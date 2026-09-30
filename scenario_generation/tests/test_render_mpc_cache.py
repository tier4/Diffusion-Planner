"""The single-segment rollout tracks only model-refresh ticks with MPC."""

import json

import numpy as np
import pytest
import torch

from scenario_generation import reproducer_rollout as rr
from scenario_generation.perf_timer import Timers
from scenario_generation.tests.test_reproducer_unstick import _make_route


def _run_cached(
    tmp_path,
    monkeypatch,
    *,
    interval=4,
    steps=6,
    horizon=8,
    snap_at=None,
    tracker_mode="mpc",
    warmup_steps=0,
):
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    tl = _make_route(route_dir)
    model_calls = []
    track_calls = []
    original_advance = rr._advance_step

    def pre_step(state):
        if state.k >= steps:
            state.done = True
            state.terminated = "max_steps"
            return None
        return {}, np.zeros((0, 11)), min(state.k, len(tl) - 1), None, None

    def score(state, *_args, **_kwargs):
        state.clearances[state.k] = np.inf
        state.rb_dists[state.k] = np.inf

    def fake_track(tracker, x0, ref):
        track_calls.append((np.array(x0, copy=True), np.array(ref, copy=True)))
        tracker.last_accel = 0.0
        tracker.last_steering = 0.0
        tracker.last_yaw_rate = 0.0
        return np.array([x0[0] + 0.5, x0[1], x0[2]], dtype=np.float64), 5.0

    def advance(*args, **kwargs):
        original_advance(*args, **kwargs)
        state = args[0]
        if snap_at is not None and state.k - 1 == snap_at:
            state.snap_count += 1
            state.accels[state.k - 1] = np.nan

    class Model:
        def __call__(self, _data):
            model_calls.append(1)
            pred = np.zeros((1, 1, horizon, 4), dtype=np.float32)
            pred[0, 0, :, 0] = 0.5 * np.arange(1, horizon + 1)
            pred[0, 0, :, 2] = 1.0
            return None, {"prediction": torch.from_numpy(pred)}

    from scenario_generation.mpc_tracker import MPCTracker

    monkeypatch.setattr(MPCTracker, "track", fake_track)
    monkeypatch.setattr(rr, "_pre_step", pre_step)
    monkeypatch.setattr(rr, "_score_into", score)
    monkeypatch.setattr(rr, "_gt_deviation_m", lambda *_args: rr.GTDeviation(0.0, None, 0, 0))
    monkeypatch.setattr(rr, "_to_torch_batch", lambda *_args: None)
    monkeypatch.setattr(rr, "mark_inference_step", lambda: None)
    monkeypatch.setattr(rr, "_feed_turn_indicator", lambda *_args: None)
    monkeypatch.setattr(rr, "_score_turn_indicator", lambda *_args: None)
    monkeypatch.setattr(rr, "_hold_turn_indicator", lambda *_args: None)
    if snap_at is not None:
        monkeypatch.setattr(rr, "_advance_step", advance)

    out = tmp_path / "out"
    rr.render_segment(
        Model(),
        None,
        tl,
        0,
        len(tl),
        out,
        device="cpu",
        near_miss_thresh=0.5,
        search_radius=1.5,
        warmup_steps=warmup_steps,
        unstick_after=0,
        unstick_advance_m=1.5,
        unstick_radius_mult=3.0,
        unstick_teleport_after=50,
        draw_every=None,
        replan_interval=interval,
        tracker_mode=tracker_mode,
        neighbor_history_mode="recorded",
        yaw_gate=True,
        strong_brake_mps2=-2.5,
        abort_deviation_m=0.0,
        abort_after=30,
        abort_max_snaps=0,
        deviation_collision_thresh_m=2.0,
        drop_objects=False,
        goal_mode="segment",
        title_prefix=None,
        distance_label_offset_m=1.2,
        view_half_m=50.0,
        max_stuck_steps=0,
        goal_reach_m=0.0,
        interpolate=False,
        color_by_uuid=False,
        window=None,
        max_steps=steps,
        timeline_progress_mode="pose",
        draw_pool=object(),
        timers=Timers(),
    )
    rows = [json.loads(line) for line in (out / "rollout.jsonl").read_text().splitlines()]
    return model_calls, track_calls, [row for row in rows if "speed" in row]


def test_mpc_only_tracks_refresh_ticks_and_cached_pose_uses_raw_plan(tmp_path, monkeypatch):
    model, tracks, rows = _run_cached(tmp_path, monkeypatch)
    assert len(model) == len(tracks) == 2
    assert len(rows) == 6
    assert [r["replan"] for r in rows] == [True, False, False, False, True, False]
    assert [r["plan_offset"] for r in rows] == [0, 1, 2, 3, 0, 1]
    assert all(r["strong_brake_acceleration_window_s"] == 0.1 for r in rows)
    assert all(r["strong_brake_filter_type"] == "active_plan_three_point" for r in rows)
    assert [r["executed_tracker_mode"] for r in rows] == [
        "mpc",
        "cached_pose",
        "cached_pose",
        "cached_pose",
        "mpc",
        "cached_pose",
    ]
    assert all(r["mpc_commanded_accel_mps2"] is None for r in rows if not r["replan"])
    assert all(r["mpc_commanded_steering_rad"] is None for r in rows if not r["replan"])
    assert all(r["mpc_accel_saturated"] is None for r in rows if not r["replan"])
    np.testing.assert_allclose(
        [r["ego_after_world"] for r in rows if not r["replan"]],
        [r["planned_next_world"] for r in rows if not r["replan"]],
    )


def test_replan_every_tick_and_short_reference_tail(tmp_path, monkeypatch):
    model, tracks, rows = _run_cached(tmp_path, monkeypatch, interval=1, steps=4, horizon=2)
    assert len(model) == len(tracks) == 4
    assert [r["plan_offset"] for r in rows] == [0, 0, 0, 0]
    np.testing.assert_allclose([ref[0, 0] for _, ref in tracks], [0.5, 1.0, 1.5, 2.0])


def test_snap_forced_refresh_keeps_original_modulo_cached_index(tmp_path, monkeypatch):
    model, tracks, rows = _run_cached(
        tmp_path, monkeypatch, interval=8, steps=4, horizon=3, snap_at=1
    )
    assert len(model) == len(tracks) == 2
    assert [r["replan"] for r in rows] == [True, False, True, False]
    # The forced refresh uses point zero; the next cached tick follows k % 8,
    # clamped to the short plan's tail, as in the original implementation.
    assert [r["plan_offset"] for r in rows] == [0, 1, 0, 2]
    np.testing.assert_allclose([r["ego_after_world"][0] for r in rows], [0.5, 1.0, 1.5, 2.5])


def test_perfect_still_executes_cached_world_points(tmp_path, monkeypatch):
    model, tracks, rows = _run_cached(
        tmp_path, monkeypatch, interval=4, steps=6, tracker_mode="perfect"
    )
    assert len(model) == 2
    assert tracks == []
    assert [r["plan_offset"] for r in rows] == [0, 1, 2, 3, 0, 1]
    np.testing.assert_allclose([r["ego_after_world"][0] for r in rows], np.arange(1, 7) * 0.5)
    assert all(r["executed_tracker_mode"] == "perfect" for r in rows)


def test_trace_marks_recorded_warmup_as_not_mpc(tmp_path, monkeypatch):
    _, tracks, rows = _run_cached(tmp_path, monkeypatch, interval=1, steps=3, warmup_steps=1)
    assert len(tracks) == 2
    assert rows[0]["executed_tracker_mode"] == "recorded_warmup"
    assert rows[0]["mpc_commanded_accel_mps2"] is None
    assert rows[0]["tracking_error_m"] is None
    assert [r["executed_tracker_mode"] for r in rows[1:]] == ["mpc", "mpc"]
