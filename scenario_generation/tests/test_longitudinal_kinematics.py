"""Production active-plan braking and legacy trace compatibility."""

import numpy as np
import pytest
import torch

from scenario_generation.longitudinal_kinematics import (
    PLAN_BRAKE_FILTER_TYPE,
    PLAN_BRAKE_POINTS,
    PLAN_BRAKE_SOURCE,
    brake_event_onsets,
    confirmed_brake_mask,
    plan_suffix_acceleration,
)


def test_three_raw_plan_points_score_adjacent_chord_speeds():
    points = np.array([[1.0, 0.0], [1.8, 0.0], [2.3, 0.0], [2.65, 0.0]])
    assert plan_suffix_acceleration(points) == pytest.approx(-30.0)
    assert plan_suffix_acceleration(points[1:]) == pytest.approx(-15.0)
    # The score is invariant to rigid world translation, without casting the
    # plan to map-frame float32 coordinates.
    assert plan_suffix_acceleration(points + [90000.0, 40000.0]) == pytest.approx(-30.0)


def test_incomplete_and_nonfinite_plan_suffix_is_unscored():
    for points in (
        np.empty((0, 2)),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 0.0]]),
    ):
        assert np.isnan(plan_suffix_acceleration(points))
    with pytest.raises(ValueError, match="shape"):
        plan_suffix_acceleration(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="dt"):
        plan_suffix_acceleration(np.zeros((3, 2)), dt=0)


def test_confirmation_and_five_clear_frame_event_release():
    scored = np.array([-4.0, -4.0, 0, 0, 0, 0, -4, -4, np.nan, -4, -4])
    assert np.flatnonzero(confirmed_brake_mask(scored)).tolist() == [1, 7, 10]
    assert brake_event_onsets(scored) == [1, 10]
    # The four-frame clear gap keeps the first event active. A missing score
    # resets it even without five finite clear frames.
    assert brake_event_onsets(np.array([-4, -4, 0, 0, 0, 0, -4, -4])) == [1]
    assert brake_event_onsets(np.array([-4, -4, 0, 0, 0, 0, 0, -4, -4])) == [1, 8]


def test_segment_aggregate_and_colormap_share_plan_definition():
    from scenario_generation.closed_loop_eval import aggregate
    from scenario_generation.reproducer_rollout import strong_brake_block
    from scenario_generation.tests.test_closed_loop_metrics import _segment_row
    from scenario_generation.trajectory_colormap import _risk_and_ticks

    scores = np.array([0.0, -3.0, -4.0, 0.0])
    rows = [
        {
            "speed": 0.0,
            "brake_metric_accel_mps2": float(score),
            "strong_brake_filter_type": PLAN_BRAKE_FILTER_TYPE,
            "strong_brake_plan_points": PLAN_BRAKE_POINTS,
            "strong_brake_acceleration_window_s": 0.1,
            "strong_brake_future_source": PLAN_BRAKE_SOURCE,
        }
        for score in scores
    ]
    risk, _, labels = _risk_and_ticks(rows, "strong_brake", near_miss_thresh=0.5)
    block = strong_brake_block(scores, -2.5)
    assert risk.tolist() == [0, 0, 1, 0]
    assert block["count"] == block["steps"] == 1
    assert block["strongest_mps2"] == pytest.approx(-4.0)
    assert block["filter_type"] == PLAN_BRAKE_FILTER_TYPE
    assert block["plan_points"] == 3
    assert "active plan" in labels[1]

    one, two = _segment_row(), _segment_row()
    one["strong_brake"] = block
    two["strong_brake"] = strong_brake_block(np.zeros(4), -2.5)
    summary = aggregate([one, two], 0.5)["strong_brake"]
    assert summary["filter_type"] == PLAN_BRAKE_FILTER_TYPE
    assert summary["plan_points"] == 3
    assert summary["count"] == summary["steps"] == 1
    with pytest.raises(ValueError, match="different strong-brake"):
        aggregate([one, _segment_row()], 0.5)
    bad = dict(two)
    bad["strong_brake"] = dict(two["strong_brake"], filter_type="unknown")
    with pytest.raises(ValueError, match="Unknown strong-brake"):
        aggregate([one, bad], 0.5)
    with pytest.raises(ValueError, match="mixed strong-brake"):
        _risk_and_ticks(rows[:-1] + [{"speed": 0.0}], "strong_brake", near_miss_thresh=0.5)


def test_old_speed_only_trace_does_not_invent_last_transition():
    from scenario_generation.trajectory_colormap import _risk_and_ticks

    rows = [{"speed": 10 - 0.3 * k} for k in range(3)]
    risk, _, _ = _risk_and_ticks(rows, "strong_brake", near_miss_thresh=0.5)
    assert risk.tolist() == [0, 1, 0]


def test_advance_step_score_ignores_executed_pose_and_speed(tmp_path):
    from scenario_generation.perf_timer import Timers
    from scenario_generation.reproducer_rollout import _advance_step, _seed_state
    from scenario_generation.tests.test_reproducer_unstick import N_FRAMES, _make_route

    tl = _make_route(tmp_path)
    state = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=0,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=Timers(),
        max_steps=4,
        unstick_after=0,
    )
    pred = np.zeros((4, 4), dtype=np.float32)
    pred[:, 0] = [1.0, 1.8, 2.3, 2.65]
    pred[:, 2] = 1.0
    for post_speed, displacement in ((0.0, 0.0), (12.0, 5.0)):
        post_pose = state.live_pose.copy()
        post_pose[0] += displacement
        _advance_step(
            state,
            pred,
            idx=state.k,
            device="cpu",
            timers=Timers(),
            override=(post_pose, post_speed),
        )
    np.testing.assert_allclose(state.brake_metric_accels[:2], [-30.0, -30.0], atol=1e-4)
    assert state.accels[0] != state.accels[1]


def test_renderer_scores_nonuniform_fresh_and_cached_raw_suffix(tmp_path, monkeypatch):
    from scenario_generation.tests.test_render_mpc_cache import _run_cached

    original = torch.from_numpy
    positions = np.array([1.0, 1.8, 2.3, 2.65, 2.9, 3.1], dtype=np.float32)

    def nonuniform_model_output(array):
        if array.shape == (1, 1, len(positions), 4):
            array[0, 0, :, 0] = positions
        return original(array)

    monkeypatch.setattr(torch, "from_numpy", nonuniform_model_output)
    model_calls, tracker_calls, rows = _run_cached(
        tmp_path, monkeypatch, interval=4, steps=6, horizon=len(positions)
    )
    assert len(model_calls) == len(tracker_calls) == 2
    assert [row["plan_offset"] for row in rows] == [0, 1, 2, 3, 0, 1]
    expected = [
        plan_suffix_acceleration(positions[offset:, None].repeat(2, axis=1) * [1.0, 0.0])
        for offset in (0, 1, 2, 3, 0, 1)
    ]
    np.testing.assert_allclose(
        [row["brake_metric_accel_mps2"] for row in rows], expected, atol=1e-4
    )
    assert rows[1]["executed_tracker_mode"] == "cached_pose"
    assert rows[0]["executed_tracker_mode"] == "mpc"
    assert expected[0] != expected[1] != expected[2]


def test_cached_tail_with_fewer_than_three_points_is_unscored(tmp_path, monkeypatch):
    from scenario_generation.tests.test_render_mpc_cache import _run_cached

    _, _, rows = _run_cached(tmp_path, monkeypatch, interval=8, steps=6, horizon=4)
    assert [row["plan_offset"] for row in rows] == [0, 1, 2, 3, 3, 3]
    assert [row["brake_metric_accel_mps2"] is None for row in rows] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_brake_event_count_uses_five_raw_clear_frames():
    from scenario_generation.reproducer_rollout import strong_brake_block

    merged = strong_brake_block(np.array([-4.0, -4.0, 0.0, 0.0, 0.0, 0.0, -4.0, -4.0]), -2.5)
    split = strong_brake_block(np.array([-4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, -4.0]), -2.5)
    assert merged["event_clear_frames"] == split["event_clear_frames"] == 5
    assert merged["steps"] == split["steps"] == 2
    assert merged["count"] == 1
    assert split["count"] == 2
