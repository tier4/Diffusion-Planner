"""Tests for the reproducer's unstick escalation and the cursor's normal/repeat state.

Stuck definition:
- pose mode: stuck only when the reproducer is in ``repeat`` AND ego speed <=
  ``STUCK_SPEED_MPS`` (0.5).
- clock mode: speed-only (bag frames always advance by wall time; no ``repeat``).

The cursor exposes only normal/repeat; the stuck counting and the widen/teleport actions (and their counts) live in the rollout.
Tests that drive the rollout call ``_advance_step``/``_post_step`` directly (bypassing
``_pre_step``/``cursor.step``, which need full model-input npz keys) and set
``cursor.last_was_repeat`` each tick to stand in for "the reproducer is stuck repeating".
"""

import json

import numpy as np
import pytest

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import (
    _ego_state_from_frame,
    _post_step,
    _seed_state,
)
from scenario_generation.route_timeline import RouteTimeline

N_FRAMES = 50
STEP_M = 0.01  # ~0.1 m/s => below the 0.5 m/s "stuck" threshold, so unstick fires
EGO_SHAPE = np.array([4.76, 7.24, 2.29], dtype=np.float32)


def _make_route(tmp_path):
    """A near-stationary straight route along +x: npz frames + pose sidecars."""
    paths = []
    # A distinctive (non-zero, non-constant) recorded ego history so a real
    # reconstruction is clearly different from the live-appended buffer.
    past = np.zeros((31, 3), dtype=np.float32)
    past[:, 0] = (np.arange(31) - 30) * STEP_M  # ramp of past x positions
    for i in range(N_FRAMES):
        p = tmp_path / f"route_{i:010d}.npz"
        np.savez_compressed(
            p,
            ego_agent_past=past,
            ego_shape=EGO_SHAPE,
            turn_indicators=np.zeros(31, dtype=np.int64),
        )
        sidecar = {
            "timestamp": float(i),
            "x": float(i * STEP_M),
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        (tmp_path / f"route_{i:010d}.json").write_text(json.dumps(sidecar))
        paths.append(p)
    return RouteTimeline(paths)


def _make_route_xyyaw(tmp_path, poses):
    """Route from explicit (x, y, yaw) poses (yaw baked into the sidecar quaternion).

    Used by the yaw-gate tests to build a self-crossing / U-turn route where two frames
    share an xy but have opposite headings.
    """
    past = np.zeros((31, 3), dtype=np.float32)
    paths = []
    for i, (x, y, yaw) in enumerate(poses):
        p = tmp_path / f"route_{i:010d}.npz"
        np.savez_compressed(
            p,
            ego_agent_past=past,
            ego_shape=EGO_SHAPE,
            turn_indicators=np.zeros(31, dtype=np.int64),
        )
        sidecar = {
            "timestamp": float(i),
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": float(np.sin(yaw / 2.0)),
            "qw": float(np.cos(yaw / 2.0)),
        }
        (tmp_path / f"route_{i:010d}.json").write_text(json.dumps(sidecar))
        paths.append(p)
    return RouteTimeline(paths)


def test_unstick_jump_ego_past_uses_recorded_npz_history(tmp_path):
    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=1000,  # keep _post_step in the recorded-pose branch (no model/tracker)
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=1.0,  # disable the gentle radius-widen stage: test the teleport path
    )
    pred = np.zeros((80, 4), dtype=np.float32)  # unused in the warmup branch
    neighbors = np.zeros((320, 11), dtype=np.float32)  # no valid neighbor -> inf clearance

    ego_hist_before = None
    for i in range(20):
        ego_hist_before = s.ego_hist.copy()
        s.cursor.last_was_repeat = True  # stuck repeating (see module docstring)
        _post_step(s, pred, neighbors, idx=i, device="cpu", timers=timers)
        if s.snap_count > 0:
            break
    else:
        pytest.fail("unstick never fired on the synthetic stalled route")

    # The teleport must have been counted on the rollout as an observable event.
    assert s.snap_count == 1

    # The snap copied a recorded GT pose into live_pose; find which frame.
    matches = np.where(np.all(np.isclose(tl.poses, s.live_pose), axis=1))[0]
    assert len(matches) == 1, "post-unstick live_pose must equal exactly one recorded frame pose"
    tgt = int(matches[0])

    expected_hist = _ego_state_from_frame(tl, tgt)[1]
    assert np.allclose(s.ego_hist, expected_hist), (
        "post-jump ego_hist must be the recorded npz history of the target frame"
    )
    # And it must NOT be the stale buffer from before the jump (proves replacement).
    assert not np.allclose(s.ego_hist, ego_hist_before)
    # Sanity: the last history row is the (recorded) current pose.
    assert np.allclose(s.ego_hist[-1], s.live_pose)


def test_unstick_teleport_resets_stuck_counter(tmp_path):
    """The teleport must reset ``s.stuck`` (the ``max_stuck_steps`` counter) alongside
    ``s.ego_stuck``. ``s.stuck`` is normally driven by ``_pre_step`` comparing
    ``cursor.max_idx_reached`` to ``s.prev_max_idx``; a fresh queue built right after
    ``cur.reset(tgt)`` can return ``tgt`` itself as the first pick, so ``max_idx_reached``
    does not strictly advance on the very next step. Without resetting ``s.stuck`` here,
    a segment rescued by unstick could be immediately killed by an unrelated, stale
    stuck count left over from before the rescue.
    """
    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=1000,  # keep _post_step in the recorded-pose branch (no model/tracker)
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=1.0,  # disable the gentle radius-widen stage: test the teleport path
    )
    pred = np.zeros((80, 4), dtype=np.float32)
    neighbors = np.zeros((320, 11), dtype=np.float32)

    # This test drives _post_step directly (bypassing _pre_step, see module docstring),
    # so stand in for a stale count _pre_step would have accumulated before the rescue.
    s.stuck = 7

    for i in range(20):
        s.cursor.last_was_repeat = True  # stuck repeating (see module docstring)
        _post_step(s, pred, neighbors, idx=i, device="cpu", timers=timers)
        if s.snap_count > 0:
            break
    else:
        pytest.fail("unstick never fired on the synthetic stalled route")

    assert s.stuck == 0, "teleport must reset s.stuck alongside s.ego_stuck"


def test_unstick_widens_radius_before_teleporting(tmp_path):
    """Two-stage escalation: a stalled ego first WIDENS the cursor search_radius (gentle,
    no teleport), and only teleports if it is STILL stuck ``unstick_teleport_after`` steps
    later. The widened radius is restored to nominal once the ego moves again."""
    from scenario_generation.reproducer_rollout import _advance_step

    tl = _make_route(tmp_path)
    timers = Timers()
    base_r, mult, after, teleport_after = 1.5, 3.0, 3, 5
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=base_r,
        warmup_steps=1000,  # recorded-pose branch (no model); route speed << 0.5 -> always "stuck"
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=after,
        unstick_advance_m=0.05,
        unstick_radius_mult=mult,
        unstick_teleport_after=teleport_after,
    )
    pred = np.zeros((80, 4), dtype=np.float32)

    # Step up to (not past) the teleport threshold: radius must have been widened, no snap yet.
    for i in range(after + teleport_after - 1):
        s.cursor.last_was_repeat = True  # stuck repeating (see module docstring)
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
    assert s.cursor.search_radius == base_r * mult, "stuck ego must widen the cursor radius"
    assert s.snap_count == 0, "teleport must be deferred while only the radius has been widened"
    assert s.expand_count == 1, "stage-1 widen must be counted as an expand event"

    # One more stuck step crosses unstick_after + unstick_teleport_after -> the teleport fires,
    # and a fresh start restores the nominal radius.
    _advance_step(s, pred, idx=after + teleport_after - 1, device="cpu", timers=timers)
    assert s.snap_count == 1, "still-stuck ego must teleport after the grace window"
    assert s.cursor.search_radius == base_r, "teleport restores the nominal search_radius"


def test_stuck_requires_repeat_not_just_stopped(tmp_path):
    """Pose mode: a stopped ego whose reproducer is NOT repeating must NOT accumulate
    stuck steps, so no widen/teleport ever fires."""
    from scenario_generation.reproducer_rollout import _advance_step

    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=3.0,
        unstick_teleport_after=5,
        replay_mode="pose",
    )
    pred = np.zeros((80, 4), dtype=np.float32)
    # Stopped (speed ~0.1 <= 0.5) but reproducer advancing (not repeat) -> never stuck.
    for i in range(50):
        s.cursor.last_was_repeat = False
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
    assert s.ego_stuck == 0
    assert s.expand_count == 0 and s.snap_count == 0


def test_clock_mode_unstick_is_speed_only(tmp_path):
    """Clock mode has no cursor.step / repeat; a stopped ego must still escalate to teleport."""
    from scenario_generation.reproducer_rollout import _advance_step

    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=1.0,  # skip widen; teleport at unstick_after
        unstick_teleport_after=5,
        replay_mode="clock",
    )
    pred = np.zeros((80, 4), dtype=np.float32)
    # last_was_repeat stays False (clock never sets it); speed-only gate must still fire.
    assert s.cursor.last_was_repeat is False
    for i in range(3):
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
    assert s.snap_count == 1


def test_clock_mode_unstick_via_pre_step_loop(tmp_path, monkeypatch):
    """End-to-end: stalled clock rollout through ``_pre_step`` + ``_advance_step`` teleports.

    The previous clock-mode regression was that the run loop never escalated; driving
    ``_advance_step`` alone does not cover that path. This locks the fix by going through
    the clock branch of ``_pre_step`` (credits ``normal_steps``, never sets ``repeat``)
    before each advance.
    """
    from scenario_generation import reproducer_rollout as rr

    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=1000,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        unstick_after=3,
        unstick_advance_m=0.05,
        unstick_radius_mult=1.0,  # skip widen; teleport at unstick_after
        unstick_teleport_after=5,
        replay_mode="clock",
        neighbor_history_mode="recorded",
    )
    monkeypatch.setattr(
        rr,
        "build_input_np",
        lambda *a, **k: ({"turn_indicators": np.zeros((1, 31), dtype=np.int64)}, np.zeros((1, 11))),
    )
    pred = np.zeros((80, 4), dtype=np.float32)
    assert s.cursor.last_was_repeat is False
    for i in range(3):
        out = rr._pre_step(s)
        assert out is not None
        _np_dict, _neighbors_live, idx, *_rest = out
        rr._advance_step(s, pred, idx=idx, device="cpu", timers=timers)
    assert s.snap_count == 1
    assert s.cursor.normal_steps == 3
    assert s.cursor.repeat_steps == 0


def test_clock_pre_step_counts_normal_steps(tmp_path, monkeypatch):
    """Each clock tick updates cursor normal_steps even though cursor.step is bypassed."""
    from scenario_generation import reproducer_rollout as rr

    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=0,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        replay_mode="clock",
        neighbor_history_mode="recorded",
    )
    # Fixture npz lacks full model-input keys; stub the build so we only exercise the
    # clock frame-index + counter path.
    monkeypatch.setattr(
        rr,
        "build_input_np",
        lambda *a, **k: ({"turn_indicators": np.zeros((1, 31), dtype=np.int64)}, np.zeros((1, 11))),
    )
    n = 5
    for _ in range(n):
        out = rr._pre_step(s)
        assert out is not None
        s.k += 1  # mimic advance so start+k walks forward
    assert s.cursor.normal_steps == n
    assert s.cursor.repeat_steps == 0
    assert s.cursor.normal_steps + s.cursor.repeat_steps == n


def test_step_sets_repeat_state_and_counters(tmp_path):
    """``step`` maintains the observable base state: ``normal`` while advancing, ``repeat``
    once the neighborhood is drained/cooled, with the matching duration counters."""
    from scenario_generation.perception_reproducer import PerceptionReproducer

    tl = _make_route(tmp_path)
    # Radius < frame spacing (STEP_M=0.01) so each query returns just the current frame;
    # after it is consumed and cool-listed the next step has nothing left -> repeat.
    cur = PerceptionReproducer(tl, search_radius=0.005)
    cur.step(np.array([0.0, 0.0]), 0.0, 0.0)  # consumes frame 0 -> advancing
    assert cur.last_was_repeat is False and cur.state == "normal"
    assert cur.normal_steps == 1

    cur.step(np.array([0.0, 0.0]), 0.0, 0.1)  # frame 0 cooled, none else -> repeat
    assert cur.last_was_repeat is True and cur.state == "repeat"
    assert cur.repeat_steps >= 1
    assert cur.state_run_steps >= 1


def test_event_count_rising_edges():
    """``_event_count`` counts rising edges with a 3-frame falling-edge debounce."""
    from scenario_generation.reproducer_rollout import _event_count

    assert _event_count(np.array([], dtype=bool)) == 0
    assert _event_count(np.zeros(5, dtype=bool)) == 0
    assert _event_count(np.array([1, 1, 1], dtype=bool)) == 1  # one run, leading True
    # Gaps of 1–2 False stay latched (same event); gap of 3 releases for a new count.
    assert _event_count(np.array([1, 0, 1], dtype=bool)) == 1  # gap 1
    assert _event_count(np.array([1, 0, 0, 1], dtype=bool)) == 1  # gap 2
    assert _event_count(np.array([1, 0, 0, 0, 1], dtype=bool)) == 2  # gap 3
    assert _event_count(np.array([1, 0, 1, 1, 0, 1], dtype=bool)) == 1  # two short gaps


def test_finalize_strong_brake_steps_and_count(tmp_path):
    """``strong_brake.steps`` needs two consecutive over-threshold frames;
    ``strong_brake.count`` is the number of discrete braking events (debounced rising edges)."""
    from scenario_generation.reproducer_rollout import _finalize

    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=0,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        strong_brake_mps2=-2.5,
    )
    # Raw over-thresh: F T T F F F T  — consecutive mask: F F T F F F F
    # One sustained pair -> steps=1, count=1. Trailing single-frame spike ignored.
    s.k = 7
    s.accels[:7] = np.array([0.0, -5.0, -4.5, 0.0, 0.0, 0.0, -6.0], dtype=np.float32)
    metrics = _finalize(s)
    assert metrics["strong_brake"]["steps"] == 1
    assert metrics["strong_brake"]["count"] == 1
    # Mask keeps only the 2nd frame of the consecutive pair (-4.5); lone -6.0 spike excluded.
    assert abs(metrics["strong_brake"]["strongest_mps2"] - (-4.5)) < 1e-6


def test_finalize_road_border_collision_thresh(tmp_path):
    """Unsigned curb distance < 0.1 m is a collision; (0.1, miss_thresh] is miss-only."""
    from scenario_generation.reproducer_rollout import _finalize

    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        N_FRAMES,
        search_radius=1.5,
        warmup_steps=0,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=1000,
        strong_brake_mps2=-2.5,
    )
    s.k = 4
    # 0.05 -> collision+miss; 0.2 -> miss only; 0.6 -> neither; inf -> neither
    s.rb_dists[:4] = np.array([0.05, 0.2, 0.6, np.inf], dtype=np.float32)
    metrics = _finalize(s)
    rb = metrics["road_border"]
    assert rb["collision_steps"] == 1
    assert rb["collision_count"] == 1
    assert rb["miss_steps"] == 2
    assert rb["miss_count"] == 1
    assert "collision_thresh_m" not in rb


def test_index_ahead_by_arc_length(tmp_path):
    """Teleport target index is chosen by cumulative BAG arc length, not Euclidean
    distance from the live ego (Autoware ``find_perturb_index_along_bag`` parity)."""
    tl = _make_route(tmp_path)  # frames spaced STEP_M=0.01 m along +x
    # 0.025 m of arc at 0.01 m spacing -> 3 frames ahead (0.01+0.01+0.01 >= 0.025).
    assert tl.index_ahead_by_arc_length(0, 0.025) == 3
    assert tl.index_ahead_by_arc_length(10, 0.025) == 13
    # distance <= 0 still advances >=1 frame (avoids re-teleporting onto the same anchor).
    assert tl.index_ahead_by_arc_length(10, 0.0) == 11
    assert tl.index_ahead_by_arc_length(0, 1e9) == len(tl) - 1
    assert tl.index_ahead_by_arc_length(len(tl) - 1, 0.0) == len(tl) - 1


def test_yaw_gate_rejects_opposite_heading_future_frame(tmp_path):
    """On a self-crossing route an xy-only cursor can grab a FUTURE opposite-heading
    frame; with ``yaw_gate`` on (default) it is rejected and the last same-heading
    frame is repeated. ``yaw_gate=False`` restores the ungated pick."""
    from scenario_generation.perception_reproducer import PerceptionReproducer

    # 0: origin heading +x; 1: far ahead (outside the 1.5 m radius, ignored);
    # 2: back near the origin but heading -x (a U-turn frame with a LATER index).
    tl = _make_route_xyyaw(tmp_path, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.05, 0.0, np.pi)])
    origin = np.array([0.0, 0.0])

    # Gate off: xy-only cursor grabs the opposite-heading future frame (2).
    cur_off = PerceptionReproducer(tl, search_radius=1.5, yaw_gate=False)
    assert cur_off.step(origin, 0.0, 0.0, sim_yaw=0.0) == 0
    assert cur_off.step(origin, 0.0, 0.1, sim_yaw=0.0) == 2

    # Gate on (default): reject that frame -> repeat the last good same-heading frame.
    cur = PerceptionReproducer(tl, search_radius=1.5)
    assert cur.yaw_gate is True
    assert cur.step(origin, 0.0, 0.0, sim_yaw=0.0) == 0  # same-heading admitted
    assert cur.step(origin, 0.0, 0.1, sim_yaw=0.0) == 0  # gated -> repeat, not frame 2
    assert cur.last_was_repeat is True


def test_perception_reproducer_widen_and_restore(tmp_path):
    """``widen``/``restore_radius`` set the radius and force a queue rebuild; restore is
    relative to the nominal base, not whatever the radius happened to be."""
    from scenario_generation.perception_reproducer import PerceptionReproducer

    tl = _make_route(tmp_path)
    cur = PerceptionReproducer(tl, search_radius=1.5)
    cur.step(np.array([0.0, 0.0]), 0.0, 0.0)  # build a queue
    cur.widen(4.0)
    assert cur.search_radius == 6.0
    assert len(cur._queue) == 0 and cur._last_seq_pos is None  # forced rebuild
    cur.restore_radius()
    assert cur.search_radius == 1.5


def test_precollision_window_clamps_across_unstick_snap():
    """The pre-collision window must not cross an unstick snap (else a saved scene's
    realized ego_future/ego_past would span the ~5m teleport)."""
    from scenario_generation.reproducer_rollout import _precollision_window_start

    t_c, pre = 579, 80  # window would be [499, 579)
    # no snap -> full window
    assert _precollision_window_start(t_c, pre, None) == 499
    # snap BEFORE the window -> not clamped (full window)
    assert _precollision_window_start(t_c, pre, 400) == 499
    # snap INSIDE the window -> clamp to the post-snap step (shorter, snap-free window)
    assert _precollision_window_start(t_c, pre, 540) == 540
    # early collision, no snap -> clamped to the live floor (0); recorded backfill is disabled,
    # so the window is shorter/all-live rather than starting at a negative (pre-segment) step.
    assert _precollision_window_start(50, pre, None) == 0
