"""When the unstick feature snaps a stalled ego forward, the ego_past that goes
with the jump must be the REAL recorded npz history of the target frame — not the
stale live buffer accumulated while the ego was stuck.

This drives the actual rollout step function (``_post_step``) through an unstick on
a tiny synthetic route and asserts the post-jump ``ego_hist`` equals the recorded
frame's ``ego_agent_past`` reconstruction (``_ego_state_from_frame``).
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
        np.savez_compressed(p, ego_agent_past=past, ego_shape=EGO_SHAPE)
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
        _post_step(s, pred, neighbors, idx=i, device="cpu", timers=timers)
        if s.n_snaps > 0:
            break
    else:
        pytest.fail("unstick never fired on the synthetic stalled route")

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
        _advance_step(s, pred, idx=i, device="cpu", timers=timers)
    assert s.cursor.search_radius == base_r * mult, "stuck ego must widen the cursor radius"
    assert s.n_snaps == 0, "teleport must be deferred while only the radius has been widened"

    # One more stuck step crosses unstick_after + unstick_teleport_after -> the teleport fires,
    # and a fresh start restores the nominal radius.
    _advance_step(s, pred, idx=after + teleport_after - 1, device="cpu", timers=timers)
    assert s.n_snaps == 1, "still-stuck ego must teleport after the grace window"
    assert s.cursor.search_radius == base_r, "teleport restores the nominal search_radius"


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
