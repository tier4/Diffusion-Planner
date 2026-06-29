"""PerceptionReproducer cursor: chronological queue over a spatial neighborhood,
cool-down, and the degenerate nearest-only mode.

The key behavior is the red-light/dwell case: while the ego is stationary the
recorded scene must keep advancing in time (frames popped in chronological order),
not freeze.
"""

import json

import numpy as np

from scenario_generation.perception_reproducer import PerceptionReproducer
from scenario_generation.route_timeline import RouteTimeline

SPACING_M = 0.5  # frame spacing along +x => speed 5 m/s at 10 Hz


def _straight_route(tmp_path, n=12):
    paths = []
    for i in range(n):
        p = tmp_path / f"r_{i:010d}.npz"
        np.savez_compressed(p, ego_agent_past=np.zeros((31, 3), np.float32))
        (tmp_path / f"r_{i:010d}.json").write_text(
            json.dumps(
                {"x": float(i * SPACING_M), "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
            )
        )
        paths.append(p)
    return RouteTimeline(paths)


def test_dwell_advances_scene_chronologically(tmp_path):
    """Stationary ego => queue of nearby frames is popped in time order (red light)."""
    tl = _straight_route(tmp_path, n=12)
    cur = PerceptionReproducer(tl, search_radius=1.5, cool_down_sec=80.0)
    cur.reset(0)
    # Ego parked at frame 0's position; frames within 1.5m are 0,1,2,3 (x=0,.5,1,1.5).
    served = [cur.step(np.array([0.0, 0.0]), sim_speed=0.0, sim_time=0.1 * k) for k in range(4)]
    assert served == [0, 1, 2, 3], f"dwell did not advance chronologically: {served}"
    assert cur.max_idx_reached == 3


def test_used_frames_not_reserved_during_cooldown(tmp_path):
    """Within the cool-down window a popped frame is not served again on rebuild."""
    tl = _straight_route(tmp_path, n=12)
    cur = PerceptionReproducer(tl, search_radius=1.5, cool_down_sec=80.0)
    cur.reset(0)
    seen = [cur.step(np.array([0.0, 0.0]), 0.0, 0.1 * k) for k in range(4)]  # drains 0..3
    assert sorted(seen) == [0, 1, 2, 3]  # the drain phase served each frame exactly once
    # Queue now empty and every nearby frame (0..3) is still within cool-down, so the
    # rebuild yields nothing fresh and the cursor HOLDS the last frame instead of
    # re-serving a cooled one.
    nxt = cur.step(np.array([0.0, 0.0]), 0.0, 0.5)
    assert nxt == 3, f"expected hold-last under full cool-down, got {nxt}"


def test_degenerate_radius_is_nearest_only(tmp_path):
    tl = _straight_route(tmp_path, n=12)
    cur = PerceptionReproducer(tl, search_radius=0.0)
    cur.reset(0)
    # x=1.1 is nearest to frame 2 (x=1.0, d=0.1) over frame 3 (x=1.5, d=0.4).
    assert cur.step(np.array([1.1, 0.0]), 5.0, 0.1) == 2
    assert cur.step(np.array([2.6, 0.0]), 5.0, 0.2) == 5  # x=2.5


def test_max_idx_reached_is_monotonic(tmp_path):
    tl = _straight_route(tmp_path, n=12)
    cur = PerceptionReproducer(tl, search_radius=1.5)
    cur.reset(0)
    last = -1
    for k in range(20):
        cur.step(np.array([min(k, 11) * SPACING_M, 0.0]), 5.0, 0.1 * k)
        assert cur.max_idx_reached >= last
        last = cur.max_idx_reached


def test_cursor_does_not_rewind_to_older_unserved_frames(tmp_path):
    tl = _straight_route(tmp_path, n=12)
    cur = PerceptionReproducer(tl, search_radius=1.5)
    cur.reset(5)

    assert cur.step(np.array([5 * SPACING_M, 0.0]), 5.0, 0.0) == 5

    # The live ego is now physically close to old frames 0..3. Those frames were not served
    # in this cursor instance, so cooldown alone would not block them. The reproducer should
    # still not rewind the replay once frame 5 has been reached.
    assert cur.step(np.array([0.0, 0.0]), 0.0, 0.1) == 5
    assert cur.max_idx_reached == 5
