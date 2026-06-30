"""Pre-collision save window: never backfill recorded frames (the window start is
clamped to the earliest LIVE buffer step), and shorten / bound it correctly.

These guard the behaviour change that removed the recorded->live clearance seam: an
early contact must yield a SHORTER all-live window, never a step before the live floor.
"""

import numpy as np

from scenario_generation.reproducer_rollout import _precollision_window_start


def _poses(lo, hi, step_m=1.0):
    """step k -> world pose [x, y, yaw]; x advances ``step_m`` per step (arc length)."""
    return {k: np.array([k * step_m, 0.0, 0.0], dtype=np.float64) for k in range(lo, hi + 1)}


def test_early_contact_clamps_to_zero_never_backfills():
    # t_c - pre_steps is negative; the live buffer starts at 0 -> clamp to 0 (no backfill).
    s = _precollision_window_start(44, 80, None, _poses(0, 44), pre_arc_m=0.0, max_scenes=160)
    assert s == 0


def test_full_window_when_enough_history():
    s = _precollision_window_start(120, 80, None, _poses(0, 120), pre_arc_m=0.0, max_scenes=160)
    assert s == 40  # t_c - pre_steps, all-live


def test_clamps_to_unstick_snap():
    s = _precollision_window_start(120, 80, 100, _poses(0, 120), pre_arc_m=0.0, max_scenes=160)
    assert s == 100


def test_buffer_floor_binds_when_buffer_shorter_than_pre_steps():
    # buffer only holds steps 50..120 (e.g. post-snap) -> floor is 50, not 40, never < 50.
    s = _precollision_window_start(120, 80, None, _poses(50, 120), pre_arc_m=0.0, max_scenes=160)
    assert s == 50


def test_max_scenes_caps_window():
    s = _precollision_window_start(120, 80, None, _poses(0, 120), pre_arc_m=0.0, max_scenes=30)
    assert s == 120 - 29  # t_c - (max_scenes - 1)


def test_arc_extend_never_goes_below_live_floor():
    # short baseline (pre_steps=10 -> base=110) but the ego barely moved, so the arc-extend
    # would reach back further; it must stop at the live floor (0), never below it.
    s = _precollision_window_start(
        120, 10, None, _poses(0, 120, step_m=0.001), pre_arc_m=1000.0, max_scenes=160
    )
    assert s == 0
