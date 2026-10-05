import numpy as np
import pytest

from scenario_generation.scenario_metrics import geometry  # noqa: F401  (registers the scorers)
from scenario_generation.scenario_metrics.geometry import OBJECT_AVOIDANCE_MIN_CLEARANCE_M
from scenario_generation.scenario_metrics.registry import score
from scenario_generation.scenario_metrics.testing import make_input, straight_path

DT = 0.1
ANCHOR = 10


def _turn_path(
    n: int, speed: float = 5.0, radius: float = 15.0, straight: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Straight for ``straight`` frames, then a left arc of ``radius``."""
    s = np.arange(n) * speed * DT
    s0 = straight * speed * DT
    arc = np.clip(s - s0, 0.0, None)
    theta = arc / radius
    x = np.where(s < s0, s, s0 + radius * np.sin(theta))
    y = np.where(s < s0, 0.0, radius * (1.0 - np.cos(theta)))
    return np.stack([x, y], axis=1), theta


def _lane(
    y: float, x0: float = -30.0, x1: float = 200.0, points: int = 20, half_width: float = 1.75
) -> np.ndarray:
    """One straight lanelet along +x in the tensor layout (xy, direction, left/right offsets)."""
    lane = np.zeros((points, 8))
    lane[:, 0] = np.linspace(x0, x1, points)
    lane[:, 1] = y
    lane[:, 2] = 1.0
    lane[:, 5] = half_width
    lane[:, 7] = -half_width
    return lane


def _frames(lanes: list[np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    tensor = np.stack(lanes)
    return {ANCHOR: {"lanes": tensor, "route_lanes": tensor[:1]}}


# ---------------------------------------------------------------- simple_turn


def test_simple_turn_following_the_recorded_turn_passes():
    rec_xy, rec_yaw = _turn_path(200)
    r = score(
        make_input(
            label="simple_turn",
            ego_xy=rec_xy,
            ego_yaw=rec_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
        )
    )
    assert r.passed is True
    assert r.values["max_lateral_error_m"] < 1e-6
    assert r.values["progress_ratio"] == pytest.approx(1.0)


def test_simple_turn_cutting_the_corner_fails():
    rec_xy, rec_yaw = _turn_path(200)
    # Same turn on a 2 m tighter radius: laterally off by up to 2 m.
    ego_xy, ego_yaw = _turn_path(200, radius=13.0)
    r = score(
        make_input(
            label="simple_turn",
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
        )
    )
    assert r.passed is False
    assert r.values["max_lateral_error_m"] > 1.0


def test_simple_turn_stalling_at_the_anchor_fails():
    rec_xy, rec_yaw = _turn_path(200)
    ego_xy = np.concatenate(
        [rec_xy[: ANCHOR + 1], np.repeat(rec_xy[ANCHOR : ANCHOR + 1], 150, axis=0)]
    )
    ego_yaw = np.concatenate([rec_yaw[: ANCHOR + 1], np.full(150, rec_yaw[ANCHOR])])
    r = score(
        make_input(
            label="simple_turn",
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
        )
    )
    assert r.passed is False and "cover" in r.reason
    assert r.values["max_lateral_error_m"] < 1e-6  # the stall would otherwise look perfect


def test_simple_turn_trace_ending_mid_turn_fails_but_a_goal_inside_the_stretch_counts():
    rec_xy, rec_yaw = _turn_path(200)
    short = ANCHOR + 30  # max_steps after 3 s of an 8 s stretch
    r = score(
        make_input(
            label="simple_turn",
            ego_xy=rec_xy[:short],
            ego_yaw=rec_yaw[:short],
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            terminated="max_steps",
        )
    )
    assert r.passed is False and r.values["reached_end"] == 0.0

    # Window of 60 frames: the stretch is clipped at its last frame and the goal
    # fires 5 m short of it.
    rec_xy, rec_yaw = rec_xy[:60], rec_yaw[:60]
    r = score(
        make_input(
            label="simple_turn",
            ego_xy=rec_xy[:50],
            ego_yaw=rec_yaw[:50],
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            terminated="goal",
        )
    )
    assert r.values["progress_ratio"] < 0.9 and r.passed is True


def test_simple_turn_anchor_never_reached_is_not_applicable():
    rec_xy, rec_yaw = _turn_path(200)
    r = score(
        make_input(
            label="simple_turn",
            ego_xy=rec_xy[:5],
            ego_yaw=rec_yaw[:5],
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
        )
    )
    assert r.passed is None and r.reason == "anchor never reached"


# ---------------------------------------------------------------- centerline


def test_centerline_measures_the_offset_from_the_route_lane():
    rec_xy, rec_yaw = straight_path(200, 5.0)
    frames = _frames([_lane(0.0)])
    near = rec_xy + [0.0, 0.3]
    r = score(
        make_input(
            label="centerline",
            ego_xy=near,
            ego_yaw=rec_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            frames=frames,
        )
    )
    assert r.passed is True
    assert r.values["average_lateral_error_m"] == pytest.approx(0.3)

    far = rec_xy + [0.0, 1.5]
    r = score(
        make_input(
            label="centerline",
            ego_xy=far,
            ego_yaw=rec_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            frames=frames,
        )
    )
    assert r.passed is False
    assert r.values["max_lateral_error_m"] == pytest.approx(1.5)


def test_centerline_anchor_never_reached_is_not_applicable():
    rec_xy, rec_yaw = straight_path(200, 5.0)
    r = score(
        make_input(
            label="centerline",
            ego_xy=rec_xy[:3],
            ego_yaw=rec_yaw[:3],
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
        )
    )
    assert r.passed is None


# ---------------------------------------------------------------- lane_change


def _lane_change_path(n: int, shift: float, speed: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Straight along +x, sliding ``shift`` sideways between 1 s and 5 s after the anchor."""
    xy, yaw = straight_path(n, speed)
    t = (np.arange(n) - ANCHOR) * DT
    u = np.clip((t - 1.0) / 4.0, 0.0, 1.0)
    xy = xy.copy()
    xy[:, 1] = shift * (3 * u**2 - 2 * u**3)
    return xy, yaw


def _lane_change(ego_shift: float, rec_shift: float = -3.5):
    rec_xy, rec_yaw = _lane_change_path(200, rec_shift)
    ego_xy, ego_yaw = _lane_change_path(200, ego_shift)
    # Lanes are ego-centric at the anchor's recorded pose (x = ANCHOR m along the road).
    frames = _frames([_lane(0.0), _lane(-3.5), _lane(3.5)])
    return score(
        make_input(
            label="lane_change",
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            frames=frames,
        )
    )


def test_lane_change_following_the_human_passes():
    r = _lane_change(-3.5)
    assert r.passed is True
    assert r.values["gt_direction"] == -1.0
    assert r.values["completion_ratio"] == pytest.approx(1.0)
    assert r.values["lane_tolerance_m"] == pytest.approx(1.75)
    assert 1.0 < r.values["lane_change_time_s"] < 5.0


def test_lane_change_staying_in_lane_or_going_the_wrong_way_fails():
    stay = _lane_change(0.0)
    assert stay.passed is False and stay.values["left_source_lane"] == 0.0
    wrong = _lane_change(3.5)
    assert wrong.passed is False and wrong.values["left_source_lane"] == 0.0


def test_lane_change_overshooting_the_target_lane_fails():
    r = _lane_change(-7.0)
    assert r.passed is False
    assert r.values["left_source_lane"] == 1.0 and r.values["reached_gt_lane"] == 0.0


def test_lane_change_not_completed_before_the_trace_ends_fails():
    rec_xy, rec_yaw = _lane_change_path(200, -3.5)
    end = ANCHOR + 25  # aborted 2.5 s after the anchor, half a metre into the change
    frames = _frames([_lane(0.0), _lane(-3.5)])
    r = score(
        make_input(
            label="lane_change",
            ego_xy=rec_xy[:end],
            ego_yaw=rec_yaw[:end],
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            frames=frames,
            terminated="abort",
        )
    )
    assert r.passed is False and r.values["left_source_lane"] == 0.0


def test_lane_change_without_a_recorded_lane_change_is_not_applicable():
    r = _lane_change(0.0, rec_shift=0.0)
    assert r.passed is None and r.details["gt_lane_change_detected"] is False


def test_lane_change_anchor_never_reached_is_not_applicable():
    rec_xy, rec_yaw = _lane_change_path(200, -3.5)
    r = score(
        make_input(
            label="lane_change",
            ego_xy=rec_xy[:4],
            ego_yaw=rec_yaw[:4],
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
        )
    )
    assert r.passed is None


# ---------------------------------------------------------------- object_avoidance


def _avoidance(ego_xy, ego_yaw, rec_xy, rec_yaw, **kw):
    return score(
        make_input(
            label="object_avoidance",
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            rec_xy=rec_xy,
            rec_yaw=rec_yaw,
            anchor_frame=ANCHOR,
            **kw,
        )
    )


def test_object_avoidance_passing_the_obstacle_cleanly_passes():
    xy, yaw = straight_path(200, 5.0)
    r = _avoidance(xy, yaw, xy, yaw, clearance_m=np.full(200, 1.2))
    assert r.passed is True
    assert r.values["min_clearance_m"] == pytest.approx(1.2) and r.values["collision"] == 0.0


def test_object_avoidance_collision_fails():
    xy, yaw = straight_path(200, 5.0)
    collision = np.zeros(200, dtype=bool)
    collision[ANCHOR + 30] = True
    r = _avoidance(xy, yaw, xy, yaw, clearance_m=np.full(200, 1.2), collision=collision)
    assert r.passed is False and r.details["first_collision_step"] == ANCHOR + 30
    # A collision before the anchor is not this label's concern.
    early = np.zeros(200, dtype=bool)
    early[ANCHOR - 5] = True
    assert (
        _avoidance(xy, yaw, xy, yaw, clearance_m=np.full(200, 1.2), collision=early).passed is True
    )


def test_object_avoidance_passing_too_close_fails():
    xy, yaw = straight_path(200, 5.0)
    clearance = np.full(200, 1.2)
    clearance[ANCHOR + 30] = 0.3
    r = _avoidance(xy, yaw, xy, yaw, clearance_m=clearance)
    assert r.passed is False and r.values["collision"] == 0.0
    assert r.reason == f"passed a neighbor closer than {OBJECT_AVOIDANCE_MIN_CLEARANCE_M} m"
    # Exactly at the threshold is wide enough.
    clearance[ANCHOR + 30] = OBJECT_AVOIDANCE_MIN_CLEARANCE_M
    assert _avoidance(xy, yaw, xy, yaw, clearance_m=clearance).passed is True


def test_object_avoidance_by_stopping_forever_fails():
    rec_xy, rec_yaw = straight_path(200, 5.0)
    ego_xy = np.concatenate(
        [rec_xy[: ANCHOR + 1], np.repeat(rec_xy[ANCHOR : ANCHOR + 1], 189, axis=0)]
    )
    r = _avoidance(
        ego_xy, rec_yaw, rec_xy, rec_yaw, clearance_m=np.full(200, 3.0), terminated="max_steps"
    )
    assert r.passed is False and r.values["collision"] == 0.0
    assert r.values["progress_ratio"] == pytest.approx(0.0)


def test_object_avoidance_without_neighbors_is_not_applicable():
    xy, yaw = straight_path(200, 5.0)
    assert _avoidance(xy, yaw, xy, yaw).passed is None
    assert _avoidance(xy[:5], yaw[:5], xy, yaw).passed is None
