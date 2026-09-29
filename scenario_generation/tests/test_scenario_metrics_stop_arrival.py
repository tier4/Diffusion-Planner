import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.stop_arrival import GOAL_REACH_M
from scenario_generation.scenario_metrics.testing import (
    make_input,
    speed_profile_path,
    straight_path,
)

ANCHOR = 10


def _profile(moving: int, stopped: int, then_moving: int = 0, v: float = 5.0) -> np.ndarray:
    return np.r_[np.full(moving, v), np.zeros(stopped), np.full(then_moving, v)]


def _stop_input(ego_speeds, rec_speeds=None, label="traffic_light_stop", **kw):
    """Human drives 5 m/s for 50 frames (25 m), then stops for the rest of the window."""
    rec_xy, rec_yaw = speed_profile_path(_profile(50, 100) if rec_speeds is None else rec_speeds)
    ego_xy, ego_yaw = speed_profile_path(ego_speeds)
    return make_input(
        label=label,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        **kw,
    )


@pytest.mark.parametrize("label", ["traffic_light_stop", "obstacle_stop"])
def test_stop_matching_the_human_passes(label):
    r = registry.score(_stop_input(_profile(50, 100), label=label))
    assert r.metric == "stop_overshoot" and r.passed is True
    assert r.values["human_stop_s_m"] == pytest.approx(25.0)
    assert r.values["overshoot_m"] == pytest.approx(0.0)


def test_stop_short_passes_and_reports_undershoot():
    r = registry.score(_stop_input(_profile(44, 100)))  # 3 m short
    assert r.passed is True
    assert r.values["undershoot_m"] == pytest.approx(3.0)
    assert r.values["overshoot_m"] == 0.0


def test_stop_past_tolerance_fails():
    # Overshoot is visible even though the recorded path ends at the human's stop.
    r = registry.score(_stop_input(_profile(56, 100)))  # 3 m past
    assert r.passed is False
    assert r.values["overshoot_m"] == pytest.approx(3.0)
    assert r.values["ego_sustained_stop"] == 1.0


def test_stop_within_tolerance_passes():
    r = registry.score(_stop_input(_profile(51, 100)))  # 0.5 m past == tolerance
    assert r.passed is True and r.values["overshoot_m"] == pytest.approx(0.5)


def test_ego_that_never_stops_fails_with_its_furthest_point():
    r = registry.score(_stop_input(np.full(150, 5.0), terminated="max_steps"))
    assert r.passed is False
    assert r.values["ego_sustained_stop"] == 0.0
    assert r.values["overshoot_m"] == pytest.approx(149 * 0.5 - 25.0)


def test_ego_stopping_then_driving_on_after_release_passes():
    rec = _profile(50, 40, 60)
    r = registry.score(_stop_input(_profile(50, 40, 60), rec_speeds=rec))
    assert r.passed is True and r.values["overshoot_m"] == pytest.approx(0.0)


def test_human_never_stopping_is_not_scored():
    r = registry.score(_stop_input(np.full(150, 5.0), rec_speeds=np.full(150, 5.0)))
    assert r.passed is None and "never stops" in r.reason


def test_goal_termination_before_the_stop_is_not_scored():
    # Rollout ends GOAL_REACH_M short of the human stop (= window end), still moving.
    n = int((25.0 - GOAL_REACH_M) / 0.5)
    r = registry.score(_stop_input(np.full(n, 5.0), terminated="goal"))
    assert r.passed is None and "goal" in r.reason
    assert r.values["shortfall_to_human_stop_m"] > GOAL_REACH_M - 0.5


def test_stop_anchor_never_reached_is_not_scored():
    r = registry.score(_stop_input(np.full(5, 5.0)))
    assert r.passed is None and "never replayed" in r.reason


def test_traffic_light_stop_reports_red_light_steps_after_anchor_only():
    red = np.zeros(150, dtype=bool)
    red[[3, 60, 61]] = True  # step 3 is before the anchor
    r = registry.score(_stop_input(_profile(50, 100), red_light_violation=red))
    assert r.values["red_light_violation_steps"] == 2.0
    assert r.details["first_red_light_violation_step"] == 60
    assert r.passed is True  # verdict is the overshoot criterion
    assert (
        "red_light_violation_steps"
        not in registry.score(_stop_input(_profile(50, 100), label="obstacle_stop")).values
    )


# --- arrival ---------------------------------------------------------------------


def _arrival_input(ego_xy, ego_yaw, terminated="goal"):
    rec_xy, rec_yaw = straight_path(100, 5.0)  # endpoint at x = 49.5
    return make_input(
        label="arrival",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        terminated=terminated,
    )


def test_arrival_at_the_endpoint_passes():
    xy, yaw = straight_path(100, 5.0)
    r = registry.score(_arrival_input(xy, yaw))
    assert r.metric == "arrival" and r.passed is True
    assert r.values["closest_distance_m"] == pytest.approx(0.0)


@pytest.mark.parametrize("short_m", [3.0, 4.5])
def test_arrival_goal_terminated_short_of_the_endpoint_passes(short_m):
    xy, yaw = straight_path(100, 5.0)
    keep = xy[:, 0] <= 49.5 - short_m
    r = registry.score(_arrival_input(xy[keep], yaw[keep]))
    assert r.passed is True
    assert r.values["longitudinal_shortfall_m"] == pytest.approx(r.values["closest_distance_m"])
    assert r.values["longitudinal_shortfall_m"] >= short_m


def test_arrival_laterally_off_fails():
    xy, yaw = straight_path(100, 5.0)
    xy = xy[:92] + [0.0, 3.0]
    r = registry.score(_arrival_input(xy, yaw[:92]))
    assert r.passed is False and not r.details["lateral_within_tolerance"]
    assert r.values["lateral_offset_m"] == pytest.approx(3.0)


def test_arrival_heading_off_fails():
    xy, yaw = straight_path(100, 5.0)
    r = registry.score(_arrival_input(xy[:92], yaw[:92] + np.radians(20.0)))
    assert r.passed is False and r.values["heading_error_deg"] == pytest.approx(20.0)


def test_arrival_never_reaching_the_endpoint_fails():
    xy, yaw = straight_path(100, 5.0)
    xy[60:] = xy[60]  # stops 19.5 m short
    r = registry.score(_arrival_input(xy, yaw, terminated="max_steps"))
    assert r.passed is False and not r.details["reached"]
    assert r.values["closest_distance_m"] == pytest.approx(19.5)


def test_arrival_anchor_never_reached_is_not_scored():
    xy, yaw = straight_path(5, 5.0)
    r = registry.score(_arrival_input(xy, yaw))
    assert r.passed is None
