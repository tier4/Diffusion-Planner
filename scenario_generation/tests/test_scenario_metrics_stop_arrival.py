import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.stop_arrival import (
    GOAL_REACH_M,
    TEMPORAL_STOP_TOLERANCE_M,
)
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


@pytest.mark.parametrize("label", ["traffic_light_stop", "obstacle_stop", "temporal_stop"])
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


def _route_frames(lanes: np.ndarray | None = None) -> dict[int, dict[str, np.ndarray]]:
    """Anchor frame with one straight route lane along its +x, from 10 m behind the ego."""
    if lanes is None:
        x = np.linspace(-10.0, 90.0, 21)
        lanes = np.stack([x, np.zeros_like(x)], axis=-1)[None]
    return {ANCHOR: {"route_lanes": lanes}}


def test_open_loop_stop_reference_uses_the_final_stop():
    # The human stops at 25 m for 1 s, creeps 2 m and stops again at 27 m; the ego drives
    # straight to 27 m. Closed loop measures the first human stop (fail, 2 m over), open
    # loop the final one within 8 s (pass). The anchor pose is at 5 m, the route lane
    # starts 10 m behind it, so route s = path arc + 5 m.
    rec = np.r_[_profile(50, 10, 4), np.zeros(86)]
    r = registry.score(_stop_input(_profile(54, 96), rec_speeds=rec, frames=_route_frames()))
    assert r.passed is False
    assert r.values["human_stop_s_m"] == pytest.approx(25.0)
    assert r.values["overshoot_m"] == pytest.approx(2.0)
    assert r.values["ol_gt_stop_s_m"] == pytest.approx(27.0 + 5.0)
    assert r.values["ol_ego_stop_s_m"] == pytest.approx(27.0 + 5.0)
    assert r.values["ol_stop_overshoot_m"] == pytest.approx(0.0)
    assert r.values["ol_gt_sustained_stop"] == 1.0 and r.values["ol_ego_sustained_stop"] == 1.0
    assert r.values["ol_passed"] == 1.0


def test_open_loop_stop_reference_leaves_verdicts_and_values_unchanged():
    ego = _profile(56, 100)  # 3 m past
    plain = registry.score(_stop_input(ego))
    ref = registry.score(_stop_input(ego, frames=_route_frames()))
    assert not any(k.startswith("ol_") for k in plain.values)
    assert ref.passed is plain.passed is False and ref.reason == plain.reason
    assert ref.details == plain.details
    assert {k: v for k, v in ref.values.items() if not k.startswith("ol_")} == plain.values
    assert ref.values["ol_stop_overshoot_m"] == pytest.approx(3.0)
    assert ref.values["ol_passed"] == 0.0
    # A frame without a usable route lane: omitted, nothing raised.
    empty = registry.score(_stop_input(ego, frames=_route_frames(np.zeros((2, 20, 4)))))
    assert empty.values == plain.values


# --- arrival ---------------------------------------------------------------------

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


def test_open_loop_arrival_reference_uses_the_final_pose():
    # Goal-terminated 3 m short: closed loop passes, open loop's FDE fails on the goal radius.
    xy, yaw = straight_path(100, 5.0)
    keep = xy[:, 0] <= 49.5 - 3.0
    r = registry.score(_arrival_input(xy[keep], yaw[keep] + np.radians(5.0)))
    assert r.passed is True
    assert r.values["ol_final_displacement_error_m"] == pytest.approx(3.0)
    assert r.values["ol_final_heading_error_deg"] == pytest.approx(5.0)
    assert r.values["ol_passed"] == 0.0
    at_end = registry.score(_arrival_input(xy, yaw))
    assert at_end.passed is True and at_end.values["ol_passed"] == 1.0
    assert at_end.values["ol_final_displacement_error_m"] == pytest.approx(0.0)


def test_temporal_stop_is_a_stop_with_the_closed_loop_tolerance():
    # The human stops at the line and moves on: a stop past the 0.5 m tolerance fails
    # even though the trace ends well past the stop line.
    human = _profile(50, 20, 80)
    rolled_through = registry.score(_stop_input(_profile(56, 20, 74), human, label="temporal_stop"))
    assert rolled_through.metric == "stop_overshoot" and rolled_through.passed is False
    assert rolled_through.values["tolerance_m"] == TEMPORAL_STOP_TOLERANCE_M
    assert rolled_through.values["overshoot_m"] == pytest.approx(3.0)
    held = registry.score(_stop_input(_profile(50, 20, 80), human, label="temporal_stop"))
    assert held.passed is True


def _extended_arrival(ego_speeds, terminated="goal"):
    """Bus drives 5 m/s, holds at the stop (x = 25 m) for 3 s, then moves on 25 m."""
    rec_xy, rec_yaw = speed_profile_path(_profile(50, 30, 50))
    ego_xy, ego_yaw = speed_profile_path(ego_speeds)
    return make_input(
        label="arrival",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        terminated=terminated,
    )


def test_extended_arrival_is_scored_at_the_humans_stop():
    r = registry.score(_extended_arrival(_profile(50, 30, 50)))
    assert r.passed is True and r.details["arrival_mode"] == "stop_at_arrival_point"
    assert r.values["arrival_point_s_m"] == pytest.approx(25.0)
    assert r.values["stop_distance_m"] == pytest.approx(0.0)
    assert "ol_passed" not in r.values


def test_extended_arrival_stopping_away_or_driving_through_fails():
    short = registry.score(_extended_arrival(_profile(42, 30, 58)))  # 4 m short
    assert short.passed is False
    assert short.values["longitudinal_offset_m"] == pytest.approx(-4.0)
    assert short.reason == "ego stopped away from the arrival point"
    through = registry.score(_extended_arrival(np.full(130, 5.0)))
    assert through.passed is False and through.reason == "ego never stopped after the anchor"


def test_extended_arrival_trace_ending_before_the_point_is_not_scored():
    r = registry.score(_extended_arrival(np.full(30, 5.0), terminated="max_steps"))
    assert r.passed is None
    assert r.reason.startswith("trace ended (max_steps) before the ego reached")


def test_a_jittering_stopped_ego_still_counts_as_stopped():
    # Stopped at the human's stop, but the pose jitters 8 cm back and forth each step and
    # the logged speed swings across the 0.5 m/s threshold, as in closed-loop rollouts.
    ego_xy, ego_yaw = speed_profile_path(_profile(50, 100))
    ego_xy[50:, 0] += np.where(np.arange(100) % 2, 0.08, 0.0)
    swinging = np.where(np.arange(150) % 2, 0.8, 0.1)
    r = registry.score(
        make_input(
            label="traffic_light_stop",
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            rec_xy=speed_profile_path(_profile(50, 100))[0],
            rec_yaw=np.zeros(150),
            anchor_frame=ANCHOR,
            ego_speed=swinging,
        )
    )
    assert r.passed is True and r.values["ego_sustained_stop"] == 1.0
    assert r.values["overshoot_m"] < 0.1


def test_creeping_past_the_line_after_stopping_fails():
    # Stops 1 m short, then creeps on at 0.3 m/s (still a "stop" by speed) to 2 m past.
    ego = np.r_[np.full(48, 5.0), np.zeros(10), np.full(100, 0.3)]
    r = registry.score(_stop_input(ego))
    assert r.passed is False and r.values["ego_sustained_stop"] == 1.0
    assert r.values["overshoot_m"] > 1.5


def test_a_brief_stop_short_of_the_line_then_rolling_through_fails():
    ego = np.r_[np.full(44, 5.0), np.zeros(10), np.full(96, 5.0)]  # stops 3 m short
    r = registry.score(_stop_input(ego))
    assert r.passed is False
    assert r.values["overshoot_m"] > 10.0


def _line_frames(line_x_ahead: float):
    """The human's stop frame (99, pose x = 25 m) with a stop line ``line_x_ahead`` metres
    ahead of its rear axle, across the lane; ego front 3 m ahead of the axle."""
    line = np.array([[[line_x_ahead, -2.0], [line_x_ahead, 2.0]]] + [[[0.0, 0.0]] * 2] * 2)
    return {99: {"stop_lines": line, "ego_shape": np.array([2.0, 4.0, 2.0])}}


def _line_input(line_x_ahead: float):
    """Human stops at x = 25 m; the ego rolls on 1.8 m at 1 m/s and stops at 26.8 m."""
    ego_xy, ego_yaw = speed_profile_path(np.r_[np.full(50, 5.0), np.full(18, 1.0), np.zeros(82)])
    return make_input(
        label="traffic_light_stop",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=speed_profile_path(_profile(50, 100))[0],
        rec_yaw=np.zeros(150),
        anchor_frame=ANCHOR,
        frames=_line_frames(line_x_ahead),
    )


def test_stop_line_labels_are_judged_by_the_front_against_the_line():
    # Line at x = 30 m, 2 m ahead of the human's front (28 m). The ego's front stops at
    # 29.8 m: short of the line, though 1.8 m past the human's stop.
    r = registry.score(_line_input(5.0))
    assert r.details["stop_reference"] == "stop_line"
    assert r.values["stop_line_s_m"] == pytest.approx(30.0)
    assert r.values["human_front_past_line_m"] == pytest.approx(-2.0)
    assert r.values["past_human_stop_m"] == pytest.approx(1.8)
    assert r.passed is True and r.values["overshoot_m"] == 0.0


def test_stop_line_crossing_by_the_front_fails():
    # Line at x = 28.5 m: the same stop puts the ego's front 1.3 m past it.
    r = registry.score(_line_input(3.5))
    assert r.passed is False
    assert r.values["overshoot_m"] == pytest.approx(1.3)


def test_without_a_stop_line_the_human_stop_is_the_reference():
    r = registry.score(_stop_input(_profile(50, 100)))
    assert r.details["stop_reference"] == "human_stop"
    obstacle = registry.score(
        make_input(
            label="obstacle_stop",
            ego_xy=speed_profile_path(_profile(50, 100))[0],
            ego_yaw=np.zeros(150),
            rec_xy=speed_profile_path(_profile(50, 100))[0],
            rec_yaw=np.zeros(150),
            anchor_frame=ANCHOR,
            frames=_line_frames(3.5),
        )
    )
    assert obstacle.details["stop_reference"] == "human_stop"  # not a stop-line label
