import dataclasses

import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.stop_arrival import (
    ARRIVAL_STOP_SEARCH_M,
    GOAL_REACH_M,
    HOLD_CREEP_MARGIN_M,
    QUEUED_REASON,
    STOP_FOR_LINE_MAX_SHORT_M,
    TEMPORAL_STOP_MIN_HOLD_S,
    TEMPORAL_STOP_TOLERANCE_M,
    TRAFFIC_LIGHT_MIN_STOP_S,
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
    # The human stops at the line and moves on: a stop past the tolerance fails
    # even though the trace ends well past the stop line.
    human = _profile(50, 20, 80)
    rolled_through = registry.score(_stop_input(_profile(56, 20, 74), human, label="temporal_stop"))
    assert rolled_through.metric == "stop_overshoot" and rolled_through.passed is False
    assert rolled_through.values["tolerance_m"] == TEMPORAL_STOP_TOLERANCE_M
    assert rolled_through.values["overshoot_m"] == pytest.approx(3.0)
    held = registry.score(_stop_input(_profile(50, 20, 80), human, label="temporal_stop"))
    assert held.passed is True


def test_temporal_stop_rolling_stop_fails():
    # The ego rests 0.8 s at the line (a 0.5 s sustained stop) and moves on as the human
    # departs: a stop by ``sustained_stop_s`` and never past the line while the human
    # holds, yet too short to be a temporal stop.
    human = _profile(50, 8, 92)
    ego = np.r_[np.full(50, 5.0), np.zeros(8), np.full(92, 2.0)]
    r = registry.score(_stop_input(ego, human, label="temporal_stop"))
    assert r.passed is False and r.values["ego_sustained_stop"] == 0.0
    # The same rest is enough for obstacle_stop, which keeps the 0.5 s rule.
    assert registry.score(_stop_input(ego, human, label="obstacle_stop")).passed is True


def test_temporal_stop_held_at_the_line_then_departing_passes():
    # The human holds 4 s; the ego rests 1.9 s at the line (a 1.5 s sustained stop) and
    # departs while the human still holds. Its departure is not overshoot.
    human = _profile(50, 40, 60)
    ego = _profile(50, 19, 81)
    r = registry.score(_stop_input(ego, human, label="temporal_stop"))
    stop = r.details["ego_stop_steps"]
    assert stop[1] - stop[0] == 15 and 1.5 > TEMPORAL_STOP_MIN_HOLD_S
    assert r.passed is True and r.values["overshoot_m"] == pytest.approx(0.0)
    # traffic_light_stop judges the first stop; leaving it while the human holds fails
    # only the hold.
    red = registry.score(_stop_input(ego, human))
    assert red.passed is True and red.values["hold_passed"] == 0.0


def test_temporal_stop_held_past_the_line_fails():
    human = _profile(50, 40, 60)
    r = registry.score(_stop_input(_profile(53, 20, 77), human, label="temporal_stop"))
    assert r.passed is False and r.values["overshoot_m"] == pytest.approx(1.5)


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


def test_creeping_past_the_line_after_stopping_passes_and_fails_the_hold():
    # Stops 1 m short, then creeps on at 0.3 m/s (still a "stop" by speed) to 2 m past.
    ego = np.r_[np.full(48, 5.0), np.zeros(10), np.full(100, 0.3)]
    r = registry.score(_stop_input(ego))
    assert r.passed is True and r.values["ego_sustained_stop"] == 1.0
    assert r.values["hold_creep_m"] > 1.5 + HOLD_CREEP_MARGIN_M
    assert r.values["human_hold_creep_m"] == pytest.approx(0.0)
    assert r.values["max_hold_creep_m"] == pytest.approx(HOLD_CREEP_MARGIN_M)
    assert r.values["hold_passed"] == 0.0


def test_a_brief_stop_short_of_the_line_then_rolling_through_fails():
    # Rests 1 s 3 m short (a 0.6 s sustained stop): a rolling stop, not the first stop.
    ego = np.r_[np.full(44, 5.0), np.zeros(10), np.full(96, 5.0)]
    r = registry.score(_stop_input(ego))
    assert r.passed is False
    assert r.reason == "ego passed the stop line without stopping at it"
    assert np.isnan(r.values["first_stop_overshoot_m"])
    assert r.values["required_stop_s"] == TRAFFIC_LIGHT_MIN_STOP_S


def test_traffic_light_stop_holding_behind_the_line_passes_the_hold():
    r = registry.score(_stop_input(_profile(50, 100)))
    assert r.passed is True
    assert r.values["hold_creep_m"] == pytest.approx(0.0)
    assert r.values["hold_passed"] == 1.0


def test_traffic_light_stop_first_stop_past_the_limit_fails():
    r = registry.score(_stop_input(_profile(52, 100)))  # 1 m past, holds there
    assert r.passed is False
    assert r.values["first_stop_overshoot_m"] == pytest.approx(1.0)
    assert r.values["overshoot_m"] == pytest.approx(1.0)
    assert r.reason == "ego's first stop was past the stop limit"
    assert r.values["hold_passed"] == 1.0  # holding is reported apart from the verdict


def test_traffic_light_stop_never_stopping_fails_and_has_no_hold_verdict():
    r = registry.score(_stop_input(np.full(150, 5.0), terminated="max_steps"))
    assert r.passed is False
    assert r.reason == "ego passed the stop line without stopping at it"
    assert np.isnan(r.values["hold_creep_m"]) and "hold_passed" not in r.values


def test_human_hold_creep_is_the_humans_advance_while_holding():
    # The human stops at 25 m, creeps 1 m at 0.4 m/s (a queue moving up), holds again and
    # departs. The ego stops at 25 m and creeps 1.8 m: within 1 m + the margin.
    human = np.r_[np.full(50, 5.0), np.zeros(20), np.full(25, 0.4), np.zeros(30), np.full(25, 5.0)]
    ego = np.r_[np.full(50, 5.0), np.zeros(20), np.full(45, 0.4), np.zeros(35)]
    r = registry.score(_stop_input(ego, human))
    assert r.values["human_hold_creep_m"] == pytest.approx(1.0)
    assert r.values["max_hold_creep_m"] == pytest.approx(1.0 + HOLD_CREEP_MARGIN_M)
    assert r.values["hold_creep_m"] == pytest.approx(1.8, abs=0.05)
    assert r.passed is True and r.values["hold_passed"] == 1.0
    # The same creep without the human's is beyond the margin.
    alone = registry.score(_stop_input(ego))
    assert alone.values["human_hold_creep_m"] == pytest.approx(0.0)
    assert alone.passed is True and alone.values["hold_passed"] == 0.0


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


def _queue_input(line_x_ahead: float, lead_xy: tuple[float, float], lead_speed: float = 0.0):
    """``_line_input`` with one vehicle at ``lead_xy`` (relative to the human's stop pose)
    in the stop frame, moving along +x at ``lead_speed``."""
    inp = _line_input(line_x_ahead)
    frame = _line_frames(line_x_ahead)[99]
    t = np.arange(-30, 1) * 0.1
    past = np.zeros((2, 31, 4))
    past[0, :, 0] = lead_xy[0] + lead_speed * t
    past[0, :, 1] = lead_xy[1]
    past[0, :, 2] = 1.0
    frame["neighbor_agents_past"] = past
    frame["agent_label"] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return dataclasses.replace(inp, load_frame=lambda i: {99: frame}[i])


def test_traffic_light_stop_behind_a_queued_vehicle_is_not_scored():
    # Line 9 m ahead of the axle (6 m ahead of the human's front), a car stopped 6.5 m
    # ahead in the lane: the human stopped behind it, not at the line.
    r = registry.score(_queue_input(9.0, (6.5, 0.2)))
    assert r.passed is None and r.reason == QUEUED_REASON
    assert r.values["queued"] == 1.0
    assert r.values["human_front_to_line_m"] == pytest.approx(6.0)
    assert "first_stop_overshoot_m" in r.values  # the values are kept


def test_traffic_light_stop_queue_ignores_the_next_lane_and_moving_vehicles():
    line_only = registry.score(_line_input(9.0))
    assert "queued" not in line_only.values  # no neighbor data, no check
    for lead_xy, speed in (((6.5, -3.0), 0.0), ((6.5, 0.0), 5.0)):
        r = registry.score(_queue_input(9.0, lead_xy, speed))
        assert r.values["queued"] == 0.0
        assert r.passed == line_only.passed and r.reason == line_only.reason


def test_extended_arrival_reports_the_share_of_the_dwell_the_ego_stayed():
    # The bus holds at the stop over recorded frames 50..80 (3 s). The ego stops there
    # too; the replay either plays the dwell out one frame per step (it stays) or jumps
    # past it after 1 s (it edged on and pulled the recording forward).
    rec_xy, rec_yaw = speed_profile_path(_profile(50, 30, 50))
    ego_xy, ego_yaw = speed_profile_path(_profile(50, 30, 50))
    stayed = make_input(
        label="arrival",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
    )
    r = registry.score(stayed)
    assert r.passed is True and r.values["wait_ratio"] == pytest.approx(1.0)
    jumped = np.r_[np.arange(60), np.arange(81, 81 + 70)]
    left = make_input(
        label="arrival",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        rec_idx=np.minimum(jumped, 129),
    )
    r = registry.score(left)
    # Reported only: the verdict is the first stop.
    assert r.passed is True and r.reason == ""
    assert r.values["wait_ratio"] == pytest.approx(1.0 / 3.0, abs=0.05)


def test_traffic_light_stop_reports_the_share_of_the_red_waited_out():
    # The human waits over frames 50..150 (window end). The ego stops at the human's stop;
    # in one run the replay plays the wait out, in the other it jumps past the wait (and to
    # the window's departure frames) after 2 s, as when the ego edges on.
    human = _profile(50, 60, 40)  # waits over frames 50..110, then departs
    ok = _stop_input(_profile(50, 100), human)
    r = registry.score(ok)
    assert r.passed is True and r.values["wait_ratio"] >= 0.8
    jumped = np.r_[np.arange(70), np.arange(111, 111 + 80)]
    early = _stop_input(_profile(50, 100), human, rec_idx=np.minimum(jumped, 149)[:150])
    r = registry.score(early)
    # Reported only: leaving the red early no longer fails by itself.
    assert r.passed is True and r.reason == ""
    assert r.values["wait_ratio"] < 0.8
    assert r.values["overshoot_m"] == pytest.approx(0.0)
    temporal = registry.score(
        _stop_input(
            _profile(50, 100), human, label="temporal_stop", rec_idx=np.minimum(jumped, 149)[:150]
        )
    )
    assert temporal.passed is True and "wait_ratio" not in temporal.values


# --- first stop vs hold ----------------------------------------------------------


def test_stop_splits_the_first_stop_from_the_creep_while_holding():
    # Stops 1 m short at 24 m, then creeps at 0.3 m/s to 26.97 m while the human holds.
    r = registry.score(_stop_input(np.r_[np.full(48, 5.0), np.zeros(10), np.full(100, 0.3)]))
    assert r.values["first_stop_overshoot_m"] == pytest.approx(-1.0)
    assert r.values["hold_creep_m"] == pytest.approx(2.97)
    # The verdict and the stop values are the first stop's; the creep is the hold's.
    assert r.passed is True and r.reason == ""
    assert r.details["ego_stop_steps"] == [50, 158]
    stop_keys = (
        "tolerance_m",
        "red_light_violation_steps",
        "human_stop_s_m",
        "max_s_after_anchor_m",
        "ego_stop_s_m",
        "overshoot_m",
        "undershoot_m",
        "past_human_stop_m",
        "ego_sustained_stop",
        "human_hold_creep_m",
        "max_hold_creep_m",
        "hold_passed",
    )
    assert {k: r.values[k] for k in stop_keys} == pytest.approx(
        {
            "tolerance_m": 0.5,
            "red_light_violation_steps": 0.0,
            "human_stop_s_m": 25.0,
            "max_s_after_anchor_m": 26.97,
            "ego_stop_s_m": 24.0,
            "overshoot_m": 0.0,
            "undershoot_m": 1.0,
            "past_human_stop_m": -1.0,
            "ego_sustained_stop": 1.0,
            "human_hold_creep_m": 0.0,
            "max_hold_creep_m": HOLD_CREEP_MARGIN_M,
            "hold_passed": 0.0,
        }
    )


def test_first_stop_is_against_the_stop_line_and_nan_without_a_stop():
    line = registry.score(_line_input(5.0))  # rear axle 26.8 m vs line - front = 27 m
    assert line.values["first_stop_overshoot_m"] == pytest.approx(-0.2)
    assert line.values["hold_creep_m"] == pytest.approx(0.0)
    never = registry.score(_stop_input(np.full(150, 5.0), terminated="max_steps"))
    assert np.isnan(never.values["first_stop_overshoot_m"])
    assert np.isnan(never.values["hold_creep_m"])


def test_hold_creep_covers_the_steps_the_verdict_judges():
    # Rests at the line, departs while the human still holds: creep for a red light,
    # departure (not creep) for a temporal stop.
    human, ego = _profile(50, 40, 60), _profile(50, 19, 81)
    red = registry.score(_stop_input(ego, human))
    assert red.values["first_stop_overshoot_m"] == pytest.approx(0.0)
    assert red.values["hold_creep_m"] == pytest.approx(10.0)
    temporal = registry.score(_stop_input(ego, human, label="temporal_stop"))
    assert temporal.values["first_stop_overshoot_m"] == pytest.approx(0.0)
    assert temporal.values["hold_creep_m"] == pytest.approx(0.0)


def test_extended_arrival_splits_the_first_stop_from_the_creep_while_dwelling():
    # Stops at the bus stop (25 m) for 1 s, then creeps at 0.3 m/s until the bus leaves.
    ego = np.r_[np.full(50, 5.0), np.zeros(10), np.full(20, 0.3), np.full(50, 5.0)]
    r = registry.score(_extended_arrival(ego))
    assert r.values["first_stop_distance_m"] == pytest.approx(0.0)
    assert r.values["hold_creep_m"] == pytest.approx(0.57)
    # The verdict and the stop values are the first stop's.
    assert r.passed is True and r.reason == ""
    assert r.values["ego_stop_s_m"] == pytest.approx(25.0)
    assert r.values["stop_distance_m"] == pytest.approx(0.0)
    assert r.values["wait_ratio"] == pytest.approx(1.0)
    assert r.values["human_hold_creep_m"] == pytest.approx(0.0)
    assert r.values["hold_passed"] == 1.0
    matched = registry.score(_extended_arrival(_profile(50, 30, 50)))
    assert matched.values["first_stop_distance_m"] == pytest.approx(0.0)
    assert matched.values["hold_creep_m"] == pytest.approx(0.0)
    through = registry.score(_extended_arrival(np.full(130, 5.0)))
    assert np.isnan(through.values["first_stop_distance_m"])
    assert np.isnan(through.values["hold_creep_m"])


def test_extended_arrival_first_stop_within_tolerance_then_creeping_far_passes():
    # Stops at the bus stop for 1 s, then creeps at 1 m/s while the bus still holds.
    ego = np.r_[np.full(50, 5.0), np.zeros(10), np.full(70, 1.0)]
    r = registry.score(_extended_arrival(ego))
    assert r.passed is True and r.values["first_stop_distance_m"] == pytest.approx(0.0)
    assert r.values["max_hold_creep_m"] == pytest.approx(HOLD_CREEP_MARGIN_M)
    assert r.values["hold_creep_m"] == pytest.approx(1.9, abs=0.1)
    assert r.values["hold_passed"] == 0.0


def test_extended_arrival_first_stop_3m_off_fails_even_if_it_moves_up():
    # Stops 3 m short of the bus stop, then moves up and stops at it.
    ego = np.r_[np.full(44, 5.0), np.zeros(10), np.full(6, 5.0), np.zeros(70)]
    r = registry.score(_extended_arrival(ego))
    assert r.passed is False and r.reason == "ego stopped away from the arrival point"
    assert r.values["first_stop_distance_m"] == pytest.approx(3.0)
    assert r.values["longitudinal_offset_m"] == pytest.approx(-3.0)


# --- which stop is the first stop ------------------------------------------------


def _rest_steps(run_s: float) -> int:
    """Steps at rest giving a sustained-stop run of ``run_s`` (the net-displacement window
    makes the run 0.4 s shorter than the rest)."""
    return round(run_s / 0.1) + 4


def test_traffic_light_brief_stop_far_before_the_line_then_through_fails():
    # A 0.5 s stop 10 m short (behind a lead vehicle, and brief), then through the line.
    ego = np.r_[np.full(30, 5.0), np.zeros(_rest_steps(0.5)), np.full(111, 5.0)]
    r = registry.score(_stop_input(ego))
    assert r.passed is False
    assert r.reason == "ego passed the stop line without stopping at it"
    assert np.isnan(r.values["first_stop_duration_s"])
    assert r.details["ego_stop_steps"] is None


def test_traffic_light_stop_waiting_far_back_through_the_red_then_on_passes():
    # The human waits over frames 50..110. The ego waits 8 m short of the line (too far
    # back to be a stop for it) until the light turns green, then drives through.
    assert 8.0 > STOP_FOR_LINE_MAX_SHORT_M
    human = _profile(50, 60, 40)
    ego = np.r_[np.full(34, 5.0), np.zeros(80), np.full(36, 5.0)]
    r = registry.score(_stop_input(ego, human))
    assert r.passed is True and r.reason == ""
    assert r.values["crossed_on_red"] == 0.0
    assert np.isnan(r.values["first_stop_overshoot_m"])
    assert r.details["first_step_past_limit"] > 110
    assert "human_hold_creep_m" in r.values


def test_traffic_light_stop_brief_stop_far_back_then_through_on_red_fails():
    # A 0.5 s stop 10 m short, then through the line while the human still waits.
    human = _profile(50, 60, 40)
    ego = np.r_[np.full(30, 5.0), np.zeros(_rest_steps(0.5)), np.full(111, 5.0)]
    r = registry.score(_stop_input(ego, human))
    assert r.passed is False
    assert r.reason == "ego passed the stop line without stopping at it"
    assert r.values["crossed_on_red"] == 1.0
    assert r.details["first_step_past_limit"] < 110


def test_traffic_light_stop_held_long_enough_just_before_the_line_passes():
    # A 1.5 s stop with the front 0.3 m short of the line, then on as the light holds.
    ego_xy, ego_yaw = speed_profile_path(
        np.r_[np.full(54, 5.0), np.zeros(_rest_steps(1.5)), np.full(77, 5.0)]
    )
    # Line 5.3 m ahead of the human's axle (25 m): the front (3 m ahead) is to stop by
    # 30.3 m, the axle by 27.3 m; the ego's axle stops at 27 m.
    r = registry.score(
        make_input(
            label="traffic_light_stop",
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            rec_xy=speed_profile_path(_profile(50, 100))[0],
            rec_yaw=np.zeros(150),
            anchor_frame=ANCHOR,
            frames=_line_frames(5.3),
        )
    )
    assert r.passed is True
    assert r.values["first_stop_overshoot_m"] == pytest.approx(-0.3)
    assert r.values["first_stop_duration_s"] == pytest.approx(1.5)


def test_traffic_light_stop_brief_stop_counts_when_the_human_barely_stopped():
    # The human stops 0.6 s (the light turned green on its arrival); a 0.6 s ego stop at
    # the same point is then enough.
    human = _profile(50, 6, 94)
    ego = np.r_[np.full(50, 5.0), np.zeros(_rest_steps(0.6)), np.full(90, 5.0)]
    r = registry.score(_stop_input(ego, human))
    assert r.values["human_wait_s"] == pytest.approx(0.6)
    assert r.values["required_stop_s"] == pytest.approx(0.6)
    assert r.values["first_stop_duration_s"] == pytest.approx(0.6)
    assert r.passed is True and r.values["first_stop_overshoot_m"] == pytest.approx(0.0)
    # Against a long red the same stop is a rolling stop.
    held = registry.score(_stop_input(ego, _profile(50, 60, 40)))
    assert held.passed is False
    assert held.reason == "ego passed the stop line without stopping at it"


def test_traffic_light_stop_skips_a_queue_stop_far_before_the_line():
    # Stops 2 s 8 m short (a queue), moves up and stops 1.5 s at the human's stop.
    assert 8.0 > STOP_FOR_LINE_MAX_SHORT_M
    ego = np.r_[
        np.full(34, 5.0),
        np.zeros(_rest_steps(2.0)),
        np.full(16, 5.0),
        np.zeros(_rest_steps(1.5)),
        np.full(60, 5.0),
    ]
    r = registry.score(_stop_input(ego))
    assert r.passed is True
    assert r.values["first_stop_overshoot_m"] == pytest.approx(0.0)
    assert r.values["first_stop_duration_s"] == pytest.approx(1.5)
    assert r.details["ego_stop_steps"][0] > 50


def test_extended_arrival_skips_a_queue_stop_far_from_the_bus_stop():
    # Stops 6 m short (a queue), then pulls up and stops at the bus stop.
    assert 6.0 > ARRIVAL_STOP_SEARCH_M
    ego = np.r_[np.full(38, 5.0), np.zeros(10), np.full(12, 5.0), np.zeros(70)]
    r = registry.score(_extended_arrival(ego))
    assert r.passed is True and r.reason == ""
    assert r.values["first_stop_distance_m"] == pytest.approx(0.0)
    assert r.details["ego_stop_steps"][0] > 48


def test_extended_arrival_only_a_queue_stop_then_rolling_past_fails():
    ego = np.r_[np.full(38, 5.0), np.zeros(10), np.full(82, 5.0)]
    r = registry.score(_extended_arrival(ego))
    assert r.passed is False and r.reason == "ego stopped away from the arrival point"
    assert r.values["first_stop_distance_m"] == pytest.approx(6.0)
