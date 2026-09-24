"""Tests for the lane-change completion metric."""

import math

import pytest
import torch

from planner_metrics.lane_change import evaluate_lane_change_with_details

_PARAMETERS = {"horizon_seconds": 8.0}
_HALF_WIDTH = 1.75
_LANE_WIDTH = 2 * _HALF_WIDTH
_STEPS = 80


def _straight_lane(center_y: float, x_start: float = -20.0, x_end: float = 120.0) -> torch.Tensor:
    """A straight lanelet along +x at ``center_y``, in the canonical 13-column layout."""
    lane = torch.zeros(20, 13)
    x = torch.linspace(x_start, x_end, 20)
    lane[:, 0] = x
    lane[:, 1] = center_y
    lane[:, 2] = x[1] - x[0]  # centerline tangent
    lane[:, 5] = _HALF_WIDTH  # left boundary offset
    lane[:, 7] = -_HALF_WIDTH  # right boundary offset
    return lane


def _turn_lane(
    start_x: float, start_y: float, radius: float = 15.0, lead_in: float = 2.0
) -> torch.Tensor:
    """A left-turn lanelet: a short straight lead-in along +x, then a quarter arc to +y.

    The lead-in makes the first segment identical in direction to a straight
    lanelet leaving the same point, exactly as a lanelet2 turn branch does.
    """
    lane = torch.zeros(20, 13)
    theta = torch.linspace(0.0, math.pi / 2, 18)
    x = torch.cat(
        [torch.tensor([start_x, start_x + lead_in]), start_x + lead_in + radius * torch.sin(theta)]
    )
    y = torch.cat([torch.tensor([start_y, start_y]), start_y + radius * (1 - torch.cos(theta))])
    lane[:, 0] = x
    lane[:, 1] = y
    lane[:, 5] = _HALF_WIDTH
    lane[:, 7] = -_HALF_WIDTH
    return lane


def _crossing_lane(through_x: float = 0.0, angle_deg: float = 60.0) -> torch.Tensor:
    """A lanelet crossing the ego's road through ``(through_x, 0)`` at ``angle_deg``."""
    lane = torch.zeros(20, 13)
    t = torch.linspace(-30.0, 30.0, 20)
    lane[:, 0] = through_x + t * math.cos(math.radians(angle_deg))
    lane[:, 1] = t * math.sin(math.radians(angle_deg))
    lane[:, 5] = _HALF_WIDTH
    lane[:, 7] = -_HALF_WIDTH
    return lane


def _straight_map() -> torch.Tensor:
    """Three parallel lanes; the ego starts on the middle one."""
    return torch.stack(
        [_straight_lane(0.0), _straight_lane(_LANE_WIDTH), _straight_lane(-_LANE_WIDTH)]
    )


def _trajectory(final_y: float, steps: int = _STEPS) -> torch.Tensor:
    """A trajectory driving straight ahead while ramping laterally to ``final_y``."""
    traj = torch.zeros(1, steps, 4)
    traj[0, :, 0] = torch.linspace(0.0, 100.0, steps)
    traj[0, :, 1] = torch.linspace(0.0, final_y, steps)
    traj[0, :, 2] = 1.0
    return traj


def _evaluate(prediction: torch.Tensor, gt: torch.Tensor, lanes: torch.Tensor | None = None):
    data = {"ego_agent_future": gt, "lanes": _straight_map() if lanes is None else lanes}
    return evaluate_lane_change_with_details(prediction, data, _PARAMETERS)


def test_lane_change_succeeds_when_prediction_reaches_the_gt_lane():
    gt = _trajectory(_LANE_WIDTH)
    result = _evaluate(gt.clone(), gt)

    assert result.scores["success_rate_percent"].item() == 100.0
    assert result.details["lane_change"]["completion_ratio"].item() == pytest.approx(1.0)
    assert result.details["lane_change"]["final_lateral_offset_error_m"].item() < 1e-4
    # The ramp clears the 1.75 m boundary at index 40 of 80, i.e. t = 41 * 0.1 s.
    assert result.details["lane_change"]["lane_change_time_s"].item() == pytest.approx(4.1)
    assert result.details["lane_change"]["gt_direction"].item() == 1.0
    assert result.details["lane_change"]["left_source_lane"].item() == 1.0
    assert result.details["lane_change"]["reached_gt_lane"].item() == 1.0


def test_lane_change_fails_when_prediction_stays_in_the_source_lane():
    result = _evaluate(_trajectory(0.0), _trajectory(_LANE_WIDTH))

    assert result.scores["success_rate_percent"].item() == 0.0
    assert result.details["lane_change"]["completion_ratio"].item() == pytest.approx(0.0)
    assert result.details["lane_change"]["left_source_lane"].item() == 0.0
    # Never crossing the boundary reports the full horizon as the change time.
    assert result.details["lane_change"]["lane_change_time_s"].item() == pytest.approx(8.0)


def test_lane_change_fails_when_prediction_changes_to_the_wrong_side():
    result = _evaluate(_trajectory(-_LANE_WIDTH), _trajectory(_LANE_WIDTH))

    assert result.scores["success_rate_percent"].item() == 0.0
    assert result.details["lane_change"]["completion_ratio"].item() == pytest.approx(0.0)
    assert result.details["lane_change"]["left_source_lane"].item() == 0.0


def test_lane_change_fails_when_prediction_overshoots_past_the_gt_lane():
    result = _evaluate(_trajectory(2 * _LANE_WIDTH), _trajectory(_LANE_WIDTH))

    assert result.details["lane_change"]["left_source_lane"].item() == 1.0
    assert result.details["lane_change"]["reached_gt_lane"].item() == 0.0
    assert result.scores["success_rate_percent"].item() == 0.0


def test_lane_change_ignores_longitudinal_lag():
    """A prediction that is simply too slow still completes the lane change."""
    gt = _trajectory(_LANE_WIDTH)
    slow = gt.clone()
    slow[0, :, 0] *= 0.5  # half the forward progress, same lateral behaviour

    result = _evaluate(slow, gt)

    assert result.scores["success_rate_percent"].item() == 100.0


def test_lane_change_partial_completion_is_reported_as_a_ratio():
    result = _evaluate(_trajectory(0.4 * _LANE_WIDTH), _trajectory(_LANE_WIDTH))

    assert result.details["lane_change"]["completion_ratio"].item() == pytest.approx(0.4, abs=1e-3)
    assert result.scores["success_rate_percent"].item() == 0.0


def test_lane_change_batches_independent_samples():
    gt = torch.cat([_trajectory(_LANE_WIDTH), _trajectory(-_LANE_WIDTH)], dim=0)
    prediction = torch.cat([_trajectory(_LANE_WIDTH), _trajectory(0.0)], dim=0)

    result = _evaluate(prediction, gt)

    assert result.scores["success_rate_percent"].tolist() == [100.0, 0.0]
    assert result.details["lane_change"]["gt_direction"].tolist() == [1.0, -1.0]


def test_lane_change_follows_the_source_lane_through_chained_lanelets():
    """The lateral reference must bend with the lane, not with the first lanelet."""
    bend = math.radians(30.0)
    straight = _straight_lane(0.0, x_start=0.0, x_end=30.0)

    bent = torch.zeros(20, 13)
    distance = torch.linspace(0.0, 70.0, 20)
    bent[:, 0] = 30.0 + distance * math.cos(bend)
    bent[:, 1] = distance * math.sin(bend)
    bent[:, 2] = (distance[1] - distance[0]) * math.cos(bend)
    bent[:, 3] = (distance[1] - distance[0]) * math.sin(bend)
    # Boundary offsets are perpendicular to the bent tangent.
    bent[:, 4] = -_HALF_WIDTH * math.sin(bend)
    bent[:, 5] = _HALF_WIDTH * math.cos(bend)
    bent[:, 6] = _HALF_WIDTH * math.sin(bend)
    bent[:, 7] = -_HALF_WIDTH * math.cos(bend)

    lanes = torch.stack([straight, bent])
    normal = torch.tensor([-math.sin(bend), math.cos(bend)])

    # GT drives the chained lane and ends one lane width to its left.
    gt = torch.zeros(1, _STEPS, 4)
    centers = torch.cat([straight[:, :2], bent[1:, :2]], dim=0)
    walk = torch.linspace(0.0, centers.shape[0] - 1.0, _STEPS)
    low = walk.floor().long().clamp(max=centers.shape[0] - 2)
    frac = (walk - low).unsqueeze(-1)
    gt[0, :, :2] = centers[low] * (1 - frac) + centers[low + 1] * frac
    gt[0, :, :2] += torch.linspace(0.0, _LANE_WIDTH, _STEPS).unsqueeze(-1) * normal

    result = _evaluate(gt.clone(), gt, lanes=lanes)

    shift = result.details["lane_change"]["gt_lateral_shift_m"].item()
    assert shift == pytest.approx(_LANE_WIDTH, abs=0.1)
    assert result.scores["success_rate_percent"].item() == 100.0


def test_lane_change_fails_a_scene_where_the_gt_never_leaves_its_lane():
    """A mis-curated scene fails rather than aborting the whole validation run."""
    result = _evaluate(_trajectory(_LANE_WIDTH), _trajectory(0.0))

    assert result.scores["success_rate_percent"].item() == 0.0
    assert result.details["lane_change"]["gt_lane_change_detected"].item() == 0.0
    assert result.details["lane_change"]["completion_ratio"].item() == 0.0
    assert result.details["lane_change"]["lane_change_time_s"].item() == pytest.approx(8.0)


def test_lane_change_reports_the_precondition_separately_from_the_score():
    """A bad list reads as gt_lane_change_detected, not as a planner failure."""
    gt = torch.cat([_trajectory(_LANE_WIDTH), _trajectory(0.0)], dim=0)
    prediction = torch.cat([_trajectory(_LANE_WIDTH), _trajectory(_LANE_WIDTH)], dim=0)

    result = _evaluate(prediction, gt)

    assert result.details["lane_change"]["gt_lane_change_detected"].tolist() == [1.0, 0.0]
    assert result.scores["success_rate_percent"].tolist() == [100.0, 0.0]


def test_completion_ratio_is_zero_when_the_prediction_holds_its_initial_offset():
    """Progress is measured from where the ego started, not from the lane center."""
    gt = _trajectory(_LANE_WIDTH)
    gt[0, :, 1] += 0.8  # the whole scene starts 0.8 m left of the lane center
    held = _trajectory(0.0)
    held[0, :, 1] = 0.8  # prediction keeps that offset and never changes lane

    result = _evaluate(held, gt)

    assert result.details["lane_change"]["completion_ratio"].item() == pytest.approx(0.0, abs=1e-3)
    assert result.scores["success_rate_percent"].item() == 0.0


def test_completion_ratio_does_not_reward_overshooting_the_target_lane():
    result = _evaluate(_trajectory(2 * _LANE_WIDTH), _trajectory(_LANE_WIDTH))

    # Overshooting by a full lane is as far from the GT target as not moving.
    assert result.details["lane_change"]["completion_ratio"].item() == pytest.approx(0.0, abs=1e-3)


def test_reached_tolerance_survives_a_map_without_boundary_offsets():
    """Zero half widths must fall back to the floor, not demand an exact match."""
    lanes = _straight_map()
    lanes[:, :, 4:8] = 0.0
    gt = _trajectory(_LANE_WIDTH)
    close = _trajectory(_LANE_WIDTH - 0.3)

    result = _evaluate(close, gt, lanes=lanes)

    assert result.details["lane_change"]["gt_lane_change_detected"].item() == 1.0
    assert result.details["lane_change"]["reached_gt_lane"].item() == 1.0
    assert result.scores["success_rate_percent"].item() == 100.0


def test_source_lane_ignores_a_nearer_oncoming_lanelet():
    """The closest lanelet is not the source lane when it runs the other way."""
    # Ego sits 0.5 m off its own lane center, with an oncoming lanelet 0.3 m
    # away on the other side -- so plain nearest-centerline picks the oncoming one.
    source = _straight_lane(-0.5)
    target = _straight_lane(-0.5 + _LANE_WIDTH)
    oncoming = torch.flip(_straight_lane(0.3), dims=[0])
    oncoming[:, 2] *= -1
    lanes = torch.stack([source, target, oncoming])
    gt = _trajectory(-0.5 + _LANE_WIDTH)

    result = _evaluate(gt.clone(), gt, lanes=lanes)

    assert result.details["lane_change"]["source_lane_index"].item() == 0
    assert result.details["lane_change"]["gt_direction"].item() == 1.0
    assert result.scores["success_rate_percent"].item() == 100.0


def test_source_lane_path_chains_when_the_ego_starts_deep_inside_a_long_lanelet():
    """The chaining budget must be measured forward from the ego, not from the path start."""
    bend = math.radians(30.0)
    # A 160 m source lanelet with the ego 150 m in, so only 10 m lies ahead of it.
    behind = _straight_lane(0.0, x_start=-150.0, x_end=10.0)
    ahead = torch.zeros(20, 13)
    distance = torch.linspace(0.0, 120.0, 20)
    ahead[:, 0] = 10.0 + distance * math.cos(bend)
    ahead[:, 1] = distance * math.sin(bend)
    ahead[:, 2] = (distance[1] - distance[0]) * math.cos(bend)
    ahead[:, 3] = (distance[1] - distance[0]) * math.sin(bend)
    ahead[:, 4] = -_HALF_WIDTH * math.sin(bend)
    ahead[:, 5] = _HALF_WIDTH * math.cos(bend)
    ahead[:, 6] = _HALF_WIDTH * math.sin(bend)
    ahead[:, 7] = -_HALF_WIDTH * math.cos(bend)
    lanes = torch.stack([behind, ahead])

    # The GT starts at the ego (the origin), not at the far end of the lanelet.
    forward = behind[behind[:, 0] > 0.0][:, :2]
    centers = torch.cat([torch.zeros(1, 2), forward, ahead[1:, :2]], dim=0)
    normal = torch.tensor([-math.sin(bend), math.cos(bend)])
    walk = torch.linspace(0.0, centers.shape[0] - 1.0, _STEPS)
    low = walk.floor().long().clamp(max=centers.shape[0] - 2)
    frac = (walk - low).unsqueeze(-1)
    gt = torch.zeros(1, _STEPS, 4)
    gt[0, :, :2] = centers[low] * (1 - frac) + centers[low + 1] * frac
    gt[0, :, :2] += torch.linspace(0.0, _LANE_WIDTH, _STEPS).unsqueeze(-1) * normal

    result = _evaluate(gt.clone(), gt, lanes=lanes)

    # Measuring the budget against the whole path counts the 150 m behind the
    # ego, stops chaining, and extrapolates the bend as a straight line.
    assert result.details["lane_change"]["gt_lateral_shift_m"].item() == pytest.approx(
        _LANE_WIDTH, abs=0.3
    )
    assert result.scores["success_rate_percent"].item() == 100.0


def test_lane_change_requires_lane_boundary_columns():
    lanes = _straight_map()[:, :, :4]

    with pytest.raises(ValueError, match="boundary-offset columns"):
        _evaluate(_trajectory(_LANE_WIDTH), _trajectory(_LANE_WIDTH), lanes=lanes)


def test_lane_change_requires_ground_truth():
    with pytest.raises(ValueError, match="requires ego_agent_future"):
        evaluate_lane_change_with_details(
            _trajectory(_LANE_WIDTH), {"lanes": _straight_map()}, _PARAMETERS
        )


def test_source_lane_chaining_follows_the_straight_branch_at_a_fork():
    """A turn lanelet leaves the fork tangent to the straight one; the whole-lanelet
    direction, not the first segment, must decide which successor is chained."""
    fork_x = 20.0
    lanes = torch.stack(
        [
            _straight_lane(0.0, -20.0, fork_x),  # source: ends at the fork
            _turn_lane(fork_x, 0.0),  # listed BEFORE the straight successor
            _straight_lane(0.0, fork_x, 120.0),
            _straight_lane(_LANE_WIDTH, -20.0, 120.0),
            _straight_lane(-_LANE_WIDTH, -20.0, 120.0),
        ]
    )
    gt = _trajectory(_LANE_WIDTH)
    result = _evaluate(gt.clone(), gt, lanes)

    details = result.details["lane_change"]
    assert details["source_lane_index"].item() == 0
    # Measured against a straight source path the GT moved exactly one lane;
    # against the turn branch it would read as tens of meters.
    assert details["gt_lateral_shift_m"].item() == pytest.approx(_LANE_WIDTH, abs=0.05)
    assert result.scores["success_rate_percent"].item() == 100.0


def test_source_lane_prefers_the_straight_branch_when_the_ego_sits_on_a_fork():
    fork_x = -1.0
    lanes = torch.stack(
        [
            _turn_lane(fork_x, 0.0),  # listed first, same distance as the straight lane
            _straight_lane(0.0, fork_x, 120.0),
            _straight_lane(_LANE_WIDTH, -20.0, 120.0),
        ]
    )
    gt = _trajectory(_LANE_WIDTH)
    result = _evaluate(gt.clone(), gt, lanes)

    assert result.details["lane_change"]["source_lane_index"].item() == 1
    assert result.details["lane_change"]["gt_lateral_shift_m"].item() == pytest.approx(
        _LANE_WIDTH, abs=0.05
    )


def test_source_lane_rejects_a_crossing_lanelet_through_the_ego_position():
    """A crossing road's lanelet passing through the origin is nearer than the
    ego's own centerline, but its heading rules it out."""
    lanes = torch.stack(
        [
            _crossing_lane(),  # distance 0 from the ego, 60 deg off heading
            _straight_lane(0.3),  # the ego's lane, 0.3 m to the left
            _straight_lane(0.3 + _LANE_WIDTH),
        ]
    )
    gt = _trajectory(_LANE_WIDTH)
    result = _evaluate(gt.clone(), gt, lanes)

    assert result.details["lane_change"]["source_lane_index"].item() == 1
    assert result.scores["success_rate_percent"].item() == 100.0


def test_lane_change_time_counts_the_first_future_point_as_one_timestep():
    """Prediction index 0 is already 0.1 s after t=0."""
    gt = _trajectory(_LANE_WIDTH)
    prediction = gt.clone()
    prediction[0, :, 1] = _LANE_WIDTH  # in the target lane from the very first point
    result = _evaluate(prediction, gt)

    assert result.details["lane_change"]["lane_change_time_s"].item() == pytest.approx(0.1)


def test_stationary_off_center_gt_does_not_divide_by_zero():
    """GT parked outside the source lane has zero lateral progress; the sample
    must be scored, not crash the run."""
    gt = _trajectory(0.0)
    gt[0, :, 1] = _HALF_WIDTH + 0.5  # constant offset beyond the lane boundary
    result = _evaluate(gt.clone(), gt)

    assert result.details["lane_change"]["gt_lane_change_detected"].item() == 1.0
    assert result.details["lane_change"]["completion_ratio"].item() == 0.0


def test_source_lane_flags_a_heading_rejected_fallback():
    """With only an oncoming lanelet available the scorer still runs, but marks
    the reference lane as not heading-aligned."""
    oncoming = _straight_lane(0.0).flip(0)  # same geometry, opposite direction
    lanes = torch.stack([oncoming, _straight_lane(_LANE_WIDTH).flip(0)])
    gt = _trajectory(_LANE_WIDTH)
    result = _evaluate(gt.clone(), gt, lanes)

    assert result.details["lane_change"]["source_lane_heading_aligned"].item() == 0.0


def test_source_lane_is_flagged_heading_aligned_on_a_normal_map():
    gt = _trajectory(_LANE_WIDTH)
    result = _evaluate(gt.clone(), gt)

    assert result.details["lane_change"]["source_lane_heading_aligned"].item() == 1.0
