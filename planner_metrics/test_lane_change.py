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

    assert result.scores["lane_change_success"].item() == 1.0
    assert result.scores["lane_change_completion_ratio"].item() == pytest.approx(1.0)
    assert result.scores["final_lateral_offset_error_m"].item() < 1e-4
    # The ramp clears the 1.75 m boundary exactly halfway through the horizon.
    assert result.scores["lane_change_time_s"].item() == pytest.approx(4.0, abs=0.1)
    assert result.details["lane_change"]["gt_direction"].item() == 1.0
    assert result.details["lane_change"]["left_source_lane"].item() == 1.0
    assert result.details["lane_change"]["reached_gt_lane"].item() == 1.0


def test_lane_change_fails_when_prediction_stays_in_the_source_lane():
    result = _evaluate(_trajectory(0.0), _trajectory(_LANE_WIDTH))

    assert result.scores["lane_change_success"].item() == 0.0
    assert result.scores["lane_change_completion_ratio"].item() == pytest.approx(0.0)
    assert result.details["lane_change"]["left_source_lane"].item() == 0.0
    # Never crossing the boundary reports the full horizon as the change time.
    assert result.scores["lane_change_time_s"].item() == pytest.approx(8.0)


def test_lane_change_fails_when_prediction_changes_to_the_wrong_side():
    result = _evaluate(_trajectory(-_LANE_WIDTH), _trajectory(_LANE_WIDTH))

    assert result.scores["lane_change_success"].item() == 0.0
    assert result.scores["lane_change_completion_ratio"].item() == pytest.approx(0.0)
    assert result.details["lane_change"]["left_source_lane"].item() == 0.0


def test_lane_change_fails_when_prediction_overshoots_past_the_gt_lane():
    result = _evaluate(_trajectory(2 * _LANE_WIDTH), _trajectory(_LANE_WIDTH))

    assert result.details["lane_change"]["left_source_lane"].item() == 1.0
    assert result.details["lane_change"]["reached_gt_lane"].item() == 0.0
    assert result.scores["lane_change_success"].item() == 0.0


def test_lane_change_ignores_longitudinal_lag():
    """A prediction that is simply too slow still completes the lane change."""
    gt = _trajectory(_LANE_WIDTH)
    slow = gt.clone()
    slow[0, :, 0] *= 0.5  # half the forward progress, same lateral behaviour

    result = _evaluate(slow, gt)

    assert result.scores["lane_change_success"].item() == 1.0


def test_lane_change_partial_completion_is_reported_as_a_ratio():
    result = _evaluate(_trajectory(0.4 * _LANE_WIDTH), _trajectory(_LANE_WIDTH))

    assert result.scores["lane_change_completion_ratio"].item() == pytest.approx(0.4, abs=1e-3)
    assert result.scores["lane_change_success"].item() == 0.0


def test_lane_change_batches_independent_samples():
    gt = torch.cat([_trajectory(_LANE_WIDTH), _trajectory(-_LANE_WIDTH)], dim=0)
    prediction = torch.cat([_trajectory(_LANE_WIDTH), _trajectory(0.0)], dim=0)

    result = _evaluate(prediction, gt)

    assert result.scores["lane_change_success"].tolist() == [1.0, 0.0]
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
    assert result.scores["lane_change_success"].item() == 1.0


def test_lane_change_rejects_a_scene_where_the_gt_never_leaves_its_lane():
    with pytest.raises(ValueError, match="not a lane-change scene"):
        _evaluate(_trajectory(_LANE_WIDTH), _trajectory(0.0))


def test_lane_change_requires_lane_boundary_columns():
    lanes = _straight_map()[:, :, :4]

    with pytest.raises(ValueError, match="boundary-offset columns"):
        _evaluate(_trajectory(_LANE_WIDTH), _trajectory(_LANE_WIDTH), lanes=lanes)


def test_lane_change_requires_ground_truth():
    with pytest.raises(ValueError, match="requires ego_agent_future"):
        evaluate_lane_change_with_details(
            _trajectory(_LANE_WIDTH), {"lanes": _straight_map()}, _PARAMETERS
        )
