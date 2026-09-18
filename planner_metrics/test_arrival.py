"""Tests for final arrival errors."""

import math

import pytest
import torch

from planner_metrics.arrival import (
    compute_final_displacement_error_batch,
    compute_final_heading_error_batch,
    evaluate_arrival_with_details,
)


def _prediction(final_x: float, final_y: float, final_yaw: float) -> torch.Tensor:
    prediction = torch.zeros(1, 3, 4)
    prediction[0, -1, :2] = torch.tensor([final_x, final_y])
    prediction[0, -1, 2:] = torch.tensor([math.cos(final_yaw), math.sin(final_yaw)])
    return prediction


def test_arrival_compares_final_position_and_heading_against_legacy_gt():
    gt = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, math.pi / 2]]])
    prediction = _prediction(13.0, 4.0, math.pi)

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    details = result.details["arrival"]
    assert torch.allclose(details["final_displacement_error_m"], torch.tensor([5.0]))
    assert torch.allclose(details["final_heading_error_deg"], torch.tensor([90.0]))
    assert torch.allclose(details["final_heading_error_rad"], torch.tensor([math.pi / 2]))
    assert result.scores["success_rate_percent"].tolist() == [0.0]


def test_arrival_wraps_final_heading_error_at_pi_boundary():
    gt = torch.zeros(1, 3, 3)
    gt[0, -1, 2] = math.radians(179.0)
    prediction = _prediction(0.0, 0.0, math.radians(-179.0))

    assert torch.allclose(
        compute_final_displacement_error_batch(prediction, {"ego_agent_future": gt}),
        torch.tensor([0.0]),
    )
    assert torch.allclose(
        compute_final_heading_error_batch(prediction, {"ego_agent_future": gt}),
        torch.tensor([2.0]),
        atol=1e-5,
    )


def test_arrival_compares_the_common_horizon_when_lengths_differ():
    """A GT horizon longer than the prediction must not be compared end to end."""
    gt = torch.zeros(1, 5, 3)
    gt[0, 2, :2] = torch.tensor([10.0, 0.0])  # where the GT is at the prediction's last step
    gt[0, -1, :2] = torch.tensor([30.0, 0.0])  # two steps further on
    prediction = _prediction(13.0, 4.0, 0.0)  # 3 steps, ending at (13, 4)

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    assert torch.allclose(
        result.details["arrival"]["final_displacement_error_m"], torch.tensor([5.0])
    )


def test_arrival_success_requires_both_position_and_heading_within_tolerance():
    gt = torch.zeros(1, 3, 3)
    gt[0, -1, :2] = torch.tensor([10.0, 0.0])
    parameters = {"position_tolerance_m": 2.0, "heading_tolerance_deg": 10.0}

    def score(prediction: torch.Tensor) -> float:
        return (
            evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, parameters)
            .scores["success_rate_percent"]
            .item()
        )

    assert score(_prediction(11.0, 1.0, math.radians(5.0))) == 100.0  # 1.41 m, 5 deg
    assert score(_prediction(12.5, 0.0, 0.0)) == 0.0  # 2.5 m off
    assert score(_prediction(10.0, 0.0, math.radians(15.0))) == 0.0  # 15 deg off
    assert score(_prediction(11.9, 0.0, math.radians(9.0))) == 100.0  # both just inside


def test_arrival_rejects_an_ambiguous_column_layout():
    """A 5-column GT cannot be told apart from [x, y, heading] plus extras."""
    gt = torch.zeros(1, 3, 5)

    with pytest.raises(ValueError, match="arrival ground truth must have shape"):
        evaluate_arrival_with_details(_prediction(0.0, 0.0, 0.0), {"ego_agent_future": gt}, {})
