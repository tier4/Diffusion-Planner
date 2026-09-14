"""Tests for final arrival errors."""

import math

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
    gt = torch.tensor(
        [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, math.pi / 2]]]
    )
    prediction = _prediction(13.0, 4.0, math.pi)

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    assert torch.allclose(
        result.scores["final_displacement_error_m"], torch.tensor([5.0])
    )
    assert torch.allclose(result.scores["final_heading_error_deg"], torch.tensor([90.0]))
    assert torch.allclose(
        result.details["arrival"]["final_heading_error_rad"], torch.tensor([math.pi / 2])
    )


def test_arrival_wraps_final_heading_error_at_pi_boundary():
    gt = torch.tensor([[[0.0, 0.0, math.radians(179.0)]]])
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
