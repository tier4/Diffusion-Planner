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


def _padded_gt(real_steps: int, total_steps: int, final_x: float) -> torch.Tensor:
    """A 4-column GT that stops being recorded after ``real_steps`` and is zero-padded.

    This is what ``scenario_generation`` writes for a frame near the end of a
    recording: a zeros array filled step by step until the timeline runs out.
    """
    gt = torch.zeros(1, total_steps, 4)
    gt[0, :real_steps, 0] = torch.linspace(final_x / real_steps, final_x, real_steps)
    gt[0, :real_steps, 2] = 1.0
    return gt


def _straight_prediction(final_x: float, steps: int) -> torch.Tensor:
    prediction = torch.zeros(1, steps, 4)
    prediction[0, :, 0] = torch.linspace(final_x / steps, final_x, steps)
    prediction[0, :, 2] = 1.0
    return prediction


def test_arrival_measures_against_the_last_recorded_gt_step_not_the_padding():
    """A zero-padded tail is the ego's own origin, not a pose to arrive at."""
    gt = _padded_gt(real_steps=50, total_steps=80, final_x=20.0)
    # Faster than the GT, so the two do not coincide at the GT's last real step.
    prediction = _straight_prediction(final_x=48.0, steps=80)

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    # At step 50 the GT is at x=20 and the prediction at x=30, so the error is
    # 10 m. Measured against the padded last row it would be 48 m -- the
    # prediction's whole journey, since an all-zero row is the ego origin.
    assert result.details["arrival"]["final_displacement_error_m"].item() == pytest.approx(
        10.0, abs=0.5
    )


def test_arrival_keeps_a_four_column_gt_parked_at_the_origin():
    """A stationary recorded pose carries cos=1, so it is not padding."""
    gt = torch.zeros(1, 4, 4)
    gt[0, :, 2] = 1.0  # parked at the origin, heading +x, for every step
    prediction = torch.zeros(1, 4, 4)
    prediction[0, :, 2] = 1.0

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    assert result.scores["success_rate_percent"].item() == 100.0
    assert result.details["arrival"]["final_displacement_error_m"].item() == pytest.approx(0.0)


def test_arrival_compares_a_four_column_gt_on_the_cos_sin_path():
    """The 4-column GT layout was otherwise only exercised on the prediction."""
    gt = torch.zeros(1, 3, 4)
    gt[0, -1, :2] = torch.tensor([10.0, 0.0])
    gt[0, -1, 2:] = torch.tensor([math.cos(math.pi / 2), math.sin(math.pi / 2)])
    prediction = _prediction(13.0, 4.0, math.pi)

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    details = result.details["arrival"]
    assert details["final_displacement_error_m"].item() == pytest.approx(5.0)
    assert details["final_heading_error_deg"].item() == pytest.approx(90.0)


def test_arrival_scores_each_sample_of_a_batch_independently():
    gt = torch.cat([_padded_gt(50, 80, 20.0), _padded_gt(80, 80, 20.0)])
    prediction = torch.cat([_straight_prediction(20.0, 80), _straight_prediction(32.0, 80)])

    result = evaluate_arrival_with_details(prediction, {"ego_agent_future": gt}, {})

    details = result.details["arrival"]
    # Sample 0 is 7.5 m ahead at the GT's last recorded step; sample 1 is 12 m out.
    assert result.scores["success_rate_percent"].tolist() == [0.0, 0.0]
    assert details["position_within_tolerance"].tolist() == [0.0, 0.0]
    assert details["heading_within_tolerance"].tolist() == [1.0, 1.0]
    assert details["final_displacement_error_m"].shape == (2,)


def test_arrival_rejects_an_empty_horizon():
    with pytest.raises(ValueError, match="at least one prediction"):
        evaluate_arrival_with_details(
            torch.zeros(1, 0, 4), {"ego_agent_future": torch.zeros(1, 0, 4)}, {}
        )
