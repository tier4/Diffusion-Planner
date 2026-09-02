import torch

from planner_metrics.gt_lateral_deviation import evaluate_gt_lateral_deviation_with_details

_PARAMETERS = {"horizon_seconds": 8.0}


def _straight_gt(steps: int = 80) -> torch.Tensor:
    gt = torch.zeros(1, steps, 4)
    gt[0, :, 0] = torch.linspace(0.0, 20.0, steps)
    gt[0, :, 2] = 1.0
    return gt


def test_gt_lateral_deviation_is_zero_when_prediction_matches_gt():
    gt = _straight_gt()
    pred = gt.clone()
    result = evaluate_gt_lateral_deviation_with_details(pred, {"ego_agent_future": gt}, _PARAMETERS)

    assert result.scores["average_lateral_error_m"].item() < 1e-4
    assert result.scores["final_lateral_error_m"].item() < 1e-4


def test_gt_lateral_deviation_reports_constant_lateral_offset():
    gt = _straight_gt()
    pred = gt.clone()
    pred[0, :, 1] = 1.5  # shift the whole prediction 1.5m to the side

    result = evaluate_gt_lateral_deviation_with_details(pred, {"ego_agent_future": gt}, _PARAMETERS)

    assert abs(result.scores["average_lateral_error_m"].item() - 1.5) < 1e-3
    assert abs(result.scores["final_lateral_error_m"].item() - 1.5) < 1e-3


def test_gt_lateral_deviation_keeps_origin_before_a_turn():
    gt = torch.zeros(1, 3, 4)
    gt[0, :, :2] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    pred = gt.clone()

    result = evaluate_gt_lateral_deviation_with_details(
        pred, {"ego_agent_future": gt}, {"horizon_seconds": 0.2}
    )

    assert result.scores["average_lateral_error_m"].item() < 1e-4
