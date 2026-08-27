import torch

from planner_metrics.pedestrian_stop_safety import (
    evaluate_pedestrian_stop_safety_with_details,
)


def _data(pedestrian_x: float, pedestrian: bool = True):
    future = torch.zeros(1, 4, 4)
    future[0, :, 0] = pedestrian_x
    future[0, :, 2] = 1.0
    past = torch.zeros(1, 2, 10)
    past[0, -1, 6:8] = torch.tensor([0.6, 0.8])
    past[0, -1, 9] = float(pedestrian)
    return {
        "neighbor_agents_future": future.unsqueeze(0),
        "neighbor_agents_past": past.unsqueeze(0),
        "ego_shape": torch.tensor([[1.0, 4.0, 2.0]]),
    }


def test_pedestrian_stop_safety_reports_safe_clearance():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    result = evaluate_pedestrian_stop_safety_with_details(ego, _data(3.0), {})

    assert result.details["pedestrian_stop_safety"]["collision"].tolist() == [0.0]
    assert result.scores["failure_rate_percent"].tolist() == [0.0]
    assert result.details["pedestrian_stop_safety"]["min_clearance_m"].item() > 0.0
    assert result.details["pedestrian_stop_safety"]["status_code"].tolist() == [0]


def test_pedestrian_stop_safety_reports_collision_as_risk():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    result = evaluate_pedestrian_stop_safety_with_details(ego, _data(0.5), {})

    assert result.details["pedestrian_stop_safety"]["collision"].tolist() == [1.0]
    assert result.scores["failure_rate_percent"].tolist() == [100.0]
    assert result.details["pedestrian_stop_safety"]["min_clearance_m"].item() <= 0.0
    assert result.details["pedestrian_stop_safety"]["status_code"].tolist() == [1]


def test_pedestrian_stop_safety_rejects_missing_pedestrian():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    import pytest

    with pytest.raises(ValueError, match="no pedestrian"):
        evaluate_pedestrian_stop_safety_with_details(ego, _data(10.0, False), {})
