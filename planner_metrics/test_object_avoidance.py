import pytest
import torch

from planner_metrics.object_avoidance import evaluate_object_avoidance_with_details


def _data(neighbor_x: float, *, vehicle: bool = True):
    future = torch.zeros(1, 4, 4)
    future[0, :, 0] = neighbor_x
    future[0, :, 2] = 1.0
    past = torch.zeros(1, 2, 10)
    past[0, -1, 6:8] = torch.tensor([0.6, 0.8])
    past[0, -1, 8] = float(vehicle)
    return {
        "neighbor_agents_future": future.unsqueeze(0),
        "neighbor_agents_past": past.unsqueeze(0),
        "ego_shape": torch.tensor([[1.0, 4.0, 2.0]]),
    }


def test_object_avoidance_reports_safe_clearance():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    result = evaluate_object_avoidance_with_details(ego, _data(3.0), {})

    assert result.details["object_avoidance"]["collision"].tolist() == [0.0]
    assert result.scores["success_rate_percent"].tolist() == [100.0]


def test_object_avoidance_reports_collision_for_any_neighbor_type():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    result = evaluate_object_avoidance_with_details(ego, _data(0.5, vehicle=True), {})

    assert result.details["object_avoidance"]["collision"].tolist() == [1.0]
    assert result.scores["success_rate_percent"].tolist() == [0.0]


def test_object_avoidance_converts_legacy_heading_future():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    data = _data(0.5)
    data["neighbor_agents_future"] = data["neighbor_agents_future"][..., :3]

    result = evaluate_object_avoidance_with_details(ego, data, {})

    assert result.details["object_avoidance"]["collision"].tolist() == [1.0]


def test_object_avoidance_ignores_padded_neighbor_timesteps():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    data = _data(3.0)
    data["neighbor_agents_future"][0, 0, 2:] = 0.0

    result = evaluate_object_avoidance_with_details(ego, data, {})

    assert result.details["object_avoidance"]["collision"].tolist() == [0.0]


def test_object_avoidance_rejects_missing_neighbor():
    ego = torch.zeros(1, 4, 4)
    ego[:, :, 2] = 1.0
    data = _data(10.0)
    data["neighbor_agents_future"][:] = 0.0

    with pytest.raises(ValueError, match="no valid neighbor"):
        evaluate_object_avoidance_with_details(ego, data, {})
