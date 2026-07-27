import torch
from diffusion_planner.override_validation.metrics.centerline import (
    _point_to_polylines_min_dist,
    evaluate_centerline,
)


def test_centerline_metric_projects_to_segments():
    points = torch.tensor([[0.5, 1.0], [2.0, -2.0]])
    polylines = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])

    distances = _point_to_polylines_min_dist(points, polylines, torch.tensor([[True, True]]))

    torch.testing.assert_allclose(distances, torch.tensor([1.0, 2.0]))


def test_centerline_metric_returns_ade_and_fde_at_requested_horizon():
    prediction = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [1.0, 2.0, 1.0, 0.0], [2.0, 3.0, 1.0, 0.0]]])
    # [batch, singleton context, lane, point, feature]
    route_lanes = torch.zeros((1, 1, 1, 4, 8))
    route_lanes[0, 0, 0, :, 0] = torch.arange(4)
    route_lanes[0, 0, 0, :, 2] = 1.0

    values = evaluate_centerline(
        prediction,
        {"route_lanes": route_lanes},
        {"horizon_seconds": 0.2},
    )

    torch.testing.assert_allclose(values["ade_m"], torch.tensor([1.5]))
    torch.testing.assert_allclose(values["fde_m"], torch.tensor([2.0]))
