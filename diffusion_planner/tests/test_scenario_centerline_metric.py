"""Tests for route-centerline distance, ADE, and FDE metrics."""

import torch

from planner_metrics.centerline import compute_centerline_distance_batch, evaluate_centerline


def test_centerline_metric_projects_to_segments():
    """Project each predicted point onto the nearest route-centerline segment."""
    ego_trajs = torch.tensor([[[0.5, 1.0], [2.0, -2.0]]])
    route_lanes = torch.zeros((1, 2, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 3.0])
    route_lanes[0, :, 2] = 1.0

    distances = compute_centerline_distance_batch(
        ego_trajs,
        {"route_lanes": route_lanes},
    )

    torch.testing.assert_allclose(distances, torch.tensor([[1.0, 2.0]]))


def test_centerline_metric_returns_ade_and_fde_at_requested_horizon():
    """Compute ADE/FDE using only the configured prediction horizon."""
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
