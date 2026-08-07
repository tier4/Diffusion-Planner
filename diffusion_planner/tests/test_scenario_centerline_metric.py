"""Tests for route-centerline distance and lateral error metrics."""

import torch

from planner_metrics.centerline import (
    compute_centerline_distance_batch,
    compute_centerline_error_components_batch,
    evaluate_centerline,
)


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

    torch.testing.assert_close(distances, torch.tensor([[1.0, 2.0]]))


def test_centerline_metric_ignores_duplicate_point_segments():
    """A duplicate centerline point must not report a spurious zero error."""
    prediction = torch.tensor([[[10.0, 0.0]]])
    route_lanes = torch.zeros((1, 3, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 0.0, 3.0])
    route_lanes[0, :, 2] = 1.0

    distances = compute_centerline_distance_batch(prediction, {"route_lanes": route_lanes})
    components = compute_centerline_error_components_batch(prediction, {"route_lanes": route_lanes})

    torch.testing.assert_close(distances, torch.tensor([[7.0]]))
    torch.testing.assert_close(components["lateral_error_m"], torch.tensor([[0.0]]))
    torch.testing.assert_close(components["longitudinal_error_m"], torch.tensor([[7.0]]))


def test_centerline_metric_returns_lateral_errors_at_requested_horizon():
    """Compute average/final lateral errors using the configured horizon."""
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

    torch.testing.assert_close(values["average_lateral_error_m"], torch.tensor([1.5]))
    torch.testing.assert_close(values["final_lateral_error_m"], torch.tensor([2.0]))


def test_centerline_error_components_separate_endpoint_overshoot():
    """Endpoint overshoot is longitudinal, not lateral, error."""
    prediction = torch.tensor([[[4.0, 0.0], [5.0, 1.0]]])
    route_lanes = torch.zeros((1, 2, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 3.0])
    route_lanes[0, :, 2] = 1.0

    values = compute_centerline_error_components_batch(
        prediction,
        {"route_lanes": route_lanes},
    )

    torch.testing.assert_close(values["lateral_error_m"], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(values["longitudinal_error_m"], torch.tensor([[1.0, 2.0]]))
