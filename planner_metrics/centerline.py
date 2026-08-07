"""Route-centerline trajectory metrics usable across all evaluation scenarios."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.geometry import (
    _point_to_segments_error_components,
    _point_to_segments_min_dist,
)

_PREDICTION_TIMESTEP_SECONDS = 0.1
_CENTERLINE_SEGMENT_MIN_LENGTH = 1e-6


def _centerline_segments(lanes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if lanes.ndim != 3 or lanes.shape[-1] < 4:
        raise ValueError(f"lanes must have shape (S, P, D>=4), got {tuple(lanes.shape)}")
    centerlines = lanes[..., :2]
    valid_points = lanes[..., :4].abs().sum(dim=-1) > 1e-6
    valid_segments = valid_points[:, :-1] & valid_points[:, 1:]
    segment_lengths = (centerlines[:, 1:] - centerlines[:, :-1]).norm(dim=-1)
    valid_segments &= segment_lengths > _CENTERLINE_SEGMENT_MIN_LENGTH
    if not valid_segments.any():
        raise ValueError("centerline metric found no valid route-centerline segments")
    return (
        centerlines[:, :-1][valid_segments],
        centerlines[:, 1:][valid_segments],
    )


@torch.no_grad()
def compute_centerline_distance_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return nearest centerline distance with shape ``(N, T_selected)``.

    ``data`` may contain one shared lane tensor ``(S,P,D)``, a singleton scene
    dimension, or one lane tensor per trajectory. Distance evaluation delegates
    to the chunked geometry primitive to remain safe for large maps/batches.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")
    if horizon_steps is None:
        horizon_steps = ego_trajs.shape[1]
    if not 1 <= horizon_steps <= ego_trajs.shape[1]:
        raise ValueError(f"horizon_steps must be in [1, {ego_trajs.shape[1]}]")

    lanes = data.get("route_lanes", data.get("lanes"))
    if lanes is None:
        raise ValueError("centerline metric requires route_lanes or lanes in data")
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(
                f"expected singleton route_lanes context axis, got {tuple(lanes.shape)}"
            )
        lanes = lanes[:, 0]
    if lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    if lanes.ndim != 4 or lanes.shape[0] not in (1, ego_trajs.shape[0]):
        raise ValueError(
            "lanes must have shape (S,P,D), (1,S,P,D), or (N,S,P,D); "
            f"got {tuple(lanes.shape)} for N={ego_trajs.shape[0]}"
        )

    distances = []
    for index in range(ego_trajs.shape[0]):
        scene_lanes = lanes[0 if lanes.shape[0] == 1 else index]
        seg_p1, seg_p2 = _centerline_segments(scene_lanes)
        points = ego_trajs[index, :horizon_steps, :2]
        distances.append(_point_to_segments_min_dist(points, seg_p1.to(points), seg_p2.to(points)))
    return torch.stack(distances, dim=0)


@torch.no_grad()
def compute_centerline_error_components_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return lateral and beyond-segment longitudinal errors per timestep.

    Lateral error is measured against the selected centerline segment's
    supporting line.  Thus longitudinal overshoot beyond a segment endpoint is
    reported separately instead of being folded into lateral error.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")
    if horizon_steps is None:
        horizon_steps = ego_trajs.shape[1]
    if not 1 <= horizon_steps <= ego_trajs.shape[1]:
        raise ValueError(f"horizon_steps must be in [1, {ego_trajs.shape[1]}]")

    lanes = data.get("route_lanes", data.get("lanes"))
    if lanes is None:
        raise ValueError("centerline metric requires route_lanes or lanes in data")
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(
                f"expected singleton route_lanes context axis, got {tuple(lanes.shape)}"
            )
        lanes = lanes[:, 0]
    if lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    if lanes.ndim != 4 or lanes.shape[0] not in (1, ego_trajs.shape[0]):
        raise ValueError(
            "lanes must have shape (S,P,D), (1,S,P,D), or (N,S,P,D); "
            f"got {tuple(lanes.shape)} for N={ego_trajs.shape[0]}"
        )

    lateral_errors = []
    longitudinal_errors = []
    for index in range(ego_trajs.shape[0]):
        scene_lanes = lanes[0 if lanes.shape[0] == 1 else index]
        seg_p1, seg_p2 = _centerline_segments(scene_lanes)
        points = ego_trajs[index, :horizon_steps, :2]
        lateral, longitudinal = _point_to_segments_error_components(
            points, seg_p1.to(points), seg_p2.to(points)
        )
        lateral_errors.append(lateral)
        longitudinal_errors.append(longitudinal)
    return {
        "lateral_error_m": torch.stack(lateral_errors, dim=0),
        "longitudinal_error_m": torch.stack(longitudinal_errors, dim=0),
    }


@torch.no_grad()
def compute_centerline_average_lateral_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return average lateral error for each trajectory, shape ``(N,)``."""
    return compute_centerline_error_components_batch(ego_trajs, data, horizon_steps)[
        "lateral_error_m"
    ].mean(dim=1)


@torch.no_grad()
def compute_centerline_final_lateral_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return final lateral error for each trajectory, shape ``(N,)``."""
    return compute_centerline_error_components_batch(ego_trajs, data, horizon_steps)[
        "lateral_error_m"
    ][:, -1]


def evaluate_centerline(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> dict[str, torch.Tensor]:
    """Return average/final lateral error using a configurable horizon."""
    horizon_seconds = float(parameters.get("horizon_seconds", 8.0))
    if horizon_seconds <= 0:
        raise ValueError("centerline horizon_seconds must be positive")
    steps = min(int(round(horizon_seconds / _PREDICTION_TIMESTEP_SECONDS)), ego_trajs.shape[1])
    if steps < 1:
        raise ValueError("centerline horizon selects zero prediction steps")
    components = compute_centerline_error_components_batch(ego_trajs, data, steps)
    lateral_error = components["lateral_error_m"]
    return {
        "average_lateral_error_m": lateral_error.mean(dim=1),
        "final_lateral_error_m": lateral_error[:, -1],
    }


def evaluate_centerline_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate centerline metrics using the common open-loop result format.

    Centerline currently has no additional per-sample detail fields, so the
    result contains only the average/final lateral error score tensors.
    """
    return MetricEvaluation(scores=evaluate_centerline(ego_trajs, data, parameters))


__all__ = [
    "compute_centerline_average_lateral_error_batch",
    "compute_centerline_distance_batch",
    "compute_centerline_error_components_batch",
    "compute_centerline_final_lateral_error_batch",
    "evaluate_centerline",
    "evaluate_centerline_with_details",
]
