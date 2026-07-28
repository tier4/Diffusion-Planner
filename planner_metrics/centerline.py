"""Route-centerline trajectory metrics usable across all evaluation scenarios."""

from __future__ import annotations

import torch

from planner_metrics.geometry import _point_to_segments_min_dist

_PREDICTION_TIMESTEP_SECONDS = 0.1


def _centerline_segments(lanes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if lanes.ndim != 3 or lanes.shape[-1] < 4:
        raise ValueError(f"lanes must have shape (S, P, D>=4), got {tuple(lanes.shape)}")
    centerlines = lanes[..., :2]
    valid_points = lanes[..., :4].abs().sum(dim=-1) > 1e-6
    valid_segments = valid_points[:, :-1] & valid_points[:, 1:]
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
def compute_centerline_ade_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return centerline ADE for each trajectory, shape ``(N,)``."""
    return compute_centerline_distance_batch(ego_trajs, data, horizon_steps).mean(dim=1)


@torch.no_grad()
def compute_centerline_fde_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return centerline FDE for each trajectory, shape ``(N,)``."""
    return compute_centerline_distance_batch(ego_trajs, data, horizon_steps)[:, -1]


def evaluate_centerline(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> dict[str, torch.Tensor]:
    """Return per-trajectory centerline ADE/FDE using a configurable horizon."""
    horizon_seconds = float(parameters.get("horizon_seconds", 8.0))
    if horizon_seconds <= 0:
        raise ValueError("centerline horizon_seconds must be positive")
    steps = min(int(round(horizon_seconds / _PREDICTION_TIMESTEP_SECONDS)), ego_trajs.shape[1])
    if steps < 1:
        raise ValueError("centerline horizon selects zero prediction steps")
    return {
        "ade_m": compute_centerline_ade_batch(ego_trajs, data, steps),
        "fde_m": compute_centerline_fde_batch(ego_trajs, data, steps),
    }


__all__ = [
    "compute_centerline_ade_batch",
    "compute_centerline_distance_batch",
    "compute_centerline_fde_batch",
    "evaluate_centerline",
]
