"""Final-position and final-heading errors for arrival evaluation."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation


def _prepare_inputs(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate arrival inputs and return prediction/GT tensors with batch axes."""
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 3:
        raise ValueError(
            "arrival prediction must have shape (N, T, D>=3) with [x, y, heading] "
            "or [x, y, cos, sin], "
            f"got {tuple(ego_trajs.shape)}"
        )

    gt_future = data.get("ego_agent_future")
    if gt_future is None:
        raise ValueError("arrival metric requires ego_agent_future in data")
    if gt_future.ndim == 2:
        gt_future = gt_future.unsqueeze(0)
    if gt_future.ndim != 3 or gt_future.shape[-1] < 3:
        raise ValueError(
            "arrival ground truth must have shape (N, T, D>=3) with [x, y, heading] "
            f"or [x, y, cos, sin], got {tuple(gt_future.shape)}"
        )
    if gt_future.shape[0] not in (1, ego_trajs.shape[0]):
        raise ValueError(
            "arrival ground truth batch dimension must be 1 or match predictions; "
            f"got {gt_future.shape[0]} for N={ego_trajs.shape[0]}"
        )
    gt_future = gt_future.to(device=ego_trajs.device, dtype=ego_trajs.dtype)
    if gt_future.shape[0] == 1:
        gt_future = gt_future.expand(ego_trajs.shape[0], -1, -1)
    return ego_trajs, gt_future


def _yaw_radians(trajectories: torch.Tensor) -> torch.Tensor:
    """Return yaw in radians from either heading or cos/sin trajectory columns."""
    if trajectories.shape[-1] == 3:
        return trajectories[..., 2]
    return torch.atan2(trajectories[..., 3], trajectories[..., 2])


@torch.no_grad()
def compute_final_displacement_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return endpoint Euclidean position error in meters, shape ``(N,)``."""
    ego_trajs, gt_future = _prepare_inputs(ego_trajs, data)
    return (ego_trajs[:, -1, :2] - gt_future[:, -1, :2]).norm(dim=-1)


@torch.no_grad()
def compute_final_heading_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return wrapped absolute endpoint yaw error in degrees, shape ``(N,)``."""
    ego_trajs, gt_future = _prepare_inputs(ego_trajs, data)
    prediction_yaw = _yaw_radians(ego_trajs[:, -1])
    gt_yaw = _yaw_radians(gt_future[:, -1])
    difference_rad = torch.atan2(
        torch.sin(prediction_yaw - gt_yaw),
        torch.cos(prediction_yaw - gt_yaw),
    ).abs()
    return torch.rad2deg(difference_rad)


@torch.no_grad()
def evaluate_arrival_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate final displacement and final heading errors against GT.

    Both errors compare the final available prediction point with the final
    available GT point. GT supports the legacy ``[x, y, heading]`` layout and
    the canonical ``[x, y, cos(yaw), sin(yaw)]`` layout. Heading error is the
    wrapped absolute difference in degrees, in the range ``[0, 180]``.
    """
    del parameters
    fde = compute_final_displacement_error_batch(ego_trajs, data)
    heading_error_deg = compute_final_heading_error_batch(ego_trajs, data)
    return MetricEvaluation(
        scores={
            "final_displacement_error_m": fde,
            "final_heading_error_deg": heading_error_deg,
        },
        details={
            "arrival": {
                "final_displacement_error_m": fde,
                "final_heading_error_deg": heading_error_deg,
                "final_heading_error_rad": torch.deg2rad(heading_error_deg),
            }
        },
    )


__all__ = [
    "compute_final_displacement_error_batch",
    "compute_final_heading_error_batch",
    "evaluate_arrival_with_details",
]
