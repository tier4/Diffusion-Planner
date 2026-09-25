"""Arrival success: final position and heading within tolerance of the GT endpoint."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation

_DEFAULT_POSITION_TOLERANCE_M = 2.0
_DEFAULT_HEADING_TOLERANCE_DEG = 10.0
# Padding is an all-zero row, which is what the producers write; a real pose
# parked at the ego origin still carries a heading (cos=1 in the 4-column
# layout, or a non-zero yaw in the 3-column one), so testing every column
# separates the two wherever the data allows it at all.
_GT_ROW_MIN_NORM = 1e-6


def _prepare_inputs(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate arrival inputs and return prediction, GT, and the arrival index.

    The layouts are accepted as exactly 3 or exactly 4 columns, never "at least
    3": the third column means ``heading`` in one layout and ``cos(yaw)`` in the
    other, so a wider tensor cannot be told apart and would silently be read as
    ``atan2(extra_column, heading)``.

    Both tensors are truncated to their common length, so a checkpoint whose
    ``future_len`` differs from the NPZ's GT horizon compares the two at the
    same instant instead of scoring t=8.0 s against t=9.0 s.

    The arrival index is the last step where the GT is a real recorded pose, per
    sample. A GT future can be zero-padded at the tail when the recording ran
    out (``scenario_generation/reproducer_rollout.py`` fills a zeros array and
    breaks), and an all-zero row in the ego frame IS the ego's own t=0 pose --
    so taking ``[:, -1]`` unconditionally would measure the prediction against
    the origin and report roughly its total travel as the error. Index 0 is
    always treated as real: the ego-frame GT legitimately starts at the origin.

    The one case this cannot separate is a 3-column GT parked at the origin
    with a yaw of exactly 0, which is byte-identical to padding.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] not in (3, 4):
        raise ValueError(
            "arrival prediction must have shape (N, T, 3) with [x, y, heading] "
            "or (N, T, 4) with [x, y, cos, sin], "
            f"got {tuple(ego_trajs.shape)}"
        )

    gt_future = data.get("ego_agent_future")
    if gt_future is None:
        raise ValueError("arrival metric requires ego_agent_future in data")
    if gt_future.ndim == 2:
        gt_future = gt_future.unsqueeze(0)
    if gt_future.ndim != 3 or gt_future.shape[-1] not in (3, 4):
        raise ValueError(
            "arrival ground truth must have shape (N, T, 3) with [x, y, heading] "
            f"or (N, T, 4) with [x, y, cos, sin], got {tuple(gt_future.shape)}"
        )
    if gt_future.shape[0] not in (1, ego_trajs.shape[0]):
        raise ValueError(
            "arrival ground truth batch dimension must be 1 or match predictions; "
            f"got {gt_future.shape[0]} for N={ego_trajs.shape[0]}"
        )
    gt_future = gt_future.to(device=ego_trajs.device, dtype=ego_trajs.dtype)
    if gt_future.shape[0] == 1:
        gt_future = gt_future.expand(ego_trajs.shape[0], -1, -1)
    steps = min(ego_trajs.shape[1], gt_future.shape[1])
    if steps < 1:
        raise ValueError("arrival needs at least one prediction and ground-truth step")
    ego_trajs, gt_future = ego_trajs[:, :steps], gt_future[:, :steps]

    recorded = gt_future.abs().sum(dim=-1) > _GT_ROW_MIN_NORM
    recorded[:, 0] = True
    # Last True per row: weight each index by its own position and take the max.
    positions = torch.arange(steps, device=recorded.device)
    arrival_index = (recorded * positions).amax(dim=1)
    return ego_trajs, gt_future, arrival_index


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
    """Return endpoint Euclidean position error in meters, shape ``(N,)``.

    The endpoint is the GT's last recorded step, which is not the last stored
    step when the future is zero-padded.
    """
    ego_trajs, gt_future, arrival_index = _prepare_inputs(ego_trajs, data)
    rows = torch.arange(ego_trajs.shape[0], device=ego_trajs.device)
    return (ego_trajs[rows, arrival_index, :2] - gt_future[rows, arrival_index, :2]).norm(dim=-1)


@torch.no_grad()
def compute_final_heading_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return wrapped absolute endpoint yaw error in degrees, shape ``(N,)``."""
    ego_trajs, gt_future, arrival_index = _prepare_inputs(ego_trajs, data)
    rows = torch.arange(ego_trajs.shape[0], device=ego_trajs.device)
    prediction_yaw = _yaw_radians(ego_trajs[rows, arrival_index])
    gt_yaw = _yaw_radians(gt_future[rows, arrival_index])
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
    """Score whether the prediction arrives where the GT does, as a success rate.

    A sample succeeds when its final displacement error is within
    ``position_tolerance_m`` AND its final heading error is within
    ``heading_tolerance_deg``. Both errors compare the final available
    prediction point with the final available GT point. GT supports the legacy
    ``[x, y, heading]`` layout and the canonical ``[x, y, cos(yaw), sin(yaw)]``
    layout. Heading error is the wrapped absolute difference in degrees, in the
    range ``[0, 180]``. The raw errors stay available in the details section.
    """
    position_tolerance_m = float(
        parameters.get("position_tolerance_m", _DEFAULT_POSITION_TOLERANCE_M)
    )
    heading_tolerance_deg = float(
        parameters.get("heading_tolerance_deg", _DEFAULT_HEADING_TOLERANCE_DEG)
    )
    if position_tolerance_m < 0 or heading_tolerance_deg < 0:
        raise ValueError("arrival tolerances must be non-negative")

    fde = compute_final_displacement_error_batch(ego_trajs, data)
    heading_error_deg = compute_final_heading_error_batch(ego_trajs, data)
    position_ok = fde <= position_tolerance_m
    heading_ok = heading_error_deg <= heading_tolerance_deg
    passed = position_ok & heading_ok
    return MetricEvaluation(
        scores={"success_rate_percent": passed.to(ego_trajs.dtype) * 100.0},
        details={
            "arrival": {
                "final_displacement_error_m": fde,
                "final_heading_error_deg": heading_error_deg,
                "final_heading_error_rad": torch.deg2rad(heading_error_deg),
                "position_tolerance_m": torch.full_like(fde, position_tolerance_m),
                "heading_tolerance_deg": torch.full_like(fde, heading_tolerance_deg),
                "position_within_tolerance": position_ok.to(ego_trajs.dtype),
                "heading_within_tolerance": heading_ok.to(ego_trajs.dtype),
            }
        },
    )


__all__ = [
    "compute_final_displacement_error_batch",
    "compute_final_heading_error_batch",
    "evaluate_arrival_with_details",
]
