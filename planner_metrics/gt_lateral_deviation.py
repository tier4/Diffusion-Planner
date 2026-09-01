"""GT-trajectory lateral deviation metric (backs the ``simple_turn`` label).

Same lateral/longitudinal error decomposition as ``planner_metrics/centerline.py``
(shared via ``planner_metrics/lateral_deviation.py``), but the reference path
is the GT ego future trajectory instead of the route-lane centerline.
Route-lane geometry is coarse and doesn't always capture the intended turn
shape precisely, whereas the recorded GT trajectory does — so for
simple-turn scenes, deviation from GT is the more faithful reference path.
"""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.horizon import resolve_horizon_steps
from planner_metrics.lateral_deviation import compute_lateral_longitudinal_error_batch

_PREDICTION_TIMESTEP_SECONDS = 0.1
_GT_SEGMENT_MIN_LENGTH = 1e-6


def _gt_segments(gt_future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_future.ndim != 2 or gt_future.shape[-1] < 2:
        raise ValueError(f"gt_future must have shape (T, D>=2), got {tuple(gt_future.shape)}")
    xy = gt_future[:, :2]
    valid_points = xy.abs().sum(dim=-1) > 1e-6
    valid_segments = valid_points[:-1] & valid_points[1:]
    segment_lengths = (xy[1:] - xy[:-1]).norm(dim=-1)
    valid_segments &= segment_lengths > _GT_SEGMENT_MIN_LENGTH
    if not valid_segments.any():
        raise ValueError("gt_lateral_deviation found no usable GT trajectory segments")
    return xy[:-1][valid_segments], xy[1:][valid_segments]


@torch.no_grad()
def compute_gt_lateral_deviation_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-sample lateral/longitudinal error against the GT trajectory.

    Each sample projects against its own GT future (unlike the shared/per-
    sample route-lane tensor supported by ``centerline.py``), since the GT
    trajectory is inherently per-sample.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")
    if horizon_steps is None:
        horizon_steps = ego_trajs.shape[1]
    if not 1 <= horizon_steps <= ego_trajs.shape[1]:
        raise ValueError(f"horizon_steps must be in [1, {ego_trajs.shape[1]}]")

    gt_future = data.get("ego_agent_future")
    if gt_future is None:
        raise ValueError("gt_lateral_deviation requires ego_agent_future in data")

    def segments_for_sample(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _gt_segments(gt_future[index])

    return compute_lateral_longitudinal_error_batch(ego_trajs, horizon_steps, segments_for_sample)


def evaluate_gt_lateral_deviation_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate lateral deviation from the GT trajectory using a configurable horizon."""
    horizon_seconds = float(parameters.get("horizon_seconds", 8.0))
    steps = resolve_horizon_steps(
        horizon_seconds,
        ego_trajs.shape[1],
        label="gt_lateral_deviation",
        timestep_seconds=_PREDICTION_TIMESTEP_SECONDS,
    )
    components = compute_gt_lateral_deviation_batch(ego_trajs, data, steps)
    lateral_error = components["lateral_error_m"]
    return MetricEvaluation(
        scores={
            "average_lateral_error_m": lateral_error.mean(dim=1),
            "final_lateral_error_m": lateral_error[:, -1],
        }
    )


__all__ = [
    "compute_gt_lateral_deviation_batch",
    "evaluate_gt_lateral_deviation_with_details",
]
