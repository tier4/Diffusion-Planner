"""Yield-progress metric: predicted ego must not advance far while yielding.

Shared by the ``pedestrian_yield`` and ``vehicle_yield`` scenario labels: the
scene puts a pedestrian/cyclist or vehicle in the ego's path, and the
predicted ego is expected to stay (near-)stationary for the first
``horizon_seconds`` rather than pushing through. "Forward" is the ego's own
``x`` axis at the reference timestep, matching the shared ego-relative frame
used throughout this package (see ``planner_metrics/scene_format.py``).
"""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.horizon import resolve_horizon_steps

_PREDICTION_TIMESTEP_SECONDS = 0.1


@torch.no_grad()
def compute_max_forward_progress_batch(
    ego_trajs: torch.Tensor,
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return the maximum forward (ego +x) progress per sample, shape ``(N,)``."""
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 1:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=1), got {tuple(ego_trajs.shape)}")
    if horizon_steps is None:
        horizon_steps = ego_trajs.shape[1]
    if not 1 <= horizon_steps <= ego_trajs.shape[1]:
        raise ValueError(f"horizon_steps must be in [1, {ego_trajs.shape[1]}]")
    return ego_trajs[:, :horizon_steps, 0].max(dim=1).values


def evaluate_yield_progress_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate yield failure and return per-sample diagnostic details.

    The aggregate score is the yield failure rate in percent: a sample fails
    when the ego's maximum forward progress within ``horizon_seconds``
    exceeds ``maximum_forward_progress_m``.
    """
    del data
    horizon_seconds = float(parameters["horizon_seconds"])
    tolerance = float(parameters["maximum_forward_progress_m"])
    if tolerance < 0:
        raise ValueError("yield_progress maximum_forward_progress_m must be non-negative")
    steps = resolve_horizon_steps(
        horizon_seconds,
        ego_trajs.shape[1],
        label="yield_progress",
        timestep_seconds=_PREDICTION_TIMESTEP_SECONDS,
    )
    maximum = compute_max_forward_progress_batch(ego_trajs, steps)
    yielded = maximum <= tolerance
    return MetricEvaluation(
        scores={"failure_rate_percent": (~yielded).to(ego_trajs.dtype) * 100.0},
        details={
            "yield_progress": {
                "horizon_seconds": torch.full_like(maximum, horizon_seconds),
                "maximum_forward_progress_m": torch.full_like(maximum, tolerance),
                "max_forward_progress_m": maximum,
                "yielded": yielded,
            }
        },
    )


__all__ = [
    "compute_max_forward_progress_batch",
    "evaluate_yield_progress_with_details",
]
