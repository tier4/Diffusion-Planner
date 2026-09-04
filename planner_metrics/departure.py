"""Trajectory departure metrics usable across all evaluation scenarios."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.horizon import resolve_horizon_steps

_PREDICTION_TIMESTEP_SECONDS = 0.1


@torch.no_grad()
def compute_max_displacement_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return maximum displacement from the current ego pose, shape ``(N,)``."""
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")
    if horizon_steps is None:
        horizon_steps = ego_trajs.shape[1]
    if not 1 <= horizon_steps <= ego_trajs.shape[1]:
        raise ValueError(f"horizon_steps must be in [1, {ego_trajs.shape[1]}]")

    current = data.get("ego_current_state")
    if current is None:
        raise ValueError("departure metric requires ego_current_state in data")
    if current.ndim == 1:
        current = current.unsqueeze(0)
    if current.ndim != 2 or current.shape[1] < 2 or current.shape[0] not in (1, ego_trajs.shape[0]):
        raise ValueError(
            "ego_current_state must have shape (D>=2,) or (N,D>=2); "
            f"got {tuple(current.shape)} for N={ego_trajs.shape[0]}"
        )
    initial_xy = current[:, :2].to(device=ego_trajs.device, dtype=ego_trajs.dtype)
    if initial_xy.shape[0] == 1:
        initial_xy = initial_xy.expand(ego_trajs.shape[0], -1)
    displacement = (ego_trajs[:, :horizon_steps, :2] - initial_xy[:, None]).norm(dim=-1)
    return displacement.max(dim=1).values


@torch.no_grad()
def compute_departure_failure_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    minimum_displacement_m: float,
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return 1.0 for failure and 0.0 for successful departure, shape ``(N,)``."""
    if minimum_displacement_m < 0:
        raise ValueError("minimum_displacement_m must be non-negative")
    maximum = compute_max_displacement_batch(ego_trajs, data, horizon_steps)
    return (maximum < minimum_displacement_m).to(ego_trajs.dtype)


def evaluate_departure(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> dict[str, torch.Tensor]:
    """Return departure success as 0 or 100 percent per trajectory."""
    return evaluate_departure_with_details(ego_trajs, data, parameters).scores


def evaluate_departure_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate departure metrics and return per-sample diagnostic details.

    The aggregate score is the departure success rate in percent. The detail
    section additionally records the configured threshold and horizon, the
    maximum displacement observed for each sample, and the resulting departure
    decision.
    """
    horizon_seconds = float(parameters["horizon_seconds"])
    minimum = float(parameters["minimum_displacement_m"])
    steps = resolve_horizon_steps(
        horizon_seconds,
        ego_trajs.shape[1],
        label="departure",
        timestep_seconds=_PREDICTION_TIMESTEP_SECONDS,
    )
    maximum = compute_max_displacement_batch(ego_trajs, data, steps)
    return MetricEvaluation(
        scores={"success_rate_percent": (maximum >= minimum).to(ego_trajs.dtype) * 100.0},
        details={
            "departure": {
                "horizon_seconds": torch.full_like(maximum, horizon_seconds),
                "minimum_displacement_m": torch.full_like(maximum, minimum),
                "max_displacement_m": maximum,
                "departed": maximum >= minimum,
            }
        },
    )


__all__ = [
    "compute_departure_failure_batch",
    "compute_max_displacement_batch",
    "evaluate_departure",
    "evaluate_departure_with_details",
]
