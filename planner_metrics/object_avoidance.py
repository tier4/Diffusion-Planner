"""Safety of predicted ego motion against all GT neighbors (any actor type)."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.subscores import compute_ego_neighbor_signed_clearance


def _sample_tensor(value: torch.Tensor, index: int, *, batched: bool) -> torch.Tensor:
    return value[index] if batched else value


@torch.no_grad()
def evaluate_object_avoidance_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate predicted ego OBB clearance against every GT neighbor.

    No actor-type filter is applied: the minimum clearance is taken across
    all valid neighbors, since the relevant actor to avoid is not yet
    annotated. Missing required data or a sample with no valid neighbor is
    rejected because this metric is defined for object-avoidance scenes.
    """
    del parameters
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 4:
        raise ValueError("ego_trajs must have shape (B, T, D>=4)")

    batch_size, prediction_steps, _ = ego_trajs.shape
    required = ("neighbor_agents_future", "neighbor_agents_past", "ego_shape")
    if not all(key in data and torch.is_tensor(data[key]) for key in required):
        missing = [key for key in required if key not in data or not torch.is_tensor(data[key])]
        raise ValueError(f"object_avoidance requires tensor data: {missing}")

    collision = torch.zeros(batch_size, dtype=torch.bool, device=ego_trajs.device)
    clearance = torch.zeros(batch_size, dtype=ego_trajs.dtype, device=ego_trajs.device)
    # Samples skipped below (malformed/missing per-sample data) keep this
    # default, matching their default collision=False (no clearance evidence
    # of a collision, so not counted against the success rate).
    success_rate = torch.full((batch_size,), 100.0, dtype=ego_trajs.dtype, device=ego_trajs.device)
    status = torch.full((batch_size,), 2, dtype=torch.int64, device=ego_trajs.device)

    future_all = data["neighbor_agents_future"]
    past_all = data["neighbor_agents_past"]
    shape_all = data["ego_shape"]

    for index in range(batch_size):
        future = _sample_tensor(future_all, index, batched=future_all.ndim == 4)
        past = _sample_tensor(past_all, index, batched=past_all.ndim == 4)
        ego_shape = _sample_tensor(shape_all, index, batched=shape_all.ndim == 2)
        if future.ndim != 3 or future.shape[-1] < 4 or past.ndim != 3:
            continue
        if past.shape[-1] < 10 or past.shape[1] < 1 or ego_shape.numel() < 3:
            continue

        steps = min(prediction_steps, future.shape[1])
        future = future[:, :steps, :4].to(device=ego_trajs.device, dtype=ego_trajs.dtype)
        past = past.to(device=ego_trajs.device)
        nonzero_future = future[..., :2].abs().sum(dim=-1).gt(0).any(dim=-1)
        if not nonzero_future.any():
            raise ValueError(f"object_avoidance found no valid neighbor for sample {index}")

        future = future[nonzero_future]
        neighbor_shapes = past[nonzero_future, -1, 6:8].to(
            device=ego_trajs.device, dtype=ego_trajs.dtype
        )
        neighbor_valid = torch.ones(
            future.shape[0], steps, dtype=torch.bool, device=ego_trajs.device
        )
        distances = compute_ego_neighbor_signed_clearance(
            ego_trajs[index : index + 1, :steps, :4],
            ego_shape[:3].to(device=ego_trajs.device, dtype=ego_trajs.dtype),
            future,
            neighbor_shapes,
            neighbor_valid,
        )
        minimum = distances.min()
        clearance[index] = minimum
        collision[index] = minimum <= 0.0
        status[index] = 1 if collision[index] else 0
        success_rate[index] = (~collision[index]).to(ego_trajs.dtype) * 100.0

    return MetricEvaluation(
        scores={
            "success_rate_percent": success_rate,
        },
        details={
            "object_avoidance": {
                "collision": collision.to(ego_trajs.dtype),
                "min_clearance_m": clearance,
                "status_code": status,
            }
        },
    )


__all__ = ["evaluate_object_avoidance_with_details"]
