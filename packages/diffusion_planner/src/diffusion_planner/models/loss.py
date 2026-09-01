"""Diffusion-planner-specific training loss construction."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F

from diffusion_planner.data.dimensions import TRAJECTORY_DIM

from .diffusion_planner import DiffusionPlanner
from .flow_matching import compute_x0_flow_matching_loss, x0_velocity_error


class DiffusionPlannerLoss(TypedDict):
    """Loss values and turn-indicator counts for one training batch."""

    total: torch.Tensor
    trajectory: torch.Tensor
    turn_indicator: torch.Tensor
    turn_indicator_correct: torch.Tensor
    turn_indicator_valid_count: torch.Tensor


def compute_turn_indicator_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    transition_weight: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted loss, correct count, and valid-label count."""
    target = batch["turn_indicators_future"][:, 0].to(torch.long)
    current = batch["turn_indicators"][:, -1].to(torch.long)
    valid = (target >= 1) & (target <= 3)
    current_valid = (current >= 1) & (current <= 3)
    transition = valid & current_valid & (current != target)
    class_target = (target - 1).clamp(0, 2)
    per_sample_loss = F.cross_entropy(logits, class_target, reduction="none")
    sample_weight = torch.where(
        transition,
        per_sample_loss.new_tensor(transition_weight),
        per_sample_loss.new_tensor(1.0),
    )
    sample_weight = sample_weight * valid
    valid_count = valid.sum()
    loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
    correct = ((logits.argmax(dim=-1) == class_target) & valid).sum()
    return loss, correct, valid_count


def trajectory_error_in_target_frame(
    error: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Rotate global xy error into target longitudinal/lateral coordinates."""
    position_error = error[..., :2]
    target_cos = target[..., 2]
    target_sin = target[..., 3]
    longitudinal = (
        position_error[..., 0] * target_cos + position_error[..., 1] * target_sin
    )
    lateral = -position_error[..., 0] * target_sin + position_error[..., 1] * target_cos
    return torch.cat(
        (longitudinal.unsqueeze(-1), lateral.unsqueeze(-1), error[..., 2:]),
        dim=-1,
    )


def trajectory_huber_loss(
    x_prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    time_epsilon: float,
    ego_loss_weight: float = 1.0,
    neighbor_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Apply agent-weighted Huber loss after target-frame position rotation."""
    target_frame_error = trajectory_error_in_target_frame(x_prediction - target, target)
    target_frame_error = x0_velocity_error(target_frame_error, time, time_epsilon)
    elementwise_loss = F.huber_loss(
        target_frame_error,
        torch.zeros_like(target_frame_error),
        reduction="none",
    )
    agent_weights = torch.cat(
        (
            elementwise_loss.new_full((1,), ego_loss_weight),
            elementwise_loss.new_full(
                (elementwise_loss.shape[1] - 1,), neighbor_loss_weight
            ),
        )
    )
    return elementwise_loss * agent_weights.view(1, -1, 1, 1)


def create_target_trajectory(
    input_data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Combine labels into `(B, A, T, 4)` ego and neighbor trajectories."""
    ego_future = input_data["ego_agent_future"][..., :TRAJECTORY_DIM].unsqueeze(1)
    return torch.cat((ego_future, input_data["neighbor_agents_future"]), dim=1)


def compute_diffusion_planner_loss(
    model: DiffusionPlanner,
    input_data: dict[str, torch.Tensor],
    *,
    time_mean: float,
    time_std: float,
    time_epsilon: float,
    noise_scale: float,
    ego_loss_weight: float = 1.0,
    neighbor_loss_weight: float = 1.0,
    turn_indicator_loss_weight: float = 1.0,
    turn_indicator_transition_loss_weight: float = 5.0,
) -> DiffusionPlannerLoss:
    """Compute the joint planner loss and turn-indicator metrics."""
    target = create_target_trajectory(input_data)
    training_mask = (torch.count_nonzero(target, dim=-1) == 0).any(dim=-1)
    turn_indicator_logits: list[torch.Tensor] = []

    def predict(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        trajectory, logits = model(
            state,
            training_mask,
            input_data,
            time,
        )
        turn_indicator_logits.append(logits)
        return trajectory

    trajectory_loss = compute_x0_flow_matching_loss(
        x0_model=predict,
        loss_function=lambda x_prediction, clean_target, time: trajectory_huber_loss(
            x_prediction,
            clean_target,
            time,
            time_epsilon,
            ego_loss_weight,
            neighbor_loss_weight,
        ),
        target=target,
        mask=training_mask,
        time_mean=time_mean,
        time_std=time_std,
        noise_scale=noise_scale,
    )
    turn_indicator_loss, correct, valid_count = compute_turn_indicator_loss(
        turn_indicator_logits[0],
        input_data,
        transition_weight=turn_indicator_transition_loss_weight,
    )
    return {
        "total": trajectory_loss + turn_indicator_loss_weight * turn_indicator_loss,
        "trajectory": trajectory_loss,
        "turn_indicator": turn_indicator_loss,
        "turn_indicator_correct": correct,
        "turn_indicator_valid_count": valid_count,
    }
