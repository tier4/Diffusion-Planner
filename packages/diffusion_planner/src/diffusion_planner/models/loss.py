"""Diffusion-planner-specific training loss construction."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F

from diffusion_planner.data.dimensions import (
    EGO_VELOCITY_INDEX,
    TRAJECTORY_DIM,
)

from .control import (
    DT,
    ControlNormalizer,
    denormalize_action,
    denormalize_positions,
    waypoints_to_control,
)
from .diffusion_planner import DiffusionPlanner
from .flow_matching import compute_x0_flow_matching_loss, x0_velocity_error


class DiffusionPlannerLoss(TypedDict):
    """Loss values and turn-indicator counts for one training batch."""

    total: torch.Tensor
    control: torch.Tensor
    control_trajectory: torch.Tensor
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


def control_huber_loss(
    x_prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    time_epsilon: float,
) -> torch.Tensor:
    """Apply an elementwise Huber loss to the ego control.

    Control channels are already normalized to a comparable scale, so unlike a
    pose target there is no frame to rotate the error into first.
    """
    error = x0_velocity_error(x_prediction - target, time, time_epsilon)
    return F.huber_loss(error, torch.zeros_like(error), reduction="none")


def create_control_target(
    input_data: dict[str, torch.Tensor],
    control_normalizer: ControlNormalizer,
    position_scale: float,
) -> torch.Tensor:
    """Fit the `(B, T, 2)` ego control target from the recorded trajectory."""
    ego_past = input_data["ego_agent_past"]
    history = denormalize_positions(ego_past[..., :TRAJECTORY_DIM], position_scale)
    future = denormalize_positions(
        input_data["ego_agent_future"][..., :TRAJECTORY_DIM], position_scale
    )
    return control_normalizer(
        waypoints_to_control(history, future, ego_past[:, -1, EGO_VELOCITY_INDEX])
    )


def create_ego_padding_mask(input_data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mark `(B,)` samples whose ego pose label contains a padded timestep.

    The mask is read off the pose label rather than the control target: a
    stationary ego has a genuinely all-zero control sequence, and deriving the
    mask from control would drop exactly those frames from training.
    """
    poses = input_data["ego_agent_future"][..., :TRAJECTORY_DIM]
    return (torch.count_nonzero(poses, dim=-1) == 0).any(dim=-1)


def compute_control_trajectory_loss(
    ego_control: torch.Tensor,
    input_data: dict[str, torch.Tensor],
    control_normalizer: ControlNormalizer,
    position_scale: float,
    horizon: int,
) -> torch.Tensor:
    """Score predicted ego control in trajectory space over sliding windows.

    Matching control channel by channel says nothing about where the vehicle ends
    up: a small curvature error integrates into metres of lateral offset. For each
    start step `t` the reference state is obtained by rolling the prediction out
    from the current pose (that rollout is detached, so the gradient of window `t`
    does not fight the windows before it), then `control[t : t + horizon]` is
    integrated from there and compared with the recorded trajectory expressed in
    that same reference frame.

    Args:
        ego_control: `(B, T, 2)` predicted ego control in `ControlNormalizer` space.
        input_data: Batched planner inputs, normalized as the model sees them.
        control_normalizer: Statistics the prediction is expressed in.
        position_scale: `PlannerDataNormalizer.position_scale`.
        horizon: Steps integrated per window; zero or more than `T` disables the loss.

    Returns:
        Scalar loss, or zero when the horizon leaves no complete window.
    """
    batch, steps = ego_control.shape[:2]
    device = ego_control.device
    window_count = steps - horizon + 1
    if horizon <= 0 or window_count <= 0:
        return ego_control.new_zeros(())

    control = denormalize_action(control_normalizer.inverse(ego_control))
    accel = control[..., 0]
    curvature = control[..., 1]

    dt = DT
    half_dt = 0.5 * dt
    half_dt_squared = 0.5 * dt * dt

    # The conversion frame puts the current ego pose at the origin facing +x, so
    # the recorded future needs no rotation -- only the metre rescaling.
    future = denormalize_positions(
        input_data["ego_agent_future"][..., :TRAJECTORY_DIM], position_scale
    )
    zero_pose = future.new_zeros((batch, 1, TRAJECTORY_DIM))
    zero_pose[:, 0, 2] = 1.0
    poses = torch.cat((zero_pose, future), dim=1)  # (B, T + 1, 4)
    recorded_xy = poses[..., :2]
    recorded_heading = torch.atan2(poses[..., 3], poses[..., 2])

    # Full rollout from the current pose; every window starts from a detached slice.
    initial_speed = input_data["ego_agent_past"][:, -1, EGO_VELOCITY_INDEX]
    speed = torch.cat(
        (
            initial_speed[:, None],
            initial_speed[:, None] + torch.cumsum(accel * dt, dim=-1),
        ),
        dim=-1,
    )  # (B, T + 1)
    heading = torch.cat(
        (
            speed.new_zeros((batch, 1)),
            torch.cumsum(
                curvature * speed[:, :-1] * dt + curvature * accel * half_dt_squared,
                dim=-1,
            ),
        ),
        dim=-1,
    )  # (B, T + 1)
    step_x = (
        speed[:, :-1] * torch.cos(heading[:, :-1])
        + speed[:, 1:] * torch.cos(heading[:, 1:])
    ) * half_dt
    step_y = (
        speed[:, :-1] * torch.sin(heading[:, :-1])
        + speed[:, 1:] * torch.sin(heading[:, 1:])
    ) * half_dt
    rolled_x = torch.cat(
        (step_x.new_zeros((batch, 1)), torch.cumsum(step_x, dim=-1)), dim=-1
    )
    rolled_y = torch.cat(
        (step_y.new_zeros((batch, 1)), torch.cumsum(step_y, dim=-1)), dim=-1
    )

    reference_speed = speed[:, :window_count].detach()
    reference_x = rolled_x[:, :window_count].detach()
    reference_y = rolled_y[:, :window_count].detach()
    reference_heading = heading[:, :window_count].detach()

    # Re-integrate each window from its detached reference state.
    accel_windows = accel.unfold(1, horizon, 1)  # (B, W, H)
    curvature_windows = curvature.unfold(1, horizon, 1)
    window_speed = torch.cat(
        (
            reference_speed.unsqueeze(-1),
            reference_speed.unsqueeze(-1) + torch.cumsum(accel_windows * dt, dim=-1),
        ),
        dim=-1,
    )  # (B, W, H + 1)
    window_heading = torch.cat(
        (
            window_speed.new_zeros((batch, window_count, 1)),
            torch.cumsum(
                curvature_windows * window_speed[..., :-1] * dt
                + curvature_windows * accel_windows * half_dt_squared,
                dim=-1,
            ),
        ),
        dim=-1,
    )
    predicted_x = torch.cumsum(
        (
            window_speed[..., :-1] * torch.cos(window_heading[..., :-1])
            + window_speed[..., 1:] * torch.cos(window_heading[..., 1:])
        )
        * half_dt,
        dim=-1,
    )
    predicted_y = torch.cumsum(
        (
            window_speed[..., :-1] * torch.sin(window_heading[..., :-1])
            + window_speed[..., 1:] * torch.sin(window_heading[..., 1:])
        )
        * half_dt,
        dim=-1,
    )
    predicted_heading = window_heading[..., 1:]

    # The recorded trajectory in each window's reference frame.
    indices = torch.arange(1, horizon + 1, device=device).unsqueeze(0) + torch.arange(
        window_count, device=device
    ).unsqueeze(1)  # (W, H)
    target_xy = recorded_xy[:, indices]
    target_heading = recorded_heading[:, indices] - reference_heading[..., None]
    delta_x = target_xy[..., 0] - reference_x[..., None]
    delta_y = target_xy[..., 1] - reference_y[..., None]
    reference_cos = torch.cos(reference_heading)[..., None]
    reference_sin = torch.sin(reference_heading)[..., None]
    target_x = delta_x * reference_cos + delta_y * reference_sin
    target_y = -delta_x * reference_sin + delta_y * reference_cos

    position_error = (predicted_x - target_x) ** 2 + (predicted_y - target_y) ** 2
    heading_error = (torch.cos(predicted_heading) - torch.cos(target_heading)) ** 2 + (
        torch.sin(predicted_heading) - torch.sin(target_heading)
    ) ** 2
    return (position_error + heading_error).mean()


def compute_diffusion_planner_loss(
    model: DiffusionPlanner,
    input_data: dict[str, torch.Tensor],
    *,
    time_mean: float,
    time_std: float,
    time_epsilon: float,
    noise_scale: float,
    control_normalizer: ControlNormalizer,
    position_scale: float,
    ego_loss_weight: float = 1.0,
    turn_indicator_loss_weight: float = 1.0,
    turn_indicator_transition_loss_weight: float = 5.0,
    control_trajectory_loss_weight: float = 0.4,
    control_trajectory_loss_horizon: int = 80,
) -> DiffusionPlannerLoss:
    """Compute the joint planner loss and turn-indicator metrics.

    `model` is only ever called, never inspected: under DDP and `torch.compile`
    the planner arrives wrapped, and neither wrapper forwards attribute lookups
    to the module it holds. Anything the loss needs from the model comes in as an
    argument instead.
    """
    target = create_control_target(input_data, control_normalizer, position_scale)
    training_mask = create_ego_padding_mask(input_data)
    turn_indicator_logits: list[torch.Tensor] = []
    control_predictions: list[torch.Tensor] = []

    def predict(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        control, logits = model(state, input_data, time)
        turn_indicator_logits.append(logits)
        control_predictions.append(control)
        return control

    control_loss = compute_x0_flow_matching_loss(
        x0_model=predict,
        loss_function=lambda x_prediction, clean_target, time: control_huber_loss(
            x_prediction,
            clean_target,
            time,
            time_epsilon,
        ),
        target=target,
        mask=training_mask,
        time_mean=time_mean,
        time_std=time_std,
        noise_scale=noise_scale,
    )
    control_trajectory_loss = compute_control_trajectory_loss(
        control_predictions[0],
        input_data,
        control_normalizer,
        position_scale,
        control_trajectory_loss_horizon,
    )
    turn_indicator_loss, correct, valid_count = compute_turn_indicator_loss(
        turn_indicator_logits[0],
        input_data,
        transition_weight=turn_indicator_transition_loss_weight,
    )
    total = (
        ego_loss_weight * control_loss
        + control_trajectory_loss_weight * control_trajectory_loss
        + turn_indicator_loss_weight * turn_indicator_loss
    )
    return {
        "total": total,
        "control": control_loss,
        "control_trajectory": control_trajectory_loss,
        "turn_indicator": turn_indicator_loss,
        "turn_indicator_correct": correct,
        "turn_indicator_valid_count": valid_count,
    }
