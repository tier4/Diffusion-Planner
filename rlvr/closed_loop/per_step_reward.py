"""Per-step reward for closed-loop rollout.

Computes lightweight per-step signals (collision, road border, progress)
from a 2-step mini-trajectory [prev_pos, curr_pos]. Reuses functions from
rlvr.reward where possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rlvr.reward import (
    RewardConfig,
    compute_road_border_penalty,
    compute_safety_score_batch,
)


@dataclass
class StepRewardConfig:
    w_progress: float = 1.0
    w_alive: float = 0.5
    w_collision: float = 10.0
    w_rb_crossing: float = 5.0
    dt: float = 0.1


@dataclass
class StepReward:
    total: float
    collision: bool
    rb_crossing: bool
    progress: float
    terminal: bool


def compute_step_reward(
    ego_prev: torch.Tensor,
    ego_curr: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_prev: torch.Tensor,
    neighbor_curr: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
    data: dict[str, torch.Tensor],
    goal_xy: torch.Tensor,
    config: StepRewardConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> StepReward:
    """Compute reward for a single simulation step.

    Uses T=2 mini-trajectories to reuse existing reward functions.

    Args:
        ego_prev: [4] (x, y, cos_h, sin_h) at t-1 in current ego frame.
        ego_curr: [4] (x, y, cos_h, sin_h) at t in current ego frame.
        ego_shape: [3] (wheel_base, length, width).
        neighbor_prev: [N_nb, 4] neighbor positions at t-1.
        neighbor_curr: [N_nb, 4] neighbor positions at t.
        neighbor_shapes: [N_nb, 2] (width, length).
        neighbor_valid: [N_nb] bool mask for valid neighbors.
        data: Scene dict with 'line_strings' for road border check.
        goal_xy: [2] goal position in current frame.
        config: Step reward weights.
        reward_config: RewardConfig for underlying collision/border functions.

    Returns:
        StepReward with total, collision, rb_crossing, progress, terminal.
    """
    if config is None:
        config = StepRewardConfig()
    if reward_config is None:
        reward_config = RewardConfig()

    device = ego_curr.device

    # --- Build T=2 mini-trajectory: [1, 2, 4] ---
    ego_mini = torch.stack([ego_prev, ego_curr], dim=0).unsqueeze(0)  # [1, 2, 4]

    # --- Collision check ---
    collision = False
    N_nb = neighbor_curr.shape[0]
    if N_nb > 0:
        nb_mini = torch.stack([neighbor_prev, neighbor_curr], dim=1)  # [N_nb, 2, 4]
        nb_valid = neighbor_valid.unsqueeze(1).expand(-1, 2)  # [N_nb, 2]

        safety_scores, collision_steps = compute_safety_score_batch(
            ego_mini, ego_shape, nb_mini, neighbor_shapes, nb_valid, reward_config,
        )
        collision = collision_steps[0] is not None

    # --- Road border check ---
    rb_crossing = False
    crossing_gate, _, _, rb_steps, _ = compute_road_border_penalty(
        ego_mini, ego_shape, data,
    )
    if crossing_gate[0].item() < 0.5:
        rb_crossing = True

    # --- Progress toward goal ---
    progress = 0.0
    if goal_xy.abs().sum() > 1e-6:
        dist_prev = (ego_prev[:2] - goal_xy).norm().item()
        dist_curr = (ego_curr[:2] - goal_xy).norm().item()
        progress = dist_prev - dist_curr  # positive when approaching
    else:
        # Fallback: forward distance traveled
        progress = (ego_curr[:2] - ego_prev[:2]).norm().item()

    # --- Aggregate ---
    terminal = collision or rb_crossing
    total = (
        config.w_progress * progress
        + config.w_alive
        - config.w_collision * float(collision)
        - config.w_rb_crossing * float(rb_crossing)
    )

    return StepReward(
        total=total,
        collision=collision,
        rb_crossing=rb_crossing,
        progress=progress,
        terminal=terminal,
    )
