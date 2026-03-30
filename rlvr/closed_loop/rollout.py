"""Closed-loop rollout manager.

Orchestrates N-step closed-loop simulation for a single scene:
  1. Encode scene → get reference trajectory → explorer samples eta
  2. DiT generates guided trajectory → execute first step
  3. Compute per-step reward → update scene state
  4. After rollout: compute GAE advantages

The simulation backend (DiT-based state update) is abstracted behind
the step logic, making it possible to swap in an external simulator later.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from exploration_policy.model import ExplorationPolicy
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data as _load_npz_data_raw
from rlvr.closed_loop.gae import compute_gae
from rlvr.closed_loop.per_step_reward import StepRewardConfig, compute_step_reward
from rlvr.closed_loop.state_update import (
    advance_neighbor_past,
    transform_positions_to_ego_frame,
    update_scene_state,
)
from rlvr.reward import RewardConfig


def _load_npz(npz_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    data = _load_npz_data_raw(npz_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data


@dataclass
class RolloutStep:
    """Data collected at one simulation step (all detached, no grad graph)."""
    scene_encoding: torch.Tensor   # [1, N, D]
    x_ref: torch.Tensor            # [1, T, 4]
    eta_lat_01: float              # sampled value in (0, 1)
    eta_lon_01: float              # sampled value in (0, 1)
    log_prob: float
    value: float
    reward: float
    terminal: bool


@dataclass
class RolloutBuffer:
    """Complete rollout data for one scene, ready for training."""
    steps: list[RolloutStep] = field(default_factory=list)
    advantages: torch.Tensor | None = None
    value_targets: torch.Tensor | None = None
    npz_path: str = ""
    total_return: float = 0.0
    episode_length: int = 0


class RolloutManager:
    """Manages closed-loop rollout for one scene at a time.

    Decoupled from the training loop: collects (s, a, r, V, log_prob) tuples.
    The trainer handles gradient computation and policy updates.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        model_args,
        exploration_policy: ExplorationPolicy,
        device: torch.device,
        lambda_lat: float = 2.5,
        lambda_lon: float = 0.25,
        guidance_scale: float = 1.0,
        rollout_steps: int = 40,
        noise_range: tuple[float, float] = (0.5, 2.0),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        step_reward_config: StepRewardConfig | None = None,
        reward_config: RewardConfig | None = None,
    ):
        self.policy_model = policy_model
        self.model_args = model_args
        self.exploration_policy = exploration_policy
        self.device = device
        self.lambda_lat = lambda_lat
        self.lambda_lon = lambda_lon
        self.guidance_scale = guidance_scale
        self.rollout_steps = rollout_steps
        self.noise_range = noise_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.step_reward_config = step_reward_config or StepRewardConfig()
        self.reward_config = reward_config or RewardConfig()

    def _normalize_data(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply observation normalizer (on a copy to avoid corrupting original)."""
        norm_data = {}
        normalizer = copy.deepcopy(self.model_args.observation_normalizer)
        for k, v in data.items():
            norm_data[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return normalizer(norm_data)

    def _build_composer(self, eta_lat: float, eta_lon: float) -> GuidanceComposer:
        """Build GuidanceComposer for a given eta pair."""
        guidance_fns = [
            GuidanceConfig(
                name="lateral", enabled=True, scale=1.0,
                params={"lambda_lat": self.lambda_lat, "eta_lat": eta_lat},
            ),
            GuidanceConfig(
                name="longitudinal", enabled=True, scale=1.0,
                params={"lambda_lon": self.lambda_lon, "eta_lon": eta_lon},
            ),
        ]
        set_cfg = GuidanceSetConfig(functions=guidance_fns, global_scale=self.guidance_scale)
        return GuidanceComposer(set_cfg)

    @torch.no_grad()
    def run_rollout(self, npz_path: str) -> RolloutBuffer | None:
        """Run closed-loop rollout for a single scene.

        Returns:
            RolloutBuffer with steps, GAE advantages, and value targets.
            None if the scene fails to load or has no valid data.
        """
        try:
            data = _load_npz(npz_path, self.device)
        except Exception as e:
            print(f"  [rollout] Failed to load {npz_path}: {e}")
            return None

        # Extract GT neighbor futures for ghost replay [N_nb, T_future, 3] (x, y, heading_rad)
        neighbor_futures = None
        if "neighbor_agents_future" in data:
            nf = data["neighbor_agents_future"]
            if nf.dim() == 3:
                neighbor_futures = nf  # [1, N_nb, T_future, 3] or [N_nb, T_future, 3]
            if nf.dim() == 4:
                neighbor_futures = nf[0]  # remove batch dim
            else:
                neighbor_futures = nf
        # Ensure [N_nb, T_future, 3]
        if neighbor_futures is not None and neighbor_futures.dim() == 4:
            neighbor_futures = neighbor_futures[0]

        # Extract static info
        ego_shape = data.get("ego_shape", torch.tensor([[2.79, 4.34, 1.70]], device=self.device))
        if ego_shape.dim() == 2:
            ego_shape = ego_shape[0]  # [3]

        # Goal position for progress reward
        goal_xy = torch.zeros(2, device=self.device)
        if "goal_pose" in data:
            gp = data["goal_pose"]
            if gp.dim() == 2:
                goal_xy = gp[0, :2]
            elif gp.dim() == 1:
                goal_xy = gp[:2]

        # Neighbor shapes [N_nb, 2] (width, length)
        nb_shapes = torch.zeros(0, 2, device=self.device)
        if "neighbor_agents_past" in data:
            nb_past = data["neighbor_agents_past"]
            if nb_past.dim() == 4:
                nb_past_0 = nb_past[0]  # [N_nb, T, 11]
            else:
                nb_past_0 = nb_past
            # width=col6, length=col7
            nb_shapes = nb_past_0[:, -1, 6:8]  # [N_nb, 2]

        # Track ego absolute pose in original frame for neighbor transform
        ego_abs_x, ego_abs_y, ego_abs_heading = 0.0, 0.0, 0.0

        # Previous ego position for reward (starts at origin)
        ego_prev = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)

        buffer = RolloutBuffer(npz_path=npz_path)

        self.exploration_policy.eval()

        for step_t in range(self.rollout_steps):
            # 1. Normalize and encode
            norm_data = self._normalize_data(data)

            scene_encoding = run_frozen_encoder(self.policy_model, norm_data)

            # 2. Generate reference trajectory
            x_ref_np = generate_reference_trajectory(
                self.policy_model, self.model_args, norm_data, self.device,
            )
            x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(self.device)  # [1, T, 4]
            norm_data["x_ref"] = x_ref

            # 3. Explorer policy: sample eta, get value + log_prob
            policy_out = self.exploration_policy(scene_encoding, x_ref, deterministic=False)
            eta_lat_01 = policy_out.eta_lat.item()  # NOTE: .eta_lat is in [-1,1]
            eta_lon_01 = policy_out.eta_lon.item()
            # Convert back to (0,1) for log_prob storage
            eta_lat_01_raw = (eta_lat_01 + 1.0) / 2.0
            eta_lon_01_raw = (eta_lon_01 + 1.0) / 2.0
            log_prob = (policy_out.log_prob_lat.item() + policy_out.log_prob_lon.item())
            value = policy_out.value.item()

            # 4. Generate 1 guided trajectory with noise
            eta_lat = eta_lat_01  # already in [-1, 1]
            eta_lon = eta_lon_01
            composer = self._build_composer(eta_lat, eta_lon)
            noise = random.uniform(*self.noise_range)

            traj_np = generate_samples(
                model=self.policy_model, model_args=self.model_args,
                data=norm_data, noise_scale=noise, n_samples=1,
                composer=composer, device=self.device,
            )[0]  # (T, 4)
            trajectory = torch.from_numpy(traj_np).to(self.device)

            # 5. Get ego new position and neighbor positions for reward
            ego_curr = trajectory[0].clone()  # [4] new ego in current frame

            # Neighbor current positions for collision check
            nb_prev = torch.zeros(0, 4, device=self.device)
            nb_curr = torch.zeros(0, 4, device=self.device)
            nb_valid = torch.zeros(0, dtype=torch.bool, device=self.device)

            if "neighbor_agents_past" in data:
                nb_data = data["neighbor_agents_past"]
                if nb_data.dim() == 4:
                    nb_data = nb_data[0]  # [N_nb, T, 11]
                nb_prev = nb_data[:, -1, :4].clone()  # [N_nb, 4]

                # Get neighbor position at step t+1 from GT futures
                if neighbor_futures is not None and step_t < neighbor_futures.shape[1]:
                    nb_gt_orig = neighbor_futures[:, step_t, :]  # [N_nb, 3] in original frame
                    nb_curr_4d = transform_positions_to_ego_frame(
                        nb_gt_orig, ego_abs_x, ego_abs_y, ego_abs_heading, self.device,
                    )
                    nb_curr = nb_curr_4d  # [N_nb, 4] in current ego frame
                else:
                    # No GT future available, use static positions
                    nb_curr = nb_prev.clone()

                nb_valid_mask = nb_prev[:, :2].abs().sum(dim=-1) > 0.1
                nb_valid = nb_valid_mask

            # 6. Compute per-step reward
            step_reward = compute_step_reward(
                ego_prev=ego_prev,
                ego_curr=ego_curr,
                ego_shape=ego_shape,
                neighbor_prev=nb_prev,
                neighbor_curr=nb_curr,
                neighbor_shapes=nb_shapes,
                neighbor_valid=nb_valid,
                data=data,
                goal_xy=goal_xy,
                config=self.step_reward_config,
                reward_config=self.reward_config,
            )

            # 7. Store step
            buffer.steps.append(RolloutStep(
                scene_encoding=scene_encoding.detach().cpu(),
                x_ref=x_ref.detach().cpu(),
                eta_lat_01=eta_lat_01_raw,
                eta_lon_01=eta_lon_01_raw,
                log_prob=log_prob,
                value=value,
                reward=step_reward.total,
                terminal=step_reward.terminal,
            ))

            buffer.total_return += step_reward.total
            buffer.episode_length = step_t + 1

            if step_reward.terminal:
                break

            # 8. Update ego absolute pose (for neighbor GT transform)
            cos_h = math.cos(ego_abs_heading)
            sin_h = math.sin(ego_abs_heading)
            dx_local = ego_curr[0].item()
            dy_local = ego_curr[1].item()
            ego_abs_x += dx_local * cos_h - dy_local * sin_h
            ego_abs_y += dx_local * sin_h + dy_local * cos_h
            delta_heading = math.atan2(ego_curr[3].item(), ego_curr[2].item())
            ego_abs_heading += delta_heading

            # 9. Advance neighbors in data
            if nb_curr.numel() > 0:
                advance_neighbor_past(data, nb_curr, dt=0.1)

            # 10. Update scene state (re-center to new ego pose)
            data, _ = update_scene_state(data, trajectory, step_idx=0, dt=0.1)

            # Update prev ego position (always [0,0,1,0] after re-centering)
            ego_prev = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)

            # Update goal in new frame
            if "goal_pose" in data:
                gp = data["goal_pose"]
                if gp.dim() == 2:
                    goal_xy = gp[0, :2]
                elif gp.dim() == 1:
                    goal_xy = gp[:2]

        # 11. Compute GAE
        if len(buffer.steps) > 0:
            rewards = [s.reward for s in buffer.steps]
            values = [s.value for s in buffer.steps]
            terminal_value = 0.0 if buffer.steps[-1].terminal else values[-1]

            advantages, value_targets = compute_gae(
                rewards, values, terminal_value,
                gamma=self.gamma, lam=self.gae_lambda,
            )
            buffer.advantages = advantages
            buffer.value_targets = value_targets

        return buffer
