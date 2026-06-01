"""Batched closed-loop rollout manager.

Processes all scenes in parallel at each timestep instead of sequentially.
Chunks scenes into mini-batches to fit GPU memory.

Speedup: O(N_scenes × 40 × model_call) → O(40 × ceil(N_scenes/chunk) × model_call)
Expected 5-8x faster than sequential rollout.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig


def make_initial_latent(
    B: int, P: int, future_len: int, device: torch.device,
    noise_scale: float = 0.0,
) -> torch.Tensor:
    """Build the initial sampled_trajectories tensor for DPM-Solver.

    Matches the C++ production node: randn * temperature (zeros when
    temperature/noise_scale is 0).  Shape: (B, P, future_len + 1, 4).
    """
    if noise_scale > 0.0:
        return noise_scale * torch.randn(B, P, future_len + 1, 4, device=device)
    return torch.zeros(B, P, future_len + 1, 4, device=device)
from torch import nn

from exploration_policy.model import ExplorationPolicy
from preference_optimization.utils import load_npz_data as _load_npz_data_raw
from rlvr.closed_loop.gae import compute_gae
from rlvr.closed_loop.per_step_reward import StepRewardConfig, compute_step_reward
from rlvr.closed_loop.rollout import RolloutBuffer, RolloutStep
from rlvr.closed_loop.state_update import (
    advance_neighbor_past,
    build_transform_matrix,
    transform_positions_to_ego_frame,
    update_scene_state,
)
from rlvr.reward import RewardConfig


def _load_npz(npz_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    data = _load_npz_data_raw(npz_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data


@torch.no_grad()
def _batched_generate(
    model: nn.Module,
    model_args,
    batch_data: dict[str, torch.Tensor],
    noise_scale: float,
    composer: GuidanceComposer | None,
    device: torch.device,
) -> torch.Tensor:
    """Generate one trajectory per scene in a batch.

    Args:
        model: Diffusion planner model.
        model_args: Config with predicted_neighbor_num, future_len.
        batch_data: Batched observation dict with B>1.
        noise_scale: Noise for initial latent.
        composer: GuidanceComposer or None.
        device: Torch device.

    Returns:
        [B, T, 4] ego trajectories (x, y, cos, sin).
    """
    # NOTE: directly accessing decoder private attrs (_guidance_fn, _guidance_scale)
    # because the decoder has no public API for temporary guidance override.
    # The try/finally blocks below ensure restoration even on exceptions.
    _orig_fn = model.decoder._guidance_fn
    _orig_scale = model.decoder._guidance_scale
    model.decoder._guidance_fn = composer
    if composer is not None:
        model.decoder._guidance_scale = composer._set_config.global_scale
    else:
        model.decoder._guidance_scale = 0.5

    B = batch_data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    batch_data["sampled_trajectories"] = make_initial_latent(
        B, P, future_len, device, noise_scale,
    )

    try:
        _, decoder_output = model(batch_data)
        # [B, P, T, 4] -> [B, T, 4] (ego only, index 0)
        ego_trajs = decoder_output["prediction"][:, 0].detach()
    finally:
        model.decoder._guidance_fn = _orig_fn
        model.decoder._guidance_scale = _orig_scale

    return ego_trajs


@torch.no_grad()
def _batched_encoder(model: nn.Module, batch_data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run encoder on a batch of scenes.

    Args:
        model: Diffusion planner (possibly LoRA-wrapped).
        batch_data: Batched observation dict.

    Returns:
        [B, N, D_enc] scene encoding.
    """
    inner = model.module if hasattr(model, "module") else model
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        planner = inner.base_model.model
    else:
        planner = inner
    return planner.encoder(batch_data).detach()


@torch.no_grad()
def _batched_generate_varied_noise(
    model: nn.Module,
    model_args,
    batch_data: dict[str, torch.Tensor],
    noise_min: float,
    noise_max: float,
    first_deterministic: bool,
    composer: GuidanceComposer | None,
    device: torch.device,
) -> torch.Tensor:
    """Generate trajectories with per-element independent noise scales.

    Unlike _batched_generate which uses a single noise_scale for all B elements,
    this generates independent random noise_scale per trajectory, preserving
    the diversity of sequential generation while keeping batched GPU execution.

    Args:
        noise_min, noise_max: Range for uniform random noise scale per trajectory.
        first_deterministic: If True, element 0 gets noise=0 (deterministic).

    Returns:
        [B, T, 4] ego trajectories.
    """
    # NOTE: same private-attr pattern as _batched_generate above (see comment there).
    _orig_fn = model.decoder._guidance_fn
    _orig_scale = model.decoder._guidance_scale
    model.decoder._guidance_fn = composer
    if composer is not None:
        model.decoder._guidance_scale = composer._set_config.global_scale
    else:
        model.decoder._guidance_scale = 0.5

    B = batch_data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Per-element noise with independent scales
    noise_scales = torch.zeros(B, 1, 1, 1, device=device)
    for i in range(B):
        if first_deterministic and i == 0:
            noise_scales[i] = 0.0
        else:
            noise_scales[i] = random.uniform(noise_min, noise_max)

    raw_noise = torch.randn(B, P, future_len + 1, 4, device=device)
    xT = noise_scales * raw_noise

    batch_data["sampled_trajectories"] = xT

    try:
        _, decoder_output = model(batch_data)
        ego_trajs = decoder_output["prediction"][:, 0].detach()
    finally:
        model.decoder._guidance_fn = _orig_fn
        model.decoder._guidance_scale = _orig_scale

    return ego_trajs


class BatchedRolloutManager:
    """Processes all scenes in parallel at each timestep.

    Instead of: for scene in scenes: for step in 40: model(scene)
    Does:       for step in 40: for chunk in chunks(scenes, B): model(chunk)

    This exploits GPU parallelism for massive speedup.
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
        batch_size: int = 16,
        drop_last: bool = True,
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
        # Online explorer training: update every N steps during rollout
        # 0 = offline (default, train after full rollout)
        # >0 = online PPO update every N steps (PlannerRFT-style)
        self.online_update_interval = 0
        self.online_lr = 2.5e-4
        self.online_entropy_coef = 0.01
        self.online_value_coef = 0.5
        self.explorer_mini_batch = 0  # 0 = accumulate all scenes, >0 = step every N scenes
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _normalize_batch(self, batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply observation normalizer to a batched dict."""
        normalizer = copy.deepcopy(self.model_args.observation_normalizer)
        norm = {}
        for k, v in batch_data.items():
            norm[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return normalizer(norm)

    def _build_composer(self, eta_lat: float, eta_lon: float) -> GuidanceComposer:
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

    def _build_batched_composer(
        self, eta_lat_batch: torch.Tensor, eta_lon_batch: torch.Tensor,
    ) -> GuidanceComposer:
        """Build composer with batched etas [B] for GPU-parallel guidance.

        The lateral/longitudinal guidance functions support tensor etas,
        so this creates a single composer that applies per-element guidance.
        """
        guidance_fns = [
            GuidanceConfig(
                name="lateral", enabled=True, scale=1.0,
                params={"lambda_lat": self.lambda_lat, "eta_lat": eta_lat_batch},
            ),
            GuidanceConfig(
                name="longitudinal", enabled=True, scale=1.0,
                params={"lambda_lon": self.lambda_lon, "eta_lon": eta_lon_batch},
            ),
        ]
        set_cfg = GuidanceSetConfig(functions=guidance_fns, global_scale=self.guidance_scale)
        return GuidanceComposer(set_cfg)

    def run_rollouts(self, npz_paths: list[str]) -> list[RolloutBuffer]:
        """Run closed-loop rollouts for all scenes with batched inference.

        Args:
            npz_paths: List of NPZ file paths.

        Returns:
            List of RolloutBuffer, one per scene (excluding failed loads).
        """
        # --- Phase 1: Load all scenes ---
        scene_data: list[dict[str, torch.Tensor]] = []
        scene_paths: list[str] = []
        nb_futures: list[torch.Tensor | None] = []

        for path in npz_paths:
            try:
                data = _load_npz(path, self.device)
                # Extract GT neighbor futures before they get consumed
                nf = None
                if "neighbor_agents_future" in data:
                    nf = data["neighbor_agents_future"]
                    if nf.dim() == 4:
                        nf = nf[0]  # [N_nb, T, 3]
                scene_data.append(data)
                scene_paths.append(path)
                nb_futures.append(nf)
            except Exception as e:
                print(f"  [batched_rollout] Failed to load {path}: {e}")

        N = len(scene_data)
        if N == 0:
            return []

        # Apply drop_last
        if self.drop_last and N % self.batch_size != 0:
            keep = (N // self.batch_size) * self.batch_size
            if keep == 0:
                keep = N  # don't drop everything
            scene_data = scene_data[:keep]
            scene_paths = scene_paths[:keep]
            nb_futures = nb_futures[:keep]
            N = len(scene_data)

        # --- Initialize tracking per scene ---
        ego_abs = [[0.0, 0.0, 0.0] for _ in range(N)]  # [x, y, heading] in original frame
        buffers = [RolloutBuffer(npz_path=scene_paths[i]) for i in range(N)]
        active = [True] * N  # whether scene is still running
        ego_prev = [torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device) for _ in range(N)]

        if self.exploration_policy is not None:
            self.exploration_policy.eval()

        # --- Phase 2: Step-by-step rollout with batched inference ---
        for step_t in range(self.rollout_steps):
            active_indices = [i for i in range(N) if active[i]]
            if not active_indices:
                break

            # Process in chunks
            for chunk_start in range(0, len(active_indices), self.batch_size):
                chunk_idx = active_indices[chunk_start:chunk_start + self.batch_size]
                B_chunk = len(chunk_idx)

                # Stack scene data into batch
                batch_data = {}
                for k in scene_data[chunk_idx[0]].keys():
                    vals = [scene_data[i][k] for i in chunk_idx]
                    if isinstance(vals[0], torch.Tensor):
                        batch_data[k] = torch.cat(vals, dim=0)  # [B_chunk, ...]
                    else:
                        batch_data[k] = vals[0]  # non-tensor (rare)

                # Normalize
                norm_data = self._normalize_batch(batch_data)

                # Encoder
                scene_encoding = _batched_encoder(self.policy_model, norm_data)

                # Reference trajectory (LoRA-disabled, deterministic)
                import contextlib
                inner = self.policy_model.module if hasattr(self.policy_model, "module") else self.policy_model
                use_lora_disable = hasattr(inner, "disable_adapter")
                disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()

                with disable_ctx:
                    ref_trajs = _batched_generate(
                        self.policy_model, self.model_args, norm_data,
                        noise_scale=0.0, composer=None, device=self.device,
                    )  # [B_chunk, T, 4]

                norm_data["x_ref"] = ref_trajs
                norm_data["reference_trajectory"] = ref_trajs  # Required by lateral/longitudinal guidance

                # Explorer policy (batched) — or zero-init if no explorer
                noise = random.uniform(*self.noise_range)
                if self.exploration_policy is not None:
                    policy_out = self.exploration_policy(
                        scene_encoding, ref_trajs, deterministic=False,
                    )
                    eta_lat_batch = policy_out.eta_lat[:B_chunk]
                    eta_lon_batch = policy_out.eta_lon[:B_chunk]
                else:
                    # No explorer — use zero guidance (equivalent to zero-init)
                    policy_out = None
                    eta_lat_batch = torch.zeros(B_chunk, device=self.device)
                    eta_lon_batch = torch.zeros(B_chunk, device=self.device)

                # Build batched composer — guidance functions support tensor etas [B]
                composer = self._build_batched_composer(eta_lat_batch, eta_lon_batch)

                # Batched guided trajectory generation
                guided_trajs = _batched_generate(
                    self.policy_model, self.model_args, norm_data,
                    noise_scale=noise, composer=composer, device=self.device,
                )  # [B_chunk, T, 4]
                chunk_trajs = [guided_trajs[i] for i in range(B_chunk)]

                # Process each scene in chunk: reward, state update, store
                for local_idx, global_idx in enumerate(chunk_idx):
                    trajectory = chunk_trajs[local_idx]  # [T, 4]
                    ego_curr = trajectory[0].clone()

                    # Eta values for this scene
                    if policy_out is not None:
                        eta_lat_01_raw = (policy_out.eta_lat[local_idx].item() + 1.0) / 2.0
                        eta_lon_01_raw = (policy_out.eta_lon[local_idx].item() + 1.0) / 2.0
                        log_prob = (policy_out.log_prob_lat[local_idx].item()
                                   + policy_out.log_prob_lon[local_idx].item())
                        value = policy_out.value[local_idx].item()
                    else:
                        eta_lat_01_raw = 0.5  # zero-init maps to 0.5 in (0,1) space
                        eta_lon_01_raw = 0.5
                        log_prob = 0.0
                        value = 0.0

                    # Get neighbor positions for reward
                    data_i = scene_data[global_idx]
                    ego_shape = data_i.get("ego_shape", torch.tensor([[2.79, 4.34, 1.70]], device=self.device))
                    if ego_shape.dim() == 2:
                        ego_shape = ego_shape[0]

                    goal_xy = torch.zeros(2, device=self.device)
                    if "goal_pose" in data_i:
                        gp = data_i["goal_pose"]
                        goal_xy = gp[0, :2] if gp.dim() == 2 else gp[:2]

                    nb_prev = torch.zeros(0, 4, device=self.device)
                    nb_curr = torch.zeros(0, 4, device=self.device)
                    nb_valid = torch.zeros(0, dtype=torch.bool, device=self.device)
                    nb_shapes = torch.zeros(0, 2, device=self.device)

                    if "neighbor_agents_past" in data_i:
                        nb_data = data_i["neighbor_agents_past"]
                        if nb_data.dim() == 4:
                            nb_data = nb_data[0]
                        nb_prev = nb_data[:, -1, :4].clone()
                        nb_shapes = nb_data[:, -1, 6:8]

                        nf = nb_futures[global_idx]
                        if nf is not None and step_t < nf.shape[1]:
                            ax, ay, ah = ego_abs[global_idx]
                            nb_curr = transform_positions_to_ego_frame(
                                nf[:, step_t, :], ax, ay, ah, self.device,
                            )
                        else:
                            nb_curr = nb_prev.clone()

                        nb_valid = nb_prev[:, :2].abs().sum(dim=-1) > 0.1

                    # Per-step reward
                    step_reward = compute_step_reward(
                        ego_prev=ego_prev[global_idx],
                        ego_curr=ego_curr,
                        ego_shape=ego_shape,
                        neighbor_prev=nb_prev,
                        neighbor_curr=nb_curr,
                        neighbor_shapes=nb_shapes,
                        neighbor_valid=nb_valid,
                        data=data_i,
                        goal_xy=goal_xy,
                        config=self.step_reward_config,
                        reward_config=self.reward_config,
                    )

                    # Store step
                    buffers[global_idx].steps.append(RolloutStep(
                        scene_encoding=scene_encoding[local_idx:local_idx+1].detach().cpu(),
                        x_ref=ref_trajs[local_idx:local_idx+1].detach().cpu(),
                        eta_lat_01=eta_lat_01_raw,
                        eta_lon_01=eta_lon_01_raw,
                        log_prob=log_prob,
                        value=value,
                        reward=step_reward.total,
                        terminal=step_reward.terminal,
                    ))
                    buffers[global_idx].total_return += step_reward.total
                    buffers[global_idx].episode_length = step_t + 1

                    if step_reward.terminal:
                        active[global_idx] = False
                        continue

                    # Update ego absolute pose
                    ax, ay, ah = ego_abs[global_idx]
                    dx = ego_curr[0].item()
                    dy = ego_curr[1].item()
                    cos_h = math.cos(ah)
                    sin_h = math.sin(ah)
                    ego_abs[global_idx][0] += dx * cos_h - dy * sin_h
                    ego_abs[global_idx][1] += dx * sin_h + dy * cos_h
                    dh = math.atan2(ego_curr[3].item(), ego_curr[2].item())
                    ego_abs[global_idx][2] += dh

                    # Advance neighbors
                    if nb_curr.numel() > 0:
                        advance_neighbor_past(data_i, nb_curr, dt=0.1)

                    # Update scene state
                    scene_data[global_idx], _ = update_scene_state(
                        data_i, trajectory.unsqueeze(0), step_idx=0, dt=0.1,
                    )

                    ego_prev[global_idx] = torch.tensor(
                        [0.0, 0.0, 1.0, 0.0], device=self.device,
                    )

                    # Update goal in new frame
                    if "goal_pose" in scene_data[global_idx]:
                        gp = scene_data[global_idx]["goal_pose"]
                        goal_xy = gp[0, :2] if gp.dim() == 2 else gp[:2]

            # --- Online explorer update (PlannerRFT-style) ---
            if (self.online_update_interval > 0
                and self.exploration_policy is not None
                and (step_t + 1) % self.online_update_interval == 0):
                self._online_explorer_update(buffers, active, step_t)

        # --- Phase 3: Compute GAE for all buffers ---
        for buf in buffers:
            if len(buf.steps) > 0:
                rewards = [s.reward for s in buf.steps]
                values = [s.value for s in buf.steps]
                terminal_value = 0.0 if buf.steps[-1].terminal else values[-1]
                advantages, value_targets = compute_gae(
                    rewards, values, terminal_value,
                    gamma=self.gamma, lam=self.gae_lambda,
                )
                buf.advantages = advantages
                buf.value_targets = value_targets

        return [b for b in buffers if len(b.steps) > 0]

    def _online_explorer_update(self, buffers, active, current_step):
        """Mid-rollout PPO update for exploration policy (PlannerRFT-style).

        Computes GAE on recent steps and does a gradient update on the explorer.
        This lets the explorer improve DURING the rollout, not just after.

        When explorer_mini_batch > 0, steps the optimizer every N scenes
        instead of accumulating across all scenes. This prevents the gradient
        from averaging out per-scene signal into a global bias.
        """
        if self.exploration_policy is None:
            return

        import torch

        interval = self.online_update_interval
        mini_batch = self.explorer_mini_batch
        self.exploration_policy.train()

        if not hasattr(self, '_online_optimizer'):
            from torch import optim
            self._online_optimizer = optim.AdamW(
                self.exploration_policy.parameters(), lr=self.online_lr,
            )

        self._online_optimizer.zero_grad()
        n_in_batch = 0

        for i, buf in enumerate(buffers):
            if not active[i] or len(buf.steps) < interval:
                continue

            recent = buf.steps[-interval:]
            rewards = [s.reward for s in recent]
            values = [s.value for s in recent]
            terminal_value = 0.0 if recent[-1].terminal else values[-1]

            advantages, value_targets = compute_gae(
                rewards, values, terminal_value,
                gamma=self.gamma, lam=self.gae_lambda,
            )

            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Determine divisor: mini_batch size or total active scenes
            if mini_batch > 0:
                divisor = interval * mini_batch
            else:
                divisor = interval * max(sum(active), 1)

            for t, step in enumerate(recent):
                scene_enc = step.scene_encoding.to(self.device)
                x_ref = step.x_ref.to(self.device)

                policy_out = self.exploration_policy(scene_enc, x_ref, deterministic=False)

                eta_lat_01 = torch.tensor(
                    step.eta_lat_01, dtype=torch.float32, device=self.device,
                ).clamp(1e-6, 1 - 1e-6)
                eta_lon_01 = torch.tensor(
                    step.eta_lon_01, dtype=torch.float32, device=self.device,
                ).clamp(1e-6, 1 - 1e-6)

                log_prob = (policy_out.lat_dist.log_prob(eta_lat_01)
                           + policy_out.lon_dist.log_prob(eta_lon_01))

                adv = advantages[t].to(self.device)
                reinforce_loss = -(log_prob * adv.detach())
                value_loss = (policy_out.value.squeeze() - value_targets[t].to(self.device).detach()) ** 2
                entropy = policy_out.lat_dist.entropy() + policy_out.lon_dist.entropy()

                step_loss = (reinforce_loss
                            + self.online_value_coef * value_loss
                            - self.online_entropy_coef * entropy)
                step_loss = step_loss / divisor
                step_loss.backward()

            n_in_batch += 1

            # Step optimizer every mini_batch scenes
            if mini_batch > 0 and n_in_batch >= mini_batch:
                torch.nn.utils.clip_grad_norm_(self.exploration_policy.parameters(), max_norm=1.0)
                self._online_optimizer.step()
                self._online_optimizer.zero_grad()
                n_in_batch = 0

        # Final step for remaining scenes (or all scenes if mini_batch=0)
        if n_in_batch > 0:
            torch.nn.utils.clip_grad_norm_(self.exploration_policy.parameters(), max_norm=1.0)
            self._online_optimizer.step()

        self.exploration_policy.eval()
