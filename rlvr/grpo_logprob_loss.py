"""DDV2-style Gaussian log-probability GRPO loss.

Two-stage approach mirroring DiffusionDriveV2:
  Stage 1 (collect_logprob_rollout): Run truncated denoising, collect chain + log-probs.
  Stage 2 (compute_logprob_grpo_loss): Re-run on stored chain, compute RL + IL loss.

Reference: DiffusionDriveV2 diffusiondrivev2_model_rl.py lines 800-1120
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import generate_prefix_mask

from rlvr.vpsde_logprob import (
    compute_discount_weights,
    create_timestep_schedule,
)


def _build_model_inputs(
    data: dict,
    trajectories_norm: torch.Tensor,
    t_value: float,
    x_t: torch.Tensor,
    model_args,
    device: torch.device,
    N: int,
) -> Tuple[dict, torch.Tensor]:
    """Build merged model input dict for N trajectories at a specific (x_t, t).

    This reuses the pattern from compute_batched_trajectory_losses in grpo_loss.py
    but with a fixed t and a specific x_t (from the denoising chain) rather than
    sampling them randomly.

    Args:
        data: Scene observation dict (already B=N expanded).
        trajectories_norm: [N, T, 4] normalized ego trajectories (for gt_future).
        t_value: Scalar diffusion time for this step.
        x_t: [N, P, T+1, 4] noisy trajectories at time t (the denoising chain state).
        model_args: Config with state_normalizer, observation_normalizer, etc.
        device: Torch device.
        N: Batch size.

    Returns:
        Merged input dict ready for model forward pass.
    """
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)

    # Build gt_future: [N, P, T, 4]
    gt_future = torch.zeros(N, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = trajectories_norm

    # Fill neighbor GT
    Pn = P - 1
    if Pn > 0 and "neighbor_agents_future" in data:
        nf = data["neighbor_agents_future"]
        nf_pn = min(nf.shape[1], Pn)
        nf_4d = torch.zeros(N, nf_pn, future_len, 4, device=device)
        nf_4d[..., :2] = nf[:, :nf_pn, :future_len, :2]
        if nf.shape[-1] >= 3:
            heading = nf[:, :nf_pn, :future_len, 2]
            nf_4d[..., 2] = torch.cos(heading)
            nf_4d[..., 3] = torch.sin(heading)
        nf_4d_norm = (nf_4d - ego_mean) / ego_std
        gt_future[:, 1:1 + nf_pn, :, :] = nf_4d_norm

    # Current states
    ego_current = data["ego_current_state"][:, :4]
    ego_current_norm = (ego_current - ego_mean) / ego_std
    if P > 1:
        neighbors_current = data["neighbor_agents_past"][:, :P - 1, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    else:
        neighbors_current_norm = torch.zeros(N, 0, 4, device=device)
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)

    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [N, P, T+1, 4]

    # Zero out invalid neighbors
    if Pn > 0 and "neighbor_agents_future" in data:
        nf = data["neighbor_agents_future"]
        nf_pn = min(nf.shape[1], Pn)
        neighbor_current_mask = (data["neighbor_agents_past"][:, :Pn, -1, :4].abs().sum(dim=-1) == 0)
        nf_valid = (nf[:, :nf_pn, :future_len, :2].abs().sum(dim=-1) > 0.1)
        nf_mask = ~nf_valid
        full_neighbor_mask = torch.cat(
            [neighbor_current_mask[:, :nf_pn].unsqueeze(-1), nf_mask], dim=-1
        )
        neighbor_slice = all_gt[:, 1:1 + nf_pn]
        neighbor_slice.masked_fill_(
            full_neighbor_mask.unsqueeze(-1).expand_as(neighbor_slice), 0.0
        )

    # Build t tensor [N, P, T+1, 1]
    t_4d = torch.full((N, P, future_len + 1, 1), t_value, device=device)

    # Prefix mask (no random delay for logprob — use fixed delay=0)
    delay = torch.zeros(N, dtype=torch.long, device=device)
    prefix_mask = generate_prefix_mask(delay, P, future_len + 1)

    # Normalize observation data
    data_normalized = model_args.observation_normalizer(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    )

    merged = {**data_normalized}
    merged["gt_trajectories"] = all_gt
    merged["sampled_trajectories"] = x_t
    merged["diffusion_time"] = t_4d
    merged["prefix_mask"] = prefix_mask
    if "delay" not in merged:
        merged["delay"] = delay

    return merged, all_gt


def _expand_data(data: dict, N: int) -> dict:
    """Expand scene data from B=1 to B=N."""
    batch_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == 1:
            batch_data[k] = v.expand(N, *v.shape[1:]).contiguous()
        else:
            batch_data[k] = v
    return batch_data


@torch.no_grad()
def collect_logprob_rollout(
    model: torch.nn.Module,
    data: dict,
    trajectories: torch.Tensor,
    model_args,
    config,
    device: torch.device,
) -> Dict[str, Any]:
    """Stage 1: Run truncated denoising rollout, collect chain states and log-probs.

    Mirrors DDV2 forward_train_rl (lines 800-939): start from noised trajectories,
    denoise step by step collecting the chain and per-step log-probabilities.

    Args:
        model: Diffusion planner model (eval mode for collection).
        data: Scene observation dict with B=1.
        trajectories: [N, T, 4] generated trajectories to denoise from.
        model_args: Config namespace.
        config: GRPOConfig with logprob settings.
        device: Torch device.

    Returns:
        Dict with:
            "chain": list of [N, P, T+1, 4] tensors (len = num_steps + 1)
            "log_probs": [N, num_steps] per-step log-probs
            "trajectories_norm": [N, T, 4] normalized ego trajectories
    """
    sde = VPSDE_linear()
    N = trajectories.shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    num_steps = config.logprob_num_steps
    t_start = config.logprob_t_start
    min_std = config.logprob_min_std

    # Normalize ego trajectories
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    traj_norm = (trajectories - ego_mean) / ego_std  # [N, T, 4]

    # Expand data from B=1 to B=N
    batch_data = _expand_data(data, N)

    # Build initial all_gt for current states + neighbor GT
    # (we need this to construct the full x_t with current state prefix)
    gt_future = torch.zeros(N, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = traj_norm

    Pn = P - 1
    if Pn > 0 and "neighbor_agents_future" in batch_data:
        nf = batch_data["neighbor_agents_future"]
        nf_pn = min(nf.shape[1], Pn)
        nf_4d = torch.zeros(N, nf_pn, future_len, 4, device=device)
        nf_4d[..., :2] = nf[:, :nf_pn, :future_len, :2]
        if nf.shape[-1] >= 3:
            heading = nf[:, :nf_pn, :future_len, 2]
            nf_4d[..., 2] = torch.cos(heading)
            nf_4d[..., 3] = torch.sin(heading)
        nf_4d_norm = (nf_4d - ego_mean) / ego_std
        gt_future[:, 1:1 + nf_pn, :, :] = nf_4d_norm

    ego_current = batch_data["ego_current_state"][:, :4]
    ego_current_norm = (ego_current - ego_mean) / ego_std
    if P > 1:
        neighbors_current = batch_data["neighbor_agents_past"][:, :P - 1, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    else:
        neighbors_current_norm = torch.zeros(N, 0, 4, device=device)
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)

    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [N, P, T+1, 4]

    # Zero invalid neighbors
    if Pn > 0 and "neighbor_agents_future" in batch_data:
        nf = batch_data["neighbor_agents_future"]
        nf_pn = min(nf.shape[1], Pn)
        neighbor_current_mask = (batch_data["neighbor_agents_past"][:, :Pn, -1, :4].abs().sum(dim=-1) == 0)
        nf_valid = (nf[:, :nf_pn, :future_len, :2].abs().sum(dim=-1) > 0.1)
        nf_mask = ~nf_valid
        full_neighbor_mask = torch.cat(
            [neighbor_current_mask[:, :nf_pn].unsqueeze(-1), nf_mask], dim=-1
        )
        neighbor_slice = all_gt[:, 1:1 + nf_pn]
        neighbor_slice.masked_fill_(
            full_neighbor_mask.unsqueeze(-1).expand_as(neighbor_slice), 0.0
        )

    # Add noise at t_start to get x_{t_start}
    # Only noise the future part (index 1:), keep current state clean
    t_start_tensor = torch.tensor(t_start, device=device).view(1, 1, 1, 1)
    future_gt = all_gt[:, :, 1:, :]  # [N, P, T, 4]
    mean_start, std_start = sde.marginal_prob(future_gt, t_start_tensor)
    noise = torch.randn_like(future_gt)
    x_t_future = mean_start + std_start * noise
    x_t = torch.cat([all_gt[:, :, :1, :], x_t_future], dim=2)  # [N, P, T+1, 4]

    # Create timestep schedule
    schedule = create_timestep_schedule(t_start, num_steps)  # (num_steps + 1,)

    chain = [x_t.clone()]
    all_log_probs = []

    model.eval()
    for step_idx in range(num_steps):
        t_current = schedule[step_idx].item()
        t_prev = schedule[step_idx + 1].item()

        # Build model inputs for current x_t
        merged, _ = _build_model_inputs(
            batch_data, traj_norm, t_current, x_t, model_args, device, N
        )

        # Forward pass → x_0 prediction
        _, outputs = model(merged)
        if "model_output" in outputs:
            x0_pred = outputs["model_output"][:, :, 1:, :]  # [N, P, T, 4]
        else:
            x0_pred = outputs["prediction"]

        # VPSDE denoising step with log-prob (ego only for log-prob)
        t_prev_tensor = torch.tensor(t_prev, device=device).view(1, 1, 1, 1)

        # Sample new x_{t-1} for ALL agents (needed for chain continuity)
        mean_all, std_all = sde.marginal_prob(x0_pred, t_prev_tensor)
        std_all = std_all.clamp(min=min_std)
        noise = torch.randn_like(mean_all)
        x_t_prev_future = mean_all + std_all * noise

        # Compute log-prob for EGO only (agent 0) — this is the policy's action
        ego_sample = x_t_prev_future[:, 0]  # [N, T, 4]
        ego_mean = mean_all[:, 0]  # [N, T, 4]
        # std_all is typically (1,1,1,1) broadcast — squeeze to scalar for clean broadcasting
        ego_std = std_all.view(-1).mean().clamp(min=min_std)  # scalar std from VPSDE

        log_prob = (
            -((ego_sample.detach() - ego_mean) ** 2) / (2 * ego_std ** 2)
            - torch.log(ego_std)
            - 0.5 * math.log(2 * math.pi)
        )
        log_prob = log_prob.reshape(N, -1).mean(dim=-1)  # [N] — mean not sum to normalize by dims

        # Reconstruct full x_{t-1} with current state
        x_t = torch.cat([all_gt[:, :, :1, :], x_t_prev_future], dim=2)

        chain.append(x_t.clone())
        all_log_probs.append(log_prob)

    log_probs = torch.stack(all_log_probs, dim=-1)  # [N, num_steps]
    return {
        "chain": chain,  # list of (num_steps+1) tensors, each [N, P, T+1, 4]
        "log_probs": log_probs,  # [N, num_steps]
        "trajectories_norm": traj_norm,  # [N, T, 4]
        "all_gt": all_gt,  # [N, P, T+1, 4] — clean reference
    }


def compute_logprob_grpo_loss(
    model: torch.nn.Module,
    rollout: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
    data: dict,
    model_args,
    config,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Stage 2: Re-run denoising on stored chain, compute RL + IL loss.

    Mirrors DDV2 get_rlloss (lines 1028-1120): re-run the model on stored chain
    states to get new log-probs, then compute advantage-weighted policy gradient.

    KL regularization uses model.disable_adapter_layers() for reference when
    the model has LoRA adapters and config.kl_coef > 0.

    Args:
        model: Diffusion planner model (train mode).
        rollout: Dict from collect_logprob_rollout with chain, log_probs, etc.
        advantages: [N] advantages (before per-step discount, will be applied here).
        data: Scene observation dict with B=1.
        model_args: Config namespace.
        config: GRPOConfig with logprob settings.
        device: Torch device.

    Returns:
        (loss, metrics_dict) where loss is a scalar tensor and metrics_dict
        contains diagnostic values.
    """
    sde = VPSDE_linear()
    chain = rollout["chain"]
    traj_norm = rollout["trajectories_norm"]
    all_gt = rollout["all_gt"]
    N = traj_norm.shape[0]
    num_steps = config.logprob_num_steps
    min_std = config.logprob_min_std

    # Expand data from B=1 to B=N
    batch_data = _expand_data(data, N)

    # Create timestep schedule (same as collection)
    schedule = create_timestep_schedule(config.logprob_t_start, num_steps)

    # Apply per-step discount to advantages: [N, num_steps]
    discount = compute_discount_weights(num_steps, config.logprob_discount).to(device)
    # advantages: [N] → [N, num_steps]
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    advantages_per_step = advantages_t.unsqueeze(-1) * discount.unsqueeze(0)  # [N, num_steps]

    all_log_probs = []
    il_losses = []

    model.train()
    for step_idx in range(num_steps):
        t_current = schedule[step_idx].item()
        t_prev = schedule[step_idx + 1].item()

        # Load stored x_t from chain (detached — no gradient through chain)
        x_t = chain[step_idx].detach()

        # Build model inputs
        merged, _ = _build_model_inputs(
            batch_data, traj_norm, t_current, x_t, model_args, device, N
        )

        # Forward pass → x_0 prediction (with gradient)
        _, outputs = model(merged)
        if "model_output" in outputs:
            x0_pred = outputs["model_output"][:, :, 1:, :]  # [N, P, T, 4]
        else:
            x0_pred = outputs["prediction"]

        # Compute new log-prob for EGO ONLY using stored x_{t-1}
        stored_x_prev = chain[step_idx + 1][:, :, 1:, :].detach()  # [N, P, T, 4]
        t_prev_tensor = torch.tensor(t_prev, device=device).view(1, 1, 1, 1)

        # Compute mean/std from model's x0_pred
        mean_all, std_all = sde.marginal_prob(x0_pred, t_prev_tensor)
        std_all = std_all.clamp(min=min_std)

        # Ego-only log-prob (agent 0)
        ego_mean = mean_all[:, 0]  # [N, T, 4]
        ego_sample = stored_x_prev[:, 0].detach()  # [N, T, 4]
        ego_std = std_all.view(-1).mean().clamp(min=min_std)  # scalar std from VPSDE

        log_prob = (
            -((ego_sample - ego_mean) ** 2) / (2 * ego_std ** 2)
            - torch.log(ego_std)
            - 0.5 * math.log(2 * math.pi)
        )
        log_prob = log_prob.reshape(N, -1).mean(dim=-1)  # [N] — mean not sum to normalize by dims

        all_log_probs.append(log_prob)

        # IL loss: MSE between model's x_0 prediction and GT (ego only)
        ego_gt = all_gt[:, 0, 1:, :]  # [N, T, 4]
        ego_pred = x0_pred[:, 0]  # [N, T, 4]
        il_loss_step = F.mse_loss(ego_pred, ego_gt, reduction='none').mean(dim=(1, 2))  # [N]
        il_losses.append(il_loss_step)

    # Stack log-probs: [N, num_steps]
    log_probs = torch.stack(all_log_probs, dim=-1)

    # RL loss: -exp(logp - logp.detach()) * advantages (DDV2 line 1096)
    # The exp(logp - logp.detach()) = 1.0 at evaluation, but provides correct
    # gradient ∇logp * advantages during backprop (standard REINFORCE trick).
    per_step_loss = -torch.exp(log_probs - log_probs.detach()) * advantages_per_step

    # Mask zero-advantage entries (DDV2 lines 1099-1101)
    mask_nz = advantages_per_step != 0  # [N, num_steps]
    if mask_nz.any():
        rl_loss_per_sample = (per_step_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp(min=1)
    else:
        rl_loss_per_sample = per_step_loss.mean(dim=1)
    rl_loss = rl_loss_per_sample.mean()

    # IL loss: average over steps and samples
    il_loss = torch.stack(il_losses, dim=-1).mean()

    # Adaptive IL weight (DDV2 lines 1113-1117)
    if config.il_adaptive:
        has_positive = (advantages_t > 0).any()
        il_weight = config.il_loss_weight if has_positive else 1.0
    else:
        il_weight = config.il_loss_weight

    total_loss = rl_loss + il_weight * il_loss

    # KL regularization against reference model (optional)
    # Uses mean-divergence KL: KL(policy || ref) ≈ mean((μ_policy - μ_ref)² / (2σ²))
    # This avoids the chain-based KL magnitude bug where both policy and ref log-probs
    # are huge negative numbers on stored chain samples.
    kl_loss = torch.tensor(0.0, device=device)
    if config.kl_coef > 0 and hasattr(model, 'disable_adapter_layers'):
        ref_ego_means = []
        model.disable_adapter_layers()
        try:
            for step_idx in range(num_steps):
                t_current = schedule[step_idx].item()
                t_prev = schedule[step_idx + 1].item()
                x_t = chain[step_idx].detach()

                # Reference model forward pass (adapters disabled = SFT base)
                merged_ref, _ = _build_model_inputs(
                    batch_data, traj_norm, t_current, x_t, model_args, device, N
                )
                with torch.no_grad():
                    _, ref_outputs = model(merged_ref)
                if "model_output" in ref_outputs:
                    ref_x0 = ref_outputs["model_output"][:, :, 1:, :]
                else:
                    ref_x0 = ref_outputs["prediction"]

                # Compute reference mean for ego
                t_prev_t = torch.tensor(t_prev, device=device).view(1, 1, 1, 1)
                ref_mean, ref_std = sde.marginal_prob(ref_x0, t_prev_t)
                ref_ego_mean = ref_mean[:, 0]  # [N, T, 4]

                ref_ego_mean_detached = ref_ego_mean.detach()
                ref_ego_means.append(ref_ego_mean_detached)
        finally:
            model.enable_adapter_layers()

        # Now compute policy means (with adapters enabled)
        policy_ego_means = []
        for step_idx in range(num_steps):
            t_current = schedule[step_idx].item()
            t_prev = schedule[step_idx + 1].item()
            x_t = chain[step_idx].detach()

            merged_pol, _ = _build_model_inputs(
                batch_data, traj_norm, t_current, x_t, model_args, device, N
            )
            _, pol_outputs = model(merged_pol)
            if "model_output" in pol_outputs:
                pol_x0 = pol_outputs["model_output"][:, :, 1:, :]
            else:
                pol_x0 = pol_outputs["prediction"]

            t_prev_t = torch.tensor(t_prev, device=device).view(1, 1, 1, 1)
            pol_mean, pol_std = sde.marginal_prob(pol_x0, t_prev_t)
            pol_ego_mean = pol_mean[:, 0]  # [N, T, 4]
            pol_std_val = pol_std.squeeze().clamp(min=min_std)

            ref_ego_mean = ref_ego_means[step_idx]

            # KL between N(μ_pol, σ) and N(μ_ref, σ) = (μ_pol - μ_ref)² / (2σ²)
            step_kl = ((pol_ego_mean - ref_ego_mean) ** 2 / (2 * pol_std_val ** 2)).mean()
            policy_ego_means.append(step_kl)

        kl_loss = torch.stack(policy_ego_means).mean()
        total_loss = total_loss + config.kl_coef * kl_loss

    metrics = {
        "rl_loss": rl_loss.item(),
        "il_loss": il_loss.item(),
        "il_weight": il_weight,
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "mean_log_prob": log_probs.mean().item(),
        "std_log_prob": log_probs.std().item(),
        "n_positive_adv": (advantages_t > 0).sum().item(),
        "n_negative_adv": (advantages_t < 0).sum().item(),
    }

    return total_loss, metrics


    # (Old _compute_ref_log_probs helper removed — KL now uses analytical
    # mean-divergence computed inline in compute_logprob_grpo_loss above.)
