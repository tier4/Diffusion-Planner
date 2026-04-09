"""GRPO loss computation with multiple loss mode support.

Supports two outer modes controlled by GRPOConfig.inner_epochs:

1. On-policy (M=1): Single pass, advantage-weighted loss + KL.
   L = (1/G) * sum_i[ A_i * loss_i ] + kl_coef * KL

2. Multi-epoch (M>1): Multiple inner epochs on the same rollout batch.
   Uses PPO-clipped importance sampling to bound policy drift within a batch.
   L = (1/G) * sum_i[ -min(r_i * A_i, clip(r_i) * A_i) ] + kl_coef * KL
   where r_i = exp(old_logprob_i - new_loss_i)  (ratio of behavior to current policy)

Additionally, supports alternative loss modes via GRPOConfig.loss_mode:

- "diffusion" (default): standard advantage-weighted diffusion loss at random t.
- "direct_best": regress the model's deterministic DPM-Solver output toward the
    best-in-group trajectory. Hypothesis: this may more directly affect the
    deterministic output than the standard diffusion loss, which evaluates at a
    random diffusion timestep that may not correspond to the DPM-Solver path.
- "diffusion_low_t": sample t from [t_min, t_max] near 0.
- "diffusion_multistep": average loss over K timesteps spread across the schedule.

Dual-reference strategy:
- Fixed SFT reference (disable_adapter): used for KL penalty.
- Behavior reference (old_log_probs): used for importance sampling ratio.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preference_optimization.dpo_loss import compute_trajectory_loss as _compute_trajectory_loss_raw
from rlvr.grpo_config import GRPOConfig


def compute_trajectory_loss(model, data, trajectory, model_args, noise, t, device):
    """V4-compatible trajectory loss using 4D diffusion timestep.

    Matches SFT training in decoder.py: includes prefix mask with random delay,
    per-timestep t modulation, and clean prefix injection.
    """
    import random as _random

    from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
    from diffusion_planner.model.module.decoder import generate_prefix_mask

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    # Expand t to [B, P, T+1, 1]
    if t.dim() == 1:
        t_4d = t.view(B, 1, 1, 1).expand(B, P, future_len + 1, 1).clone()
    elif t.dim() == 4:
        t_4d = t
    else:
        t_4d = t.view(B, 1, 1, 1).expand(B, P, future_len + 1, 1).clone()

    # Prefix mask with random delay — matches SFT (decoder.py line 95-100)
    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=device)
    prefix_mask = generate_prefix_mask(delay, P, future_len + 1)  # (B, P, T+1, 1)
    mask_coeff = _random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t_4d * mask_coeff, torch.tensor(eps, device=device))
    t_4d = torch.where(prefix_mask, curr_mask_time, t_4d)

    gt_trajectory = torch.as_tensor(trajectory, dtype=torch.float32, device=device)
    if gt_trajectory.dim() == 2:
        gt_trajectory = gt_trajectory.unsqueeze(0)  # [T, 4] → [1, T, 4]
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    gt_trajectory_norm = (gt_trajectory - ego_mean) / ego_std

    gt_future = torch.zeros(B, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = gt_trajectory_norm

    ego_current = data["ego_current_state"][:, :4]
    if P > 1:
        neighbors_current = data["neighbor_agents_past"][:, :P - 1, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    else:
        neighbors_current_norm = torch.zeros(B, 0, 4, device=device)
    ego_current_norm = (ego_current - ego_mean) / ego_std
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)

    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [B, P, T+1, 4]

    # Diffusion noise with prefix masking — matches SFT (decoder.py line 111-116)
    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t_4d[..., 1:, :])
    xT = mean + std * noise
    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)  # [B, P, T+1, 4]
    # Prefix: replace noised steps with clean GT
    xT_full = torch.where(prefix_mask, all_gt, xT_full)

    data_for_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_normalized = model_args.observation_normalizer(data_for_norm)

    merged_inputs = {**data_normalized}
    merged_inputs["gt_trajectories"] = all_gt
    merged_inputs["sampled_trajectories"] = xT_full
    merged_inputs["diffusion_time"] = t_4d  # [B, P, T+1, 1]
    merged_inputs["prefix_mask"] = prefix_mask
    if "delay" not in merged_inputs:
        merged_inputs["delay"] = delay

    _, outputs = model(merged_inputs)

    if "model_output" in outputs:
        model_output = outputs["model_output"][:, 0, 1:, :]  # [B, T, 4]
    else:
        model_output = outputs["prediction"][:, 0]

    gt_target = all_gt[:, 0, 1:, :]  # [B, T, 4]
    loss = F.mse_loss(model_output, gt_target)
    return loss


def _sample_t_for_mode(
    config: GRPOConfig,
    B: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample diffusion timestep(s) based on loss_mode.

    For 'diffusion': uniform in [eps, 1).
    For 'diffusion_low_t': uniform in [t_min, t_max] from config.diffusion_t_range.
    For 'diffusion_multistep': not used (caller handles K samples).
    """
    eps = 1e-3
    if config.loss_mode == "diffusion_low_t":
        t_min, t_max = config.diffusion_t_range
        return torch.rand(B, device=device) * (t_max - t_min) + t_min
    else:
        return torch.rand(B, device=device) * (1 - eps) + eps


def compute_direct_best_loss(
    policy_model: nn.Module,
    best_trajectory: np.ndarray,
    data: dict[str, torch.Tensor],
    model_args,
    device: torch.device,
    config: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute behavioral cloning loss toward the best-in-group trajectory.

    Since the DPM-Solver's sample() method uses torch.no_grad() internally,
    we cannot directly backpropagate through the deterministic generation path.
    Instead, this mode treats the best trajectory as a supervised target and
    trains the denoiser to reconstruct it, averaging over K diffusion timesteps
    sampled near t=0 where the denoising is closest to the final clean output.

    This differs from standard GRPO in two ways:
    1. Only the best trajectory is used (no advantage weighting over the group)
    2. Timesteps are concentrated near t=0 via config.diffusion_t_range

    The KL regularization term uses the same diffusion loss against the SFT
    reference to prevent catastrophic drift.

    Args:
        policy_model: Policy model (LoRA-wrapped).
        best_trajectory: (T, 4) best trajectory from the group [x, y, cos, sin].
        data: Raw observation dict (NOT normalized).
        model_args: Config object from load_model.
        device: Torch device.
        config: GRPOConfig.

    Returns:
        (loss, metrics_dict)
    """
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    inner = policy_model.module if hasattr(policy_model, "module") else policy_model
    use_lora_disable = hasattr(inner, "disable_adapter")

    # Average over K timestep samples for stability
    K = config.diffusion_k_steps
    t_min, t_max = config.diffusion_t_range

    policy_losses = []
    ref_losses = []

    for _ in range(K):
        noise = torch.randn(B, P, future_len, 4, device=device)
        t = torch.rand(B, device=device) * (t_max - t_min) + t_min

        # Policy loss (with grad)
        data_p = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        l_policy = compute_trajectory_loss(
            policy_model, data_p, best_trajectory, model_args, noise, t, device,
        )
        policy_losses.append(l_policy)

        # Reference loss (no grad) for KL
        if config.kl_coef > 0:
            disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()
            with disable_ctx, torch.no_grad():
                data_r = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                l_ref = compute_trajectory_loss(
                    policy_model, data_r, best_trajectory, model_args, noise.clone(), t, device,
                )
            ref_losses.append(l_ref)

    # Average over K samples
    direct_loss = torch.stack(policy_losses).mean()

    kl_loss = torch.tensor(0.0, device=device)
    if ref_losses:
        ref_loss = torch.stack(ref_losses).mean()
        kl_loss = direct_loss - ref_loss

    total_loss = config.direct_loss_weight * direct_loss + config.kl_coef * kl_loss

    metrics = {
        "loss": float(total_loss.item()),
        "policy_loss": float(direct_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "mean_advantage": 0.0,
        "advantage_std": 0.0,
        "mean_policy_logprob": float((-direct_loss).item()),
        "mean_ref_logprob": float((-ref_loss).item()) if ref_losses else 0.0,
        "clip_fraction": 0.0,
        "approx_kl_behavior": 0.0,
        "direct_mse": float(direct_loss.item()),
    }

    return total_loss, metrics


def compute_batched_trajectory_losses(
    model, data, trajectories_tensor, model_args, noise, t, device,
    neighbor_loss_weight: float = 0.0,
):
    """Compute diffusion losses for N trajectories in ONE forward pass.

    Matches the SFT training path in decoder.py: includes prefix mask with
    random delay, per-timestep t modulation, and proper noising.

    Args:
        model: Diffusion planner model (in train mode).
        data: Scene observation dict with B=1 or B=N.
        trajectories_tensor: [N, T, 4] tensor of trajectories.
        model_args: Config with state_normalizer, etc.
        noise: [1, P, T, 4] noise (will be expanded to [N, ...]).
        t: [1] diffusion timestep (will be expanded to [N]).
        device: Torch device.

    Returns:
        [N] tensor of per-trajectory MSE losses.
    """
    import random as _random

    from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
    from diffusion_planner.model.module.decoder import generate_prefix_mask

    N = trajectories_tensor.shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    # Expand data to B=N. Supports B=1 (expand) and B=N (pass through).
    # Only validate/expand tensors whose dim0 matches the scene batch size;
    # non-batched metadata tensors (e.g. ego_shape [3], lane geometry) pass through.
    B_scene = data["ego_current_state"].shape[0]
    batch_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B_scene:
            if B_scene == 1:
                batch_data[k] = v.expand(N, *v.shape[1:]).contiguous()
            elif B_scene == N:
                batch_data[k] = v
            else:
                raise ValueError(
                    f"data['{k}'] has B={B_scene}, expected 1 or N={N}"
                )
        else:
            batch_data[k] = v

    # Expand t to [N, P, T+1, 1] — matches SFT (decoder.py line 90-92)
    if t.dim() == 1:
        t_N = t.expand(N)
        t_4d = t_N.view(N, 1, 1, 1).expand(N, P, future_len + 1, 1).clone()
    else:
        t_4d = t.expand(N, *t.shape[1:]).contiguous() if t.shape[0] == 1 else t

    # Expand noise to [N, P, T, 4]
    if noise.shape[0] == 1:
        noise_N = noise.expand(N, -1, -1, -1).contiguous()
    else:
        noise_N = noise

    # Prefix mask with random delay — matches SFT (decoder.py line 95-100)
    # Forces first `delay` steps to use clean GT, training the model to
    # predict the trajectory conditioned on a clean prefix.
    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (N,), device=device)
    prefix_mask = generate_prefix_mask(delay, P, future_len + 1)  # (N, P, T+1, 1)
    mask_coeff = _random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t_4d * mask_coeff, torch.tensor(eps, device=device))
    t_4d = torch.where(prefix_mask, curr_mask_time, t_4d)

    # Normalize trajectories: [N, T, 4]
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    gt_traj_norm = (trajectories_tensor - ego_mean) / ego_std

    # Build gt_future: [N, P, T, 4] with ego + neighbor GT
    # Ego uses the GRPO sampled trajectory; neighbors use their actual GT futures.
    # This matches SFT (decoder.py line 84-86) and provides neighbor regularization.
    gt_future = torch.zeros(N, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = gt_traj_norm

    # Fill neighbor GT from data (matches SFT)
    Pn = P - 1
    neighbor_future_valid = None
    nf_pn = 0
    if Pn > 0 and "neighbor_agents_future" in batch_data:
        nf = batch_data["neighbor_agents_future"]  # [N, Pn_data, T, 3] — x, y, heading_rad
        nf_pn = min(nf.shape[1], Pn)
        nf_4d = torch.zeros(N, nf_pn, future_len, 4, device=device)
        nf_4d[..., :2] = nf[:, :nf_pn, :future_len, :2]  # x, y
        if nf.shape[-1] >= 3:
            heading = nf[:, :nf_pn, :future_len, 2]  # heading_rad
            nf_4d[..., 2] = torch.cos(heading)  # cos_yaw
            nf_4d[..., 3] = torch.sin(heading)  # sin_yaw
        nf_4d_norm = (nf_4d - ego_mean) / ego_std
        gt_future[:, 1:1 + nf_pn, :, :] = nf_4d_norm
        # Track validity for neighbor loss
        if nf.shape[-1] >= 3:
            neighbor_future_valid = (nf[:, :nf_pn, :future_len, :2].abs().sum(dim=-1) > 0.1)  # [N, Pn', T]

    # Current states — normalized (matches SFT decoder.py line 60-67)
    ego_current = batch_data["ego_current_state"][:, :4]  # [N, 4]
    if P > 1:
        neighbors_current = batch_data["neighbor_agents_past"][:, :P - 1, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    else:
        neighbors_current_norm = torch.zeros(N, 0, 4, device=device)
    ego_current_norm = (ego_current - ego_mean) / ego_std
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)

    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [N, P, T+1, 4]

    # Zero out invalid neighbor entries in all_gt (matches SFT decoder.py line 108)
    if Pn > 0:
        neighbor_current_mask_final = (batch_data["neighbor_agents_past"][:, :Pn, -1, :4].abs().sum(dim=-1) == 0)  # [N, Pn]
        if "neighbor_agents_future" in batch_data:
            nf = batch_data["neighbor_agents_future"]
            nf_pn = min(nf.shape[1], Pn)
            nf_valid = (nf[:, :nf_pn, :future_len, :2].abs().sum(dim=-1) > 0.1)
            nf_mask = ~nf_valid  # [N, Pn', T]
            full_neighbor_mask = torch.cat([neighbor_current_mask_final[:, :nf_pn].unsqueeze(-1), nf_mask], dim=-1)  # [N, Pn', T+1]
            # Use named slice + masked_fill_ to avoid chained indexing issues
            neighbor_slice = all_gt[:, 1:1 + nf_pn]  # view into all_gt
            neighbor_slice.masked_fill_(full_neighbor_mask.unsqueeze(-1).expand_as(neighbor_slice), 0.0)

    # Diffusion noise with prefix masking — matches SFT (decoder.py line 111-116)
    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t_4d[..., 1:, :])
    xT = mean + std * noise_N
    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    # Prefix: replace noised steps with clean GT for delay steps
    xT_full = torch.where(prefix_mask, all_gt, xT_full)

    # Normalize observation data
    data_for_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
    data_normalized = model_args.observation_normalizer(data_for_norm)

    merged = {**data_normalized}
    merged["gt_trajectories"] = all_gt
    merged["sampled_trajectories"] = xT_full
    merged["diffusion_time"] = t_4d
    merged["prefix_mask"] = prefix_mask
    if "delay" not in merged:
        merged["delay"] = delay

    _, outputs = model(merged)

    if "model_output" in outputs:
        full_output = outputs["model_output"][:, :, 1:, :]  # [N, P, T, 4]
    else:
        full_output = outputs["prediction"]  # [N, P, T, 4]

    full_gt = all_gt[:, :, 1:, :]  # [N, P, T, 4]

    # Ego loss: [N]
    ego_output = full_output[:, 0]  # [N, T, 4]
    ego_gt = full_gt[:, 0]  # [N, T, 4]
    per_traj_ego_loss = F.mse_loss(ego_output, ego_gt, reduction='none').mean(dim=(1, 2))

    # Neighbor regularization loss: per-trajectory MSE on valid neighbor predictions.
    # This prevents the LoRA from distorting neighbor predictions, which feeds back
    # into the joint denoising and corrupts ego output over time.
    # Disabled by default (neighbor_loss_weight=0). Set >0 to enable.
    _NEIGHBOR_LOSS_WEIGHT = neighbor_loss_weight
    if _NEIGHBOR_LOSS_WEIGHT > 0 and P > 1 and neighbor_future_valid is not None:
        neighbor_output = full_output[:, 1:1 + nf_pn]  # [N, Pn', T, 4]
        neighbor_gt = full_gt[:, 1:1 + nf_pn]  # [N, Pn', T, 4]
        neighbor_mse = F.mse_loss(neighbor_output, neighbor_gt, reduction='none')  # [N, Pn', T, 4]
        # Mask invalid neighbors to zero
        valid_mask = neighbor_future_valid.unsqueeze(-1).expand_as(neighbor_mse)  # [N, Pn', T, 4]
        neighbor_mse = neighbor_mse * valid_mask.float()
        # Per-trajectory neighbor loss: [N] — average over valid neighbors, timesteps, dims
        n_valid_per_traj = valid_mask.float().sum(dim=(1, 2, 3)).clamp(min=1)
        per_traj_neighbor_loss = neighbor_mse.sum(dim=(1, 2, 3)) / n_valid_per_traj
        per_traj_loss = per_traj_ego_loss + _NEIGHBOR_LOSS_WEIGHT * per_traj_neighbor_loss
    else:
        per_traj_loss = per_traj_ego_loss

    return per_traj_loss


def _compute_losses_and_ref(
    policy_model: nn.Module,
    trajectories: list[np.ndarray],
    data: dict[str, torch.Tensor],
    model_args,
    device: torch.device,
    noise: torch.Tensor,
    t: torch.Tensor,
    compute_ref: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute policy losses (with grad) and reference losses (no grad) for all trajectories.

    Args:
        compute_ref: If True, also compute reference (SFT base) losses for KL.

    Returns:
        (policy_losses, ref_losses) — each a list of N scalar tensors.
        If compute_ref is False, ref_losses is empty.
    """
    inner = policy_model.module if hasattr(policy_model, "module") else policy_model
    use_lora_disable = hasattr(inner, "disable_adapter")

    policy_losses = []
    ref_losses = []

    for i in range(len(trajectories)):
        data_p = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in data.items()}
        l_policy = compute_trajectory_loss(
            policy_model, data_p, trajectories[i],
            model_args, noise, t, device,
        )
        policy_losses.append(l_policy)

        if compute_ref:
            disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()
            with disable_ctx, torch.no_grad():
                data_r = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in data.items()}
                l_ref = compute_trajectory_loss(
                    policy_model, data_r, trajectories[i],
                    model_args, noise.clone(), t, device,
                )
            ref_losses.append(l_ref)

    return policy_losses, ref_losses


def _compute_batched_losses_and_ref(
    policy_model: nn.Module,
    trajectories_tensor: torch.Tensor,
    data: dict[str, torch.Tensor],
    model_args,
    device: torch.device,
    noise: torch.Tensor,
    t: torch.Tensor,
    compute_ref: bool = True,
    neighbor_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched version: compute all N trajectory losses in ONE forward pass.

    Returns:
        (policy_losses [N], ref_losses [N])
    """
    inner = policy_model.module if hasattr(policy_model, "module") else policy_model
    use_lora_disable = hasattr(inner, "disable_adapter")

    # Policy losses: one batched forward pass for all N trajectories
    policy_losses = compute_batched_trajectory_losses(
        policy_model, data, trajectories_tensor, model_args, noise, t, device,
        neighbor_loss_weight=neighbor_loss_weight,
    )  # [N]

    ref_losses = torch.zeros_like(policy_losses)
    if compute_ref:
        disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()
        with disable_ctx, torch.no_grad():
            ref_losses = compute_batched_trajectory_losses(
                policy_model, data, trajectories_tensor, model_args,
                noise.clone(), t, device,
                neighbor_loss_weight=neighbor_loss_weight,
            )  # [N]

    return policy_losses, ref_losses


def _compute_neighbor_reg_loss(
    policy_model, data, model_args, device, K, P, future_len,
):
    """Compute MSE between LoRA and base model neighbor outputs.

    Runs K forward passes with random (noise, t), compares neighbor predictions
    from the LoRA model vs the base model (LoRA disabled). Returns a scalar loss
    with gradients flowing only through the LoRA model.

    When B>1 (batched trainers expand per-scene data), uses only the first
    element since all B entries come from the same scene.
    """
    import random as _random

    from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
    from diffusion_planner.model.module.decoder import generate_prefix_mask

    B = data["ego_current_state"].shape[0]
    if B > 1:
        # All B entries must be the same scene (batched trainers expand per-scene
        # data to B=keep_per). Bail out if different scenes are mixed in.
        ego = data["ego_current_state"]
        if not torch.allclose(ego[:1].expand_as(ego), ego):
            raise ValueError(
                f"_compute_neighbor_reg_loss: B={B} with mixed scenes. "
                "Neighbor reg requires single-scene batches."
            )
        data = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    inner = policy_model.module if hasattr(policy_model, "module") else policy_model
    if not hasattr(inner, "disable_adapter"):
        return torch.tensor(0.0, device=device)

    eps = 1e-3
    Pn = P - 1

    # Build normalized scene data (B=1)
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    ego_current = data["ego_current_state"][:, :4]
    ego_current_norm = (ego_current - ego_mean) / ego_std

    if Pn > 0:
        neighbors_current = data["neighbor_agents_past"][:, :Pn, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    else:
        neighbors_current_norm = torch.zeros(1, 0, 4, device=device)

    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)

    # Build neighbor GT for the all_gt tensor
    gt_future = torch.zeros(1, P, future_len, 4, device=device)
    nf_pn = 0
    neighbor_future_valid = None
    if Pn > 0 and "neighbor_agents_future" in data:
        nf = data["neighbor_agents_future"]
        nf_pn = min(nf.shape[1], Pn)
        nf_4d = torch.zeros(1, nf_pn, future_len, 4, device=device)
        nf_4d[..., :2] = nf[:, :nf_pn, :future_len, :2]
        if nf.shape[-1] >= 3:
            heading = nf[:, :nf_pn, :future_len, 2]
            nf_4d[..., 2] = torch.cos(heading)
            nf_4d[..., 3] = torch.sin(heading)
        nf_4d_norm = (nf_4d - ego_mean) / ego_std
        gt_future[:, 1:1 + nf_pn, :, :] = nf_4d_norm
        neighbor_future_valid = (nf[:, :nf_pn, :future_len, :2].abs().sum(dim=-1) > 0.1)

    # Also need ego GT for the noise target
    if "ego_agent_future" in data:
        ego_gt = data["ego_agent_future"]
        if ego_gt.dim() == 3:
            ego_gt = ego_gt[:, :future_len, :4]
        ego_gt_norm = (ego_gt - ego_mean) / ego_std
        gt_future[:, 0, :ego_gt_norm.shape[1], :] = ego_gt_norm

    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)

    # Normalize observation data
    data_normalized = model_args.observation_normalizer(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    )

    if neighbor_future_valid is None or not neighbor_future_valid.any():
        return torch.tensor(0.0, device=device)

    # Also exclude neighbors absent at current timestep
    neighbor_current_mask = (data["neighbor_agents_past"][:, :Pn, -1, :4].abs().sum(dim=-1) == 0)
    neighbor_future_valid = neighbor_future_valid & (~neighbor_current_mask[:, :nf_pn].unsqueeze(-1))

    total_reg = torch.tensor(0.0, device=device)
    for _ in range(K):
        t = torch.rand(1, device=device) * (1 - eps) + eps
        t_4d = t.view(1, 1, 1, 1).expand(1, P, future_len + 1, 1).clone()

        max_delay = 5
        delay = torch.randint(0, max_delay + 1, (1,), device=device)
        prefix_mask = generate_prefix_mask(delay, P, future_len + 1)
        mask_coeff = _random.uniform(0.0, 1.0)
        curr_mask_time = torch.maximum(t_4d * mask_coeff, torch.tensor(eps, device=device))
        t_4d = torch.where(prefix_mask, curr_mask_time, t_4d)

        z = torch.randn(1, P, future_len, 4, device=device)
        mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t_4d[..., 1:, :])
        xT = mean + std * z
        xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        xT_full = torch.where(prefix_mask, all_gt, xT_full)

        merged = {**data_normalized}
        merged["gt_trajectories"] = all_gt
        merged["sampled_trajectories"] = xT_full
        merged["diffusion_time"] = t_4d
        merged["prefix_mask"] = prefix_mask
        if "delay" not in merged:
            merged["delay"] = delay

        # LoRA forward (with grad)
        _, lora_out = policy_model(merged)
        lora_neighbor = lora_out["model_output"][:, 1:1 + nf_pn, 1:, :]  # [1, Pn', T, 4]

        # Base forward (no grad)
        with inner.disable_adapter(), torch.no_grad():
            _, base_out = policy_model(merged)
        base_neighbor = base_out["model_output"][:, 1:1 + nf_pn, 1:, :]

        reg_mse = ((lora_neighbor - base_neighbor.detach()) ** 2).mean(dim=-1)  # [1, Pn', T]
        masked_reg = reg_mse[:, :nf_pn][neighbor_future_valid[:, :nf_pn]]
        if masked_reg.numel() > 0:
            total_reg = total_reg + masked_reg.mean()

    return total_reg / K


def compute_batched_grpo_loss(
    policy_model: nn.Module,
    trajectories_tensor: torch.Tensor,
    advantages: np.ndarray,
    data: dict[str, torch.Tensor],
    model_args,
    config: GRPOConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched GRPO loss: all N trajectories processed in ONE forward pass.

    Drop-in replacement for compute_grpo_loss for on-policy mode (inner_epochs=1).
    Processes N trajectories simultaneously instead of looping.

    Args:
        trajectories_tensor: [N, T, 4] tensor (not list of numpy arrays).
        Other args same as compute_grpo_loss.

    Returns:
        (loss, metrics_dict)
    """
    N = trajectories_tensor.shape[0]
    if N == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, _empty_metrics()

    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Average over K (noise, t) samples for stable gradients — matches DPO which
    # uses K=8. A single sample is dominated by noise at that specific timestep
    # and doesn't reliably push the deterministic (t=0) trajectory.
    K = max(config.diffusion_k_steps, 1)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

    policy_losses_sum = torch.zeros(N, device=device)
    ref_losses_sum = torch.zeros(N, device=device)

    for _ in range(K):
        noise = torch.randn(1, P, future_len, 4, device=device)
        t = _sample_t_for_mode(config, 1, device)

        policy_losses_k, ref_losses_k = _compute_batched_losses_and_ref(
            policy_model, trajectories_tensor, data, model_args, device, noise, t,
            compute_ref=True,
            neighbor_loss_weight=getattr(config, 'neighbor_loss_weight', 0.0),
        )
        policy_losses_sum = policy_losses_sum + policy_losses_k
        ref_losses_sum = ref_losses_sum + ref_losses_k

    policy_losses = policy_losses_sum / K
    ref_losses = ref_losses_sum / K

    kl_loss = (policy_losses - ref_losses).mean()

    if torch.isnan(policy_losses).any() or torch.isnan(ref_losses).any():
        raise RuntimeError("NaN in batched GRPO loss computation")

    # On-policy mode: advantage-weighted loss + KL
    weighted_loss = (advantages_t * policy_losses).sum() / max(N, 1)
    total_loss = weighted_loss + config.kl_coef * kl_loss

    # Neighbor regularization: MSE(lora_neighbor, base_neighbor) at same inputs.
    # Prevents LoRA from distorting neighbor predictions through shared attention.
    neighbor_reg_w = getattr(config, 'neighbor_reg_weight', 0.0)
    neighbor_reg_loss_val = 0.0
    if neighbor_reg_w > 0 and P > 1:
        neighbor_reg_loss_val = _compute_neighbor_reg_loss(
            policy_model, data, model_args, device, K, P, future_len,
        )
        total_loss = total_loss + neighbor_reg_w * neighbor_reg_loss_val

    metrics = {
        "loss": total_loss.item(),
        "policy_loss_mean": policy_losses.mean().item(),
        "ref_loss_mean": ref_losses.mean().item(),
        "kl": kl_loss.item(),
        "weighted_loss": weighted_loss.item(),
        "neighbor_reg_loss": neighbor_reg_loss_val.item() if isinstance(neighbor_reg_loss_val, torch.Tensor) else 0.0,
    }

    return total_loss, metrics


def compute_log_probs(
    policy_model: nn.Module,
    trajectories: list[np.ndarray],
    data: dict[str, torch.Tensor],
    model_args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute log-probabilities (negative diffusion loss) for each trajectory.

    Used to store old_log_probs at rollout time for importance sampling.
    Also returns the (noise, t) used so they can be reused during training
    for a consistent importance sampling ratio.

    Returns:
        (old_log_probs, noise, t):
        - old_log_probs: (N,) tensor of log-probs (negative MSE losses).
        - noise: (B, P, T, 4) shared noise sample.
        - t: (B,) diffusion timestep.
    """
    N = len(trajectories)
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    noise = torch.randn(B, P, future_len, 4, device=device)
    t = torch.rand(B, device=device) * (1 - eps) + eps

    # Compute in train mode to match the mode used during GRPO loss computation.
    # This ensures the IS ratio starts at exactly 1.0 on the first inner epoch
    # (no bias from dropout differences between eval and train mode).
    was_training = policy_model.training
    policy_model.train()

    log_probs = []
    with torch.no_grad():
        for i in range(N):
            data_c = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in data.items()}
            loss = compute_trajectory_loss(
                policy_model, data_c, trajectories[i],
                model_args, noise, t, device,
            )
            log_probs.append(-loss)

    if not was_training:
        policy_model.eval()

    return torch.stack(log_probs), noise, t  # (N,), (B,P,T,4), (B,)


def compute_grpo_loss(
    policy_model: nn.Module,
    trajectories: list[np.ndarray],
    advantages: np.ndarray,
    data: dict[str, torch.Tensor],
    model_args,
    config: GRPOConfig,
    device: torch.device,
    old_log_probs: torch.Tensor | None = None,
    old_noise: torch.Tensor | None = None,
    old_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss supporting both on-policy and multi-epoch modes.

    On-policy mode (inner_epochs=1, old_log_probs=None):
        L = (1/G) * sum_i[ A_i * loss_i ] + kl_coef * mean(loss_i - loss_ref_i)

    Multi-epoch mode (inner_epochs>1, old_log_probs provided):
        ratio_i = exp(new_logp_i - old_logp_i)
            Since log_prob ≈ -loss: ratio = exp(-new_loss + old_loss)
        clipped_ratio = clip(ratio, 1 - eps, 1 + eps)
        L = -(1/G) * sum_i[ min(ratio * A_i, clipped_ratio * A_i) ]
             + kl_coef * mean(loss_i - loss_ref_i)

    Args:
        policy_model: Policy model (LoRA-wrapped for adapter toggling).
        trajectories: N trajectories, each (T, 4) [x, y, cos, sin].
        advantages: (N,) group-relative advantages.
        data: Raw observation dict (NOT normalized).
        model_args: Config object from load_model.
        config: GRPOConfig with kl_coef, ppo_clip_epsilon, etc.
        device: Torch device.
        old_log_probs: (N,) tensor of log-probs from rollout time.
            Required for multi-epoch mode (inner_epochs > 1).
            None for on-policy mode (inner_epochs = 1).

    Returns:
        (loss, metrics_dict)
    """
    N = len(trajectories)
    if N == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, _empty_metrics()

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    # Reuse stored (noise, t) from rollout time for consistent IS ratio.
    # For on-policy mode or first inner epoch, generate fresh samples.
    if old_noise is not None and old_t is not None:
        noise = old_noise.to(device)
        t = old_t.to(device)
    else:
        noise = torch.randn(B, P, future_len, 4, device=device)
        t = _sample_t_for_mode(config, B, device)

    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

    # For diffusion_multistep: average loss over K different timesteps
    if config.loss_mode == "diffusion_multistep":
        K = config.diffusion_k_steps
        all_policy = []
        all_ref = []
        for k_idx in range(K):
            t_k = _sample_t_for_mode(config, B, device)
            noise_k = torch.randn(B, P, future_len, 4, device=device)
            p_losses, r_losses = _compute_losses_and_ref(
                policy_model, trajectories, data, model_args, device, noise_k, t_k,
                compute_ref=True,
            )
            all_policy.append(torch.stack(p_losses))
            all_ref.append(torch.stack(r_losses))
        policy_loss_stack = torch.stack(all_policy).mean(dim=0)  # (N,)
        ref_loss_stack = torch.stack(all_ref).mean(dim=0)        # (N,)
    else:
        # Standard single-sample: diffusion or diffusion_low_t
        policy_losses, ref_losses = _compute_losses_and_ref(
            policy_model, trajectories, data, model_args, device, noise, t,
            compute_ref=True,
        )
        policy_loss_stack = torch.stack(policy_losses)  # (N,)
        ref_loss_stack = torch.stack(ref_losses)         # (N,)

    # KL divergence against fixed SFT reference (always computed)
    kl_loss = (policy_loss_stack - ref_loss_stack).mean()

    # Hard check: NaN losses indicate a bug, not a recoverable condition
    if torch.isnan(policy_loss_stack).any() or torch.isnan(ref_loss_stack).any():
        nan_policy = int(torch.isnan(policy_loss_stack).sum().item())
        nan_ref = int(torch.isnan(ref_loss_stack).sum().item())
        raise RuntimeError(
            f"NaN in GRPO loss computation: {nan_policy}/{N} policy losses, "
            f"{nan_ref}/{N} ref losses are NaN. Model weights may be corrupted."
        )

    if old_log_probs is not None and config.uses_importance_sampling:
        # Multi-epoch mode: PPO-clipped importance sampling
        # new_log_probs = -policy_losses (log_prob ≈ -MSE_loss)
        new_log_probs = -policy_loss_stack  # (N,)

        # Importance sampling ratio: pi_new / pi_old
        # Clamp log_ratio to prevent exp() overflow → inf → NaN
        log_ratio = new_log_probs - old_log_probs.to(device)
        # Numerical safety: clamp to prevent exp() overflow. With consistent
        # (noise, t) the ratio starts at 1.0 and drifts slowly; |log_ratio|>10
        # means exp()>22000 which is far beyond PPO clip range and indicates
        # something unexpected.
        if (log_ratio.abs() > 10.0).any():
            max_lr = float(log_ratio.abs().max().item())
            print(f"  [grpo_loss] WARNING: log_ratio magnitude {max_lr:.1f} exceeds 10, clamping")
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)  # (N,)

        # PPO clipping
        eps_clip = config.ppo_clip_epsilon
        clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)

        # PPO surrogate objective (maximize → negate for minimization)
        surr1 = ratio * advantages_t
        surr2 = clipped_ratio * advantages_t
        # Take the pessimistic bound: min for positive advantages, max for negative
        policy_loss = -torch.min(surr1, surr2).mean()

        # Fraction of ratios that were clipped (diagnostic)
        clip_frac = float(((ratio - 1.0).abs() > eps_clip).float().mean().item())
        approx_kl_behavior = float((log_ratio ** 2).mean().item() * 0.5)
    else:
        # On-policy mode: direct advantage-weighted loss
        policy_loss = (advantages_t * policy_loss_stack).mean()
        clip_frac = 0.0
        approx_kl_behavior = 0.0

    total_loss = policy_loss + config.kl_coef * kl_loss

    if torch.isnan(total_loss):
        raise RuntimeError(
            f"NaN total loss: policy_loss={policy_loss.item()}, "
            f"kl_loss={kl_loss.item()}, kl_coef={config.kl_coef}. "
            f"Model weights may be corrupted."
        )

    metrics = {
        "loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "mean_advantage": float(advantages_t.mean().item()),
        "advantage_std": float(advantages_t.std().item()),
        "mean_policy_logprob": float((-policy_loss_stack).mean().item()),
        "mean_ref_logprob": float((-ref_loss_stack).mean().item()),
        "clip_fraction": clip_frac,
        "approx_kl_behavior": approx_kl_behavior,
    }

    return total_loss, metrics


def _empty_metrics() -> dict[str, float]:
    return {
        "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0,
        "mean_advantage": 0.0, "advantage_std": 0.0,
        "mean_policy_logprob": 0.0, "mean_ref_logprob": 0.0,
        "clip_fraction": 0.0, "approx_kl_behavior": 0.0,
    }
