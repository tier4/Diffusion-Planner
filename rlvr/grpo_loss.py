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

    The v4 DiT requires t as [B, P, T+1, 1] with t.shape[2] == x.shape[2].
    The original dpo_loss.compute_trajectory_loss passes a single t to both
    marginal_prob (which needs [B,P,T,1]) and diffusion_time (which needs
    [B,P,T+1,1]). To resolve this, we replicate the essential logic here
    with proper 4D t handling, matching the v4 training code in decoder.py.
    """
    from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Expand t to [B, P, T+1, 1]
    if t.dim() == 1:
        t_4d = t.view(B, 1, 1, 1).expand(B, P, future_len + 1, 1).clone()
    elif t.dim() == 4:
        t_4d = t
    else:
        t_4d = t.view(B, 1, 1, 1).expand(B, P, future_len + 1, 1).clone()

    gt_trajectory = torch.tensor(trajectory, dtype=torch.float32, device=device).unsqueeze(0)
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

    # marginal_prob on future part with sliced t (matching decoder.py:111)
    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t_4d[..., 1:, :])
    xT = mean + std * noise

    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)  # [B, P, T+1, 4]

    data_for_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_normalized = model_args.observation_normalizer(data_for_norm)

    merged_inputs = {**data_normalized}
    merged_inputs["gt_trajectories"] = all_gt
    merged_inputs["sampled_trajectories"] = xT_full
    merged_inputs["diffusion_time"] = t_4d  # [B, P, T+1, 1]
    if "delay" not in merged_inputs:
        merged_inputs["delay"] = torch.zeros(B, dtype=torch.long, device=device)

    _, outputs = model(merged_inputs)

    if "model_output" in outputs:
        # model_output is [B, P, T+1, 4] from _forward_training; skip prefix at index 0
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
):
    """Compute diffusion losses for N trajectories in ONE forward pass.

    Instead of looping N times with B=1, expands scene data to B=N
    and processes all trajectories at once.

    Args:
        model: Diffusion planner model (in train mode).
        data: Scene observation dict with B=1.
        trajectories_tensor: [N, T, 4] tensor of trajectories.
        model_args: Config with state_normalizer, etc.
        noise: [1, P, T, 4] noise (will be expanded to [N, ...]).
        t: [1] diffusion timestep (will be expanded to [N]).
        device: Torch device.

    Returns:
        [N] tensor of per-trajectory MSE losses.
    """
    from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear

    N = trajectories_tensor.shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Expand data from B=1 to B=N
    batch_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == 1:
            batch_data[k] = v.expand(N, *v.shape[1:]).contiguous()
        else:
            batch_data[k] = v

    # Expand t to [N, P, T+1, 1]
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

    # Normalize trajectories: [N, T, 4]
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    gt_traj_norm = (trajectories_tensor - ego_mean) / ego_std

    # Build gt_future: [N, P, T, 4] with ego trajectory only
    gt_future = torch.zeros(N, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = gt_traj_norm

    # Current states
    ego_current = batch_data["ego_current_state"][:, :4]  # [N, 4]
    if P > 1:
        neighbors_current = batch_data["neighbor_agents_past"][:, :P - 1, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    else:
        neighbors_current_norm = torch.zeros(N, 0, 4, device=device)
    ego_current_norm = (ego_current - ego_mean) / ego_std
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)

    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [N, P, T+1, 4]

    # Diffusion noise
    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t_4d[..., 1:, :])
    xT = mean + std * noise_N
    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

    # Normalize observation data
    data_for_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
    data_normalized = model_args.observation_normalizer(data_for_norm)

    merged = {**data_normalized}
    merged["gt_trajectories"] = all_gt
    merged["sampled_trajectories"] = xT_full
    merged["diffusion_time"] = t_4d
    if "delay" not in merged:
        merged["delay"] = torch.zeros(N, dtype=torch.long, device=device)

    _, outputs = model(merged)

    if "model_output" in outputs:
        model_output = outputs["model_output"][:, 0, 1:, :]  # [N, T, 4]
    else:
        model_output = outputs["prediction"][:, 0]

    gt_target = all_gt[:, 0, 1:, :]  # [N, T, 4]

    # Per-trajectory MSE loss: [N]
    per_traj_loss = F.mse_loss(model_output, gt_target, reduction='none').mean(dim=(1, 2))
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
    )  # [N]

    ref_losses = torch.zeros_like(policy_losses)
    if compute_ref:
        disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()
        with disable_ctx, torch.no_grad():
            ref_losses = compute_batched_trajectory_losses(
                policy_model, data, trajectories_tensor, model_args,
                noise.clone(), t, device,
            )  # [N]

    return policy_losses, ref_losses


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

    B = data["ego_current_state"].shape[0]  # should be 1
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    noise = torch.randn(1, P, future_len, 4, device=device)
    t = torch.rand(1, device=device) * (1 - eps) + eps

    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

    policy_losses, ref_losses = _compute_batched_losses_and_ref(
        policy_model, trajectories_tensor, data, model_args, device, noise, t,
        compute_ref=True,
    )

    kl_loss = (policy_losses - ref_losses).mean()

    if torch.isnan(policy_losses).any() or torch.isnan(ref_losses).any():
        raise RuntimeError("NaN in batched GRPO loss computation")

    # On-policy mode: advantage-weighted loss + KL
    weighted_loss = (advantages_t * policy_losses).sum() / max(N, 1)
    total_loss = weighted_loss + config.kl_coef * kl_loss

    metrics = {
        "loss": total_loss.item(),
        "policy_loss_mean": policy_losses.mean().item(),
        "ref_loss_mean": ref_losses.mean().item(),
        "kl": kl_loss.item(),
        "weighted_loss": weighted_loss.item(),
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
