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

from preference_optimization.dpo_loss import compute_trajectory_loss

from rlvr.grpo_config import GRPOConfig


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
    """Compute direct regression loss from deterministic output to best trajectory.

    Runs the model in eval mode through DPM-Solver to produce the deterministic
    output, then computes MSE against the best-in-group trajectory (in normalized
    space). This directly optimizes the trajectory that would be deployed.

    The gradient flows through the DPM-Solver sampling process back to the model
    parameters, requiring torch.enable_grad() during inference.

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

    # Convert target to normalized space
    gt_traj = torch.tensor(best_trajectory, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 4]
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    gt_norm = (gt_traj - ego_mean) / ego_std  # [1, T, 4]

    # Clone and normalize observations
    data_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_norm = model_args.observation_normalizer(data_norm)

    # Run model in eval mode with gradients enabled to get deterministic output
    # through DPM-Solver. The gradient flows through the solver steps.
    was_training = policy_model.training
    policy_model.eval()

    # Build zero-noise input (deterministic)
    data_norm["sampled_trajectories"] = torch.zeros(
        B, P, future_len + 1, 4, device=device
    )

    # Temporarily disable guidance for clean deterministic output
    inner = policy_model.module if hasattr(policy_model, "module") else policy_model
    decoder = inner.decoder if hasattr(inner, "decoder") else inner
    orig_guidance_fn = decoder._guidance_fn
    decoder._guidance_fn = None

    with torch.enable_grad():
        _, outputs = policy_model(data_norm)

    decoder._guidance_fn = orig_guidance_fn

    # Extract ego prediction in normalized space
    # outputs["prediction"] is in physical space (inverse-normalized by decoder).
    # We need to re-normalize it for loss computation against gt_norm.
    pred_physical = outputs["prediction"][:, 0]  # [B, T, 4]
    pred_norm = (pred_physical - ego_mean) / ego_std

    # MSE loss between deterministic output and best trajectory
    direct_loss = F.mse_loss(pred_norm, gt_norm, reduction="mean")

    # KL term: run one diffusion loss for policy and ref to get KL estimate
    kl_loss = torch.tensor(0.0, device=device)
    if config.kl_coef > 0:
        noise = torch.randn(B, P, future_len, 4, device=device)
        t = _sample_t_for_mode(config, B, device)
        data_p = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        policy_model.train()
        l_policy = compute_trajectory_loss(
            policy_model, data_p, best_trajectory, model_args, noise, t, device,
        )
        use_lora_disable = hasattr(inner, "disable_adapter")
        disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()
        with disable_ctx, torch.no_grad():
            data_r = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            l_ref = compute_trajectory_loss(
                policy_model, data_r, best_trajectory, model_args, noise.clone(), t, device,
            )
        kl_loss = l_policy - l_ref

    total_loss = config.direct_loss_weight * direct_loss + config.kl_coef * kl_loss

    if was_training:
        policy_model.train()

    metrics = {
        "loss": float(total_loss.item()),
        "policy_loss": float(direct_loss.item()),
        "kl_loss": float(kl_loss.item()) if isinstance(kl_loss, torch.Tensor) else 0.0,
        "mean_advantage": 0.0,
        "advantage_std": 0.0,
        "mean_policy_logprob": 0.0,
        "mean_ref_logprob": 0.0,
        "clip_fraction": 0.0,
        "approx_kl_behavior": 0.0,
        "direct_mse": float(direct_loss.item()),
    }

    return total_loss, metrics


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
