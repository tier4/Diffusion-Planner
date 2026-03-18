"""GRPO loss computation with dual-mode support.

Supports two modes controlled by GRPOConfig.inner_epochs:

1. On-policy (M=1): Single pass, advantage-weighted loss + KL.
   L = (1/G) * sum_i[ A_i * loss_i ] + kl_coef * KL

2. Multi-epoch (M>1): Multiple inner epochs on the same rollout batch.
   Uses PPO-clipped importance sampling to bound policy drift within a batch.
   L = (1/G) * sum_i[ -min(r_i * A_i, clip(r_i) * A_i) ] + kl_coef * KL
   where r_i = exp(old_logprob_i - new_loss_i)  (ratio of behavior to current policy)

Dual-reference strategy:
- Fixed SFT reference (disable_adapter): used for KL penalty.
- Behavior reference (old_log_probs): used for importance sampling ratio.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn

from preference_optimization.dpo_loss import compute_trajectory_loss

from rlvr.grpo_config import GRPOConfig


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
) -> torch.Tensor:
    """Compute log-probabilities (negative diffusion loss) for each trajectory.

    Used to store old_log_probs at rollout time for importance sampling.
    Runs in eval mode, no gradient.

    Returns:
        (N,) tensor of log-probs (negative MSE losses). Higher = more likely.
    """
    N = len(trajectories)
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    eps = 1e-3

    noise = torch.randn(B, P, future_len, 4, device=device)
    t = torch.rand(B, device=device) * (1 - eps) + eps

    was_training = policy_model.training
    policy_model.eval()

    log_probs = []
    with torch.no_grad():
        for i in range(N):
            data_c = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in data.items()}
            loss = compute_trajectory_loss(
                policy_model, data_c, trajectories[i],
                model_args, noise, t, device,
            )
            # log_prob ≈ -loss (MSE loss is proportional to -log pi)
            log_probs.append(-loss)

    if was_training:
        policy_model.train()

    return torch.stack(log_probs)  # (N,)


def compute_grpo_loss(
    policy_model: nn.Module,
    trajectories: list[np.ndarray],
    advantages: np.ndarray,
    data: dict[str, torch.Tensor],
    model_args,
    config: GRPOConfig,
    device: torch.device,
    old_log_probs: torch.Tensor | None = None,
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

    # K=1: shared noise sample across the group
    noise = torch.randn(B, P, future_len, 4, device=device)
    t = torch.rand(B, device=device) * (1 - eps) + eps

    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

    # Compute policy losses (with grad) and SFT reference losses (no grad)
    policy_losses, ref_losses = _compute_losses_and_ref(
        policy_model, trajectories, data, model_args, device, noise, t,
        compute_ref=True,
    )

    policy_loss_stack = torch.stack(policy_losses)  # (N,)
    ref_loss_stack = torch.stack(ref_losses)         # (N,)

    # KL divergence against fixed SFT reference (always computed)
    kl_loss = (policy_loss_stack - ref_loss_stack).mean()

    if old_log_probs is not None and config.uses_importance_sampling:
        # Multi-epoch mode: PPO-clipped importance sampling
        # new_log_probs = -policy_losses (log_prob ≈ -MSE_loss)
        new_log_probs = -policy_loss_stack  # (N,)

        # Importance sampling ratio: pi_new / pi_old
        log_ratio = new_log_probs - old_log_probs.to(device)
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
