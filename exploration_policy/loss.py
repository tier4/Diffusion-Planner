"""Loss functions for the Exploration Policy (joint GRPO training).

The policy outputs a Beta distribution, K η values are sampled from it,
each generates a trajectory scored by the reward function. GRPO-style
group-relative advantages are used to train the policy via REINFORCE.

Loss = policy_gradient + c_e * entropy_loss + c_kl * kl_loss

Where:
- policy_gradient: -mean(A_k * log_prob_k) — REINFORCE with GRPO advantages
- entropy_loss: -mean(H(lat_dist) + H(lon_dist)) — encourage exploration
- kl_loss: KL(current || init) — anchor to initial uniform-ish policy
"""

from __future__ import annotations

import math

import torch
from torch.distributions import Beta, kl_divergence


# Initial Beta params from zero-initialized GuidanceHead: softplus(0) + 1
_INIT_PARAM = math.log(2) + 1.0
_INIT_DIST_LAT = None
_INIT_DIST_LON = None


def _get_init_distributions(device: torch.device) -> tuple[Beta, Beta]:
    """Get the initial (zero-init) Beta distributions for KL computation."""
    global _INIT_DIST_LAT, _INIT_DIST_LON
    if _INIT_DIST_LAT is None or _INIT_DIST_LAT.concentration1.device != device:
        param = torch.tensor(_INIT_PARAM, device=device)
        _INIT_DIST_LAT = Beta(param, param)
        _INIT_DIST_LON = Beta(param, param)
    return _INIT_DIST_LAT, _INIT_DIST_LON


def compute_exploration_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    lat_dist: Beta,
    lon_dist: Beta,
    entropy_coef: float = 0.05,
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the exploration policy loss for one scene.

    Args:
        advantages: [K] GRPO group-relative advantages for each η sample.
        log_probs: [K] log π(η_k) = log_prob_lat_k + log_prob_lon_k.
        lat_dist: Beta distribution for η_lat (batch_shape=[] or (B,)).
        lon_dist: Beta distribution for η_lon (batch_shape=[] or (B,)).
        entropy_coef: Weight for entropy bonus (higher = more exploration).
        kl_coef: Weight for KL penalty against initial policy.

    Returns:
        (loss, metrics_dict)
    """
    device = advantages.device

    # --- REINFORCE policy gradient ---
    # Negative because we maximize reward (minimize negative advantage-weighted log_prob)
    policy_loss = -(advantages * log_probs).mean()

    # --- Entropy bonus ---
    # Maximize entropy to prevent η collapse. Beta entropy is well-defined.
    # Reduce to scalar with .mean() since distributions may have batch dims.
    entropy_value = (lat_dist.entropy() + lon_dist.entropy()).mean()
    entropy_loss = -entropy_value

    # --- KL divergence against initial policy ---
    # Prevents the policy from straying too far from the zero-mean init
    init_lat, init_lon = _get_init_distributions(device)
    kl_value = (kl_divergence(lat_dist, init_lat) + kl_divergence(lon_dist, init_lon)).mean()
    kl_loss = kl_value

    # --- Total loss (guaranteed scalar) ---
    total_loss = policy_loss + entropy_coef * entropy_loss + kl_coef * kl_loss

    metrics = {
        "exploration_policy_loss": policy_loss.item(),
        "exploration_entropy": entropy_value.item(),
        "exploration_kl": kl_value.item(),
        "exploration_total_loss": total_loss.item(),
        "exploration_eta_lat_mean": lat_dist.mean.mean().item() * 2 - 1,
        "exploration_eta_lon_mean": lon_dist.mean.mean().item() * 2 - 1,
        "exploration_eta_lat_std": (lat_dist.variance.mean().item() * 4) ** 0.5,
        "exploration_eta_lon_std": (lon_dist.variance.mean().item() * 4) ** 0.5,
    }

    return total_loss, metrics
