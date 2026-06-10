"""Loss functions for the Exploration Policy (joint GRPO training).

The policy outputs one Beta distribution per guidance head, K η values are
sampled from them, each generates a trajectory scored by the reward function.
GRPO-style group-relative advantages are used to train the policy via
advantage-weighted log probability (advantage_logprob mode).

Loss = policy_gradient + c_e * entropy_loss + c_kl * kl_loss + c_a * action_cost

Where:
- policy_gradient: -mean(A_k * log_prob_k) — advantage-weighted log_prob
- entropy_loss: -mean(Σ_h H(dist_h)) — encourage exploration
- kl_loss: Σ_h KL(current_h || init) — anchor to initial zero-mean policy
- action_cost: Σ_h (2*mean_h - 1)^2 — differentiable pull of the deterministic
  action (the Beta mean) toward η=0. A tie-breaker for inertness: negligible
  next to real advantages, but breaks reward-indifference plateaus on scenes
  where doing nothing is already optimal.
"""

from __future__ import annotations

import math

import torch
from torch.distributions import Beta, kl_divergence

# Initial Beta params from zero-initialized GuidanceHead: softplus(0) + 1
_INIT_PARAM = math.log(2) + 1.0
_INIT_DIST = {}


def _get_init_distribution(device: torch.device) -> Beta:
    """The initial (zero-init) Beta distribution for KL computation."""
    key = str(device)
    if key not in _INIT_DIST:
        param = torch.tensor(_INIT_PARAM, device=device)
        _INIT_DIST[key] = Beta(param, param)
    return _INIT_DIST[key]


def _get_init_distributions(device: torch.device) -> tuple[Beta, Beta]:
    """Backward-compat: the (lat, lon) pair of init distributions."""
    d = _get_init_distribution(device)
    return d, d


def compute_exploration_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    lat_dist: Beta | None = None,
    lon_dist: Beta | None = None,
    entropy_coef: float = 0.05,
    kl_coef: float = 0.01,
    dists: dict[str, Beta] | None = None,
    action_cost_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the exploration policy loss for one scene.

    Args:
        advantages: [K] GRPO group-relative advantages for each η sample.
        log_probs: [K] log π(η_k) summed over heads.
        lat_dist / lon_dist: legacy 2-head API — Beta distributions for
            η_lat / η_lon. Ignored when ``dists`` is given.
        entropy_coef: Weight for entropy bonus (higher = more exploration).
        kl_coef: Weight for KL penalty against initial policy.
        dists: head name -> Beta distribution (any number of heads).
        action_cost_coef: Weight for the η=0 action-cost tie-breaker.

    Returns:
        (loss, metrics_dict)
    """
    device = advantages.device
    if dists is None:
        if lat_dist is None or lon_dist is None:
            raise ValueError("pass either dists or both lat_dist and lon_dist")
        dists = {"lateral": lat_dist, "longitudinal": lon_dist}

    # --- Advantage-weighted policy gradient ---
    # Negative because we maximize reward (minimize negative advantage-weighted log_prob)
    policy_loss = -(advantages * log_probs).mean()

    # --- Entropy bonus ---
    # Maximize entropy to prevent η collapse. Beta entropy is well-defined.
    # Reduce to scalar with .mean() since distributions may have batch dims.
    entropy_value = sum(d.entropy().mean() for d in dists.values())
    entropy_loss = -entropy_value

    # --- KL divergence against initial policy ---
    # Prevents the policy from straying too far from the zero-mean init
    init_dist = _get_init_distribution(device)
    kl_value = sum(kl_divergence(d, init_dist).mean() for d in dists.values())
    kl_loss = kl_value

    # --- Action cost: pull the deterministic action (Beta mean) toward 0 ---
    action_cost = sum(((2.0 * d.mean - 1.0) ** 2).mean() for d in dists.values())

    # --- Total loss (guaranteed scalar) ---
    total_loss = (
        policy_loss
        + entropy_coef * entropy_loss
        + kl_coef * kl_loss
        + action_cost_coef * action_cost
    )

    metrics = {
        "exploration_policy_loss": policy_loss.item(),
        "exploration_entropy": entropy_value.item(),
        "exploration_kl": kl_value.item(),
        "exploration_action_cost": float(action_cost),
        "exploration_total_loss": total_loss.item(),
    }
    for name, d in dists.items():
        metrics[f"exploration_eta_{name}_mean"] = d.mean.mean().item() * 2 - 1
        metrics[f"exploration_eta_{name}_std"] = (d.variance.mean().item() * 4) ** 0.5
    # Legacy metric aliases for the original 2-head layout
    if "lateral" in dists:
        metrics["exploration_eta_lat_mean"] = metrics["exploration_eta_lateral_mean"]
        metrics["exploration_eta_lat_std"] = metrics["exploration_eta_lateral_std"]
    if "longitudinal" in dists:
        metrics["exploration_eta_lon_mean"] = metrics["exploration_eta_longitudinal_mean"]
        metrics["exploration_eta_lon_std"] = metrics["exploration_eta_longitudinal_std"]

    return total_loss, metrics
