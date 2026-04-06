"""Generalized Advantage Estimation (GAE) for closed-loop rollout."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: list[float],
    values: list[float],
    terminal_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets.

    Args:
        rewards: [r_0, ..., r_{T-1}] per-step rewards.
        values: [V(s_0), ..., V(s_{T-1})] value estimates.
        terminal_value: V(s_T) — 0 if episode terminated, otherwise bootstrap.
        gamma: Discount factor.
        lam: GAE lambda for bias-variance tradeoff.

    Returns:
        advantages: [T] tensor of GAE advantages A_t.
        value_targets: [T] tensor of discounted return targets for value function.
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = terminal_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae

    value_targets = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, value_targets
