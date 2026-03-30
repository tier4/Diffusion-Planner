"""Unit tests for rlvr.closed_loop.gae.

Run: python -m rlvr.closed_loop.test_gae
"""

from __future__ import annotations

import torch

from rlvr.closed_loop.gae import compute_gae


def test_single_step():
    """One step: advantage = r + gamma*V_terminal - V(s_0)."""
    advantages, targets = compute_gae(
        rewards=[1.0], values=[0.5], terminal_value=0.0, gamma=0.99, lam=0.95,
    )
    assert advantages.shape == (1,)
    expected_adv = 1.0 + 0.99 * 0.0 - 0.5  # = 0.5
    assert abs(advantages[0].item() - expected_adv) < 1e-6, f"adv={advantages[0].item()}"
    print("  PASS: single_step")


def test_constant_reward():
    """Constant reward with zero values => advantages should be positive."""
    T = 10
    advantages, targets = compute_gae(
        rewards=[1.0] * T, values=[0.0] * T, terminal_value=0.0,
        gamma=0.99, lam=0.95,
    )
    assert advantages.shape == (T,)
    # All advantages should be positive (reward > 0, value = 0)
    assert (advantages > 0).all(), f"Expected all positive, got {advantages}"
    # Earlier steps should have higher advantage (more future reward)
    assert advantages[0] > advantages[-1], "Earlier steps should have higher advantage"
    print("  PASS: constant_reward")


def test_value_targets():
    """Value targets = advantages + values."""
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    advantages, targets = compute_gae(rewards, values, terminal_value=0.0)
    expected_targets = advantages + torch.tensor(values)
    assert torch.allclose(targets, expected_targets, atol=1e-6)
    print("  PASS: value_targets")


def test_terminal_episode():
    """Terminal episode with terminal_value=0 and large negative reward at end."""
    rewards = [0.5, 0.5, -10.0]
    values = [1.0, 1.0, 1.0]
    advantages, targets = compute_gae(
        rewards, values, terminal_value=0.0, gamma=0.99, lam=0.95,
    )
    # Last step should have very negative advantage
    assert advantages[-1] < -5, f"Last advantage should be very negative: {advantages[-1]}"
    # Earlier steps should also be negative (GAE propagates)
    assert advantages[0] < 0, f"First advantage should be negative: {advantages[0]}"
    print("  PASS: terminal_episode")


if __name__ == "__main__":
    print("Running GAE tests...")
    test_single_step()
    test_constant_reward()
    test_value_targets()
    test_terminal_episode()
    print("\nAll GAE tests passed!")
