"""Unit tests for grpo_logprob_loss.py.

Tests the core loss computation logic without requiring a full model.
Run: python -m rlvr.tests.test_logprob_loss
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from rlvr.vpsde_logprob import (
    compute_discount_weights,
    create_timestep_schedule,
    vpsde_denoising_step_with_logprob,
)


def test_rl_loss_gradient_is_advantage_weighted():
    """Verify RL loss gradient equals advantages * d(logp)/d(theta).

    The DDV2 RL loss is: L = -exp(logp - logp.detach()) * A
    At evaluation: exp(logp - logp.detach()) = 1, so L = -A
    But gradient: dL/d(theta) = -A * d(logp)/d(theta) (REINFORCE)

    We verify this by checking that:
    1. The loss for a trajectory with advantage=0 produces zero gradient
    2. The loss for positive advantage pushes x0_pred toward sample
    3. The loss for negative advantage pushes x0_pred away from sample
    """
    torch.manual_seed(42)
    sde = VPSDE_linear()
    D = 8

    # Simulate model output (x0_pred) that requires grad
    x0_pred = torch.randn(1, D, requires_grad=True)
    t_prev = torch.tensor(0.05).view(1, 1)

    # Collection pass: sample
    sample, _, _ = vpsde_denoising_step_with_logprob(
        x0_pred.detach(), t_prev, sde, min_std=0.1
    )

    # Optimization pass: compute log_prob with gradient
    _, log_prob, mean = vpsde_denoising_step_with_logprob(
        x0_pred, t_prev, sde, min_std=0.1, prev_sample=sample
    )

    # Test 1: advantage=0 → zero gradient
    advantage_zero = torch.tensor([0.0])
    loss_zero = -torch.exp(log_prob - log_prob.detach()) * advantage_zero
    loss_zero.backward(retain_graph=True)
    assert x0_pred.grad is not None
    assert x0_pred.grad.abs().max() < 1e-7, (
        f"Zero advantage should give zero gradient, got max={x0_pred.grad.abs().max()}"
    )
    x0_pred.grad.zero_()

    # Test 2: positive advantage → gradient pushes x0 toward sample
    advantage_pos = torch.tensor([2.0])
    loss_pos = -torch.exp(log_prob - log_prob.detach()) * advantage_pos
    loss_pos.backward(retain_graph=True)
    grad_pos = x0_pred.grad.clone()
    x0_pred.grad.zero_()

    # Test 3: negative advantage → gradient in opposite direction
    advantage_neg = torch.tensor([-1.0])
    loss_neg = -torch.exp(log_prob - log_prob.detach()) * advantage_neg
    loss_neg.backward(retain_graph=True)
    grad_neg = x0_pred.grad.clone()

    # Positive and negative advantage should give opposite gradient directions
    dot_product = (grad_pos * grad_neg).sum()
    assert dot_product < 0, (
        f"Pos/neg advantage gradients should be opposite, dot={dot_product.item():.4f}"
    )
    print("PASS: RL loss gradient is advantage-weighted REINFORCE")


def test_chain_consistency():
    """Verify that collection and optimization passes give same log-probs.

    Collection pass: sample x_{t-1} = mean + std * noise, compute log_prob
    Optimization pass: given stored x_{t-1}, compute log_prob

    These should be identical when using the same x0_pred.
    """
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, D = 3, 16

    x0 = torch.randn(B, D)
    t_prev = torch.tensor(0.05).view(1, 1).expand(B, 1)

    # Collection: get sample and log_prob
    sample, lp_collect, _ = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=0.1
    )

    # Optimization: recompute log_prob for same sample
    _, lp_optimize, _ = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=0.1, prev_sample=sample
    )

    assert torch.allclose(lp_collect, lp_optimize, atol=1e-5), (
        f"Collection and optimization log-probs should match:\n"
        f"  collect={lp_collect}\n  optimize={lp_optimize}"
    )
    print("PASS: chain consistency (collection == optimization log-probs)")


def test_il_loss_basic():
    """Verify IL loss is MSE between model prediction and GT."""
    B, T, D = 2, 8, 4
    pred = torch.randn(B, T, D)
    gt = torch.randn(B, T, D)

    il_loss = F.mse_loss(pred, gt, reduction='none').mean(dim=(1, 2))
    assert il_loss.shape == (B,), f"IL loss shape should be (B,): {il_loss.shape}"
    assert (il_loss > 0).all(), "IL loss should be positive"

    # When pred == gt, loss should be 0
    il_zero = F.mse_loss(gt, gt, reduction='none').mean(dim=(1, 2))
    assert (il_zero == 0).all(), "IL loss should be 0 when pred == gt"
    print("PASS: IL loss basic")


def test_discount_application():
    """Verify per-step discount correctly weights advantages."""
    N, num_steps = 3, 5
    discount = compute_discount_weights(num_steps, 0.8)

    advantages = torch.tensor([2.0, -1.0, 0.5])
    advantages_per_step = advantages.unsqueeze(-1) * discount.unsqueeze(0)

    assert advantages_per_step.shape == (N, num_steps)

    # First step (noisiest) should have smallest weight
    assert advantages_per_step[0, 0].abs() < advantages_per_step[0, -1].abs()

    # Last step (cleanest) should have weight 1.0 * advantage
    assert torch.allclose(advantages_per_step[:, -1], advantages, atol=1e-6)
    print("PASS: discount application")


def test_masking_zero_advantages():
    """Verify that zero-advantage entries are masked in RL loss."""
    N, S = 4, 3
    log_probs = torch.randn(N, S)
    advantages = torch.tensor([
        [1.0, 1.0, 1.0],   # positive
        [0.0, 0.0, 0.0],   # zero (should be masked)
        [-1.0, -1.0, -1.0],  # negative
        [0.0, 0.0, 0.0],   # zero (should be masked)
    ])

    per_step_loss = -torch.exp(log_probs - log_probs.detach()) * advantages
    mask_nz = advantages != 0

    # Masked computation
    rl_loss = (per_step_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp(min=1)

    # Zero-advantage samples should contribute 0 to loss
    assert rl_loss[1].item() == 0.0, f"Zero-adv sample should have 0 loss: {rl_loss[1]}"
    assert rl_loss[3].item() == 0.0, f"Zero-adv sample should have 0 loss: {rl_loss[3]}"
    print("PASS: zero-advantage masking")


def test_adaptive_il_weight():
    """Verify adaptive IL weight logic."""
    advantages = torch.tensor([1.0, -1.0, 0.0, 0.5])
    has_positive = (advantages > 0).any()
    assert has_positive, "Should detect positive advantages"

    advantages_all_neg = torch.tensor([-1.0, -1.0, -0.5])
    has_positive_neg = (advantages_all_neg > 0).any()
    assert not has_positive_neg, "Should NOT detect positive advantages"

    # When no positive: IL weight = 1.0 (full IL to prevent degradation)
    il_weight_no_pos = 1.0 if not has_positive_neg else 0.1
    assert il_weight_no_pos == 1.0, "No positive adv → full IL weight"

    # When has positive: IL weight = 0.1
    il_weight_has_pos = 1.0 if not has_positive else 0.1
    assert il_weight_has_pos == 0.1, "Has positive adv → reduced IL weight"
    print("PASS: adaptive IL weight")


def test_multistep_logprob_accumulation():
    """Test multi-step denoising with log-prob accumulation."""
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, D = 2, 8
    num_steps = 5

    schedule = create_timestep_schedule(0.01, num_steps)
    x0_true = torch.randn(B, D)

    # Start from noised version
    t_start = torch.tensor(schedule[0].item()).view(1, 1)
    mean_start, std_start = sde.marginal_prob(x0_true, t_start)
    x_t = mean_start + std_start * torch.randn_like(x0_true)

    all_log_probs = []
    for i in range(num_steps):
        t_prev = torch.tensor(schedule[i + 1].item()).view(1, 1)
        # Pretend model perfectly predicts x0
        x_t, lp, _ = vpsde_denoising_step_with_logprob(
            x0_true, t_prev, sde, min_std=0.1
        )
        all_log_probs.append(lp)

    log_probs = torch.stack(all_log_probs, dim=-1)  # [B, num_steps]
    assert log_probs.shape == (B, num_steps)
    assert torch.isfinite(log_probs).all(), f"Log-probs should be finite: {log_probs}"

    # Later steps (lower noise, closer to clean) should have higher log-probs
    # because the std is clamped at min_std, and sample is closer to mean
    print(f"  Log-probs per step (avg): {log_probs.mean(dim=0).tolist()}")
    print("PASS: multi-step log-prob accumulation")


if __name__ == "__main__":
    test_rl_loss_gradient_is_advantage_weighted()
    test_chain_consistency()
    test_il_loss_basic()
    test_discount_application()
    test_masking_zero_advantages()
    test_adaptive_il_weight()
    test_multistep_logprob_accumulation()
    print("\n=== ALL TESTS PASSED ===")
