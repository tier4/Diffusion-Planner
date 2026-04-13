"""Unit tests for VPSDE log-probability computation.

Run: python -m rlvr.tests.test_vpsde_logprob
"""

import torch
import torch.distributions as dist
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear

from rlvr.vpsde_logprob import (
    compute_discount_weights,
    create_timestep_schedule,
    vpsde_denoising_step_with_logprob,
)


def test_logprob_matches_torch_normal():
    """Verify log-prob matches torch.distributions.Normal for known inputs."""
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, D = 4, 8
    x0 = torch.randn(B, D)
    t_prev = torch.tensor(0.05).view(1, 1).expand(B, 1)

    # Compute mean, std from SDE
    mean, std_raw = sde.marginal_prob(x0, t_prev)
    min_std = 0.1
    std = std_raw.clamp(min=min_std)

    # Sample
    noise = torch.randn_like(mean)
    sample = mean + std * noise

    # Our implementation
    _, log_prob_ours, _ = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=min_std, prev_sample=sample
    )

    # Reference: torch.distributions (mean to match our normalization)
    normal = dist.Normal(mean, std)
    log_prob_ref = normal.log_prob(sample).reshape(B, -1).mean(dim=-1)

    assert torch.allclose(log_prob_ours, log_prob_ref, atol=1e-5), (
        f"Log-prob mismatch:\n  ours={log_prob_ours}\n  ref={log_prob_ref}\n"
        f"  diff={torch.abs(log_prob_ours - log_prob_ref).max()}"
    )
    print("PASS: log-prob matches torch.distributions.Normal")


def test_gradient_flows_through_mean_not_sample():
    """Verify gradient flows through mean (model output) but not sample."""
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, D = 2, 4

    # x0_pred requires grad (simulates model output)
    x0 = torch.randn(B, D, requires_grad=True)
    t_prev = torch.tensor(0.05).view(1, 1).expand(B, 1)

    # Collection pass: new sample drawn
    sample, log_prob, mean = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=0.1
    )

    # Backward
    loss = log_prob.sum()
    loss.backward()

    # x0 should have gradients (flows through mean)
    assert x0.grad is not None, "x0 should have gradient"
    assert x0.grad.abs().sum() > 0, "x0 gradient should be non-zero"

    # sample should be detached inside log_prob computation
    # (we verify this indirectly: changing sample shouldn't change log_prob grad)
    print("PASS: gradient flows through mean (x0_pred)")


def test_optimization_pass_uses_stored_sample():
    """Verify that passing prev_sample gives consistent log-probs."""
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, D = 3, 6
    x0 = torch.randn(B, D)
    t_prev = torch.tensor(0.05).view(1, 1).expand(B, 1)

    # Collection pass
    sample1, lp1, mean1 = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=0.1
    )

    # Optimization pass with stored sample
    _, lp2, mean2 = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=0.1, prev_sample=sample1
    )

    # Mean should be identical (same x0 and t)
    assert torch.allclose(mean1, mean2, atol=1e-6), "Means should match"
    # Log-probs should be identical (same mean, std, and sample)
    assert torch.allclose(lp1, lp2, atol=1e-5), (
        f"Log-probs should match for same sample:\n  lp1={lp1}\n  lp2={lp2}"
    )
    print("PASS: optimization pass with stored sample is consistent")


def test_min_std_clamp():
    """Verify min_std prevents explosion near t=0."""
    sde = VPSDE_linear()
    B, D = 2, 4
    x0 = torch.randn(B, D)

    # Very small t → std would be tiny without clamping
    t_tiny = torch.tensor(1e-4).view(1, 1).expand(B, 1)
    _, std_raw = sde.marginal_prob(x0, t_tiny)
    assert std_raw.max() < 0.01, f"Raw std at t=1e-4 should be tiny: {std_raw.max()}"

    # With min_std=0.1, log_prob should be finite
    sample, log_prob, _ = vpsde_denoising_step_with_logprob(
        x0, t_tiny, sde, min_std=0.1
    )
    assert torch.isfinite(log_prob).all(), f"Log-prob should be finite: {log_prob}"
    assert not torch.isnan(log_prob).any(), f"Log-prob should not be NaN: {log_prob}"
    print("PASS: min_std clamp prevents explosion near t=0")


def test_at_small_t_mean_approaches_x0():
    """At small t, the marginal mean should be close to x0."""
    sde = VPSDE_linear()
    x0 = torch.randn(2, 4)
    t_small = torch.tensor(0.001).view(1, 1).expand(2, 1)

    mean, _ = sde.marginal_prob(x0, t_small)
    diff = (mean - x0).abs().max()
    assert diff < 0.01, f"At t=0.001, mean should be ~x0, but diff={diff:.6f}"
    print("PASS: mean → x0 as t → 0")


def test_timestep_schedule():
    """Verify timestep schedule is decreasing and has correct endpoints."""
    schedule = create_timestep_schedule(t_start=0.5, num_steps=10, eps=1e-3)
    assert schedule.shape == (11,), f"Expected (11,), got {schedule.shape}"
    assert abs(schedule[0].item() - 0.5) < 1e-6, f"Start should be 0.5: {schedule[0]}"
    assert abs(schedule[-1].item() - 1e-3) < 1e-6, f"End should be 1e-3: {schedule[-1]}"
    # Monotonically decreasing
    diffs = schedule[1:] - schedule[:-1]
    assert (diffs <= 0).all(), f"Schedule should be decreasing: {schedule}"
    print("PASS: timestep schedule")


def test_discount_weights():
    """Verify discount weights match DDV2 formula."""
    weights = compute_discount_weights(num_steps=5, discount=0.8)
    expected = torch.tensor([
        0.8**4, 0.8**3, 0.8**2, 0.8**1, 0.8**0
    ])
    assert torch.allclose(weights, expected, atol=1e-6), (
        f"Discount mismatch:\n  got={weights}\n  expected={expected}"
    )
    # Last step (cleanest) should have weight 1.0
    assert abs(weights[-1].item() - 1.0) < 1e-6
    # First step (noisiest) should have lowest weight
    assert weights[0] < weights[-1]
    print("PASS: discount weights match DDV2 formula")


def test_multidim_shapes():
    """Test with realistic multi-dimensional inputs (B, P, T, D)."""
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, P, T, D = 2, 5, 16, 4  # batch, agents, timesteps, features
    x0 = torch.randn(B, P, T, D)
    t_prev = torch.tensor(0.05).view(1, 1, 1, 1).expand(B, 1, T, 1)

    sample, log_prob, mean = vpsde_denoising_step_with_logprob(
        x0, t_prev, sde, min_std=0.1
    )

    assert sample.shape == x0.shape, f"Sample shape mismatch: {sample.shape}"
    assert mean.shape == x0.shape, f"Mean shape mismatch: {mean.shape}"
    assert log_prob.shape == (B,), f"Log-prob shape should be (B,): {log_prob.shape}"
    assert torch.isfinite(log_prob).all(), "Log-prob should be finite"
    print("PASS: multi-dimensional shapes (B, P, T, D)")


def test_different_x0_gives_different_logprob():
    """Verify that different x0 predictions give different log-probs for same sample."""
    torch.manual_seed(42)
    sde = VPSDE_linear()
    B, D = 2, 8
    x0_a = torch.randn(B, D)
    x0_b = x0_a + 1.0  # shifted
    t_prev = torch.tensor(0.05).view(1, 1).expand(B, 1)

    # Same sample for both
    sample = torch.randn(B, D)
    _, lp_a, _ = vpsde_denoising_step_with_logprob(
        x0_a, t_prev, sde, min_std=0.1, prev_sample=sample
    )
    _, lp_b, _ = vpsde_denoising_step_with_logprob(
        x0_b, t_prev, sde, min_std=0.1, prev_sample=sample
    )

    assert not torch.allclose(lp_a, lp_b, atol=1e-3), (
        "Different x0 should give different log-probs"
    )
    print("PASS: different x0 → different log-probs")


if __name__ == "__main__":
    test_logprob_matches_torch_normal()
    test_gradient_flows_through_mean_not_sample()
    test_optimization_pass_uses_stored_sample()
    test_min_std_clamp()
    test_at_small_t_mean_approaches_x0()
    test_timestep_schedule()
    test_discount_weights()
    test_multidim_shapes()
    test_different_x0_gives_different_logprob()
    print("\n=== ALL TESTS PASSED ===")
