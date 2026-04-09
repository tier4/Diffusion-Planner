"""Unit tests for SpeedGuidance stretch mode."""

import torch
import pytest


def _make_trajectory(B=1, T=20, speed=2.0, dt=0.1):
    """Create a straight-line trajectory at constant speed."""
    # x increases by speed*dt each step, y=0, heading=(1,0)
    positions = torch.zeros(B, 1, T + 1, 4)
    for t in range(T + 1):
        positions[:, 0, t, 0] = speed * dt * t  # x
        positions[:, 0, t, 2] = 1.0  # cos_yaw
    return positions


def _build_guidance(stretch=1.0, v_low=0.0, v_high=14.0):
    from diffusion_planner.model.guidance.config import GuidanceConfig
    from diffusion_planner.model.guidance.speed_guidance import SpeedGuidance

    cfg = GuidanceConfig(
        name="speed", enabled=True, scale=1.0,
        params={"stretch": stretch, "v_low": v_low, "v_high": v_high},
    )
    return SpeedGuidance(cfg)


def test_stretch_1_is_noop():
    """stretch=1.0 should not activate stretch mode (falls through to band mode)."""
    g = _build_guidance(stretch=1.0, v_high=100.0)
    x = _make_trajectory(speed=2.0)
    x.requires_grad_(True)
    energy = g._compute(x, {})
    # Speed is well within [0, 100] band → no correction → energy ≈ 0
    assert energy.abs().item() < 1e-6


def test_stretch_gt1_positive_energy():
    """stretch>1 should produce positive energy (pushes trajectory faster)."""
    g = _build_guidance(stretch=1.5)
    x = _make_trajectory(speed=2.0)
    energy = g._compute(x, {})
    assert energy.item() > 0


def test_stretch_lt1_negative_energy():
    """stretch<1 should produce negative energy (pushes trajectory slower)."""
    g = _build_guidance(stretch=0.5)
    x = _make_trajectory(speed=2.0)
    energy = g._compute(x, {})
    assert energy.item() < 0


def test_stretch_gradient_direction():
    """Gradient of stretch>1 energy should push positions forward (increase x)."""
    g = _build_guidance(stretch=1.5)
    x = _make_trajectory(speed=2.0)
    x.requires_grad_(True)
    energy = g._compute(x, {})
    # Surrogate: dot(correction.detach(), pos). grad_x = correction.
    # correction = disp * (1.5 - 1) = positive along x direction
    # So gradient should be positive in x for later timesteps
    grad = torch.autograd.grad(energy.sum(), x)[0]
    # Later positions should get positive x gradient
    mean_x_grad = grad[0, 0, 10:, 0].mean()
    assert mean_x_grad.item() > 0


def test_stretch_stronger_with_higher_value():
    """stretch=2.0 should produce stronger energy than stretch=1.2."""
    x = _make_trajectory(speed=2.0)
    e12 = _build_guidance(stretch=1.2)._compute(x, {}).item()
    e20 = _build_guidance(stretch=2.0)._compute(x, {}).item()
    assert abs(e20) > abs(e12)


def test_band_mode_unchanged():
    """With stretch=1.0, overspeed should produce nonzero correction energy."""
    g = _build_guidance(stretch=1.0, v_high=1.0)  # speed=2 > v_high=1
    x = _make_trajectory(speed=2.0)
    energy = g._compute(x, {})
    # Overspeed → correction active → nonzero energy
    assert abs(energy.item()) > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
