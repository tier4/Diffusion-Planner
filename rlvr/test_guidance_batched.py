"""Unit tests for rlvr/guidance_batched.py (per-sample guidance params)."""

import pytest
import torch

import guidance_gui.custom_guidance  # noqa: F401 -- registers collision_swerve
import rlvr.guidance_batched  # noqa: F401 -- registers the batched variants
from diffusion_planner.model.guidance.config import GuidanceConfig
from diffusion_planner.model.guidance.registry import build
from diffusion_planner.model.guidance.speed_guidance import SpeedGuidance

B, T, PN, HIST = 3, 10, 4, 5


def _make_x(requires_grad=False):
    torch.manual_seed(0)
    # Forward-moving ego with slight lateral variation per batch element.
    x = torch.zeros(B, 1, T + 1, 4)
    x[:, 0, :, 0] = torch.linspace(0, 20, T + 1)          # x forward
    x[:, 0, :, 1] = torch.randn(B, 1) * 0.3                # constant lateral offset
    x[:, 0, :, 2] = 1.0                                    # cos(0)
    x.requires_grad_(requires_grad)
    return x


def _make_inputs(with_neighbors=True):
    nb = torch.zeros(B, PN, HIST, 11)
    if with_neighbors:
        # One stopped neighbour 8 m ahead, slightly right; rest are empty slots.
        nb[:, 0, -1, 0] = 8.0
        nb[:, 0, -1, 1] = -0.5
        nb[:, 0, -1, 2] = 1.0
    return {"neighbor_agents_past": nb}


def _swerve(name, **params):
    return build(GuidanceConfig(name=name, enabled=True, scale=1.0, params=params))


# ---------------------------------------------------------------------------
# collision_swerve_batched
# ---------------------------------------------------------------------------

def test_swerve_batched_matches_scalar_original():
    x = _make_x()
    inputs = _make_inputs()
    for side in (1.0, -1.0):
        orig = _swerve("collision_swerve", side=side, range=8.0)
        batched = _swerve("collision_swerve_batched", eta_col=side, range=8.0)
        assert torch.allclose(orig._compute(x, inputs), batched._compute(x, inputs))


def test_swerve_batched_per_sample_eta():
    x = _make_x()
    x.data[:, 0, :, 1] = 0.2  # identical lateral position for all batch elems
    inputs = _make_inputs()
    eta = torch.tensor([1.0, -1.0, 0.5])
    fn = _swerve("collision_swerve_batched", eta_col=eta, range=8.0)
    out = fn._compute(x, inputs)
    assert torch.allclose(out[0], -out[1])
    assert torch.allclose(out[2], 0.5 * out[0])


def test_swerve_batched_zero_eta_is_inert():
    x = _make_x(requires_grad=True)
    fn = _swerve("collision_swerve_batched", eta_col=0.0, range=8.0)
    out = fn._compute(x, _make_inputs())
    assert torch.all(out == 0)
    grad = torch.autograd.grad(out.sum(), x, allow_unused=True)[0]
    assert grad is None or torch.all(grad == 0)


def test_swerve_batched_no_neighbors_zero_energy():
    fn = _swerve("collision_swerve_batched", eta_col=1.0, range=8.0)
    out = fn._compute(_make_x(), _make_inputs(with_neighbors=False))
    assert torch.all(out == 0)
    out = fn._compute(_make_x(), {})
    assert torch.all(out == 0)


def test_swerve_batched_gradient_pushes_left_for_positive_eta():
    x = _make_x(requires_grad=True)
    fn = _swerve("collision_swerve_batched", eta_col=1.0, range=8.0)
    out = fn._compute(x, _make_inputs())
    grad = torch.autograd.grad(out.sum(), x)[0]
    # Energy is maximised by the solver: positive dE/dy = push toward +y (left)
    lat_grad = grad[:, 0, 1:, 1]
    assert (lat_grad >= 0).all()
    assert lat_grad.sum() > 0


def test_swerve_batched_shape_mismatch_raises():
    fn = _swerve("collision_swerve_batched", eta_col=torch.ones(B + 1), range=8.0)
    with pytest.raises(ValueError, match="expected scalar"):
        fn._compute(_make_x(), _make_inputs())


def test_swerve_batched_energy_time_gated():
    x = _make_x(requires_grad=True)
    fn = _swerve("collision_swerve_batched", eta_col=1.0, range=8.0)
    t_in = torch.full((B,), 0.05)   # inside (t_min, t_max) window
    e = fn.energy(x, t_in, _make_inputs())
    grad = torch.autograd.grad(e.sum(), x)[0]
    assert grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# speed_stretch_batched
# ---------------------------------------------------------------------------

def test_stretch_batched_matches_scalar_speed_guidance():
    x = _make_x()
    for stretch in (0.8, 1.2):
        orig = SpeedGuidance(GuidanceConfig(name="speed", params={"stretch": stretch}))
        batched = _swerve("speed_stretch_batched", stretch=stretch)
        assert torch.allclose(orig._compute(x, {}), batched._compute(x, {}))


def test_stretch_batched_one_is_inert():
    x = _make_x(requires_grad=True)
    fn = _swerve("speed_stretch_batched", stretch=1.0)
    out = fn._compute(x, {})
    assert torch.all(out == 0)
    grad = torch.autograd.grad(out.sum(), x, allow_unused=True)[0]
    assert grad is None or torch.all(grad == 0)


def test_stretch_batched_per_sample():
    x = _make_x(requires_grad=True)
    stretch = torch.tensor([0.8, 1.0, 1.2])
    fn = _swerve("speed_stretch_batched", stretch=stretch)
    out = fn._compute(x, {})
    grad = torch.autograd.grad(out.sum(), x)[0]
    fwd_grad = grad[:, 0, 1:, 0]  # gradient along travel (+x) direction
    assert (fwd_grad[0] <= 0).all() and fwd_grad[0].sum() < 0   # slow down
    assert torch.all(grad[1] == 0)                              # inert
    assert (fwd_grad[2] >= 0).all() and fwd_grad[2].sum() > 0   # speed up


def test_lateral_batched_matches_stock_when_unprotected():
    from diffusion_planner.model.guidance.lateral_guidance import LateralGuidance
    x = _make_x()
    ref = torch.zeros(2 if False else B, T, 4); ref[..., 2] = 1.0
    inputs = {"reference_trajectory": ref}
    for eta in (0.0, 0.5, torch.tensor([0.3, -0.7, 1.0])):
        stock = LateralGuidance(GuidanceConfig(name="lateral",
                                params={"lambda_lat": 4.0, "eta_lat": eta}))
        mine = _swerve("lateral_batched", lambda_lat=4.0, eta_lat=eta, head_protect=0)
        assert torch.allclose(stock._compute(x, inputs), mine._compute(x, inputs), atol=1e-6)


def test_head_protect_zeroes_early_gradient():
    x = _make_x(requires_grad=True)
    ref = torch.zeros(B, T, 4); ref[..., 2] = 1.0
    inputs = {"reference_trajectory": ref, **_make_inputs()}
    lat = _swerve("lateral_batched", lambda_lat=4.0, eta_lat=1.0, head_protect=5)
    col = _swerve("collision_swerve_batched", eta_col=1.0, range=8.0, head_protect=5)
    out = lat._compute(x, inputs) + col._compute(x, inputs)
    grad = torch.autograd.grad(out.sum(), x)[0]
    assert torch.all(grad[:, 0, 1:6, :] == 0), "first 5 future steps must carry no gradient"
    assert grad[:, 0, 6:, :].abs().sum() > 0
