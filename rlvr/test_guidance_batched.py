"""Unit tests for rlvr/guidance_batched.py (per-sample guidance params)."""

import pytest
import torch
from diffusion_planner.model.guidance.config import GuidanceConfig
from diffusion_planner.model.guidance.registry import build
from diffusion_planner.model.guidance.speed_guidance import SpeedGuidance

import guidance_gui.custom_guidance  # noqa: F401 -- registers collision_swerve
import rlvr.guidance_batched  # noqa: F401 -- registers the batched variants

B, T, PN, HIST = 3, 10, 4, 5


def _make_x(requires_grad=False):
    torch.manual_seed(0)
    # Forward-moving ego with slight lateral variation per batch element.
    x = torch.zeros(B, 1, T + 1, 4)
    x[:, 0, :, 0] = torch.linspace(0, 20, T + 1)  # x forward
    x[:, 0, :, 1] = torch.randn(B, 1) * 0.3  # constant lateral offset
    x[:, 0, :, 2] = 1.0  # cos(0)
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
    t_in = torch.full((B,), 0.05)  # inside (t_min, t_max) window
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
    assert (fwd_grad[0] <= 0).all() and fwd_grad[0].sum() < 0  # slow down
    assert torch.all(grad[1] == 0)  # inert
    assert (fwd_grad[2] >= 0).all() and fwd_grad[2].sum() > 0  # speed up


def test_lateral_batched_matches_stock_when_unprotected():
    from diffusion_planner.model.guidance.lateral_guidance import LateralGuidance

    x = _make_x()
    ref = torch.zeros(B, T, 4)
    ref[..., 2] = 1.0
    inputs = {"reference_trajectory": ref}
    for eta in (0.0, 0.5, torch.tensor([0.3, -0.7, 1.0])):
        stock = LateralGuidance(
            GuidanceConfig(name="lateral", params={"lambda_lat": 4.0, "eta_lat": eta})
        )
        mine = _swerve("lateral_batched", lambda_lat=4.0, eta_lat=eta, head_protect=0)
        assert torch.allclose(stock._compute(x, inputs), mine._compute(x, inputs), atol=1e-6)


def test_head_protect_zeroes_early_gradient():
    x = _make_x(requires_grad=True)
    ref = torch.zeros(B, T, 4)
    ref[..., 2] = 1.0
    inputs = {"reference_trajectory": ref, **_make_inputs()}
    lat = _swerve("lateral_batched", lambda_lat=4.0, eta_lat=1.0, head_protect=5)
    col = _swerve("collision_swerve_batched", eta_col=1.0, range=8.0, head_protect=5)
    out = lat._compute(x, inputs) + col._compute(x, inputs)
    grad = torch.autograd.grad(out.sum(), x)[0]
    assert torch.all(grad[:, 0, 1:6, :] == 0), "first 5 future steps must carry no gradient"
    assert grad[:, 0, 6:, :].abs().sum() > 0


# ---------------------------------------------------------------------------
# FastGuidanceComposer._all_inert
# ---------------------------------------------------------------------------


def test_fast_composer_inert_detection():
    from rlvr.guidance_batched import build_head_composer

    # truly inert: every head at its inert point
    comp = build_head_composer({"lateral": 0.0, "collision": 0.0, "stretch": 0.0})
    assert comp._all_inert()

    # legacy longitudinal head ALONE must count as active (regression:
    # _eta_lon was unchecked, silently disabling guidance)
    comp = build_head_composer({"lateral": 0.0, "longitudinal": 0.7})
    assert not comp._all_inert()
    comp = build_head_composer({"lateral": 0.0, "longitudinal": 0.0})
    assert comp._all_inert()

    # a function exposing NO recognized eta attribute must force the
    # composer active
    comp = build_head_composer({"lateral": 0.0, "collision": 0.0})
    assert comp._all_inert()

    class _NoEtaFn:
        pass

    comp._functions.append(_NoEtaFn())
    assert not comp._all_inert()

    # a REAL stock SpeedGuidance band fn (stretch==1.0 but v_low/v_high
    # clamping active — the --speed_floor pattern) must force active too
    band = build(
        GuidanceConfig(name="speed", enabled=True, scale=1.0, params={"v_low": 2.0, "v_high": 14.0})
    )
    assert getattr(band, "_stretch", None) == 1.0  # the trap: looks inert
    comp = build_head_composer({"lateral": 0.0, "collision": 0.0})
    comp._functions.append(band)
    assert not comp._all_inert()


# ---------------------------------------------------------------------------
# DiTForwardMemo / dit_memo
# ---------------------------------------------------------------------------


class _CountingNet(torch.nn.Module):
    """Stand-in DiT: deterministic fn of (x, t, cross_c), counts forwards."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t, cross_c=None, neighbor_current_mask=None):
        self.calls += 1
        out = x * 2.0 + t.reshape(-1, *([1] * (x.dim() - 1)))
        if cross_c is not None:
            out = out + cross_c.mean()
        return out


def test_dit_memo_hit_on_equal_values_miss_on_change():
    from rlvr.guidance_batched import DiTForwardMemo

    net = _CountingNet()
    memo = DiTForwardMemo(net)
    x = torch.randn(2, 3, 11, 4)
    t = torch.tensor([0.5, 0.5])
    cond = torch.randn(2, 8)

    out1 = memo(x, t, cross_c=cond)
    assert net.calls == 1 and memo.misses == 1 and memo.hits == 0

    # value-equal x in a DIFFERENT tensor (the composer detaches/reshapes)
    # + same conditioning object -> hit, no extra forward
    out2 = memo(x.detach().clone().requires_grad_(True), t, cross_c=cond)
    assert net.calls == 1 and memo.hits == 1
    assert out2 is out1

    # changed x values -> miss
    memo(x + 1e-6, t, cross_c=cond)
    assert net.calls == 2

    # same x values but a DIFFERENT conditioning tensor object -> miss
    memo(x + 1e-6, t, cross_c=cond.clone())
    assert net.calls == 3


def test_dit_memo_inplace_mutation_cannot_false_hit():
    from rlvr.guidance_batched import DiTForwardMemo

    net = _CountingNet()
    memo = DiTForwardMemo(net)
    x = torch.randn(2, 3, 11, 4)
    t = torch.tensor([0.5, 0.5])

    out1 = memo(x, t)
    x.add_(1.0)  # caller mutates its tensor in place after the cached call
    out2 = memo(x, t)
    # the cached key was cloned, so the mutated x must NOT match the stale
    # entry — a fresh forward is required
    assert net.calls == 2
    assert not torch.equal(out1, out2)


def test_dit_memo_output_matches_unwrapped_and_is_detached():
    from rlvr.guidance_batched import DiTForwardMemo

    net = _CountingNet()
    memo = DiTForwardMemo(net)
    x = torch.randn(2, 3, 11, 4)
    t = torch.tensor([0.3, 0.3])
    expected = net(x, t)

    with torch.enable_grad():
        out = memo(x.clone().requires_grad_(True), t)
    assert torch.equal(out, expected)
    assert not out.requires_grad  # forward runs under no_grad


def test_dit_memo_context_manager_installs_and_restores():
    from rlvr.guidance_batched import DiTForwardMemo, dit_memo

    class _Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = _CountingNet()

    dec = _Decoder()
    orig = dec.dit
    with dit_memo(dec) as memo:
        assert isinstance(dec.dit, DiTForwardMemo)
        # wrapped module hidden from state_dict while installed
        assert all(not k.startswith("dit.") for k in dec.state_dict())
        x = torch.randn(1, 2, 11, 4)
        t = torch.tensor([0.1])
        dec.dit(x, t)
        dec.dit(x.clone(), t.clone())
        assert memo.hits == 1 and memo.misses == 1
    assert dec.dit is orig
    assert "dit.calls" not in dec.state_dict()  # plain restore

    # exception inside the block must still restore
    try:
        with dit_memo(dec):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert dec.dit is orig
