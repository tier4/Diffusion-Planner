"""Unit tests for the redesigned anchor_following guidance (path mode).

Run: python -m pytest rlvr/test_anchor_guidance.py -q
"""

import numpy as np
import pytest
import torch
from diffusion_planner.model.guidance.anchor_following import AnchorFollowingGuidance
from diffusion_planner.model.guidance.config import GuidanceConfig

T = 80


@pytest.fixture
def prototypes_path(tmp_path):
    """Three prototypes: straight slow, left turn, stationary (degenerate)."""
    protos = np.zeros((3, T, 2), dtype=np.float32)
    protos[0, :, 0] = np.linspace(0.3, 24.0, T)  # straight, ~3 m/s
    s = np.linspace(0.3, 24.0, T)
    protos[1, :, 0] = np.sin(s / 24.0 * np.pi / 2) * 24.0 * 2 / np.pi
    protos[1, :, 1] = (1 - np.cos(s / 24.0 * np.pi / 2)) * 24.0 * 2 / np.pi  # quarter left arc
    # protos[2] stays at origin (stop mode)
    path = tmp_path / "protos.npy"
    np.save(path, protos)
    return str(path)


def make_guidance(prototypes_path, idx=0, scale=1.0, **params):
    cfg = GuidanceConfig(
        name="anchor_following",
        enabled=True,
        scale=scale,
        params={"prototypes_path": prototypes_path, "anchor_index": idx, **params},
    )
    return AnchorFollowingGuidance(cfg)


def make_x(traj_xy: torch.Tensor, requires_grad=False) -> torch.Tensor:
    """[B=1, P=1, T+1, 4] with future xy = traj_xy [T, 2]."""
    x = torch.zeros(1, 1, T + 1, 4)
    x[0, 0, 1:, :2] = traj_xy
    if requires_grad:
        x.requires_grad_(True)
    return x


def straight_traj(speed=3.0, lateral=0.0):
    xy = torch.zeros(T, 2)
    xy[:, 0] = torch.linspace(0.1 * speed, 8.0 * speed, T)
    xy[:, 1] = lateral
    return xy


class TestPathBuilding:
    def test_extension_along_final_heading(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=0, extend_len=100.0)
        path = g._anchor_path
        assert path.shape[0] == T + 1
        # straight +x anchor: tail is 100 m beyond the last point, y unchanged
        assert torch.isclose(path[-1, 0], torch.tensor(124.0), atol=1e-3)
        assert torch.isclose(path[-1, 1], torch.tensor(0.0), atol=1e-4)

    def test_degenerate_anchor_extends_forward(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=2, extend_len=50.0)
        assert torch.isclose(g._anchor_path[-1, 0], torch.tensor(50.0), atol=1e-4)


class TestPathDistance:
    def test_on_path_is_zero(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=0)
        d = g._path_distance(straight_traj(speed=3.0).unsqueeze(0))
        assert d.max() < 1e-4

    def test_faster_ego_uses_extension(self, prototypes_path):
        # ego at 10 m/s ends 80 m out, far beyond the 24 m anchor, but the
        # extension covers it: distance stays ~0 (no backward drag).
        g = make_guidance(prototypes_path, idx=0)
        d = g._path_distance(straight_traj(speed=10.0).unsqueeze(0))
        assert d.max() < 1e-4

    def test_lateral_offset_measured(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=0)
        d = g._path_distance(straight_traj(speed=3.0, lateral=1.5).unsqueeze(0))
        assert torch.allclose(d, torch.full_like(d, 1.5), atol=1e-3)


class TestEnergy:
    def test_gradient_active_at_high_noise(self, prototypes_path):
        """Path mode must have gradient at mode-forming steps (t ~ 0.5)."""
        g = make_guidance(prototypes_path, idx=1)
        x = make_x(straight_traj(), requires_grad=True)
        e = g.energy(x, torch.tensor([0.5]), {})
        grad = torch.autograd.grad(e.sum(), x)[0]
        assert grad.abs().max() > 0

    def test_gradient_gated_outside_window(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=1, t_max=0.4)
        x = make_x(straight_traj(), requires_grad=True)
        e = g.energy(x, torch.tensor([0.5]), {})
        assert not e.requires_grad or torch.autograd.grad(e.sum(), x)[0].abs().max() == 0

    def test_gradient_bounded_by_dist_cap(self, prototypes_path):
        """Huber: per-waypoint gradient magnitude saturates at dist_cap."""
        g = make_guidance(prototypes_path, idx=0, dist_cap=2.0)
        far = straight_traj(speed=3.0, lateral=50.0)
        x = make_x(far, requires_grad=True)
        raw = g._compute_path(x)
        grad = torch.autograd.grad(raw.sum(), x)[0][0, 0, 1:, :2]
        assert grad.norm(dim=-1).max() <= 2.0 + 1e-4

    def test_time_compensation_constant_x0_shift(self, prototypes_path):
        """energy * sigma^2/alpha (the solver's amplification) is t-invariant."""
        g = make_guidance(prototypes_path, idx=0)
        shifts = []
        for t in (0.9, 0.5, 0.1, 0.02):
            x = make_x(straight_traj(lateral=1.0), requires_grad=True)
            ts = torch.tensor([t])
            e = g.energy(x, ts, {})
            grad = torch.autograd.grad(e.sum(), x)[0]
            log_a = -0.25 * ts**2 * 19.9 - 0.5 * ts * 0.1
            amp = (1 - torch.exp(2 * log_a)) / torch.exp(log_a)  # sigma^2/alpha
            shifts.append((grad.abs().max() * amp).item())
        assert max(shifts) / min(shifts) < 1.01

    def test_step_gain_calibration(self, prototypes_path):
        """At 1 m offset, effective x0 shift = step_gain * scale / std^2 * ... :
        grad * sigma^2/alpha * std^2 == step_gain * scale * offset."""
        g = make_guidance(prototypes_path, idx=0, step_gain=1.0, scale=2.0)
        x = make_x(straight_traj(lateral=1.0), requires_grad=True)
        ts = torch.tensor([0.3])
        e = g.energy(x, ts, {})
        grad = torch.autograd.grad(e.sum(), x)[0][0, 0, 1:, 1]  # y coords
        log_a = -0.25 * ts**2 * 19.9 - 0.5 * ts * 0.1
        amp = (1 - torch.exp(2 * log_a)) / torch.exp(log_a)
        shift = grad.abs() * amp * 20.0**2
        # every waypoint sits at exactly 1 m lateral offset -> shift = 2.0 m
        assert torch.allclose(shift, torch.full_like(shift, 2.0), atol=0.02)


class TestLegacyWaypointMode:
    def test_waypoint_energy_matches_legacy_formula(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=0, mode="waypoint", scale=3.0)
        traj = straight_traj(speed=5.0)
        x = make_x(traj)
        e = g.energy(x, torch.tensor([0.05]), {})
        anchor = torch.from_numpy(np.load(prototypes_path)[0])
        expected = -0.05 * 3.0 * ((traj - anchor) ** 2).sum()
        assert torch.isclose(e[0], expected, atol=1e-3)

    def test_waypoint_keeps_legacy_window(self, prototypes_path):
        """Legacy mode must still be gated to (0.005, 0.1)."""
        g = make_guidance(prototypes_path, idx=0, mode="waypoint")
        x = make_x(straight_traj(), requires_grad=True)
        e = g.energy(x, torch.tensor([0.5]), {})
        assert not e.requires_grad or torch.autograd.grad(e.sum(), x)[0].abs().max() == 0

    def test_invalid_mode_raises(self, prototypes_path):
        with pytest.raises(ValueError, match="mode"):
            make_guidance(prototypes_path, idx=0, mode="bogus")


class TestReward:
    def test_reward_ranks_closer_trajectory_higher(self, prototypes_path):
        g = make_guidance(prototypes_path, idx=1)
        on_left = torch.from_numpy(np.load(prototypes_path)[1]).float()
        straight = straight_traj(speed=3.0)
        traj = torch.stack([on_left, straight])  # [2, T, 2]
        traj4 = torch.cat([traj, torch.zeros(2, T, 2)], dim=-1)
        r = g.reward(traj4, {})
        assert r[0] > r[1]
