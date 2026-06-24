"""Tests for the temporal-stability probe metrics (SAGE-JEPA plan §4).

Run with the OnePlanner venv (planner_metrics is dependency-isolated: numpy + torch):

    PYTHONPATH=/mnt/nvme/Diffusion-Planner \
        /mnt/nvme/OnePlanner/.venv/bin/python -m pytest \
        planner_metrics/test_temporal_stability.py -q
"""

from __future__ import annotations

import math

import torch

from planner_metrics.config import RewardConfig


# --------------------------------------------------------------------------
# Trajectory builders (analytic, noise-free) used across the metric tests.
# Each returns (T, 4): x, y, cos(heading), sin(heading).
# --------------------------------------------------------------------------
def _straight(T: int, v: float = 5.0, dt: float = 0.1) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32) * dt
    x = v * t
    y = torch.zeros_like(t)
    cos = torch.ones_like(t)
    sin = torch.zeros_like(t)
    return torch.stack([x, y, cos, sin], dim=-1)


def _arc(T: int, R: float = 20.0, v: float = 5.0, dt: float = 0.1) -> torch.Tensor:
    """Constant-radius, constant-speed circular arc: constant curvature 1/R,
    so curvature-RATE is ~0 even though curvature is large."""
    omega = v / R
    t = torch.arange(T, dtype=torch.float32) * dt
    theta = omega * t
    x = R * torch.sin(theta)
    y = R * (1.0 - torch.cos(theta))
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([x, y, cos, sin], dim=-1)


def _curvature_ramp(T: int, c: float, v: float = 5.0, dt: float = 0.1) -> torch.Tensor:
    """Heading turns ever faster (theta = 0.5*c*t^2) at constant speed, so the
    path curvature increases roughly linearly in time -> nonzero curvature-rate.
    Larger c -> steeper curvature ramp."""
    xs = [0.0]
    ys = [0.0]
    coss = []
    sins = []
    for k in range(T):
        tk = k * dt
        theta = 0.5 * c * tk * tk
        coss.append(math.cos(theta))
        sins.append(math.sin(theta))
        if k < T - 1:
            xs.append(xs[-1] + v * dt * math.cos(theta))
            ys.append(ys[-1] + v * dt * math.sin(theta))
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    cos = torch.tensor(coss, dtype=torch.float32)
    sin = torch.tensor(sins, dtype=torch.float32)
    return torch.stack([x, y, cos, sin], dim=-1)


# --------------------------------------------------------------------------
# compute_curvature_rate_score_batch
# --------------------------------------------------------------------------
def test_curvature_rate_straight_line_is_near_zero():
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    traj = _straight(80, dt=cfg.dt)[None]  # (1, 80, 4)
    score = compute_curvature_rate_score_batch(traj, cfg)
    assert score.shape == (1,)
    assert abs(score.item()) < 1e-2, f"straight line should score ~0, got {score.item()}"


def test_curvature_rate_steady_turn_is_near_zero():
    """The discriminating test: a steady circular turn has LARGE curvature but
    ZERO curvature-rate, so this metric (unlike a curvature metric) must score ~0."""
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    traj = _arc(80, R=20.0, v=5.0, dt=cfg.dt)[None]
    score = compute_curvature_rate_score_batch(traj, cfg)
    assert abs(score.item()) < 1e-2, (
        f"constant-curvature arc should score ~0 (rate, not curvature), got {score.item()}"
    )


def test_curvature_rate_penalizes_changing_curvature():
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    arc = _arc(80, R=20.0, v=5.0, dt=cfg.dt)[None]
    ramp = _curvature_ramp(80, c=0.5, dt=cfg.dt)[None]
    arc_score = compute_curvature_rate_score_batch(arc, cfg)
    ramp_score = compute_curvature_rate_score_batch(ramp, cfg)
    # scores are negative (penalties); a curvature ramp must be clearly worse
    # than a steady arc of the same curvature scale.
    assert ramp_score.item() < arc_score.item() - 1e-2, (
        f"changing-curvature ramp ({ramp_score.item()}) should score worse than "
        f"steady arc ({arc_score.item()})"
    )


def test_curvature_rate_monotonic_in_ramp_steepness():
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    mild = _curvature_ramp(80, c=0.3, dt=cfg.dt)[None]
    steep = _curvature_ramp(80, c=0.9, dt=cfg.dt)[None]
    mild_score = compute_curvature_rate_score_batch(mild, cfg)
    steep_score = compute_curvature_rate_score_batch(steep, cfg)
    assert steep_score.item() < mild_score.item(), (
        f"steeper curvature ramp ({steep_score.item()}) should score worse than "
        f"milder ({mild_score.item()})"
    )


def test_curvature_rate_short_traj_returns_zeros():
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    traj = _straight(6, dt=cfg.dt)[None]  # below SG window -> graceful zeros
    score = compute_curvature_rate_score_batch(traj, cfg)
    assert score.shape == (1,)
    assert score.item() == 0.0


def test_subscores_batch_surfaces_curvature_rate():
    """The temporal-stability probe reads curvature_rate off the shared subscore
    dict, so compute_subscores_batch must expose it (as an (N,) tensor matching
    the standalone metric)."""
    from planner_metrics.aggregate import compute_subscores_batch
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    ego = _curvature_ramp(80, c=0.6, dt=cfg.dt)[None]  # (1, 80, 4)
    data = {"ego_shape": torch.tensor([[2.7, 4.5, 2.0]])}
    subs = compute_subscores_batch(ego, data, cfg)
    assert "curvature_rate" in subs, "compute_subscores_batch must expose curvature_rate"
    assert subs["curvature_rate"].shape == (1,)
    expected = compute_curvature_rate_score_batch(ego, cfg)
    assert torch.allclose(subs["curvature_rate"], expected)


def test_curvature_rate_batched():
    from planner_metrics.subscores import compute_curvature_rate_score_batch

    cfg = RewardConfig()
    batch = torch.cat(
        [
            _straight(80, dt=cfg.dt)[None],
            _curvature_ramp(80, c=0.6, dt=cfg.dt)[None],
        ],
        dim=0,
    )  # (2, 80, 4)
    score = compute_curvature_rate_score_batch(batch, cfg)
    assert score.shape == (2,)
    # row 0 straight ~0, row 1 ramp clearly negative
    assert abs(score[0].item()) < 1e-2
    assert score[1].item() < score[0].item() - 1e-2


# --------------------------------------------------------------------------
# compute_replan_consistency_batch (open-loop proxy for SAGE prefix-cascade)
#
# Setup: a prediction tau_a is made at frame t (frame-t ego coords). One step-
# offset g later, the ego (having followed tau_a) is at pose (rel_pos, rel_heading)
# expressed in frame t, and makes a fresh prediction tau_b in frame-(t+g) coords.
# The two predictions describe overlapping wall-clock time and must agree on the
# overlap once tau_a is re-expressed in frame-(t+g). The jump on the overlap is
# the open-loop temporal-(in)stability signal.
# --------------------------------------------------------------------------
def _angle_of(traj: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj[..., 3], traj[..., 2])


def test_replan_consistency_perfect_straight_is_zero():
    from planner_metrics.replan_consistency import compute_replan_consistency_batch

    dt, v, g = 0.1, 5.0, 5
    Ta = 80
    traj_a = _straight(Ta, v=v, dt=dt)[None]  # (1, Ta, 4), along +x
    L = Ta - g
    # ego followed the plan: at step g it is at (g*v*dt, 0), heading 0
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    # perfect re-prediction of the same world line, in frame-(t+g): (i*v*dt, 0)
    t = torch.arange(L, dtype=torch.float32) * dt
    traj_b = torch.stack(
        [v * t, torch.zeros(L), torch.ones(L), torch.zeros(L)], dim=-1
    )[None]
    out = compute_replan_consistency_batch(traj_a, traj_b, g, rel_pos, rel_heading)
    assert out["overlap_len"] == L
    assert out["position_jump"].item() < 1e-4, out["position_jump"].item()
    assert out["heading_jump"].item() < 1e-4, out["heading_jump"].item()


def test_replan_consistency_lateral_offset_recovered():
    from planner_metrics.replan_consistency import compute_replan_consistency_batch

    dt, v, g, delta = 0.1, 5.0, 5, 0.7
    Ta = 80
    traj_a = _straight(Ta, v=v, dt=dt)[None]
    L = Ta - g
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    t = torch.arange(L, dtype=torch.float32) * dt
    # next-frame prediction shifted laterally by delta (planner changed its mind)
    traj_b = torch.stack(
        [v * t, torch.full((L,), delta), torch.ones(L), torch.zeros(L)], dim=-1
    )[None]
    out = compute_replan_consistency_batch(traj_a, traj_b, g, rel_pos, rel_heading)
    assert abs(out["position_jump"].item() - delta) < 1e-4, out["position_jump"].item()


def test_replan_consistency_heading_offset_recovered():
    from planner_metrics.replan_consistency import compute_replan_consistency_batch

    dt, v, g, dphi = 0.1, 5.0, 5, 0.2
    Ta = 80
    traj_a = _straight(Ta, v=v, dt=dt)[None]
    L = Ta - g
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    t = torch.arange(L, dtype=torch.float32) * dt
    # same positions, headings rotated by a constant dphi
    traj_b = torch.stack(
        [v * t, torch.zeros(L), torch.full((L,), math.cos(dphi)), torch.full((L,), math.sin(dphi))],
        dim=-1,
    )[None]
    out = compute_replan_consistency_batch(traj_a, traj_b, g, rel_pos, rel_heading)
    assert abs(out["heading_jump"].item() - dphi) < 1e-4, out["heading_jump"].item()


def test_replan_consistency_perfect_arc_is_zero():
    """Rotation test: a constant-curvature arc is self-similar, so if the ego
    follows it, the next-frame prediction in the new ego frame equals the
    original prediction's first L steps. The function must align the later,
    rotated+translated chunk of tau_a back onto tau_b and find ~0 jump."""
    from planner_metrics.replan_consistency import compute_replan_consistency_batch

    dt, R, v, g = 0.1, 20.0, 5.0, 5
    Ta = 80
    traj_a = _arc(Ta, R=R, v=v, dt=dt)[None]
    L = Ta - g
    rel_pos = traj_a[:, g, :2].clone()  # (1, 2)
    rel_heading = _angle_of(traj_a[:, g])  # (1,)
    traj_b = traj_a[:, :L].clone()  # self-similarity of the constant arc
    out = compute_replan_consistency_batch(traj_a, traj_b, g, rel_pos, rel_heading)
    assert out["overlap_len"] == L
    assert out["position_jump"].item() < 1e-3, out["position_jump"].item()
    assert out["heading_jump"].item() < 1e-3, out["heading_jump"].item()


def test_replan_consistency_overlap_len_is_min():
    from planner_metrics.replan_consistency import compute_replan_consistency_batch

    dt, v, g = 0.1, 5.0, 30
    Ta, Tb = 80, 40
    traj_a = _straight(Ta, v=v, dt=dt)[None]
    traj_b = _straight(Tb, v=v, dt=dt)[None]
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    out = compute_replan_consistency_batch(traj_a, traj_b, g, rel_pos, rel_heading)
    assert out["overlap_len"] == min(Ta - g, Tb)  # = 40 vs 50 -> 40


def test_replan_consistency_batched():
    from planner_metrics.replan_consistency import compute_replan_consistency_batch

    dt, v, g = 0.1, 5.0, 5
    Ta = 80
    L = Ta - g
    traj_a = torch.cat([_straight(Ta, v=v, dt=dt)[None], _straight(Ta, v=v, dt=dt)[None]], dim=0)
    rel_pos = torch.tensor([[g * v * dt, 0.0], [g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0, 0.0])
    t = torch.arange(L, dtype=torch.float32) * dt
    # row 0 perfect, row 1 offset by 1.0
    b0 = torch.stack([v * t, torch.zeros(L), torch.ones(L), torch.zeros(L)], dim=-1)
    b1 = torch.stack([v * t, torch.ones(L), torch.ones(L), torch.zeros(L)], dim=-1)
    traj_b = torch.stack([b0, b1], dim=0)
    out = compute_replan_consistency_batch(traj_a, traj_b, g, rel_pos, rel_heading)
    assert out["position_jump"].shape == (2,)
    assert out["position_jump"][0].item() < 1e-4
    assert abs(out["position_jump"][1].item() - 1.0) < 1e-4


# --------------------------------------------------------------------------
# temporal_consistency_loss (differentiable training loss; reduces flicker)
# --------------------------------------------------------------------------
def test_temporal_consistency_loss_zero_when_aligned():
    from planner_metrics.replan_consistency import temporal_consistency_loss

    dt, v, g = 0.1, 5.0, 5
    Ta = 80
    traj_a = _straight(Ta, v=v, dt=dt)[None]
    L = Ta - g
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    t = torch.arange(L, dtype=torch.float32) * dt
    traj_b = torch.stack([v * t, torch.zeros(L), torch.ones(L), torch.zeros(L)], dim=-1)[None]
    loss = temporal_consistency_loss(traj_a, traj_b, g, rel_pos, rel_heading)
    assert loss.item() < 1e-4


def test_temporal_consistency_loss_recovers_offset():
    from planner_metrics.replan_consistency import temporal_consistency_loss

    dt, v, g, delta = 0.1, 5.0, 5, 0.7
    Ta = 80
    traj_a = _straight(Ta, v=v, dt=dt)[None]
    L = Ta - g
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    t = torch.arange(L, dtype=torch.float32) * dt
    traj_b = torch.stack([v * t, torch.full((L,), delta), torch.ones(L), torch.zeros(L)], dim=-1)[None]
    loss = temporal_consistency_loss(traj_a, traj_b, g, rel_pos, rel_heading, w_heading=0.0)
    assert abs(loss.item() - delta) < 1e-4


def test_temporal_consistency_loss_differentiable_and_stopgrad():
    from planner_metrics.replan_consistency import temporal_consistency_loss

    dt, v, g = 0.1, 5.0, 5
    Ta = 80
    traj_a = _straight(Ta, v=v, dt=dt)[None].clone().requires_grad_(True)
    # perturb BEFORE requires_grad so traj_b stays a leaf (else .grad is None)
    traj_b = (_straight(Ta, v=v, dt=dt)[None] + 0.01).clone().requires_grad_(True)
    rel_pos = torch.tensor([[g * v * dt, 0.0]])
    rel_heading = torch.tensor([0.0])
    # stop_grad_a=True: gradient flows only into traj_b (the current frame), not the past anchor
    loss = temporal_consistency_loss(traj_a, traj_b, g, rel_pos, rel_heading, stop_grad_a=True)
    loss.backward()
    assert traj_b.grad is not None and traj_b.grad.abs().sum() > 0
    assert traj_a.grad is None or traj_a.grad.abs().sum() == 0
