"""Tests for per-sample / scene-gated temporal_consistency_loss.

The synthesis run (coeff 0.5) cut flicker −34% but copied the prior on scene-change frames,
because the consistency loss fired uniformly. Scene-aware gating weights each paired sample's
consistency by w = exp(-gt_dev / tau): normal frames (gt_dev ~ 0 => w ~ 1) keep full
consistency; scene-change frames (large gt_dev => w ~ 0) drop it so GT wins. This needs
temporal_consistency_loss to expose a per-sample value and accept a sample_weight — tested here.
"""
import torch

from planner_metrics.replan_consistency import temporal_consistency_loss


def _mk(N=3, T=8):
    # straight-line plans (x = 0..T-1, heading 0). frame_b gets a DISTINCT per-sample y-offset
    # so the per-sample consistency losses are distinct (makes the weighting tests meaningful).
    a = torch.zeros(N, T, 4)
    a[..., 2] = 1.0  # cos=1, sin=0 => heading 0
    a[..., 0] = torch.arange(T, dtype=torch.float32)
    b = a.clone()
    b[..., 1] = b[..., 1] + torch.tensor([0.5, 1.5, 3.0])[:N].view(N, 1)
    rel_pos = torch.zeros(N, 2)
    rel_h = torch.zeros(N)
    return a, b, rel_pos, rel_h


def test_per_sample_shape_and_mean_matches_scalar():
    a, b, rp, rh = _mk()
    ps = temporal_consistency_loss(a, b, 3, rp, rh, per_sample=True)
    assert ps.shape == (3,), f"expected per-sample [3], got {tuple(ps.shape)}"
    assert len(torch.unique(ps)) == 3, "distinct inputs should give distinct per-sample losses"
    scalar = temporal_consistency_loss(a, b, 3, rp, rh)
    assert torch.allclose(ps.mean(), scalar, atol=1e-5), f"{ps.mean()} != {scalar}"


def test_uniform_weight_equals_plain_mean():
    a, b, rp, rh = _mk()
    scalar = temporal_consistency_loss(a, b, 3, rp, rh)
    weighted = temporal_consistency_loss(a, b, 3, rp, rh, sample_weight=torch.ones(3))
    assert torch.allclose(weighted, scalar, atol=1e-5), f"{weighted} != {scalar}"


def test_zero_weight_excludes_sample():
    a, b, rp, rh = _mk()
    ps = temporal_consistency_loss(a, b, 3, rp, rh, per_sample=True)
    # zero-weight the 3rd sample -> weighted mean == mean of the first two
    weighted = temporal_consistency_loss(
        a, b, 3, rp, rh, sample_weight=torch.tensor([1.0, 1.0, 0.0])
    )
    assert torch.allclose(weighted, ps[:2].mean(), atol=1e-5), f"{weighted} != {ps[:2].mean()}"


def test_weight_keeps_gradient_to_traj_b():
    # the gate must not break the gradient path into traj_b (the supervised frame_{t+g} plan)
    a, b, rp, rh = _mk()
    b = b.clone().requires_grad_(True)
    w = torch.tensor([1.0, 0.3, 0.0])
    loss = temporal_consistency_loss(a, b, 3, rp, rh, sample_weight=w)
    loss.backward()
    assert b.grad is not None
    # zero-weighted sample 2 gets no gradient; weighted samples do
    assert torch.count_nonzero(b.grad[0]) > 0
    assert torch.count_nonzero(b.grad[2]) == 0, "zero-weight sample must get zero gradient"
