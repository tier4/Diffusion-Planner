"""Tests for JEPAEnergy — the frozen energy used by the training-time aux loss (Use A)."""

from __future__ import annotations

import torch

from diffusion_planner.model.jepa.encoder import TrajStateEncoder
from diffusion_planner.model.jepa.predictor import ACLatentPredictor


def _zeroed_pair(embed=8):
    enc = TrajStateEncoder(state_dim=4, embed_dim=embed, hidden=16, layers=3)
    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
    pred = ACLatentPredictor(z_dim=embed, a_dim=4, hidden=16, layers=2, nhead=2, delta_pred=True)
    with torch.no_grad():
        last = [m for m in pred.head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_(); last.bias.zero_()
    return enc, pred


def test_energy_zero_for_constant_latent_identity_predictor():
    from diffusion_planner.model.jepa.energy_module import JEPAEnergy

    enc, pred = _zeroed_pair()
    je = JEPAEnergy(enc, pred, pose_mean=torch.zeros(4), pose_std=torch.ones(4), prefix_K=10)
    ego_current = torch.randn(3, 4)
    ego_pred_world = torch.randn(3, 80, 4)
    e = je.energy(ego_current, ego_pred_world)
    assert e.shape == (3,)
    assert torch.allclose(e, torch.zeros(3), atol=1e-5)


def test_energy_grad_flows_to_traj_not_frozen_params():
    from diffusion_planner.model.jepa.energy_module import JEPAEnergy

    enc = TrajStateEncoder(state_dim=4, embed_dim=8, hidden=16, layers=3)
    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2)
    je = JEPAEnergy(enc, pred, pose_mean=torch.zeros(4), pose_std=torch.ones(4), prefix_K=10)
    ego_current = torch.randn(2, 4)
    ego_pred_world = torch.randn(2, 80, 4, requires_grad=True)
    je.energy(ego_current, ego_pred_world).sum().backward()
    assert ego_pred_world.grad is not None and ego_pred_world.grad.abs().sum() > 0
    for p in je.parameters():
        assert p.grad is None  # frozen


def test_energy_uses_only_prefix_K():
    """Energy must depend only on the first K predicted steps; perturbing steps
    beyond K leaves it unchanged."""
    from diffusion_planner.model.jepa.energy_module import JEPAEnergy

    enc = TrajStateEncoder(state_dim=4, embed_dim=8, hidden=16, layers=3)
    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2)
    je = JEPAEnergy(enc, pred, pose_mean=torch.zeros(4), pose_std=torch.ones(4), prefix_K=10)
    ego_current = torch.randn(2, 4)
    ep = torch.randn(2, 80, 4)
    e1 = je.energy(ego_current, ep)
    ep2 = ep.clone()
    ep2[:, 20:, :] += 7.0  # perturb only beyond the K=10 prefix
    e2 = je.energy(ego_current, ep2)
    assert torch.allclose(e1, e2, atol=1e-6)


def test_from_ckpts_loads_trained_egoonly():
    import os

    import pytest

    from diffusion_planner.model.jepa.energy_module import JEPAEnergy

    enc_ck = "/mnt/nvme/Diffusion-Planner/checkpoints/jepa_egoonly/jepa_encoder_ema.pt"
    pred_ck = "/mnt/nvme/Diffusion-Planner/checkpoints/jepa_egoonly/jepa_predictor.pt"
    if not (os.path.isfile(enc_ck) and os.path.isfile(pred_ck)):
        pytest.skip("trained ego-only JEPA ckpts not present")
    je = JEPAEnergy.from_ckpts(enc_ck, pred_ck, device="cpu", prefix_K=10)
    e = je.energy(torch.randn(4, 4), torch.randn(4, 80, 4))
    assert e.shape == (4,) and torch.isfinite(e).all() and (e >= 0).all()
