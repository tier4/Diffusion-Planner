"""Synthetic-tensor unit tests for the frozen JEPA energy module (SAGE-JEPA plan §1).

Ported/adapted from refer/sage/jepa/utils.py to our 4-d ego pose + velocity action +
optional scene context. No data / no training here — these pin module behaviour
(shapes, frozen-grad routing, delta-prediction, loss properties, energy ranking).

Run (diffusion_planner package needs only torch for these modules):

    PYTHONPATH=/mnt/nvme/Diffusion-Planner/diffusion_planner \
        /mnt/nvme/OnePlanner/.venv/bin/python -m pytest \
        diffusion_planner/diffusion_planner/model/jepa/test_jepa_modules.py -q
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# TrajStateEncoder + EMA
# --------------------------------------------------------------------------
def test_encoder_maps_last_dim_to_embed():
    from diffusion_planner.model.jepa.encoder import TrajStateEncoder

    enc = TrajStateEncoder(state_dim=4, embed_dim=256, hidden=512, layers=3)
    # arbitrary leading dims: [B, T, state_dim] -> [B, T, embed_dim]
    z = enc(torch.randn(2, 11, 4))
    assert z.shape == (2, 11, 256)
    # also a flat [B, state_dim]
    assert enc(torch.randn(5, 4)).shape == (5, 256)
    assert torch.isfinite(z).all()


def test_encoder_context_concat_state_dim():
    from diffusion_planner.model.jepa.encoder import TrajStateEncoder

    # ego(4) + pooled scene context(256) = 260-d state (flag-switchable design)
    enc = TrajStateEncoder(state_dim=4 + 256, embed_dim=256)
    s = torch.randn(3, 11, 260)
    assert enc(s).shape == (3, 11, 256)


def test_update_ema_decay_extremes():
    from diffusion_planner.model.jepa.encoder import TrajStateEncoder, update_ema

    online = TrajStateEncoder(state_dim=4)
    teacher = TrajStateEncoder(state_dim=4)
    # randomise online so it differs from teacher
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p))

    # decay=0 -> teacher copies online exactly
    update_ema(teacher, online, decay=0.0)
    for pt, po in zip(teacher.parameters(), online.parameters()):
        assert torch.allclose(pt, po)

    # decay=1 -> teacher unchanged by a further online change
    snapshot = [p.clone() for p in teacher.parameters()]
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    update_ema(teacher, online, decay=1.0)
    for pt, snap in zip(teacher.parameters(), snapshot):
        assert torch.allclose(pt, snap)


# --------------------------------------------------------------------------
# ACLatentPredictor (block-causal, latent-delta)
# --------------------------------------------------------------------------
def test_predictor_teacher_forced_shape():
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=256, a_dim=4, hidden=256, layers=2, nhead=4)
    z = torch.randn(2, 11, 256)  # T = 11 latents
    a = torch.randn(2, 10, 4)  # T-1 actions
    out = pred.forward_teacher(z, a)  # predict z_{1..T-1}: [B, T-1, z_dim]
    assert out.shape == (2, 10, 256)


def test_predictor_delta_identity_when_head_zeroed():
    """With latent-delta prediction and a zeroed output head, the predictor
    returns z_t unchanged (z_{t+1} = z_t + 0). Pins the delta parameterisation."""
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, delta_pred=True)
    with torch.no_grad():
        # zero the final linear of the head so delta == 0
        last = [m for m in pred.head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_()
        last.bias.zero_()
    z = torch.randn(2, 6, 8)
    a = torch.randn(2, 5, 4)
    out = pred.forward_teacher(z, a)
    assert torch.allclose(out, z[:, :5, :], atol=1e-6)


def test_predictor_block_causal_no_future_leak():
    """Output for step t must not depend on actions/latents after t (block-causal):
    changing a future action leaves earlier predictions unchanged."""
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    torch.manual_seed(0)
    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2)
    pred.eval()
    z = torch.randn(1, 8, 8)
    a = torch.randn(1, 7, 4)
    out1 = pred.forward_teacher(z, a)
    a2 = a.clone()
    a2[:, -1, :] += 5.0  # perturb only the LAST action
    out2 = pred.forward_teacher(z, a2)
    # earlier predictions (before the last step) must be unchanged
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)


def test_predictor_rollout_shape_and_grad():
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2)
    z = torch.randn(2, 11, 8, requires_grad=True)
    a = torch.randn(2, 10, 4)
    zh = pred.forward_rollout(z, a, horizon=4)  # [B, Dz] at step `horizon`
    assert zh.shape == (2, 8)
    zh.sum().backward()  # rollout is grad-enabled (unlike SAGE's no_grad variant)
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_rollout_identity_predictor_constant_latent_is_zero_loss():
    from diffusion_planner.model.jepa.losses import ac_rollout_loss
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, delta_pred=True)
    with torch.no_grad():
        last = [m for m in pred.head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_()
        last.bias.zero_()
    z = torch.zeros(3, 8, 8)  # constant latent sequence
    a = torch.randn(3, 7, 4)
    loss = ac_rollout_loss(pred, z, a, horizon=4)
    assert loss.item() < 1e-6


# --------------------------------------------------------------------------
# compute_traj_energy (frozen enc + pred; grad flows through the trajectory)
# --------------------------------------------------------------------------
def _frozen_pair(z_dim=8, a_dim=4):
    from diffusion_planner.model.jepa.encoder import TrajStateEncoder
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    enc = TrajStateEncoder(state_dim=4, embed_dim=z_dim, hidden=16, layers=3).eval()
    pred = ACLatentPredictor(z_dim=z_dim, a_dim=a_dim, hidden=16, layers=2, nhead=2).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in pred.parameters():
        p.requires_grad_(False)
    return enc, pred


def test_energy_shape_and_nonneg():
    from diffusion_planner.model.jepa.energy import compute_traj_energy

    enc, pred = _frozen_pair()
    ego = torch.randn(3, 21, 4)  # [B, T+1, 4]
    vel = torch.randn(3, 21, 4)  # per-step action (velocity)
    e = compute_traj_energy(enc, pred, ego, vel, ctx=None, K=10)
    assert e.shape == (3,)
    assert (e >= 0).all()


def test_energy_grad_flows_through_traj_not_params():
    """The whole point of Use A: energy is differentiable w.r.t. the predicted
    trajectory, while encoder/predictor stay frozen (no param grad)."""
    from diffusion_planner.model.jepa.energy import compute_traj_energy

    enc, pred = _frozen_pair()
    ego = torch.randn(2, 21, 4, requires_grad=True)
    vel = torch.randn(2, 21, 4, requires_grad=True)
    e = compute_traj_energy(enc, pred, ego, vel, ctx=None, K=10)
    e.sum().backward()
    assert ego.grad is not None and torch.isfinite(ego.grad).all()
    assert ego.grad.abs().sum() > 0
    # frozen modules accumulate no gradient
    for p in list(enc.parameters()) + list(pred.parameters()):
        assert p.grad is None


def test_energy_zero_for_constant_latent_identity_predictor():
    """If the predictor is delta=identity (head zeroed) and the latent sequence
    is constant, every one-step residual is 0 -> energy 0. Pins the energy formula
    E = mean_k ||f(z_k,a_k) - z_{k+1}||_1."""
    from diffusion_planner.model.jepa.encoder import TrajStateEncoder
    from diffusion_planner.model.jepa.energy import compute_traj_energy
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    # encoder that maps everything to the same latent: zero all weights, so
    # enc(s) = bias (constant) for every step -> constant latent sequence.
    enc = TrajStateEncoder(state_dim=4, embed_dim=8, hidden=16, layers=3).eval()
    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
        # LayerNorm of a constant vector is well-defined; bias makes latent constant
    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, delta_pred=True).eval()
    with torch.no_grad():
        last = [m for m in pred.head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_()
        last.bias.zero_()
    for p in list(enc.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)

    ego = torch.randn(2, 21, 4)
    vel = torch.randn(2, 21, 4)
    e = compute_traj_energy(enc, pred, ego, vel, ctx=None, K=10)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-5), e


def test_energy_context_conditions_predictor_not_encoder():
    """Scene context enters as PREDICTOR conditioning (option c): the encoder stays
    pose-only (state_dim=4) and ctx is a separate input to f_η. Verifies the energy
    runs with a pose-only encoder + ctx-conditioned predictor, and that the energy is
    ctx-sensitive (different context -> different energy)."""
    from diffusion_planner.model.jepa.encoder import TrajStateEncoder
    from diffusion_planner.model.jepa.energy import compute_traj_energy
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    enc = TrajStateEncoder(state_dim=4, embed_dim=8, hidden=16, layers=3).eval()  # pose-only
    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, ctx_dim=16).eval()
    for p in list(enc.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)
    ego = torch.randn(2, 21, 4)
    vel = torch.randn(2, 21, 4)
    ctx = torch.randn(2, 16)
    e = compute_traj_energy(enc, pred, ego, vel, ctx=ctx, K=10)
    assert e.shape == (2,)
    # ctx-sensitivity: a different context yields a different energy
    e2 = compute_traj_energy(enc, pred, ego, vel, ctx=ctx + 3.0, K=10)
    assert not torch.allclose(e, e2)


def test_predictor_context_conditioning_changes_output():
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, ctx_dim=12).eval()
    z = torch.randn(2, 6, 8)
    a = torch.randn(2, 5, 4)
    ctx_a = torch.randn(2, 12)
    out_a = pred.forward_teacher(z, a, ctx=ctx_a)
    out_b = pred.forward_teacher(z, a, ctx=ctx_a + 5.0)
    assert out_a.shape == (2, 5, 8)
    assert not torch.allclose(out_a, out_b)  # context conditioning actually conditions


# --------------------------------------------------------------------------
# losses: Stage-I JEPA (sim + VICReg) and Stage-II AC (tf + rollout + hinge)
# --------------------------------------------------------------------------
def test_jepa_loss_low_when_aligned():
    from diffusion_planner.model.jepa.losses import jepa_loss

    torch.manual_seed(0)
    target = torch.randn(64, 16)
    # pred == target (after stop-grad on target) -> sim term ~0
    pred = target.clone().requires_grad_(True)
    total, parts = jepa_loss(pred, target)
    assert parts["sim"] < 1e-5


def test_jepa_loss_variance_penalizes_collapse():
    from diffusion_planner.model.jepa.losses import jepa_loss

    # collapsed predictions (all equal) -> high variance penalty vs spread-out
    collapsed = torch.zeros(64, 16, requires_grad=True)
    spread = torch.randn(64, 16, requires_grad=True)
    _, p_collapsed = jepa_loss(collapsed, torch.randn(64, 16))
    _, p_spread = jepa_loss(spread, torch.randn(64, 16))
    assert p_collapsed["var"] > p_spread["var"]


def test_ac_action_usage_hinge_fires_on_action_insensitive():
    """The hinge penalises a predictor whose error stays low under permuted
    actions (action-insensitive). A predictor that ignores actions -> hinge > 0."""
    from diffusion_planner.model.jepa.losses import ac_action_usage_hinge
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, delta_pred=True)
    with torch.no_grad():  # zero head -> predictor ignores actions entirely
        last = [m for m in pred.head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_()
        last.bias.zero_()
    # constant latent sequence: a head-zeroed (delta=0) predictor is BOTH accurate
    # (predicts z_t == z_{t+1}) AND action-blind -> neg error stays ~0 under permuted
    # actions -> hinge = margin - 0 > 0. (On unpredictable latents the hinge wouldn't
    # fire, which is correct: the hinge only flags accurate-yet-action-insensitive.)
    z = torch.zeros(8, 6, 8)
    a = torch.randn(8, 5, 4)
    hinge = ac_action_usage_hinge(pred, z, a, margin=0.10)
    assert hinge.item() > 0.0


def test_ac_teacher_forced_loss_zero_on_self_consistent():
    from diffusion_planner.model.jepa.losses import ac_teacher_forced_loss
    from diffusion_planner.model.jepa.predictor import ACLatentPredictor

    pred = ACLatentPredictor(z_dim=8, a_dim=4, hidden=16, layers=2, nhead=2, delta_pred=True)
    with torch.no_grad():
        last = [m for m in pred.head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_()
        last.bias.zero_()
    # constant latent sequence + identity(delta=0) predictor -> tf target met exactly
    z = torch.zeros(4, 6, 8)
    a = torch.randn(4, 5, 4)
    loss = ac_teacher_forced_loss(pred, z, a)
    assert loss.item() < 1e-6
