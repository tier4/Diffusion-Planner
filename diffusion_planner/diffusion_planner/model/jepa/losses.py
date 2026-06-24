"""JEPA training losses (SAGE Stage-I and Stage-II), ported from refer/sage.

Stage-I: latent alignment (L2 on normalised embeddings to stop-grad EMA targets) +
VICReg variance/covariance anti-collapse (``main.tex`` §A VICReg, λ_var=1, λ_cov=0.1).
Stage-II: teacher-forced one-step L1 ``L_tf`` and action-usage hinge ``L_neg`` (permute
actions in-batch). The short-rollout term ``L_ro`` lives with the trainer (deferred —
needs data) since it is only used during Stage-II optimisation.

These are used only by the offline, annotation-free JEPA trainers; the planner sees only
the frozen modules via ``energy.compute_traj_energy``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "jepa_loss",
    "ac_teacher_forced_loss",
    "ac_rollout_loss",
    "ac_action_usage_hinge",
]


def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sim_coef: float = 1.0,
    var_coef: float = 1.0,
    cov_coef: float = 0.1,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SAGE Stage-I loss: sim (alignment) + VICReg variance + covariance.

    pred / target: [B, M, d] or [B, d]. ``target`` is stop-grad (EMA teacher output).
    Returns (total, {"sim", "var", "cov"}) with detached scalar parts for logging.
    """
    if pred.dim() == 3:
        B, M, d = pred.shape
        pred = pred.reshape(B * M, d)
        target = target.reshape(B * M, d)
    target = target.detach()

    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    sim = F.mse_loss(pred_n, targ_n)

    std = torch.sqrt(pred.var(dim=0) + eps)  # [d]
    var_term = F.relu(gamma - std).mean()

    x = pred - pred.mean(dim=0, keepdim=True)
    n = max(1, x.shape[0] - 1)
    cov = (x.T @ x) / n  # [d, d]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = off_diag.pow(2).mean()

    total = sim_coef * sim + var_coef * var_term + cov_coef * cov_term
    parts = {"sim": sim.detach(), "var": var_term.detach(), "cov": cov_term.detach()}
    return total, parts


def ac_teacher_forced_loss(predictor, z: torch.Tensor, a: torch.Tensor, ctx=None) -> torch.Tensor:
    """Teacher-forced one-step L1: ``L_tf = mean ||ẑ_{t+1} − z_{t+1}||₁``.

    z: [B, T, Dz], a: [B, T-1, Da], ctx: [B, ctx_dim] or None.
    """
    z_pred = predictor.forward_teacher(z, a, ctx=ctx)  # [B, T-1, Dz]
    return F.l1_loss(z_pred, z[:, 1:, :])


def ac_rollout_loss(predictor, z: torch.Tensor, a: torch.Tensor, horizon: int = 4, ctx=None) -> torch.Tensor:
    """Short-horizon rollout L1: ``L_ro = ||ẑ_{t+H} − z_{t+H}||₁`` (H = ``horizon``).

    z: [B, T, Dz], a: [B, T-1, Da], ctx: [B, ctx_dim] or None. Uses ``forward_rollout``.
    """
    z_hat = predictor.forward_rollout(z, a, horizon, ctx=ctx)  # [B, Dz]
    return F.l1_loss(z_hat, z[:, horizon, :])


def ac_action_usage_hinge(
    predictor,
    z: torch.Tensor,
    a: torch.Tensor,
    margin: float = 0.10,
    ctx=None,
) -> torch.Tensor:
    """Action-usage hinge ``L_neg = [m − E_neg]_+`` with batch-permuted actions.

    Permuting actions across the batch and requiring the prediction error to RISE
    forces the predictor to actually use actions; the hinge only fires when the error
    under mismatched actions stays below the margin (i.e. action-insensitivity). z:
    [B, T, Dz], a: [B, T-1, Da].
    """
    B = z.shape[0]
    perm = torch.randperm(B, device=z.device)
    a_neg = a[perm]
    z_pred_neg = predictor.forward_teacher(z, a_neg, ctx=ctx)
    neg_err = (z_pred_neg - z[:, 1:, :]).abs().mean()
    return F.relu(margin - neg_err)
