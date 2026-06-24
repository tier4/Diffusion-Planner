"""SAGE latent-consistency energy E(τ), single-mode adaptation.

E(τ) = (1/K) Σ_{k=0..K-1} || f_η(z_k, a_k) − z_{k+1} ||₁,  z_k = e_θ̄(s_k)

Ported from refer/sage/pipelines/energy.py (``compute_energy_from_traj``), but NOT
wrapped in ``torch.no_grad``: for the training-time auxiliary loss (Use A) the gradient
must flow back through the predicted trajectory ``ego_traj`` / ``velocity`` into the
planner, while the encoder/predictor stay frozen (``requires_grad_(False)`` set by the
caller). For inference scoring/guidance the caller supplies the no-grad context.

State choice is flag-switchable (plan §6): pass ``ctx=None`` for ego-only, or a pooled
scene-context tensor ``[B, ctx_dim]`` to encode ``[ego_k ⊕ ctx]`` (penalises
scene-inconsistent motion).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["compute_traj_energy"]


def compute_traj_energy(
    encoder: nn.Module,
    predictor: nn.Module,
    ego_traj: torch.Tensor,
    velocity: torch.Tensor,
    ctx: torch.Tensor | None,
    K: int,
) -> torch.Tensor:
    """Per-trajectory latent-consistency energy.

    Args:
        encoder: frozen TrajStateEncoder (e_θ̄).
        predictor: frozen ACLatentPredictor (f_η).
        ego_traj: [B, T+1, state_pose_dim] ego state sequence s_0..s_T (≥ K+1 steps).
        velocity: [B, T, a_dim] (or ≥ K) per-step action a_k (the model's velocity
            output IS the action — no inverse-dynamics model needed).
        ctx: [B, ctx_dim] pooled scene context broadcast over time, or None (ego-only).
        K: prefix length (= 1 s = 10 steps at 10 Hz; paper sweet spot, K≥20 degrades).

    Returns:
        E: [B] non-negative energy (lower = more dynamically self-consistent).
    """
    K = int(K)
    assert K >= 1, "K must be >= 1"

    s = ego_traj[:, : K + 1, :]  # [B, K+1, pose_dim]
    a = velocity[:, :K, :]  # [B, K, a_dim]

    # Encoder stays pose-only (latent is pose-derived -> action-sensitive). Scene context
    # enters as predictor conditioning (option c), NOT concatenated into the state — a
    # constant-within-window context concatenated here collapses action-sensitivity.
    z = encoder(s)  # [B, K+1, Dz]  (grad flows via s -> ego_traj)
    z_pred = predictor.forward_teacher(z, a, ctx=ctx)  # [B, K, Dz]

    step_err = (z_pred - z[:, 1:, :]).abs().mean(dim=-1)  # [B, K]
    return step_err.mean(dim=-1)  # [B]
