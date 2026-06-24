"""JEPA state encoder (SAGE Stage-I representation) + EMA update.

Ported from refer/sage/jepa/utils.py (``MLP`` / ``Encoder``), adapted to our state
choice: ego pose (4-d) optionally concatenated with pooled scene context. The encoder
maps a per-step state vector to a latent; it operates on the last dim, so it accepts
any leading shape ([B, T, state_dim] -> [B, T, embed_dim]).

This is the only learned representation SAGE keeps frozen at planner train/infer time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["TrajStateEncoder", "update_ema"]


class TrajStateEncoder(nn.Module):
    """MLP encoder ``state_dim -> hidden -> ... -> embed_dim`` (GELU + LayerNorm).

    With the SAGE defaults (hidden=512, layers=3) this is ``state_dim → 512 → 512 →
    256``, matching ``tab:impl_jepa_used`` and the planner ``hidden_dim=256`` (= d_z).
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int = 256,
        hidden: int = 512,
        layers: int = 3,
    ):
        super().__init__()
        dims = [state_dim] + [hidden] * (layers - 1) + [embed_dim]
        mods: list[nn.Module] = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.LayerNorm(dims[i + 1])]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    """EMA teacher update: ``target ← decay·target + (1-decay)·source`` (in place).

    decay=0 copies ``source`` onto ``target``; decay=1 leaves ``target`` unchanged.
    Mirrors SAGE ``JEPAStateModel.update_ema`` (cosine 0.99→0.9999 schedule applied
    by the trainer).
    """
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)
    for bt, bs in zip(target.buffers(), source.buffers()):
        bt.data.copy_(bs.data)
