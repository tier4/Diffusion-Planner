"""Action-conditioned latent predictor f_η (SAGE Stage-II).

Ported from refer/sage/jepa/utils.py (``ACTinyTransformer.forward_teacher``), bundle=2
([z_t, a_t]) only — our "state" is already the encoded latent, so the optional
state-token path is dropped. Block-causal Transformer over per-step [latent, action]
bundles with a latent-delta head ``ẑ_{t+1} = z_t + Δz``. Defaults: 2 layers / 4 heads /
hidden 256 (``tab:impl_ac_used``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ACLatentPredictor"]


class ACLatentPredictor(nn.Module):
    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        hidden: int = 256,
        layers: int = 2,
        nhead: int = 4,
        delta_pred: bool = True,
        dropout: float = 0.0,
        max_T: int = 1024,
        ctx_dim: int | None = None,
    ):
        super().__init__()
        assert hidden % nhead == 0, "hidden must be divisible by nhead"
        self.delta_pred = delta_pred
        self.bundle = 2  # [z_t, a_t]
        self.max_T = max_T

        self.z_in = nn.Linear(z_dim, hidden)
        self.a_in = nn.Linear(a_dim, hidden)
        # Scene-context conditioning (option c): context modulates the predictor as an
        # additive token bias, so the diffed latent stays pose-derived (action-sensitive)
        # while the transition is scene-aware. None = ego-only.
        self.ctx_in = nn.Linear(ctx_dim, hidden) if ctx_dim is not None else None
        # LN + action gain helps prevent the predictor ignoring actions.
        self.z_ln = nn.LayerNorm(hidden)
        self.a_ln = nn.LayerNorm(hidden)
        self.a_gain = nn.Parameter(torch.tensor(1.0))

        self.type_z = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_a = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.type_z, std=0.02)
        nn.init.trunc_normal_(self.type_a, std=0.02)

        self.time_pos = nn.Embedding(max_T, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=4 * hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        # norm_first=True makes the nested-tensor fast path inapplicable; disable it
        # explicitly to avoid the runtime warning (no behavioural change).
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, z_dim),
        )

    def _block_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # time-level future mask, expanded over the per-step bundle
        time_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        tok_mask = time_mask.repeat_interleave(self.bundle, dim=0).repeat_interleave(
            self.bundle, dim=1
        )
        return tok_mask.float() * -1e9  # [S, S]

    def _build_tokens(self, z: torch.Tensor, a: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        """z: [B, T, Dz], a: [B, T-1, Da], ctx: [B, ctx_dim] or None -> [B, T*bundle, H]."""
        B, T, _ = z.shape
        assert T <= self.max_T, f"T={T} exceeds max_T={self.max_T}"

        z_proj = self.z_ln(self.z_in(z))  # [B, T, H]
        a_pad = torch.zeros(B, 1, a.shape[-1], device=a.device, dtype=a.dtype)
        a_full = torch.cat([a, a_pad], dim=1)  # [B, T, Da]
        a_proj = self.a_ln(self.a_in(a_full) * self.a_gain)  # [B, T, H]

        time = torch.arange(T, device=z.device)
        time_emb = self.time_pos(time).view(1, T, -1)  # [1, T, H]

        z_tok = z_proj + self.type_z + time_emb
        a_tok = a_proj + self.type_a + time_emb
        if self.ctx_in is not None:
            if ctx is None:
                raise ValueError("predictor was built with ctx_dim; ctx must be provided")
            ctx_emb = self.ctx_in(ctx).unsqueeze(1)  # [B, 1, H] broadcast over time
            z_tok = z_tok + ctx_emb
            a_tok = a_tok + ctx_emb
        toks = torch.stack([z_tok, a_tok], dim=2)  # [B, T, 2, H]
        return toks.view(B, T * self.bundle, -1)  # [B, S, H]

    def forward_teacher(self, z: torch.Tensor, a: torch.Tensor,
                        ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Predict z_{t+1} for t=0..T-2 from teacher-forced latents/actions.

        z: [B, T, Dz], a: [B, T-1, Da], ctx: [B, ctx_dim] (if ctx-conditioned) -> [B, T-1, Dz].
        """
        B, T, _ = z.shape
        W = T - 1
        seq = self._build_tokens(z, a, ctx)
        attn_mask = self._block_causal_mask(T, seq.device)
        out = self.tr(seq, mask=attn_mask)  # [B, S, H]

        z_slots = (torch.arange(W, device=z.device) * self.bundle).long()  # [W]
        z_repr = out.gather(
            dim=1, index=z_slots.view(1, -1, 1).expand(B, -1, out.size(-1))
        )  # [B, W, H]
        dz = self.head(z_repr)  # [B, W, Dz]
        if self.delta_pred:
            return z[:, :W, :] + dz
        return dz

    def forward_rollout(self, z: torch.Tensor, a: torch.Tensor, horizon: int,
                        ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Autoregressive rollout from z_0 under actions a, returning ẑ at ``horizon``.

        Grad-enabled (the SAGE reference wraps this in no_grad, which silently zeroes the
        rollout loss; here it is a real training signal). Built with cat (no in-place) so
        autograd flows. z: [B, T, Dz], a: [B, T-1, Da]; returns [B, Dz].
        """
        T = z.shape[1]
        assert 1 <= horizon < T, "horizon must satisfy 1 <= horizon < T"
        roll = z[:, :1, :]  # [B, 1, Dz] — z_0 (ground truth)
        for t in range(horizon):
            # placeholder GT latent at position t+1 keeps the block-causal mask happy;
            # it cannot leak into the prediction of step t+1.
            seq = torch.cat([roll, z[:, t + 1 : t + 2, :]], dim=1)  # [B, t+2, Dz]
            z_pred = self.forward_teacher(seq, a[:, : t + 1, :], ctx=ctx)  # [B, t+1, Dz]
            roll = torch.cat([roll, z_pred[:, -1:, :]], dim=1)  # append predicted z_{t+1}
        return roll[:, horizon, :]
