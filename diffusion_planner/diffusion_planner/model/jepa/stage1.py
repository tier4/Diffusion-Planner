"""Stage-I JEPA model: online encoder + EMA teacher + mask-token predictor g_φ.

Ported from refer/sage/jepa/utils.py (MaskedTokenPredictor, JEPAStateModel). g_φ is a
mask-token Transformer that maps the encoded context window to the EMA-teacher embeddings
of future target states at offsets k. Only used by the offline Stage-I trainer; after
training the EMA encoder is frozen and handed to Stage-II / the energy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_planner.model.jepa.encoder import TrajStateEncoder, update_ema

__all__ = ["MaskedTokenPredictor", "JEPAStage1Model"]


class MaskedTokenPredictor(nn.Module):
    """Transformer readout: (context latents [B,W,d], offsets ks [B,M]) -> preds [B,M,d]."""

    def __init__(self, d: int, nhead: int = 4, layers: int = 2, ff_mult: int = 4, max_pos: int = 4096):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff_mult * d, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.0,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers, enable_nested_tensor=False)
        self.pos = nn.Embedding(max_pos, d)
        self.k_embed = nn.Embedding(max_pos, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.k_embed.weight, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Linear(d * 2, d * 2), nn.GELU(), nn.Linear(d * 2, d)
        )
        self.max_pos = max_pos

    def forward(self, h_ctx: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
        B, W, d = h_ctx.shape
        M = ks.shape[1]
        device = h_ctx.device
        pos_ctx = torch.arange(W, device=device).unsqueeze(0).expand(B, W)
        h_ctx = h_ctx + self.pos(pos_ctx)
        pos_mask = torch.clamp((W - 1) + ks, max=self.max_pos - 1)
        mask_tok = self.mask_token.expand(B, M, d) + self.pos(pos_mask)
        seq = torch.cat([h_ctx, mask_tok], dim=1)  # [B, W+M, d]
        out = self.tr(seq)
        mask_out = out[:, W:, :]  # [B, M, d]
        k_emb = self.k_embed(ks)  # [B, M, d]
        return self.head(torch.cat([mask_out, k_emb], dim=-1))


class JEPAStage1Model(nn.Module):
    """Online encoder + EMA teacher + mask-token predictor (SAGE Stage-I)."""

    def __init__(self, state_dim: int, embed_dim: int = 256, enc_hidden: int = 512,
                 enc_layers: int = 3, pred_nhead: int = 4, pred_layers: int = 2):
        super().__init__()
        self.encoder = TrajStateEncoder(state_dim, embed_dim, hidden=enc_hidden, layers=enc_layers)
        self.encoder_ema = TrajStateEncoder(state_dim, embed_dim, hidden=enc_hidden, layers=enc_layers)
        self.predictor = MaskedTokenPredictor(embed_dim, nhead=pred_nhead, layers=pred_layers)
        update_ema(self.encoder_ema, self.encoder, decay=0.0)  # init teacher = online
        for p in self.encoder_ema.parameters():
            p.requires_grad_(False)

    def update_ema(self, decay: float) -> None:
        update_ema(self.encoder_ema, self.encoder, decay)

    def forward(self, ctx1, ctx2, targets, ks):
        # ctx*: [B,W,Ds], targets: [B,M,Ds], ks: [B,M]
        pred1 = self.predictor(self.encoder(ctx1), ks)  # [B,M,d]
        pred2 = self.predictor(self.encoder(ctx2), ks)
        with torch.no_grad():
            B, M, Ds = targets.shape
            targ = self.encoder_ema(targets.reshape(B * M, Ds)).view(B, M, -1)
        return pred1, pred2, targ
