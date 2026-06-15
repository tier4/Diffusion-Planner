"""Reference-Scene Fusion via Cross-Attention.

ref_token [B, H] queries the frozen scene encoding [B, N, D_enc] via
multi-head cross-attention, producing a scene-aware representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefFusionAttention(nn.Module):
    """Cross-attention: ref_token queries frozen scene encoding.

    Architecture:
        Q = Linear(ref_token)       [B, 1, H]
        K = Linear(scene_encoding)  [B, N, H]
        V = Linear(scene_encoding)  [B, N, H]
        out = softmax(QK^T / sqrt(d)) V  -> [B, 1, H]
        return ref_token + out_proj(out)  (residual)
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(encoder_dim, hidden_dim)
        self.v_proj = nn.Linear(encoder_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(encoder_dim)

    def forward(
        self,
        ref_token: torch.Tensor,
        scene_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ref_token: [B, H] pooled reference trajectory representation.
            scene_encoding: [B, N, D_enc] frozen encoder output.

        Returns:
            [B, H] ref_token enriched with scene context (residual).
        """
        B = ref_token.shape[0]

        q = self.norm_q(ref_token).unsqueeze(1)  # [B, 1, H]
        kv = self.norm_kv(scene_encoding)  # [B, N, D_enc]

        q = self.q_proj(q)  # [B, 1, H]
        k = self.k_proj(kv)  # [B, N, H]
        v = self.v_proj(kv)  # [B, N, H]

        # Reshape for multi-head attention
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, 1, d]
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, h, 1, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, h, 1, d]
        out = out.transpose(1, 2).reshape(B, 1, -1)  # [B, 1, H]
        out = self.out_proj(out).squeeze(1)  # [B, H]

        return ref_token + out
