"""Reference Trajectory Mixer — compresses x_ref [B, T, 4] into a single token.

Uses the same MLP-Mixer architecture as diffusion_planner.model.module.mixer:
alternating token-mixing and channel-mixing MLPs with pre-norm residuals.
"""

import torch
import torch.nn as nn
from timm.models.layers import Mlp


class _MixerBlock(nn.Module):
    """Pre-norm residual MLP-Mixer block.

    Identical pattern to diffusion_planner.model.module.mixer.MixerBlock.
    """

    def __init__(self, tokens_dim: int, channels_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_dim, hidden_features=tokens_dim,
            act_layer=nn.GELU, drop=dropout,
        )
        self.norm2 = nn.LayerNorm(channels_dim)
        self.channels_mlp = Mlp(
            in_features=channels_dim, hidden_features=channels_dim,
            act_layer=nn.GELU, drop=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        y = self.norm1(x)
        y = y.permute(0, 2, 1)  # [B, C, T]
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)  # [B, T, C]
        x = x + y

        # Channel mixing
        y = self.norm2(x)
        return x + self.channels_mlp(y)


class RefTrajectoryMixer(nn.Module):
    """MLP-Mixer that compresses a reference trajectory [B, T, 4] into [B, H].

    Architecture:
        Linear(4 -> H) -> [MixerBlock] x L -> LayerNorm -> mean-pool
    """

    def __init__(self, seq_len: int, hidden_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(4, hidden_dim)

        self.layers = nn.ModuleList([
            _MixerBlock(seq_len, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ref: [B, T, 4] reference trajectory in (x, y, cos, sin) format.

        Returns:
            [B, H] pooled representation.
        """
        x = self.input_proj(x_ref)  # [B, T, H]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x.mean(dim=1)  # [B, H]
