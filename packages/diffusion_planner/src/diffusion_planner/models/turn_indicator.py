"""Turn-indicator prediction from scene tokens and the current indicator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import TrajectoryEncoder
from .encoder import OneHotEncoder


class TurnIndicatorDecoder(nn.Module):
    """Predict the next indicator state from a scene and its current state."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        trajectory_encoder_depth: int = 2,
        trajectory_mixer_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.turn_indicator_encoder = OneHotEncoder(4, hidden_dim)
        self.trajectory_encoder = TrajectoryEncoder(
            hidden_dim=hidden_dim,
            depth=trajectory_encoder_depth,
            mixer_hidden_dim=trajectory_mixer_hidden_dim,
        )
        self.trajectory_scene_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        turn_indicator: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Return DISABLE/LEFT/RIGHT logits.

        Args:
            scene: Scene tokens with shape ``(B, N, H)``.
            scene_mask: Invalid scene-token mask with shape ``(B, N)``.
            turn_indicator: Current report with shape ``(B,)``. Values 0, 1,
                2, and 3 represent missing, disabled, left, and right.
            trajectory: Normalized ego trajectory with shape ``(B, T, 4)``.

        Returns:
            Next-state logits with shape ``(B, 3)``.
        """
        current = turn_indicator.to(torch.long).clamp(0, 3)
        current_one_hot = F.one_hot(current, num_classes=4).to(scene.dtype)
        current_token = self.turn_indicator_encoder(current_one_hot).unsqueeze(1)
        trajectory_token = self.trajectory_encoder(trajectory.unsqueeze(1))
        trajectory_context, _ = self.trajectory_scene_attention(
            trajectory_token,
            scene,
            scene,
            key_padding_mask=scene_mask,
            need_weights=False,
        )
        trajectory_token = trajectory_token + trajectory_context
        query = self.fusion_norm(current_token + trajectory_token)
        attended, _ = self.cross_attention(
            query,
            scene,
            scene,
            key_padding_mask=scene_mask,
            need_weights=False,
        )
        token = query + attended
        token = token + self.mlp(self.norm(token))
        return self.classifier(self.output_norm(token[:, 0]))
