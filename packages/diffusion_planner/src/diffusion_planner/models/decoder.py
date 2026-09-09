"""Flow-matching trajectory decoder with scene cross-attention."""

from __future__ import annotations

import math

import torch
from timm.layers.mlp import Mlp
from timm.models.mlp_mixer import MixerBlock
from torch import nn

from diffusion_planner.data.dimensions import (
    CONTROL_DIM,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar flow times with sinusoidal features and an MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.frequency_dim = hidden_dim
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            out_features=hidden_dim,
            act_layer=nn.SiLU,
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Encode flow times `(B,)` or `(B, 1)` into `(B, H)`."""
        time = time.reshape(-1)
        half_dim = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, device=time.device, dtype=time.dtype)
            / max(half_dim - 1, 1)
        )
        angles = time[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.frequency_dim:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.mlp(embedding)


class AdaptiveLayerNorm(nn.Module):
    """Apply LayerNorm modulated by a flow-time embedding."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Modulate values `(..., H)` using time features `(B, H)`."""
        scale, shift = self.modulation(time).chunk(2, dim=-1)
        while scale.ndim < values.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(values) * (1 + scale) + shift


class TrajectoryEncoder(nn.Module):
    """Encode one complete per-timestep sequence into a single token.

    Two callers share this and they do not agree on the width of a timestep, so
    `state_dim` has to be given rather than read off a constant:

    - `TrajectoryDecoder` encodes the flow state, which is control
      `(accel, curvature)`, so `state_dim` is `CONTROL_DIM`.
    - `TurnIndicatorDecoder` encodes an ego pose trajectory
      `(x, y, cos_yaw, sin_yaw)`, so `state_dim` is `TRAJECTORY_DIM`.

    Nothing here is specific to either: the sequence is projected, mixed over
    time, and pooled.
    """

    def __init__(
        self,
        hidden_dim: int,
        mixer_hidden_dim: int,
        depth: int,
        state_dim: int,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.trajectory_len = TRAJECTORY_LENGTH
        self.input_projection = nn.Linear(state_dim, mixer_hidden_dim)
        self.blocks = nn.ModuleList(
            MixerBlock(mixer_hidden_dim, TRAJECTORY_LENGTH, drop_path=drop_path_rate)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(mixer_hidden_dim)
        self.output_projection = nn.Linear(mixer_hidden_dim, hidden_dim)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode a trajectory `(B, T, D)` into one token `(B, H)`.

        Args:
            trajectory: Trajectory with shape `(B, T, D)`, where `D` is the
                encoder's `state_dim`.

        Returns:
            One token with shape `(B, H)`.
        """
        features = self.input_projection(trajectory)
        for block in self.blocks:
            features = block(features)
        return self.output_projection(self.norm(features).mean(dim=1))


class TrajectoryDecoderBlock(nn.Module):
    """Fuse the ego trajectory token with scene memory."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_norm = AdaptiveLayerNorm(hidden_dim)
        self.feedforward_norm = AdaptiveLayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = Mlp(
            in_features=hidden_dim,
            hidden_features=feedforward_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        scene: torch.Tensor,
        time: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the ego token `(B, 1, H)` using masked scene memory and flow time."""
        query = self.cross_norm(x, time)
        memory = self.scene_norm(scene)
        cross = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=scene_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(cross)
        return x + self.dropout(self.feedforward(self.feedforward_norm(x, time)))


class TrajectoryDecoder(nn.Module):
    """Decode the ego token into a clean control prediction.

    The decoder predicts the ego vehicle only. Control is defined by rolling a
    unicycle out from the agent's own pose, so a neighbor's `(accel, curvature)`
    in the ego frame has no meaning and there is no agent axis to carry it.
    Neighbors still reach the model as scene tokens through the encoder.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
        trajectory_encoder_depth: int = 2,
        trajectory_mixer_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.trajectory_len = TRAJECTORY_LENGTH
        self.output_dim = CONTROL_DIM
        self.trajectory_encoder = TrajectoryEncoder(
            hidden_dim=hidden_dim,
            depth=trajectory_encoder_depth,
            mixer_hidden_dim=trajectory_mixer_hidden_dim,
            state_dim=self.output_dim,
        )
        self.ego_pose_embedding = nn.Linear(TRAJECTORY_DIM, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            TrajectoryDecoderBlock(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = Mlp(
            in_features=hidden_dim,
            hidden_features=feedforward_dim,
            out_features=TRAJECTORY_LENGTH * self.output_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )
        nn.init.zeros_(self.output_projection.fc2.weight)
        if self.output_projection.fc2.bias is not None:
            nn.init.zeros_(self.output_projection.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        ego_pose: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the clean ego control from a noisy flow state.

        Args:
            x: Ego control sequence with shape `(B, T, CONTROL_DIM)`, where each
                state is `[accel, curvature]` and `T` equals `TRAJECTORY_LENGTH`.
                The whole sequence is encoded into one token before attention.
            scene: Scene tokens with shape `(B, S, H)`.
            scene_mask: Invalid-scene-token mask with shape `(B, S)`.
            ego_pose: Normalized current ego pose with shape `(B, 4)`, laid out
                as `[x, y, cos_yaw, sin_yaw]`.
            time: Flow times with shape `(B,)` or `(B, 1)`.

        Returns:
            Predicted clean control with shape `(B, T, CONTROL_DIM)`.
        """
        features = self.trajectory_encoder(x) + self.ego_pose_embedding(ego_pose)
        # One query token per scene: the decoder attends to the scene, never
        # between agents.
        features = features.unsqueeze(1)

        time_features = self.time_embedding(time.to(dtype=features.dtype))
        for block in self.blocks:
            features = block(features, scene, time_features, scene_mask)
        output = self.output_projection(self.output_norm(features[:, 0]))
        return output.reshape(-1, self.trajectory_len, self.output_dim)
