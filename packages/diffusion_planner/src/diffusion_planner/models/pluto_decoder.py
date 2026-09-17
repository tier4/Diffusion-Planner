"""PLUTO-style one-shot planning decoder on top of ``SceneEncoder`` tokens.

Ported from PLUTO (Cheng et al. 2024, https://github.com/jchengai/pluto,
``src/models/pluto/modules/planning_decoder.py``) and adapted to the tier4
scene tokens:

- lateral queries come from the preferred route (one reference line built from
  the route-lane tokens) instead of PLUTO's reference-line PointNet,
- every masked operation uses ``masked_fill`` instead of boolean indexing so the
  graph stays static for ONNX / TensorRT,
- the output keeps the planner contract ``[x, y, cos_yaw, sin_yaw]``.
"""

from __future__ import annotations

import torch
from torch import nn

from diffusion_planner.data.dimensions import (
    MAX_NUM_NEIGHBORS,
    NUM_INTERSECTION_AREAS,
    NUM_LANE_SEGMENTS,
    NUM_ROAD_BORDERS,
    NUM_ROUTE_SEGMENTS,
    NUM_STOP_LINES,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)

# Token layout produced by ``SceneEncoder.forward``.
ROUTE_TOKEN_START = MAX_NUM_NEIGHBORS + NUM_LANE_SEGMENTS
ROUTE_TOKEN_END = ROUTE_TOKEN_START + NUM_ROUTE_SEGMENTS
NUM_SCENE_TOKENS = (
    ROUTE_TOKEN_END + NUM_INTERSECTION_AREAS + NUM_STOP_LINES + NUM_ROAD_BORDERS + 3
)
EGO_TOKEN_INDEX = NUM_SCENE_TOKENS - 3  # ego-history singleton token
# Logit assigned to candidates of padded reference lines (PLUTO uses -1e6).
PADDED_LOGIT = -1e6


def mlp_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """PLUTO ``MLPLayer``: Linear, LayerNorm, ReLU, Linear."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def zero_init_last_linear(head: nn.Sequential) -> None:
    """Start a head at zero so every candidate begins as the normalized-space mean."""
    last = head[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


class PreferredRouteQuery(nn.Module):
    """One reference-line query (R = 1) from the preferred-route tokens.

    The route tokens are consecutive lanelets of the preferred lane in route
    order, so their masked mean plus the start pose of the first valid segment
    describes the path the ego is expected to follow.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.start_pose_embedding = nn.Linear(TRAJECTORY_DIM, hidden_dim)

    def forward(
        self,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        route_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the reference-line query.

        Args:
            scene: Fused scene tokens ``(B, N, H)``, zero where masked.
            scene_mask: Invalid-token mask ``(B, N)``.
            route_lanes: Normalized route geometry ``(B, S, P, 6)``.

        Returns:
            Queries ``(B, 1, H)`` and a padding mask ``(B, 1)`` that is True when
            no route segment is valid.
        """
        tokens = scene[:, ROUTE_TOKEN_START:ROUTE_TOKEN_END]
        invalid = scene_mask[:, ROUTE_TOKEN_START:ROUTE_TOKEN_END]
        valid = (~invalid).to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        batch, _, points, channels = route_lanes.shape
        first_valid = torch.argmax((~invalid).to(route_lanes.dtype), dim=1)
        index = first_valid.view(batch, 1, 1, 1).expand(-1, 1, points, channels)
        segment = route_lanes.gather(1, index).squeeze(1)  # (B, P, 6)
        start_xy = segment[:, 0, :2]
        direction = segment[:, 1, :2] - segment[:, 0, :2]
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        start_pose = torch.cat((start_xy, direction), dim=-1)

        no_route = invalid.all(dim=1, keepdim=True)  # (B, 1)
        query = pooled + self.start_pose_embedding(start_pose).to(pooled.dtype)
        query = query.masked_fill(no_route, 0.0)
        return query.unsqueeze(1), no_route


class PlutoDecoderLayer(nn.Module):
    """PLUTO ``DecoderLayer``: line-to-line, mode-to-mode, scene cross-attention, FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.line_norm = nn.LayerNorm(hidden_dim)
        self.line_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mode_norm = nn.LayerNorm(hidden_dim)
        self.mode_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        line_padding: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        mode_position: torch.Tensor,
    ) -> torch.Tensor:
        """Refine ``(B, R, M, H)`` queries with ``(B, R)`` line padding."""
        batch, num_lines, num_modes, hidden = queries.shape

        if num_lines > 1:
            # Attention over an all-masked row is NaN: keep line 0 attendable and
            # mask its logit later instead.
            all_padded = line_padding.all(dim=1, keepdim=True)
            safe_padding = torch.cat(
                (line_padding[:, :1] & ~all_padded, line_padding[:, 1:]), dim=1
            )
            x = queries.transpose(1, 2).reshape(batch * num_modes, num_lines, hidden)
            y = self.line_norm(x)
            attended = self.line_attention(
                y,
                y,
                y,
                key_padding_mask=safe_padding.repeat_interleave(num_modes, dim=0),
                need_weights=False,
            )[0]
            x = x + self.dropout(attended)
            queries = x.reshape(batch, num_modes, num_lines, hidden).transpose(1, 2)

        x = queries.reshape(batch * num_lines, num_modes, hidden)
        y = self.mode_norm(x)
        y_positioned = y + mode_position
        attended = self.mode_attention(
            y_positioned, y_positioned, y, need_weights=False
        )[0]
        x = x + self.dropout(attended)
        x = x.masked_fill(line_padding.reshape(batch * num_lines, 1, 1), 0.0)

        x = x.reshape(batch, num_lines * num_modes, hidden)
        y = self.cross_norm(x)
        attended = self.cross_attention(
            y, scene, scene, key_padding_mask=scene_mask, need_weights=False
        )[0]
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feedforward(self.feedforward_norm(x)))
        return x.reshape(batch, num_lines, num_modes, hidden)


class PlutoDecoder(nn.Module):
    """Decode ``R x M`` ego candidates and their logits from reference-line queries."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        num_modes: int = 12,
        cat_ego_token: bool = False,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.cat_ego_token = cat_ego_token
        self.mode_embedding = nn.Parameter(torch.empty(1, 1, num_modes, hidden_dim))
        self.mode_position = nn.Parameter(torch.empty(1, num_modes, hidden_dim))
        self.query_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            PlutoDecoderLayer(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(depth)
        )
        if cat_ego_token:
            self.ego_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.location_head = mlp_head(hidden_dim, 2 * hidden_dim, TRAJECTORY_LENGTH * 2)
        self.yaw_head = mlp_head(hidden_dim, 2 * hidden_dim, TRAJECTORY_LENGTH * 2)
        self.probability_head = mlp_head(hidden_dim, hidden_dim, 1)
        nn.init.normal_(self.mode_embedding, std=0.01)
        nn.init.normal_(self.mode_position, std=0.01)
        zero_init_last_linear(self.location_head)
        zero_init_last_linear(self.yaw_head)

    def forward(
        self,
        line_queries: torch.Tensor,
        line_padding: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        ego_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidates ``(B, R, M, T, 4)`` and logits ``(B, R, M)``.

        Args:
            line_queries: Reference-line embeddings ``(B, R, H)``.
            line_padding: True for padded reference lines ``(B, R)``.
            scene: Scene tokens ``(B, S, H)``.
            scene_mask: Invalid scene-token mask ``(B, S)``.
            ego_token: Optional ego token ``(B, H)`` concatenated before the heads.
        """
        batch, num_lines, _ = line_queries.shape
        lines = line_queries.unsqueeze(2).expand(-1, -1, self.num_modes, -1)
        modes = self.mode_embedding.expand(batch, num_lines, -1, -1)
        queries = self.query_projection(torch.cat((lines, modes), dim=-1))
        for block in self.blocks:
            queries = block(
                queries, line_padding, scene, scene_mask, self.mode_position
            )
        if self.cat_ego_token:
            if ego_token is None:
                raise ValueError("cat_ego_token requires an ego token")
            ego = ego_token[:, None, None, :].expand(-1, num_lines, self.num_modes, -1)
            queries = self.ego_projection(torch.cat((queries, ego), dim=-1))
        shape = (batch, num_lines, self.num_modes, TRAJECTORY_LENGTH, 2)
        location = self.location_head(queries).reshape(shape)
        yaw = self.yaw_head(queries).reshape(shape)
        candidates = torch.cat((location, yaw), dim=-1)
        logits = self.probability_head(queries).squeeze(-1)
        return candidates, logits


class NeighborPredictor(nn.Module):
    """PLUTO ``AgentPredictor`` without the velocity head: MLP heads on neighbor tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.location_head = mlp_head(hidden_dim, 2 * hidden_dim, TRAJECTORY_LENGTH * 2)
        self.yaw_head = mlp_head(hidden_dim, 2 * hidden_dim, TRAJECTORY_LENGTH * 2)
        zero_init_last_linear(self.location_head)
        zero_init_last_linear(self.yaw_head)

    def forward(self, scene: torch.Tensor, scene_mask: torch.Tensor) -> torch.Tensor:
        """Predict neighbor futures ``(B, A, T, 4)`` from the neighbor tokens."""
        tokens = scene[:, :MAX_NUM_NEIGHBORS]
        invalid = scene_mask[:, :MAX_NUM_NEIGHBORS]
        batch, agents, _ = tokens.shape
        shape = (batch, agents, TRAJECTORY_LENGTH, 2)
        location = self.location_head(tokens).reshape(shape)
        yaw = self.yaw_head(tokens).reshape(shape)
        prediction = torch.cat((location, yaw), dim=-1)
        return prediction.masked_fill(invalid[:, :, None, None], 0.0)


def select_candidate(
    candidates: torch.Tensor,
    logits: torch.Tensor,
    line_padding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the highest-logit candidate; padded lines never win.

    Returns the trajectory ``(B, T, 4)`` and the flat ``R * M`` index ``(B,)``.
    """
    batch, num_lines, num_modes, steps, dim = candidates.shape
    masked = logits.masked_fill(line_padding.unsqueeze(-1), PADDED_LOGIT)
    index = masked.reshape(batch, num_lines * num_modes).argmax(dim=-1)
    flat = candidates.reshape(batch, num_lines * num_modes, steps, dim)
    gather_index = index.view(batch, 1, 1, 1).expand(-1, 1, steps, dim)
    return flat.gather(1, gather_index).squeeze(1), index
