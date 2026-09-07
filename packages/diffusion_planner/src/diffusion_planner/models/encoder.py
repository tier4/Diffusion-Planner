"""Composable MLP-Mixer encoders for vectorized planner inputs."""

from __future__ import annotations

import torch
from timm.models.mlp_mixer import MixerBlock
from torch import nn

from diffusion_planner.data.dimensions import (
    AGENT_LABEL_DIM,
    AGENT_POSE_DIM,
    AGENT_SHAPE_DIM,
    EGO_HISTORY_LENGTH,
    EGO_SHAPE_DIM,
    EGO_VELOCITY_INDEX,
    GOAL_POSE_DIM,
    INTERSECTION_AREA_LENGTH,
    LANE_GEOMETRY_DIM,
    LANE_LENGTH,
    LANE_TYPE_DIM,
    ROAD_BORDER_LENGTH,
    STOP_LINE_LENGTH,
    TRAFFIC_LIGHT_DIM,
    TRAFFIC_LIGHT_FUTURE_LENGTH,
    TRAFFIC_LIGHT_PAST_LENGTH,
)


def _element_embedding(hidden_dim: int) -> nn.Parameter:
    embedding = nn.Parameter(torch.empty(hidden_dim))
    nn.init.normal_(embedding, std=0.02)
    return embedding


class OneHotEncoder(nn.Module):
    """Encode one-hot vectors."""

    def __init__(self, num_classes: int, output_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.projection = nn.Linear(num_classes, output_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode one-hot vectors.

        Args:
            values: One-hot vectors with shape `(..., C)`.

        Returns:
            Encoded features with shape `(..., output_dim)`.
        """
        return self.projection(values)


class FloatVectorEncoder(nn.Module):
    """Encode continuous vectors."""

    def __init__(self, vector_dim: int, output_dim: int) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        self.projection = nn.Linear(vector_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode continuous vectors.

        Args:
            values: Continuous vectors with shape `(..., D)`.

        Returns:
            Encoded features with shape `(..., output_dim)`.
        """
        return self.projection(values)


class OneHotSequenceEncoder(nn.Module):
    """Encode one-hot sequences with an MLP-Mixer."""

    def __init__(
        self,
        sequence_len: int,
        num_classes: int,
        hidden_dim: int,
        mixer_hidden_dim: int,
        depth: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.num_classes = num_classes
        self.one_hot_encoder = OneHotEncoder(num_classes, mixer_hidden_dim)
        self.blocks = nn.ModuleList(
            MixerBlock(mixer_hidden_dim, sequence_len, drop_path=drop_path_rate)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(mixer_hidden_dim)
        self.output = nn.Linear(mixer_hidden_dim, hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode one-hot sequences.

        Args:
            values: One-hot sequences with shape `(..., T, C)`.

        Returns:
            Encoded features with shape `(..., hidden_dim)`.
        """
        prefix = values.shape[:-2]
        features = values.reshape(-1, self.sequence_len, self.num_classes)
        features = self.one_hot_encoder(features)
        for block in self.blocks:
            features = block(features)
        features = self.output(self.norm(features).mean(dim=1))
        return features.reshape(*prefix, -1)


class FloatVectorSequenceEncoder(nn.Module):
    """Encode continuous-vector sequences with an MLP-Mixer."""

    def __init__(
        self,
        sequence_len: int,
        vector_dim: int,
        hidden_dim: int,
        mixer_hidden_dim: int,
        depth: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.vector_dim = vector_dim
        self.vector_encoder = FloatVectorEncoder(vector_dim, mixer_hidden_dim)
        self.blocks = nn.ModuleList(
            MixerBlock(mixer_hidden_dim, sequence_len, drop_path=drop_path_rate)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(mixer_hidden_dim)
        self.output = nn.Linear(mixer_hidden_dim, hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode continuous-vector sequences.

        Args:
            values: Continuous-vector sequences with shape `(..., T, D)`.

        Returns:
            Encoded features with shape `(..., hidden_dim)`.
        """
        prefix = values.shape[:-2]
        features = values.reshape(-1, self.sequence_len, self.vector_dim)
        features = self.vector_encoder(features)
        for block in self.blocks:
            features = block(features)
        features = self.output(self.norm(features).mean(dim=1))
        return features.reshape(*prefix, -1)


def _mask_invalid(
    features: torch.Tensor, invalid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return features.masked_fill(invalid_mask.unsqueeze(-1), 0.0), invalid_mask


class IntersectionAreaEncoder(nn.Module):
    """Encode intersection-area boundaries."""

    def __init__(
        self,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        mixer_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.geometry_encoder = FloatVectorSequenceEncoder(
            INTERSECTION_AREA_LENGTH,
            2,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(
        self, intersection_area: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode intersection-area boundaries.

        Args:
            intersection_area: Boundary points with shape `(B, N, V, 2)`.

        Returns:
            Tokens with shape `(B, N, H)` and masks with shape `(B, N)`.
        """
        invalid_mask = intersection_area.abs().sum(dim=(-2, -1)) == 0
        features = self.geometry_encoder(intersection_area) + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class RoadBorderEncoder(nn.Module):
    """Encode road-border polylines."""

    def __init__(
        self,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        mixer_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.geometry_encoder = FloatVectorSequenceEncoder(
            ROAD_BORDER_LENGTH,
            2,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(self, road_borders: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode road-border polylines.

        Args:
            road_borders: Polyline points with shape `(B, N, V, 2)`.

        Returns:
            Tokens with shape `(B, N, H)` and masks with shape `(B, N)`.
        """
        invalid_mask = road_borders.abs().sum(dim=(-2, -1)) == 0
        features = self.geometry_encoder(road_borders) + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class StopLineEncoder(nn.Module):
    """Encode stop lines."""

    def __init__(
        self,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        mixer_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.geometry_encoder = FloatVectorSequenceEncoder(
            STOP_LINE_LENGTH,
            2,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(self, stop_lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode stop lines.

        Args:
            stop_lines: Line points with shape `(B, N, V, 2)`.

        Returns:
            Tokens with shape `(B, N, H)` and masks with shape `(B, N)`.
        """
        invalid_mask = stop_lines.abs().sum(dim=(-2, -1)) == 0
        features = self.geometry_encoder(stop_lines) + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class NeighborAgentEncoder(nn.Module):
    """Encode neighbor motion history, shape, and semantic label."""

    def __init__(
        self,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        mixer_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.history_encoder = FloatVectorSequenceEncoder(
            EGO_HISTORY_LENGTH,
            AGENT_POSE_DIM,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.shape_encoder = FloatVectorEncoder(AGENT_SHAPE_DIM, hidden_dim)
        self.label_encoder = OneHotEncoder(AGENT_LABEL_DIM, hidden_dim)
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(
        self,
        neighbor_agents_past: torch.Tensor,
        agent_shape: torch.Tensor,
        agent_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode neighbor agents.

        Args:
            neighbor_agents_past: Pose histories with shape `(B, N, T, 4)`.
            agent_shape: Width and length with shape `(B, N, 2)`.
            agent_label: Class one-hot vectors with shape `(B, N, 4)`.

        Returns:
            Tokens with shape `(B, N, H)` and masks with shape `(B, N)`.
        """
        invalid_mask = neighbor_agents_past.abs().sum(dim=(-2, -1)) == 0
        features = self.history_encoder(neighbor_agents_past)
        features = features + self.shape_encoder(agent_shape)
        features = features + self.label_encoder(agent_label)
        features = features + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class LaneEncoder(nn.Module):
    """Encode lane geometry and attributes."""

    def __init__(
        self,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        mixer_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.geometry_encoder = FloatVectorSequenceEncoder(
            LANE_LENGTH,
            LANE_GEOMETRY_DIM,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.lane_type_encoder = OneHotEncoder(LANE_TYPE_DIM, hidden_dim)
        self.past_traffic_encoder = OneHotSequenceEncoder(
            TRAFFIC_LIGHT_PAST_LENGTH,
            TRAFFIC_LIGHT_DIM,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.future_traffic_encoder = OneHotSequenceEncoder(
            TRAFFIC_LIGHT_FUTURE_LENGTH,
            TRAFFIC_LIGHT_DIM,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.speed_limit_encoder = FloatVectorEncoder(1, hidden_dim)
        self.unknown_speed_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.element_embedding = _element_embedding(hidden_dim)
        nn.init.normal_(self.unknown_speed_embedding, std=0.02)

    def forward(
        self,
        lanes: torch.Tensor,
        lane_types: torch.Tensor,
        speed_limits: torch.Tensor,
        traffic_light_past: torch.Tensor,
        traffic_light_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode lanes.

        Args:
            lanes: Lane geometry with shape `(B, N, V, 6)`.
            lane_types: Boundary type vectors with shape `(B, N, 20)`.
            speed_limits: Speed limits with shape `(B, N, 1)`.
            traffic_light_past: Past states with shape `(B, N, T_past, 6)`.
            traffic_light_future: Future states with shape `(B, N, T_future, 6)`.

        Returns:
            Tokens with shape `(B, N, H)` and masks with shape `(B, N)`.
        """
        invalid_mask = lanes.abs().sum(dim=(-2, -1)) == 0
        features = self.geometry_encoder(lanes)
        features = features + self.lane_type_encoder(lane_types)
        features = features + self.past_traffic_encoder(traffic_light_past)
        features = features + self.future_traffic_encoder(traffic_light_future)
        features = features + self.element_embedding

        known_speed = speed_limits > 0
        speed_features = self.speed_limit_encoder(speed_limits)
        unknown_speed = self.unknown_speed_embedding.expand_as(speed_features)
        features = features + torch.where(known_speed, speed_features, unknown_speed)
        return _mask_invalid(features, invalid_mask)


class EgoHistoryEncoder(nn.Module):
    """Encode ego stop history and current velocity."""

    def __init__(
        self,
        velocity_threshold: float,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        mixer_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.velocity_threshold = velocity_threshold
        self.stop_history_encoder = OneHotSequenceEncoder(
            EGO_HISTORY_LENGTH,
            2,
            hidden_dim,
            mixer_hidden_dim,
            depth,
            drop_path_rate,
        )
        self.current_velocity_encoder = FloatVectorEncoder(1, hidden_dim)
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(
        self, ego_agent_past: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ego stop history and current velocity.

        Args:
            ego_agent_past: Ego history with shape `(B, T, 6)`.

        Returns:
            Tokens with shape `(B, H)` and masks with shape `(B,)`.
        """
        invalid_mask = torch.zeros(
            ego_agent_past.shape[0], dtype=torch.bool, device=ego_agent_past.device
        )
        velocity_history = ego_agent_past[..., EGO_VELOCITY_INDEX]
        stopped = velocity_history <= self.velocity_threshold
        stop_history = torch.stack((~stopped, stopped), dim=-1).to(ego_agent_past.dtype)
        current_velocity = velocity_history[:, -1].unsqueeze(-1)
        features = self.stop_history_encoder(stop_history)
        features = features + self.current_velocity_encoder(current_velocity)
        features = features + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class GoalPoseEncoder(nn.Module):
    """Encode goal position and orientation."""

    def __init__(
        self,
        hidden_dim: int,
        max_distance: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.goal_pose_encoder = FloatVectorEncoder(GOAL_POSE_DIM, hidden_dim)
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(self, goal_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode goal poses.

        Args:
            goal_pose: Normalized goal pose vectors with shape `(B, 4)`.

        Returns:
            Tokens with shape `(B, H)` and masks with shape `(B,)`.
        """
        distance = torch.linalg.vector_norm(goal_pose[..., :2], dim=-1)
        invalid_mask = distance > self.max_distance
        features = self.goal_pose_encoder(goal_pose) + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class EgoShapeEncoder(nn.Module):
    """Encode ego vehicle dimensions."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.ego_shape_encoder = FloatVectorEncoder(EGO_SHAPE_DIM, hidden_dim)
        self.element_embedding = _element_embedding(hidden_dim)

    def forward(self, ego_shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ego vehicle dimensions.

        Args:
            ego_shape: Vehicle dimensions with shape `(B, 3)`.

        Returns:
            Tokens with shape `(B, H)` and masks with shape `(B,)`.
        """
        invalid_mask = torch.zeros(
            ego_shape.shape[0], dtype=torch.bool, device=ego_shape.device
        )
        features = self.ego_shape_encoder(ego_shape) + self.element_embedding
        return _mask_invalid(features, invalid_mask)


class FusionEncoder(nn.Module):
    """Fuse scene tokens with padding-aware self-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        dropout: float = 0.0,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Fuse scene tokens with self-attention.

        Args:
            tokens: Scene tokens with shape `(B, N, H)`.
            padding_mask: Invalid-token mask with shape `(B, N)`.

        Returns:
            Fused tokens with shape `(B, N, H)`.
        """
        features = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return features.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SceneEncoder(nn.Module):
    """Tokenize and fuse an input-data map."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        fusion_depth: int,
        encoder_depth: int,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        mixer_hidden_dim: int = 128,
        velocity_threshold: float = 0.1,
        goal_max_distance: float = 2.0,
    ) -> None:
        super().__init__()
        self.neighbor_agent_encoder = NeighborAgentEncoder(
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.lane_encoder = LaneEncoder(
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.route_lane_encoder = LaneEncoder(
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.intersection_area_encoder = IntersectionAreaEncoder(
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.stop_line_encoder = StopLineEncoder(
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.road_border_encoder = RoadBorderEncoder(
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.ego_history_encoder = EgoHistoryEncoder(
            velocity_threshold,
            drop_path_rate,
            hidden_dim,
            encoder_depth,
            mixer_hidden_dim,
        )
        self.goal_pose_encoder = GoalPoseEncoder(hidden_dim, goal_max_distance)
        self.ego_shape_encoder = EgoShapeEncoder(hidden_dim)
        self.fusion_encoder = FusionEncoder(
            hidden_dim, num_heads, fusion_depth, dropout
        )

    def forward(
        self, input_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batched planner input map.

        Args:
            input_data: Normalized planner tensors keyed by their schema names.

        Returns:
            Fused scene tokens with shape `(B, N, H)` and padding masks with
            shape `(B, N)`.
        """
        neighbor_tokens, neighbor_mask = self.neighbor_agent_encoder(
            input_data["neighbor_agents_past"],
            input_data["agent_shape"],
            input_data["agent_label"],
        )
        lane_tokens, lane_mask = self.lane_encoder(
            input_data["lanes"],
            input_data["lane_types"],
            input_data["lanes_speed_limit"],
            input_data["lane_traffic_light_past"],
            input_data["lane_traffic_light_future"],
        )
        route_tokens, route_mask = self.route_lane_encoder(
            input_data["route_lanes"],
            input_data["route_lane_types"],
            input_data["route_lanes_speed_limit"],
            input_data["route_traffic_light_past"],
            input_data["route_traffic_light_future"],
        )

        intersection_tokens, intersection_mask = self.intersection_area_encoder(
            input_data["intersection_area"]
        )
        stop_line_tokens, stop_line_mask = self.stop_line_encoder(
            input_data["stop_lines"]
        )
        road_border_tokens, road_border_mask = self.road_border_encoder(
            input_data["road_borders"]
        )

        ego_history_token, ego_history_mask = self.ego_history_encoder(
            input_data["ego_agent_past"]
        )
        goal_pose_token, goal_pose_mask = self.goal_pose_encoder(
            input_data["goal_pose"]
        )
        ego_shape_token, ego_shape_mask = self.ego_shape_encoder(
            input_data["ego_shape"]
        )
        singleton_tokens = torch.stack(
            (ego_history_token, goal_pose_token, ego_shape_token), dim=1
        )
        singleton_mask = torch.stack(
            (ego_history_mask, goal_pose_mask, ego_shape_mask), dim=1
        )

        masks = (
            neighbor_mask,
            lane_mask,
            route_mask,
            intersection_mask,
            stop_line_mask,
            road_border_mask,
        )
        tokens = torch.cat(
            (
                neighbor_tokens,
                lane_tokens,
                route_tokens,
                intersection_tokens,
                stop_line_tokens,
                road_border_tokens,
                singleton_tokens,
            ),
            dim=1,
        )
        padding_mask = torch.cat((*masks, singleton_mask), dim=1)
        return self.fusion_encoder(tokens, padding_mask), padding_mask
