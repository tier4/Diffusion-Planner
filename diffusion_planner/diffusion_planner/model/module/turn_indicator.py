import torch
from timm.layers import DropPath
from timm.models.layers import Mlp
from timm.models.mlp_mixer import MixerBlock
from torch import nn
from torch.nn import functional as F

from diffusion_planner.dimensions import *
from diffusion_planner.model.module.encoder import LaneEncoder

# This network keeps its own, deliberately small, token-type vocabulary so that
# it stays independent from the main diffusion Encoder (which owns a different,
# larger CLASS_TYPE_* set). Keeping them separate means changes here never shift
# the main encoder's positional-embedding dimensions or invalidate its
# checkpoints.
CLASS_TYPE_TURN_INDICATOR_HISTORY = 0
CLASS_TYPE_LANE = 1
CLASS_TYPE_ROUTE = 2
CLASS_TYPE_NUM = 3

# Raw turn-indicator states stored in ``turn_indicators`` (none / disable /
# enable-left / enable-right). Note this is distinct from
# ``TURN_INDICATOR_OUTPUT_DIM``, which additionally carries the "keep" class the
# network predicts.
TURN_INDICATOR_HISTORY_NUM_CLASSES = 4


def add_class_type(x, class_type):
    """
    Add class type to the input tensor.
    Args:
        x: Tensor of shape (B, T, D=4) where D=4 represents (x, y, cos, sin)
        class_type: Class type to add (int)
    Returns:
        x: Tensor with class type added at the end
    """
    B, T, D = x.shape
    assert D == 4, "Input tensor must have 4 features (x, y, cos, sin)"
    class_type_tensor = F.one_hot(
        torch.full((B, T), class_type, device=x.device, dtype=torch.long),
        num_classes=CLASS_TYPE_NUM,
    ).to(dtype=x.dtype)
    return torch.cat([x, class_type_tensor], dim=-1)


def make_baselink_pose(batch_size: int, device: torch.device):
    return torch.cat(
        [
            torch.zeros((batch_size, 1, 2), device=device),  # x, y
            torch.ones((batch_size, 1, 1), device=device),  # cos(yaw)
            torch.zeros((batch_size, 1, 1), device=device),  # sin(yaw)
        ],
        dim=-1,
    )  # (B, 1, 4)


def make_lane_pos(x: torch.Tensor, lane_len: int, class_type: int):
    """Build a per-element positional token from raw lane/route geometry.

    Mirrors ``LaneEncoder``'s positional logic but emits it in *this* network's
    token vocabulary, so the positions of the reused ``LaneEncoder`` (which are
    encoded in the main encoder's vocabulary) can be discarded.

    Args:
        x: (B, P, V, D) raw lane tensor. D layout starts with (x, y, dx, dy).
        lane_len: Number of points per element (V).
        class_type: This network's class-type id for the element.
    Returns:
        (B, P, 4 + CLASS_TYPE_NUM) positional tokens.
    """
    pos = x[:, :, int(lane_len / 2), :4].clone()  # x, y, dx, dy
    heading = torch.atan2(pos[..., 3], pos[..., 2])
    pos = torch.stack([pos[..., 0], pos[..., 1], torch.cos(heading), torch.sin(heading)], dim=-1)
    return add_class_type(pos, class_type)


class TrajectoryEncoder(nn.Module):
    """Encode an ego trajectory (B, T, D) into a single cross-attention query.

    Uses timm's standard ``MixerBlock`` at ``hidden_dim`` width throughout: a
    linear pose embedding, ``depth`` mixer blocks, a final norm and global
    average pooling over time.
    """

    def __init__(
        self,
        time_len: int,
        pose_dim: int,
        hidden_dim: int,
        drop_path_rate: float = 0.0,
        depth: int = 3,
    ):
        super(TrajectoryEncoder, self).__init__()

        self.stem = nn.Linear(pose_dim, hidden_dim)
        # Token mixing runs on the time axis (seq_len = time_len); channel
        # mixing runs at hidden_dim, so no width bottleneck / re-projection.
        self.blocks = nn.ModuleList(
            [MixerBlock(hidden_dim, time_len, drop_path=drop_path_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D) tensor of trajectory data, where D is the feature dimension.
        Returns:
            x: (B, 1, hidden_dim) query token.
        """
        x = self.stem(x)  # (B, T, hidden_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # global average pooling over time
        return torch.mean(x, dim=1, keepdim=True)  # (B, 1, hidden_dim)


class TurnIndicatorHistoryEncoder(nn.Module):
    """Encode the past turn-indicator states into a single fusion token."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = TURN_INDICATOR_HISTORY_NUM_CLASSES,
    ):
        super(TurnIndicatorHistoryEncoder, self).__init__()
        self.num_classes = num_classes
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T) tensor of turn indicator history, where each element is an
               integer representing the turn indicator class.
        Returns:
            x: (B, 1, hidden_dim) encoded turn indicator history token.
            mask: (B, 1) boolean mask (never masked).
            pos: (B, 1, 4 + CLASS_TYPE_NUM) positional token.
        """
        B = x.shape[0]

        x = F.one_hot(x, num_classes=self.num_classes).float()  # (B, T, num_classes)
        x = x.view(B, -1)  # (B, T * num_classes)
        x = self.mlp(x)  # (B, hidden_dim)
        x = x.unsqueeze(1)  # (B, 1, hidden_dim)

        mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
        pos = add_class_type(make_baselink_pose(B, x.device), CLASS_TYPE_TURN_INDICATOR_HISTORY)

        return x, mask, pos


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP block (query attends to key/value)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        mlp_ratio = 4.0

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout
        )

    def forward(self, q, kv, kv_mask):
        kv = self.norm_kv(kv)
        q = q + self.drop_path(
            self.attn(self.norm_q(q), kv, kv, key_padding_mask=kv_mask, need_weights=False)[0]
        )
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class CrossFusionEncoder(nn.Module):
    """Stack of cross-attention blocks fusing key/value context into the query."""

    def __init__(self, hidden_dim, num_heads, drop_path_rate, depth):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout=drop_path_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, q, kv, kv_mask):
        for b in self.blocks:
            q = b(q, kv, kv_mask)
        return self.norm(q)


class TurnIndicatorNetwork(nn.Module):
    """Standalone network predicting the turn-indicator logit.

    Independent from the diffusion Encoder/Decoder. The (future) ego trajectory
    is encoded into a query token which cross-attends to key/value tokens built
    from the past turn-indicator states, the lane map, and the route; the fused
    query then predicts a logit over ``TURN_INDICATOR_OUTPUT_DIM`` classes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mixer_depth: int,
        fusion_depth: int,
        drop_path_rate: float = 0.0,
    ):
        super(TurnIndicatorNetwork, self).__init__()

        self.trajectory_encoder = TrajectoryEncoder(
            time_len=OUTPUT_T,
            pose_dim=POSE_DIM,
            hidden_dim=hidden_dim,
            drop_path_rate=drop_path_rate,
            depth=mixer_depth,
        )
        self.turn_indicator_history_encoder = TurnIndicatorHistoryEncoder(
            input_dim=INPUT_T * TURN_INDICATOR_HISTORY_NUM_CLASSES,
            hidden_dim=hidden_dim,
            num_classes=TURN_INDICATOR_HISTORY_NUM_CLASSES,
        )
        # Reused for feature/mask extraction only; their positional outputs are
        # recomputed locally (make_lane_pos) so the class_type here just tags the
        # discarded internal positions.
        self.lane_encoder = LaneEncoder(
            POINTS_PER_LANELET,
            class_type=CLASS_TYPE_LANE,
            drop_path_rate=drop_path_rate,
            hidden_dim=hidden_dim,
            depth=mixer_depth,
            embed_dim=64,
        )
        self.route_encoder = LaneEncoder(
            POINTS_PER_LANELET,
            class_type=CLASS_TYPE_ROUTE,
            drop_path_rate=drop_path_rate,
            hidden_dim=hidden_dim,
            depth=mixer_depth,
            embed_dim=64,
        )

        # position embedding for key/value tokens encodes x, y, cos, sin, type
        self.pos_emb = nn.Linear(4 + CLASS_TYPE_NUM, hidden_dim)

        self.fusion = CrossFusionEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            depth=fusion_depth,
        )

        self.head = nn.Linear(hidden_dim, TURN_INDICATOR_OUTPUT_DIM)

        # Initialize layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.attribute_emb.weight, std=0.02)
        nn.init.normal_(self.route_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.route_encoder.attribute_emb.weight, std=0.02)

    def _encode_query(self, ego_trajectory: torch.Tensor) -> torch.Tensor:
        """Encode the ego trajectory into the cross-attention query.

        Args:
            ego_trajectory: (B, T, D) ego trajectory to condition on.
        Returns:
            query: (B, 1, hidden) query token.
        """
        return self.trajectory_encoder(ego_trajectory)

    def _encode_key_value(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the key/value context tokens from history, lanes and route.

        Args:
            inputs: Dict containing lanes / route / turn_indicators tensors.
        Returns:
            kv: (B, N, hidden) key/value tokens.
            kv_mask: (B, N) boolean padding mask (True = masked out).
            pos: (B, N, 4 + CLASS_TYPE_NUM) positional tokens.
        """
        lanes = inputs["lanes"]
        route = inputs["route_lanes"]

        # Drop the current step so the history matches INPUT_T.
        turn_indicator_history = inputs["turn_indicators"][:, :-1].long()  # (B, INPUT_T)

        enc_hist, mask_hist, pos_hist = self.turn_indicator_history_encoder(turn_indicator_history)
        enc_lane, mask_lane, _ = self.lane_encoder(
            lanes, inputs["lanes_speed_limit"], inputs["lanes_has_speed_limit"]
        )
        enc_route, mask_route, _ = self.route_encoder(
            route, inputs["route_lanes_speed_limit"], inputs["route_lanes_has_speed_limit"]
        )

        # LaneEncoder emits positions in the main encoder's vocabulary, so build
        # them locally in this network's vocabulary instead.
        pos_lane = make_lane_pos(lanes, POINTS_PER_LANELET, CLASS_TYPE_LANE)
        pos_route = make_lane_pos(route, POINTS_PER_LANELET, CLASS_TYPE_ROUTE)

        kv = torch.cat([enc_hist, enc_lane, enc_route], dim=1)  # (B, N, hidden)
        kv_mask = torch.cat([mask_hist, mask_lane, mask_route], dim=1)  # (B, N)
        pos = torch.cat([pos_hist, pos_lane, pos_route], dim=1)  # (B, N, 4 + CLASS_TYPE_NUM)

        return kv, kv_mask, pos

    def _add_positional_embedding(
        self, kv: torch.Tensor, kv_mask: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Add positional embeddings to key/value tokens, zeroed on masked ones.

        Args:
            kv: (B, N, hidden) key/value tokens.
            kv_mask: (B, N) boolean padding mask (True = masked out).
            pos: (B, N, 4 + CLASS_TYPE_NUM) positional tokens.
        Returns:
            kv: (B, N, hidden) key/value tokens with positional embeddings added.
        """
        B, N, _ = kv.shape

        pos_emb = self.pos_emb(pos.view(B * N, -1))
        pos_emb = torch.where(
            (~kv_mask.view(-1)).unsqueeze(-1),
            pos_emb,
            torch.zeros_like(pos_emb),
        )
        return kv + pos_emb.view(B, N, -1)

    def forward(self, ego_trajectory: torch.Tensor, inputs: dict[str, torch.Tensor]):
        """
        Args:
            ego_trajectory: (B, T, D) ego trajectory to condition on.
            inputs: Dict containing lanes / route / turn_indicators tensors.
        Returns:
            turn_indicator_logit: (B, TURN_INDICATOR_OUTPUT_DIM)
        """
        query = self._encode_query(ego_trajectory)  # (B, 1, hidden)

        kv, kv_mask, pos = self._encode_key_value(inputs)
        kv = self._add_positional_embedding(kv, kv_mask, pos)

        # The history token is always valid, so no key/value row is fully masked
        # (which would make attention produce NaNs).
        fused = self.fusion(query, kv, kv_mask)  # (B, 1, hidden)

        return self.head(fused.squeeze(1))
