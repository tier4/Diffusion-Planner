"""Road border repulsion guidance for the diffusion planner.

Repels the ego trajectory away from road border line_strings using 80
perimeter sample points around the ego rectangle (20 per side), matching
the approach used in the reward function for accurate ego-edge distance.

Line string data layout (v4 C++ binary / NPZ format):
  line_strings shape: (B, NUM_LINE_STRINGS=60, POINTS_PER_LINE_STRING=20, 4)
    Channel 0: X coordinate (ego-centric metres)
    Channel 1: Y coordinate (ego-centric metres)
    Channel 2: one-hot stop_line flag
    Channel 3: one-hot road_border flag (1.0 = road border)
"""

import torch

from .base import BaseGuidance
from .registry import register

_ROAD_BORDER_TYPE_IDX = 3
_MIN_LINE_STRING_DIM = 4

# Distance thresholds (metres) from ego EDGE to road border.
_DIST_HARD = 0.25  # within 25cm of edge: full penalty
_DIST_SOFT = 0.60  # within 60cm: decaying penalty

_MAX_BORDER_DIST = 30.0
_PTS_PER_SIDE = 20  # 20 per side × 4 sides = 80 perimeter points


def _build_ego_perimeter(ego_shape_params):
    """Build 80 local-frame perimeter points for the ego rectangle.

    Args:
        ego_shape_params: tuple (wheel_base, length, width) in metres.

    Returns:
        (80, 2) tensor of local-frame (x, y) points.
    """
    wb, length, width = ego_shape_params
    ro = (length - wb) / 2
    half_w = width / 2
    pts = []
    for j in range(_PTS_PER_SIDE):
        f = j / (_PTS_PER_SIDE - 1)
        pts.append((-ro + f * length, -half_w))  # bottom edge
        pts.append((-ro + f * length, half_w))  # top edge
        pts.append((-ro, -half_w + f * width))  # left edge
        pts.append((length - ro, -half_w + f * width))  # right edge
    return torch.tensor(pts, dtype=torch.float32)  # (80, 2)


@register
class RoadBorderGuidance(BaseGuidance):
    """Repulsive guidance using ego perimeter sampling against road borders.

    Samples 80 points around the ego rectangle at each trajectory timestep,
    computes minimum distance from each perimeter point to the nearest road
    border point, and applies a smooth penalty that increases sharply as the
    ego edge approaches the border.

    _energy_scale = 5.0 so that scale=1 in the GUI produces meaningful repulsion.
    The raw reward sums over 80 timesteps with penalty in [0,1], giving values
    up to -80. With _energy_scale=5.0 and scale=1, effective energy is up to -400.
    """

    name = "road_border"
    _energy_scale = 5.0

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units.

        Returns [B] unscaled reward (higher = farther from road borders).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device

        ego_traj = x[:, 0, 1:, :]  # [B, T, 4] — skip t=0 prefix

        line_strings = inputs["line_strings"]  # [B, N_ls, N_pts, D]
        if line_strings.shape[-1] < _MIN_LINE_STRING_DIM:
            return torch.zeros(B, device=device)

        # Extract road border points
        border_flag = line_strings[..., _ROAD_BORDER_TYPE_IDX]  # [B, N_ls, N_pts]
        border_xy = line_strings[..., :2]  # [B, N_ls, N_pts, 2]
        is_border = border_flag > 0.5
        has_coords = border_xy.norm(dim=-1) > 1e-3
        valid = is_border & has_coords

        N_flat = line_strings.shape[1] * line_strings.shape[2]
        border_pts = border_xy.reshape(B, N_flat, 2)  # [B, K, 2]
        valid_flat = valid.reshape(B, N_flat)  # [B, K]

        if not valid_flat.any():
            return torch.zeros(B, device=device)

        # Build ego perimeter in local frame (80 points)
        # Use default ego dimensions — guidance doesn't have access to ego_shape
        # in the same way as reward, but 4.34×1.70 is the standard vehicle.
        local_pts = _build_ego_perimeter((2.75, 4.34, 1.70)).to(device)  # (80, 2)
        N_perim = local_pts.shape[0]

        # Transform perimeter to world frame at each timestep
        cos_h = ego_traj[..., 2]  # [B, T]
        sin_h = ego_traj[..., 3]  # [B, T]
        h_norm = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
        cos_h = cos_h / h_norm
        sin_h = sin_h / h_norm

        # Rotation matrix: [B, T, 2, 2]
        rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(B, T, 2, 2)
        # Rotated perimeter: [B, T, 80, 2]
        rotated = torch.einsum("btij,kj->btki", rot, local_pts)
        world_pts = ego_traj[..., :2].unsqueeze(2) + rotated  # [B, T, 80, 2]

        # Min distance from each perimeter point to nearest border point
        # world_pts: [B, T, 80, 2], border_pts: [B, K, 2]
        # Reshape for cdist: [B, T*80, 2] vs [B, K, 2]
        world_flat = world_pts.reshape(B, T * N_perim, 2)
        dists = torch.cdist(world_flat, border_pts)  # [B, T*80, K]

        # Mask invalid border points
        dists = dists.masked_fill(~valid_flat.unsqueeze(1).expand(-1, T * N_perim, -1), 1e6)

        # Min dist per perimeter point
        min_per_point = dists.min(dim=-1).values  # [B, T*80]
        min_per_point = min_per_point.reshape(B, T, N_perim)

        # Min dist per timestep = min across all perimeter points
        min_dist = min_per_point.min(dim=-1).values  # [B, T]

        # Smooth two-zone penalty on ego EDGE distance
        range_width = _DIST_SOFT - _DIST_HARD
        t_norm = ((min_dist - _DIST_HARD) / range_width).clamp(0.0, 1.0)
        penalty = (1.0 - t_norm) ** 2  # [B, T]

        too_far = min_dist > _MAX_BORDER_DIST
        penalty = penalty.masked_fill(too_far, 0.0)

        reward = -penalty.sum(dim=-1)  # [B]
        return reward


def road_border_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Deprecated. Use RoadBorderGuidance via GuidanceComposer."""
    from .config import GuidanceConfig

    fn = RoadBorderGuidance(GuidanceConfig(name="road_border"))
    return fn.energy(x, t, inputs)
