"""Lane keeping guidance for the diffusion planner.

Penalises ego trajectory points where the vehicle footprint protrudes beyond
the boundaries of the nearest lane segment.  The vehicle width is read from
``inputs["ego_shape"]`` so that the guidance respects the actual vehicle
dimensions rather than a hardcoded constant.

Lane segment data layout (SEGMENT_POINT_DIM = 33, indices from dimensions.py):
  0-1 : centerline X, Y
  2-3 : direction  dX, dY
  4-5 : left boundary  LB_X, LB_Y
  6-7 : right boundary RB_X, RB_Y
"""

import torch
import torch.nn.functional as F

from .base import BaseGuidance
from .registry import register

# Indices into the lane feature vector (must match dimensions.py)
_X, _Y = 0, 1
_DX, _DY = 2, 3
_LBX, _LBY = 4, 5
_RBX, _RBY = 6, 7

# Only penalise violations above this threshold (metres).
_MARGIN = 0.0

# Ignore the nearest lane if farther than this distance from the ego position.
_MAX_LANE_DIST = 30.0


@register
class LaneKeepingGuidance(BaseGuidance):
    """Penalises ego footprint protrusion beyond lane boundaries.

    Finds the nearest lane segment point for each future ego position and
    computes how far the vehicle extends beyond the left and right boundaries.

    _energy_scale = 0.05 keeps the correction comparable to route-following
    guidance at default settings.
    """

    name = "lane_keeping"
    _energy_scale = 0.05

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units.

        Returns [B] unscaled reward (higher = fewer lane violations).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1

        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

        lanes = inputs["lanes"]  # [B, N_seg, N_pts, 33]
        N = lanes.shape[1] * lanes.shape[2]

        lane_centers = lanes[..., _X:_Y + 1].reshape(B, N, 2)
        lane_dirs    = lanes[..., _DX:_DY + 1].reshape(B, N, 2)
        lane_left    = lanes[..., _LBX:_LBY + 1].reshape(B, N, 2)
        lane_right   = lanes[..., _RBX:_RBY + 1].reshape(B, N, 2)

        lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
        lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # [B, N, 2]

        lane_valid = (lane_left.norm(dim=-1) + lane_right.norm(dim=-1)) > 1e-3  # [B, N]

        dist = (ego_pos.unsqueeze(2) - lane_centers.unsqueeze(1)).norm(dim=-1)  # [B, T, N]
        dist = dist.masked_fill(~lane_valid.unsqueeze(1).expand(-1, T, -1), 1e6)
        nearest = dist.argmin(dim=-1)      # [B, T]
        min_dist = dist.min(dim=-1).values  # [B, T]

        def gather2(tensor):
            idx = nearest.unsqueeze(-1).expand(-1, -1, 2)
            return tensor.unsqueeze(1).expand(-1, T, -1, -1) \
                         .gather(2, idx.unsqueeze(2)).squeeze(2)

        c   = gather2(lane_centers)
        lat = gather2(lane_lat)
        lb  = gather2(lane_left)   # offset vectors from centerline, not absolute positions
        rb  = gather2(lane_right)  # offset vectors from centerline, not absolute positions

        ego_lat  = ((ego_pos - c) * lat).sum(dim=-1)    # [B, T]
        left_hw  = (lb * lat).sum(dim=-1)                 # [B, T]
        right_hw = (rb * lat).sum(dim=-1)                 # [B, T]

        half_w = inputs["ego_shape"][:, 2:3] / 2         # [B, 1]

        viol_left  = F.relu(ego_lat + half_w - left_hw  + _MARGIN)
        viol_right = F.relu(right_hw - ego_lat + half_w + _MARGIN)

        no_lane = min_dist > _MAX_LANE_DIST
        viol_left  = viol_left.masked_fill(no_lane, 0.0)
        viol_right = viol_right.masked_fill(no_lane, 0.0)

        reward = -(viol_left + viol_right).sum(dim=-1)  # [B]
        return reward


# ---------------------------------------------------------------------------
# Backward-compatible module-level function alias
# ---------------------------------------------------------------------------

def lane_keeping_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Deprecated. Use LaneKeepingGuidance via GuidanceComposer."""
    from .config import GuidanceConfig
    fn = LaneKeepingGuidance(GuidanceConfig(name="lane_keeping"))
    return fn.energy(x, t, inputs)
