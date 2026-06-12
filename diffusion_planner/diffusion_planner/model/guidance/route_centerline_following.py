"""Route-centerline-following guidance for the diffusion planner.

Identical to ``centerline_following.py`` except it uses the planned
``route_lanes`` tensor instead of all ``lanes``. This matches
``rlvr.reward.compute_centerline_score_batch`` which scores against
route_lanes, so guidance and ranking are aligned.

For each future ego position:
  1. Find the nearest point among all valid route-lane polyline points.
  2. Project the ego offset onto the route-lane LATERAL direction
     (perpendicular to the route tangent).
  3. Penalise lateral_offset^2 (quadratic pull toward the route centerline).

Points farther than ``_MAX_LANE_DIST`` are masked out to avoid pulling
ego toward distant/disconnected route pieces.
"""

import torch

from .base import BaseGuidance
from .registry import register

_X, _Y = 0, 1
_DX, _DY = 2, 3

_MAX_LANE_DIST = 30.0


@register
class RouteCenterlineFollowingGuidance(BaseGuidance):
    """Quadratic penalty for lateral deviation from the nearest ROUTE lane centerline.

    Identical to ``CenterlineFollowingGuidance`` but reads ``route_lanes``
    from ``inputs`` so the guidance and the reward (which also uses
    ``route_lanes``) are aligned.

    _energy_scale = 0.1 matches CenterlineFollowingGuidance — gradient of
    ego_lat^2 at 1 m offset is ~40 in normalised trajectory coordinates so
    scale=0.1 gives a ~4 unit correction.
    """

    name = "route_centerline_following"
    _energy_scale = 0.1

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units (must contain route_lanes).

        Returns [B] unscaled reward (higher = closer to route centerline).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1

        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

        # route_lanes shape: [B, 25, 20, 33] (25 segments × 20 points each)
        route_lanes = inputs["route_lanes"]
        N = route_lanes.shape[1] * route_lanes.shape[2]

        lane_centers = route_lanes[..., _X:_Y + 1].reshape(B, N, 2)
        lane_dirs    = route_lanes[..., _DX:_DY + 1].reshape(B, N, 2)

        lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
        lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)

        lane_valid = lane_centers.norm(dim=-1) > 1e-3

        dist = (ego_pos.unsqueeze(2) - lane_centers.unsqueeze(1)).norm(dim=-1)
        dist = dist.masked_fill(~lane_valid.unsqueeze(1).expand(-1, T, -1), 1e6)
        nearest   = dist.argmin(dim=-1)
        min_dist  = dist.min(dim=-1).values

        def gather2(tensor):
            idx = nearest.unsqueeze(-1).expand(-1, -1, 2)
            return tensor.unsqueeze(1).expand(-1, T, -1, -1) \
                         .gather(2, idx.unsqueeze(2)).squeeze(2)

        c   = gather2(lane_centers)
        lat = gather2(lane_lat)

        ego_lat = ((ego_pos - c) * lat).sum(dim=-1)  # [B, T]

        no_lane = min_dist > _MAX_LANE_DIST
        ego_lat = ego_lat.masked_fill(no_lane, 0.0)

        reward = -(ego_lat ** 2).sum(dim=-1)  # [B]
        return reward


# ---------------------------------------------------------------------------
# Backward-compatible module-level function alias
# ---------------------------------------------------------------------------

def route_centerline_following_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Deprecated. Use RouteCenterlineFollowingGuidance via GuidanceComposer."""
    from .config import GuidanceConfig
    fn = RouteCenterlineFollowingGuidance(GuidanceConfig(name="route_centerline_following"))
    return fn.energy(x, t, inputs)
