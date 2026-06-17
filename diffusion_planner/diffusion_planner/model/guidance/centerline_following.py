"""Centerline following guidance for the diffusion planner.

Pulls the ego trajectory toward the nearest lane centerline by penalising
lateral deviation continuously (quadratic cost), unlike lane_keeping which
only fires when the vehicle protrudes beyond the boundary.

The gradient grows linearly with lateral offset, so the correction is
stronger the further the vehicle is from the center.
"""

import torch

from .base import BaseGuidance
from .registry import register

_X, _Y = 0, 1
_DX, _DY = 2, 3

_MAX_LANE_DIST = 30.0


@register
class CenterlineFollowingGuidance(BaseGuidance):
    """Quadratic penalty for lateral deviation from the nearest lane centerline.

    _energy_scale = 0.1 keeps the DPM-Solver correction at a reasonable
    magnitude; the gradient of ego_lat^2 at 1 m lateral offset is ~40 in
    normalised trajectory coordinates so scale=0.1 gives a ~4 unit correction.
    """

    name = "centerline_following"
    _energy_scale = 0.1

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units.

        Returns [B] unscaled reward (higher = closer to lane centerline).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1

        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

        lanes = inputs["lanes"]  # [B, 140, 20, 33]
        N = lanes.shape[1] * lanes.shape[2]

        lane_centers = lanes[..., _X : _Y + 1].reshape(B, N, 2)
        lane_dirs = lanes[..., _DX : _DY + 1].reshape(B, N, 2)

        lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
        lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)

        lane_valid = lane_centers.norm(dim=-1) > 1e-3

        dist = (ego_pos.unsqueeze(2) - lane_centers.unsqueeze(1)).norm(dim=-1)
        dist = dist.masked_fill(~lane_valid.unsqueeze(1).expand(-1, T, -1), 1e6)
        nearest = dist.argmin(dim=-1)
        min_dist = dist.min(dim=-1).values

        def gather2(tensor):
            idx = nearest.unsqueeze(-1).expand(-1, -1, 2)
            return tensor.unsqueeze(1).expand(-1, T, -1, -1).gather(2, idx.unsqueeze(2)).squeeze(2)

        c = gather2(lane_centers)
        lat = gather2(lane_lat)

        ego_lat = ((ego_pos - c) * lat).sum(dim=-1)  # [B, T]

        no_lane = min_dist > _MAX_LANE_DIST
        ego_lat = ego_lat.masked_fill(no_lane, 0.0)

        reward = -(ego_lat**2).sum(dim=-1)  # [B]
        return reward


# ---------------------------------------------------------------------------
# Backward-compatible module-level function alias
# ---------------------------------------------------------------------------


def centerline_following_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Deprecated. Use CenterlineFollowingGuidance via GuidanceComposer."""
    from .config import GuidanceConfig

    fn = CenterlineFollowingGuidance(GuidanceConfig(name="centerline_following"))
    return fn.energy(x, t, inputs)
