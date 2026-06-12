"""Route-following guidance for the Diffusion Planner.

Minimises the distance from ego trajectory points to the nearest route lane
point, pulling the predicted trajectory toward the planned route.
"""

import torch

from .base import BaseGuidance
from .registry import register


@register
class RouteFollowingGuidance(BaseGuidance):
    """Attracts the ego trajectory toward the planned route lane points.

    For each future ego position, finds the nearest point among all route lane
    segment points and accumulates the minimum distance as a negative reward.

    _energy_scale = 0.05 keeps the DPM-Solver correction at a comparable
    magnitude to lane-keeping guidance.
    """

    name = "route_following"
    _energy_scale = 0.05

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units.

        Returns [B] unscaled reward (higher = closer to route).
        """
        B, P, T, _ = x.shape
        route_lanes = inputs["route_lanes"]  # [B, SegNum=25, PointNum=20, SEGMENT_POINT_DIM=33]
        route_lanes = route_lanes.reshape(B, 25 * 20, route_lanes.shape[-1])  # [B, 500, 33]
        route_lanes = route_lanes[:, :, :2]  # [B, 500, 2]

        predictions = x[:, 0, :, :2]  # [B, T+1, 2]
        pred_points = predictions[:, 1:]  # [B, T, 2]  (exclude pinned current state)

        expanded_routes = route_lanes.unsqueeze(1)  # [B, 1, 500, 2]
        expanded_preds = pred_points.unsqueeze(2)  # [B, T, 1, 2]
        distances = torch.norm(expanded_preds - expanded_routes, dim=-1)  # [B, T, 500]
        min_distances = torch.min(distances, dim=2)[0]  # [B, T]

        reward = -torch.sum(min_distances, dim=1)  # [B]
        return reward


# ---------------------------------------------------------------------------
# Backward-compatible module-level function alias
# ---------------------------------------------------------------------------


def route_following_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Deprecated. Use RouteFollowingGuidance via GuidanceComposer."""
    from .config import GuidanceConfig

    fn = RouteFollowingGuidance(GuidanceConfig(name="route_following"))
    return fn.energy(x, t, inputs)
