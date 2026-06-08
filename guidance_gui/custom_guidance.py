"""Custom guidance functions for the Scene Branch Editor.

These live OUTSIDE ``diffusion_planner/`` (which is read-only here) but register
into the same guidance registry, so they are usable by name through the standard
``GuidanceComposer`` once this module is imported.

Currently provides:
    collision_swerve -- directional collision avoidance. The built-in
        ``collision`` guidance only corrects laterally and, for an obstacle
        directly ahead and centred, the left/right push is symmetric and nearly
        cancels (no preferred side), so the ego will not budge. ``collision_swerve``
        adds an explicit, caller-chosen side: when the ego comes within ``range``
        metres of a (static) neighbour, it rewards lateral displacement toward
        ``side`` (+1 = left, -1 = right), gated by proximity so the swerve
        concentrates around the obstacle. It does NOT touch the longitudinal axis
        (no braking) -- purely a left/right swerve, by design.
"""

import torch

from diffusion_planner.model.guidance.base import BaseGuidance
from diffusion_planner.model.guidance.registry import register


@register
class CollisionSwerveGuidance(BaseGuidance):
    """Directional, proximity-gated lateral swerve around neighbours.

    Params:
        side (float):  +1.0 = swerve left, -1.0 = swerve right. Default +1.0.
        range (float): proximity radius in metres within which the swerve
                       activates (centroid gap ego<->neighbour). Default 8.0.

    Energy (maximised by the solver):
        E = side * Σ_t  w_t * y_t
    where y_t is the ego lateral position (ego frame, +y = left) at future step t
    and w_t = max(0, 1 - d_t / range) is the proximity to the nearest valid
    neighbour at that step (detached, so it only gates -- it does not pull the ego
    toward the obstacle). ∂E/∂y_t = side * w_t, i.e. a clean lateral push toward the
    chosen side, strongest next to the obstacle.
    """

    name = "collision_swerve"
    _energy_scale = 1.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._side = float(config.params.get("side", 1.0))
        self._range = float(config.params.get("range", 8.0))

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units (must hold neighbor_agents_past).

        Returns [B] energy (higher = more displacement toward the chosen side
        near obstacles).
        """
        B = x.shape[0]
        device = x.device

        nb = inputs.get("neighbor_agents_past")
        if nb is None:
            return torch.zeros(B, device=device)

        # Ego future positions (keep grad on xy).
        ego_xy = x[:, 0, 1:, :2]  # [B, T, 2]
        T = ego_xy.shape[1]

        # Static neighbour current positions + validity mask.
        nb_cur = nb[:, :, -1, :4].detach()        # [B, Pn, 4]
        nb_xy = nb_cur[..., :2]                    # [B, Pn, 2]
        nb_valid = nb_cur.abs().sum(dim=-1) > 0    # [B, Pn]

        if nb_valid.sum() == 0:
            return torch.zeros(B, device=device)

        # Centroid gap ego_t <-> neighbour (detached -> proximity gate only).
        d = torch.cdist(ego_xy.detach(), nb_xy)    # [B, T, Pn]
        big = torch.full_like(d, 1e6)
        d = torch.where(nb_valid[:, None, :], d, big)
        w = torch.clamp(1.0 - d / self._range, min=0.0)  # [B, T, Pn]
        w = w.max(dim=-1).values                          # [B, T] nearest-neighbour proximity

        y = ego_xy[..., 1]                                # [B, T] lateral, +y = left
        reward = (self._side * w.detach() * y).sum(dim=-1)  # [B]
        return reward
