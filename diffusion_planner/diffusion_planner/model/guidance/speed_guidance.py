"""Target speed maintenance and trajectory stretch guidance.

Two modes (stretch takes precedence when != 1.0):

1. **Stretch mode** (``stretch`` param != 1.0): scales every per-step
   displacement by the stretch factor, pushing the trajectory to travel
   faster (>1) or slower (<1) along its current direction.

2. **Band mode** (default, ``stretch`` == 1.0): clamps speed to
   [v_low, v_high] using a surrogate gradient that computes the explicit
   position correction needed to cap/boost speed.

Both modes use the surrogate gradient trick ``dot(correction.detach(), x)``
so that ``grad_x(energy) = correction``, producing stable gradients
through DPM-Solver sampling.

Compatible with BaseGuidance interface:
    x:       [B, P, T+1, 4]  physical ego-centric metres
    outputs: [B] reward (higher = better speed compliance)
"""

import torch

from .base import BaseGuidance
from .registry import register


@register
class SpeedGuidance(BaseGuidance):
    """Surrogate-gradient speed guidance for DPM-Solver.

    Supports two modes controlled by the ``stretch`` param:
    - stretch != 1.0: scale all displacements by stretch (speed up/slow down).
    - stretch == 1.0 (default): clamp speed to [v_low, v_high] band.
    """

    name = "speed"
    _energy_scale = 1.0
    _t_min = 0.005
    _t_max = 0.1

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        p = config.params
        self._v_low = p.get("v_low", 0.0)
        self._v_high = p.get("v_high", 14.0)
        self._dt = p.get("dt", 0.1)
        self._stretch = p.get("stretch", 1.0)  # >1 = speed up, <1 = slow down

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x:       [B, P, T+1, 4]  physical ego-centric metres
        inputs:  observation dict in physical units

        Returns [B] surrogate energy for DPM-Solver guidance.
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1

        # Ego future positions
        pos = x[:, 0, 1:, :2]  # [B, T, 2]
        disp = pos[:, 1:, :] - pos[:, :-1, :]  # [B, T-1, 2]
        dist = disp.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, T-1, 1]
        v = dist.squeeze(-1) / self._dt  # [B, T-1]

        # Stretch mode: scale all displacements by stretch factor.
        # Each point gets pushed along its current travel direction to achieve
        # v_desired = v_current * stretch.  Uses surrogate gradient approach
        # (correction.detach() dotted with pos) for stable DPM-Solver gradients.
        # Ref: "Safe and Stylized Trajectory Planning" (2026), Eq. speed energy.
        if abs(self._stretch - 1.0) > 1e-6:
            correction = disp * (self._stretch - 1.0)  # [B, T-1, 2]
            reward = torch.sum(correction.detach() * pos[:, 1:, :2], dim=(1, 2))
            return reward

        # v_low / v_high mode: clamp speed to [v_low, v_high] band.
        direction = disp / dist  # [B, T-1, 2] unit direction
        target_dist = self._v_high * self._dt  # scalar
        overspeed = (v > self._v_high).float().unsqueeze(-1)  # [B, T-1, 1]
        correction = (direction * target_dist - disp) * overspeed  # [B, T-1, 2]

        if self._v_low > 0:
            target_dist_low = self._v_low * self._dt
            underspeed = (v < self._v_low).float().unsqueeze(-1)
            correction = correction + (direction * target_dist_low - disp) * underspeed

        # Surrogate energy: dot(correction.detach(), pos[1:])
        # ∇_x dot(c, x) = c — so the gradient of this energy equals the correction.
        # DPM-Solver does x += gs * ∇energy = gs * correction → moves toward target speed.
        reward = torch.sum(correction.detach() * pos[:, 1:, :2], dim=(1, 2))  # [B]

        return reward
