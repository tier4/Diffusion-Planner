"""Target speed maintenance guidance for the Diffusion Planner.

Uses a surrogate gradient approach (same pattern as collision guidance) to
produce stable gradients through DPM-Solver sampling. For each timestep where
speed exceeds v_high, computes the target position that would achieve v_high
and uses dot(target_correction.detach(), x) as the energy — making the
gradient of energy w.r.t. x equal to the desired correction direction.

Compatible with BaseGuidance interface:
    x:       [B, P, T+1, 4]  physical ego-centric metres
    outputs: [B] reward (higher = speed within acceptable band)
"""

import torch

from .base import BaseGuidance
from .registry import register


@register
class SpeedGuidance(BaseGuidance):
    """Surrogate-gradient speed guidance for DPM-Solver.

    Instead of autograd through relu(v-v_high)² (which produces unstable
    gradients through multi-step denoising), compute the explicit position
    correction needed to cap speed at v_high. The correction vector dotted
    with x creates a surrogate energy whose gradient equals the correction.

    _energy_scale = 0.05 similar to route/lane guidance.
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

        # For timesteps where v > v_high, compute the target displacement
        # that would achieve exactly v_high in the same direction.
        direction = disp / dist  # [B, T-1, 2] unit direction
        target_dist = (self._v_high * self._dt)  # scalar
        # Correction: pull pos[t+1] back toward pos[t] to match target_dist
        # correction = target_pos - actual_pos = (direction * target_dist - disp)
        # Only active when v > v_high
        overspeed = (v > self._v_high).float().unsqueeze(-1)  # [B, T-1, 1]
        correction = (direction * target_dist - disp) * overspeed  # [B, T-1, 2]

        # Also handle v < v_low (push forward)
        if self._v_low > 0:
            target_dist_low = self._v_low * self._dt
            underspeed = (v < self._v_low).float().unsqueeze(-1)
            correction = correction + (direction * target_dist_low - disp) * underspeed

        # Surrogate energy: dot(correction.detach(), pos[1:])
        # ∇_x dot(c, x) = c — so the gradient of this energy equals the correction.
        # DPM-Solver does x += gs * ∇energy = gs * correction → moves toward target speed.
        reward = torch.sum(correction.detach() * pos[:, 1:, :2], dim=(1, 2))  # [B]

        return reward
