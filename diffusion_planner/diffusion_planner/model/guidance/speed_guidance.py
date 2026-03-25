"""Target speed maintenance guidance for the Diffusion Planner.

Penalises ego path speed deviations outside a deadband [v_low, v_high]
using a squared hinge loss.  Within the band the energy is zero (no
interference with normal driving); outside the band the gradient grows
linearly with violation magnitude — making speed control predictable.

Implements Eq. (10) from the paper (same hinge form; speed is path speed in
the ego plane rather than x-only):
    E = Σ_τ [ max(v_τ - v_high, 0)² + max(v_low - v_τ, 0)² ]

Path speed is ||Δp|| / dt where p = (x, y) in the ego-centric frame and Δ
is between consecutive future waypoints (index 0 along time is the pinned
current state).  Inputs to _compute are already physical metres
(GuidanceComposer applies inverse normalisation before energy()).

Compatible with BaseGuidance interface:
    x:       [B, P, T+1, 4]  physical ego-centric metres
    outputs: [B] reward (higher = speed within acceptable band)
"""

import torch
import torch.nn.functional as F

from .base import BaseGuidance
from .registry import register


@register
class SpeedGuidance(BaseGuidance):
    """
    Deadband squared-hinge path-speed energy (Eq. 10 form, Appendix C.3).

    Energy is zero when ego path speed is in [v_low, v_high].
    Outside the band a squared penalty applies — linear gradient growth
    makes speed corrections predictable and tunable.

    _energy_scale = 300.0 aligns the gradient magnitude with collision guidance.
    """

    name = "speed"
    _energy_scale = 20.0
    _t_min = 0.001
    _t_max = 0.2

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        p = config.params
        self._v_low    = p.get("v_low", 0.0)      # m/s  lower speed bound
        self._v_high   = p.get("v_high", 14.0)   # m/s  upper speed bound
        self._dt       = p.get("dt", 0.1)         # s    time step between waypoints

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x:       [B, P, T+1, 4]  physical ego-centric metres (x, y, cos_h, sin_h)
        inputs:  observation dict in physical units

        Returns [B] reward (higher = speed within acceptable band).
        """
        # Ego future positions (excluding pinned current state at index 0)
        pos = x[:, 0, 1:, :2]  # [B, T, 2]
        disp = pos[:, 1:, :] - pos[:, :-1, :]  # [B, T-1, 2]
        v = torch.linalg.vector_norm(disp, dim=-1) / self._dt  # [B, T-1]

        # Squared hinge penalty: zero inside [v_low, v_high]
        penalty_high = F.relu(v - self._v_high) ** 2      # penalise v > v_high
        penalty_low  = F.relu(self._v_low - v) ** 2      # penalise v < v_low

        raw_energy = (penalty_high + penalty_low).mean(dim=1)   # [B]

        # BaseGuidance convention: higher reward = better trajectory
        reward = -raw_energy
        return reward
