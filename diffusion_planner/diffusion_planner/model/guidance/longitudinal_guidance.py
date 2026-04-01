"""PlannerRFT longitudinal guidance energy (Eq. 3 from arxiv 2601.12901).

Energy-based classifier guidance that modulates the ego's speed relative
to a reference trajectory during DPM-Solver denoising.

At each timestep τ, the longitudinal energy penalizes the squared
deviation of the ego's tangential velocity from a scaled reference velocity:

    Ψ_lon = (1/(T-1)) Σ_τ (n∥_τ · (v_τ - λ_lon · η_lon · v^ref_τ))²

where:
    n∥_τ   = unit tangent (heading direction) of the reference
    v_τ    = ego velocity (finite difference of positions / dt)
    v^ref  = reference velocity (finite difference of reference / dt)
    λ_lon  = speed scaling constant (target = λ · η · v_ref)
    η_lon  = guidance scale in [-1, 1] (later learned by PPO)

Target tangential velocity = λ_lon · η_lon · v^ref_tangential.
    η_lon = 0  → target = 0 (penalizes all tangential velocity, encourages
                 stopping; only neutral when overall guidance scale is also 0)
    η_lon = 1, λ_lon = 1.0  → target = v_ref (match reference speed)
    η_lon = 1, λ_lon = 0.5  → target = 0.5 * v_ref (half reference speed)
    η_lon = -1 → target = -λ·v_ref (penalizes forward motion)

The PPO exploration policy will learn to output (η_lat, η_lon) from
Beta distributions per scene, controlling the exploration direction.

Requires ``reference_trajectory`` in the inputs dict:
    inputs["reference_trajectory"]: [B, T, 4] — (x, y, cos_yaw, sin_yaw)

Params:
    lambda_lon (float): Speed scaling constant. Default 0.5.
    eta_lon (float): Guidance scale in [-1, 1]. Default 0.0.
        Later replaced by PPO policy output.
    dt (float): Timestep for velocity computation. Default 0.1s.
"""

import torch

from .base import BaseGuidance
from .registry import register


@register
class LongitudinalGuidance(BaseGuidance):
    """PlannerRFT longitudinal classifier guidance (Eq. 3).

    Operates on velocity (speed scaling) not position (arc-length offset).
    _energy_scale = 10.0; tune via config.scale and eta_lon.
    """

    name = "longitudinal"
    _energy_scale = 10.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._lambda_lon = config.params.get("lambda_lon", 0.5)
        self._eta_lon = config.params.get("eta_lon", 0.0)
        self._dt = config.params.get("dt", 0.1)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: must contain "reference_trajectory" [B, T, 4].

        Returns [B] negative energy (higher = better alignment with target speed).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device

        ref = inputs.get("reference_trajectory")
        if ref is None:
            return torch.zeros(B, device=device)

        ref = ref[:, :T, :]

        # Reference heading → unit tangent
        cos_h = ref[..., 2]  # [B, T]
        sin_h = ref[..., 3]
        h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
        cos_h = cos_h / h_norm
        sin_h = sin_h / h_norm
        # n∥ = (cos, sin) — tangent direction
        n_par_x = cos_h  # [B, T]
        n_par_y = sin_h

        # Ego velocity via finite differences
        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]
        ego_vel = (ego_pos[:, 1:, :] - ego_pos[:, :-1, :]) / self._dt  # [B, T-1, 2]

        # Reference velocity via finite differences
        ref_pos = ref[..., :2]  # [B, T, 2]
        ref_vel = (ref_pos[:, 1:, :] - ref_pos[:, :-1, :]) / self._dt  # [B, T-1, 2]

        # Slice tangent to T-1 to match velocity dimensions
        n_par_x = n_par_x[:, :T - 1]
        n_par_y = n_par_y[:, :T - 1]

        # Project ego velocity onto tangent: n∥ · v
        ego_v_tangent = n_par_x * ego_vel[..., 0] + n_par_y * ego_vel[..., 1]  # [B, T-1]

        # Project reference velocity onto tangent: n∥ · v^ref
        ref_v_tangent = n_par_x * ref_vel[..., 0] + n_par_y * ref_vel[..., 1]  # [B, T-1]

        # Target: λ_lon · η_lon · v^ref_tangent
        # Supports both scalar eta and batched [B] tensor eta
        eta = self._eta_lon
        if isinstance(eta, torch.Tensor) and eta.dim() >= 1:
            eta = eta.unsqueeze(-1)  # [B, 1] for broadcasting with [B, T-1]
        target = self._lambda_lon * eta * ref_v_tangent  # [B, T-1]

        # Ψ_lon = (1/(T-1)) Σ (n∥ · (v - λ·η·v^ref))²
        psi = ((ego_v_tangent - target) ** 2).mean(dim=-1)  # [B]
        return -psi
