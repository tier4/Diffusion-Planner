"""PlannerRFT lateral guidance energy (Eq. 2 from arxiv 2601.12901).

Energy-based classifier guidance that steers the ego trajectory laterally
relative to a reference trajectory during DPM-Solver denoising.

At each timestep τ, the lateral energy penalizes the squared deviation
of the ego's perpendicular projection from a target offset:

    Ψ_lat = (1/T) Σ_τ (n⊥_τ · (x_τ - x^ref_τ) - λ_lat · η_lat)²

where:
    n⊥_τ   = unit normal (left-perpendicular) to the reference heading
    λ_lat  = maximum lateral offset in metres (config param)
    η_lat  = guidance scale in [-1, 1] (config param, later learned by PPO)

The gradient ∇_x Ψ_lat is injected into the denoising process via
classifier guidance (Eq. 5), enabling multi-modal trajectory generation
through different η_lat values.

Requires ``reference_trajectory`` in the inputs dict:
    inputs["reference_trajectory"]: [B, T, 4] — (x, y, cos_yaw, sin_yaw)

Params:
    lambda_lat (float): Maximum lateral offset in metres. Default 3.0.
    eta_lat (float): Guidance scale in [-1, 1]. Default 0.0 (no offset).
        Positive = left, negative = right. Later replaced by PPO policy output.
"""

import torch

from .base import BaseGuidance
from .registry import register


@register
class LateralGuidance(BaseGuidance):
    """PlannerRFT lateral classifier guidance (Eq. 2).

    _energy_scale = 10.0; tune via config.scale and eta_lat.
    """

    name = "lateral"
    _energy_scale = 10.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._lambda_lat = config.params.get("lambda_lat", 3.0)
        self._eta_lat = config.params.get("eta_lat", 0.0)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: must contain "reference_trajectory" [B, T, 4].

        Returns [B] negative energy (higher = better alignment with target offset).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device

        ref = inputs.get("reference_trajectory")
        if ref is None:
            return torch.zeros(B, device=device)

        ref = ref[:, :T, :]

        # Reference heading → unit normal (left-perpendicular)
        cos_h = ref[..., 2]  # [B, T]
        sin_h = ref[..., 3]
        h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
        cos_h = cos_h / h_norm
        sin_h = sin_h / h_norm
        # n⊥ = (-sin, cos) — left-perpendicular
        n_perp_x = -sin_h  # [B, T]
        n_perp_y = cos_h

        # Ego position relative to reference
        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]
        ref_pos = ref[..., :2]      # [B, T, 2]
        dx = ego_pos[..., 0] - ref_pos[..., 0]  # [B, T]
        dy = ego_pos[..., 1] - ref_pos[..., 1]

        # Perpendicular projection: n⊥ · (x - x_ref)
        lateral_proj = n_perp_x * dx + n_perp_y * dy  # [B, T]

        # Target offset
        target = self._lambda_lat * self._eta_lat

        # Ψ_lat = (1/T) Σ (lateral_proj - target)²
        # Return negative because guidance maximizes energy (lower Ψ = better)
        psi = ((lateral_proj - target) ** 2).mean(dim=-1)  # [B]
        return -psi
