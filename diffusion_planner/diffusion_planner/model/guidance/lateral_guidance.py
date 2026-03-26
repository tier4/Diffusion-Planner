"""PlannerRFT-style lateral guidance for structured trajectory exploration.

Applies a perpendicular offset from a reference trajectory, pushing the ego
trajectory laterally (left/right) relative to the reference heading at each
timestep. This decomposes exploration into an interpretable lateral axis
rather than relying on isotropic Gaussian noise.

Reference: PlannerRFT (arxiv 2601.12901) — lateral/longitudinal guidance
decomposition for reinforcement fine-tuning exploration.

Requires ``reference_trajectory`` in the inputs dict:
    inputs["reference_trajectory"]: [B, T, 4] — (x, y, cos_yaw, sin_yaw)
    in physical ego-centric metres, typically the deterministic model output.

Params:
    lateral_offset (float): Perpendicular offset in metres.
        Positive = left (driver side in right-hand-traffic).
        Negative = right.
"""

import torch

from .base import BaseGuidance
from .registry import register


@register
class LateralGuidance(BaseGuidance):
    """Pulls ego trajectory toward reference + lateral offset.

    At each timestep t, computes the target position as:
        target_t = ref_pos_t + offset * perp_direction_t

    where perp_direction = (-sin_yaw, cos_yaw) is the left-perpendicular
    to the reference heading.

    Energy = -sum_t ||ego_pos_t - target_t||^2

    _energy_scale = 0.05 matches route_following magnitude.
    """

    name = "lateral"
    _energy_scale = 1.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._offset = config.params.get("lateral_offset", 1.0)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: must contain "reference_trajectory" [B, T, 4].

        Returns [B] reward (higher = closer to lateral target).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device

        ref = inputs.get("reference_trajectory")
        if ref is None:
            return torch.zeros(B, device=device)

        # ref: [B, T, 4] — (x, y, cos_yaw, sin_yaw)
        ref = ref[:, :T, :]  # clip to match trajectory length

        ref_pos = ref[..., :2]       # [B, T, 2]
        ref_cos = ref[..., 2:3]      # [B, T, 1]
        ref_sin = ref[..., 3:4]      # [B, T, 1]

        # Left-perpendicular direction: (-sin, cos)
        perp = torch.cat([-ref_sin, ref_cos], dim=-1)  # [B, T, 2]

        # Target = reference position + lateral offset along perpendicular
        target = ref_pos + self._offset * perp  # [B, T, 2]

        # Ego future positions
        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

        # Quadratic attraction toward target
        diff = ego_pos - target  # [B, T, 2]
        reward = -(diff ** 2).sum(dim=-1).sum(dim=-1)  # [B]

        return reward
