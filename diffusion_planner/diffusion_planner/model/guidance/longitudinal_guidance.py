"""PlannerRFT-style longitudinal guidance for structured trajectory exploration.

Applies an arc-length offset along a reference trajectory in the Frenet
frame, making the ego travel faster or slower than the reference. The
target trajectory is computed by shifting each reference point along
the path by the specified arc-length offset.

This follows PlannerRFT's (arxiv 2601.12901) lateral/longitudinal
decomposition of the exploration space. In PlannerRFT, a learned PPO
policy outputs Beta-distributed longitudinal offsets per scene; here we
accept a fixed offset parameter (random sampling is done by the GRPO
sampler). The Frenet frame math is in frenet.py.

Can be used in two ways:
  1. As a guidance energy during DPM-Solver denoising (via GuidanceComposer)
  2. The underlying frenet.perturb_trajectory() can create perturbed xT
     initial conditions for the sampler (bypassing guidance entirely).

Requires ``reference_trajectory`` in the inputs dict:
    inputs["reference_trajectory"]: [B, T, 4] — (x, y, cos_yaw, sin_yaw)
    in physical ego-centric metres, typically the deterministic model output.

Params:
    longitudinal_offset (float): Arc-length offset in metres (Frenet Δs).
        Positive = ahead on path (faster). Negative = behind (slower).
"""

import torch

from .base import BaseGuidance
from .frenet import perturb_trajectory
from .registry import register


@register
class LongitudinalGuidance(BaseGuidance):
    """Guidance energy that pulls ego toward a Frenet-frame longitudinal offset.

    Computes the target trajectory by applying a longitudinal (arc-length)
    offset in the Frenet frame of the reference, then returns negative
    squared distance from ego to target as the energy.

    _energy_scale = 1.0; use config.scale to tune strength.
    """

    name = "longitudinal"
    _energy_scale = 1.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._longitudinal_offset = config.params.get("longitudinal_offset", 2.0)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: must contain "reference_trajectory" [B, T, 4].

        Returns [B] reward (higher = closer to Frenet-offset target).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device

        ref = inputs.get("reference_trajectory")
        if ref is None:
            return torch.zeros(B, device=device)

        ref = ref[:, :T, :]

        # Compute target via Frenet perturbation (longitudinal only, no lateral)
        target = perturb_trajectory(ref, lateral_offset=0.0, longitudinal_offset=self._longitudinal_offset)
        target_xy = target[..., :2]  # [B, T, 2]

        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

        diff = ego_pos - target_xy
        reward = -(diff ** 2).sum(dim=-1).sum(dim=-1)  # [B]
        return reward
