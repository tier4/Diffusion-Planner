"""Anchor-following (MTR-style) guidance for the Diffusion Planner.

Guides the ego trajectory toward a pre-clustered prototype trajectory
(a "motion mode") extracted from the training set. The anchor defines
a desired shape in the ego-centric frame (straight, turn-left, etc.)
and the guidance energy attracts the predicted trajectory toward it.
"""

import numpy as np
import torch

from .base import BaseGuidance
from .registry import register


@register
class AnchorFollowingGuidance(BaseGuidance):
    """Soft guidance toward a prototype trajectory shape.

    The anchor is one row of a prototypes array of shape (K, 80, 2) loaded
    from a .npy file. The energy penalises squared Euclidean distance from
    the ego future positions to the anchor at each timestep.

    Required params in GuidanceConfig.params:
        prototypes_path (str): Path to .npy file of shape (K, 80, 2).
        anchor_index (int):    Index of the prototype to follow (0 ≤ idx < K).

    _energy_scale = 0.05 produces a correction comparable to route-following
    at default settings.
    """

    name = "anchor_following"
    _energy_scale = 0.05

    def __init__(self, config: "GuidanceConfig"):  # noqa: F821
        super().__init__(config)
        protos = np.load(config.params["prototypes_path"])  # (K, 80, 2)
        idx = config.params["anchor_index"]
        self._anchor = torch.tensor(protos[idx], dtype=torch.float32)  # (80, 2)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict (unused by this function).

        Returns [B] unscaled reward (higher = closer to anchor shape).
        """
        B = x.shape[0]
        T = x.shape[2] - 1           # number of future timesteps
        ego_pred = x[:, 0, 1:, :2]   # [B, T, 2]
        anchor = self._anchor.to(x.device)[:T]  # [T, 2]
        sq_dist = ((ego_pred - anchor.unsqueeze(0)) ** 2).sum(dim=-1)  # [B, T]
        return -sq_dist.sum(dim=-1)   # [B]
