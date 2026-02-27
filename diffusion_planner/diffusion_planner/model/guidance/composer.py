"""GuidanceComposer: drop-in replacement for GuidanceWrapper."""

import torch

from .config import GuidanceSetConfig
from .registry import build


class GuidanceComposer:
    """
    Composes multiple guidance functions from a GuidanceSetConfig.

    Compatible with the DPM-Solver classifier_fn interface: same __call__
    signature as GuidanceWrapper so it can be assigned to
    model.decoder._guidance_fn without any Decoder changes.

    Also exposes compute_rewards() for DPO/GRPO reward evaluation.
    """

    def __init__(self, set_config: GuidanceSetConfig, **build_kwargs):
        self._set_config = set_config
        self._functions = [
            build(fn_cfg, **build_kwargs)
            for fn_cfg in set_config.active_functions()
        ]

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """
        Called by DPM-Solver at each denoising step.

        Applies x_start correction and inverse normalisation (identical to
        GuidanceWrapper), then sums energy from all active guidance functions.
        """
        state_normalizer = kwargs["state_normalizer"]
        observation_normalizer = kwargs["observation_normalizer"]

        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]

        # x_start denoising correction — identical to GuidanceWrapper logic.
        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x_in = x_in + x_fix.reshape(B, P, -1)

        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
        inputs = observation_normalizer.inverse(kwargs["inputs"])

        energy = torch.zeros(B, device=x_in.device)
        for fn in self._functions:
            e = fn.energy(x_in, t_input, inputs)
            if torch.isnan(e).any():
                print(f"Warning: NaN energy from {fn.name}, skipping")
                continue
            energy = energy + e

        return energy

    def compute_rewards(
        self,
        trajectory: torch.Tensor,
        inputs: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Evaluate all active guidance functions as scalar reward signals.

        Used by DPO (trajectory pair scoring) and GRPO (rollout scoring).

        Args:
            trajectory: [B, T, 4] completed ego trajectory in physical
                ego-centric metres (x, y, cos_yaw, sin_yaw).
            inputs: Observation dict in physical units.

        Returns:
            Dict with one [B] tensor per guidance function keyed by name,
            plus "total" containing the sum of all individual rewards.
        """
        rewards: dict[str, torch.Tensor] = {}
        total = torch.zeros(trajectory.shape[0], device=trajectory.device)
        for fn in self._functions:
            r = fn.reward(trajectory, inputs)
            rewards[fn.name] = r
            total = total + r
        rewards["total"] = total
        return rewards
