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

        B = x_in.shape[0]
        P = x_in.shape[1]
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]

        # x_in may be 3D [B,P,T*4] (after prefix_constraint) or 4D [B,P,T,4].
        # The DiT model requires 4D, so reshape for the x_start correction.
        x_4d = x_in.reshape(B, P, -1, 4)
        t_4d = t_input if t_input.dim() == 4 else t_input
        x_fix = model(x_4d, t_4d, **model_condition).detach() - x_4d.detach()
        x_fix[:, :, 0] = 0.0
        x_corrected = x_4d + x_fix

        x_phys = state_normalizer.inverse(x_corrected.detach())
        inputs = observation_normalizer.inverse(kwargs["inputs"])

        t_scalar = t_input.reshape(-1)[0].unsqueeze(0) if t_input.dim() > 1 else t_input

        # Compute guidance gradient on detached 4D trajectory, then use
        # surrogate energy = dot(grad, x_in) so autograd returns a gradient
        # matching x_in's shape (3D or 4D), compatible with the DPM solver.
        x_phys_grad = x_phys.detach().requires_grad_(True)
        raw_energy = torch.zeros(B, device=x_in.device)
        for fn in self._functions:
            e = fn.energy(x_phys_grad, t_scalar, inputs)
            if torch.isnan(e).any():
                continue
            raw_energy = raw_energy + e

        if raw_energy.requires_grad:
            grad_phys = torch.autograd.grad(raw_energy.sum(), x_phys_grad)[0]
        else:
            grad_phys = torch.zeros_like(x_phys_grad)

        # Transform gradient from physical space back to normalized space
        # and reshape to match x_in's shape for the surrogate dot product.
        grad_flat = grad_phys.detach().reshape(x_in.shape)

        # Surrogate energy: autograd.grad(dot(grad, x_in), x_in) = grad
        surrogate = (grad_flat * x_in).sum()
        return surrogate

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
