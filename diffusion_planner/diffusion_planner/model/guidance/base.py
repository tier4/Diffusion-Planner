"""Abstract base class for all guidance functions."""

from abc import ABC, abstractmethod

import torch


class BaseGuidance(ABC):
    """
    Base class for all guidance functions.

    Subclasses must define:
        name: ClassVar[str]    -- the registry key
        _energy_scale: float   -- normalisation constant that sets the natural unit
                                  of the energy so that guidance_scale=0.5 produces
                                  reasonable DPM-Solver corrections at default settings.

    And implement:
        _compute(x, inputs) -> torch.Tensor [B]

    energy() is called during DPM-Solver sampling; gradients flow through x.
    reward() is called on completed trajectories for DPO/GRPO scoring; no_grad.

    Scale formula applied by both methods:
        output = _energy_scale * config.scale * _compute(x, inputs)
    """

    name: str
    _energy_scale: float = 1.0

    # Diffusion timestep window in which guidance gradients are allowed to flow.
    # Outside this range x is detached to avoid instability at high noise levels
    # (t >> 0.1, trajectory is still dominated by noise) and at near-zero noise
    # (t < 0.005, last DPM step, gradient directions become unreliable).
    # With 10 logSNR-spaced DPM steps these bounds activate guidance at ~3 steps.
    # Values originate from the upstream Diffusion-Planner guidance implementation.
    # Subclasses may override if a different window is needed.
    _t_min: float = 0.005
    _t_max: float = 0.1

    def __init__(self, config: "GuidanceConfig"):  # noqa: F821
        self.config = config

    @abstractmethod
    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        Core physics computation shared by energy() and reward().

        Args:
            x: [B, P, T+1, 4] trajectory in physical ego-centric metres (base_link frame).
               Index 0 along T+1 is the pinned current state.
               In reward() mode P=1 (ego only); guidance functions that require
               neighbours must override reward() if neighbour data is needed.
            inputs: Observation dict already in physical units.

        Returns:
            [B] raw (unscaled) reward. Higher value = better trajectory.
        """
        ...

    def energy(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        """
        Guidance energy for DPM-Solver classifier guidance.

        Gradients flow through x only when t ∈ (0.005, 0.1); outside that
        window x is detached to avoid numerical instability.

        Args:
            x: [B, P, T+1, 4] trajectory in physical ego-centric metres,
               x_start-corrected and state_normalizer.inverse applied by
               GuidanceComposer before this call.
            t: [B] diffusion timestep in [0, 1].
            inputs: Observation dict in physical units.

        Returns:
            [B] energy tensor.
        """
        # t may be scalar [B] (v3) or per-timestep [B, P, T, 1] (v4).
        # Use the first element per batch for the time window check.
        t_scalar = t.reshape(t.shape[0], -1)[:, 0] if t.dim() > 1 else t
        mask = (t_scalar < self._t_max) * (t_scalar > self._t_min)
        mask = mask.view(x.shape[0], *([1] * (x.dim() - 1)))
        x_gated = torch.where(mask, x, x.detach())
        raw = self._compute(x_gated, inputs)
        return self._energy_scale * self.config.scale * raw

    @torch.no_grad()
    def reward(
        self,
        trajectory: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        """
        Scalar quality score for a completed ego trajectory.

        Used in DPO preference scoring and GRPO reward evaluation.

        Args:
            trajectory: [B, T, 4] completed ego trajectory in physical
                ego-centric metres (x, y, cos_yaw, sin_yaw). No current-state
                slot prepended.
            inputs: Observation dict in physical units.

        Returns:
            [B] reward tensor. Higher = better trajectory.
        """
        B, T, D = trajectory.shape
        # Prepend a zeroed current-state slot so _compute indices match
        # energy() convention: x[:, 0, 0, :] = current state, x[:, 0, 1:, :] = future.
        current_slot = torch.zeros(B, 1, 1, D, device=trajectory.device)
        x_padded = torch.cat([current_slot, trajectory.unsqueeze(1)], dim=2)  # [B, 1, T+1, 4]
        raw = self._compute(x_padded, inputs)
        return self._energy_scale * self.config.scale * raw
