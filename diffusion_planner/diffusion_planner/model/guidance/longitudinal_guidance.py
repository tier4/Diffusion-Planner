"""PlannerRFT-style longitudinal guidance for structured trajectory exploration.

Applies a time-shift along a reference trajectory, making the ego travel
faster or slower than the reference by pulling it toward a time-shifted
version of the reference path. This decomposes exploration into an
interpretable longitudinal (speed) axis.

Reference: PlannerRFT (arxiv 2601.12901) — lateral/longitudinal guidance
decomposition for reinforcement fine-tuning exploration.

Requires ``reference_trajectory`` in the inputs dict:
    inputs["reference_trajectory"]: [B, T, 4] — (x, y, cos_yaw, sin_yaw)
    in physical ego-centric metres, typically the deterministic model output.

Params:
    time_shift (float): Number of timesteps to shift. At dt=0.1s:
        Positive = ego should be ahead of reference (faster).
        Negative = ego should be behind reference (slower).
        Fractional values are interpolated linearly.
"""

import torch

from .base import BaseGuidance
from .registry import register


def _time_shift_trajectory(ref: torch.Tensor, shift: float) -> torch.Tensor:
    """Shift a reference trajectory by a fractional number of timesteps.

    Args:
        ref: [B, T, D] reference trajectory.
        shift: Fractional timestep shift. Positive = look ahead in time.

    Returns:
        [B, T, D] shifted trajectory. Positions beyond the trajectory
        boundary are clamped to the last/first valid position.
    """
    B, T, D = ref.shape

    if abs(shift) < 1e-6:
        return ref.clone()

    # Integer and fractional parts for linear interpolation
    shift_floor = int(shift) if shift >= 0 else -int(-shift) - 1
    shift_ceil = shift_floor + 1
    alpha = shift - shift_floor  # interpolation weight for ceil

    def _gather_shifted(s: int) -> torch.Tensor:
        """Gather ref positions at indices shifted by s, clamping at boundaries."""
        indices = torch.arange(T, device=ref.device) + s
        indices = indices.clamp(0, T - 1)  # [T]
        return ref[:, indices, :]  # [B, T, D]

    ref_floor = _gather_shifted(shift_floor)
    ref_ceil = _gather_shifted(shift_ceil)

    return (1.0 - alpha) * ref_floor + alpha * ref_ceil


@register
class LongitudinalGuidance(BaseGuidance):
    """Pulls ego trajectory toward a time-shifted reference.

    At each timestep t, the target position is the reference position at
    time t + time_shift (with linear interpolation for fractional shifts
    and clamping at trajectory boundaries).

    This naturally handles curves and varying speeds — a positive shift
    means "be where the reference will be N steps from now", which
    effectively asks the ego to travel faster along the same path.

    Energy = -sum_t ||ego_pos_t - ref_pos_{t+shift}||^2

    _energy_scale = 0.05 matches route_following magnitude.
    """

    name = "longitudinal"
    _energy_scale = 1.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._time_shift = config.params.get("time_shift", 5.0)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: must contain "reference_trajectory" [B, T, 4].

        Returns [B] reward (higher = closer to longitudinal target).
        """
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device

        ref = inputs.get("reference_trajectory")
        if ref is None:
            return torch.zeros(B, device=device)

        # ref: [B, T, 4] — (x, y, cos_yaw, sin_yaw)
        ref = ref[:, :T, :]  # clip to match trajectory length

        # Compute time-shifted target positions
        shifted_ref = _time_shift_trajectory(ref, self._time_shift)  # [B, T, 4]
        target = shifted_ref[..., :2]  # [B, T, 2]

        # Ego future positions
        ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

        # Quadratic attraction toward time-shifted target
        diff = ego_pos - target  # [B, T, 2]
        reward = -(diff ** 2).sum(dim=-1).sum(dim=-1)  # [B]

        return reward
