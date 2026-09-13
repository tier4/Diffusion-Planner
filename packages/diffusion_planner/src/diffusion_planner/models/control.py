"""Conversion between pose trajectories and unicycle control sequences.

The decoder can predict a per-timestep control `(accel, curvature)` channel
alongside (or instead of) the usual `(x, y, cos_yaw, sin_yaw)` pose channel.
This module holds the pieces that translate between the two representations and
the normalizer that puts control into the same rough scale as the pose channels.

Two conventions are load-bearing and silently produce wrong numbers if broken:

Units
    Trajectories are in **metres**. ``PlannerDataNormalizer`` divides xy by
    ``position_scale``, so callers holding batched model inputs must scale back
    first -- see :func:`denormalize_positions`. Speeds are in m/s, which the
    dataset already stores unscaled.

Frame
    The unicycle rollout integrates from the origin with heading zero, so both
    directions of the conversion assume an **ego-centric frame whose origin is
    the last history pose**: ``history[..., -1, :] == [0, 0, 1, 0]``. The H5
    dataset stores ``ego_agent_past`` in exactly this frame (and
    ``PlannerRigidAugmentation`` re-centres onto the augmented pose, so it still
    holds downstream). Feeding a trajectory in any other frame makes the fit
    absorb the offset into a spurious initial acceleration and curvature.
"""

from __future__ import annotations

from typing import cast

import torch

from ..data.dimensions import TRAJECTORY_LENGTH
from .unicycle import (
    UnicycleAccelCurvatureActionSpace,
    action_to_traj4d,
    traj4d_to_action,
)

# One shared action space: it holds only constant buffers (dt, bounds, the
# normalization statistics of the action itself), so a module-level instance
# avoids rebuilding the smoothing matrices on every batch.
ACTION_SPACE = UnicycleAccelCurvatureActionSpace(n_waypoints=TRAJECTORY_LENGTH)

# Integration step of the action space, in seconds.
DT = float(ACTION_SPACE.dt)


def _action_buffer(name: str) -> torch.Tensor:
    """Read one of the action space's registered constant buffers.

    ``register_buffer`` leaves the attribute typed as ``Tensor | Module``, so the
    cast belongs here rather than at every call site.
    """
    return cast(torch.Tensor, getattr(ACTION_SPACE, name))


def denormalize_action(control: torch.Tensor) -> torch.Tensor:
    """Undo the action space's own `(accel, curvature)` scaling.

    :func:`waypoints_to_control` returns control in the action space's normalized
    units. Anything that integrates the control itself -- rather than handing it
    back to :func:`control_to_waypoints` -- needs the physical values.

    Args:
        control: `(..., 2)` control in the action space's normalized units.

    Returns:
        `(..., 2)` control as m/s^2 and 1/m.
    """
    mean = torch.stack((_action_buffer("accel_mean"), _action_buffer("curvature_mean")))
    std = torch.stack((_action_buffer("accel_std"), _action_buffer("curvature_std")))
    return control * std.to(control.device) + mean.to(control.device)


class ControlNormalizer:
    """Scale control channels to roughly unit variance for the diffusion target."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, control: torch.Tensor) -> torch.Tensor:
        """Normalize raw `(accel, curvature)` into the training scale."""
        return (control - self.mean.to(control.device)) / self.std.to(control.device)

    def inverse(self, control: torch.Tensor) -> torch.Tensor:
        """Restore raw `(accel, curvature)` from the training scale."""
        return control * self.std.to(control.device) + self.mean.to(control.device)

    def to_dict(self) -> dict[str, list[float]]:
        """Return the statistics in a JSON-serializable form."""
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }


def denormalize_positions(
    trajectory: torch.Tensor, position_scale: float
) -> torch.Tensor:
    """Undo ``PlannerDataNormalizer``'s xy scaling, leaving other channels alone."""
    xy = trajectory[..., :2] * position_scale
    return torch.cat((xy, trajectory[..., 2:]), dim=-1)


def waypoints_to_control(
    history: torch.Tensor,
    future: torch.Tensor,
    initial_speed: torch.Tensor | None,
) -> torch.Tensor:
    """Fit the control sequence that reproduces `future` from `history`.

    This is a ground-truth-side conversion and carries no gradient
    (``traj4d_to_action`` runs under ``torch.no_grad``).

    Args:
        history: `(..., T_hist, 4)` past poses in metres, ending at the origin.
        future: `(..., T, 4)` future poses in metres.
        initial_speed: `(...)` speed at the last history step in m/s, or None to
            estimate it from `history`.

    Returns:
        `(..., T, 2)` raw `(accel, curvature)`.
    """
    t0_states = None if initial_speed is None else {"v": initial_speed}
    return traj4d_to_action(ACTION_SPACE, history, future, t0_states=t0_states)


def control_to_waypoints(
    control: torch.Tensor,
    history: torch.Tensor,
    initial_speed: torch.Tensor | None,
) -> torch.Tensor:
    """Roll `control` out from the end of `history` into poses.

    Unlike :func:`waypoints_to_control` this is differentiable, so it can carry a
    trajectory-space loss back into the predicted control.

    Args:
        control: `(..., T, 2)` raw `(accel, curvature)`.
        history: `(..., T_hist, 4)` past poses in metres, ending at the origin.
        initial_speed: `(...)` speed at the last history step in m/s, or None to
            estimate it from `history`.

    Returns:
        `(..., T, 4)` poses in metres.
    """
    t0_states = None if initial_speed is None else {"v": initial_speed}
    return action_to_traj4d(ACTION_SPACE, history, control, t0_states=t0_states)
