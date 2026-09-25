"""Shared shape normalization for the NPZ ``lanes``/``route_lanes`` tensors."""

from __future__ import annotations

import torch


def resolve_lane_tensor(lanes: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return ``lanes`` as a view of shape ``(B, S, P, D)``, ``B`` 1 or ``batch_size``.

    ``B == 1`` is one scene shared by the whole batch; callers index with
    ``0 if lanes.shape[0] == 1 else sample_index``. ``D`` is not validated --
    callers read different columns and check the ones they read.
    """
    original_shape = tuple(lanes.shape)
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(f"expected a singleton lane context axis, got {original_shape}")
        lanes = lanes[:, 0]
    if lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    if lanes.ndim != 4 or lanes.shape[0] not in (1, batch_size):
        raise ValueError(
            "lanes must have shape (S,P,D), (1,S,P,D), (N,S,P,D), or (N,1,S,P,D); "
            f"got {original_shape} for N={batch_size}"
        )
    return lanes


__all__ = ["resolve_lane_tensor"]
