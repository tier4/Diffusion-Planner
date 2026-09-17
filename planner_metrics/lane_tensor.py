"""Shared shape normalization for the NPZ lane tensors.

``lanes`` and ``route_lanes`` reach metric evaluators in several equivalent
shapes depending on whether the caller handed over a single scene, a collated
batch, or a batch that still carries the model's singleton context axis. Every
lane-based metric has to flatten that to one canonical ``(B, S, P, D)`` layout
first; this module holds that one shared implementation so the accepted shapes
and their error messages cannot drift apart between metrics.
"""

from __future__ import annotations

import torch


def resolve_lane_tensor(lanes: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return ``lanes`` as ``(B, S, P, D)`` with ``B`` either 1 or ``batch_size``.

    ``B == 1`` means one lane tensor shared by every trajectory in the batch;
    callers index it with ``0 if lanes.shape[0] == 1 else sample_index``.
    """
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(
                f"expected singleton route_lanes context axis, got {tuple(lanes.shape)}"
            )
        lanes = lanes[:, 0]
    if lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    if lanes.ndim != 4 or lanes.shape[0] not in (1, batch_size):
        raise ValueError(
            "lanes must have shape (S,P,D), (1,S,P,D), or (N,S,P,D); "
            f"got {tuple(lanes.shape)} for N={batch_size}"
        )
    return lanes


__all__ = ["resolve_lane_tensor"]
