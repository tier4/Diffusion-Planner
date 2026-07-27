"""Canonical NPZ scene-format helpers.

Trainable/reward futures are ALWAYS 4-col ``[x, y, cos, sin]`` — the cpp
data_converter writes this layout natively since npz version 3. Legacy 3-col
``[x, y, heading]`` arrays are widened HERE: this is the implementation the
repo shares (do not copy it into tools). Sole exception: ``diffusion_planner``
must stay installable standalone, so ``utils/neighbor_db.py`` carries its own
numpy/torch equivalents rather than importing this package.
"""

from __future__ import annotations

import numpy as np


def future_to_4col(arr: np.ndarray, *, zero_rows_are_padding: bool = True) -> np.ndarray:
    """Widen a heading future to the canonical 4-col layout.

    ``(..., 3)`` ``[x, y, heading]`` -> ``(..., 4)`` ``[x, y, cos, sin]``;
    already-4-col input passes through unchanged (idempotent).

    ``zero_rows_are_padding=True`` (neighbor arrays): rows whose x and y are both
    exactly zero are empty slots and stay fully zero instead of getting
    ``cos(0)=1``. Set it to ``False`` for single-agent trajectories (e.g. an ego
    future in the ego frame, which legitimately starts at the origin).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] == 4:
        return arr
    if arr.shape[-1] != 3:
        raise ValueError(f"expected last dim 3 or 4, got {arr.shape}")
    h = arr[..., 2]
    out = np.concatenate(
        [arr[..., :2], np.cos(h)[..., None], np.sin(h)[..., None]], axis=-1
    ).astype(np.float32)
    if zero_rows_are_padding:
        out[np.abs(arr[..., :2]).sum(-1) == 0] = 0.0
    return out
