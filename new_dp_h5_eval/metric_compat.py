"""Exact, scoring-only views of native fields expected by legacy metrics."""

from __future__ import annotations

import numpy as np


def legacy_route_lanes(frame: dict[str, np.ndarray]) -> np.ndarray:
    """Build old 33-column route lanes from native geometry and current TL state.

    Native lane columns are xy/left-offset/right-offset. Legacy red-light scoring
    also needs a centerline tangent, which is determined from adjacent native xy.
    """
    native = frame["route_lanes"]
    out = np.zeros((*native.shape[:-1], 33), dtype=np.float32)
    out[..., :2] = native[..., :2]
    valid = np.linalg.norm(native[..., :2], axis=-1) > 0
    delta = np.zeros_like(native[..., :2])
    delta[:, :-1] = native[:, 1:, :2] - native[:, :-1, :2]
    delta[:, -1] = delta[:, -2]
    pair_valid = valid & np.concatenate((valid[:, 1:], valid[:, -1:]), axis=1)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    out[..., 2:4] = np.where(pair_valid[..., None], delta / np.maximum(norm, 1e-6), 0)
    out[..., 4:6] = native[..., 2:4]
    out[..., 6:8] = native[..., 4:6]
    # Old one-hot columns: green=8, yellow=9, red=10, white=11, none=12.
    tl = frame["route_traffic_light_past"][:, -1]
    out[..., 8] = tl[:, None, 0]
    out[..., 9] = tl[:, None, 1]
    out[..., 10] = tl[:, None, 2]
    out[..., 11] = tl[:, None, 4]
    out[..., 12] = tl[:, None, 3]
    return out

