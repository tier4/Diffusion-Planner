"""Strong-brake metric from realized tangential acceleration."""

from __future__ import annotations

import numpy as np


def strong_brake_mask(
    accels: np.ndarray,
    *,
    thresh_mps2: float = -2.5,
) -> np.ndarray:
    """Vectorized strong-brake flag over a 1-D accel series.

    A step counts only when *this* frame and the *previous* frame both satisfy
    ``accel <= thresh`` (two consecutive over-threshold samples). Single-frame
    spikes from tracker / replan chord noise are ignored.
    """
    raw = np.asarray(accels, dtype=np.float32) <= float(thresh_mps2)
    if raw.size == 0:
        return raw
    out = np.zeros_like(raw, dtype=bool)
    out[1:] = raw[:-1] & raw[1:]
    return out
