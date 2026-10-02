"""Strong-brake metric from executed-speed acceleration."""

from __future__ import annotations

import numpy as np


def strong_brake_mask(
    accels: np.ndarray,
    *,
    thresh_mps2: float = -2.5,
) -> np.ndarray:
    """Vectorized strong-brake flag over a 1-D accel series.

    A step counts after two consecutive frames satisfy ``accel <= thresh``.
    """
    values = np.asarray(accels, dtype=np.float64)
    raw = np.isfinite(values) & (values <= float(thresh_mps2))
    if raw.size == 0:
        return raw
    out = np.zeros_like(raw, dtype=bool)
    out[1:] = raw[:-1] & raw[1:]
    return out


def strong_brake_count(accels: np.ndarray, *, thresh_mps2: float = -2.5) -> int:
    """Confirm after two crossings; end after five raw clear ticks (10 Hz)."""
    count = pending = clear = 0
    active = False
    for value in accels:
        if not np.isfinite(value):
            active = False
            pending = clear = 0
        elif value <= thresh_mps2:
            clear = 0
            if not active:
                pending += 1
                if pending == 2:
                    count += 1
                    active = True
        else:
            pending = 0
            if active:
                clear += 1
                if clear == 5:
                    active = False
    return count
