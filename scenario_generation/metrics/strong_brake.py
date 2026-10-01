"""Strong-brake metric from active-plan acceleration."""

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
    values = np.asarray(accels, dtype=np.float64)
    raw = np.isfinite(values) & (values <= float(thresh_mps2))
    if raw.size == 0:
        return raw
    out = np.zeros_like(raw, dtype=bool)
    out[1:] = raw[:-1] & raw[1:]
    return out


def plan_acceleration(plan_xy: np.ndarray, dt: float = 0.1) -> float:
    """Acceleration between p0→p1 and p1→p2 of the active raw plan suffix."""
    points = np.asarray(plan_xy[:3], dtype=np.float64)
    if len(points) < 3 or not np.isfinite(points).all():
        return float("nan")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float((distances[1] - distances[0]) / dt**2)


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
