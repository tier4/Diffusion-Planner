"""Braking acceleration from the active raw plan, plus event confirmation."""

from __future__ import annotations

import numpy as np

PLAN_BRAKE_FILTER_TYPE = "active_plan_three_point"
PLAN_BRAKE_SOURCE = "active_plan_raw_local_suffix"
PLAN_BRAKE_POINTS = 3


def plan_suffix_acceleration(plan_xy: np.ndarray, *, dt: float = 0.1) -> float:
    """Score the first three points of the active plan suffix in m/s².

    The two adjacent chord speeds are differenced over one tick. This uses no
    executed ego state or temporal smoothing. An incomplete/nonfinite suffix is
    unscored instead of extrapolated or padded.
    """
    points = np.asarray(plan_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("plan_xy must have shape (N, 2)")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if len(points) < PLAN_BRAKE_POINTS or not np.isfinite(points[:PLAN_BRAKE_POINTS]).all():
        return float("nan")
    first = float(np.linalg.norm(points[1] - points[0]))
    second = float(np.linalg.norm(points[2] - points[1]))
    result = (second - first) / (dt * dt)
    return result if np.isfinite(result) else float("nan")


def confirmed_brake_mask(scored_accels: np.ndarray, *, thresh_mps2: float = -2.5) -> np.ndarray:
    """Confirm two adjacent already-scored threshold crossings."""
    values = np.asarray(scored_accels, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scored_accels must be one-dimensional")
    below = np.isfinite(values) & (values <= float(thresh_mps2))
    mask = np.zeros_like(below, dtype=bool)
    mask[1:] = below[:-1] & below[1:]
    return mask


def brake_event_onsets(
    scored_accels: np.ndarray,
    *,
    thresh_mps2: float = -2.5,
    confirm_frames: int = 2,
    clear_frames: int = 5,
) -> list[int]:
    """Confirm consecutive threshold crossings; release on raw clear frames.

    Missing scores reset the state. Onsets refer to the confirmation frame.
    An isolated crossing during an active event resets its clear timer.
    """
    if confirm_frames < 1 or clear_frames < 1:
        raise ValueError("confirm_frames and clear_frames must be positive")
    onsets = []
    active = False
    pending = clear = 0
    for k, value in enumerate(np.asarray(scored_accels, dtype=np.float64)):
        if not np.isfinite(value):
            active = False
            pending = clear = 0
        elif value <= thresh_mps2:
            clear = 0
            if not active:
                pending += 1
                if pending >= confirm_frames:
                    onsets.append(k)
                    active = True
                    pending = 0
        else:
            pending = 0
            if active:
                clear += 1
                if clear >= clear_frames:
                    active = False
                    clear = 0
    return onsets
