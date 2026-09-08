"""Traffic-light preprocessing shared by training and visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Frame, FrameLike

UNKNOWN_INDEX = 3


def _forward_fill_unknown(past: NDArray[Any], future: NDArray[Any]) -> NDArray[Any]:
    """Replace future Unknown states with the preceding traffic-light state.

    Every future step takes the most recent known state at or before it, falling
    back to the last past state when no earlier future step is known. The gather
    indices are built with a running maximum so the whole element grid is filled
    in a few vectorized passes instead of one Python loop per step.
    """
    horizon = future.shape[-2]
    known_source = np.arange(1, horizon + 1)
    # Index 0 selects the last past state; index t + 1 selects future step t.
    source = np.where(future[..., UNKNOWN_INDEX] > 0.5, 0, known_source)
    np.maximum.accumulate(source, axis=-1, out=source)
    candidates = np.concatenate((past[..., -1:, :], future), axis=-2)
    filled = np.take_along_axis(candidates, source[..., None], axis=-2)
    return filled.astype(future.dtype, copy=False)


def fill_unknown_traffic_light_futures(
    frame: FrameLike,
) -> Frame:
    """Forward-fill Unknown lane and route future states without mutating input."""
    result = dict(frame)
    for past_key, future_key in (
        ("lane_traffic_light_past", "lane_traffic_light_future"),
        ("route_traffic_light_past", "route_traffic_light_future"),
    ):
        if past_key not in result or future_key not in result:
            continue
        result[future_key] = _forward_fill_unknown(
            np.asarray(result[past_key]), np.asarray(result[future_key])
        )
    return result


class FillUnknownTrafficLightFutures:
    """Forward-fill unknown traffic-light futures as a dataset transform."""

    def __call__(self, frame: FrameLike) -> Frame:
        return fill_unknown_traffic_light_futures(frame)
