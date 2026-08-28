"""Traffic-light preprocessing shared by training and visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Frame, FrameLike


def _forward_fill_unknown(past: NDArray[Any], future: NDArray[Any]) -> NDArray[Any]:
    """Replace future Unknown states with the preceding traffic-light state."""
    filled = np.array(future, copy=True)
    flattened_past = past.reshape(-1, past.shape[-2], past.shape[-1])
    flattened_future = filled.reshape(-1, filled.shape[-2], filled.shape[-1])
    unknown_index = 3
    for history, future_sequence in zip(flattened_past, flattened_future, strict=True):
        previous = history[-1].copy()
        for state in future_sequence:
            if state[unknown_index] > 0.5:
                state[:] = previous
            else:
                previous = state.copy()
    return filled


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
