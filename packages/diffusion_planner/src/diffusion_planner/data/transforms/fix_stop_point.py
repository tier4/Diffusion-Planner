"""Fix the ego future at its first detected stop point."""

from __future__ import annotations

import numpy as np

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike


class PlannerFixStopPoint:
    """Hold the ego future pose after its first low-speed point."""

    def __init__(self, stop_speed_threshold: float = 0.1) -> None:
        self.stop_speed_threshold = stop_speed_threshold

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        future = input_data.get("ego_agent_future")
        if future is None or len(future) == 0:
            return output

        stop_indices = np.flatnonzero(
            future[:, EGO_VELOCITY_INDEX] <= self.stop_speed_threshold
        )
        if len(stop_indices) == 0:
            return output

        stop_index = int(stop_indices[0])
        fixed_future = np.array(future, copy=True)
        fixed_future[stop_index:, :4] = future[stop_index, :4]
        fixed_future[stop_index:, 4:] = 0.0
        output["ego_agent_future"] = fixed_future
        return output
