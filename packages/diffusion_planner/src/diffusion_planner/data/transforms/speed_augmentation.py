"""NumPy ego-speed augmentation for planner dataset frames."""

from __future__ import annotations

import numpy as np

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike


class PlannerSpeedAugmentation:
    """Randomly scale only the ego velocity history."""

    def __init__(
        self,
        speed_scale_range: tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5,
    ) -> None:
        self.speed_scale_range = speed_scale_range
        self.probability = probability

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        if np.random.random() >= self.probability:
            return output

        output["ego_agent_past"] = input_data["ego_agent_past"].copy()
        speed_scale = np.random.uniform(*self.speed_scale_range)
        output["ego_agent_past"][..., EGO_VELOCITY_INDEX] *= speed_scale
        return output
