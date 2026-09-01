"""Random ego-shape reduction for planner dataset frames."""

from __future__ import annotations

import numpy as np

from .base import Frame, FrameLike


class PlannerEgoShapeAugmentation:
    """Randomly reduce ego-shape dimensions with independent scales."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.9, 1.0),
        probability: float = 0.5,
    ) -> None:
        self.scale_range = scale_range
        self.probability = probability

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        if "ego_shape" not in input_data or np.random.random() >= self.probability:
            return output

        scales = np.random.uniform(
            *self.scale_range, size=input_data["ego_shape"].shape
        )
        output["ego_shape"] = (input_data["ego_shape"] * scales).astype(
            input_data["ego_shape"].dtype,
            copy=False,
        )
        return output
