"""Shift recent ego starts from history into the future."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike
from .pose_augmentation import recenter_frame_to_pose


class PlannerStartDecisionAugmentation:
    """Move a recent start transition from ego history to ego future."""

    def __init__(
        self,
        probability: float = 0.5,
        stop_speed_threshold: float = 0.1,
        max_shift_steps: int = 10,
    ) -> None:
        self.probability = probability
        self.stop_speed_threshold = stop_speed_threshold
        self.max_shift_steps = max_shift_steps
        if max_shift_steps < 1:
            raise ValueError("max_shift_steps must be positive")

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        past = input_data.get("ego_agent_past")
        future = input_data.get("ego_agent_future")
        if past is None or future is None or len(past) == 0 or len(future) == 0:
            return output

        sequence = np.concatenate((past, future), axis=0)
        start_index = self._find_start_index(sequence[:, EGO_VELOCITY_INDEX], len(past))
        if start_index is None or np.random.random() >= self.probability:
            return output

        shift_steps = len(past) - start_index
        shifted = self._shift_right(sequence, shift_steps)
        output["ego_agent_past"] = shifted[: len(past)]
        output["ego_agent_future"] = shifted[len(past) :]
        current_pose = output["ego_agent_past"][-1, :4]
        heading = current_pose[2:4]
        heading = heading / max(float(np.linalg.norm(heading)), 1e-6)
        return recenter_frame_to_pose(output, current_pose[:2], heading)

    def _find_start_index(self, speeds: NDArray[Any], past_length: int) -> int | None:
        first_candidate = max(1, past_length - self.max_shift_steps)
        for index in range(first_candidate, past_length):
            was_stopped = speeds[index - 1] <= self.stop_speed_threshold
            starts_moving = speeds[index] > self.stop_speed_threshold
            if was_stopped and starts_moving:
                return index
        return None

    @staticmethod
    def _shift_right(sequence: NDArray[Any], shift_steps: int) -> NDArray[Any]:
        shifted = np.empty_like(sequence)
        shifted[:shift_steps] = sequence[0]
        shifted[shift_steps:] = sequence[:-shift_steps]
        return shifted
