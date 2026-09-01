"""Goal-position correction from a nearby stopped ego future."""

from __future__ import annotations

import numpy as np

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike
from .rigid_augmentation import recenter_frame_to_pose


class PlannerGoalTransform:
    """Replace a nearby goal position with the stopped ego-future endpoint."""

    def __init__(
        self,
        stop_speed_threshold: float = 0.1,
        time_shift_probability: float = 0.5,
        goal_tolerance_m: float = 1.0,
    ) -> None:
        self.stop_speed_threshold = stop_speed_threshold
        self.time_shift_probability = time_shift_probability
        self.goal_tolerance_m = goal_tolerance_m

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        goal_pose = input_data.get("goal_pose")
        ego_future = input_data.get("ego_agent_future")
        if goal_pose is None or ego_future is None or len(ego_future) == 0:
            return output

        terminal_state = ego_future[-1]
        goal_distance = float(np.linalg.norm(terminal_state[:2] - goal_pose[:2]))
        terminal_speed = float(terminal_state[EGO_VELOCITY_INDEX])
        if (
            goal_distance <= self.goal_tolerance_m
            and terminal_speed <= self.stop_speed_threshold
        ):
            output["goal_pose"] = np.array(goal_pose, copy=True)
            output["goal_pose"][:2] = terminal_state[:2]

        return self._shift_ego_time(output)

    def _shift_ego_time(self, input_data: FrameLike) -> Frame:
        """Move ego time forward while leaving every other sequence unshifted."""
        output = dict(input_data)
        past = input_data["ego_agent_past"]
        future = input_data["ego_agent_future"]
        goal_pose = input_data["goal_pose"]
        if (
            np.linalg.norm(future[-1, :2] - goal_pose[:2]) > self.goal_tolerance_m
            or np.random.random() >= self.time_shift_probability
        ):
            return output

        if len(future) < 2:
            return output
        shift_steps = int(np.random.randint(1, len(future)))

        ego_trajectory = np.concatenate((past, future), axis=0)
        new_past = ego_trajectory[shift_steps : shift_steps + len(past)].copy()
        new_future = np.empty_like(future)
        remaining = len(future) - shift_steps
        new_future[:remaining] = future[shift_steps:]
        goal_state = np.zeros(future.shape[-1], dtype=future.dtype)
        goal_state[:4] = goal_pose[:4]
        new_future[remaining:] = goal_state

        shifted = dict(input_data)
        shifted["ego_agent_past"] = new_past
        shifted["ego_agent_future"] = new_future
        current_pose = new_past[-1, :4]
        heading = current_pose[2:4]
        heading = heading / max(float(np.linalg.norm(heading)), 1e-6)
        return recenter_frame_to_pose(shifted, current_pose[:2], heading)
