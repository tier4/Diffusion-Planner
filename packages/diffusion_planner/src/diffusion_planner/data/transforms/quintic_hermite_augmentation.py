"""Ego-pose augmentation with quintic Hermite future refinement."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike
from .rigid_augmentation import apply_rigid_pose_augmentation


class PlannerQuinticHermiteAugmentation:
    """Move the ego pose and reconnect its future with a quintic Hermite curve."""

    def __init__(
        self,
        longitudinal_offset_range: tuple[float, float] = (0.0, 0.0),
        lateral_offset_range: tuple[float, float] = (-1.0, 1.0),
        yaw_offset_range: tuple[float, float] = (-math.radians(5), math.radians(5)),
        pose_probability: float = 0.5,
        num_refine: int = 10,
        time_step_s: float = 0.1,
        pose_augmentation_speed_threshold: float = 0.1,
    ) -> None:
        self.longitudinal_offset_range = longitudinal_offset_range
        self.lateral_offset_range = lateral_offset_range
        self.yaw_offset_range = yaw_offset_range
        self.pose_probability = pose_probability
        self.num_refine = num_refine
        self.time_step_s = time_step_s
        self.pose_augmentation_speed_threshold = pose_augmentation_speed_threshold

    def __call__(self, input_data: FrameLike) -> Frame:
        """Apply pose augmentation and smoothly reconnect the ego future."""
        output = dict(input_data)
        transformed_ego_future: NDArray[Any] | None = None
        current_speed = float(input_data["ego_agent_past"][-1, EGO_VELOCITY_INDEX])
        if (
            current_speed >= self.pose_augmentation_speed_threshold
            and np.random.random() < self.pose_probability
        ):
            longitudinal_offset = 0.0
            if any(value != 0.0 for value in self.longitudinal_offset_range):
                longitudinal_offset = np.random.uniform(*self.longitudinal_offset_range)
            lateral_offset = np.random.uniform(*self.lateral_offset_range)
            yaw_offset = np.random.uniform(*self.yaw_offset_range)
            output, transformed_ego_future = apply_rigid_pose_augmentation(
                input_data, longitudinal_offset, lateral_offset, yaw_offset
            )

        if transformed_ego_future is not None:
            output["ego_agent_future"] = _refine_ego_future(
                transformed_ego_future,
                output["ego_agent_past"],
                self.num_refine,
                self.time_step_s,
            )

        return output


def _refine_ego_future(
    future: NDArray[Any],
    past: NDArray[Any],
    num_refine: int,
    time_step_s: float,
) -> NDArray[Any]:
    """Join the augmented current state to the original future with a quintic."""
    if len(future) == 0 or num_refine < 0 or time_step_s <= 0.0:
        return future

    endpoint_index = min(num_refine, len(future) - 1)
    if endpoint_index < 2:
        return future

    result = future.copy()
    start = past[-1]
    endpoint = future[endpoint_index]
    start_heading = math.atan2(float(start[3]), float(start[2]))
    endpoint_heading = math.atan2(float(endpoint[3]), float(endpoint[2]))

    start_speed = float(start[4])
    start_acceleration = 0.0
    if len(past) >= 2:
        start_acceleration = (float(past[-1, 4]) - float(past[-2, 4])) / time_step_s
    start_yaw_rate = float(start[5])

    p0 = future[endpoint_index - 2, :2].astype(np.float64)
    p1 = future[endpoint_index - 1, :2].astype(np.float64)
    p2 = endpoint[:2].astype(np.float64)
    endpoint_speed = float(np.linalg.norm(p2 - p1) / time_step_s)
    endpoint_acceleration = float(np.linalg.norm(p2 - 2.0 * p1 + p0) / time_step_s**2)
    previous_heading = math.atan2(
        float(future[endpoint_index - 1, 3]),
        float(future[endpoint_index - 1, 2]),
    )
    endpoint_yaw_rate = (
        _normalize_angle(endpoint_heading - previous_heading) / time_step_s
    )

    duration = (endpoint_index + 1) * time_step_s
    start_velocity = start_speed * np.asarray(
        [math.cos(start_heading), math.sin(start_heading)]
    )
    start_second_derivative = np.asarray(
        [
            start_acceleration * math.cos(start_heading)
            - start_speed * math.sin(start_heading) * start_yaw_rate,
            start_acceleration * math.sin(start_heading)
            + start_speed * math.cos(start_heading) * start_yaw_rate,
        ]
    )
    endpoint_velocity = endpoint_speed * np.asarray(
        [math.cos(endpoint_heading), math.sin(endpoint_heading)]
    )
    endpoint_second_derivative = np.asarray(
        [
            endpoint_acceleration * math.cos(endpoint_heading)
            - endpoint_speed * math.sin(endpoint_heading) * endpoint_yaw_rate,
            endpoint_acceleration * math.sin(endpoint_heading)
            + endpoint_speed * math.cos(endpoint_heading) * endpoint_yaw_rate,
        ]
    )
    coefficients = _quintic_coefficients(
        start[:2],
        start_velocity,
        start_second_derivative,
        endpoint[:2],
        endpoint_velocity,
        endpoint_second_derivative,
        duration,
    )

    previous_position = start[:2].astype(np.float64)
    previous_refined_heading = start_heading
    for index in range(endpoint_index + 1):
        time = (index + 1) * time_step_s
        powers = np.asarray([1.0, time, time**2, time**3, time**4, time**5])
        position = powers @ coefficients
        velocity = (
            np.asarray(
                [0.0, 1.0, 2.0 * time, 3.0 * time**2, 4.0 * time**3, 5.0 * time**4]
            )
            @ coefficients
        )
        heading = math.atan2(
            position[1] - previous_position[1],
            position[0] - previous_position[0],
        )
        result[index, :2] = position
        result[index, 2:4] = (math.cos(heading), math.sin(heading))
        result[index, 4] = np.linalg.norm(velocity)
        result[index, 5] = (
            _normalize_angle(heading - previous_refined_heading) / time_step_s
        )
        previous_position = position
        previous_refined_heading = heading
    return result


def _quintic_coefficients(
    start_position: NDArray[Any],
    start_velocity: NDArray[Any],
    start_acceleration: NDArray[Any],
    end_position: NDArray[Any],
    end_velocity: NDArray[Any],
    end_acceleration: NDArray[Any],
    duration: float,
) -> NDArray[np.float64]:
    """Return polynomial coefficients as rows a0..a5 and columns x/y."""
    coefficients = np.zeros((6, 2), dtype=np.float64)
    coefficients[0] = start_position
    coefficients[1] = start_velocity
    coefficients[2] = start_acceleration / 2.0
    matrix = np.asarray(
        [
            [duration**3, duration**4, duration**5],
            [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
            [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
        ]
    )
    known_end = (
        coefficients[0] + coefficients[1] * duration + coefficients[2] * duration**2
    )
    right_hand_side = np.stack(
        (
            end_position - known_end,
            end_velocity - coefficients[1] - 2.0 * coefficients[2] * duration,
            end_acceleration - 2.0 * coefficients[2],
        )
    )
    coefficients[3:] = np.linalg.solve(matrix, right_hand_side)
    return coefficients


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
