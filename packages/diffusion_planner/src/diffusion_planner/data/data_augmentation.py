"""NumPy ego-pose augmentation for planner dataset frames."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dimensions import EGO_VELOCITY_INDEX


class PlannerDataAugmentation:
    """Move ego along the y-axis, rotate it, and recenter one planner frame."""

    def __init__(
        self,
        lateral_offset_range: tuple[float, float] = (-1.0, 1.0),
        yaw_offset_range: tuple[float, float] = (-math.radians(5), math.radians(5)),
        pose_probability: float = 0.5,
        ego_speed_scale_range: tuple[float, float] = (0.8, 1.2),
        speed_probability: float = 0.5,
        num_refine: int = 10,
        time_step_s: float = 0.1,
        pose_augmentation_speed_threshold: float = 0.1,
    ) -> None:
        self.lateral_offset_range = lateral_offset_range
        self.yaw_offset_range = yaw_offset_range
        self.pose_probability = pose_probability
        self.ego_speed_scale_range = ego_speed_scale_range
        self.speed_probability = speed_probability
        self.num_refine = num_refine
        self.time_step_s = time_step_s
        self.pose_augmentation_speed_threshold = pose_augmentation_speed_threshold

    def __call__(self, input_data: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        """Apply pose and speed augmentation as independent random events."""
        output = input_data.copy()
        transformed_ego_future: NDArray[Any] | None = None
        current_speed = float(input_data["ego_agent_past"][-1, EGO_VELOCITY_INDEX])
        if (
            current_speed >= self.pose_augmentation_speed_threshold
            and np.random.random() < self.pose_probability
        ):
            ego_current = input_data["ego_agent_past"][-1, :4]
            ego_position = ego_current[:2]
            ego_heading = _normalize(ego_current[2:4])
            lateral_offset = np.random.uniform(*self.lateral_offset_range)
            yaw_offset = np.random.uniform(*self.yaw_offset_range)

            augmented_position = ego_position.copy()
            augmented_position[1] += lateral_offset
            augmented_heading = _rotate(ego_heading, yaw_offset)

            output["ego_agent_past"] = _transform_pose_tensor(
                input_data["ego_agent_past"], ego_position, ego_heading
            )
            if "ego_agent_future" in input_data:
                transformed_ego_future = _transform_pose_tensor(
                    input_data["ego_agent_future"],
                    augmented_position,
                    augmented_heading,
                )
                output["ego_agent_future"] = transformed_ego_future
            output["neighbor_agents_past"] = _transform_pose_tensor(
                input_data["neighbor_agents_past"],
                augmented_position,
                augmented_heading,
            )
            if "neighbor_agents_future" in input_data:
                output["neighbor_agents_future"] = _transform_pose_tensor(
                    input_data["neighbor_agents_future"],
                    augmented_position,
                    augmented_heading,
                )
            output["goal_pose"] = _transform_pose_tensor(
                input_data["goal_pose"], augmented_position, augmented_heading
            )
            output["lanes"] = _transform_lane_tensor(
                input_data["lanes"], augmented_position, augmented_heading
            )
            output["route_lanes"] = _transform_lane_tensor(
                input_data["route_lanes"], augmented_position, augmented_heading
            )
            for key in ("intersection_area", "stop_lines", "road_borders"):
                output[key] = _transform_point_tensor(
                    input_data[key], augmented_position, augmented_heading
                )

        if np.random.random() < self.speed_probability:
            output["ego_agent_past"] = output["ego_agent_past"].copy()
            ego_speed_scale = np.random.uniform(*self.ego_speed_scale_range)
            output["ego_agent_past"][..., EGO_VELOCITY_INDEX] *= ego_speed_scale

        if transformed_ego_future is not None:
            output["ego_agent_future"] = _refine_ego_future(
                transformed_ego_future,
                output["ego_agent_past"],
                self.num_refine,
                self.time_step_s,
            )

        return output


def _normalize(vector: NDArray[Any]) -> NDArray[Any]:
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _rotate(vector: NDArray[Any], angle: float) -> NDArray[Any]:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        )
    )


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


def _vectors_to_local(vectors: NDArray[Any], heading: NDArray[Any]) -> NDArray[Any]:
    x = vectors[..., 0] * heading[0] + vectors[..., 1] * heading[1]
    y = -vectors[..., 0] * heading[1] + vectors[..., 1] * heading[0]
    return np.stack((x, y), axis=-1)


def _points_to_local(
    points: NDArray[Any], position: NDArray[Any], heading: NDArray[Any]
) -> NDArray[Any]:
    return _vectors_to_local(points - position, heading)


def _transform_pose_tensor(
    values: NDArray[Any],
    position: NDArray[Any],
    heading: NDArray[Any],
) -> NDArray[Any]:
    valid = np.count_nonzero(values[..., :4], axis=-1) > 0
    result = values.copy()
    result[..., :2] = _points_to_local(values[..., :2], position, heading)
    result[..., 2:4] = _vectors_to_local(values[..., 2:4], heading)
    result[~valid] = 0
    return result


def _transform_point_tensor(
    values: NDArray[Any],
    position: NDArray[Any],
    heading: NDArray[Any],
) -> NDArray[Any]:
    valid = np.count_nonzero(values, axis=(-2, -1)) > 0
    result = _points_to_local(values, position, heading)
    result[~valid] = 0
    return result.astype(values.dtype, copy=False)


def _transform_lane_tensor(
    values: NDArray[Any],
    position: NDArray[Any],
    heading: NDArray[Any],
) -> NDArray[Any]:
    valid = np.count_nonzero(values, axis=(-2, -1)) > 0
    result = values.copy()
    result[..., :2] = _points_to_local(values[..., :2], position, heading)
    result[..., 2:4] = _vectors_to_local(values[..., 2:4], heading)
    result[..., 4:6] = _vectors_to_local(values[..., 4:6], heading)
    result[~valid] = 0
    return result
