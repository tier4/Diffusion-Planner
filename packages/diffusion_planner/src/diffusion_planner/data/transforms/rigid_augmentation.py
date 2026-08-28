"""Ego-pose augmentation without future-trajectory refinement."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Frame, FrameLike


class PlannerRigidDataAugmentation:
    """Move the ego pose and rigidly recenter the scene without path refinement."""

    def __init__(
        self,
        lateral_offset_range: tuple[float, float] = (-1.0, 1.0),
        yaw_offset_range: tuple[float, float] = (-math.radians(5), math.radians(5)),
        pose_probability: float = 0.5,
    ) -> None:
        self.lateral_offset_range = lateral_offset_range
        self.yaw_offset_range = yaw_offset_range
        self.pose_probability = pose_probability

    def __call__(self, input_data: FrameLike) -> Frame:
        """Apply the pre-refinement pose augmentation behavior."""
        if np.random.random() >= self.pose_probability:
            return dict(input_data)
        output, _ = _apply_rigid_pose_augmentation(
            input_data,
            np.random.uniform(*self.lateral_offset_range),
            np.random.uniform(*self.yaw_offset_range),
        )
        return output


def _apply_rigid_pose_augmentation(
    input_data: FrameLike,
    lateral_offset: float,
    yaw_offset: float,
) -> tuple[Frame, NDArray[Any] | None]:
    """Move the ego pose and rigidly recenter every spatial scene tensor."""
    output = dict(input_data)
    ego_current = input_data["ego_agent_past"][-1, :4]
    ego_position = ego_current[:2]
    ego_heading = _normalize(ego_current[2:4])
    augmented_position = ego_position.copy()
    augmented_position[1] += lateral_offset
    augmented_heading = _rotate(ego_heading, yaw_offset)

    output["ego_agent_past"] = _transform_pose_tensor(
        input_data["ego_agent_past"], ego_position, ego_heading
    )
    transformed_ego_future: NDArray[Any] | None = None
    if "ego_agent_future" in input_data:
        transformed_ego_future = _transform_pose_tensor(
            input_data["ego_agent_future"], augmented_position, augmented_heading
        )
        output["ego_agent_future"] = transformed_ego_future
    output["neighbor_agents_past"] = _transform_pose_tensor(
        input_data["neighbor_agents_past"], augmented_position, augmented_heading
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
    return output, transformed_ego_future


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
