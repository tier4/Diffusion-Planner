"""Ego-pose augmentation without future-trajectory refinement."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike


class PlannerRigidDataAugmentation:
    """Move the ego pose and rigidly recenter the scene without path refinement."""

    def __init__(
        self,
        longitudinal_offset_range: tuple[float, float] = (0.0, 0.0),
        lateral_offset_range: tuple[float, float] = (-1.0, 1.0),
        yaw_offset_range: tuple[float, float] = (-math.radians(5), math.radians(5)),
        pose_probability: float = 0.5,
        pose_augmentation_speed_threshold: float = 0.1,
        pose_augmentation_speed_check_index: int = 20,
    ) -> None:
        self.longitudinal_offset_range = longitudinal_offset_range
        self.lateral_offset_range = lateral_offset_range
        self.yaw_offset_range = yaw_offset_range
        self.pose_probability = pose_probability
        self.pose_augmentation_speed_threshold = pose_augmentation_speed_threshold
        self.pose_augmentation_speed_check_index = pose_augmentation_speed_check_index

    def __call__(self, input_data: FrameLike) -> Frame:
        """Apply the pre-refinement pose augmentation behavior."""
        if (
            not has_sufficient_future_speed(
                input_data,
                self.pose_augmentation_speed_check_index,
                self.pose_augmentation_speed_threshold,
            )
            or np.random.random() >= self.pose_probability
        ):
            return dict(input_data)
        longitudinal_offset = 0.0
        if any(value != 0.0 for value in self.longitudinal_offset_range):
            longitudinal_offset = np.random.uniform(*self.longitudinal_offset_range)
        lateral_offset = np.random.uniform(*self.lateral_offset_range)
        yaw_offset = np.random.uniform(*self.yaw_offset_range)
        output, _ = apply_rigid_pose_augmentation(
            input_data, longitudinal_offset, lateral_offset, yaw_offset
        )
        return output


def has_sufficient_future_speed(
    input_data: FrameLike,
    check_index: int,
    speed_threshold: float,
) -> bool:
    """Return whether every ego-future speed through an index meets a threshold."""
    future = input_data.get("ego_agent_future")
    if future is None or len(future) == 0 or check_index < 0:
        return False
    endpoint = min(check_index, len(future) - 1)
    speeds = future[: endpoint + 1, EGO_VELOCITY_INDEX]
    return bool(np.all(speeds >= speed_threshold))


def apply_rigid_pose_augmentation(
    input_data: FrameLike,
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
) -> tuple[Frame, NDArray[Any] | None]:
    """Move the ego pose and rigidly recenter every spatial scene tensor."""
    ego_pose = input_data["ego_agent_past"][-1, :4]
    shifted_pose = get_shifted_pose(
        ego_pose, longitudinal_offset, lateral_offset, yaw_offset
    )
    output = recenter_frame_to_pose(input_data, shifted_pose[:2], shifted_pose[2:4])
    output["ego_agent_past"] = _transform_pose_tensor(
        input_data["ego_agent_past"], ego_pose[:2], _normalize(ego_pose[2:4])
    )
    return output, output.get("ego_agent_future")


def get_shifted_pose(
    ego_pose: NDArray[Any],
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
) -> NDArray[Any]:
    """Return ego xy and heading after applying local pose offsets."""
    shifted_pose = np.array(ego_pose, copy=True)
    shifted_pose[0] += longitudinal_offset
    shifted_pose[1] += lateral_offset
    shifted_pose[2:4] = _rotate(_normalize(ego_pose[2:4]), yaw_offset)
    return shifted_pose


def recenter_frame_to_pose(
    input_data: FrameLike,
    position: NDArray[Any],
    heading: NDArray[Any],
) -> Frame:
    """Express every spatial frame tensor relative to one ego pose."""
    output = dict(input_data)
    for key in (
        "ego_agent_past",
        "ego_agent_future",
        "neighbor_agents_past",
        "neighbor_agents_future",
        "goal_pose",
    ):
        if key in input_data:
            output[key] = _transform_pose_tensor(input_data[key], position, heading)
    for key in ("lanes", "route_lanes"):
        if key in input_data:
            output[key] = _transform_lane_tensor(input_data[key], position, heading)
    for key in ("intersection_area", "stop_lines", "road_borders"):
        if key in input_data:
            output[key] = _transform_point_tensor(input_data[key], position, heading)
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
