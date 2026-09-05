"""Random ego-pose augmentation and ego-centric scene transformation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike

POSE_AUGMENTATION_APPLIED_KEY = "_pose_augmentation_applied"


class PoseAugmentationCase(Enum):
    """Pose-noise profile selected for one planner frame."""

    NORMAL = "normal_case"
    STOPPED_IN_INTERSECTION = "stopped_in_intersection"


class PlannerPoseAugmentation:
    """Move the ego pose and recenter the scene without refining its future."""

    def __init__(
        self,
        normal_case: Mapping[str, Any],
        stopped_in_intersection: Mapping[str, Any],
    ) -> None:
        self.normal_case = dict(normal_case)
        self.stopped_in_intersection = dict(stopped_in_intersection)
        self._case_configs = {
            PoseAugmentationCase.NORMAL: self.normal_case,
            PoseAugmentationCase.STOPPED_IN_INTERSECTION: self.stopped_in_intersection,
        }

    def __call__(self, input_data: FrameLike) -> Frame:
        """Apply a pose offset and record whether it was applied."""
        case = self._select_case(input_data)
        case_config = self._case_configs[case]
        if case is PoseAugmentationCase.NORMAL and not has_sufficient_future_speed(
            input_data,
            int(case_config["pose_augmentation_speed_check_endpoint_index"]),
            float(case_config["pose_augmentation_endpoint_speed_threshold"]),
        ):
            output = dict(input_data)
            output[POSE_AUGMENTATION_APPLIED_KEY] = np.asarray(False)
            return output
        if np.random.random() >= float(case_config["probability"]):
            output = dict(input_data)
            output[POSE_AUGMENTATION_APPLIED_KEY] = np.asarray(False)
            return output
        longitudinal_range = tuple(case_config["longitudinal_offset_range"])
        lateral_range = tuple(case_config["lateral_offset_range"])
        yaw_range = tuple(case_config["yaw_offset_range"])
        longitudinal_offset = self._sample_offset(longitudinal_range)
        lateral_offset = self._sample_offset(lateral_range)
        yaw_offset = self._sample_offset(yaw_range)
        output, _ = apply_pose_augmentation(
            input_data, longitudinal_offset, lateral_offset, yaw_offset
        )
        output[POSE_AUGMENTATION_APPLIED_KEY] = np.asarray(True)
        return output

    def _select_case(self, input_data: FrameLike) -> PoseAugmentationCase:
        current_state = input_data["ego_agent_past"][-1]
        is_stopped = current_state[EGO_VELOCITY_INDEX] <= float(
            self.stopped_in_intersection["stopped_speed_threshold"]
        )
        if is_stopped and is_point_in_intersection(input_data, current_state[:2]):
            return PoseAugmentationCase.STOPPED_IN_INTERSECTION
        return PoseAugmentationCase.NORMAL

    @staticmethod
    def _sample_offset(offset_range: tuple[float, float]) -> float:
        if offset_range == (0.0, 0.0):
            return 0.0
        return float(np.random.uniform(*offset_range))


def has_sufficient_future_speed(
    input_data: FrameLike,
    check_index: int,
    speed_threshold: float,
) -> bool:
    """Check speed at a future endpoint index."""
    future = input_data.get("ego_agent_future")
    if future is None or len(future) == 0 or check_index < 0:
        return False
    endpoint = min(check_index, len(future) - 1)
    return bool(future[endpoint, EGO_VELOCITY_INDEX] > speed_threshold)


def is_point_in_intersection(input_data: FrameLike, point: NDArray[Any]) -> bool:
    """Return whether a point lies inside or on any valid intersection polygon."""
    areas = input_data.get("intersection_area")
    if areas is None:
        return False
    return any(
        _point_in_polygon(point, polygon)
        for polygon in areas
        if np.count_nonzero(polygon) > 0
    )


def _point_in_polygon(point: NDArray[Any], polygon: NDArray[Any]) -> bool:
    """Return whether a 2D point is inside a polygon using ray casting."""
    if len(polygon) < 3:
        return False
    px, py = map(float, point)
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = map(float, previous)
        x2, y2 = map(float, current)
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-9
            and min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9
        ):
            return True
        crosses_ray = (y1 > py) != (y2 > py)
        if crosses_ray:
            intersection_x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < intersection_x:
                inside = not inside
        previous = current
    return inside


def apply_pose_augmentation(
    input_data: FrameLike,
    longitudinal_offset: float,
    lateral_offset: float,
    yaw_offset: float,
) -> tuple[Frame, NDArray[Any] | None]:
    """Move the ego pose and recenter every spatial scene tensor."""
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
