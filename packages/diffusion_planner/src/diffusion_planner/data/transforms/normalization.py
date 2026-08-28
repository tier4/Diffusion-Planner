"""NumPy preprocessing for planner feature normalization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Frame, FrameLike


class PlannerDataNormalizer:
    """Normalize continuous frame features before conversion to tensors."""

    def __init__(
        self,
        position_scale: float = 50.0,
        speed_scale: float = 15.0,
        vehicle_shape_scale: float = 10.0,
    ) -> None:
        self.position_scale = position_scale
        self.speed_scale = speed_scale
        self.vehicle_shape_scale = vehicle_shape_scale

    def __call__(self, frame: FrameLike) -> Frame:
        """Return a normalized shallow copy of one unbatched frame."""
        normalized = dict(frame)
        for key in (
            "ego_agent_past",
            "neighbor_agents_past",
            "ego_agent_future",
            "neighbor_agents_future",
            "goal_pose",
        ):
            if key in frame:
                normalized[key] = self.normalize_trajectory(frame[key])

        for key in (
            "lanes",
            "route_lanes",
            "intersection_area",
            "stop_lines",
            "road_borders",
        ):
            if key in frame:
                normalized[key] = frame[key] / self.position_scale

        for key in ("lanes_speed_limit", "route_lanes_speed_limit"):
            if key in frame:
                normalized[key] = frame[key] / self.speed_scale

        for key in ("agent_shape", "ego_shape"):
            if key in frame:
                normalized[key] = frame[key] / self.vehicle_shape_scale
        return normalized

    def normalize_trajectory(self, trajectory: NDArray[Any]) -> NDArray[Any]:
        """Normalize xy while preserving all remaining trajectory features."""
        result = np.array(trajectory, copy=True)
        result[..., :2] /= self.position_scale
        return result

    def denormalize_trajectory(self, trajectory: NDArray[Any]) -> NDArray[Any]:
        """Restore meter units and project trajectory yaw onto the unit circle."""
        result = np.array(trajectory, copy=True)
        result[..., :2] *= self.position_scale
        return self.normalize_yaw_vector(result)

    @staticmethod
    def normalize_yaw_vector(trajectory: NDArray[Any]) -> NDArray[Any]:
        """Project trajectory cos/sin pairs onto the unit circle."""
        result = np.array(trajectory, copy=True)
        yaw = result[..., 2:4]
        norm = np.maximum(np.linalg.norm(yaw, axis=-1, keepdims=True), 1e-6)
        result[..., 2:4] = yaw / norm
        return result
