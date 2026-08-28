"""Numerical checks for planner data augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

POSE_KEYS = frozenset(
    {
        "ego_agent_past",
        "ego_agent_future",
        "neighbor_agents_past",
        "neighbor_agents_future",
        "goal_pose",
    }
)
LANE_KEYS = frozenset({"lanes", "route_lanes"})
POINT_KEYS = frozenset({"intersection_area", "stop_lines", "road_borders"})
COORDINATE_KEYS = POSE_KEYS | LANE_KEYS | POINT_KEYS


def _padding_preserved(key: str, original: np.ndarray, augmented: np.ndarray) -> bool:
    if key in POSE_KEYS and original.ndim >= 2:
        invalid = np.count_nonzero(original[..., :4], axis=-1) == 0
        return bool(np.all(augmented[invalid] == 0))
    if key in LANE_KEYS | POINT_KEYS and original.ndim >= 2:
        invalid = np.count_nonzero(original, axis=tuple(range(1, original.ndim))) == 0
        return bool(np.all(augmented[invalid] == 0))
    return True


def _valid_points(key: str, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if key in POSE_KEYS:
        features = array[..., :4]
        points = array[..., :2]
    elif key in LANE_KEYS:
        features = array
        points = array[..., :2]
    else:
        features = array
        points = array
    valid = np.count_nonzero(features, axis=-1) > 0
    return points, valid


def _rigid_distance_error(
    key: str, original: np.ndarray, augmented: np.ndarray
) -> float | None:
    # Ego future is intentionally reshaped near the current pose by the quintic
    # refinement, so it is not a rigid transform of the original trajectory.
    if key not in COORDINATE_KEYS or key == "ego_agent_future":
        return None
    original_points, valid = _valid_points(key, original)
    augmented_points, _ = _valid_points(key, augmented)
    original_valid = original_points[valid].reshape(-1, 2)
    augmented_valid = augmented_points[valid].reshape(-1, 2)
    if len(original_valid) < 2:
        return 0.0
    original_distance = np.linalg.norm(original_valid - original_valid[0], axis=-1)
    augmented_distance = np.linalg.norm(augmented_valid - augmented_valid[0], axis=-1)
    return float(np.max(np.abs(original_distance - augmented_distance)))


def _yaw_norm_error(key: str, augmented: np.ndarray) -> float | None:
    if key not in POSE_KEYS:
        return None
    poses = augmented[..., :4]
    valid = np.count_nonzero(poses, axis=-1) > 0
    if not np.any(valid):
        return 0.0
    yaw_norm = np.linalg.norm(poses[..., 2:4][valid], axis=-1)
    return float(np.max(np.abs(yaw_norm - 1.0)))


def inspect_augmentation(
    original: Mapping[str, Any], augmented: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return per-tensor invariants and differences for dashboard display."""
    rows: list[dict[str, Any]] = []
    for key in sorted(original):
        original_array = np.asarray(original[key])
        augmented_array = np.asarray(augmented[key])
        numeric = np.issubdtype(original_array.dtype, np.number)
        max_difference = (
            float(np.max(np.abs(augmented_array - original_array)))
            if numeric and original_array.size
            else 0.0
        )
        padding_preserved = _padding_preserved(key, original_array, augmented_array)
        rigid_error = _rigid_distance_error(key, original_array, augmented_array)
        yaw_error = _yaw_norm_error(key, augmented_array)
        unchanged_required = key not in COORDINATE_KEYS
        valid = padding_preserved
        if unchanged_required:
            valid = valid and np.array_equal(original_array, augmented_array)
        if rigid_error is not None:
            valid = valid and rigid_error < 1e-4
        if yaw_error is not None:
            valid = valid and yaw_error < 1e-4
        rows.append(
            {
                "tensor": key,
                "changed": not np.array_equal(original_array, augmented_array),
                "padding_preserved": padding_preserved,
                "max_difference": max_difference,
                "rigid_distance_error": rigid_error,
                "yaw_norm_error": yaw_error,
                "valid": valid,
            }
        )
    return rows
