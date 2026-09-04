"""Rigid transforms for the new schema, kept byte-for-byte equivalent in meaning to new DP."""

from __future__ import annotations

import numpy as np


def _vectors_to_local(vectors: np.ndarray, heading: np.ndarray) -> np.ndarray:
    return np.stack((vectors[..., 0] * heading[0] + vectors[..., 1] * heading[1],
                     -vectors[..., 0] * heading[1] + vectors[..., 1] * heading[0]), axis=-1)


def _pose(values: np.ndarray, position: np.ndarray, heading: np.ndarray) -> np.ndarray:
    valid = np.count_nonzero(values[..., :4], axis=-1) > 0
    out = values.copy()
    out[..., :2] = _vectors_to_local(values[..., :2] - position, heading)
    out[..., 2:4] = _vectors_to_local(values[..., 2:4], heading)
    out[~valid] = 0
    return out


def _points(values: np.ndarray, position: np.ndarray, heading: np.ndarray) -> np.ndarray:
    valid = np.count_nonzero(values, axis=(-2, -1)) > 0
    out = _vectors_to_local(values - position, heading)
    out[~valid] = 0
    return out.astype(values.dtype, copy=False)


def _lanes(values: np.ndarray, position: np.ndarray, heading: np.ndarray) -> np.ndarray:
    valid = np.count_nonzero(values, axis=(-2, -1)) > 0
    out = values.copy()
    out[..., :2] = _vectors_to_local(values[..., :2] - position, heading)
    out[..., 2:4] = _vectors_to_local(values[..., 2:4], heading)
    out[..., 4:6] = _vectors_to_local(values[..., 4:6], heading)
    out[~valid] = 0
    return out


def recenter_frame_to_pose(frame: dict[str, np.ndarray], position: np.ndarray,
                           heading: np.ndarray) -> dict[str, np.ndarray]:
    """Express every spatial new-schema tensor relative to ``position/heading``."""
    heading = np.asarray(heading) / max(float(np.linalg.norm(heading)), 1e-6)
    out = dict(frame)
    for key in ("ego_agent_past", "ego_agent_future", "neighbor_agents_past",
                "neighbor_agents_future", "goal_pose"):
        if key in frame:
            out[key] = _pose(frame[key], position, heading)
    for key in ("lanes", "route_lanes"):
        if key in frame:
            out[key] = _lanes(frame[key], position, heading)
    for key in ("intersection_area", "stop_lines", "road_borders"):
        if key in frame:
            out[key] = _points(frame[key], position, heading)
    return out

