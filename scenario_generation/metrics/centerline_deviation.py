"""Centerline lateral offset from route_lanes in the current ego frame."""

from __future__ import annotations

import numpy as np


def _valid_centerline_points(route_lanes: np.ndarray) -> np.ndarray:
    valid = np.abs(route_lanes[..., :2]).sum(axis=-1) > 1e-6
    pts = route_lanes[..., :2][valid]
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def lateral_offset_from_route_lanes(route_lanes: np.ndarray) -> float:
    """Min distance from ego origin (0,0) to any route_lanes centerline point (m).

      At each closed-loop step the observation is re-centered so the live ego sits
    at the origin; this matches the training centerline reward geometry.
    """
    pts = _valid_centerline_points(np.asarray(route_lanes))
    if pts.shape[0] == 0:
        return float("inf")
    return float(np.linalg.norm(pts, axis=-1).min())


def lateral_offset_series(route_lanes_list: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [lateral_offset_from_route_lanes(rl) for rl in route_lanes_list], dtype=np.float32
    )
