"""Coordinate transform utilities for ego-centric conversion.

Follows the same convention as StatePerturbation / state_update.py:
    R = [[cos_h, sin_h], [-sin_h, cos_h]]
This rotates by -heading, making the heading direction become [1, 0].
"""

from __future__ import annotations

import math

import numpy as np


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (heading about +Z) from a quaternion using the standard
    ZYX-Euler reduction.

    For a unit quaternion this matches ``tf2::getYaw`` (ROS) and
    ``Rotation.from_quat([qx, qy, qz, qw]).as_euler('xyz')[2]``. Returned
    angle is in radians, range ``(-pi, pi]``.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _rotation_matrix(heading: float) -> np.ndarray:
    """Build 2x2 rotation matrix for world-to-ego transform.

    Returns R such that R @ (p - ego_xy) gives ego-centric coordinates
    when ego has the given heading.
    """
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def transform_positions(
    xy: np.ndarray,
    R: np.ndarray,
    ego_xy: np.ndarray,
) -> np.ndarray:
    """Translate then rotate positions into ego frame.

    Args:
        xy: (..., 2) positions in world frame.
        R: (2, 2) rotation matrix from _rotation_matrix().
        ego_xy: (2,) ego position in world frame.

    Returns:
        (..., 2) positions in ego frame.
    """
    translated = xy - ego_xy
    shape = translated.shape
    flat = translated.reshape(-1, 2)
    rotated = (R @ flat.T).T
    return rotated.reshape(shape).astype(np.float32)


def transform_directions(
    dxy: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Rotate direction vectors (no translation).

    Used for velocity, acceleration, lane direction (dX, dY),
    and lane boundary offsets (which are relative to centerline).

    Args:
        dxy: (..., 2) direction vectors in world frame.
        R: (2, 2) rotation matrix.

    Returns:
        (..., 2) direction vectors in ego frame.
    """
    shape = dxy.shape
    flat = dxy.reshape(-1, 2)
    rotated = (R @ flat.T).T
    return rotated.reshape(shape).astype(np.float32)


def transform_headings(
    headings: np.ndarray | float,
    ego_heading: float,
) -> np.ndarray:
    """Subtract ego heading to get ego-relative heading.

    Args:
        headings: (...) heading angles in radians (world frame).
        ego_heading: Ego heading in radians.

    Returns:
        (...) ego-relative headings in radians.
    """
    return np.asarray(headings - ego_heading, dtype=np.float32)


def transform_cos_sin(
    cos_sin: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Rotate (cos_h, sin_h) pairs using the rotation matrix.

    This is equivalent to vector_transform on heading unit vectors,
    matching the convention in state_update.py.

    Args:
        cos_sin: (..., 2) [cos(heading), sin(heading)] in world frame.
        R: (2, 2) rotation matrix.

    Returns:
        (..., 2) [cos(heading), sin(heading)] in ego frame.
    """
    return transform_directions(cos_sin, R)


def world_to_ego_frame(
    scene_data: dict[str, np.ndarray],
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> dict[str, np.ndarray]:
    """Transform a full model-input dict from scene frame to ego-centric frame.

    Applies the same transforms as state_update.py:
    - Positions (x, y): translate by -ego_xy then rotate
    - Direction vectors (dx, dy, vx, vy, boundary offsets): rotate only
    - Headings (cos, sin): rotate as unit vectors
    - Non-spatial features (traffic lights, types, speeds): unchanged

    Args:
        scene_data: Dict of numpy arrays matching model input keys, in scene frame.
        ego_x, ego_y, ego_heading: Ego pose in scene frame.

    Returns:
        New dict with all arrays transformed to ego-centric frame.
    """
    R = _rotation_matrix(ego_heading)
    ego_xy = np.array([ego_x, ego_y], dtype=np.float64)
    out: dict[str, np.ndarray] = {}

    for key, arr in scene_data.items():
        out[key] = arr.copy()

    # --- ego_agent_past [1, T, 4]: x, y, cos_h, sin_h ---
    if "ego_agent_past" in out:
        ep = out["ego_agent_past"]
        ep[0, :, :2] = transform_positions(ep[0, :, :2], R, ego_xy)
        ep[0, :, 2:4] = transform_cos_sin(ep[0, :, 2:4], R)

    # --- ego_current_state [1, 10]: x, y, cos, sin, vx, vy, ax, ay, steer, yaw ---
    if "ego_current_state" in out:
        ec = out["ego_current_state"]
        ec[0, :2] = transform_positions(ec[0, :2].reshape(1, 2), R, ego_xy).flatten()
        ec[0, 2:4] = transform_cos_sin(ec[0, 2:4].reshape(1, 2), R).flatten()
        ec[0, 4:6] = transform_directions(ec[0, 4:6].reshape(1, 2), R).flatten()
        ec[0, 6:8] = transform_directions(ec[0, 6:8].reshape(1, 2), R).flatten()

    # --- neighbor_agents_past [1, N, T, 11] ---
    if "neighbor_agents_past" in out:
        nb = out["neighbor_agents_past"]
        mask = np.sum(np.abs(nb[0, :, :, :6]), axis=-1) == 0
        nb[0, :, :, :2] = transform_positions(nb[0, :, :, :2], R, ego_xy)
        nb[0, :, :, 2:4] = transform_cos_sin(nb[0, :, :, 2:4], R)
        nb[0, :, :, 4:6] = transform_directions(nb[0, :, :, 4:6], R)
        nb[0][mask] = 0.0

    # --- lanes [1, N_lanes, 20, 33] ---
    if "lanes" in out:
        la = out["lanes"]
        mask = np.sum(np.abs(la[0, :, :, :8]), axis=-1) == 0
        la[0, :, :, :2] = transform_positions(la[0, :, :, :2], R, ego_xy)
        la[0, :, :, 2:4] = transform_directions(la[0, :, :, 2:4], R)
        la[0, :, :, 4:6] = transform_directions(la[0, :, :, 4:6], R)  # left boundary offset
        la[0, :, :, 6:8] = transform_directions(la[0, :, :, 6:8], R)  # right boundary offset
        la[0][mask] = 0.0

    # --- route_lanes [1, N_route, 20, 33] ---
    if "route_lanes" in out:
        rl = out["route_lanes"]
        mask = np.sum(np.abs(rl[0, :, :, :8]), axis=-1) == 0
        rl[0, :, :, :2] = transform_positions(rl[0, :, :, :2], R, ego_xy)
        rl[0, :, :, 2:4] = transform_directions(rl[0, :, :, 2:4], R)
        rl[0, :, :, 4:6] = transform_directions(rl[0, :, :, 4:6], R)
        rl[0, :, :, 6:8] = transform_directions(rl[0, :, :, 6:8], R)
        rl[0][mask] = 0.0

    # --- polygons [1, N_poly, 40, 3]: x, y, type ---
    if "polygons" in out:
        pg = out["polygons"]
        mask = np.sum(np.abs(pg[0]), axis=-1) == 0
        pg[0, :, :, :2] = transform_positions(pg[0, :, :, :2], R, ego_xy)
        pg[0][mask] = 0.0

    # --- line_strings [1, N_ls, 20, 4]: x, y, type1, type2 ---
    if "line_strings" in out:
        ls = out["line_strings"]
        mask = np.sum(np.abs(ls[0]), axis=-1) == 0
        ls[0, :, :, :2] = transform_positions(ls[0, :, :, :2], R, ego_xy)
        ls[0][mask] = 0.0

    # --- static_objects [1, 5, 10]: x, y, cos_h, sin_h, w, l, type(4) ---
    if "static_objects" in out:
        so = out["static_objects"]
        mask = np.sum(np.abs(so[0, :, :10]), axis=-1) == 0
        so[0, :, :2] = transform_positions(so[0, :, :2], R, ego_xy)
        so[0, :, 2:4] = transform_cos_sin(so[0, :, 2:4], R)
        so[0][mask] = 0.0

    # --- goal_pose [1, 4]: x, y, cos_h, sin_h ---
    if "goal_pose" in out:
        gp = out["goal_pose"]
        if gp.ndim == 2:
            gp[0, :2] = transform_positions(gp[0, :2].reshape(1, 2), R, ego_xy).flatten()
            gp[0, 2:4] = transform_cos_sin(gp[0, 2:4].reshape(1, 2), R).flatten()

    return out
