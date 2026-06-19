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


def world_to_ego_frame_torch(
    batch: dict,  # dict[str, torch.Tensor], each [B, ...] on `device`
    dx,  # torch.Tensor [B]  ego_x per sample (scene frame)
    dy,  # torch.Tensor [B]  ego_y per sample
    dyaw,  # torch.Tensor [B] ego_heading per sample (radians)
) -> dict:
    """Batched, on-device mirror of :func:`world_to_ego_frame`.

    Applies the SAME rigid transform as the numpy version, but to a stacked batch of
    ``B`` frames each with its own ``(dx, dy, dyaw)``, entirely in torch on the tensors'
    device. Used by the reproducer's ``--gpu_transform`` path: the recorded frames are
    H2D'd untransformed and re-centered on the GPU (one batched op) instead of per-segment
    numpy on the CPU. Numerically equivalent to N calls of ``world_to_ego_frame`` (verified
    by ``scenario_generation/tests``); kept bit-for-bit faithful, INCLUDING the
    transform-then-zero-invalid order (translated zeros are non-zero, so masked padding is
    re-zeroed AFTER the transform).

    Operates in place on the tensors in ``batch`` and returns it. Only the spatial keys
    present are touched; non-spatial features (types, speeds, traffic lights) are untouched.
    """
    import torch

    c = torch.cos(dyaw)
    s = torch.sin(dyaw)
    # R[b] = [[c, s], [-s, c]] (rotate by -heading), matching _rotation_matrix.
    R = torch.stack(
        [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-2
    )  # [B, 2, 2]
    t = torch.stack([dx, dy], dim=-1)  # [B, 2]

    def _rot(vec, Rb):
        # vec [B, ..., 2], Rb [B, 2, 2]; out[...,k] = sum_j vec[...,j] * R[k,j]
        return torch.einsum("b...j,bkj->b...k", vec, Rb)

    def _pos(vec, Rb, tb):
        # broadcast t over the middle dims
        tt = tb.view(tb.shape[0], *([1] * (vec.dim() - 2)), 2)
        return _rot(vec - tt, Rb)

    # Match the float dtype of the spatial tensors. Pick the first FLOATING tensor (not just
    # the first key) — the batch also holds bool (*_has_speed_limit) / long (turn_indicators)
    # tensors, and R/t must stay float regardless of dict order.
    ref_dtype = next((v.dtype for v in batch.values() if v.is_floating_point()), torch.float32)
    R = R.to(ref_dtype)
    t = t.to(ref_dtype)

    if "ego_agent_past" in batch:
        ep = batch["ego_agent_past"]
        ep[:, :, :2] = _pos(ep[:, :, :2], R, t)
        ep[:, :, 2:4] = _rot(ep[:, :, 2:4], R)
    if "ego_current_state" in batch:
        ec = batch["ego_current_state"]
        ec[:, :2] = _pos(ec[:, :2].unsqueeze(1), R, t).squeeze(1)
        ec[:, 2:4] = _rot(ec[:, 2:4].unsqueeze(1), R).squeeze(1)
        ec[:, 4:6] = _rot(ec[:, 4:6].unsqueeze(1), R).squeeze(1)
        ec[:, 6:8] = _rot(ec[:, 6:8].unsqueeze(1), R).squeeze(1)
    if "neighbor_agents_past" in batch:
        nb = batch["neighbor_agents_past"]  # [B, N, T, 11]
        m = nb[:, :, :, :6].abs().sum(-1) == 0  # pre-transform padding mask [B,N,T]
        nb[:, :, :, :2] = _pos(nb[:, :, :, :2], R, t)
        nb[:, :, :, 2:4] = _rot(nb[:, :, :, 2:4], R)
        nb[:, :, :, 4:6] = _rot(nb[:, :, :, 4:6], R)
        nb[m] = 0.0
    for key in ("lanes", "route_lanes"):
        if key in batch:
            la = batch[key]  # [B, N, P, 33]
            m = la[:, :, :, :8].abs().sum(-1) == 0
            la[:, :, :, :2] = _pos(la[:, :, :, :2], R, t)
            la[:, :, :, 2:4] = _rot(la[:, :, :, 2:4], R)
            la[:, :, :, 4:6] = _rot(la[:, :, :, 4:6], R)
            la[:, :, :, 6:8] = _rot(la[:, :, :, 6:8], R)
            la[m] = 0.0
    if "polygons" in batch:
        pg = batch["polygons"]  # [B, N, 40, 3]
        m = pg.abs().sum(-1) == 0
        pg[:, :, :, :2] = _pos(pg[:, :, :, :2], R, t)
        pg[m] = 0.0
    if "line_strings" in batch:
        ls = batch["line_strings"]  # [B, N, 20, 4]
        m = ls.abs().sum(-1) == 0
        ls[:, :, :, :2] = _pos(ls[:, :, :, :2], R, t)
        ls[m] = 0.0
    if "static_objects" in batch:
        so = batch["static_objects"]  # [B, 5, 10]
        m = so[:, :, :10].abs().sum(-1) == 0
        so[:, :, :2] = _pos(so[:, :, :2], R, t)
        so[:, :, 2:4] = _rot(so[:, :, 2:4], R)
        so[m] = 0.0
    if "goal_pose" in batch:
        gp = batch["goal_pose"]  # [B, 4]
        gp[:, :2] = _pos(gp[:, :2].unsqueeze(1), R, t).squeeze(1)
        gp[:, 2:4] = _rot(gp[:, 2:4].unsqueeze(1), R).squeeze(1)

    return batch
