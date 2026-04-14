"""Convert SceneContext to normalized model input tensors.

Produces a dict of torch tensors ready for model.forward(), with:
1. Ego-centric coordinate transform centered on the chosen ego agent
2. Proper tensor shapes matching model expectations (dimensions.py)
3. Full observation normalization via model_args.observation_normalizer
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from scenario_generation.scene_context import Agent, AgentType, SceneContext
from scenario_generation.transforms import (
    _rotation_matrix,
    transform_cos_sin,
    transform_directions,
    transform_headings,
    transform_positions,
)

if TYPE_CHECKING:
    pass

# Model dimension constants (from diffusion_planner.dimensions)
_INPUT_T = 30
_OUTPUT_T = 80
_POSE_DIM = 4
_MAX_NUM_NEIGHBORS = 32
_MAX_NUM_AGENTS = _MAX_NUM_NEIGHBORS + 1
_NUM_LANES = 140
_NUM_ROUTE = 25
_NUM_POLYGONS = 10
_NUM_LINE_STRINGS = 60
_POINTS_PER_LANELET = 20
_POINTS_PER_POLYGON = 40
_POINTS_PER_LINE_STRING = 20
_SEGMENT_POINT_DIM = 33
_NUM_STATIC = 5


def _pad_or_truncate(arr: np.ndarray, target_len: int, axis: int = 0) -> np.ndarray:
    """Pad with zeros or truncate (keeping the most recent entries) along an axis."""
    current = arr.shape[axis]
    if current == target_len:
        return arr
    if current > target_len:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(current - target_len, current)
        return arr[tuple(slices)]
    # Pad at the beginning (oldest entries are zeros)
    pad_shape = list(arr.shape)
    pad_shape[axis] = target_len - current
    padding = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([padding, arr], axis=axis)


def _heading_to_cos_sin(heading: np.ndarray) -> np.ndarray:
    """Convert heading radians to [cos, sin] pairs. Input: (...), Output: (..., 2)."""
    return np.stack([np.cos(heading), np.sin(heading)], axis=-1).astype(np.float32)


def _build_ego_agent_past(
    ego: Agent,
    R: np.ndarray,
    ego_xy: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """Build ego_agent_past tensor: [1, INPUT_T+1, 4] [x, y, cos_h, sin_h].

    In ego-centric frame, the current position is (0, 0) with heading=0.
    """
    T_needed = _INPUT_T + 1
    traj = ego.past_trajectory  # (T, 3) [x, y, heading_rad]

    # Transform to ego frame
    xy_ego = transform_positions(traj[:, :2], R, ego_xy)
    h_ego = transform_headings(traj[:, 2], ego_heading)
    cos_sin = _heading_to_cos_sin(h_ego)

    # [T, 4]: x, y, cos_h, sin_h
    past = np.concatenate([xy_ego, cos_sin], axis=-1).astype(np.float32)
    past = _pad_or_truncate(past, T_needed, axis=0)
    return past[np.newaxis]  # [1, T, 4]


def _build_ego_current_state(
    ego: Agent,
    R: np.ndarray,
    ego_xy: np.ndarray,
) -> np.ndarray:
    """Build ego_current_state: [1, 10] all zeros for position (at origin)."""
    vel = ego.current_velocity
    vel_ego = transform_directions(vel.reshape(1, 2), R).flatten()

    accel = ego.acceleration
    accel_ego = transform_directions(accel.reshape(1, 2), R).flatten()

    state = np.zeros(10, dtype=np.float32)
    state[0] = 0.0  # x (at origin)
    state[1] = 0.0  # y (at origin)
    state[2] = 1.0  # cos(0)
    state[3] = 0.0  # sin(0)
    state[4] = vel_ego[0]
    state[5] = vel_ego[1]
    state[6] = accel_ego[0]
    state[7] = accel_ego[1]
    state[8] = ego.steering_angle
    state[9] = ego.yaw_rate
    return state[np.newaxis]  # [1, 10]


def _build_neighbor_agents_past(
    scene: SceneContext,
    ego_id: str,
    R: np.ndarray,
    ego_xy: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """Build neighbor_agents_past: [1, 32, INPUT_T+1, 11].

    Neighbors sorted by distance from ego (closest first).
    """
    T_needed = _INPUT_T + 1
    out = np.zeros((1, _MAX_NUM_NEIGHBORS, T_needed, 11), dtype=np.float32)

    # Collect non-ego agents with their distance to ego
    neighbors_with_dist: list[tuple[float, Agent]] = []
    for agent in scene.agents:
        if agent.id == ego_id:
            continue
        pos = agent.current_position
        dist = np.sqrt((pos[0] - ego_xy[0]) ** 2 + (pos[1] - ego_xy[1]) ** 2)
        neighbors_with_dist.append((dist, agent))

    # Sort by distance
    neighbors_with_dist.sort(key=lambda x: x[0])

    for slot_idx, (_, agent) in enumerate(neighbors_with_dist[:_MAX_NUM_NEIGHBORS]):
        traj = agent.past_trajectory  # (T, 3) [x, y, heading_rad]
        T_agent = traj.shape[0]

        # Transform positions and headings
        xy_ego = transform_positions(traj[:, :2], R, ego_xy)
        h_ego = transform_headings(traj[:, 2], ego_heading)
        cos_h = np.cos(h_ego).astype(np.float32)
        sin_h = np.sin(h_ego).astype(np.float32)

        # Transform velocities
        if agent.past_velocities is not None:
            vel = transform_directions(agent.past_velocities, R)
        else:
            vel = np.zeros((T_agent, 2), dtype=np.float32)
            if T_agent >= 2:
                diffs = np.diff(traj[:, :2], axis=0) / scene.dt
                vel[1:] = transform_directions(diffs, R)

        # Agent type one-hot
        type_vec = np.zeros(3, dtype=np.float32)
        if agent.agent_type == AgentType.VEHICLE:
            type_vec[0] = 1.0
        elif agent.agent_type == AgentType.PEDESTRIAN:
            type_vec[1] = 1.0
        elif agent.agent_type == AgentType.BICYCLE:
            type_vec[2] = 1.0

        # Build per-timestep features: [x, y, cos_h, sin_h, vx, vy, width, length, type(3)]
        feats = np.zeros((T_agent, 11), dtype=np.float32)
        feats[:, 0] = xy_ego[:, 0]
        feats[:, 1] = xy_ego[:, 1]
        feats[:, 2] = cos_h
        feats[:, 3] = sin_h
        feats[:, 4] = vel[:, 0]
        feats[:, 5] = vel[:, 1]
        feats[:, 6] = agent.width
        feats[:, 7] = agent.length
        feats[:, 8:11] = type_vec

        # Zero out timesteps where position is (0,0) -- indicates missing data
        invalid = (np.abs(feats[:, 0]) < 1e-8) & (np.abs(feats[:, 1]) < 1e-8)
        # But don't zero out if the agent genuinely is at origin
        if np.sum(np.abs(traj[:, :2])) > 1e-6:
            orig_invalid = np.sum(np.abs(traj[:, :2]), axis=-1) == 0
            feats[orig_invalid] = 0.0

        feats = _pad_or_truncate(feats, T_needed, axis=0)
        out[0, slot_idx] = feats

    return out


def _build_lanes(
    lanes: np.ndarray,
    R: np.ndarray,
    ego_xy: np.ndarray,
    target_n: int,
) -> np.ndarray:
    """Transform lane segments to ego frame. Returns [1, target_n, 20, 33]."""
    N_src = lanes.shape[0]
    out = np.zeros((1, target_n, _POINTS_PER_LANELET, _SEGMENT_POINT_DIM), dtype=np.float32)

    n = min(N_src, target_n)
    src = lanes[:n].copy().astype(np.float32)

    mask = np.sum(np.abs(src[:, :, :8]), axis=-1) == 0

    # Center XY: translate + rotate
    src[:, :, :2] = transform_positions(src[:, :, :2], R, ego_xy)
    # Direction dXdY: rotate only
    src[:, :, 2:4] = transform_directions(src[:, :, 2:4], R)
    # Left boundary offset: rotate only
    src[:, :, 4:6] = transform_directions(src[:, :, 4:6], R)
    # Right boundary offset: rotate only
    src[:, :, 6:8] = transform_directions(src[:, :, 6:8], R)

    src[mask] = 0.0
    out[0, :n] = src
    return out


def _build_static_objects(
    static: np.ndarray,
    R: np.ndarray,
    ego_xy: np.ndarray,
) -> np.ndarray:
    """Transform static objects to ego frame. Returns [1, 5, 10]."""
    out = np.zeros((1, _NUM_STATIC, 10), dtype=np.float32)
    N_src = min(static.shape[0], _NUM_STATIC)
    src = static[:N_src].copy().astype(np.float32)

    mask = np.sum(np.abs(src[:, :10]), axis=-1) == 0
    src[:, :2] = transform_positions(src[:, :2], R, ego_xy)
    src[:, 2:4] = transform_cos_sin(src[:, 2:4], R)
    src[mask] = 0.0

    out[0, :N_src] = src
    return out


def _build_polygons(
    polygons: np.ndarray,
    R: np.ndarray,
    ego_xy: np.ndarray,
) -> np.ndarray:
    """Transform polygons to ego frame. Returns [1, N_poly, 40, D].

    D matches the input (typically 2 for [x,y] or 3 for [x,y,type]).
    """
    D = polygons.shape[-1] if polygons.ndim >= 3 else 2
    out = np.zeros((1, _NUM_POLYGONS, _POINTS_PER_POLYGON, D), dtype=np.float32)
    N_src = min(polygons.shape[0], _NUM_POLYGONS)
    src = polygons[:N_src].copy().astype(np.float32)

    mask = np.sum(np.abs(src), axis=-1) == 0
    src[:, :, :2] = transform_positions(src[:, :, :2], R, ego_xy)
    src[mask] = 0.0

    out[0, :N_src] = src
    return out


def _build_line_strings(
    line_strings: np.ndarray,
    R: np.ndarray,
    ego_xy: np.ndarray,
) -> np.ndarray:
    """Transform line strings to ego frame. Returns [1, N_ls, 20, D].

    D matches the input (typically 2 for [x,y] or 4 for [x,y,t1,t2]).
    """
    D = line_strings.shape[-1] if line_strings.ndim >= 3 else 2
    out = np.zeros((1, _NUM_LINE_STRINGS, _POINTS_PER_LINE_STRING, D), dtype=np.float32)
    N_src = min(line_strings.shape[0], _NUM_LINE_STRINGS)
    src = line_strings[:N_src].copy().astype(np.float32)

    mask = np.sum(np.abs(src), axis=-1) == 0
    src[:, :, :2] = transform_positions(src[:, :, :2], R, ego_xy)
    src[mask] = 0.0

    out[0, :N_src] = src
    return out


def _build_goal_pose(
    ego: Agent,
    R: np.ndarray,
    ego_xy: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """Build goal_pose: [1, 4] [x, y, cos_h, sin_h] in ego frame."""
    out = np.zeros((1, 4), dtype=np.float32)
    if ego.goal_pose is not None:
        gp = ego.goal_pose  # (3,) [x, y, heading_rad]
        xy_ego = transform_positions(gp[:2].reshape(1, 2), R, ego_xy).flatten()
        h_ego = float(gp[2]) - ego_heading
        out[0] = [xy_ego[0], xy_ego[1], math.cos(h_ego), math.sin(h_ego)]
    return out


def _build_ego_shape(ego: Agent) -> np.ndarray:
    """Build ego_shape: [1, 3] [wheelbase, length, width]."""
    return np.array([[ego.wheelbase, ego.length, ego.width]], dtype=np.float32)


def _build_turn_indicators(ego: Agent) -> np.ndarray:
    """Build turn_indicators: [1, INPUT_T+1] int. Default: all zeros (NONE).

    The encoder slices [:, :-1] internally, so we provide INPUT_T+1 entries.
    """
    T = _INPUT_T + 1
    out = np.zeros((1, T), dtype=np.int32)
    if ego.turn_indicators is not None:
        ti = ego.turn_indicators
        ti = _pad_or_truncate(ti, T, axis=0)
        out[0] = ti.astype(np.int32)
    return out


def _build_route_lanes(
    ego: Agent,
    R: np.ndarray,
    ego_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build route lane tensors in ego frame.

    Returns:
        (route_lanes [1, 25, 20, 33],
         route_lanes_speed_limit [1, 25, 1],
         route_lanes_has_speed_limit [1, 25, 1])
    """
    if ego.route_lanes is not None:
        lanes = _build_lanes(ego.route_lanes, R, ego_xy, _NUM_ROUTE)
    else:
        lanes = np.zeros((1, _NUM_ROUTE, _POINTS_PER_LANELET, _SEGMENT_POINT_DIM), dtype=np.float32)

    sl = np.zeros((1, _NUM_ROUTE, 1), dtype=np.float32)
    hsl = np.zeros((1, _NUM_ROUTE, 1), dtype=np.float32)

    if ego.route_speed_limit is not None:
        n = min(ego.route_speed_limit.shape[0], _NUM_ROUTE)
        sl[0, :n] = ego.route_speed_limit[:n].astype(np.float32)
    if ego.route_has_speed_limit is not None:
        n = min(ego.route_has_speed_limit.shape[0], _NUM_ROUTE)
        hsl[0, :n] = ego.route_has_speed_limit[:n].astype(np.float32)

    return lanes, sl, hsl


def to_model_tensors(
    scene: SceneContext,
    ego_agent_id: str,
    model_args,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert SceneContext to normalized model input tensors.

    Transforms all scene data into the ego-centric frame of the specified
    agent, builds tensors matching the model's expected shapes, and applies
    observation normalization.

    Args:
        scene: The scene to convert.
        ego_agent_id: ID of the agent to use as ego.
        model_args: Model arguments object with:
            - observation_normalizer: ObservationNormalizer instance
            - predicted_neighbor_num: int
            - future_len: int (typically 80)
        device: Target device for tensors.

    Returns:
        Dict of normalized torch tensors ready for model.forward().
        Includes sampled_trajectories initialized to zeros (deterministic).
    """
    ego = scene.get_agent(ego_agent_id)
    ego_xy = ego.current_position.astype(np.float64)
    ego_heading = ego.current_heading

    R = _rotation_matrix(ego_heading)

    # Build all numpy arrays
    data_np: dict[str, np.ndarray] = {}
    data_np["ego_agent_past"] = _build_ego_agent_past(ego, R, ego_xy, ego_heading)
    data_np["ego_current_state"] = _build_ego_current_state(ego, R, ego_xy)
    data_np["neighbor_agents_past"] = _build_neighbor_agents_past(
        scene, ego_agent_id, R, ego_xy, ego_heading,
    )
    data_np["static_objects"] = _build_static_objects(scene.map_data.static_objects, R, ego_xy)
    data_np["lanes"] = _build_lanes(scene.map_data.lanes, R, ego_xy, _NUM_LANES)
    data_np["lanes_speed_limit"] = np.zeros((1, _NUM_LANES, 1), dtype=np.float32)
    data_np["lanes_has_speed_limit"] = np.zeros((1, _NUM_LANES, 1), dtype=np.float32)
    n_sl = min(scene.map_data.lanes_speed_limit.shape[0], _NUM_LANES)
    data_np["lanes_speed_limit"][0, :n_sl] = scene.map_data.lanes_speed_limit[:n_sl].astype(np.float32)
    data_np["lanes_has_speed_limit"][0, :n_sl] = scene.map_data.lanes_has_speed_limit[:n_sl].astype(np.float32)

    route_lanes, route_sl, route_hsl = _build_route_lanes(ego, R, ego_xy)
    data_np["route_lanes"] = route_lanes
    data_np["route_lanes_speed_limit"] = route_sl
    data_np["route_lanes_has_speed_limit"] = route_hsl

    data_np["polygons"] = _build_polygons(scene.map_data.polygons, R, ego_xy)
    data_np["line_strings"] = _build_line_strings(scene.map_data.line_strings, R, ego_xy)
    data_np["goal_pose"] = _build_goal_pose(ego, R, ego_xy, ego_heading)
    data_np["ego_shape"] = _build_ego_shape(ego)
    data_np["turn_indicators"] = _build_turn_indicators(ego)

    # Convert to torch tensors
    _bool_keys = {"lanes_has_speed_limit", "route_lanes_has_speed_limit"}
    data_torch: dict[str, torch.Tensor] = {}
    for key, arr in data_np.items():
        if key in _bool_keys:
            data_torch[key] = torch.tensor(arr, dtype=torch.bool, device=device)
        else:
            data_torch[key] = torch.tensor(arr, dtype=torch.float32, device=device)
    data_torch["turn_indicators"] = torch.tensor(
        data_np["turn_indicators"], dtype=torch.long, device=device
    )

    # Delay: always 0 for inference
    data_torch["delay"] = torch.zeros(1, dtype=torch.long, device=device)

    # Sampled trajectories: zeros = deterministic (caller can override for stochastic)
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    data_torch["sampled_trajectories"] = torch.zeros(
        1, P, future_len + 1, _POSE_DIM, dtype=torch.float32, device=device,
    )

    # Apply observation normalization
    data_torch = model_args.observation_normalizer(data_torch)

    return data_torch
