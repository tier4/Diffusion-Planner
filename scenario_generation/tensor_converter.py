"""Convert SceneContext to normalized model input tensors.

Produces a dict of torch tensors ready for model.forward(), with:
1. Ego-centric coordinate transform centered on the chosen ego agent
2. Proper tensor shapes matching model expectations (dimensions.py)
3. Full observation normalization via model_args.observation_normalizer
"""

from __future__ import annotations

import math

import numpy as np
import torch

from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext
from scenario_generation.transforms import (
    _rotation_matrix,
    transform_cos_sin,
    transform_directions,
    transform_headings,
    transform_positions,
)

# Model dimension constants (from diffusion_planner.dimensions)
_INPUT_T = 30
_POSE_DIM = 4
_MAX_NUM_NEIGHBORS = 32
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
) -> np.ndarray:
    """Build ego_current_state: ``[1, 10]`` in ego frame.

    Convention matches the training data / C++ Autoware:
    - position (x, y) is (0, 0) — self-frame
    - heading (cos, sin) is (1, 0) — self-frame
    - **vx carries the full global speed magnitude**, vy is exactly 0
    - ax is the longitudinal projection of world-frame acceleration onto
      ego forward, ay is exactly 0
    - steering_angle and yaw_rate are scalars set by the physics updater

    Forcing vy=ay=0 matches the training NPZ convention — the car's motion
    is canonicalized to its own heading axis so the lateral dimension is
    only a kinematics cue (rate of heading change), never a velocity.
    """
    vel = ego.current_velocity  # world frame [Vx_w, Vy_w]
    speed = float(np.sqrt(vel[0] ** 2 + vel[1] ** 2))

    # Longitudinal acceleration = projection onto ego forward. Lateral = 0.
    accel = ego.acceleration  # world frame
    accel_ego = transform_directions(accel.reshape(1, 2), R).flatten()

    state = np.zeros(10, dtype=np.float32)
    state[0] = 0.0           # x
    state[1] = 0.0           # y
    state[2] = 1.0           # cos(0)
    state[3] = 0.0           # sin(0)
    state[4] = speed         # vx = |V| (full magnitude in ego frame)
    state[5] = 0.0           # vy = 0 (canonicalized to forward axis)
    state[6] = accel_ego[0]  # ax = longitudinal accel
    state[7] = 0.0           # ay = 0
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

        # Velocity, canonicalized to each neighbor's own heading axis — this
        # matches the C++ Autoware convention in
        # ``autoware_diffusion_planner/src/conversion/agent.cpp::as_array``
        # (lines 73-76): velocity_norm = hypot(vx_world, vy_world); then
        # (velocity_x, velocity_y) = velocity_norm * (cos(θ), sin(θ)) with θ
        # being the neighbor's own yaw. This zeros any lateral drift
        # (skid) and only encodes forward speed. We then rotate into ego
        # frame via R.
        if agent.past_velocities is not None:
            raw_vel = agent.past_velocities
        elif T_agent >= 2:
            raw_vel = np.zeros((T_agent, 2), dtype=np.float32)
            raw_vel[1:] = np.diff(traj[:, :2], axis=0) / scene.dt
        else:
            raw_vel = np.zeros((T_agent, 2), dtype=np.float32)
        speed = np.linalg.norm(raw_vel, axis=1).astype(np.float32)  # (T,)
        # Canonical world-frame velocity pointing along the neighbor's own
        # heading. traj[:, 2] is heading_rad.
        cos_hw = np.cos(traj[:, 2]).astype(np.float32)
        sin_hw = np.sin(traj[:, 2]).astype(np.float32)
        vel_world_canonical = np.stack(
            [speed * cos_hw, speed * sin_hw], axis=1,
        )
        vel = transform_directions(vel_world_canonical, R)

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

        # Zero out timesteps that were missing in the original (pre-transform) trajectory
        orig_invalid = np.sum(np.abs(traj[:, :2]), axis=-1) == 0
        feats[orig_invalid] = 0.0

        # Zero out pre-spawn history so other agents see the neighbor as
        # "just appeared". age_steps=0 means only the current frame (last
        # row) is real; age_steps>=T_agent means the full history is valid.
        if hasattr(agent, "age_steps") and agent.age_steps < T_agent:
            n_valid = max(1, agent.age_steps + 1)  # at least current frame
            feats[:T_agent - n_valid] = 0.0

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
    """Transform polygons to ego frame. Returns [1, N_poly, 40, 3].

    Always outputs D=3 [x, y, type] to match normalizer/model expectations.
    Inputs with D=2 are zero-padded in the type channel.
    """
    out = np.zeros((1, _NUM_POLYGONS, _POINTS_PER_POLYGON, 3), dtype=np.float32)
    N_src = min(polygons.shape[0], _NUM_POLYGONS)
    src_in = polygons[:N_src].copy().astype(np.float32)

    src = np.zeros((N_src, _POINTS_PER_POLYGON, 3), dtype=np.float32)
    D_in = min(src_in.shape[-1], 3) if src_in.ndim >= 3 else 2
    src[:, :, :D_in] = src_in[:, :, :D_in]

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
    """Transform line strings to ego frame. Returns [1, N_ls, 20, 4].

    Always outputs D=4 [x, y, stop_flag, border_flag] to match normalizer/model
    expectations. Inputs with D=2 are zero-padded in the flag channels.
    """
    out = np.zeros((1, _NUM_LINE_STRINGS, _POINTS_PER_LINE_STRING, 4), dtype=np.float32)
    N_src = min(line_strings.shape[0], _NUM_LINE_STRINGS)
    src_in = line_strings[:N_src].copy().astype(np.float32)

    src = np.zeros((N_src, _POINTS_PER_LINE_STRING, 4), dtype=np.float32)
    D_in = min(src_in.shape[-1], 4) if src_in.ndim >= 3 else 2
    src[:, :, :D_in] = src_in[:, :, :D_in]

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


class MapTensorCache:
    """Pre-computes padded, masked world-frame map arrays once per scene.

    Avoids repeated .copy().astype() and mask computation when converting
    the same scene for multiple agents (each with a different ego frame).
    """

    def __init__(self, map_data: MapData) -> None:
        # -- lanes: (n, 20, 33) --
        n_lanes = min(map_data.lanes.shape[0], _NUM_LANES)
        self._lanes = np.zeros((_NUM_LANES, _POINTS_PER_LANELET, _SEGMENT_POINT_DIM), dtype=np.float32)
        self._lanes[:n_lanes] = map_data.lanes[:n_lanes].astype(np.float32)
        self._lanes_mask = np.sum(np.abs(self._lanes[:, :, :8]), axis=-1) == 0

        # speed limits (no per-agent transform needed)
        self._lanes_speed_limit = np.zeros((1, _NUM_LANES, 1), dtype=np.float32)
        self._lanes_has_speed_limit = np.zeros((1, _NUM_LANES, 1), dtype=bool)
        n_sl = min(map_data.lanes_speed_limit.shape[0], _NUM_LANES)
        self._lanes_speed_limit[0, :n_sl] = map_data.lanes_speed_limit[:n_sl].astype(np.float32)
        self._lanes_has_speed_limit[0, :n_sl] = map_data.lanes_has_speed_limit[:n_sl].astype(bool)

        # -- static objects: (n, 10) --
        n_static = min(map_data.static_objects.shape[0], _NUM_STATIC)
        self._static = np.zeros((_NUM_STATIC, 10), dtype=np.float32)
        self._static[:n_static] = map_data.static_objects[:n_static].astype(np.float32)
        self._static_mask = np.sum(np.abs(self._static[:, :10]), axis=-1) == 0
        self._n_static = n_static

        # -- polygons: (n, 40, 3) --
        n_poly = min(map_data.polygons.shape[0], _NUM_POLYGONS)
        self._polygons = np.zeros((_NUM_POLYGONS, _POINTS_PER_POLYGON, 3), dtype=np.float32)
        if n_poly > 0:
            src_in = map_data.polygons[:n_poly].astype(np.float32)
            D_in = min(src_in.shape[-1], 3) if src_in.ndim >= 3 else 2
            self._polygons[:n_poly, :, :D_in] = src_in[:, :, :D_in]
        self._polygons_mask = np.sum(np.abs(self._polygons), axis=-1) == 0
        self._n_poly = n_poly

        # -- line strings: (n, 20, 4) --
        n_ls = min(map_data.line_strings.shape[0], _NUM_LINE_STRINGS)
        self._line_strings = np.zeros((_NUM_LINE_STRINGS, _POINTS_PER_LINE_STRING, 4), dtype=np.float32)
        if n_ls > 0:
            src_in = map_data.line_strings[:n_ls].astype(np.float32)
            D_in = min(src_in.shape[-1], 4) if src_in.ndim >= 3 else 2
            self._line_strings[:n_ls, :, :D_in] = src_in[:, :, :D_in]
        self._line_strings_mask = np.sum(np.abs(self._line_strings), axis=-1) == 0
        self._n_ls = n_ls

    def sync_tl_state(self, map_data: "MapData") -> None:
        """Refresh the TL one-hot channels (``[:, :, 8:13]``) from ``map_data``.

        ``MapTensorCache.__init__`` snapshots ``map_data.lanes`` (the
        ``.astype(np.float32)`` forces a fresh allocation, so the cache
        does NOT share storage with ``scene.map_data.lanes`` as earlier
        replay comments implied). Callers that mutate the live lanes
        tensor post-build — notably ``TrafficLightController.tick`` —
        must call this after every mutation, or the model will keep
        reading the snapshotted state (TL_NONE on every lanelet right
        after a lanelet-set refresh, since ``lanelet_to_33dim`` defaults
        to TL_NONE before the controller's first tick).

        Only the 5-dim TL one-hot is synced; geometry (``[0:8]``) and
        line-type channels (``[13:33]``) never change post-build.
        """
        n = min(map_data.lanes.shape[0], self._lanes.shape[0])
        if n > 0:
            # ``np.copyto`` with ``same_kind`` avoids the fresh allocation that
            # ``.astype(np.float32)`` forces every tick — map_data.lanes is
            # already float32 in the production path, so this is a pure memcpy.
            np.copyto(
                self._lanes[:n, :, 8:13],
                map_data.lanes[:n, :, 8:13],
                casting="same_kind",
            )

    def get_lanes_ego(self, R: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
        """Return lanes in ego frame: [1, NUM_LANES, 20, 33]."""
        src = self._lanes.copy()
        src[:, :, :2] = transform_positions(src[:, :, :2], R, ego_xy)
        src[:, :, 2:4] = transform_directions(src[:, :, 2:4], R)
        src[:, :, 4:6] = transform_directions(src[:, :, 4:6], R)
        src[:, :, 6:8] = transform_directions(src[:, :, 6:8], R)
        src[self._lanes_mask] = 0.0
        return src[np.newaxis]

    def get_static_objects_ego(self, R: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
        """Return static objects in ego frame: [1, 5, 10]."""
        src = self._static.copy()
        src[:, :2] = transform_positions(src[:, :2], R, ego_xy)
        src[:, 2:4] = transform_cos_sin(src[:, 2:4], R)
        src[self._static_mask] = 0.0
        return src[np.newaxis]

    def get_polygons_ego(self, R: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
        """Return polygons in ego frame: [1, NUM_POLYGONS, 40, 3]."""
        src = self._polygons.copy()
        src[:, :, :2] = transform_positions(src[:, :, :2], R, ego_xy)
        src[self._polygons_mask] = 0.0
        return src[np.newaxis]

    def get_line_strings_ego(self, R: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
        """Return line strings in ego frame: [1, NUM_LINE_STRINGS, 20, 4]."""
        src = self._line_strings.copy()
        src[:, :, :2] = transform_positions(src[:, :, :2], R, ego_xy)
        src[self._line_strings_mask] = 0.0
        return src[np.newaxis]

    @property
    def lanes_speed_limit(self) -> np.ndarray:
        """Return the cached speed-limit array. Treated as immutable by callers."""
        view = self._lanes_speed_limit.view()
        view.flags.writeable = False
        return view

    @property
    def lanes_has_speed_limit(self) -> np.ndarray:
        """Return the cached has-speed-limit mask. Treated as immutable by callers."""
        view = self._lanes_has_speed_limit.view()
        view.flags.writeable = False
        return view


def to_model_tensors(
    scene: SceneContext,
    ego_agent_id: str,
    model_args,
    device: str | torch.device = "cpu",
    map_cache: MapTensorCache | None = None,
    inference_delay: int = 0,
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
        map_cache: Optional pre-computed map tensor cache. When provided,
            avoids repeated copy/pad/mask of static map arrays across
            multiple agent conversions within the same scene.

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
    data_np["ego_current_state"] = _build_ego_current_state(ego, R)
    data_np["neighbor_agents_past"] = _build_neighbor_agents_past(
        scene, ego_agent_id, R, ego_xy, ego_heading,
    )
    if map_cache is not None:
        data_np["static_objects"] = map_cache.get_static_objects_ego(R, ego_xy)
        data_np["lanes"] = map_cache.get_lanes_ego(R, ego_xy)
        data_np["lanes_speed_limit"] = map_cache.lanes_speed_limit
        data_np["lanes_has_speed_limit"] = map_cache.lanes_has_speed_limit
        data_np["polygons"] = map_cache.get_polygons_ego(R, ego_xy)
        data_np["line_strings"] = map_cache.get_line_strings_ego(R, ego_xy)
    else:
        data_np["static_objects"] = _build_static_objects(scene.map_data.static_objects, R, ego_xy)
        data_np["lanes"] = _build_lanes(scene.map_data.lanes, R, ego_xy, _NUM_LANES)
        data_np["lanes_speed_limit"] = np.zeros((1, _NUM_LANES, 1), dtype=np.float32)
        data_np["lanes_has_speed_limit"] = np.zeros((1, _NUM_LANES, 1), dtype=bool)
        n_sl = min(scene.map_data.lanes_speed_limit.shape[0], _NUM_LANES)
        data_np["lanes_speed_limit"][0, :n_sl] = scene.map_data.lanes_speed_limit[:n_sl].astype(np.float32)
        data_np["lanes_has_speed_limit"][0, :n_sl] = scene.map_data.lanes_has_speed_limit[:n_sl].astype(bool)
        data_np["polygons"] = _build_polygons(scene.map_data.polygons, R, ego_xy)
        data_np["line_strings"] = _build_line_strings(scene.map_data.line_strings, R, ego_xy)

    route_lanes, route_sl, route_hsl = _build_route_lanes(ego, R, ego_xy)
    data_np["route_lanes"] = route_lanes
    data_np["route_lanes_speed_limit"] = route_sl
    data_np["route_lanes_has_speed_limit"] = route_hsl
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

    # Delay: number of prefix timesteps to keep fixed during diffusion.
    data_torch["delay"] = torch.tensor([inference_delay], dtype=torch.long, device=device)

    # Sampled trajectories: zeros = deterministic (caller can override for stochastic)
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    data_torch["sampled_trajectories"] = torch.zeros(
        1, P, future_len + 1, _POSE_DIM, dtype=torch.float32, device=device,
    )

    # Apply observation normalization
    data_torch = model_args.observation_normalizer(data_torch)

    return data_torch


def dump_step_npz(
    scene: SceneContext,
    map_cache: MapTensorCache,
    future_len: int,
    predicted_neighbor_num: int = _MAX_NUM_NEIGHBORS,
) -> dict[str, np.ndarray]:
    """Build un-normalized per-step observation arrays in training-NPZ format.

    Captures the scene as the model sees it for one replay step, but WITHOUT
    applying ``observation_normalizer`` (so world-scale lane widths etc. are
    preserved). GT-future arrays are zero-filled; downstream training/ranked-
    SFT generates its own futures.

    Call sites (e.g. replay.py dump_npz_dir) then np.savez the returned dict.

    Args:
        scene: Current scene at this replay step.
        map_cache: Pre-computed map tensor cache for the scene's map.
        future_len: Number of future timesteps (typically 80 — from model_args).
        predicted_neighbor_num: Neighbor slot count for the future placeholder.
            Must equal ``_MAX_NUM_NEIGHBORS`` (32) — the past array is built at
            that fixed shape, so a mismatch would produce NPZs where past and
            future disagree on the neighbor dimension and break the training
            NPZ loader.

    Returns:
        Dict with the standard NPZ keys (ego_agent_past, ego_current_state,
        neighbor_agents_past, lanes, line_strings, polygons, route_lanes,
        goal_pose, ego_shape, turn_indicators, ego_agent_future,
        neighbor_agents_future, version, plus speed_limit fields). Arrays are
        stripped of batch dim and have dtypes compatible with the training
        NPZ loader.
    """
    if predicted_neighbor_num != _MAX_NUM_NEIGHBORS:
        raise ValueError(
            f"predicted_neighbor_num={predicted_neighbor_num} disagrees with "
            f"the fixed past-neighbor shape {_MAX_NUM_NEIGHBORS}. Pass "
            f"{_MAX_NUM_NEIGHBORS} or omit (default). Threading a per-call "
            f"neighbor count through _build_neighbor_agents_past is a "
            f"follow-up if you need non-default future slots."
        )
    ego = scene.get_agent(scene.ego_agent_id)
    ego_xy = ego.current_position.astype(np.float64)
    ego_h = ego.current_heading
    R = _rotation_matrix(ego_h)

    data: dict[str, np.ndarray] = {}
    data["ego_agent_past"] = _build_ego_agent_past(ego, R, ego_xy, ego_h)
    data["ego_current_state"] = _build_ego_current_state(ego, R)
    data["neighbor_agents_past"] = _build_neighbor_agents_past(
        scene, scene.ego_agent_id, R, ego_xy, ego_h,
    )
    data["static_objects"] = map_cache.get_static_objects_ego(R, ego_xy)
    data["lanes"] = map_cache.get_lanes_ego(R, ego_xy)
    data["lanes_speed_limit"] = map_cache.lanes_speed_limit
    data["lanes_has_speed_limit"] = map_cache.lanes_has_speed_limit
    data["polygons"] = map_cache.get_polygons_ego(R, ego_xy)
    data["line_strings"] = map_cache.get_line_strings_ego(R, ego_xy)
    rl, rsl, rhsl = _build_route_lanes(ego, R, ego_xy)
    data["route_lanes"] = rl
    data["route_lanes_speed_limit"] = rsl
    data["route_lanes_has_speed_limit"] = rhsl
    data["goal_pose"] = _build_goal_pose(ego, R, ego_xy, ego_h)
    data["ego_shape"] = _build_ego_shape(ego)
    data["turn_indicators"] = _build_turn_indicators(ego)

    # Strip batch dim for NPZ storage (B=1 → [...]).
    for k, v in list(data.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1:
            data[k] = v[0]

    # Training NPZ format uses bool for has_speed_limit fields.
    for bk in ("lanes_has_speed_limit", "route_lanes_has_speed_limit"):
        if bk in data:
            data[bk] = data[bk].astype(bool)

    # Convert cos/sin back to heading-rad for keys the training loader re-
    # expands via heading_to_cos_sin at load time.
    if data["ego_agent_past"].shape[-1] == 4:
        ap = data["ego_agent_past"]
        data["ego_agent_past"] = np.stack(
            [ap[..., 0], ap[..., 1], np.arctan2(ap[..., 3], ap[..., 2])], axis=-1
        ).astype(np.float32)
    if data["goal_pose"].shape[-1] == 4:
        gp = data["goal_pose"]
        data["goal_pose"] = np.array(
            [gp[0], gp[1], float(np.arctan2(gp[3], gp[2]))], dtype=np.float32,
        )

    # GT-future placeholders (caller fills if desired; ranked-SFT ignores).
    data["ego_agent_future"] = np.zeros((future_len, 3), dtype=np.float32)
    data["neighbor_agents_future"] = np.zeros(
        (predicted_neighbor_num, future_len, 3), dtype=np.float32,
    )
    data["version"] = np.array(1, dtype=np.int64)

    return data
