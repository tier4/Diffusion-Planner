"""Load NPZ files into SceneContext.

NPZ files store data in ego-centric frame (ego at origin, heading=0).
This becomes the scene frame for the resulting SceneContext.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext

# Default vehicle dimensions when ego_shape is absent in NPZ
_DEFAULT_WHEELBASE = 2.79
_DEFAULT_LENGTH = 4.34
_DEFAULT_WIDTH = 1.70

# Wheelbase-to-length ratio for deriving wheelbase from neighbor shape
_WHEELBASE_LENGTH_RATIO = 0.65


def _correct_heading_flip(
    past_traj: np.ndarray,
    past_vel: np.ndarray | None,
    future_traj: np.ndarray | None = None,
    speed_threshold: float = 0.3,
) -> bool:
    """Detect and correct 180-degree heading errors using velocity direction.

    Checks the agent's heading against its movement direction. If they differ
    by more than 90 degrees, flips all headings (past and future) by pi.

    Args:
        past_traj: (T, 3) [x, y, heading_rad] -- modified in-place.
        past_vel: (T, 2) [vx, vy] or None.
        future_traj: (T_future, 3) [x, y, heading_rad] or None -- modified in-place.
        speed_threshold: Minimum speed (m/s) to trust velocity direction.

    Returns:
        True if headings were flipped.
    """
    # Try velocity-based check on recent timesteps
    for t in range(past_traj.shape[0] - 1, max(past_traj.shape[0] - 10, -1), -1):
        if past_vel is not None:
            vx, vy = past_vel[t, 0], past_vel[t, 1]
        elif t > 0:
            dt = 0.1
            vx = (past_traj[t, 0] - past_traj[t - 1, 0]) / dt
            vy = (past_traj[t, 1] - past_traj[t - 1, 1]) / dt
        else:
            continue

        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed < speed_threshold:
            continue

        vel_heading = np.arctan2(vy, vx)
        agent_heading = past_traj[t, 2]
        diff = (agent_heading - vel_heading + np.pi) % (2 * np.pi) - np.pi

        if abs(diff) > np.pi / 2:
            past_traj[:, 2] += np.pi
            past_traj[:, 2] = (past_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
            if future_traj is not None:
                future_traj[:, 2] += np.pi
                future_traj[:, 2] = (future_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
            return True
        return False

    # Fallback: try future trajectory direction if past had no speed
    if future_traj is not None and future_traj.shape[0] >= 2:
        dx = future_traj[1, 0] - future_traj[0, 0]
        dy = future_traj[1, 1] - future_traj[0, 1]
        speed = np.sqrt(dx ** 2 + dy ** 2) / 0.1
        if speed >= speed_threshold:
            move_heading = np.arctan2(dy, dx)
            agent_heading = past_traj[-1, 2]
            diff = (agent_heading - move_heading + np.pi) % (2 * np.pi) - np.pi
            if abs(diff) > np.pi / 2:
                past_traj[:, 2] += np.pi
                past_traj[:, 2] = (past_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
                future_traj[:, 2] += np.pi
                future_traj[:, 2] = (future_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
                return True

    return False


def _agent_type_from_onehot(is_vehicle: float, is_ped: float, is_bike: float) -> AgentType:
    """Map one-hot neighbor type flags to AgentType enum."""
    vals = [is_vehicle, is_ped, is_bike]
    idx = int(np.argmax(vals))
    return [AgentType.VEHICLE, AgentType.PEDESTRIAN, AgentType.BICYCLE][idx]


def _cos_sin_to_heading(cos_h: np.ndarray, sin_h: np.ndarray) -> np.ndarray:
    """Convert (cos, sin) arrays to heading angle in radians."""
    return np.arctan2(sin_h, cos_h).astype(np.float32)


def _extract_ego_agent(
    data: dict[str, np.ndarray],
    wheelbase: float,
    length: float,
    width: float,
) -> Agent:
    """Build the ego Agent from NPZ arrays."""
    # ego_agent_past: (T, 3) [x, y, heading_rad] or (T, 4) [x, y, cos_h, sin_h]
    ego_past = data["ego_agent_past"]
    if ego_past.shape[-1] == 3:
        past_traj = ego_past.astype(np.float32)
    else:
        past_traj = np.stack([
            ego_past[:, 0],
            ego_past[:, 1],
            _cos_sin_to_heading(ego_past[:, 2], ego_past[:, 3]),
        ], axis=-1).astype(np.float32)

    # ego_current_state: (10,) [x, y, cos, sin, vx, vy, ax, ay, steer, yaw_rate]
    eco = data.get("ego_current_state")
    if eco is not None:
        eco = eco.flatten()
        vx, vy = float(eco[4]), float(eco[5])
        accel = np.array([eco[6], eco[7]], dtype=np.float32)
        steering = float(eco[8])
        yaw_rate = float(eco[9])
    else:
        vx, vy = 0.0, 0.0
        accel = np.zeros(2, dtype=np.float32)
        steering = 0.0
        yaw_rate = 0.0

    # Ego velocity: only current state (per-timestep not stored in NPZ)
    T = past_traj.shape[0]
    past_vel = np.zeros((T, 2), dtype=np.float32)
    past_vel[-1] = [vx, vy]
    # Derive past velocities from trajectory differences
    if T >= 2:
        diffs = np.diff(past_traj[:, :2], axis=0) / 0.1
        past_vel[1:] = diffs

    # Ground truth future
    future = data.get("ego_agent_future")
    if future is not None:
        future = future.astype(np.float32)
        if future.shape[-1] == 4:
            future = np.stack([
                future[:, 0], future[:, 1],
                _cos_sin_to_heading(future[:, 2], future[:, 3]),
            ], axis=-1).astype(np.float32)

    # Goal pose: (3,) or (4,) [x, y, heading] or [x, y, cos, sin]
    goal = data.get("goal_pose")
    if goal is not None:
        goal = goal.flatten()
        if goal.shape[0] == 4:
            goal = np.array([goal[0], goal[1], _cos_sin_to_heading(goal[2:3], goal[3:4]).item()],
                            dtype=np.float32)
        else:
            goal = goal[:3].astype(np.float32)

    # Route lanes
    route_lanes = data.get("route_lanes")
    route_sl = data.get("route_lanes_speed_limit")
    route_hsl = data.get("route_lanes_has_speed_limit")

    # Turn indicators
    turn_ind = data.get("turn_indicators")
    if turn_ind is not None:
        turn_ind = turn_ind.flatten().astype(np.int32)

    return Agent(
        id="ego",
        agent_type=AgentType.VEHICLE,
        length=length,
        width=width,
        wheelbase=wheelbase,
        past_trajectory=past_traj,
        past_velocities=past_vel,
        acceleration=accel,
        steering_angle=steering,
        yaw_rate=yaw_rate,
        future_trajectory=future,
        goal_pose=goal,
        route_lanes=route_lanes,
        route_speed_limit=route_sl,
        route_has_speed_limit=route_hsl,
        turn_indicators=turn_ind,
    )


def _extract_neighbors(data: dict[str, np.ndarray]) -> list[Agent]:
    """Build neighbor Agent list from NPZ arrays."""
    nb_past = data.get("neighbor_agents_past")
    if nb_past is None:
        return []

    # (N_nb, T, 11) [x, y, cos_h, sin_h, vx, vy, width, length, is_veh, is_ped, is_bike]
    N_nb, T, _ = nb_past.shape

    nb_future = data.get("neighbor_agents_future")

    agents: list[Agent] = []
    for i in range(N_nb):
        traj_i = nb_past[i]  # (T, 11)

        # Skip invalid neighbors (all zeros in positional data)
        if np.sum(np.abs(traj_i[:, :6])) == 0:
            continue

        # Convert cos/sin heading to radians
        headings = _cos_sin_to_heading(traj_i[:, 2], traj_i[:, 3])
        past_traj = np.stack([traj_i[:, 0], traj_i[:, 1], headings], axis=-1).astype(np.float32)
        past_vel = traj_i[:, 4:6].astype(np.float32).copy()

        width = float(np.max(np.abs(traj_i[:, 6])))
        length = float(np.max(np.abs(traj_i[:, 7])))
        if width == 0.0:
            width = _DEFAULT_WIDTH
        if length == 0.0:
            length = _DEFAULT_LENGTH
        wheelbase = length * _WHEELBASE_LENGTH_RATIO

        # Agent type from last valid timestep
        last_valid = -1
        for t in range(T - 1, -1, -1):
            if np.any(traj_i[t, :2] != 0):
                last_valid = t
                break
        if last_valid < 0:
            last_valid = T - 1
        atype = _agent_type_from_onehot(
            traj_i[last_valid, 8], traj_i[last_valid, 9], traj_i[last_valid, 10]
        )

        # Current dynamics from last timestep
        accel = np.zeros(2, dtype=np.float32)
        if T >= 2:
            dv = past_vel[-1] - past_vel[-2]
            accel = (dv / 0.1).astype(np.float32)

        # Ground truth future (if available)
        future = None
        if nb_future is not None:
            fut_i = nb_future[i]  # (T_future, 3) [x, y, heading_rad]
            if np.any(fut_i != 0):
                future = fut_i.astype(np.float32)
                if future.shape[-1] == 4:
                    future = np.stack([
                        future[:, 0], future[:, 1],
                        _cos_sin_to_heading(future[:, 2], future[:, 3]),
                    ], axis=-1).astype(np.float32)

        _correct_heading_flip(past_traj, past_vel, future)

        agents.append(Agent(
            id=f"neighbor_{i}",
            agent_type=atype,
            length=length,
            width=width,
            wheelbase=wheelbase,
            past_trajectory=past_traj,
            past_velocities=past_vel,
            acceleration=accel,
            future_trajectory=future,
        ))

    return agents


def _extract_map_data(data: dict[str, np.ndarray]) -> MapData:
    """Build MapData from NPZ arrays."""
    def _get(key: str, default_shape: tuple[int, ...]) -> np.ndarray:
        arr = data.get(key)
        if arr is None:
            return np.zeros(default_shape, dtype=np.float32)
        return arr.astype(np.float32)

    def _get_bool(key: str, default_shape: tuple[int, ...]) -> np.ndarray:
        arr = data.get(key)
        if arr is None:
            return np.zeros(default_shape, dtype=bool)
        return arr.astype(bool)

    return MapData(
        lanes=_get("lanes", (140, 20, 33)),
        lanes_speed_limit=_get("lanes_speed_limit", (140, 1)),
        lanes_has_speed_limit=_get_bool("lanes_has_speed_limit", (140, 1)),
        polygons=_get("polygons", (10, 40, 3)),
        line_strings=_get("line_strings", (60, 20, 4)),
        static_objects=_get("static_objects", (5, 10)),
    )


def from_npz(path: str | Path) -> SceneContext:
    """Load an NPZ file into a SceneContext.

    The resulting scene frame is the original ego-centric frame from the NPZ
    (ego at origin, heading=0 at the current timestep).

    Args:
        path: Path to the .npz file.

    Returns:
        SceneContext with ego + neighbors + map data.
    """
    with np.load(str(path), allow_pickle=True) as loaded:
        data = {k: v for k, v in loaded.items() if k not in {"map_name", "token"}}

    # Ego shape: [wheelbase, length, width]
    ego_shape = data.get("ego_shape")
    if ego_shape is not None:
        ego_shape = ego_shape.flatten()
        wheelbase = float(ego_shape[0])
        length = float(ego_shape[1])
        width = float(ego_shape[2])
    else:
        wheelbase = _DEFAULT_WHEELBASE
        length = _DEFAULT_LENGTH
        width = _DEFAULT_WIDTH

    ego = _extract_ego_agent(data, wheelbase, length, width)
    neighbors = _extract_neighbors(data)
    map_data = _extract_map_data(data)

    return SceneContext(
        agents=[ego] + neighbors,
        map_data=map_data,
        ego_agent_id="ego",
    )
