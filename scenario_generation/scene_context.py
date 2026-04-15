"""Core data types for scenario generation.

All spatial data is stored in a scene-level coordinate frame (world frame).
When loaded from NPZ, this frame corresponds to the original ego vehicle's
frame at the current timestep. Heading angles are stored as radians for
human readability; cos/sin conversion happens during tensor export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class AgentType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    BICYCLE = "bicycle"


@dataclass
class Agent:
    """A dynamic agent (vehicle, pedestrian, or bicycle) in the scene.

    Attributes:
        id: Unique identifier within the scene.
        agent_type: Classification of the agent.
        length: Bounding box length in metres.
        width: Bounding box width in metres.
        wheelbase: Distance between front and rear axles in metres.
            Derived from length (~0.65 * length) when not available.
        past_trajectory: (T_past, 3) float32 array of [x, y, heading_rad]
            in scene frame. Index -1 is the current timestep.
        past_velocities: (T_past, 2) float32 array of [vx, vy] in m/s,
            scene frame. If None, derived from trajectory differences
            during tensor conversion.
        acceleration: (2,) float32 [ax, ay] m/s^2 at current timestep.
        steering_angle: Current steering angle in radians.
        yaw_rate: Current yaw rate in rad/s.
        future_trajectory: Optional (T_future, 3) float32 [x, y, heading_rad].
            Ground truth future; used for training/evaluation only.
        goal_pose: Optional (3,) float32 [x, y, heading_rad] in scene frame.
        route_lanes: Optional (N_segments, 20, 33) lane segments for this
            agent's route. Same 33-dim format as map lanes.
        route_speed_limit: Optional (N_segments, 1) float32.
        route_has_speed_limit: Optional (N_segments, 1) bool.
        turn_indicators: Optional (T_past,) int32 turn signal history.
            Values: 0=NONE, 1=DISABLE, 2=LEFT, 3=RIGHT, 4=KEEP.
    """

    id: str
    agent_type: AgentType
    length: float
    width: float
    wheelbase: float

    past_trajectory: np.ndarray
    past_velocities: np.ndarray | None = None

    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    steering_angle: float = 0.0
    yaw_rate: float = 0.0

    future_trajectory: np.ndarray | None = None
    goal_pose: np.ndarray | None = None
    route_lanes: np.ndarray | None = None
    route_speed_limit: np.ndarray | None = None
    route_has_speed_limit: np.ndarray | None = None
    turn_indicators: np.ndarray | None = None

    @property
    def current_position(self) -> np.ndarray:
        """(2,) [x, y] at current timestep."""
        return self.past_trajectory[-1, :2].copy()

    @property
    def current_heading(self) -> float:
        """Heading in radians at current timestep."""
        return float(self.past_trajectory[-1, 2])

    @property
    def current_velocity(self) -> np.ndarray:
        """(2,) [vx, vy] at current timestep."""
        if self.past_velocities is not None:
            return self.past_velocities[-1].copy()
        if self.past_trajectory.shape[0] >= 2:
            dt = 0.1
            diff = self.past_trajectory[-1, :2] - self.past_trajectory[-2, :2]
            return (diff / dt).astype(np.float32)
        return np.zeros(2, dtype=np.float32)


@dataclass
class MapData:
    """Static map elements in scene frame.

    Lane segment 33-dim breakdown per point:
        [0:2]   centerline XY (absolute position in scene frame)
        [2:4]   dX, dY (direction to next point -- rotate only)
        [4:6]   left boundary XY (offset from centerline -- rotate only)
        [6:8]   right boundary XY (offset from centerline -- rotate only)
        [8:13]  traffic light one-hot (green, yellow, red, white, none)
        [13:23] left line type one-hot (10 types)
        [23:33] right line type one-hot (10 types)

    Attributes:
        lanes: (N_lanes, 20, 33) lane segments.
        lanes_speed_limit: (N_lanes, 1) speed limits in m/s.
        lanes_has_speed_limit: (N_lanes, 1) boolean mask.
        polygons: (N_poly, 40, 2|3) intersection areas [x, y] or [x, y, type].
        line_strings: (N_ls, 20, 2|4) stop lines / boundaries [x, y] or
            [x, y, stop_flag, border_flag].
        static_objects: (N_static, 10) parked objects
            [x, y, cos_h, sin_h, width, length, type(4)].
    """

    lanes: np.ndarray
    lanes_speed_limit: np.ndarray
    lanes_has_speed_limit: np.ndarray
    polygons: np.ndarray
    line_strings: np.ndarray
    static_objects: np.ndarray


@dataclass
class SceneContext:
    """Complete scene representation in a world/scene coordinate frame.

    Designed to be the interchange format between data sources (NPZ, rosbag,
    scenario generators) and the Diffusion-Planner model.

    Attributes:
        agents: All dynamic agents. Index 0 is typically the original ego
            when loaded from NPZ, but any agent can serve as ego during
            tensor conversion.
        map_data: Static map geometry and road furniture.
        ego_agent_id: ID of the original ego agent (informational).
        dt: Seconds per trajectory timestep (default 0.1 = 10 Hz).
    """

    agents: list[Agent]
    map_data: MapData
    ego_agent_id: str | None = None
    dt: float = 0.1

    def get_agent(self, agent_id: str) -> Agent:
        """Look up an agent by ID. Raises KeyError if not found."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Agent '{agent_id}' not found. IDs: {[a.id for a in self.agents]}")

    @property
    def ego_agent(self) -> Agent | None:
        """The original ego agent, or None if not set."""
        if self.ego_agent_id is None:
            return None
        try:
            return self.get_agent(self.ego_agent_id)
        except KeyError:
            return None
