"""Tensor layout constants for diffusion planner frame data."""

from __future__ import annotations

from enum import IntEnum

PAST_TIME_STEP_S = 0.1
FUTURE_TIME_STEP_S = 0.1


class PoseIndex(IntEnum):
    """Indices shared by pose-like tensors."""

    X = 0
    Y = 1
    COS_YAW = 2
    SIN_YAW = 3


class EgoIndex(IntEnum):
    """Indices in ego past and future tensors."""

    X = 0
    Y = 1
    COS_YAW = 2
    SIN_YAW = 3
    VELOCITY = 4
    YAW_RATE = 5


class NeighborIndex(IntEnum):
    """Indices in ``neighbor_agents_past``."""

    X = 0
    Y = 1
    COS_YAW = 2
    SIN_YAW = 3


class AgentShapeIndex(IntEnum):
    """Indices in ``agent_shape``."""

    WIDTH = 0
    LENGTH = 1


class AgentLabelIndex(IntEnum):
    """Indices in ``agent_label``."""

    IS_VEHICLE = 0
    IS_PEDESTRIAN = 1
    IS_BICYCLE = 2
    IS_UNKNOWN = 3


class LaneIndex(IntEnum):
    """Indices in lane point tensors."""

    X = 0
    Y = 1
    LEFT_OFFSET_X = 2
    LEFT_OFFSET_Y = 3
    RIGHT_OFFSET_X = 4
    RIGHT_OFFSET_Y = 5


class TrafficLightIndex(IntEnum):
    """Indices in traffic-light one-hot tensors."""

    GREEN = 0
    AMBER = 1
    RED = 2
    UNKNOWN = 3
    WHITE_OR_NONE = 4
    IS_ARROW = 5


# Expected rank after removing an optional leading batch dimension.
FRAME_KEY_NDIMS: dict[str, int] = {
    "ego_agent_past": 2,
    "neighbor_agents_past": 3,
    "agent_shape": 2,
    "agent_label": 2,
    "lanes": 3,
    "lane_types": 2,
    "lanes_speed_limit": 2,
    "route_lanes": 3,
    "route_lane_types": 2,
    "route_lanes_speed_limit": 2,
    "lane_traffic_light_past": 3,
    "route_traffic_light_past": 3,
    "intersection_area": 3,
    "stop_lines": 3,
    "road_borders": 3,
    "goal_pose": 1,
    "ego_shape": 1,
    "turn_indicators": 1,
    "ego_agent_future": 2,
    "neighbor_agents_future": 3,
    "turn_indicators_future": 1,
    "lane_traffic_light_future": 3,
    "route_traffic_light_future": 3,
}

REQUIRED_FRAME_KEYS = frozenset(
    {
        "ego_agent_past",
        "neighbor_agents_past",
        "agent_shape",
        "agent_label",
        "lanes",
        "lane_types",
        "route_lanes",
        "route_lane_types",
        "intersection_area",
        "stop_lines",
        "road_borders",
        "goal_pose",
        "ego_shape",
    }
)
