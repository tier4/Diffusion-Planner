"""The versioned native-H5/ONNX contract used by the current new DP."""

from __future__ import annotations

MODEL_INPUT_NAMES = (
    "ego_agent_past",
    "neighbor_agents_past",
    "agent_shape",
    "agent_label",
    "lanes",
    "lane_types",
    "lanes_speed_limit",
    "lane_traffic_light_past",
    "lane_traffic_light_future",
    "route_lanes",
    "route_lane_types",
    "route_lanes_speed_limit",
    "route_traffic_light_past",
    "route_traffic_light_future",
    "intersection_area",
    "stop_lines",
    "road_borders",
    "goal_pose",
    "ego_shape",
    "turn_indicators",
)

H5_FORMAT = "diffusion_planner_frame_dataset"
H5_FORMAT_VERSION = 4

NUM_AGENTS = 321
FUTURE_STEPS = 80
TRAJECTORY_DIM = 4
TURN_LOGIT_DIM = 3
INITIAL_NOISE_SHAPE = (NUM_AGENTS, FUTURE_STEPS, TRAJECTORY_DIM)
