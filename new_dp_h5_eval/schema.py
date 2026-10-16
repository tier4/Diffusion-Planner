"""The versioned native-H5/ONNX contract used by the current new DP."""

from __future__ import annotations

from dataclasses import dataclass

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
# Later versions are read too: the evaluator interprets tensors by their width and reads
# version 5 metadata, so a layout it does not know fails on the width or a missing key.
H5_MIN_FORMAT_VERSION = 4
H5_PACKED_DATASET = "frames_packed"
H5_PACKED_LAYOUT_ATTR = "packed_frame_layout"

NUM_AGENTS = 321
FUTURE_STEPS = 80
TRAJECTORY_DIM = 4
TURN_LOGIT_DIM = 3
INITIAL_NOISE_SHAPE = (NUM_AGENTS, FUTURE_STEPS, TRAJECTORY_DIM)


@dataclass(frozen=True)
class EgoPastLayout:
    """Column meaning of native ``ego_agent_past``, which depends on its width.

    A history with steering inserts ``steering_tire_angle`` before ``yaw_rate``, so the yaw-rate
    column moves.
    """

    width: int
    yaw_rate: int
    steering: int | None

    @classmethod
    def of(cls, width: int) -> "EgoPastLayout":
        if width == 6:
            return cls(width=6, yaw_rate=5, steering=None)
        if width == 7:
            return cls(width=7, yaw_rate=6, steering=5)
        raise ValueError(f"unexpected native ego_agent_past width {width}; expected 6 or 7")


EGO_SPEED_COLUMN = 4
