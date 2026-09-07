"""ONNX boundary for complete planner sampling."""

from __future__ import annotations

import torch
from torch import nn

from ..data.dimensions import CONTROL_DIM, TRAJECTORY_DIM
from .diffusion_planner import DiffusionPlanner

PLANNER_INPUT_NAMES = (
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


class DiffusionPlannerOnnxWrapper(nn.Module):
    """Expose fixed 10-step Heun sampling as one ONNX graph.

    Both tensors the deployed node builds its TensorRT profile around keep their
    shape, `(B, 1 + MAX_NUM_NEIGHBORS, T, TRAJECTORY_DIM)`, even though the
    planner now predicts the ego alone and denoises control rather than poses.
    Changing either would force a matching change in the node.

    The output needs only the agent axis put back: `sample` already returns poses,
    so the ego trajectory goes to index 0 and every neighbor slot stays zero --
    the same encoding the node already receives for an agent with no prediction.

    The input needs one adaptation beyond that axis. What gets denoised here is
    control, so `sample` wants `(B, T, CONTROL_DIM)`; the node supplies a wider
    per-agent tensor, and agent 0's first `CONTROL_DIM` channels are taken from
    it. That is a valid draw because every element of the supplied noise is
    i.i.d. standard normal.
    """

    def __init__(self, planner: DiffusionPlanner) -> None:
        super().__init__()
        self.planner = planner

    def forward(
        self,
        initial_noise: torch.Tensor,
        ego_agent_past: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        agent_shape: torch.Tensor,
        agent_label: torch.Tensor,
        lanes: torch.Tensor,
        lane_types: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lane_traffic_light_past: torch.Tensor,
        lane_traffic_light_future: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lane_types: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_traffic_light_past: torch.Tensor,
        route_traffic_light_future: torch.Tensor,
        intersection_area: torch.Tensor,
        stop_lines: torch.Tensor,
        road_borders: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
        turn_indicators: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectories and turn-indicator logits."""
        input_data: dict[str, torch.Tensor] = dict(
            zip(
                PLANNER_INPUT_NAMES,
                (
                    ego_agent_past,
                    neighbor_agents_past,
                    agent_shape,
                    agent_label,
                    lanes,
                    lane_types,
                    lanes_speed_limit,
                    lane_traffic_light_past,
                    lane_traffic_light_future,
                    route_lanes,
                    route_lane_types,
                    route_lanes_speed_limit,
                    route_traffic_light_past,
                    route_traffic_light_future,
                    intersection_area,
                    stop_lines,
                    road_borders,
                    goal_pose,
                    ego_shape,
                    turn_indicators,
                ),
                strict=True,
            )
        )
        trajectory, turn_indicator_logits = self.planner.sample(
            input_data,
            initial_noise[:, 0, :, :CONTROL_DIM],
            num_steps=6,
            time_epsilon=1e-5,
        )
        # The agent count comes from the noise the caller supplied, so the output
        # axis always matches the input axis the caller built its profile around.
        neighbor_trajectory = trajectory.new_zeros(
            (
                trajectory.shape[0],
                initial_noise.shape[1] - 1,
                trajectory.shape[1],
                TRAJECTORY_DIM,
            )
        )
        trajectory = torch.cat((trajectory.unsqueeze(1), neighbor_trajectory), dim=1)
        return trajectory, turn_indicator_logits
