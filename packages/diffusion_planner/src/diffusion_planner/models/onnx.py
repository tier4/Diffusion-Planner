"""ONNX boundary for complete planner sampling."""

from __future__ import annotations

import torch
from torch import nn

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
    """Expose fixed 10-step Heun sampling as one ONNX graph."""

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
        return self.planner.sample(
            input_data, initial_noise, num_steps=6, time_epsilon=1e-5
        )
