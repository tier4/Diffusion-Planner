"""Tests for the conditional flow-matching planner."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import (
    CONTROL_DIM,
    EGO_HISTORY_LENGTH,
    INTERSECTION_AREA_LENGTH,
    LANE_LENGTH,
    ROAD_BORDER_LENGTH,
    STOP_LINE_LENGTH,
    TRAFFIC_LIGHT_FUTURE_LENGTH,
    TRAFFIC_LIGHT_PAST_LENGTH,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
    TURN_INDICATOR_HISTORY_LENGTH,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.flow_matching import sample_time
from diffusion_planner.models.loss import (
    compute_diffusion_planner_loss,
    control_huber_loss,
    create_ego_padding_mask,
)


def make_input_data() -> dict[str, torch.Tensor]:
    batch = 1
    neighbors = 2
    data = {
        "ego_agent_past": torch.zeros(batch, EGO_HISTORY_LENGTH, 6),
        "neighbor_agents_past": torch.zeros(batch, neighbors, EGO_HISTORY_LENGTH, 4),
        "agent_shape": torch.zeros(batch, neighbors, 2),
        "agent_label": torch.zeros(batch, neighbors, 3),
        "lanes": torch.zeros(batch, 2, LANE_LENGTH, 6),
        "lane_types": torch.zeros(batch, 2, 20),
        "lanes_speed_limit": torch.zeros(batch, 2, 1),
        "lane_traffic_light_past": torch.zeros(batch, 2, TRAFFIC_LIGHT_PAST_LENGTH, 6),
        "lane_traffic_light_future": torch.zeros(
            batch, 2, TRAFFIC_LIGHT_FUTURE_LENGTH, 6
        ),
        "route_lanes": torch.zeros(batch, 1, LANE_LENGTH, 6),
        "route_lane_types": torch.zeros(batch, 1, 20),
        "route_lanes_speed_limit": torch.zeros(batch, 1, 1),
        "route_traffic_light_past": torch.zeros(batch, 1, TRAFFIC_LIGHT_PAST_LENGTH, 6),
        "route_traffic_light_future": torch.zeros(
            batch, 1, TRAFFIC_LIGHT_FUTURE_LENGTH, 6
        ),
        "intersection_area": torch.zeros(batch, 1, INTERSECTION_AREA_LENGTH, 2),
        "stop_lines": torch.zeros(batch, 1, STOP_LINE_LENGTH, 2),
        "road_borders": torch.zeros(batch, 1, ROAD_BORDER_LENGTH, 2),
        "goal_pose": torch.tensor([[10.0, 0.0, 1.0, 0.0]]),
        "ego_shape": torch.tensor([[3.8, 4.9, 1.9]]),
        "ego_agent_future": torch.zeros(batch, TRAJECTORY_LENGTH, 6),
        "neighbor_agents_future": torch.zeros(batch, neighbors, TRAJECTORY_LENGTH, 4),
        "turn_indicators": torch.ones(batch, TURN_INDICATOR_HISTORY_LENGTH),
        "turn_indicators_future": torch.ones(batch, TRAJECTORY_LENGTH),
    }
    data["ego_agent_past"][..., 2] = 1.0
    data["neighbor_agents_past"][:, 0, :, 2] = 1.0
    data["agent_shape"][:, 0] = torch.tensor([2.0, 4.5])
    data["agent_label"][:, 0, 0] = 1.0
    data["ego_agent_future"][..., 2] = 1.0
    data["neighbor_agents_future"][:, 0, :, 2] = 1.0
    return data


class DiffusionPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            element_mixer_hidden_dim=8,
        )
        self.input_data = make_input_data()

    def test_compute_loss(self) -> None:
        self.input_data["ego_agent_future"][:, :, 0] = torch.arange(TRAJECTORY_LENGTH)
        turn_indicator_trajectories: list[torch.Tensor] = []

        def capture_turn_indicator_trajectory(
            _module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            turn_indicator_trajectories.append(args[3].detach().clone())

        handle = self.model.turn_indicator_decoder.register_forward_hook(
            capture_turn_indicator_trajectory
        )
        losses = compute_diffusion_planner_loss(
            self.model,
            self.input_data,
            time_mean=-0.4,
            time_std=1.0,
            time_epsilon=1e-5,
            noise_scale=1.0,
        )
        handle.remove()

        self.assertEqual(losses["total"].ndim, 0)
        self.assertTrue(torch.isfinite(losses["total"]))
        torch.testing.assert_close(
            turn_indicator_trajectories[0],
            self.input_data["ego_agent_future"][..., :TRAJECTORY_DIM],
        )
        losses["total"].backward()

    def test_partial_future_padding_masks_the_sample(self) -> None:
        self.assertFalse(create_ego_padding_mask(self.input_data).any())

        self.input_data["ego_agent_future"][:, TRAJECTORY_LENGTH // 2 :] = 0.0

        self.assertTrue(create_ego_padding_mask(self.input_data).all())

    def test_stationary_ego_is_not_masked(self) -> None:
        """A stationary ego has all-zero control; the mask must read poses instead."""
        self.input_data["ego_agent_past"][..., 4] = 0.0
        self.input_data["ego_agent_future"][..., 4] = 0.0

        self.assertFalse(create_ego_padding_mask(self.input_data).any())

    def test_turn_indicator_loss_backpropagates_into_scene_encoder(self) -> None:
        control, logits = self.model(
            torch.randn(1, TRAJECTORY_LENGTH, CONTROL_DIM),
            self.input_data,
            torch.full((1,), 0.5),
        )
        del control

        logits.sum().backward()

        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in self.model.scene_encoder.parameters()
            )
        )
        self.assertIsNotNone(
            self.model.turn_indicator_decoder.trajectory_scene_attention.in_proj_weight.grad
        )

    def test_turn_indicator_loss_weight_controls_total_loss(self) -> None:
        losses = compute_diffusion_planner_loss(
            self.model,
            self.input_data,
            time_mean=-0.4,
            time_std=1.0,
            time_epsilon=1e-5,
            noise_scale=1.0,
            turn_indicator_loss_weight=0.0,
            control_trajectory_loss_weight=0.4,
        )

        torch.testing.assert_close(
            losses["total"],
            losses["control"] + 0.4 * losses["control_trajectory"],
        )

    def test_control_loss_uses_elementwise_huber(self) -> None:
        target = torch.tensor([[[0.0, 0.0]]])
        prediction = torch.tensor([[[-2.0, 0.5]]])

        loss = control_huber_loss(prediction, target, torch.zeros(1), time_epsilon=1e-5)

        torch.testing.assert_close(loss, torch.tensor([[[1.5, 0.125]]]))

    def test_logistic_normal_time_is_inside_unit_interval(self) -> None:
        time = sample_time(
            128,
            torch.device("cpu"),
            torch.float32,
            -0.4,
            1.0,
        )

        self.assertTrue(torch.all(time > 0))
        self.assertTrue(torch.all(time < 1))

    def test_sample_encodes_scene_once_and_returns_ego_poses(self) -> None:
        call_count = 0
        decoder_call_count = 0
        turn_indicator_trajectories: list[torch.Tensor] = []

        def count_scene_calls(
            _module: torch.nn.Module,
            _args: tuple[dict[str, torch.Tensor]],
            _output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            nonlocal call_count
            call_count += 1

        def count_decoder_calls(
            _module: torch.nn.Module,
            _args: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            nonlocal decoder_call_count
            decoder_call_count += 1

        def capture_turn_indicator_trajectory(
            _module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            turn_indicator_trajectories.append(args[3].detach().clone())

        handle = self.model.scene_encoder.register_forward_hook(count_scene_calls)
        decoder_handle = self.model.trajectory_decoder.register_forward_hook(
            count_decoder_calls
        )
        turn_indicator_handle = self.model.turn_indicator_decoder.register_forward_hook(
            capture_turn_indicator_trajectory
        )
        trajectories, turn_indicator_logits = self.model.sample(
            self.input_data,
            torch.randn(1, TRAJECTORY_LENGTH, CONTROL_DIM),
            num_steps=2,
        )
        handle.remove()
        decoder_handle.remove()
        turn_indicator_handle.remove()

        self.assertEqual(trajectories.shape, (1, TRAJECTORY_LENGTH, 4))
        self.assertEqual(turn_indicator_logits.shape, (1, 3))
        yaw_norm = torch.linalg.vector_norm(trajectories[..., 2:4], dim=-1)
        torch.testing.assert_close(yaw_norm, torch.ones_like(yaw_norm))
        self.assertEqual(call_count, 1)
        self.assertEqual(decoder_call_count, 3)
        torch.testing.assert_close(turn_indicator_trajectories[0], trajectories)


if __name__ == "__main__":
    unittest.main()
