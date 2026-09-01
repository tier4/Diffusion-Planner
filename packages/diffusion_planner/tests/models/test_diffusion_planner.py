"""Tests for the conditional flow-matching planner."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import (
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
    create_target_trajectory,
    trajectory_error_in_target_frame,
    trajectory_huber_loss,
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

    def test_partial_future_padding_masks_complete_agent(self) -> None:
        target = create_target_trajectory(self.input_data)
        target[:, 0, TRAJECTORY_LENGTH // 2 :] = 0.0

        missing_future = (torch.count_nonzero(target, dim=-1) == 0).any(dim=-1)

        self.assertTrue(missing_future[:, 0].all())
        self.assertFalse(missing_future[:, 1].any())
        self.assertTrue(missing_future[:, 2].all())

    def test_turn_indicator_loss_backpropagates_into_scene_encoder(self) -> None:
        agent_count = self.input_data["neighbor_agents_past"].shape[1] + 1
        trajectory, logits = self.model(
            torch.randn(1, agent_count, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
            torch.zeros(1, agent_count, dtype=torch.bool),
            self.input_data,
            torch.full((1,), 0.5),
        )
        del trajectory

        logits.sum().backward()

        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in self.model.scene_encoder.parameters()
            )
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
        )

        torch.testing.assert_close(losses["total"], losses["trajectory"])

    def test_position_error_uses_target_longitudinal_lateral_frame(self) -> None:
        error = torch.tensor([[[[1.0, 0.0, 0.25, -0.5]]]])
        target = torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]])

        transformed = trajectory_error_in_target_frame(error, target)

        torch.testing.assert_close(
            transformed, torch.tensor([[[[0.0, -1.0, 0.25, -0.5]]]])
        )

    def test_trajectory_loss_uses_elementwise_huber(self) -> None:
        error = torch.tensor([[[[0.0, -2.0, 0.5, -0.5]]]])
        target = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
        prediction = target + error

        loss = trajectory_huber_loss(
            prediction, target, torch.zeros(1), time_epsilon=1e-5
        )

        torch.testing.assert_close(loss, torch.tensor([[[[0.0, 1.5, 0.125, 0.125]]]]))

    def test_trajectory_loss_applies_ego_and_neighbor_weights(self) -> None:
        target = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]])
        prediction = target.clone()
        prediction[..., 0] += 0.5

        loss = trajectory_huber_loss(
            prediction,
            target,
            torch.zeros(1),
            time_epsilon=1e-5,
            ego_loss_weight=2.0,
            neighbor_loss_weight=0.5,
        )

        torch.testing.assert_close(loss[0, 0, 0, 0], torch.tensor(0.25))
        torch.testing.assert_close(loss[0, 1, 0, 0], torch.tensor(0.0625))

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

    def test_sample_encodes_scene_once_and_masks_missing_agents(self) -> None:
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
            torch.randn(1, 3, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
            num_steps=2,
        )
        handle.remove()
        decoder_handle.remove()
        turn_indicator_handle.remove()

        self.assertEqual(trajectories.shape, (1, 3, TRAJECTORY_LENGTH, 4))
        self.assertEqual(turn_indicator_logits.shape, (1, 3))
        torch.testing.assert_close(
            trajectories[:, 2], torch.zeros_like(trajectories[:, 2])
        )
        self.assertEqual(call_count, 1)
        self.assertEqual(decoder_call_count, 3)
        torch.testing.assert_close(turn_indicator_trajectories[0], trajectories[:, 0])


if __name__ == "__main__":
    unittest.main()
