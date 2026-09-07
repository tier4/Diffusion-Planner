"""Tests for the planner ONNX module boundaries."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.onnx import (
    PLANNER_INPUT_NAMES,
    DiffusionPlannerOnnxWrapper,
)

from .test_diffusion_planner import make_input_data


class OnnxWrapperTest(unittest.TestCase):
    def test_wrapper_keeps_the_agent_axis_the_node_expects(self) -> None:
        """The graph's tensor shapes are a contract with the deployed ROS node.

        The node builds a fixed TensorRT profile from them, so the agent axis has
        to survive on both the noise input and the trajectory output even though
        the planner predicts the ego alone.
        """
        model = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            element_mixer_hidden_dim=8,
        ).eval()
        input_data = make_input_data()
        wrapper = DiffusionPlannerOnnxWrapper(model).eval()
        agents = 5
        initial_noise = torch.randn(
            1, agents, 80, 4, generator=torch.Generator().manual_seed(42)
        )

        with torch.inference_mode():
            trajectory, turn_indicator = wrapper(
                initial_noise,
                *(input_data[name] for name in PLANNER_INPUT_NAMES),
            )
            expected_ego, expected_turn_indicator = model.sample(
                input_data, initial_noise[:, 0], num_steps=6
            )

        self.assertEqual(trajectory.shape, (1, agents, 80, 4))
        torch.testing.assert_close(trajectory[:, 0], expected_ego)
        torch.testing.assert_close(
            trajectory[:, 1:], torch.zeros_like(trajectory[:, 1:])
        )
        torch.testing.assert_close(turn_indicator, expected_turn_indicator)

    def test_sampler_wrapper_matches_planner_sample(self) -> None:
        model = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            element_mixer_hidden_dim=8,
        ).eval()
        input_data = make_input_data()
        wrapper = DiffusionPlannerOnnxWrapper(model).eval()
        initial_noise = torch.randn(
            1, 3, 80, 4, generator=torch.Generator().manual_seed(42)
        )

        with torch.inference_mode():
            actual_trajectory, actual_turn_indicator = wrapper(
                initial_noise,
                *(input_data[name] for name in PLANNER_INPUT_NAMES),
            )
            expected_trajectory, expected_turn_indicator = model.sample(
                input_data, initial_noise[:, 0], num_steps=6
            )

        torch.testing.assert_close(actual_trajectory[:, 0], expected_trajectory)
        torch.testing.assert_close(actual_turn_indicator, expected_turn_indicator)


if __name__ == "__main__":
    unittest.main()
