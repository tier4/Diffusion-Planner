"""Tests for planner tensor normalization."""

from __future__ import annotations

import unittest

import numpy as np

from diffusion_planner.data import PlannerDataNormalizer
from diffusion_planner.data.dimensions import AGENT_LABEL_DIM


class PlannerDataNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = PlannerDataNormalizer(
            position_scale=10.0,
            speed_scale=5.0,
            vehicle_shape_scale=2.0,
        )

    def test_trajectory_round_trip(self) -> None:
        trajectory = np.array([[[20.0, -10.0, 3.0, 4.0]]], dtype=np.float32)

        normalized = self.normalizer.normalize_trajectory(trajectory)
        restored = self.normalizer.denormalize_trajectory(normalized)

        np.testing.assert_allclose(normalized, [[[2.0, -1.0, 3.0, 4.0]]])
        np.testing.assert_allclose(restored, [[[20.0, -10.0, 0.6, 0.8]]])

    def test_input_normalization_is_non_destructive(self) -> None:
        input_data = {
            "neighbor_agents_past": np.array([[[10.0, 20.0, 1.0, 0.0]]]),
            "lanes": np.full((1, 1, 6), 10.0),
            "route_lanes": np.full((1, 1, 6), 10.0),
            "intersection_area": np.full((1, 1, 2), 10.0),
            "stop_lines": np.full((1, 1, 2), 10.0),
            "road_borders": np.full((1, 1, 2), 10.0),
            "goal_pose": np.array([20.0, -10.0, 1.0, 0.0]),
            "lanes_speed_limit": np.array([[10.0]]),
            "route_lanes_speed_limit": np.array([[10.0]]),
            "agent_shape": np.array([[2.0, 4.0]]),
            "ego_shape": np.array([4.0, 6.0, 2.0]),
            "agent_label": np.ones((1, AGENT_LABEL_DIM)),
        }

        normalized = self.normalizer(input_data)

        self.assertEqual(input_data["goal_pose"][0], 20.0)
        np.testing.assert_allclose(normalized["goal_pose"], [2.0, -1.0, 1.0, 0.0])
        np.testing.assert_allclose(normalized["lanes_speed_limit"], [[2.0]])
        np.testing.assert_allclose(normalized["ego_shape"], [2.0, 3.0, 1.0])
        self.assertIs(normalized["agent_label"], input_data["agent_label"])

    def test_normalizes_yaw_vectors(self) -> None:
        trajectory = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)

        normalized = self.normalizer.normalize_yaw_vector(trajectory)

        np.testing.assert_allclose(normalized, [[[1.0, 2.0, 0.6, 0.8]]])


if __name__ == "__main__":
    unittest.main()
