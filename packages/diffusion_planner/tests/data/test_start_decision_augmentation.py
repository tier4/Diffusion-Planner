"""Tests for ego start-decision augmentation."""

from __future__ import annotations

import unittest

import numpy as np

from diffusion_planner.data.transforms import PlannerStartDecisionAugmentation


def _states(speeds: list[float]) -> np.ndarray:
    values = np.zeros((len(speeds), 6), dtype=np.float32)
    values[:, 0] = np.arange(len(speeds), dtype=np.float32)
    values[:, 2] = 1.0
    values[:, 4] = speeds
    return values


class PlannerStartDecisionAugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.augmentation = PlannerStartDecisionAugmentation(
            probability=1.0,
            max_shift_steps=4,
        )

    def test_moves_recent_start_from_past_into_future(self) -> None:
        past = _states([0.0, 0.0, 0.0, 0.2])
        future = _states([0.3, 1.0, 3.0, 4.0])

        result = self.augmentation({"ego_agent_past": past, "ego_agent_future": future})

        np.testing.assert_array_equal(result["ego_agent_past"][:, 4], 0.0)
        np.testing.assert_allclose(
            result["ego_agent_future"][:, 4], [0.2, 0.3, 1.0, 3.0]
        )
        np.testing.assert_array_equal(
            result["ego_agent_past"][-1, :4], [0.0, 0.0, 1.0, 0.0]
        )
        self.assertEqual(result["ego_agent_future"][0, 0], 1.0)

    def test_one_stopped_point_is_sufficient(self) -> None:
        past = _states([2.0, 0.5, 0.1, 0.2])
        future = _states([0.3, 1.0, 3.0, 4.0])

        result = self.augmentation({"ego_agent_past": past, "ego_agent_future": future})

        self.assertAlmostEqual(float(result["ego_agent_past"][-1, 4]), 0.1)
        np.testing.assert_allclose(
            result["ego_agent_future"][:, 4], [0.2, 0.3, 1.0, 3.0]
        )

    def test_shifts_only_ego_in_time_and_recenters_spatial_tensors(self) -> None:
        past = _states([0.0, 0.0, 0.0, 0.2, 0.3])
        future = _states([1.0, 3.0, 4.0])
        neighbors = np.asarray(
            [[10.0, 0.0, 1.0, 0.0], [11.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        frame = {
            "ego_agent_past": past,
            "ego_agent_future": future,
            "neighbor_agents_future": neighbors,
        }

        result = self.augmentation(frame)

        np.testing.assert_array_equal(
            result["ego_agent_past"][-1, :4], [0.0, 0.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            result["neighbor_agents_future"][:, 0], [8.0, 9.0]
        )
        np.testing.assert_array_equal(frame["ego_agent_past"], past)
        np.testing.assert_array_equal(frame["ego_agent_future"], future)
        np.testing.assert_array_equal(frame["neighbor_agents_future"], neighbors)

    def test_shifts_without_confirming_later_acceleration(self) -> None:
        past = _states([0.0, 0.0, 0.0, 0.2])
        future = _states([0.0, 0.0, 0.0])

        result = self.augmentation({"ego_agent_past": past, "ego_agent_future": future})

        self.assertEqual(result["ego_agent_past"][-1, 4], 0.0)
        np.testing.assert_allclose(result["ego_agent_future"][:, 4], [0.2, 0.0, 0.0])

    def test_does_not_shift_start_already_in_future(self) -> None:
        past = _states([0.0, 0.0, 0.0, 0.0])
        future = _states([0.2, 0.3, 1.0, 3.0])

        result = self.augmentation({"ego_agent_past": past, "ego_agent_future": future})

        self.assertIs(result["ego_agent_past"], past)
        self.assertIs(result["ego_agent_future"], future)
