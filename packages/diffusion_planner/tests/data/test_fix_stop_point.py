"""Tests for fixing an ego future after its first stop point."""

from __future__ import annotations

import unittest

import numpy as np

from diffusion_planner.data.transforms import PlannerFixStopPoint


class PlannerFixStopPointTest(unittest.TestCase):
    def test_holds_pose_and_zeros_motion_from_first_stop(self) -> None:
        future = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0, 2.0, 0.1],
                [2.0, 0.1, 0.9, 0.1, 0.1, 0.2],
                [3.0, 0.2, 0.8, 0.2, 1.0, 0.3],
            ],
            dtype=np.float32,
        )
        frame = {"ego_agent_future": future}

        result = PlannerFixStopPoint(stop_speed_threshold=0.1)(frame)

        np.testing.assert_array_equal(result["ego_agent_future"][0], future[0])
        np.testing.assert_array_equal(
            result["ego_agent_future"][1:, :4],
            np.repeat(future[1:2, :4], 2, axis=0),
        )
        np.testing.assert_array_equal(result["ego_agent_future"][1:, 4:], 0.0)
        np.testing.assert_array_equal(frame["ego_agent_future"], future)

    def test_keeps_future_when_no_stop_exists(self) -> None:
        future = np.zeros((3, 6), dtype=np.float32)
        future[:, 2] = 1.0
        future[:, 4] = 0.2

        result = PlannerFixStopPoint(stop_speed_threshold=0.1)(
            {"ego_agent_future": future}
        )

        self.assertIs(result["ego_agent_future"], future)

    def test_threshold_is_inclusive(self) -> None:
        future = np.zeros((2, 6), dtype=np.float32)
        future[:, 2] = 1.0
        future[:, 4] = (0.1, 1.0)

        result = PlannerFixStopPoint(stop_speed_threshold=0.1)(
            {"ego_agent_future": future}
        )

        np.testing.assert_array_equal(result["ego_agent_future"][:, 4:], 0.0)
