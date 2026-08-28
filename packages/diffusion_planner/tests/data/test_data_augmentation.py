"""Tests for ego-pose planner data augmentation."""

from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.typing import NDArray

from diffusion_planner.data.transforms import (
    PlannerQuinticHermiteAugmentation,
    PlannerSpeedAugmentation,
)


def _pose(
    x: float, y: float, cosine: float = 1.0, sine: float = 0.0
) -> NDArray[np.float32]:
    return np.asarray([x, y, cosine, sine], dtype=np.float32)


def _frame() -> dict[str, NDArray[np.float32]]:
    ego_state = np.concatenate((_pose(0.0, 0.0), np.asarray([1.0, 0.0])))
    frame = {
        "ego_agent_past": np.tile(ego_state, (2, 1)).astype(np.float32),
        "ego_agent_future": np.tile(ego_state, (2, 1)).astype(np.float32),
        "neighbor_agents_past": np.zeros((2, 2, 4), dtype=np.float32),
        "neighbor_agents_future": np.zeros((2, 2, 4), dtype=np.float32),
        "goal_pose": _pose(4.0, 2.0),
        "lanes": np.zeros((2, 3, 6), dtype=np.float32),
        "route_lanes": np.zeros((2, 3, 6), dtype=np.float32),
        "intersection_area": np.zeros((2, 3, 2), dtype=np.float32),
        "stop_lines": np.zeros((2, 2, 2), dtype=np.float32),
        "road_borders": np.zeros((2, 3, 2), dtype=np.float32),
        "agent_shape": np.ones((2, 2), dtype=np.float32),
    }
    frame["neighbor_agents_past"][0] = np.tile(_pose(3.0, 2.0), (2, 1))
    frame["neighbor_agents_future"][0] = np.tile(_pose(3.0, 2.0), (2, 1))
    frame["route_lanes"][0, :, 0] = np.asarray([-1.0, 0.0, 1.0])
    frame["route_lanes"][0, :, 2] = 1.0
    frame["lanes"][0] = frame["route_lanes"][0]
    frame["intersection_area"][0, :, 0] = 1.0
    frame["stop_lines"][0, :, 0] = 1.0
    frame["road_borders"][0, :, 0] = 1.0
    return frame


class PlannerQuinticHermiteAugmentationTest(unittest.TestCase):
    def test_transforms_scene_into_augmented_ego_frame(self) -> None:
        frame = _frame()
        augmentation = PlannerQuinticHermiteAugmentation(
            lateral_offset_range=(2.0, 2.0),
            yaw_offset_range=(math.pi / 2, math.pi / 2),
            pose_probability=1.0,
        )

        result = augmentation(frame)

        np.testing.assert_allclose(
            result["ego_agent_past"][-1, :4], _pose(0.0, 0.0), atol=1e-6
        )
        np.testing.assert_allclose(
            result["neighbor_agents_past"][0, -1],
            _pose(0.0, -3.0, 0.0, -1.0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["ego_agent_future"][0, :4],
            _pose(-2.0, 0.0, 0.0, -1.0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["goal_pose"],
            _pose(0.0, -4.0, 0.0, -1.0),
            atol=1e-6,
        )

    def test_preserves_padding_and_non_coordinate_tensors(self) -> None:
        frame = _frame()
        augmentation = PlannerQuinticHermiteAugmentation(
            (1.0, 1.0), (0.1, 0.1), pose_probability=1.0
        )

        result = augmentation(frame)

        self.assertEqual(np.count_nonzero(result["neighbor_agents_past"][1]), 0)
        self.assertEqual(np.count_nonzero(result["lanes"][1]), 0)
        self.assertIs(result["agent_shape"], frame["agent_shape"])

    def test_scales_only_ego_speed_history(self) -> None:
        frame = _frame()
        augmentation = PlannerSpeedAugmentation(
            speed_scale_range=(1.2, 1.2),
            probability=1.0,
        )

        result = augmentation(frame)

        np.testing.assert_allclose(result["ego_agent_past"][:, 4], 1.2)
        np.testing.assert_allclose(result["ego_agent_future"][:, 4], 1.0)

    def test_speed_augmentation_does_not_transform_scene(self) -> None:
        frame = _frame()
        augmentation = PlannerSpeedAugmentation(
            speed_scale_range=(1.2, 1.2),
            probability=1.0,
        )

        result = augmentation(frame)

        np.testing.assert_array_equal(
            result["neighbor_agents_past"], frame["neighbor_agents_past"]
        )
        np.testing.assert_allclose(result["ego_agent_past"][:, 4], 1.2)


if __name__ == "__main__":
    unittest.main()
