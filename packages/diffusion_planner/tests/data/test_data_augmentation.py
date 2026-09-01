"""Tests for ego-pose planner data augmentation."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from diffusion_planner.data.transforms import (
    PlannerEgoShapeAugmentation,
    PlannerGoalTransform,
    PlannerQuinticHermiteAugmentation,
    PlannerRigidDataAugmentation,
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
        "ego_shape": np.asarray([3.5, 4.8, 1.8], dtype=np.float32),
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

    def test_perturbs_only_ego_speed_history(self) -> None:
        frame = _frame()
        augmentation = PlannerSpeedAugmentation(
            speed_scale_range=(1.2, 1.2),
            speed_noise_range=(0.3, 0.3),
            probability=1.0,
        )

        result = augmentation(frame)

        np.testing.assert_allclose(result["ego_agent_past"][:, 4], 1.5)
        np.testing.assert_allclose(result["ego_agent_future"][:, 4], 1.0)

    def test_clips_ego_speed_at_zero(self) -> None:
        frame = _frame()
        augmentation = PlannerSpeedAugmentation(
            speed_scale_range=(0.5, 0.5),
            speed_noise_range=(-1.0, -1.0),
            probability=1.0,
        )

        result = augmentation(frame)

        np.testing.assert_array_equal(result["ego_agent_past"][:, 4], 0.0)

    def test_speed_augmentation_does_not_transform_scene(self) -> None:
        frame = _frame()
        augmentation = PlannerSpeedAugmentation(
            speed_scale_range=(1.2, 1.2),
            speed_noise_range=(0.0, 0.0),
            probability=1.0,
        )

        result = augmentation(frame)

        np.testing.assert_array_equal(
            result["neighbor_agents_past"], frame["neighbor_agents_past"]
        )
        np.testing.assert_allclose(result["ego_agent_past"][:, 4], 1.2)

    def test_pose_augmentation_checks_future_speed_without_current_speed(self) -> None:
        for augmentation_type in (
            PlannerRigidDataAugmentation,
            PlannerQuinticHermiteAugmentation,
        ):
            with self.subTest(augmentation_type=augmentation_type.__name__):
                frame = _frame()
                frame["ego_agent_past"][-1, 4] = 0.0
                augmentation = augmentation_type(
                    lateral_offset_range=(1.0, 1.0),
                    yaw_offset_range=(0.0, 0.0),
                    pose_probability=1.0,
                    pose_augmentation_speed_threshold=0.5,
                    pose_augmentation_speed_check_index=1,
                )

                result = augmentation(frame)

                self.assertFalse(
                    np.array_equal(result["goal_pose"], frame["goal_pose"])
                )

    def test_pose_augmentation_skips_when_speed_before_index_is_too_low(self) -> None:
        for augmentation_type in (
            PlannerRigidDataAugmentation,
            PlannerQuinticHermiteAugmentation,
        ):
            with self.subTest(augmentation_type=augmentation_type.__name__):
                frame = _frame()
                frame["ego_agent_future"][1, 4] = 0.4
                augmentation = augmentation_type(
                    lateral_offset_range=(1.0, 1.0),
                    yaw_offset_range=(0.0, 0.0),
                    pose_probability=1.0,
                    pose_augmentation_speed_threshold=0.5,
                    pose_augmentation_speed_check_index=1,
                )

                result = augmentation(frame)

                self.assertIs(result["goal_pose"], frame["goal_pose"])


class PlannerGoalTransformTest(unittest.TestCase):
    def test_replaces_nearby_goal_with_stopped_future_endpoint(self) -> None:
        frame = _frame()
        frame["goal_pose"] = _pose(10.0, 0.0, 1.0, 0.0)
        frame["ego_agent_future"][-1] = np.asarray(
            [8.0, 1.0, 0.0, 1.0, 0.05, 0.0], dtype=np.float32
        )

        result = PlannerGoalTransform(
            stop_speed_threshold=0.1,
            time_shift_probability=0.0,
            goal_tolerance_m=3.0,
        )(frame)

        np.testing.assert_array_equal(result["goal_pose"], _pose(8.0, 1.0, 1.0, 0.0))
        np.testing.assert_array_equal(frame["goal_pose"], _pose(10.0, 0.0))

    def test_keeps_goal_when_future_endpoint_is_too_far(self) -> None:
        frame = _frame()
        frame["goal_pose"] = _pose(21.0, 0.0)
        frame["ego_agent_future"][-1, 4] = 0.0

        result = PlannerGoalTransform(goal_tolerance_m=1.0)(frame)

        self.assertIs(result["goal_pose"], frame["goal_pose"])

    def test_keeps_goal_when_future_endpoint_is_moving(self) -> None:
        frame = _frame()
        frame["goal_pose"] = _pose(10.0, 0.0)
        frame["ego_agent_future"][-1, 4] = 0.2

        result = PlannerGoalTransform(stop_speed_threshold=0.1)(frame)

        self.assertIs(result["goal_pose"], frame["goal_pose"])

    def test_shifts_only_ego_time_and_pads_future_with_goal(self) -> None:
        frame = _frame()
        ego_states = np.asarray(
            [
                [-2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [3.0, 0.0, 1.0, 0.0, 0.5, 0.0],
                [4.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        frame["ego_agent_past"] = ego_states[:3]
        frame["ego_agent_future"] = ego_states[3:]
        frame["goal_pose"] = _pose(4.0, 0.0)
        original_neighbor_future = frame["neighbor_agents_future"].copy()
        transform = PlannerGoalTransform(
            time_shift_probability=1.0,
            goal_tolerance_m=0.1,
        )

        with patch("numpy.random.randint", return_value=2):
            result = transform(frame)

        np.testing.assert_allclose(result["ego_agent_past"][:, 0], [-2.0, -1.0, 0.0])
        np.testing.assert_allclose(
            result["ego_agent_future"][:, 0], [1.0, 2.0, 2.0, 2.0]
        )
        np.testing.assert_allclose(result["ego_agent_future"][-2:, 4:], 0.0)
        np.testing.assert_array_equal(
            result["neighbor_agents_future"][0, :, 0] + 2.0,
            original_neighbor_future[0, :, 0],
        )

    def test_does_not_shift_when_endpoint_does_not_contain_goal(self) -> None:
        frame = _frame()
        frame["goal_pose"] = _pose(10.0, 0.0)
        frame["ego_agent_future"][-1, :2] = (4.0, 0.0)
        transform = PlannerGoalTransform(
            time_shift_probability=1.0,
            goal_tolerance_m=1.0,
        )

        result = transform(frame)

        self.assertIs(result["ego_agent_future"], frame["ego_agent_future"])


class PlannerEgoShapeAugmentationTest(unittest.TestCase):
    def test_reduces_ego_shape_without_mutating_input(self) -> None:
        frame = _frame()
        original_shape = frame["ego_shape"].copy()
        augmentation = PlannerEgoShapeAugmentation(probability=1.0)

        with patch(
            "numpy.random.uniform",
            return_value=np.asarray([0.9, 0.95, 0.85], dtype=np.float32),
        ):
            result = augmentation(frame)

        np.testing.assert_allclose(
            result["ego_shape"], frame["ego_shape"] * [0.9, 0.95, 0.85]
        )
        self.assertEqual(result["ego_shape"].dtype, frame["ego_shape"].dtype)
        np.testing.assert_array_equal(frame["ego_shape"], original_shape)

    def test_keeps_ego_shape_when_skipped(self) -> None:
        frame = _frame()

        result = PlannerEgoShapeAugmentation(probability=0.0)(frame)

        self.assertIs(result["ego_shape"], frame["ego_shape"])


if __name__ == "__main__":
    unittest.main()
