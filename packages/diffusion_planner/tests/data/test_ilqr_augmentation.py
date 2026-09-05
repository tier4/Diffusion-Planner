"""Minimal behavioral checks for iLQR trajectory augmentation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from diffusion_planner.data.transforms import PlannerILQRAugmentation


class PlannerILQRAugmentationTest(unittest.TestCase):
    def test_speed_skip_uses_only_refine_endpoint_for_minimum(self) -> None:
        optimizer = PlannerILQRAugmentation(
            num_refine=2,
            pose_augmentation_endpoint_speed_threshold=2.0,
        )
        future = np.zeros((4, 6), dtype=np.float32)
        future[:, 4] = (3.0, 0.0, 3.0, 0.0)

        self.assertTrue(
            optimizer._has_sufficient_future_speed({"ego_agent_future": future})
        )

    def test_skips_pose_augmentation_when_future_speed_is_too_low(self) -> None:
        optimizer = PlannerILQRAugmentation(
            lateral_offset_range=(1.0, 1.0),
            yaw_offset_range=(0.1, 0.1),
            pose_probability=1.0,
            pose_augmentation_endpoint_speed_threshold=2.0,
            num_refine=2,
        )
        past = np.zeros((2, 6), dtype=np.float32)
        past[:, 2] = 1.0
        past[:, 4] = 3.0
        future = np.zeros((4, 6), dtype=np.float32)
        future[:, 2] = 1.0
        future[:, 4] = (3.0, 3.0, 1.0, 3.0)
        frame = {"ego_agent_past": past, "ego_agent_future": future}

        result = optimizer(frame)

        self.assertIs(result["ego_agent_past"], past)
        self.assertIs(result["ego_agent_future"], future)

    def test_tracks_straight_reference_without_steering(self) -> None:
        optimizer = PlannerILQRAugmentation(
            num_refine=19, max_iterations=20, steering_rate_weight=1.0
        )
        horizon = 20
        speed = 5.0
        times = np.arange(horizon + 1) * optimizer.dt
        reference = np.column_stack(
            (speed * times, np.zeros(horizon + 1), np.zeros(horizon + 1))
        )
        velocity_reference = np.full(horizon, speed)
        initial_state = np.asarray((0.0, 0.0, 0.0, speed, 0.0))
        initial_controls = np.column_stack((velocity_reference, np.zeros(horizon)))

        solution = optimizer.solve(
            initial_state, reference, velocity_reference, initial_controls
        )

        self.assertIsNotNone(solution)
        states, controls = solution  # type: ignore[misc]
        np.testing.assert_allclose(states[:, :3], reference, atol=1e-8)
        np.testing.assert_allclose(controls[:, 1], 0.0, atol=1e-8)

    def test_finds_steering_for_constant_curvature(self) -> None:
        optimizer = PlannerILQRAugmentation(
            num_refine=39,
            max_iterations=30,
            state_weights=(10.0, 10.0, 5.0),
            terminal_weight_scale=20.0,
            steering_weight=0.001,
            steering_rate_weight=0.1,
        )
        horizon = 40
        speed = 5.0
        radius = 20.0
        yaw = speed * np.arange(horizon + 1) * optimizer.dt / radius
        reference = np.column_stack(
            (radius * np.sin(yaw), radius * (1.0 - np.cos(yaw)), yaw)
        )
        velocity_reference = np.full(horizon, speed)
        expected_steering = math.atan(optimizer.wheelbase / radius)
        initial_state = np.asarray((0.0, 0.0, 0.0, speed, expected_steering))
        initial_controls = np.column_stack(
            (velocity_reference, np.full(horizon, expected_steering))
        )

        solution = optimizer.solve(
            initial_state, reference, velocity_reference, initial_controls
        )

        self.assertIsNotNone(solution)
        states, controls = solution  # type: ignore[misc]
        self.assertLess(np.linalg.norm(states[-1, :2] - reference[-1, :2]), 0.2)
        self.assertAlmostEqual(
            float(np.median(controls[:, 1])), expected_steering, delta=0.03
        )

    def test_refined_future_is_finite_and_preserves_suffix(self) -> None:
        optimizer = PlannerILQRAugmentation(num_refine=5)
        past = np.zeros((2, 6), dtype=np.float32)
        past[:, 2] = 1.0
        past[:, 4] = 3.0
        future = np.zeros((10, 6), dtype=np.float32)
        future[:, 0] = 3.0 * optimizer.dt * np.arange(1, 11)
        future[:, 2] = 1.0
        future[:, 4] = 3.0

        refined = optimizer.refine_future(future, past)

        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertTrue(np.all(np.isfinite(refined)))
        np.testing.assert_array_equal(refined[6:], future[6:])
        np.testing.assert_allclose(np.linalg.norm(refined[:6, 2:4], axis=1), 1.0)
