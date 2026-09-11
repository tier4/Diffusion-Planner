"""Tests for closed-loop rollout bookkeeping."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from diffusion_planner.analysis.closed_loop import (
    _advance_histories,
    _relative_pose,
    map_coverage_m,
    rollout,
)
from diffusion_planner.data.dimensions import (
    EGO_HISTORY_LENGTH,
    EGO_STATE_DIM,
    LANE_GEOMETRY_DIM,
    LANE_LENGTH,
    PLANNER_INPUT_SHAPES,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.visualizer.schema import EgoIndex, NeighborIndex, PoseIndex

NUM_SLOTS = 4


class RelativePoseTest(unittest.TestCase):
    def test_identity_when_the_origin_is_the_frame_origin(self) -> None:
        origin = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        target = np.array([3.0, 4.0, 1.0, 0.0], dtype=np.float32)

        result = _relative_pose(origin, target)

        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_subtracts_translation(self) -> None:
        origin = np.array([2.0, 1.0, 1.0, 0.0], dtype=np.float32)
        target = np.array([5.0, 1.0, 1.0, 0.0], dtype=np.float32)

        result = _relative_pose(origin, target)

        self.assertAlmostEqual(float(result[PoseIndex.X]), 3.0, places=5)
        self.assertAlmostEqual(float(result[PoseIndex.Y]), 0.0, places=5)

    def test_rotates_into_the_origin_heading(self) -> None:
        """A pose straight ahead of an origin facing +y is +x in its frame."""
        origin = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        target = np.array([0.0, 5.0, 0.0, 1.0], dtype=np.float32)

        result = _relative_pose(origin, target)

        self.assertAlmostEqual(float(result[PoseIndex.X]), 5.0, places=5)
        self.assertAlmostEqual(float(result[PoseIndex.Y]), 0.0, places=5)
        self.assertAlmostEqual(float(result[PoseIndex.COS_YAW]), 1.0, places=5)

    def test_preserves_relative_heading(self) -> None:
        origin = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        quarter = np.array(
            [1.0, 0.0, math.cos(math.pi / 2), math.sin(math.pi / 2)], dtype=np.float32
        )

        result = _relative_pose(origin, quarter)

        self.assertAlmostEqual(float(result[PoseIndex.COS_YAW]), 0.0, places=5)
        self.assertAlmostEqual(float(result[PoseIndex.SIN_YAW]), 1.0, places=5)


class MapCoverageTest(unittest.TestCase):
    def test_reports_the_furthest_lane_point_ahead(self) -> None:
        lanes = np.zeros((2, LANE_LENGTH, LANE_GEOMETRY_DIM), dtype=np.float32)
        lanes[0, :, PoseIndex.X] = np.linspace(1.0, 40.0, LANE_LENGTH)
        lanes[1, :, PoseIndex.X] = np.linspace(1.0, 90.0, LANE_LENGTH)

        self.assertAlmostEqual(map_coverage_m({"lanes": lanes}), 90.0, places=4)

    def test_ignores_empty_lanes_and_geometry_behind_the_ego(self) -> None:
        lanes = np.zeros((2, LANE_LENGTH, LANE_GEOMETRY_DIM), dtype=np.float32)
        lanes[0, :, PoseIndex.X] = -25.0

        self.assertEqual(map_coverage_m({"lanes": lanes}), 0.0)


class AdvanceHistoriesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = {
            "ego_agent_past": np.zeros(
                (EGO_HISTORY_LENGTH, EGO_STATE_DIM), dtype=np.float32
            ),
            "neighbor_agents_past": np.zeros(
                (NUM_SLOTS, EGO_HISTORY_LENGTH, len(NeighborIndex)), dtype=np.float32
            ),
        }
        self.frame["ego_agent_past"][:, EgoIndex.COS_YAW] = 1.0
        self.frame["neighbor_agents_past"][0, :, NeighborIndex.X] = 9.0
        self.frame["neighbor_agents_past"][0, :, NeighborIndex.COS_YAW] = 1.0
        self.prediction = np.zeros(
            (1 + NUM_SLOTS, TRAJECTORY_LENGTH, 4), dtype=np.float32
        )
        self.prediction[:, :, PoseIndex.COS_YAW] = 1.0
        self.prediction[1, 0, NeighborIndex.X] = 11.0

    def test_ego_history_keeps_its_length_and_ends_at_the_origin(self) -> None:
        result = _advance_histories(dict(self.frame), self.prediction, 0, 0, 1.3, 0.1)

        ego = result["ego_agent_past"]
        self.assertEqual(ego.shape, (EGO_HISTORY_LENGTH, EGO_STATE_DIM))
        self.assertAlmostEqual(float(ego[-1, EgoIndex.X]), 0.0)
        self.assertAlmostEqual(float(ego[-1, EgoIndex.COS_YAW]), 1.0)

    def test_ego_velocity_follows_the_step_taken(self) -> None:
        result = _advance_histories(dict(self.frame), self.prediction, 0, 0, 1.3, 0.1)

        self.assertAlmostEqual(
            float(result["ego_agent_past"][-1, EgoIndex.VELOCITY]), 13.0, places=4
        )

    def test_occupied_neighbors_move_and_empty_slots_stay_empty(self) -> None:
        result = _advance_histories(dict(self.frame), self.prediction, 0, 0, 1.3, 0.1)

        neighbors = result["neighbor_agents_past"]
        self.assertAlmostEqual(float(neighbors[0, -1, NeighborIndex.X]), 11.0, places=4)
        self.assertEqual(float(np.abs(neighbors[1:]).sum()), 0.0)

    def test_history_shape_is_preserved(self) -> None:
        result = _advance_histories(dict(self.frame), self.prediction, 0, 0, 1.3, 0.1)

        self.assertEqual(
            result["neighbor_agents_past"].shape,
            (NUM_SLOTS, EGO_HISTORY_LENGTH, len(NeighborIndex)),
        )


class RolloutValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in PLANNER_INPUT_SHAPES.items()
        }

    def test_rejects_invalid_arguments_before_running_the_model(self) -> None:
        model = torch.nn.Identity()

        for kwargs in (
            {"steps": 0},
            {"steps": 1, "horizon_index": TRAJECTORY_LENGTH},
            {"steps": 1, "replan_every": 0},
            {"steps": 1, "horizon_index": TRAJECTORY_LENGTH - 1, "replan_every": 5},
        ):
            with self.assertRaises(ValueError):
                next(rollout(model, self.frame, **kwargs))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
