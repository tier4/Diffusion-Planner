"""Tests for placing and removing agents in a planner frame."""

from __future__ import annotations

import math
import unittest

import numpy as np

from diffusion_planner.analysis import neighbor_classes, neighbor_valid
from diffusion_planner.analysis.scene_edit import (
    DEFAULT_TIMESTEP_S,
    PlacedAgent,
    agent_history,
    edited_slots,
    free_slots,
    insert_agent,
    occupied_slots,
    read_agent,
    remove_agent,
    update_agent,
)
from diffusion_planner.data.dimensions import (
    AGENT_LABEL_DIM,
    AGENT_SHAPE_DIM,
    EGO_HISTORY_LENGTH,
)
from diffusion_planner.visualizer.schema import AgentLabelIndex, NeighborIndex

NUM_SLOTS = 6


def _frame(occupied: int = 2) -> dict[str, np.ndarray]:
    """A minimal frame with a few recorded agents."""
    poses = np.zeros((NUM_SLOTS, EGO_HISTORY_LENGTH, len(NeighborIndex)), np.float32)
    shapes = np.zeros((NUM_SLOTS, AGENT_SHAPE_DIM), np.float32)
    labels = np.zeros((NUM_SLOTS, AGENT_LABEL_DIM), np.float32)
    for slot in range(occupied):
        poses[slot, :, NeighborIndex.X] = 10.0 + slot
        poses[slot, :, NeighborIndex.COS_YAW] = 1.0
        shapes[slot] = (2.0, 4.0)
        labels[slot, AgentLabelIndex.IS_VEHICLE] = 1.0
    return {
        "neighbor_agents_past": poses,
        "agent_shape": shapes,
        "agent_label": labels,
    }


class AgentHistoryTest(unittest.TestCase):
    def test_last_row_is_the_placed_pose(self) -> None:
        agent = PlacedAgent(x_m=3.0, y_m=-4.0, heading_rad=0.5, speed_mps=6.0)

        history = agent_history(agent, EGO_HISTORY_LENGTH)

        self.assertEqual(history.shape, (EGO_HISTORY_LENGTH, len(NeighborIndex)))
        self.assertAlmostEqual(float(history[-1, NeighborIndex.X]), 3.0, places=5)
        self.assertAlmostEqual(float(history[-1, NeighborIndex.Y]), -4.0, places=5)
        self.assertAlmostEqual(
            float(history[-1, NeighborIndex.COS_YAW]), math.cos(0.5), places=5
        )

    def test_a_moving_agent_travels_speed_times_timestep(self) -> None:
        agent = PlacedAgent(x_m=0.0, y_m=0.0, speed_mps=10.0)

        history = agent_history(agent, 3)

        step = float(history[-1, NeighborIndex.X] - history[-2, NeighborIndex.X])
        self.assertAlmostEqual(step, 10.0 * DEFAULT_TIMESTEP_S, places=5)

    def test_a_stationary_agent_repeats_one_pose(self) -> None:
        agent = PlacedAgent(x_m=7.0, y_m=1.0, speed_mps=0.0)

        history = agent_history(agent, 4)

        for row in history:
            self.assertAlmostEqual(float(row[NeighborIndex.X]), 7.0, places=5)

    def test_placed_agents_are_always_occupied(self) -> None:
        """cos and sin cannot both be zero, so a placed row is never all-zero."""
        agent = PlacedAgent(x_m=0.0, y_m=0.0, heading_rad=0.0, speed_mps=0.0)

        history = agent_history(agent, EGO_HISTORY_LENGTH)

        self.assertTrue(bool(neighbor_valid(history[None])[0]))


class InsertAgentTest(unittest.TestCase):
    def test_fills_the_lowest_free_slot_without_touching_the_input(self) -> None:
        frame = _frame(occupied=2)
        before = frame["neighbor_agents_past"].copy()

        edited, slot = insert_agent(frame, PlacedAgent(x_m=5.0, y_m=2.0))

        self.assertEqual(slot, 2)
        np.testing.assert_array_equal(frame["neighbor_agents_past"], before)
        self.assertEqual(list(occupied_slots(edited)), [0, 1, 2])

    def test_inserted_unknown_agent_reads_back_as_unknown(self) -> None:
        frame = _frame()

        edited, slot = insert_agent(
            frame, PlacedAgent(x_m=8.0, y_m=-3.0, agent_class="unknown")
        )

        names = neighbor_classes(edited["agent_label"], edited["neighbor_agents_past"])
        self.assertEqual(names[slot], "unknown")
        self.assertEqual(
            float(edited["agent_label"][slot, AgentLabelIndex.IS_UNKNOWN]), 1.0
        )
        self.assertEqual(float(edited["agent_label"][slot].sum()), 1.0)

    def test_unlabeled_agent_is_valid_but_carries_no_class(self) -> None:
        frame = _frame()

        edited, slot = insert_agent(
            frame, PlacedAgent(x_m=4.0, y_m=0.0, agent_class="unlabeled")
        )

        names = neighbor_classes(edited["agent_label"], edited["neighbor_agents_past"])
        self.assertEqual(float(edited["agent_label"][slot].sum()), 0.0)
        self.assertTrue(bool(neighbor_valid(edited["neighbor_agents_past"])[slot]))
        self.assertEqual(names[slot], "unlabeled")

    def test_honours_an_explicit_slot_and_rejects_an_occupied_one(self) -> None:
        frame = _frame(occupied=2)

        edited, slot = insert_agent(frame, PlacedAgent(x_m=1.0, y_m=1.0), slot=5)

        self.assertEqual(slot, 5)
        with self.assertRaises(ValueError):
            insert_agent(frame, PlacedAgent(x_m=1.0, y_m=1.0), slot=0)

    def test_rejects_a_full_frame(self) -> None:
        frame = _frame(occupied=NUM_SLOTS)

        self.assertEqual(free_slots(frame).size, 0)
        with self.assertRaises(ValueError):
            insert_agent(frame, PlacedAgent(x_m=1.0, y_m=1.0))

    def test_rejects_an_unknown_class_name_and_bad_size(self) -> None:
        with self.assertRaises(ValueError):
            PlacedAgent(x_m=0.0, y_m=0.0, agent_class="truck")
        with self.assertRaises(ValueError):
            PlacedAgent(x_m=0.0, y_m=0.0, width_m=0.0)


class RemoveAndUpdateTest(unittest.TestCase):
    def test_insert_then_remove_restores_the_original_frame(self) -> None:
        frame = _frame(occupied=3)

        edited, slot = insert_agent(frame, PlacedAgent(x_m=9.0, y_m=9.0))
        restored = remove_agent(edited, slot)

        for name in ("neighbor_agents_past", "agent_shape", "agent_label"):
            np.testing.assert_array_equal(restored[name], frame[name])
        self.assertEqual(edited_slots(frame, restored).size, 0)

    def test_removing_an_empty_slot_is_an_error(self) -> None:
        frame = _frame(occupied=1)

        with self.assertRaises(ValueError):
            remove_agent(frame, 4)

    def test_update_replaces_class_and_pose_in_place(self) -> None:
        frame = _frame(occupied=1)

        updated = update_agent(
            frame, 0, PlacedAgent(x_m=-6.0, y_m=2.0, agent_class="pedestrian")
        )

        names = neighbor_classes(
            updated["agent_label"], updated["neighbor_agents_past"]
        )
        self.assertEqual(names[0], "pedestrian")
        self.assertAlmostEqual(
            float(updated["neighbor_agents_past"][0, -1, NeighborIndex.X]), -6.0, 5
        )
        self.assertEqual(list(occupied_slots(updated)), [0])


class ReadAgentTest(unittest.TestCase):
    def test_round_trips_a_placed_agent(self) -> None:
        frame = _frame(occupied=0)
        placed = PlacedAgent(
            x_m=12.0,
            y_m=-5.0,
            heading_rad=0.3,
            speed_mps=7.0,
            width_m=1.8,
            length_m=4.2,
            agent_class="unknown",
        )

        edited, slot = insert_agent(frame, placed)
        recovered = read_agent(edited, slot)

        assert recovered is not None
        self.assertAlmostEqual(recovered.x_m, placed.x_m, places=4)
        self.assertAlmostEqual(recovered.y_m, placed.y_m, places=4)
        self.assertAlmostEqual(recovered.heading_rad, placed.heading_rad, places=4)
        self.assertAlmostEqual(recovered.speed_mps, placed.speed_mps, places=3)
        self.assertEqual(recovered.agent_class, placed.agent_class)

    def test_returns_none_for_an_empty_slot(self) -> None:
        self.assertIsNone(read_agent(_frame(occupied=1), 3))


class EditedSlotsTest(unittest.TestCase):
    def test_reports_only_the_changed_slot(self) -> None:
        frame = _frame(occupied=2)

        edited, slot = insert_agent(frame, PlacedAgent(x_m=3.0, y_m=3.0))

        np.testing.assert_array_equal(edited_slots(frame, edited), [slot])

    def test_detects_a_label_only_change(self) -> None:
        frame = _frame(occupied=2)
        edited = {name: value.copy() for name, value in frame.items()}
        edited["agent_label"][1] = 0.0
        edited["agent_label"][1, AgentLabelIndex.IS_UNKNOWN] = 1.0

        np.testing.assert_array_equal(edited_slots(frame, edited), [1])


if __name__ == "__main__":
    unittest.main()
