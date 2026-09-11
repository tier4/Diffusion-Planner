"""Tests for fusion attention capture and summaries."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from diffusion_planner.analysis import (
    AGENT_CLASS_NAMES,
    EMPTY_SLOT_NAME,
    UNLABELED_NAME,
    SceneTokenLayout,
    capture_fusion_attention,
    class_summary,
    fusion_attention,
    neighbor_classes,
    neighbor_valid,
    token_records,
)
from diffusion_planner.data.dimensions import (
    AGENT_LABEL_DIM,
    EGO_HISTORY_LENGTH,
    MAX_NUM_NEIGHBORS,
    NUM_INTERSECTION_AREAS,
    NUM_LANE_SEGMENTS,
    NUM_ROAD_BORDERS,
    NUM_ROUTE_SEGMENTS,
    NUM_STOP_LINES,
    PLANNER_INPUT_SHAPES,
)
from diffusion_planner.models.encoder import SceneEncoder

FUSION_DEPTH = 2
NUM_HEADS = 2
HIDDEN_DIM = 8


def _scene_input(batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Build a full-size but zero-valued planner input map."""
    return {
        name: torch.zeros(batch_size, *shape)
        for name, shape in PLANNER_INPUT_SHAPES.items()
    }


def _occupy_neighbor(
    input_data: dict[str, torch.Tensor],
    slot: int,
    label: int | None,
    position: tuple[float, float] = (5.0, 0.0),
) -> None:
    """Mark one neighbor slot occupied, optionally with a class one-hot."""
    input_data["neighbor_agents_past"][0, slot, :, 0] = position[0]
    input_data["neighbor_agents_past"][0, slot, :, 1] = position[1]
    if label is not None:
        input_data["agent_label"][0, slot, label] = 1.0


class SceneTokenLayoutTest(unittest.TestCase):
    def test_total_matches_the_input_schema(self) -> None:
        layout = SceneTokenLayout.from_dimensions()

        expected = (
            MAX_NUM_NEIGHBORS
            + NUM_LANE_SEGMENTS
            + NUM_ROUTE_SEGMENTS
            + NUM_INTERSECTION_AREAS
            + NUM_STOP_LINES
            + NUM_ROAD_BORDERS
            + 3
        )
        self.assertEqual(layout.total, expected)

    def test_blocks_are_contiguous_and_ordered(self) -> None:
        layout = SceneTokenLayout.from_dimensions()

        cursor = 0
        for block in layout.blocks:
            self.assertEqual(block.start, cursor)
            self.assertGreater(len(block), 0)
            cursor = block.stop
        self.assertEqual(cursor, layout.total)

    def test_class_of_round_trips_every_block(self) -> None:
        layout = SceneTokenLayout.from_dimensions()

        for block in layout.blocks:
            name, local = layout.class_of(block.start)
            self.assertEqual((name, local), (block.name, 0))
            name, local = layout.class_of(block.stop - 1)
            self.assertEqual((name, local), (block.name, len(block) - 1))

    def test_ego_query_index_is_the_ego_history_token(self) -> None:
        layout = SceneTokenLayout.from_dimensions()

        self.assertEqual(layout.ego_query_index, layout.block("ego_history").start)
        self.assertEqual(len(layout.block("ego_history")), 1)

    def test_rejects_unknown_block_and_out_of_range_token(self) -> None:
        layout = SceneTokenLayout.from_dimensions()

        with self.assertRaises(KeyError):
            layout.block("nonexistent")
        with self.assertRaises(IndexError):
            layout.class_of(layout.total)


class NeighborClassesTest(unittest.TestCase):
    def test_empty_slots_never_become_a_class(self) -> None:
        """``argmax`` of an all-zero label row would otherwise report vehicle."""
        poses = np.zeros((4, EGO_HISTORY_LENGTH, 4), dtype=np.float32)
        labels = np.zeros((4, AGENT_LABEL_DIM), dtype=np.float32)
        poses[1, :, 0] = 3.0
        labels[1, 0] = 1.0

        names = neighbor_classes(labels, poses)

        self.assertEqual(names[0], EMPTY_SLOT_NAME)
        self.assertEqual(names[1], "vehicle")
        self.assertEqual(names[2], EMPTY_SLOT_NAME)
        self.assertEqual(names[3], EMPTY_SLOT_NAME)

    def test_names_every_class_including_unknown(self) -> None:
        count = AGENT_LABEL_DIM
        poses = np.zeros((count, EGO_HISTORY_LENGTH, 4), dtype=np.float32)
        labels = np.zeros((count, AGENT_LABEL_DIM), dtype=np.float32)
        for index in range(count):
            poses[index, :, 0] = 1.0
            labels[index, index] = 1.0

        names = neighbor_classes(labels, poses)

        self.assertEqual(tuple(names), AGENT_CLASS_NAMES)
        self.assertIn("unknown", AGENT_CLASS_NAMES)

    def test_occupied_but_unlabeled_is_distinct_from_unknown(self) -> None:
        poses = np.zeros((2, EGO_HISTORY_LENGTH, 4), dtype=np.float32)
        labels = np.zeros((2, AGENT_LABEL_DIM), dtype=np.float32)
        poses[:, :, 0] = 2.0
        labels[1, AGENT_LABEL_DIM - 1] = 1.0

        names = neighbor_classes(labels, poses)

        self.assertEqual(names[0], UNLABELED_NAME)
        self.assertEqual(names[1], "unknown")

    def test_validity_comes_from_poses_not_labels(self) -> None:
        poses = np.zeros((2, EGO_HISTORY_LENGTH, 4), dtype=np.float32)
        labels = np.zeros((2, AGENT_LABEL_DIM), dtype=np.float32)
        labels[0, 0] = 1.0  # a label on an empty slot must not make it valid
        poses[1, :, 1] = 4.0

        valid = neighbor_valid(poses)

        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[1]))
        self.assertEqual(neighbor_classes(labels, poses)[0], EMPTY_SLOT_NAME)


class CaptureFusionAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.encoder = SceneEncoder(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            fusion_depth=FUSION_DEPTH,
            encoder_depth=1,
            mixer_hidden_dim=4,
        ).eval()
        self.layout = SceneTokenLayout.from_dimensions()
        self.input_data = _scene_input()
        _occupy_neighbor(self.input_data, 0, label=0, position=(5.0, 1.0))
        _occupy_neighbor(self.input_data, 1, AGENT_LABEL_DIM - 1, position=(10.0, 2.0))

    def test_captures_every_fusion_layer(self) -> None:
        with capture_fusion_attention(self.encoder) as capture, torch.no_grad():
            self.encoder(self.input_data)

        self.assertEqual(capture.layer_count, FUSION_DEPTH)
        weights = capture.by_layer()[0].weights
        self.assertEqual(
            weights.shape,
            (1, NUM_HEADS, self.layout.total, self.layout.total),
        )

    def test_rows_sum_to_one_and_padding_receives_nothing(self) -> None:
        with capture_fusion_attention(self.encoder) as capture, torch.no_grad():
            _, mask = self.encoder(self.input_data)

        attention = fusion_attention(capture)
        torch.testing.assert_close(
            attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1))
        )
        self.assertEqual(float(attention[0][:, mask[0]].abs().max()), 0.0)

    def test_restores_forward_and_the_fast_path(self) -> None:
        layers = self.encoder.fusion_encoder.transformer.layers
        before = [layer.self_attn.forward for layer in layers]
        fastpath = torch.backends.mha.get_fastpath_enabled()

        with capture_fusion_attention(self.encoder), torch.no_grad():
            self.encoder(self.input_data)

        self.assertEqual([layer.self_attn.forward for layer in layers], before)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), fastpath)

    def test_raises_when_nothing_was_captured(self) -> None:
        """A silent bypass would read as 'this scene has no attention'."""
        with self.assertRaises(RuntimeError), capture_fusion_attention(self.encoder):
            pass

    def test_layer_and_head_selectors(self) -> None:
        with capture_fusion_attention(self.encoder) as capture, torch.no_grad():
            self.encoder(self.input_data)

        mean = fusion_attention(capture, layer="mean")
        last = fusion_attention(capture, layer="last")
        first = fusion_attention(capture, layer=0)
        head = fusion_attention(capture, layer=0, head=1)

        self.assertEqual(mean.shape, last.shape)
        self.assertFalse(torch.equal(first, last))
        torch.testing.assert_close(head.sum(dim=-1), torch.ones_like(head.sum(dim=-1)))
        with self.assertRaises(ValueError):
            fusion_attention(capture, layer=FUSION_DEPTH)
        with self.assertRaises(ValueError):
            fusion_attention(capture, layer=0, head=NUM_HEADS)

    def test_second_pass_supersedes_the_first(self) -> None:
        with capture_fusion_attention(self.encoder) as capture, torch.no_grad():
            self.encoder(self.input_data)
            self.assertEqual(len(capture.records), FUSION_DEPTH)
            self.encoder(self.input_data)

        self.assertEqual(len(capture.records), 2 * FUSION_DEPTH)
        self.assertEqual(capture.layer_count, FUSION_DEPTH)


class TokenRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layout = SceneTokenLayout.from_dimensions()
        self.frame = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in PLANNER_INPUT_SHAPES.items()
        }
        self.frame["neighbor_agents_past"][0, :, 0] = 6.0
        self.frame["agent_label"][0, 0] = 1.0
        self.frame["neighbor_agents_past"][1, :, 0] = 8.0
        self.frame["agent_label"][1, AGENT_LABEL_DIM - 1] = 1.0
        self.frame["lanes"][0, :, 0] = 12.0

    def test_skips_empty_slots_and_sorts_by_attention(self) -> None:
        attention = np.zeros(self.layout.total, dtype=np.float32)
        attention[0] = 0.2
        attention[1] = 0.5
        attention[self.layout.block("lanes").start] = 0.3

        records = token_records(self.frame, attention, self.layout)

        neighbors = [r for r in records if r["block"] == "neighbors"]
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(
            [r["attention"] for r in records],
            sorted((r["attention"] for r in records), reverse=True),
        )
        self.assertEqual(records[0]["block"], "neighbors")
        self.assertEqual(records[0]["agent_class"], "unknown")

    def test_records_carry_position_and_distance(self) -> None:
        attention = np.zeros(self.layout.total, dtype=np.float32)

        records = token_records(self.frame, attention, self.layout)
        first = next(
            r for r in records if r["block_index"] == 0 and r["block"] == "neighbors"
        )

        self.assertAlmostEqual(first["x_m"], 6.0, places=5)
        self.assertAlmostEqual(first["distance_m"], 6.0, places=5)

    def test_rejects_mismatched_attention_length(self) -> None:
        with self.assertRaises(ValueError):
            token_records(self.frame, np.zeros(3, dtype=np.float32), self.layout)


class ClassSummaryTest(unittest.TestCase):
    def test_shares_and_selectivity(self) -> None:
        records = [
            {"block": "neighbors", "agent_class": "vehicle", "attention": 0.6},
            {"block": "neighbors", "agent_class": "unknown", "attention": 0.2},
            {"block": "lanes", "attention": 0.2},
        ]

        by_block = class_summary(records)
        by_class = class_summary(records, key="agent_class")

        self.assertAlmostEqual(sum(v["share"] for v in by_block.values()), 1.0)
        self.assertAlmostEqual(by_block["neighbors"]["share"], 0.8)
        # Two of three tokens hold 80% of attention: 0.8 / (2/3).
        self.assertAlmostEqual(by_block["neighbors"]["selectivity"], 1.2)
        self.assertAlmostEqual(by_class["unknown"]["share"], 0.25)

    def test_ignores_records_missing_the_key(self) -> None:
        records = [
            {"block": "neighbors", "agent_class": "vehicle", "attention": 1.0},
            {"block": "lanes", "attention": 1.0},
        ]

        by_class = class_summary(records, key="agent_class")

        self.assertEqual(list(by_class), ["vehicle"])


if __name__ == "__main__":
    unittest.main()
