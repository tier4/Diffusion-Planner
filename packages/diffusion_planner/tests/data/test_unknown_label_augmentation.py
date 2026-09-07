"""Tests for the unknown-class agent-label augmentation."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from diffusion_planner.data.dimensions import AGENT_LABEL_DIM
from diffusion_planner.data.transforms import PlannerUnknownLabelAugmentation
from diffusion_planner.data.transforms.unknown_label_augmentation import (
    KNOWN_LABEL_DIM,
    UNKNOWN_LABEL_INDEX,
)

HISTORY = 4


def _frame(
    classes: list[int | None], distances: list[float], label_dim: int = KNOWN_LABEL_DIM
) -> dict[str, NDArray[np.float32]]:
    """Build a frame whose agents carry the given classes at the given distances.

    A ``None`` class marks an empty neighbor slot: an all-zero pose history and
    an all-zero label row.
    """
    count = len(classes)
    neighbors = np.zeros((count, HISTORY, 4), dtype=np.float32)
    labels = np.zeros((count, label_dim), dtype=np.float32)
    for index, (label, distance) in enumerate(zip(classes, distances, strict=True)):
        if label is None:
            continue
        neighbors[index, :, 0] = distance
        neighbors[index, :, 2] = 1.0
        labels[index, label] = 1.0
    return {"neighbor_agents_past": neighbors, "agent_label": labels}


class WidenLabelsTest(unittest.TestCase):
    def test_appends_unknown_column_to_legacy_labels(self) -> None:
        frame = _frame([0, 1, None], [5.0, 5.0, 0.0])

        result = PlannerUnknownLabelAugmentation()(frame)

        self.assertEqual(result["agent_label"].shape, (3, AGENT_LABEL_DIM))
        np.testing.assert_array_equal(result["agent_label"][0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["agent_label"][1], [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["agent_label"][2], np.zeros(4))

    def test_keeps_four_column_labels_unchanged(self) -> None:
        frame = _frame([0, None], [5.0, 0.0], label_dim=AGENT_LABEL_DIM)

        result = PlannerUnknownLabelAugmentation()(frame)

        np.testing.assert_array_equal(result["agent_label"], frame["agent_label"])

    def test_does_not_mutate_the_input_frame(self) -> None:
        frame = _frame([0, 1], [5.0, 5.0])
        original = frame["agent_label"].copy()

        PlannerUnknownLabelAugmentation(probability=1.0)(frame)

        np.testing.assert_array_equal(frame["agent_label"], original)

    def test_rejects_labels_wider_than_the_model(self) -> None:
        frame = _frame([0], [5.0], label_dim=AGENT_LABEL_DIM + 1)

        with self.assertRaises(ValueError):
            PlannerUnknownLabelAugmentation()(frame)


class RenameTest(unittest.TestCase):
    def test_probability_zero_renames_nothing(self) -> None:
        frame = _frame([0, 1, 2], [5.0, 5.0, 5.0])

        result = PlannerUnknownLabelAugmentation(probability=0.0)(frame)

        self.assertEqual(result["agent_label"][:, UNKNOWN_LABEL_INDEX].sum(), 0.0)

    def test_probability_one_renames_every_labelled_agent(self) -> None:
        frame = _frame([0, 1, 2, None], [5.0, 5.0, 5.0, 0.0])

        result = PlannerUnknownLabelAugmentation(probability=1.0)(frame)

        labels = result["agent_label"]
        np.testing.assert_array_equal(labels[:3, UNKNOWN_LABEL_INDEX], np.ones(3))
        np.testing.assert_array_equal(labels[:3, :KNOWN_LABEL_DIM], np.zeros((3, 3)))
        np.testing.assert_array_equal(labels[3], np.zeros(AGENT_LABEL_DIM))

    def test_renamed_rows_stay_one_hot(self) -> None:
        frame = _frame([0, 1, 2], [5.0, 5.0, 5.0])

        labels = PlannerUnknownLabelAugmentation(probability=1.0)(frame)["agent_label"]

        np.testing.assert_array_equal(labels.sum(axis=-1), np.ones(3))

    def test_already_unknown_agents_are_left_alone(self) -> None:
        frame = _frame([UNKNOWN_LABEL_INDEX], [5.0], label_dim=AGENT_LABEL_DIM)

        labels = PlannerUnknownLabelAugmentation(probability=1.0)(frame)["agent_label"]

        np.testing.assert_array_equal(labels[0], [0.0, 0.0, 0.0, 1.0])

    def test_per_class_probabilities_hold_statistically(self) -> None:
        augmentation = PlannerUnknownLabelAugmentation(
            probability=0.0,
            probability_vehicle=0.2,
            probability_pedestrian=0.8,
            probability_bicycle=0.5,
        )
        counts = np.zeros(KNOWN_LABEL_DIM)
        trials = 4000
        np.random.seed(0)
        for _ in range(trials):
            frame = _frame([0, 1, 2], [5.0, 5.0, 5.0])
            labels = augmentation(frame)["agent_label"]
            counts += labels[:, UNKNOWN_LABEL_INDEX]
        rates = counts / trials
        np.testing.assert_allclose(rates, [0.2, 0.8, 0.5], atol=0.03)

    def test_distance_scaling_raises_the_rate_with_range(self) -> None:
        augmentation = PlannerUnknownLabelAugmentation(
            probability=0.25,
            distance_scale_max=2.0,
            distance_scale_range_m=50.0,
        )
        counts = np.zeros(3)
        trials = 4000
        np.random.seed(1)
        for _ in range(trials):
            frame = _frame([0, 0, 0], [0.0, 25.0, 100.0])
            labels = augmentation(frame)["agent_label"]
            counts += labels[:, UNKNOWN_LABEL_INDEX]
        rates = counts / trials
        # 0 m -> 0.25, 25 m -> 0.25 * 1.5, beyond the range -> 0.25 * 2.0.
        np.testing.assert_allclose(rates, [0.25, 0.375, 0.5], atol=0.03)

    def test_probability_cap_bounds_the_scaled_rate(self) -> None:
        augmentation = PlannerUnknownLabelAugmentation(
            probability=0.5,
            distance_scale_max=2.0,
            distance_scale_range_m=50.0,
            probability_cap=0.6,
        )
        counts = 0.0
        trials = 4000
        np.random.seed(2)
        for _ in range(trials):
            frame = _frame([0], [100.0])
            counts += augmentation(frame)["agent_label"][0, UNKNOWN_LABEL_INDEX]
        # Without the cap the rate would be 1.0.
        self.assertAlmostEqual(counts / trials, 0.6, delta=0.03)

    def test_frame_cap_limits_how_many_agents_are_renamed(self) -> None:
        augmentation = PlannerUnknownLabelAugmentation(
            probability=1.0, max_renamed_fraction=0.5
        )
        np.random.seed(3)
        for _ in range(20):
            frame = _frame([0, 1, 2, 0], [5.0, 5.0, 5.0, 5.0])
            labels = augmentation(frame)["agent_label"]
            self.assertEqual(labels[:, UNKNOWN_LABEL_INDEX].sum(), 2.0)

    def test_empty_slots_never_become_unknown(self) -> None:
        frame = _frame([None, None, 0], [0.0, 0.0, 5.0])

        labels = PlannerUnknownLabelAugmentation(probability=1.0)(frame)["agent_label"]

        np.testing.assert_array_equal(labels[0], np.zeros(AGENT_LABEL_DIM))
        np.testing.assert_array_equal(labels[1], np.zeros(AGENT_LABEL_DIM))
        np.testing.assert_array_equal(labels[2], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
