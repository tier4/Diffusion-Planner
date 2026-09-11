"""Tests for the attention view's pure display helpers."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from diffusion_planner.analysis import SceneTokenLayout
from diffusion_planner_dashboard.services.attention import AttentionReading
from diffusion_planner_dashboard.views.attention import (
    _attention_overlay,
    _relative_cutoff,
)

ALL_CLASSES = ["vehicle", "pedestrian", "bicycle", "unknown", "unlabeled"]


def _record(
    block: str, index: int, pct: float, agent_class: str | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "token_index": index,
        "block": block,
        "block_index": index,
        "attention": pct / 100.0,
        "attention_pct": pct,
        "x_m": float(index),
        "y_m": 0.0,
        "distance_m": float(index),
    }
    if agent_class is not None:
        record["agent_class"] = agent_class
    return record


def _reading(records: list[dict[str, Any]]) -> AttentionReading:
    layout = SceneTokenLayout.from_dimensions()
    return AttentionReading(
        attention=np.zeros(layout.total, dtype=np.float32),
        records=records,
        layout=layout,
        layer_count=6,
        head_count=12,
    )


class RelativeCutoffTest(unittest.TestCase):
    def test_scales_against_the_frames_top_token(self) -> None:
        """Absolute shares sit under 1%, so the slider must be relative."""
        reading = _reading([_record("lanes", 0, 0.8), _record("lanes", 1, 0.2)])

        self.assertAlmostEqual(_relative_cutoff(reading, 0.0), 0.0)
        self.assertAlmostEqual(_relative_cutoff(reading, 50.0), 0.4)
        self.assertAlmostEqual(_relative_cutoff(reading, 100.0), 0.8)

    def test_handles_a_frame_with_no_records(self) -> None:
        self.assertEqual(_relative_cutoff(_reading([]), 50.0), 0.0)


class AttentionOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reading = _reading(
            [
                _record("neighbors", 0, 1.0, "vehicle"),
                _record("neighbors", 1, 0.5, "unknown"),
                _record("lanes", 2, 0.25),
            ]
        )

    def test_keeps_every_token_at_zero_threshold(self) -> None:
        overlay = _attention_overlay(
            self.reading, 0.0, ["neighbors", "lanes"], ALL_CLASSES
        )

        assert overlay is not None
        self.assertEqual(len(overlay.x), 3)

    def test_threshold_is_relative_to_the_top_token(self) -> None:
        overlay = _attention_overlay(
            self.reading, 50.0, ["neighbors", "lanes"], ALL_CLASSES
        )

        assert overlay is not None
        self.assertEqual(len(overlay.x), 2)

    def test_filters_by_block_and_by_agent_class(self) -> None:
        blocks_only = _attention_overlay(self.reading, 0.0, ["lanes"], ALL_CLASSES)
        unknown_only = _attention_overlay(self.reading, 0.0, ["neighbors"], ["unknown"])

        assert blocks_only is not None
        assert unknown_only is not None
        self.assertEqual(len(blocks_only.x), 1)
        self.assertEqual(len(unknown_only.x), 1)
        self.assertIn("unknown", unknown_only.text[0])

    def test_returns_none_when_nothing_survives(self) -> None:
        self.assertIsNone(_attention_overlay(self.reading, 0.0, [], ALL_CLASSES))

    def test_marker_size_grows_with_share(self) -> None:
        overlay = _attention_overlay(
            self.reading, 0.0, ["neighbors", "lanes"], ALL_CLASSES
        )

        assert overlay is not None
        sizes = list(overlay.marker.size)
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_skips_tokens_without_a_position(self) -> None:
        record = _record("ego_shape", 3, 0.9)
        record["x_m"] = None
        reading = _reading([*self.reading.records, record])

        overlay = _attention_overlay(
            reading, 0.0, ["neighbors", "lanes", "ego_shape"], ALL_CLASSES
        )

        assert overlay is not None
        self.assertEqual(len(overlay.x), 3)


if __name__ == "__main__":
    unittest.main()
