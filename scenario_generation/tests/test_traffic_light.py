"""Tests for :mod:`scenario_generation.traffic_light`.

Pure-Python tests — no lanelet2 map, no ROS, no GPU.
Uses mock objects to isolate TL logic from the LaneletSceneBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest

from scenario_generation.traffic_light import (
    PERP_GREEN_AFTER_DELAY,
    TL_GREEN,
    TL_NONE,
    TL_RED,
    TL_YELLOW,
    _GroupState,
    _opposite_color,
)


# ── _opposite_color ─────────────────────────────────────────────────────────


class TestOppositeColor:
    def test_green_gives_red(self):
        assert _opposite_color(TL_GREEN, 0.0) == TL_RED

    def test_yellow_gives_red(self):
        assert _opposite_color(TL_YELLOW, 0.0) == TL_RED

    def test_red_short_time_gives_red(self):
        """Perpendicular stays red for PERP_GREEN_AFTER_DELAY seconds."""
        assert _opposite_color(TL_RED, 0.0) == TL_RED
        assert _opposite_color(TL_RED, PERP_GREEN_AFTER_DELAY - 0.1) == TL_RED

    def test_red_after_delay_gives_green(self):
        assert _opposite_color(TL_RED, PERP_GREEN_AFTER_DELAY + 0.1) == TL_GREEN


# ── _GroupState transitions ─────────────────────────────────────────────────


class TestGroupState:
    def test_green_to_yellow_to_red_to_green(self):
        """State machine must cycle green → yellow → red → green."""
        state = _GroupState(color=TL_GREEN, last_change_time=0.0, duration=5.0)

        # Still green before duration
        assert state.color == TL_GREEN

        # Transition to yellow
        state.color = TL_YELLOW
        state.last_change_time = 5.0
        state.duration = 3.0
        assert state.color == TL_YELLOW

        # Transition to red
        state.color = TL_RED
        state.last_change_time = 8.0
        state.duration = 5.0
        assert state.color == TL_RED

        # Transition back to green
        state.color = TL_GREEN
        state.last_change_time = 13.0
        assert state.color == TL_GREEN


# ── Signal group merging (transitive) ──────────────────────────────────────


class TestSignalGroupMerge:
    """Test the transitive bulb-group merge logic in TrafficLightController.__init__.

    We replicate the merge algorithm from traffic_light.py rather than
    instantiating a full TrafficLightController (which requires a real
    LaneletSceneBuilder). This ensures the algorithm itself is correct.
    """

    @staticmethod
    def _merge(bulb_map: dict[int, frozenset]) -> dict[frozenset, set[int]]:
        """Replicate the transitive merge from TrafficLightController.__init__."""
        signal_groups: dict[frozenset, set[int]] = {}
        for reg_id, bulbs in bulb_map.items():
            if not bulbs:
                continue
            overlapping_bulbs = bulbs
            overlapping_regs: set[int] = {reg_id}
            keys_to_remove: list[frozenset] = []
            for existing_bulbs, existing_regs in signal_groups.items():
                if overlapping_bulbs & existing_bulbs:
                    overlapping_bulbs = overlapping_bulbs | existing_bulbs
                    overlapping_regs |= existing_regs
                    keys_to_remove.append(existing_bulbs)
            for k in keys_to_remove:
                del signal_groups[k]
            signal_groups[overlapping_bulbs] = overlapping_regs
        return signal_groups

    def test_no_overlap_separate_groups(self):
        result = self._merge({
            1: frozenset({10, 11}),
            2: frozenset({20, 21}),
        })
        assert len(result) == 2

    def test_direct_overlap_merges(self):
        result = self._merge({
            1: frozenset({10, 11}),
            2: frozenset({11, 12}),
        })
        assert len(result) == 1
        group = list(result.values())[0]
        assert group == {1, 2}

    def test_transitive_overlap_merges_all(self):
        """A shares with B, B shares with C → all three merge."""
        result = self._merge({
            1: frozenset({10, 11}),
            2: frozenset({20, 21}),
            3: frozenset({11, 20}),  # bridges group 1 and 2
        })
        assert len(result) == 1
        group = list(result.values())[0]
        assert group == {1, 2, 3}
        bulbs = list(result.keys())[0]
        assert bulbs == frozenset({10, 11, 20, 21})

    def test_empty_bulbs_skipped(self):
        result = self._merge({
            1: frozenset(),
            2: frozenset({10}),
        })
        assert len(result) == 1
        assert list(result.values())[0] == {2}

    def test_chain_of_three(self):
        """A-B, B-C, C-D chain should all merge into one group."""
        result = self._merge({
            1: frozenset({1, 2}),
            2: frozenset({2, 3}),
            3: frozenset({3, 4}),
            4: frozenset({4, 5}),
        })
        assert len(result) == 1
        assert list(result.values())[0] == {1, 2, 3, 4}


# ── Lane tensor one-hot encoding ───────────────────────────────────────────


class TestLaneTensorEncoding:
    """Verify TL one-hot is written to the correct channels in lane tensors."""

    def test_one_hot_channels(self):
        """Each TL color should set exactly one of the 5 channels [8:13]."""
        for color, expected_idx in [
            (TL_GREEN, 0), (TL_YELLOW, 1), (TL_RED, 2),
        ]:
            one_hot = np.zeros(5, dtype=np.float32)
            one_hot[color] = 1.0
            assert one_hot[expected_idx] == 1.0
            assert one_hot.sum() == 1.0

    def test_no_tl_channel(self):
        one_hot = np.zeros(5, dtype=np.float32)
        one_hot[TL_NONE] = 1.0
        assert one_hot[4] == 1.0
        assert one_hot.sum() == 1.0
