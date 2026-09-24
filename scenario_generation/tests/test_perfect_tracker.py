"""Tests for the perfect tracker (exact placement, heading from the path)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scenario_generation.perfect_tracker import PerfectTracker


class TestPerfectTracker:
    def test_lands_exactly_on_first_point(self):
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        ref = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos, speed = tracker.track(x0, ref)
        assert pos[0] == pytest.approx(0.5)
        assert pos[1] == pytest.approx(0.0)
        assert speed == pytest.approx(5.0)

    def test_lands_exactly_on_first_point_on_a_curve(self):
        # Regression: the old tracker stepped along the CURRENT heading, so on a curve the
        # vehicle drifted sideways off the predicted point every step.
        tracker = PerfectTracker(dt=0.1)
        r = 20.0
        ang = np.linspace(0.05, 1.0, 40)
        ref = np.column_stack([r * np.sin(ang), r - r * np.cos(ang), ang])
        x0 = np.array([0.0, 0.0, 0.0, 10.0])
        pos, _ = tracker.track(x0, ref)
        assert pos[0] == pytest.approx(ref[0, 0])
        assert pos[1] == pytest.approx(ref[0, 1])

    def test_heading_from_path_not_from_reference_heading(self):
        # The path runs along +x; the reference's heading channel says 45 degrees.
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.3, 5.0])
        ref = np.array([[0.5, 0.0, math.pi / 4], [1.0, 0.0, math.pi / 4], [1.5, 0.0, math.pi / 4]])
        pos, _ = tracker.track(x0, ref)
        assert pos[2] == pytest.approx(0.0, abs=1e-6)

    def test_short_reference_keeps_current_heading(self):
        # Points closer together than MIN_HEADING_DISTANCE_M give no reliable direction.
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.7, 0.0])
        ref = np.array([[0.01, 0.0, 0.0], [0.02, 0.01, 0.0]])
        pos, _ = tracker.track(x0, ref)
        assert pos[2] == pytest.approx(0.7)

    def test_empty_reference(self):
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        pos, speed = tracker.track(x0, np.zeros((0, 3)))
        assert pos[0] == pytest.approx(0.0)
        assert speed == 0.0

    def test_reset_is_noop(self):
        tracker = PerfectTracker()
        tracker.reset()  # should not raise
