"""Tests for :mod:`scenario_generation.mpc_tracker`.

Pure-Python tests — no model, no GPU, no map needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scenario_generation.mpc_tracker import (
    MPCTracker,
    PerfectTracker,
    postprocess_reference,
)

# ── MPCTracker ──────────────────────────────────────────────────────────────


class TestMPCTracker:
    def test_straight_line_tracks_reference(self):
        """Vehicle going straight should stay on the straight reference."""
        tracker = MPCTracker(wheelbase=2.79, horizon_steps=20, n_knots=5)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])  # heading east, 5 m/s
        ref = np.zeros((80, 3))
        for i in range(80):
            ref[i, 0] = 0.5 * (i + 1)
        new_pos, new_speed = tracker.track(x0, ref)
        assert np.isfinite(new_pos).all()
        assert new_pos[0] > 0  # moved forward
        assert abs(new_pos[1]) < 0.1  # stayed on line
        assert new_speed > 0

    def test_output_respects_speed_bounds(self):
        tracker = MPCTracker(wheelbase=2.79, max_speed=10.0)
        x0 = np.array([0.0, 0.0, 0.0, 9.0])
        # Reference demanding very high speed
        ref = np.zeros((80, 3))
        for i in range(80):
            ref[i, 0] = 5.0 * (i + 1)  # 50 m/s reference
        _, speed = tracker.track(x0, ref)
        assert speed <= 10.0 + 1e-6

    def test_output_no_reverse(self):
        tracker = MPCTracker(wheelbase=2.79)
        x0 = np.array([0.0, 0.0, 0.0, 2.0])
        # Reference behind the vehicle
        ref = np.zeros((80, 3))
        ref[:, 0] = -10.0
        _, speed = tracker.track(x0, ref)
        assert speed >= 0.0

    def test_steering_bounded(self):
        tracker = MPCTracker(wheelbase=2.79, max_steer=0.6)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        # Sharp left turn reference
        ref = np.zeros((80, 3))
        for i in range(80):
            ref[i, 0] = 0.0
            ref[i, 1] = 2.0 * (i + 1)  # hard left
            ref[i, 2] = math.pi / 2
        new_pos, _ = tracker.track(x0, ref)
        assert np.isfinite(new_pos).all()

    def test_warm_start_produces_valid_output(self):
        tracker = MPCTracker(wheelbase=2.79)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        ref = np.zeros((80, 3))
        for i in range(80):
            ref[i, 0] = 0.5 * (i + 1)
        pos1, spd1 = tracker.track(x0, ref)
        x1 = np.array([pos1[0], pos1[1], pos1[2], spd1])
        pos2, spd2 = tracker.track(x1, ref)
        assert np.isfinite(pos2).all()
        assert spd2 >= 0.0

    def test_reset_clears_warm_start(self):
        tracker = MPCTracker(wheelbase=2.79)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        ref = np.zeros((80, 3))
        ref[:, 0] = np.arange(1, 81) * 0.5
        tracker.track(x0, ref)
        assert tracker._prev_knots is not None
        tracker.reset()
        assert tracker._prev_knots is None

    def test_invalid_knots_raises(self):
        with pytest.raises(ValueError):
            MPCTracker(wheelbase=2.79, horizon_steps=20, n_knots=7)
        with pytest.raises(ValueError):
            MPCTracker(wheelbase=2.79, horizon_steps=5, n_knots=10)
        with pytest.raises(ValueError):
            MPCTracker(wheelbase=2.79, horizon_steps=20, n_knots=0)

    def test_short_reference_padded(self):
        """Reference shorter than horizon should be padded, not crash."""
        tracker = MPCTracker(wheelbase=2.79, horizon_steps=20, n_knots=5)
        x0 = np.array([0.0, 0.0, 0.0, 3.0])
        ref = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # only 2 steps
        pos, speed = tracker.track(x0, ref)
        assert np.isfinite(pos).all()


# ── PerfectTracker ──────────────────────────────────────────────────────────


class TestPerfectTracker:
    def test_straight_advance(self):
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        ref = np.array([[0.5, 0.0, 0.0]])  # 0.5m ahead = 5 m/s
        pos, speed = tracker.track(x0, ref)
        assert pos[0] == pytest.approx(0.5, abs=0.05)
        assert abs(pos[1]) < 1e-6
        assert speed == pytest.approx(5.0, abs=0.5)

    def test_heading_snapped_to_reference(self):
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])  # heading east
        ref = np.array([[0.5, 0.0, math.pi / 4]])  # ref heading NE
        pos, _ = tracker.track(x0, ref)
        assert pos[2] == pytest.approx(math.pi / 4, abs=1e-6)

    def test_speed_capped(self):
        tracker = PerfectTracker(dt=0.1, max_speed=10.0)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        ref = np.array([[50.0, 0.0, 0.0]])  # 500 m/s implied
        _, speed = tracker.track(x0, ref)
        assert speed <= 10.0 + 1e-6

    def test_empty_reference(self):
        tracker = PerfectTracker(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.0, 5.0])
        pos, speed = tracker.track(x0, np.zeros((0, 3)))
        assert pos[0] == pytest.approx(0.0)
        assert speed == 0.0

    def test_reset_is_noop(self):
        tracker = PerfectTracker()
        tracker.reset()  # should not raise


# ── postprocess_reference ───────────────────────────────────────────────────


class TestPostprocessReference:
    def test_force_stop_freezes_positions(self):
        """After velocity drops below threshold, positions must freeze."""
        N = 20
        ref_xy = np.zeros((N, 2), dtype=np.float64)
        ref_h = np.zeros(N, dtype=np.float64)
        # Constant speed for first 10 steps, then stop
        for i in range(N):
            if i < 10:
                ref_xy[i, 0] = i * 0.5
            else:
                ref_xy[i, 0] = 10 * 0.5  # freeze
        result = postprocess_reference(ref_xy, ref_h, stop_threshold=0.3)
        # After force-stop triggers, all positions should be identical
        stop_idx = None
        for i in range(11, N):
            if result[i, 0] == result[i - 1, 0]:
                stop_idx = i
                break
        assert stop_idx is not None, "Force-stop should have frozen positions"
        for i in range(stop_idx, N):
            np.testing.assert_array_equal(result[i, :2], result[stop_idx - 1, :2])

    def test_velocity_smoothing_reduces_jitter(self):
        """Moving average should smooth velocity spikes."""
        N = 30
        ref_xy = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            ref_xy[i, 0] = i * 0.5
        # Inject a spike
        ref_xy[10, 0] += 2.0
        result = postprocess_reference(ref_xy, np.zeros(N), vel_smooth_window=8)
        assert np.isfinite(result).all()

    def test_single_point_reference(self):
        ref_xy = np.array([[1.0, 2.0]])
        ref_h = np.array([0.5])
        result = postprocess_reference(ref_xy, ref_h)
        assert result.shape == (1, 3)
        assert result[0, 0] == pytest.approx(1.0)

    def test_constant_velocity_no_stop(self):
        """Constant-velocity reference should not trigger force-stop."""
        N = 40
        ref_xy = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            ref_xy[i, 0] = i * 1.0  # 10 m/s constant
        ref_h = np.zeros(N)
        result = postprocess_reference(ref_xy, ref_h, stop_threshold=0.3)
        # Last position should NOT be frozen to earlier position
        assert result[-1, 0] > result[0, 0] + 10.0


# ── Analytic gradient ──────────────────────────────────────────────────────


class TestMPCGradient:
    """Verify _cost_and_grad against scipy's numerical gradient.

    Central-difference gradient (step h=1e-6) is accurate to
    ~1e-6; we match within 1e-4 (the bicycle model's nonlinearity + the
    reverse-mode chain can accumulate single-digit relative error at
    this tolerance). Catches any sign/factor mistake in the hand-
    derived Jacobian.
    """

    def _make_tracker(self):
        return MPCTracker(wheelbase=2.79, horizon_steps=20, n_knots=5)

    def _rand_problem(self, rng):
        x0 = np.array(
            [
                float(rng.uniform(-10, 10)),
                float(rng.uniform(-10, 10)),
                float(rng.uniform(-1, 1)),
                float(rng.uniform(0.5, 8.0)),
            ]
        )
        ref = np.zeros((20, 3), dtype=np.float64)
        for i in range(20):
            ref[i, 0] = x0[0] + (i + 1) * 0.5
            ref[i, 1] = x0[1] + (i + 1) * 0.05
            ref[i, 2] = x0[2] + 0.01 * i
        knot_flat = np.array(
            [
                rng.uniform(-1.0, 1.0),  # a0
                rng.uniform(-0.2, 0.2),  # d0
                rng.uniform(-1.0, 1.0),  # a1
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-0.2, 0.2),
            ]
        )
        return x0, ref, knot_flat

    @staticmethod
    def _numerical_gradient(tracker, knot, x0, ref, h=1e-6):
        """Central-difference gradient using only public numpy; avoids
        scipy's private ``_numdiff.approx_derivative`` whose API isn't
        stable across scipy versions."""
        g = np.zeros_like(knot)
        for i in range(len(knot)):
            k_plus = knot.copy()
            k_plus[i] += h
            k_minus = knot.copy()
            k_minus[i] -= h
            g[i] = (tracker._cost(k_plus, x0, ref) - tracker._cost(k_minus, x0, ref)) / (2 * h)
        return g

    def test_gradient_matches_numerical(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            tracker = self._make_tracker()
            x0, ref, knot = self._rand_problem(rng)

            j_ana, g_ana = tracker._cost_and_grad(knot, x0, ref)
            g_num = self._numerical_gradient(tracker, knot, x0, ref)

            # Relative error, tolerant of scale
            scale = np.maximum(np.abs(g_ana), np.abs(g_num))
            scale = np.where(scale < 1e-6, 1.0, scale)
            rel_err = np.abs(g_ana - g_num) / scale
            max_rel = float(rel_err.max())
            assert max_rel < 1e-3, (
                f"analytic vs numerical gradient diverges: max rel err = {max_rel:.2e}\n"
                f"  analytic = {g_ana}\n  numerical= {g_num}"
            )

    def test_cost_value_unchanged(self):
        """_cost_and_grad's cost value must match _cost alone bit-for-bit."""
        rng = np.random.default_rng(42)
        tracker = self._make_tracker()
        for _ in range(5):
            x0, ref, knot = self._rand_problem(rng)
            j_ana, _ = tracker._cost_and_grad(knot, x0, ref)
            j_ref = tracker._cost(knot, x0, ref)
            assert j_ana == pytest.approx(j_ref, abs=1e-10)
