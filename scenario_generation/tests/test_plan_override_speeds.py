"""Tests for perfect-track / cached-plan override speed profiles."""

from __future__ import annotations

import numpy as np

from scenario_generation.reproducer_rollout import (
    DT,
    _blend_override_speed,
    _plan_override_speeds,
)


def test_plan_override_speeds_constant_cruise():
    """Uniform waypoint spacing -> constant smoothed speed ≈ spacing/dt."""
    # 8 m/s at dt=0.1 -> 0.8 m per step
    n = 20
    xy = np.stack([np.arange(n, dtype=np.float64) * 0.8, np.zeros(n)], axis=1)
    spd = _plan_override_speeds(xy, DT, vel_smooth_window=8)
    assert spd.shape == (n,)
    np.testing.assert_allclose(spd, 8.0, atol=1e-6)


def test_plan_override_speeds_empty_and_singleton():
    assert _plan_override_speeds(np.zeros((0, 2)), DT).shape == (0,)
    np.testing.assert_array_equal(_plan_override_speeds(np.zeros((1, 2)), DT), [0.0])


def test_plan_override_speeds_kills_replan_chord_brake_artifact():
    """Single-step chord speeds oscillate across a short-first / long-later plan;
    smoothed + EMA override speeds must not produce a <= -4 m/s^2 spike at the
    replan boundary (the false strong-brake pattern under replan_interval>1).

    Synthetic profile mimicking the observed sawtooth: each replan cycle's first
    segment is short (~0.40 m / 4 m/s), then segments recover to ~0.80 m / 8 m/s.
    Crossing from a previous-cycle cruise (~8 m/s) into the short first chord
    invents a hard brake; the smoothed+EMA path must not.
    """
    segs = np.array([0.40, 0.80, 0.81, 0.79, 0.80, 0.80, 0.80], dtype=np.float64)
    xy = np.zeros((len(segs) + 1, 2), dtype=np.float64)
    xy[1:, 0] = np.cumsum(segs)

    prev_speed = 8.0  # end of previous replan cycle (cruise)
    chord0 = float(segs[0] / DT)  # ||live - plan[0]||/dt on a short first hop
    assert chord0 == 4.0
    assert (chord0 - prev_speed) / DT <= -4.0, "precondition: chord path must spike"

    target = float(_plan_override_speeds(xy, DT, vel_smooth_window=8)[0])
    # Forward MA alone still sits below cruise (short hop pulls the window down).
    assert target < prev_speed
    assert (target - prev_speed) / DT <= -4.0, "precondition: MA alone still spikes"

    spd = _blend_override_speed(prev_speed, target, alpha=1.0 / 8.0)
    assert (spd - prev_speed) / DT > -4.0

    # Walking the EMA along the plan must not invent a hard brake either.
    profile = _plan_override_speeds(xy, DT, vel_smooth_window=8)
    v = prev_speed
    for t in profile:
        v_next = _blend_override_speed(v, float(t), alpha=1.0 / 8.0)
        assert (v_next - v) / DT > -4.0
        v = v_next


def test_blend_override_speed_preserves_sustained_deceleration():
    """A genuine stop plan (target 0) still accumulates hard braking over frames."""
    v = 8.0
    saw_strong = False
    for _ in range(16):
        v_next = _blend_override_speed(v, 0.0, alpha=1.0 / 8.0)
        if (v_next - v) / DT <= -4.0:
            saw_strong = True
            break
        v = v_next
    assert saw_strong


def test_plan_override_speeds_preserves_real_deceleration():
    """A genuine coast-down (steadily shortening segments) still shows up as
    negative accel after smoothing — we only kill chord noise, not real brakes."""
    speeds = np.linspace(8.0, 0.0, 11)
    segs = speeds[:-1] * DT
    xy = np.zeros((len(segs) + 1, 2), dtype=np.float64)
    xy[1:, 0] = np.cumsum(segs)
    spd = _plan_override_speeds(xy, DT, vel_smooth_window=3)
    acc = np.diff(spd) / DT
    assert acc.min() < -2.0
