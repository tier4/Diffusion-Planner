"""Speed of an ego placed directly on plan poses (``tracker_mode="perfect"``).

Pure-Python tests — no model, no GPU, no map needed.
"""

from types import SimpleNamespace

import numpy as np

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import DT, _advance_step, _plan_offset, _plan_override

REPLAN = 8


def _placed_state(ego_hist):
    return SimpleNamespace(
        k=0,
        warmup_steps=0,
        tracker=None,
        live_pose=ego_hist[-1, :3].copy(),
        dyn=SimpleNamespace(speed=0.0),
        ego_hist=ego_hist,
        sim_time=0.0,
        accels=np.zeros(200, dtype=np.float32),
        unstick_after=0,
        tl=SimpleNamespace(native_h5=True),
    )


def _run(s, plan_x, steps):
    """Replan every ``REPLAN`` steps: ``plan_x(x0, t0)`` gives the plan's x (80,) along +x from the
    ego at ``x0`` (time ``t0``); the ego is placed on it like ``render_segment`` does."""
    out = []
    for k in range(steps):
        if k % REPLAN == 0:
            xs = plan_x(float(s.live_pose[0]), k * DT)
            plan = (np.stack([xs, np.zeros_like(xs)], axis=-1), np.zeros_like(xs))
        override = _plan_override(plan, k % REPLAN)
        _advance_step(s, np.zeros((80, 4)), k, "cpu", Timers(), override=override)
        out.append((float(s.live_pose[0]), s.dyn.speed, s.dyn.accel))
    return out


def test_stopped_zigzag_plan_reads_near_zero_speed():
    """A stopped plan whose points zigzag a few cm back and forth (the per-index pattern a stopped
    cached plan showed in closed loop) must give a near-zero speed and accel — not the
    0.3-1.0 m/s one-step hop that fed back into the model's ego history and made it creep."""
    hops = np.array([0.09, -0.076, 0.077, -0.017, 0.089, -0.065, 0.048, -0.146])  # sums to 0
    s = _placed_state(np.zeros((31, 5), dtype=np.float64))
    out = _run(s, lambda x0, t0: x0 + np.cumsum(np.resize(hops, 80)), 40)
    assert max(abs(h) for h in hops) / DT > 0.5  # the one-step hop alone reads a moving ego
    # From k=4 the window's realized half is on the zigzag too (before, it spans the flat start).
    for k, (_, v, a) in enumerate(out[4:], start=4):
        assert v < 0.05, (k, v)
        if k >= 5:
            assert abs(a) < 0.5, (k, a)
    assert s.ego_hist[-1, 3] == s.dyn.speed  # the speed column the model reads


def test_braking_speed_has_no_lag():
    """Braking at a constant 2 m/s^2 the placed ego reports its actual speed: the window is
    centered on it, so it does not read the speed of 0.4 s ago (which braked late)."""
    a = -2.0
    v0 = 8.0
    hist = np.zeros((31, 5), dtype=np.float64)
    t_hist = np.arange(-30, 1) * DT
    hist[:, 0] = v0 * t_hist + 0.5 * a * t_hist**2  # same braking before the start
    s = _placed_state(hist)
    s.dyn = SimpleNamespace(speed=v0)

    def plan_x(x0, t0):
        t = np.arange(1, 81) * DT
        return x0 + (v0 + a * t0) * t + 0.5 * a * t**2  # stays braking (stop not reached)

    for k, (_, v, _acc) in enumerate(_run(s, plan_x, 24)):
        np.testing.assert_allclose(v, v0 + a * (k + 1) * DT, atol=1e-6)


def test_backward_motion_reads_zero():
    s = _placed_state(np.zeros((31, 5), dtype=np.float64))
    out = _run(s, lambda x0, t0: x0 - 0.1 * np.arange(1, 81), 8)
    assert all(v == 0.0 for _, v, _ in out)


def test_plan_offset_counts_from_the_plan():
    # On the replan grid: the same steps as k % REPLAN.
    assert [_plan_offset(k, 8, REPLAN) for k in range(8, 17)] == [*range(8), None]
    # A plan made off the grid (after an unstick teleport at step 12) starts from its first pose
    # and is kept for a full interval.
    assert [_plan_offset(k, 13, REPLAN) for k in range(13, 22)] == [*range(8), None]
