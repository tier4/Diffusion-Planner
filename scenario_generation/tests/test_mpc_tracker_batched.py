"""Unit tests for the batched MPC solver against the serial reference."""

import numpy as np
import pytest

from scenario_generation.mpc_tracker import MPCTracker
from scenario_generation.mpc_tracker_batched import _batched_cost_and_grad, track_many


def _random_problem(rng, moving=True):
    x0 = np.array(
        [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi), rng.uniform(0, 8)],
        dtype=np.float64,
    )
    # A plausible forward reference: roughly constant-speed arc from x0.
    n = 80
    speed = rng.uniform(1.0, 8.0) if moving else 0.0
    yaw_rate = rng.uniform(-0.2, 0.2)
    yaws = x0[2] + yaw_rate * 0.1 * np.arange(1, n + 1)
    steps = speed * 0.1
    xs = x0[0] + np.cumsum(steps * np.cos(yaws))
    ys = x0[1] + np.cumsum(steps * np.sin(yaws))
    ref_world = np.column_stack([xs, ys, yaws])
    return x0, ref_world


def test_batched_cost_and_grad_matches_scalar():
    """The vectorized cost/adjoint must equal the scalar reference per block."""
    rng = np.random.default_rng(0)
    B = 6
    trackers = [MPCTracker(wheelbase=rng.uniform(2.5, 5.0)) for _ in range(B)]
    proto = trackers[0]
    x0s = np.stack([_random_problem(rng)[0] for _ in range(B)])
    refs = np.stack(
        [proto._build_ref_and_init(x0s[i], _random_problem(rng)[1])[0] for i in range(B)]
    )
    knots = rng.uniform(-1, 1, size=(B, proto.n_knots, 2))
    wheelbases = np.array([t.wheelbase for t in trackers])

    # Batched cost must equal the sum of scalar costs; the gradient must equal
    # the concatenation of scalar gradients (each scalar solve uses its own wb).
    J_batch, g_batch = _batched_cost_and_grad(knots.ravel(), x0s, refs, wheelbases, proto)
    J_scalar, g_scalar = 0.0, []
    for i, trk in enumerate(trackers):
        J_i, g_i = trk._cost_and_grad(knots[i].ravel(), x0s[i], refs[i])
        J_scalar += J_i
        g_scalar.append(g_i)
    np.testing.assert_allclose(J_batch, J_scalar, rtol=1e-12)
    np.testing.assert_allclose(g_batch, np.concatenate(g_scalar), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("warm", [False, True])
def test_track_many_close_to_serial_track(warm):
    """One batched receding-horizon step lands within tight tolerance of the
    serial solver (not bit-identical: L-BFGS memory/line search are shared)."""
    rng = np.random.default_rng(1)
    B = 8
    problems = [_random_problem(rng) for _ in range(B)]
    serial = [MPCTracker(wheelbase=4.76) for _ in range(B)]
    batched = [MPCTracker(wheelbase=4.76) for _ in range(B)]
    if warm:  # exercise the warm-start roll path too
        for s_trk, b_trk in zip(serial, batched):
            knots = rng.uniform(-0.5, 0.5, size=(s_trk.n_knots, 2))
            s_trk._prev_knots = knots.copy()
            b_trk._prev_knots = knots.copy()

    serial_out = [trk.track(x0, ref) for trk, (x0, ref) in zip(serial, problems)]
    batched_out = track_many(batched, np.stack([p[0] for p in problems]), [p[1] for p in problems])

    for (pos_s, v_s), (pos_b, v_b) in zip(serial_out, batched_out):
        # One dt step from identical x0: divergence only via the optimized
        # first control. a/delta enter position at O(dt^2) — poses must be
        # extremely close even when the solvers stop at different iterates.
        # Both solvers stop at ftol=1e-4; the batched ftol acts on the summed
        # cost, so iterates differ slightly. Bounds = the M2 validation
        # protocol's per-tick limits (0.05 m / 0.01 rad).
        np.testing.assert_allclose(pos_b[:2], pos_s[:2], atol=0.05)
        np.testing.assert_allclose(pos_b[2], pos_s[2], atol=0.01)
        # Speed applies a*dt with a in [-4, 3]; a 0.25 m/s iterate gap moves
        # next-tick position by <0.025 m, inside the pose bound above.
        assert abs(v_b - v_s) < 0.25


def test_track_many_updates_warm_start_and_telemetry():
    rng = np.random.default_rng(2)
    trackers = [MPCTracker(wheelbase=4.76) for _ in range(3)]
    problems = [_random_problem(rng) for _ in range(3)]
    out = track_many(trackers, np.stack([p[0] for p in problems]), [p[1] for p in problems])
    assert len(out) == 3
    for trk in trackers:
        assert trk._prev_knots is not None and trk._prev_knots.shape == (trk.n_knots, 2)
        assert np.isfinite(trk.last_accel) and np.isfinite(trk.last_yaw_rate)


def test_track_many_rejects_mismatched_knobs():
    a = MPCTracker(wheelbase=4.76)
    b = MPCTracker(wheelbase=4.76, horizon_steps=10, n_knots=5)
    with pytest.raises(ValueError, match="identical MPC knobs"):
        track_many([a, b], np.zeros((2, 4)), [np.zeros((80, 3))] * 2)


def test_track_many_empty():
    assert track_many([], np.zeros((0, 4)), []) == []


def test_advance_step_tracked_fast_path():
    """_advance_step with a precomputed ``tracked`` result must apply it verbatim
    (no tracker.track call) and read telemetry off the tracker like the serial
    branch does — this is the mpc_batched integration seam in the rollout."""
    from types import SimpleNamespace

    from scenario_generation.perf_timer import Timers
    from scenario_generation.reproducer_rollout import DT, _advance_step

    class _MustNotTrack:
        last_yaw_rate = 0.123
        last_steering = -0.05

        def track(self, x0, ref):
            raise AssertionError("tracked= fast path must not invoke tracker.track")

    s = SimpleNamespace(
        k=3,
        warmup_steps=0,
        tracker=_MustNotTrack(),
        live_pose=np.array([1.0, 2.0, 0.1], dtype=np.float64),
        dyn=SimpleNamespace(speed=4.0),
        ego_hist=np.zeros((31, 3), dtype=np.float64),
        sim_time=0.3,
        accels=np.zeros(100, dtype=np.float32),
        unstick_after=0,
    )
    new_pose = np.array([1.4, 2.05, 0.12], dtype=np.float64)
    _advance_step(s, np.zeros((80, 4)), 3, "cpu", Timers(), tracked=(new_pose, 4.5))

    np.testing.assert_array_equal(s.live_pose, new_pose)
    assert s.k == 4
    assert s.dyn.speed == 4.5
    np.testing.assert_allclose(s.dyn.accel, (4.5 - 4.0) / DT)
    assert s.dyn.yaw_rate == 0.123  # read from tracker telemetry, like the serial branch
    assert s.dyn.steering == -0.05
    np.testing.assert_array_equal(s.ego_hist[-1], new_pose)
    np.testing.assert_allclose(s.accels[3], (4.5 - 4.0) / DT)
