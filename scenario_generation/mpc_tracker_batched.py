"""Batched MPC solve across independent rollout segments.

The closed-loop reproducer advances B segments per tick; with
``tracker_mode="mpc"`` that is B *serial* ``scipy.optimize.minimize`` calls,
each evaluating a cost/adjoint written as Python loops over the 20-step
horizon — the single largest CPU sink of the mining rollout (44% of elapsed
on the perf-audit workload).

``track_many`` solves all B problems in ONE L-BFGS-B call: the problems are
independent, so the sum of their costs decouples and its gradient is the
concatenation of the per-problem gradients. Every cost/gradient evaluation is
vectorized over B (the H-step forward rollout and reverse adjoint sweep run as
(B,)-shaped numpy ops), replacing B × (Python-loop evaluations + solver
overhead) with one vectorized solve.

Note on device choice: at B≈16 segments × 10 decision variables the
arithmetic is far too small for a GPU to win (kernel-launch overhead would
dominate); vectorized CPU numpy is the right execution model here. The
speedup comes from batching out the Python/solver overhead, not from CUDA.

Semantics: the per-segment reference building (idle recovery), warm start,
bounds, first-control application and telemetry are shared with
``MPCTracker`` (``_build_ref_and_init`` and the same application formulas).
The optimizer trajectory differs from B independent solves (L-BFGS curvature
estimates and the line search are shared across the concatenated problem), so
results are NOT bit-identical to ``tracker_mode="mpc"`` — they must stay
statistically equivalent, which ``validate_mpc_batched.py`` checks (tracking
RMSE, per-label mined events, credit-row overlap).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from scenario_generation.mpc_tracker import MPCTracker

_SHARED_KNOBS = (
    "horizon",
    "n_knots",
    "dt",
    "max_accel",
    "min_accel",
    "max_steer",
    "max_speed",
    "w_pos",
    "w_head",
    "w_accel",
    "w_steer",
    "w_jerk",
    "w_steer_rate",
)


def _batched_cost_and_grad(
    flat: np.ndarray,
    x0s: np.ndarray,
    refs: np.ndarray,
    wheelbases: np.ndarray,
    proto: MPCTracker,
) -> tuple[float, np.ndarray]:
    """Sum of per-segment MPC costs + concatenated gradient, vectorized over B.

    Same math as ``MPCTracker._cost_and_grad`` with a leading batch dimension
    (verified term-by-term against it in the unit tests); ``wheelbases`` is
    per-segment, every other knob is shared (asserted by ``track_many``).
    """
    B = x0s.shape[0]
    K, H, sk = proto.n_knots, proto.horizon, proto.steps_per_knot
    dt, v_max, wb = proto.dt, proto.max_speed, wheelbases

    knots = flat.reshape(B, K, 2)
    controls = np.repeat(knots, sk, axis=1)[:, :H]

    states = np.empty((B, H + 1, 4), dtype=np.float64)
    states[:, 0] = x0s
    clamped = np.zeros((B, H), dtype=bool)
    for t in range(H):
        x, y, yaw, v = (states[:, t, j] for j in range(4))
        a = controls[:, t, 0]
        delta = controls[:, t, 1]
        states[:, t + 1, 0] = x + v * np.cos(yaw) * dt
        states[:, t + 1, 1] = y + v * np.sin(yaw) * dt
        states[:, t + 1, 2] = yaw + v * np.tan(delta) / wb * dt
        v_new = v + a * dt
        clamped[:, t] = (v_new < 0.0) | (v_new > v_max)
        states[:, t + 1, 3] = np.clip(v_new, 0.0, v_max)

    pred = states[:, 1:]
    pos_err = pred[:, :, :2] - refs[:, :, :2]
    dyaw = pred[:, :, 2] - refs[:, :, 2]

    J = proto.w_pos * (pos_err**2).sum(axis=(1, 2))
    J += proto.w_head * (1.0 - np.cos(dyaw)).sum(axis=1)
    J += proto.w_accel * (controls[:, :, 0] ** 2).sum(axis=1)
    J += proto.w_steer * (controls[:, :, 1] ** 2).sum(axis=1)
    if K > 1:
        dk = np.diff(knots, axis=1)
        J += proto.w_jerk * (dk[:, :, 0] ** 2).sum(axis=1)
        J += proto.w_steer_rate * (dk[:, :, 1] ** 2).sum(axis=1)

    dJ_dstate_direct = np.zeros((B, H + 1, 4), dtype=np.float64)
    dJ_dstate_direct[:, 1:, 0] = 2.0 * proto.w_pos * pos_err[:, :, 0]
    dJ_dstate_direct[:, 1:, 1] = 2.0 * proto.w_pos * pos_err[:, :, 1]
    dJ_dstate_direct[:, 1:, 2] = proto.w_head * np.sin(dyaw)

    dJ_dctrl = np.zeros((B, H, 2), dtype=np.float64)
    adj = dJ_dstate_direct[:, H].copy()
    for t in range(H - 1, -1, -1):
        yaw = states[:, t, 2]
        v = states[:, t, 3]
        delta = controls[:, t, 1]
        sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
        tan_d, cos_d = np.tan(delta), np.cos(delta)
        cos_d2 = cos_d * cos_d
        ax, ay, ayaw, av = adj[:, 0], adj[:, 1], adj[:, 2], adj[:, 3]
        not_clamped = ~clamped[:, t]
        dv_dv = not_clamped.astype(np.float64)
        dv_da = not_clamped * dt

        adj_yaw = (-v * sin_yaw * dt) * ax + (v * cos_yaw * dt) * ay + ayaw
        adj_v = (cos_yaw * dt) * ax + (sin_yaw * dt) * ay + (tan_d / wb * dt) * ayaw + dv_dv * av

        dJ_dctrl[:, t, 0] = dv_da * av
        dJ_dctrl[:, t, 1] = np.where(cos_d2 > 1e-12, v / (wb * cos_d2) * dt * ayaw, 0.0)
        adj = dJ_dstate_direct[:, t] + np.stack([ax, ay, adj_yaw, adj_v], axis=1)

    dJ_dctrl[:, :, 0] += 2.0 * proto.w_accel * controls[:, :, 0]
    dJ_dctrl[:, :, 1] += 2.0 * proto.w_steer * controls[:, :, 1]

    dJ_dknot = dJ_dctrl[:, : K * sk].reshape(B, K, sk, 2).sum(axis=2)
    if K > 1:
        dk0 = knots[:, 1:, 0] - knots[:, :-1, 0]
        dk1 = knots[:, 1:, 1] - knots[:, :-1, 1]
        dJ_dknot[:, :-1, 0] -= 2.0 * proto.w_jerk * dk0
        dJ_dknot[:, 1:, 0] += 2.0 * proto.w_jerk * dk0
        dJ_dknot[:, :-1, 1] -= 2.0 * proto.w_steer_rate * dk1
        dJ_dknot[:, 1:, 1] += 2.0 * proto.w_steer_rate * dk1

    return float(J.sum()), dJ_dknot.ravel()


def track_many(
    trackers: list[MPCTracker],
    x0s: np.ndarray,
    ref_worlds: list[np.ndarray],
    *,
    maxiter: int = 20,
    ftol: float = 1e-4,
    gtol: float = 1e-4,
) -> list[tuple[np.ndarray, float]]:
    """One receding-horizon MPC step for B segments in a single batched solve.

    Args:
        trackers: per-segment ``MPCTracker`` instances (hold warm start +
            telemetry; wheelbase may differ per segment, all other knobs must
            match).
        x0s: (B, 4) [x, y, yaw, v] current states, world frame.
        ref_worlds: B references (N_i, 3) [x, y, yaw], world frame.

    Returns:
        List of (new_pos (3,), new_speed) per segment — the same contract as
        ``MPCTracker.track``. Updates each tracker's ``_prev_knots`` and
        ``last_accel/last_steering/last_yaw_rate`` telemetry exactly like the
        serial path.
    """
    if not trackers:
        return []
    proto = trackers[0]
    for trk in trackers[1:]:
        mismatched = [k for k in _SHARED_KNOBS if getattr(trk, k) != getattr(proto, k)]
        if mismatched:
            raise ValueError(
                f"track_many requires identical MPC knobs across segments; "
                f"tracker differs on {mismatched} (only wheelbase may vary)"
            )

    B = len(trackers)
    K, H = proto.n_knots, proto.horizon
    x0s = np.asarray(x0s, dtype=np.float64).reshape(B, 4)
    refs = np.empty((B, H, 3), dtype=np.float64)
    inits = np.empty((B, K, 2), dtype=np.float64)
    for i, (trk, ref_world) in enumerate(zip(trackers, ref_worlds)):
        refs[i], inits[i] = trk._build_ref_and_init(x0s[i], ref_world)
    wheelbases = np.array([trk.wheelbase for trk in trackers], dtype=np.float64)

    result = minimize(
        _batched_cost_and_grad,
        inits.ravel(),
        args=(x0s, refs, wheelbases, proto),
        method="L-BFGS-B",
        bounds=proto._bounds * B,
        jac=True,
        # Same stopping knobs as the serial per-problem solve. ftol acts on
        # the SUMMED cost here, so individual blocks can stop at slightly
        # different iterates than B independent solves — measured per-tick
        # divergence stays within the validation bounds (0.05 m / 0.01 rad),
        # and the receding horizon re-plans from the realized state every
        # tick, so per-tick solver noise does not accumulate.
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    knots = result.x.reshape(B, K, 2)

    out: list[tuple[np.ndarray, float]] = []
    for i, trk in enumerate(trackers):
        optimal = knots[i]
        trk._prev_knots = optimal.copy()
        a = float(optimal[0, 0])
        delta = float(np.clip(optimal[0, 1], -trk.max_steer, trk.max_steer))
        x, y, yaw, v = (float(x0s[i, j]) for j in range(4))
        x_new = x + v * math.cos(yaw) * trk.dt
        y_new = y + v * math.sin(yaw) * trk.dt
        yaw_new = yaw + v * math.tan(delta) / trk.wheelbase * trk.dt
        v_new = max(0.0, min(v + a * trk.dt, trk.max_speed))
        trk.last_accel = a
        trk.last_steering = delta
        trk.last_yaw_rate = v * math.tan(delta) / trk.wheelbase
        out.append((np.array([x_new, y_new, yaw_new], dtype=np.float32), v_new))
    return out
