"""Simple MPC trajectory tracker using a bicycle kinematic model.

Replaces the teleport-based advance in closed-loop replay with a
physically plausible trajectory tracking controller. The MPC optimizes
acceleration and steering angle over a 2-second horizon to follow the
diffusion model's predicted trajectory through a kinematic bicycle
model, enforcing physical constraints that prevent aggressive behavior
(lane invasion, red-light running, unphysical heading jumps).

Bicycle kinematic model:
    dx/dt   = v * cos(yaw)
    dy/dt   = v * sin(yaw)
    dyaw/dt = v * tan(delta) / wheelbase
    dv/dt   = a

Velocity smoothing and force-stop logic ported from
``autoware_diffusion_planner/src/postprocessing/postprocessing_utils.cpp``
(lines 249-342).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


class MPCTracker:
    """MPC trajectory tracker with bicycle kinematic model.

    Optimises a short control sequence (acceleration + steering) via
    scipy L-BFGS-B to track a reference trajectory produced by the
    diffusion model.  Only the first control step is applied (receding
    horizon), and the previous solution is warm-started into the next
    call.

    Parameters
    ----------
    wheelbase : float
        Distance between front and rear axles (metres).
    horizon_steps : int
        Number of 0.1 s steps in the MPC horizon (default 20 = 2 s).
    n_knots : int
        Number of control decision points.  Controls are held constant
        between knots (piecewise-constant parameterisation).  Fewer
        knots = fewer decision variables = faster solve.
    dt : float
        Simulation timestep (seconds).
    """

    def __init__(
        self,
        wheelbase: float,
        horizon_steps: int = 20,
        n_knots: int = 5,
        dt: float = 0.1,
        # Physical bounds
        max_accel: float = 3.0,
        min_accel: float = -4.0,
        max_steer: float = 0.6,
        max_speed: float = 20.0,
        # Cost weights
        w_pos: float = 10.0,
        w_head: float = 5.0,
        w_accel: float = 0.1,
        w_steer: float = 0.5,
        w_jerk: float = 1.0,
        w_steer_rate: float = 2.0,
    ):
        self.wheelbase = max(wheelbase, 0.5)  # safety floor
        self.horizon = horizon_steps
        self.n_knots = n_knots
        self.dt = dt
        if n_knots < 1 or horizon_steps < n_knots or horizon_steps % n_knots != 0:
            raise ValueError(
                f"horizon_steps ({horizon_steps}) must be a positive "
                f"multiple of n_knots ({n_knots})"
            )
        self.steps_per_knot = horizon_steps // n_knots

        self.max_accel = max_accel
        self.min_accel = min_accel
        self.max_steer = max_steer
        self.max_speed = max_speed

        self.w_pos = w_pos
        self.w_head = w_head
        self.w_accel = w_accel
        self.w_steer = w_steer
        self.w_jerk = w_jerk
        self.w_steer_rate = w_steer_rate

        # Bounds for the optimiser (interleaved [a0, d0, a1, d1, ...])
        self._bounds = []
        for _ in range(n_knots):
            self._bounds.append((min_accel, max_accel))
            self._bounds.append((-max_steer, max_steer))

        # Warm-start buffer (n_knots, 2)
        self._prev_knots: np.ndarray | None = None

        # Last-step telemetry — populated by track(). Callers that want
        # physically-correct accel / yaw_rate / steering (instead of the
        # 5-step MA that _advance_agent derives from position history)
        # read these after the track() call.
        self.last_accel: float = 0.0
        self.last_yaw_rate: float = 0.0
        self.last_steering: float = 0.0

    # ------------------------------------------------------------------
    # Kinematic rollout
    # ------------------------------------------------------------------

    def _expand_knots(self, knots: np.ndarray) -> np.ndarray:
        """(n_knots, 2) -> (horizon, 2) via piecewise-constant hold."""
        return np.repeat(knots, self.steps_per_knot, axis=0)[: self.horizon]

    def _rollout(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Forward-integrate bicycle model for *horizon* steps.

        Args:
            x0: (4,) [x, y, yaw, v] initial state.
            controls: (horizon, 2) [acceleration, steering] per step.

        Returns:
            (horizon+1, 4) state trajectory including x0.
        """
        H = self.horizon
        states = np.empty((H + 1, 4), dtype=np.float64)
        states[0] = x0
        dt = self.dt
        wb = self.wheelbase
        v_max = self.max_speed

        for t in range(H):
            x, y, yaw, v = states[t]
            a = controls[t, 0]
            delta = controls[t, 1]
            # Euler integration
            states[t + 1, 0] = x + v * math.cos(yaw) * dt
            states[t + 1, 1] = y + v * math.sin(yaw) * dt
            states[t + 1, 2] = yaw + v * math.tan(delta) / wb * dt
            states[t + 1, 3] = max(0.0, min(v + a * dt, v_max))

        return states

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _cost(
        self,
        knot_flat: np.ndarray,
        x0: np.ndarray,
        ref: np.ndarray,
    ) -> float:
        """Scalar cost evaluated by L-BFGS-B.

        Args:
            knot_flat: (2*n_knots,) decision variables.
            x0: (4,) current state.
            ref: (horizon, 3) reference [x, y, yaw] world frame.
        """
        knots = knot_flat.reshape(self.n_knots, 2)
        controls = self._expand_knots(knots)
        states = self._rollout(x0, controls)

        # Predicted states at t=1..H aligned with ref at t=0..H-1
        pred = states[1:]

        # Position tracking  ||pos - ref||^2
        pos_err = pred[:, :2] - ref[:, :2]
        J = self.w_pos * np.dot(pos_err.ravel(), pos_err.ravel())

        # Heading tracking  (1 - cos(Δyaw))  — smooth, no wrapping issues
        dyaw = pred[:, 2] - ref[:, 2]
        J += self.w_head * np.sum(1.0 - np.cos(dyaw))

        # Control effort
        J += self.w_accel * np.dot(controls[:, 0], controls[:, 0])
        J += self.w_steer * np.dot(controls[:, 1], controls[:, 1])

        # Smoothness between consecutive knots
        if self.n_knots > 1:
            dk = np.diff(knots, axis=0)
            J += self.w_jerk * np.dot(dk[:, 0], dk[:, 0])
            J += self.w_steer_rate * np.dot(dk[:, 1], dk[:, 1])

        return float(J)

    def _cost_and_grad(
        self,
        knot_flat: np.ndarray,
        x0: np.ndarray,
        ref: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Scalar cost + analytic gradient w.r.t. knot_flat.

        Reverse-mode through the bicycle rollout. The previous
        ``_cost``-only path left scipy to compute a numerical Jacobian,
        which on the profile was ~47 ``_cost`` evaluations per L-BFGS-B
        step (one cold eval + one per control dim via forward
        perturbation). With the exact gradient returned alongside the
        cost the optimiser needs ~3-5 calls per step instead.

        Verified against scipy's numerical gradient in the unit tests —
        see ``test_mpc_tracker.py::TestMPCGradient``.
        """
        H = self.horizon
        dt = self.dt
        wb = self.wheelbase
        v_max = self.max_speed
        sk = self.steps_per_knot

        knots = knot_flat.reshape(self.n_knots, 2)
        controls = self._expand_knots(knots)

        # ---- Forward rollout (same as _rollout, inlined so we can also
        # cache per-step trig values used later in the backward sweep) ----
        states = np.empty((H + 1, 4), dtype=np.float64)
        states[0] = x0
        clamped = np.zeros(H, dtype=bool)
        for t in range(H):
            x, y, yaw, v = states[t]
            a = controls[t, 0]
            delta = controls[t, 1]
            states[t + 1, 0] = x + v * math.cos(yaw) * dt
            states[t + 1, 1] = y + v * math.sin(yaw) * dt
            states[t + 1, 2] = yaw + v * math.tan(delta) / wb * dt
            v_new = v + a * dt
            if v_new < 0.0:
                clamped[t] = True
                states[t + 1, 3] = 0.0
            elif v_new > v_max:
                clamped[t] = True
                states[t + 1, 3] = v_max
            else:
                states[t + 1, 3] = v_new

        pred = states[1:]
        pos_err = pred[:, :2] - ref[:, :2]
        dyaw = pred[:, 2] - ref[:, 2]

        # Forward cost
        J = self.w_pos * np.dot(pos_err.ravel(), pos_err.ravel())
        J += self.w_head * np.sum(1.0 - np.cos(dyaw))
        J += self.w_accel * np.dot(controls[:, 0], controls[:, 0])
        J += self.w_steer * np.dot(controls[:, 1], controls[:, 1])
        if self.n_knots > 1:
            dk = np.diff(knots, axis=0)
            J += self.w_jerk * np.dot(dk[:, 0], dk[:, 0])
            J += self.w_steer_rate * np.dot(dk[:, 1], dk[:, 1])

        # ---- Reverse sweep ----
        # Direct contributions of the tracking cost to dJ/dstate[t] for t=1..H.
        # (State[0] = x0 is a constant; tracking cost doesn't touch it.)
        dJ_dstate_direct = np.zeros((H + 1, 4), dtype=np.float64)
        dJ_dstate_direct[1:, 0] = 2.0 * self.w_pos * pos_err[:, 0]
        dJ_dstate_direct[1:, 1] = 2.0 * self.w_pos * pos_err[:, 1]
        dJ_dstate_direct[1:, 2] = self.w_head * np.sin(dyaw)

        dJ_dctrl = np.zeros((H, 2), dtype=np.float64)
        adj = dJ_dstate_direct[H].copy()  # adjoint at state[H]
        for t in range(H - 1, -1, -1):
            x, y, yaw, v = states[t]
            a = controls[t, 0]
            delta = controls[t, 1]
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            tan_d = math.tan(delta)
            cos_d = math.cos(delta)
            cos_d2 = cos_d * cos_d

            ax, ay, ayaw, av = adj
            dv_dv = 0.0 if clamped[t] else 1.0
            dv_da = 0.0 if clamped[t] else dt

            # df/dstate[t]^T @ adj
            adj_x = ax
            adj_y = ay
            adj_yaw = (-v * sin_yaw * dt) * ax + (v * cos_yaw * dt) * ay + ayaw
            adj_v = (
                (cos_yaw * dt) * ax + (sin_yaw * dt) * ay + (tan_d / wb * dt) * ayaw + dv_dv * av
            )

            # df/dctrl[t]^T @ adj — only a affects v, only delta affects yaw.
            dJ_dctrl[t, 0] = dv_da * av
            if cos_d2 > 1e-12:
                dJ_dctrl[t, 1] = (v / (wb * cos_d2) * dt) * ayaw
            # else leave 0 (delta at ±π/2 is excluded by bounds anyway).

            adj = dJ_dstate_direct[t] + np.array([adj_x, adj_y, adj_yaw, adj_v])

        # Direct control-cost contribution:
        dJ_dctrl[:, 0] += 2.0 * self.w_accel * controls[:, 0]
        dJ_dctrl[:, 1] += 2.0 * self.w_steer * controls[:, 1]

        # Aggregate per-step control grads into per-knot grads. Each knot
        # is held for ``sk`` steps of the rollout (_expand_knots uses
        # ``np.repeat``), so dJ/dknot[i] is the sum of dJ/dctrl over that
        # knot's span.
        dJ_dknot = dJ_dctrl[: self.n_knots * sk].reshape(self.n_knots, sk, 2).sum(axis=1)

        # Smoothness terms (closed-form on knot diffs).
        if self.n_knots > 1:
            dk0 = knots[1:, 0] - knots[:-1, 0]
            dk1 = knots[1:, 1] - knots[:-1, 1]
            dJ_dknot[:-1, 0] -= 2.0 * self.w_jerk * dk0
            dJ_dknot[1:, 0] += 2.0 * self.w_jerk * dk0
            dJ_dknot[:-1, 1] -= 2.0 * self.w_steer_rate * dk1
            dJ_dknot[1:, 1] += 2.0 * self.w_steer_rate * dk1

        return float(J), dJ_dknot.ravel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_ref_and_init(
        self,
        x0: np.ndarray,
        ref_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the MPC-horizon reference + warm-start knots for one solve.

        Shared verbatim by ``track()`` and the batched solver
        (``mpc_tracker_batched.track_many``), so the idle-recovery /
        warm-start semantics can never drift between the two paths.

        Returns (ref (horizon, 3), init (n_knots, 2)).
        """
        # Idle-recovery check uses the model's FULL prediction horizon,
        # not the MPC's truncated window. Post-stop the model typically
        # emits a back-loaded plan: near-zero motion in the first ~2 s
        # (gentle start-from-rest) then real acceleration through 8 s.
        # MPC's 2 s window would only see the flat start and command
        # zero accel, leaving the ego parked indefinitely even though
        # the planner has committed to 20+ m of forward motion over the
        # full 8 s.
        cur_speed = float(x0[3]) if len(x0) > 3 else 0.0
        full_n = len(ref_world)
        full_tail_reach = (
            float(np.hypot(ref_world[-1, 0] - x0[0], ref_world[-1, 1] - x0[1]))
            if full_n > 0
            else 0.0
        )
        full_horizon_time = full_n * self.dt
        avg_plan_speed = full_tail_reach / full_horizon_time if full_horizon_time > 0 else 0.0
        # Stay in recovery mode until the ego reaches a reasonable fraction
        # of the plan's average speed — without this the push fires for one
        # step (ego bumps from 0 → 0.3 m/s), then next step cur_speed > 0.1
        # drops recovery off, MPC sees the flat near-horizon ref again,
        # commands -accel, ego decelerates back to 0. Hysteresis via
        # half-target gate keeps the launch continuous.
        idle_but_plan_moves = cur_speed < max(0.1, 0.5 * avg_plan_speed) and avg_plan_speed > 0.5

        # Build the reference fed to MPC. Two modes:
        # - normal: first `horizon` steps of the model's plan (truncate/pad).
        # - idle-recovery: time-compress the full plan into the horizon
        #   so MPC sees where the model wants the ego to be in 2 s if
        #   it were following the plan's average pace. Once the ego
        #   starts moving (cur_speed >= 0.1), we drop back to the normal
        #   truncated view so the model's intended timing is respected.
        ref = np.zeros((self.horizon, 3), dtype=np.float64)
        if idle_but_plan_moves and full_n > self.horizon:
            stride = full_n / self.horizon
            indices = np.clip(
                (np.arange(self.horizon) * stride).astype(int),
                0,
                full_n - 1,
            )
            ref[:] = ref_world[indices, :3]
        else:
            n = min(full_n, self.horizon)
            ref[:n] = ref_world[:n, :3]
            if n < self.horizon:
                ref[n:] = ref[n - 1]

        # Keep the old flag name for the warm-start branch below — its
        # semantics are the same: "ego idle AND we want motion".
        idle_but_ref_moves = idle_but_plan_moves

        if self._prev_knots is not None and not idle_but_ref_moves:
            # Roll: the just-applied knot[0] is consumed, knots[1..n-1]
            # become the leading knots of this solve. Old trailing knot
            # approximates the terminal behaviour reasonably well, so
            # reuse it for the freshly-opened tail slot (was: duplicate
            # knots[-2], which drops one step of useful warm-start info
            # when the old trajectory was smoothly decelerating).
            init = np.empty_like(self._prev_knots)
            init[:-1] = self._prev_knots[1:]
            init[-1] = self._prev_knots[-1]
        elif idle_but_ref_moves:
            # Seed the warm start with an accel guess derived from the
            # model's avg plan speed (not the tail-reach/horizon² formula
            # used before — that blew up on back-loaded plans because it
            # assumed the ref tail was close, when really the ref we NOW
            # give MPC is the stretched full plan with a real far tail).
            horizon_time = self.horizon * self.dt
            a_guess = avg_plan_speed / horizon_time
            a_guess = max(self.min_accel, min(self.max_accel, a_guess))
            init = np.zeros((self.n_knots, 2), dtype=np.float64)
            init[:, 0] = a_guess
        else:
            init = np.zeros((self.n_knots, 2), dtype=np.float64)
        return ref, init

    def track(
        self,
        x0: np.ndarray,
        ref_world: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Run one MPC step: optimise controls, apply first step.

        Args:
            x0: (4,) [x, y, yaw, v] current state in world frame.
            ref_world: (N, 3) reference [x, y, yaw] in world frame.
                At least *horizon* rows; extras are ignored.

        Returns:
            new_pos: (3,) [x, y, yaw] after one dt step.
            new_speed: scalar speed after one dt step.
        """
        ref, init = self._build_ref_and_init(x0, ref_world)

        # Tightened tolerance / iter caps. The smooth quadratic-ish cost
        # converges long before maxiter=50 in the typical case; profiles
        # showed ~47 cost evals per solve with the old caps, most of
        # them after the gradient had already dropped below what ftol=1e-4
        # would accept. Dropping the caps to maxiter=20 / ftol=1e-4
        # trims cost evaluations without measurable trajectory drift
        # (see A/B on 300-step TL+NPCs run — trajectory_log identical).
        result = minimize(
            self._cost_and_grad,
            init.ravel(),
            args=(np.asarray(x0, dtype=np.float64), ref),
            method="L-BFGS-B",
            bounds=self._bounds,
            jac=True,  # analytic gradient returned alongside the cost
            options={"maxiter": 20, "ftol": 1e-4, "gtol": 1e-4},
        )
        optimal_knots = result.x.reshape(self.n_knots, 2)
        self._prev_knots = optimal_knots.copy()

        # Apply first control through the bicycle model (single step)
        a = float(optimal_knots[0, 0])
        delta = float(np.clip(optimal_knots[0, 1], -self.max_steer, self.max_steer))

        x, y, yaw, v = float(x0[0]), float(x0[1]), float(x0[2]), float(x0[3])
        x_new = x + v * math.cos(yaw) * self.dt
        y_new = y + v * math.sin(yaw) * self.dt
        yaw_new = yaw + v * math.tan(delta) / self.wheelbase * self.dt
        v_new = max(0.0, min(v + a * self.dt, self.max_speed))

        # Stash the commanded control + kinematic yaw rate so callers can
        # read the tracker's true physical state instead of MA-lagged
        # estimates from position history.
        self.last_accel = a
        self.last_steering = delta
        self.last_yaw_rate = v * math.tan(delta) / self.wheelbase

        new_pos = np.array([x_new, y_new, yaw_new], dtype=np.float32)
        return new_pos, v_new

    def reset(self):
        """Clear warm-start state (call on agent respawn)."""
        self._prev_knots = None


# ----------------------------------------------------------------------
# Reference trajectory post-processing
# ----------------------------------------------------------------------


def postprocess_reference(
    ref_xy: np.ndarray,
    ref_h: np.ndarray,
    dt: float = 0.1,
    vel_smooth_window: int = 8,
    stop_threshold: float = 0.3,
) -> np.ndarray:
    """Velocity smoothing + force-stop on a world-frame reference trajectory.

    Ports the two key post-processing passes from the C++ diffusion
    planner (``postprocessing_utils.cpp`` lines 249-342):

    1. Compute velocity from position differences, then apply a
       forward-looking moving average (window = ``vel_smooth_window``).
    2. Force-stop: once smoothed velocity drops below
       ``stop_threshold`` m/s after having been above it, freeze all
       subsequent positions and headings.  This ensures the MPC
       receives a clean stop reference instead of slow drift.

    Args:
        ref_xy: (N, 2) world-frame positions [x, y].
        ref_h: (N,) headings in radians.
        dt: timestep (seconds).
        vel_smooth_window: moving-average window for velocity.
        stop_threshold: force-stop trigger velocity (m/s).

    Returns:
        (N, 3) post-processed reference [x, y, yaw].
    """
    N = len(ref_xy)
    ref = np.column_stack([ref_xy, ref_h]).copy()
    if N < 2:
        return ref

    # Step 1: velocity from position differences
    diffs = np.diff(ref_xy, axis=0)
    velocities = np.hypot(diffs[:, 0], diffs[:, 1]) / dt
    velocities = np.concatenate([[velocities[0]], velocities])  # prepend for index alignment

    # Step 2: forward-looking moving average
    smoothed = velocities.copy()
    w = vel_smooth_window
    for i in range(N - w + 1):
        smoothed[i] = velocities[i : i + w].mean()

    # Step 3: force-stop
    force_stop = False
    for i in range(1, N):
        if not force_stop:
            if smoothed[i - 1] > stop_threshold and smoothed[i] <= stop_threshold:
                force_stop = True
        if force_stop:
            ref[i, :2] = ref[i - 1, :2]
            ref[i, 2] = ref[i - 1, 2]

    return ref


# ----------------------------------------------------------------------
# Perfect tracker
# ----------------------------------------------------------------------

# Distance along the reference over which the heading is read. Below it, point spacing at low speed
# (centimetres) makes the direction noisy, so the current heading is kept.
MIN_HEADING_DISTANCE_M = 0.5


def place_on_trajectory(
    current_pose: np.ndarray, ref_xy: np.ndarray, dt: float
) -> tuple[np.ndarray, float]:
    """Exact perfect tracking: the vehicle ends the step on the reference's first point.

    The heading is the direction the reference runs from that point, read over at least
    ``MIN_HEADING_DISTANCE_M``. It is deliberately not the model's heading output: the vehicle's
    reference point is the rear axle, which moves in the direction the vehicle points, so a vehicle
    that follows the positions exactly points along the path. When the reference is shorter than
    that distance the current heading is kept.

    Args:
        current_pose: (3,) [x, y, yaw] in world frame.
        ref_xy: (N, 2) reference positions in world frame, first point one ``dt`` ahead.
        dt: timestep (seconds).

    Returns:
        new_pose: (3,) [x, y, yaw].
        speed: distance moved this step / dt.
    """
    x, y, yaw = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
    if len(ref_xy) < 1:
        return np.array([x, y, yaw], dtype=np.float64), 0.0
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    tx, ty = float(ref_xy[0, 0]), float(ref_xy[0, 1])
    far = np.flatnonzero(np.linalg.norm(ref_xy - ref_xy[0], axis=1) >= MIN_HEADING_DISTANCE_M)
    heading = (
        math.atan2(ref_xy[far[0], 1] - ty, ref_xy[far[0], 0] - tx) if len(far) else yaw
    )
    speed = math.hypot(tx - x, ty - y) / dt
    return np.array([tx, ty, heading], dtype=np.float64), speed


class PerfectTracker:
    """Perfect trajectory tracking: every step the vehicle lands exactly on the reference's first
    point, heading along the path (see :func:`place_on_trajectory`). No dynamics, no limits.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        # Parallel to MPCTracker.last_*. No steering model: last_steering stays 0.0 and
        # last_yaw_rate is the heading change per step.
        self.last_accel: float = 0.0
        self.last_yaw_rate: float = 0.0
        self.last_steering: float = 0.0
        self._prev_speed: float = 0.0

    def track(
        self,
        x0: np.ndarray,
        ref_world: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Advance one step along the reference trajectory.

        Args:
            x0: (4,) [x, y, yaw, v] current state in world frame.
            ref_world: (N, 2+) reference [x, y, ...] in world frame; only positions are used.

        Returns:
            new_pos: (3,) [x, y, yaw] after one dt step.
            new_speed: scalar speed after one dt step.
        """
        ref_world = np.asarray(ref_world)
        ref_xy = ref_world[:, :2] if len(ref_world) else np.zeros((0, 2))
        new_pos, speed = place_on_trajectory(np.asarray(x0)[:3], ref_xy, self.dt)
        dh = (float(new_pos[2]) - float(x0[2]) + math.pi) % (2 * math.pi) - math.pi
        self.last_yaw_rate = dh / self.dt
        v_prev = float(x0[3]) if len(x0) > 3 else self._prev_speed
        self.last_accel = (speed - v_prev) / self.dt
        self._prev_speed = speed
        self.last_steering = 0.0
        return new_pos.astype(np.float32), speed

    def reset(self):
        """No-op (no internal state to clear)."""
        pass
