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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        # Pad or truncate reference to exactly *horizon* steps
        ref = np.zeros((self.horizon, 3), dtype=np.float64)
        n = min(len(ref_world), self.horizon)
        ref[:n] = ref_world[:n, :3]
        if n < self.horizon:
            ref[n:] = ref[n - 1]

        # Initial guess — warm-start from shifted previous solution
        if self._prev_knots is not None:
            init = np.roll(self._prev_knots, -1, axis=0).copy()
            init[-1] = init[-2]
        else:
            init = np.zeros((self.n_knots, 2), dtype=np.float64)

        result = minimize(
            self._cost,
            init.ravel(),
            args=(np.asarray(x0, dtype=np.float64), ref),
            method="L-BFGS-B",
            bounds=self._bounds,
            options={"maxiter": 50, "ftol": 1e-6},
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
# Perfect tracker (simple Euler integration, no optimisation)
# ----------------------------------------------------------------------

class PerfectTracker:
    """Velocity-limited Euler trajectory follower.

    Reads the target velocity from the reference trajectory's position
    differences, integrates position via Euler step, and snaps heading
    to the reference.  No optimisation, no feedback control — pure
    open-loop trajectory following with physics-limited steps.

    Much faster than :class:`MPCTracker` (~0.01 ms vs ~13 ms per call)
    but has no lookahead or kinematic steering model.
    """

    def __init__(self, dt: float = 0.1, max_speed: float = 20.0):
        self.dt = dt
        self.max_speed = max_speed

    def track(
        self,
        x0: np.ndarray,
        ref_world: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Advance one step along the reference trajectory.

        Args:
            x0: (4,) [x, y, yaw, v] current state in world frame.
            ref_world: (N, 3) reference [x, y, yaw] in world frame.

        Returns:
            new_pos: (3,) [x, y, yaw] after one dt step.
            new_speed: scalar speed after one dt step.
        """
        if len(ref_world) < 1:
            return np.array([x0[0], x0[1], x0[2]], dtype=np.float32), 0.0

        # Target velocity from reference step 0 displacement
        dx = ref_world[0, 0] - x0[0]
        dy = ref_world[0, 1] - x0[1]
        v_target = min(math.hypot(dx, dy) / self.dt, self.max_speed)

        # Euler integration using current heading and target velocity
        x, y, yaw = float(x0[0]), float(x0[1]), float(x0[2])
        x_new = x + v_target * math.cos(yaw) * self.dt
        y_new = y + v_target * math.sin(yaw) * self.dt

        # Snap heading directly to the reference orientation
        yaw_new = float(ref_world[0, 2])

        new_pos = np.array([x_new, y_new, yaw_new], dtype=np.float32)
        return new_pos, v_target

    def reset(self):
        """No-op (no internal state to clear)."""
        pass
