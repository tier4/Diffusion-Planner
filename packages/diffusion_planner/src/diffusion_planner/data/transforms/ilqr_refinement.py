"""Kinematic-bicycle iLQR refinement after ego-pose augmentation.

The solver is the hot spot of the dataloader: it runs on every augmented frame,
so its inner loops live in module-level ``numba`` kernels instead of scalar
NumPy. The 5-state / 2-control matrices are tiny, so the products are written as
explicit loops rather than BLAS calls.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..dimensions import EGO_VELOCITY_INDEX
from .base import Frame, FrameLike
from .pose_augmentation import POSE_AUGMENTATION_APPLIED_KEY

STATE_DIM = 5
CONTROL_DIM = 2
LINE_SEARCH_ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
MIN_REGULARIZATION = 1e-8
MAX_REGULARIZATION = 1e8
INITIAL_REGULARIZATION = 1e-5


@njit(inline="always")
def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@njit(inline="always")
def _clip(value: float, low: float, high: float) -> float:
    """Clip while propagating NaN so a diverged candidate stays rejectable."""
    if math.isnan(value):
        return value
    if value < low:
        return low
    if value > high:
        return high
    return value


@njit(cache=True)
def _mat_mat(x: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
    """Return ``x @ y`` for small dense matrices."""
    rows, inner = x.shape
    columns = y.shape[1]
    out = np.zeros((rows, columns))
    for i in range(rows):
        for p in range(inner):
            scale = x[i, p]
            if scale != 0.0:
                for j in range(columns):
                    out[i, j] += scale * y[p, j]
    return out


@njit(cache=True)
def _mat_t_mat(x: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
    """Return ``x.T @ y`` for small dense matrices."""
    rows, columns = x.shape
    inner = y.shape[1]
    out = np.zeros((columns, inner))
    for i in range(rows):
        for p in range(columns):
            scale = x[i, p]
            if scale != 0.0:
                for j in range(inner):
                    out[p, j] += scale * y[i, j]
    return out


@njit(cache=True)
def _mat_t_vec(x: NDArray[Any], v: NDArray[Any]) -> NDArray[Any]:
    """Return ``x.T @ v`` for small dense matrices."""
    rows, columns = x.shape
    out = np.zeros(columns)
    for i in range(rows):
        scale = v[i]
        if scale != 0.0:
            for p in range(columns):
                out[p] += x[i, p] * scale
    return out


@njit(cache=True, inline="always")
def _dynamics(
    state: NDArray[Any],
    control: NDArray[Any],
    dt: float,
    wheelbase: float,
    out: NDArray[Any],
) -> None:
    yaw = state[2]
    velocity = control[0]
    steering = control[1]
    out[0] = state[0] + velocity * math.cos(yaw) * dt
    out[1] = state[1] + velocity * math.sin(yaw) * dt
    out[2] = yaw + velocity * math.tan(steering) * dt / wheelbase
    out[3] = velocity
    out[4] = steering


@njit(cache=True)
def _rollout(
    initial_state: NDArray[Any],
    controls: NDArray[Any],
    dt: float,
    wheelbase: float,
) -> NDArray[Any]:
    horizon = controls.shape[0]
    states = np.empty((horizon + 1, STATE_DIM))
    for j in range(STATE_DIM):
        states[0, j] = initial_state[j]
    for index in range(horizon):
        _dynamics(states[index], controls[index], dt, wheelbase, states[index + 1])
    return states


@njit(cache=True)
def _cost(
    states: NDArray[Any],
    controls: NDArray[Any],
    reference: NDArray[Any],
    velocity_reference: NDArray[Any],
    q: NDArray[Any],
    qf: NDArray[Any],
    r: NDArray[Any],
    rd: NDArray[Any],
) -> float:
    value = 0.0
    for index in range(controls.shape[0]):
        error_x = states[index, 0] - reference[index, 0]
        error_y = states[index, 1] - reference[index, 1]
        error_yaw = _wrap(states[index, 2] - reference[index, 2])
        delta_velocity = controls[index, 0] - states[index, 3]
        delta_steering = controls[index, 1] - states[index, 4]
        value += 0.5 * (
            q[0] * error_x * error_x
            + q[1] * error_y * error_y
            + q[2] * error_yaw * error_yaw
        )
        velocity_error = controls[index, 0] - velocity_reference[index]
        value += 0.5 * r[0] * velocity_error * velocity_error
        value += 0.5 * r[1] * controls[index, 1] * controls[index, 1]
        value += 0.5 * (
            rd[0] * delta_velocity * delta_velocity
            + rd[1] * delta_steering * delta_steering
        )
    terminal_x = states[-1, 0] - reference[-1, 0]
    terminal_y = states[-1, 1] - reference[-1, 1]
    terminal_yaw = _wrap(states[-1, 2] - reference[-1, 2])
    return value + 0.5 * (
        qf[0] * terminal_x * terminal_x
        + qf[1] * terminal_y * terminal_y
        + qf[2] * terminal_yaw * terminal_yaw
    )


@njit(cache=True)
def _backward(
    states: NDArray[Any],
    controls: NDArray[Any],
    reference: NDArray[Any],
    velocity_reference: NDArray[Any],
    regularization: float,
    q: NDArray[Any],
    qf: NDArray[Any],
    r: NDArray[Any],
    rd: NDArray[Any],
    dt: float,
    wheelbase: float,
    feedforward: NDArray[Any],
    feedback: NDArray[Any],
) -> bool:
    """Fill the gains for one Riccati sweep and report whether it succeeded."""
    horizon = controls.shape[0]
    value_x = np.zeros(STATE_DIM)
    value_xx = np.zeros((STATE_DIM, STATE_DIM))
    value_x[0] = qf[0] * (states[-1, 0] - reference[-1, 0])
    value_x[1] = qf[1] * (states[-1, 1] - reference[-1, 1])
    value_x[2] = qf[2] * _wrap(states[-1, 2] - reference[-1, 2])
    for j in range(3):
        value_xx[j, j] = qf[j]

    lxx = np.zeros((STATE_DIM, STATE_DIM))
    lux = np.zeros((CONTROL_DIM, STATE_DIM))
    luu = np.zeros((CONTROL_DIM, CONTROL_DIM))
    for j in range(3):
        lxx[j, j] = q[j]
    for j in range(CONTROL_DIM):
        lxx[3 + j, 3 + j] = rd[j]
        lux[j, 3 + j] = -rd[j]
        luu[j, j] = r[j] + rd[j]

    a = np.zeros((STATE_DIM, STATE_DIM))
    b = np.zeros((STATE_DIM, CONTROL_DIM))
    a[0, 0] = 1.0
    a[1, 1] = 1.0
    a[2, 2] = 1.0
    b[3, 0] = 1.0
    b[4, 1] = 1.0
    lx = np.zeros(STATE_DIM)
    lu = np.zeros(CONTROL_DIM)

    for index in range(horizon - 1, -1, -1):
        yaw = states[index, 2]
        velocity = controls[index, 0]
        steering = controls[index, 1]
        delta_velocity = velocity - states[index, 3]
        delta_steering = steering - states[index, 4]

        lx[0] = q[0] * (states[index, 0] - reference[index, 0])
        lx[1] = q[1] * (states[index, 1] - reference[index, 1])
        lx[2] = q[2] * _wrap(states[index, 2] - reference[index, 2])
        lx[3] = -rd[0] * delta_velocity
        lx[4] = -rd[1] * delta_steering
        lu[0] = r[0] * (velocity - velocity_reference[index]) + rd[0] * delta_velocity
        lu[1] = r[1] * steering + rd[1] * delta_steering

        a[0, 2] = -velocity * math.sin(yaw) * dt
        a[1, 2] = velocity * math.cos(yaw) * dt
        b[0, 0] = math.cos(yaw) * dt
        b[1, 0] = math.sin(yaw) * dt
        b[2, 0] = math.tan(steering) * dt / wheelbase
        cosine = math.cos(steering)
        b[2, 1] = velocity * dt / (wheelbase * cosine * cosine)

        a_value = _mat_t_mat(a, value_xx)
        b_value = _mat_t_mat(b, value_xx)
        qx = lx + _mat_t_vec(a, value_x)
        qu = lu + _mat_t_vec(b, value_x)
        qxx = lxx + _mat_mat(a_value, a)
        quu = luu + _mat_mat(b_value, b)
        qux = lux + _mat_mat(b_value, a)

        m00 = quu[0, 0] + regularization
        m01 = quu[0, 1]
        m10 = quu[1, 0]
        m11 = quu[1, 1] + regularization
        determinant = m00 * m11 - m01 * m10
        if determinant == 0.0 or not math.isfinite(determinant):
            return False
        feedforward[index, 0] = -(m11 * qu[0] - m01 * qu[1]) / determinant
        feedforward[index, 1] = -(m00 * qu[1] - m10 * qu[0]) / determinant
        for j in range(STATE_DIM):
            feedback[index, 0, j] = -(m11 * qux[0, j] - m01 * qux[1, j]) / determinant
            feedback[index, 1, j] = -(m00 * qux[1, j] - m10 * qux[0, j]) / determinant

        k = feedforward[index]
        gain = feedback[index]
        quu_k = np.zeros(CONTROL_DIM)
        for i in range(CONTROL_DIM):
            for j in range(CONTROL_DIM):
                quu_k[i] += quu[i, j] * k[j]
        value_x = (
            qx + _mat_t_vec(gain, quu_k) + _mat_t_vec(gain, qu) + _mat_t_vec(qux, k)
        )
        gain_quu = _mat_t_mat(gain, quu)
        gain_qux = _mat_t_mat(gain, qux)
        value_xx = qxx + _mat_mat(gain_quu, gain) + gain_qux + gain_qux.T
        for i in range(STATE_DIM):
            for j in range(i + 1, STATE_DIM):
                symmetric = 0.5 * (value_xx[i, j] + value_xx[j, i])
                value_xx[i, j] = symmetric
                value_xx[j, i] = symmetric
    return True


@njit(cache=True)
def _forward_with_gains(
    initial_state: NDArray[Any],
    nominal_states: NDArray[Any],
    nominal_controls: NDArray[Any],
    feedforward: NDArray[Any],
    feedback: NDArray[Any],
    alpha: float,
    dt: float,
    wheelbase: float,
    velocity_low: float,
    velocity_high: float,
    steering_limit: float,
) -> tuple[NDArray[Any], NDArray[Any]]:
    horizon = nominal_controls.shape[0]
    states = np.empty((horizon + 1, STATE_DIM))
    controls = np.empty((horizon, CONTROL_DIM))
    for j in range(STATE_DIM):
        states[0, j] = initial_state[j]
    for index in range(horizon):
        velocity = nominal_controls[index, 0] + alpha * feedforward[index, 0]
        steering = nominal_controls[index, 1] + alpha * feedforward[index, 1]
        for j in range(STATE_DIM):
            deviation = states[index, j] - nominal_states[index, j]
            velocity += feedback[index, 0, j] * deviation
            steering += feedback[index, 1, j] * deviation
        controls[index, 0] = _clip(velocity, velocity_low, velocity_high)
        controls[index, 1] = _clip(steering, -steering_limit, steering_limit)
        _dynamics(states[index], controls[index], dt, wheelbase, states[index + 1])
    return states, controls


@njit(cache=True)
def _solve(
    initial_state: NDArray[Any],
    reference: NDArray[Any],
    velocity_reference: NDArray[Any],
    initial_controls: NDArray[Any],
    q: NDArray[Any],
    qf: NDArray[Any],
    r: NDArray[Any],
    rd: NDArray[Any],
    dt: float,
    wheelbase: float,
    velocity_low: float,
    velocity_high: float,
    steering_limit: float,
    max_iterations: int,
    convergence_tolerance: float,
) -> tuple[NDArray[Any], NDArray[Any], bool]:
    """Solve one finite-horizon tracking problem with regularized iLQR."""
    horizon = initial_controls.shape[0]
    controls = np.empty((horizon, CONTROL_DIM))
    for index in range(horizon):
        controls[index, 0] = _clip(
            initial_controls[index, 0], velocity_low, velocity_high
        )
        controls[index, 1] = _clip(
            initial_controls[index, 1], -steering_limit, steering_limit
        )
    states = _rollout(initial_state, controls, dt, wheelbase)
    cost = _cost(states, controls, reference, velocity_reference, q, qf, r, rd)
    regularization = INITIAL_REGULARIZATION
    feedforward = np.empty((horizon, CONTROL_DIM))
    feedback = np.empty((horizon, CONTROL_DIM, STATE_DIM))

    for _ in range(max_iterations):
        if not _backward(
            states,
            controls,
            reference,
            velocity_reference,
            regularization,
            q,
            qf,
            r,
            rd,
            dt,
            wheelbase,
            feedforward,
            feedback,
        ):
            regularization *= 10.0
            if regularization > MAX_REGULARIZATION:
                return states, controls, False
            continue
        accepted = False
        for alpha in LINE_SEARCH_ALPHAS:
            candidate_states, candidate_controls = _forward_with_gains(
                initial_state,
                states,
                controls,
                feedforward,
                feedback,
                alpha,
                dt,
                wheelbase,
                velocity_low,
                velocity_high,
                steering_limit,
            )
            candidate_cost = _cost(
                candidate_states,
                candidate_controls,
                reference,
                velocity_reference,
                q,
                qf,
                r,
                rd,
            )
            if math.isfinite(candidate_cost) and candidate_cost < cost:
                improvement = cost - candidate_cost
                states = candidate_states
                controls = candidate_controls
                cost = candidate_cost
                regularization = max(regularization / 5.0, MIN_REGULARIZATION)
                accepted = True
                if improvement < convergence_tolerance:
                    return states, controls, True
                break
        if not accepted:
            regularization *= 10.0
            if regularization > MAX_REGULARIZATION:
                break
    return states, controls, bool(np.all(np.isfinite(states)))


class PlannerILQRRefinement:
    """Reconnect a pose-augmented ego future using a bicycle-model iLQR."""

    def __init__(
        self,
        num_refine: int = 20,
        time_step_s: float = 0.1,
        wheelbase_m: float = 2.79,
        stop_speed_threshold: float = 0.1,
        state_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
        terminal_weight_scale: float = 10.0,
        velocity_weight: float = 0.2,
        steering_weight: float = 0.1,
        velocity_rate_weight: float = 1.0,
        steering_rate_weight: float = 10.0,
        velocity_bounds: tuple[float, float] = (0.0, 30.0),
        steering_limit_rad: float = 0.7,
        max_iterations: int = 15,
        convergence_tolerance: float = 1e-4,
    ) -> None:
        self.num_refine = num_refine
        self.dt = time_step_s
        self.wheelbase = wheelbase_m
        self.stop_speed_threshold = stop_speed_threshold
        self.q = np.asarray(state_weights, dtype=np.float64)
        self.qf = terminal_weight_scale * self.q
        self.r = np.asarray((velocity_weight, steering_weight), dtype=np.float64)
        self.rd = np.asarray(
            (velocity_rate_weight, steering_rate_weight), dtype=np.float64
        )
        self.velocity_bounds = velocity_bounds
        self.steering_limit = steering_limit_rad
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        if self.dt <= 0.0 or self.wheelbase <= 0.0:
            raise ValueError("time_step_s and wheelbase_m must be positive")

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        applied = input_data.get(POSE_AUGMENTATION_APPLIED_KEY)
        if applied is None or not bool(np.asarray(applied).item()):
            return output
        future = input_data.get("ego_agent_future")
        if future is None:
            return output
        if np.all(np.abs(future[:, EGO_VELOCITY_INDEX]) <= self.stop_speed_threshold):
            stopped_future = np.zeros_like(future)
            stopped_future[:, 2] = 1.0
            output["ego_agent_future"] = stopped_future
            return output
        refined = self.refine_future(future, input_data["ego_agent_past"])
        if refined is None:
            return output
        output["ego_agent_future"] = refined
        return output

    def refine_future(
        self, future: NDArray[Any], past: NDArray[Any]
    ) -> NDArray[Any] | None:
        """Return a dynamically feasible refined prefix, or None on failure."""
        horizon = min(self.num_refine + 1, len(future))
        if horizon < 2 or len(past) == 0:
            return np.array(future, copy=True)

        reference = np.empty((horizon + 1, 3), dtype=np.float64)
        reference[0] = (0.0, 0.0, 0.0)
        reference[1:, :2] = future[:horizon, :2]
        reference[1:, 2] = np.arctan2(future[:horizon, 3], future[:horizon, 2])
        velocity_reference = np.ascontiguousarray(future[:horizon, 4], dtype=np.float64)

        current_speed = max(float(past[-1, 4]), 0.0)
        current_steering = self._steering_from_motion(current_speed, float(past[-1, 5]))
        initial_state = np.asarray(
            (0.0, 0.0, 0.0, current_speed, current_steering), dtype=np.float64
        )
        controls = np.column_stack(
            (
                velocity_reference,
                self._steering_from_future(future[:horizon]),
            )
        )
        solution = self.solve(initial_state, reference, velocity_reference, controls)
        if solution is None:
            return None
        states, controls = solution

        result = np.array(future, copy=True)
        result[:horizon, :2] = states[1:, :2]
        yaw = states[1:, 2]
        result[:horizon, 2] = np.cos(yaw)
        result[:horizon, 3] = np.sin(yaw)
        result[:horizon, 4] = controls[:, 0]
        result[:horizon, 5] = controls[:, 0] * np.tan(controls[:, 1]) / self.wheelbase
        return result.astype(future.dtype, copy=False)

    def solve(
        self,
        initial_state: NDArray[Any],
        reference: NDArray[Any],
        velocity_reference: NDArray[Any],
        initial_controls: NDArray[Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Solve one finite-horizon tracking problem."""
        states, controls, succeeded = _solve(
            np.ascontiguousarray(initial_state, dtype=np.float64),
            np.ascontiguousarray(reference, dtype=np.float64),
            np.ascontiguousarray(velocity_reference, dtype=np.float64),
            np.ascontiguousarray(initial_controls, dtype=np.float64),
            self.q,
            self.qf,
            self.r,
            self.rd,
            self.dt,
            self.wheelbase,
            float(self.velocity_bounds[0]),
            float(self.velocity_bounds[1]),
            self.steering_limit,
            int(self.max_iterations),
            self.convergence_tolerance,
        )
        return (states, controls) if succeeded else None

    def _steering_from_future(self, future: NDArray[Any]) -> NDArray[np.float64]:
        """Return the steering implied by each future speed and yaw rate."""
        speeds = np.asarray(future[:, 4], dtype=np.float64)
        yaw_rates = np.asarray(future[:, 5], dtype=np.float64)
        steering = np.zeros(len(future), dtype=np.float64)
        moving = np.abs(speeds) >= 1e-3
        np.divide(self.wheelbase * yaw_rates, speeds, out=steering, where=moving)
        np.arctan(steering, out=steering)
        return np.clip(steering, -self.steering_limit, self.steering_limit)

    def _steering_from_motion(self, velocity: float, yaw_rate: float) -> float:
        if abs(velocity) < 1e-3:
            return 0.0
        return float(
            np.clip(
                math.atan(self.wheelbase * yaw_rate / velocity),
                -self.steering_limit,
                self.steering_limit,
            )
        )
