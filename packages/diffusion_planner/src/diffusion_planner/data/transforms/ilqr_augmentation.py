"""Ego-pose augmentation with kinematic-bicycle iLQR refinement."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Frame, FrameLike
from .rigid_augmentation import (
    apply_rigid_pose_augmentation,
    has_sufficient_future_speed,
)


def _wrap(angle: float | NDArray[Any]) -> Any:
    return np.arctan2(np.sin(angle), np.cos(angle))


class PlannerILQRAugmentation:
    """Move the ego pose and reconnect its future using a bicycle-model iLQR."""

    def __init__(
        self,
        longitudinal_offset_range: tuple[float, float] = (0.0, 0.0),
        lateral_offset_range: tuple[float, float] = (-1.0, 1.0),
        yaw_offset_range: tuple[float, float] = (-math.radians(5), math.radians(5)),
        pose_probability: float = 0.5,
        num_refine: int = 20,
        time_step_s: float = 0.1,
        wheelbase_m: float = 2.79,
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
        pose_augmentation_speed_threshold: float = 0.1,
        pose_augmentation_speed_check_index: int = 20,
    ) -> None:
        self.longitudinal_offset_range = longitudinal_offset_range
        self.lateral_offset_range = lateral_offset_range
        self.yaw_offset_range = yaw_offset_range
        self.pose_probability = pose_probability
        self.num_refine = num_refine
        self.dt = time_step_s
        self.wheelbase = wheelbase_m
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
        self.pose_augmentation_speed_threshold = pose_augmentation_speed_threshold
        self.pose_augmentation_speed_check_index = pose_augmentation_speed_check_index
        if self.dt <= 0.0 or self.wheelbase <= 0.0:
            raise ValueError("time_step_s and wheelbase_m must be positive")

    def __call__(self, input_data: FrameLike) -> Frame:
        if (
            not has_sufficient_future_speed(
                input_data,
                self.pose_augmentation_speed_check_index,
                self.pose_augmentation_speed_threshold,
            )
            or np.random.random() >= self.pose_probability
        ):
            return dict(input_data)

        longitudinal_offset = 0.0
        if any(value != 0.0 for value in self.longitudinal_offset_range):
            longitudinal_offset = np.random.uniform(*self.longitudinal_offset_range)
        lateral_offset = np.random.uniform(*self.lateral_offset_range)
        yaw_offset = np.random.uniform(*self.yaw_offset_range)
        output, future = apply_rigid_pose_augmentation(
            input_data, longitudinal_offset, lateral_offset, yaw_offset
        )
        if future is None:
            return output
        refined = self.refine_future(future, output["ego_agent_past"])
        if refined is None:
            return dict(input_data)
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
        velocity_reference = np.asarray(future[:horizon, 4], dtype=np.float64)

        current_speed = max(float(past[-1, 4]), 0.0)
        current_steering = self._steering_from_motion(current_speed, float(past[-1, 5]))
        initial_state = np.asarray(
            (0.0, 0.0, 0.0, current_speed, current_steering), dtype=np.float64
        )
        controls = np.column_stack(
            (
                velocity_reference,
                [
                    self._steering_from_motion(float(state[4]), float(state[5]))
                    for state in future[:horizon]
                ],
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
        controls = self._clip_controls(np.asarray(initial_controls, dtype=np.float64))
        states = self._rollout(initial_state, controls)
        cost = self._cost(states, controls, reference, velocity_reference)
        regularization = 1e-5

        for _ in range(self.max_iterations):
            gains = self._backward(
                states, controls, reference, velocity_reference, regularization
            )
            if gains is None:
                regularization *= 10.0
                if regularization > 1e8:
                    return None
                continue
            feedforward, feedback = gains
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
                candidate_states, candidate_controls = self._forward_with_gains(
                    initial_state, states, controls, feedforward, feedback, alpha
                )
                candidate_cost = self._cost(
                    candidate_states,
                    candidate_controls,
                    reference,
                    velocity_reference,
                )
                if np.isfinite(candidate_cost) and candidate_cost < cost:
                    improvement = cost - candidate_cost
                    states, controls, cost = (
                        candidate_states,
                        candidate_controls,
                        candidate_cost,
                    )
                    regularization = max(regularization / 5.0, 1e-8)
                    accepted = True
                    if improvement < self.convergence_tolerance:
                        return states, controls
                    break
            if not accepted:
                regularization *= 10.0
                if regularization > 1e8:
                    break
        return (states, controls) if np.all(np.isfinite(states)) else None

    def _dynamics(self, state: NDArray[Any], control: NDArray[Any]) -> NDArray[Any]:
        x, y, yaw = state[:3]
        velocity, steering = control
        return np.asarray(
            (
                x + velocity * math.cos(yaw) * self.dt,
                y + velocity * math.sin(yaw) * self.dt,
                yaw + velocity * math.tan(steering) * self.dt / self.wheelbase,
                velocity,
                steering,
            )
        )

    def _jacobians(
        self, state: NDArray[Any], control: NDArray[Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        yaw = float(state[2])
        velocity, steering = map(float, control)
        a = np.zeros((5, 5), dtype=np.float64)
        a[:3, :3] = np.eye(3)
        a[0, 2] = -velocity * math.sin(yaw) * self.dt
        a[1, 2] = velocity * math.cos(yaw) * self.dt
        b = np.zeros((5, 2), dtype=np.float64)
        b[0, 0] = math.cos(yaw) * self.dt
        b[1, 0] = math.sin(yaw) * self.dt
        b[2, 0] = math.tan(steering) * self.dt / self.wheelbase
        b[2, 1] = velocity * self.dt / (self.wheelbase * math.cos(steering) ** 2)
        b[3:, :] = np.eye(2)
        return a, b

    def _rollout(
        self, initial_state: NDArray[Any], controls: NDArray[Any]
    ) -> NDArray[np.float64]:
        states = np.empty((len(controls) + 1, 5), dtype=np.float64)
        states[0] = initial_state
        for index, control in enumerate(controls):
            states[index + 1] = self._dynamics(states[index], control)
        return states

    def _cost(
        self,
        states: NDArray[Any],
        controls: NDArray[Any],
        reference: NDArray[Any],
        velocity_reference: NDArray[Any],
    ) -> float:
        value = 0.0
        for index, control in enumerate(controls):
            error = states[index, :3] - reference[index]
            error[2] = _wrap(error[2])
            delta_control = control - states[index, 3:]
            value += 0.5 * float(error @ (self.q * error))
            value += 0.5 * self.r[0] * (control[0] - velocity_reference[index]) ** 2
            value += 0.5 * self.r[1] * control[1] ** 2
            value += 0.5 * float(delta_control @ (self.rd * delta_control))
        terminal_error = states[-1, :3] - reference[-1]
        terminal_error[2] = _wrap(terminal_error[2])
        return value + 0.5 * float(terminal_error @ (self.qf * terminal_error))

    def _backward(
        self,
        states: NDArray[Any],
        controls: NDArray[Any],
        reference: NDArray[Any],
        velocity_reference: NDArray[Any],
        regularization: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        horizon = len(controls)
        feedforward = np.empty((horizon, 2), dtype=np.float64)
        feedback = np.empty((horizon, 2, 5), dtype=np.float64)
        terminal_error = states[-1, :3] - reference[-1]
        terminal_error[2] = _wrap(terminal_error[2])
        value_x = np.zeros(5)
        value_x[:3] = self.qf * terminal_error
        value_xx = np.zeros((5, 5))
        value_xx[:3, :3] = np.diag(self.qf)

        for index in range(horizon - 1, -1, -1):
            state, control = states[index], controls[index]
            error = state[:3] - reference[index]
            error[2] = _wrap(error[2])
            delta_control = control - state[3:]
            lx = np.zeros(5)
            lx[:3] = self.q * error
            lx[3:] = -self.rd * delta_control
            lu = (
                self.r
                * np.asarray((control[0] - velocity_reference[index], control[1]))
                + self.rd * delta_control
            )
            lxx = np.zeros((5, 5))
            lxx[:3, :3] = np.diag(self.q)
            lxx[3:, 3:] = np.diag(self.rd)
            luu = np.diag(self.r + self.rd)
            lux = np.zeros((2, 5))
            lux[:, 3:] = -np.diag(self.rd)
            a, b = self._jacobians(state, control)
            qx = lx + a.T @ value_x
            qu = lu + b.T @ value_x
            qxx = lxx + a.T @ value_xx @ a
            quu = luu + b.T @ value_xx @ b
            qux = lux + b.T @ value_xx @ a
            quu_regularized = quu + regularization * np.eye(2)
            try:
                feedforward[index] = -np.linalg.solve(quu_regularized, qu)
                feedback[index] = -np.linalg.solve(quu_regularized, qux)
            except np.linalg.LinAlgError:
                return None
            k, gain = feedforward[index], feedback[index]
            value_x = qx + gain.T @ quu @ k + gain.T @ qu + qux.T @ k
            value_xx = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
            value_xx = 0.5 * (value_xx + value_xx.T)
        return feedforward, feedback

    def _forward_with_gains(
        self,
        initial_state: NDArray[Any],
        nominal_states: NDArray[Any],
        nominal_controls: NDArray[Any],
        feedforward: NDArray[Any],
        feedback: NDArray[Any],
        alpha: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        states = np.empty_like(nominal_states)
        controls = np.empty_like(nominal_controls)
        states[0] = initial_state
        for index in range(len(controls)):
            controls[index] = nominal_controls[index] + alpha * feedforward[index]
            controls[index] += feedback[index] @ (states[index] - nominal_states[index])
            controls[index] = self._clip_controls(controls[index])
            states[index + 1] = self._dynamics(states[index], controls[index])
        return states, controls

    def _clip_controls(self, controls: NDArray[Any]) -> NDArray[np.float64]:
        result = np.array(controls, dtype=np.float64, copy=True)
        result[..., 0] = np.clip(result[..., 0], *self.velocity_bounds)
        result[..., 1] = np.clip(
            result[..., 1], -self.steering_limit, self.steering_limit
        )
        return result

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
