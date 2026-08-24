from __future__ import annotations

import numpy as np
import pytest

from scenario_generation.closed_loop_delay import PlantParameters
from scenario_generation.mpc_tracker import DelayedPlantTracker


class _ConstantController:
    def __init__(self, *, accel: float, steer: float, dt: float = 0.1):
        self.accel = accel
        self.steer = steer
        self.dt = dt
        self.max_speed = 20.0
        self.max_steer = 1.0
        self.last_accel = 0.0
        self.last_steering = 0.0
        self.inputs = []
        self.reset_count = 0

    def track(self, x0, ref_world):
        self.inputs.append((np.asarray(x0).copy(), np.asarray(ref_world).copy()))
        self.last_accel = self.accel
        self.last_steering = self.steer
        return np.asarray(x0[:3], dtype=np.float32), float(x0[3])

    def reset(self):
        self.reset_count += 1


def _tracker(params, *, accel=2.0, steer=0.4, compensate=False):
    inner = _ConstantController(accel=accel, steer=steer)
    tracker = DelayedPlantTracker(
        wheelbase=4.0,
        plant_parameters=params,
        controller_compensation=compensate,
        inner_tracker=inner,
    )
    return tracker, inner


def _step(tracker, x0):
    ref = np.column_stack(
        [np.linspace(1.0, 8.0, 8), np.zeros(8), np.zeros(8)]
    )
    pose, speed = tracker.track(np.asarray(x0, dtype=np.float64), ref)
    return np.r_[pose, speed]


def test_first_order_step_response_is_separate_for_accel_and_steer():
    tracker, _inner = _tracker(
        PlantParameters(
            steer_dead_time_s=0.0,
            steer_time_constant_s=0.1,
            accel_dead_time_s=0.0,
            accel_time_constant_s=0.3,
        )
    )
    state = np.array([0.0, 0.0, 0.0, 5.0])

    state = _step(tracker, state)
    assert tracker.last_steering == pytest.approx(0.2)
    assert tracker.last_accel == pytest.approx(0.5)
    state = _step(tracker, state)
    assert tracker.last_steering == pytest.approx(0.3)
    assert tracker.last_accel == pytest.approx(0.875)


def test_dead_time_buffers_are_independent_and_tick_rounded():
    tracker, _inner = _tracker(
        PlantParameters(
            steer_dead_time_s=0.17,
            steer_time_constant_s=0.0,
            accel_dead_time_s=0.10,
            accel_time_constant_s=0.0,
        )
    )
    assert tracker.steer_delay_steps == 2
    assert tracker.accel_delay_steps == 1
    state = np.array([0.0, 0.0, 0.0, 5.0])

    state = _step(tracker, state)
    assert tracker.last_steering == 0.0
    assert tracker.last_accel == 0.0
    state = _step(tracker, state)
    assert tracker.last_steering == 0.0
    assert tracker.last_accel == pytest.approx(2.0)
    _step(tracker, state)
    assert tracker.last_steering == pytest.approx(0.4)
    assert tracker.last_accel == pytest.approx(2.0)


def test_controller_compensation_previews_pending_ticks_and_shifts_reference():
    params = PlantParameters(0.2, 0.0, 0.1, 0.0)
    tracker, inner = _tracker(params, compensate=True)
    x0 = np.array([1.0, 2.0, 0.0, 5.0])
    ref = np.column_stack(
        [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)]
    )

    tracker.track(x0, ref)

    controller_x0, controller_ref = inner.inputs[0]
    assert controller_x0[0] == pytest.approx(2.0)  # two 0.1 s preview ticks at 5 m/s
    assert controller_x0[1] == pytest.approx(2.0)
    np.testing.assert_array_equal(controller_ref, ref[2:])


def test_no_compensation_passes_current_state_and_full_reference():
    params = PlantParameters(0.2, 0.0, 0.1, 0.0)
    tracker, inner = _tracker(params, compensate=False)
    x0 = np.array([1.0, 2.0, 0.0, 5.0])
    ref = np.column_stack(
        [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)]
    )

    tracker.track(x0, ref)

    np.testing.assert_array_equal(inner.inputs[0][0], x0)
    np.testing.assert_array_equal(inner.inputs[0][1], ref)


def test_reset_clears_plant_state_and_inner_controller():
    tracker, inner = _tracker(PlantParameters(0.0, 0.1, 0.0, 0.1))
    _step(tracker, np.array([0.0, 0.0, 0.0, 5.0]))
    assert tracker.last_accel != 0.0

    tracker.reset()

    assert inner.reset_count == 1
    assert tracker.last_accel == 0.0
    assert tracker.last_steering == 0.0
    assert tracker._effective_accel == 0.0
    assert tracker._effective_steer == 0.0
