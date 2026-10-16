"""Inputs of an ML Planner that reads steering and the wheel base, from scenario_sim."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_generation.ml_planner_inputs import MlPlannerOnnx, Observer, vehicle_spec

VEHICLE_INFO = """\
/**:
  ros__parameters:
    wheel_radius: 0.3725
    wheel_width: 0.215
    wheel_base: 4.76012
    wheel_tread: 1.754
    front_overhang: 0.95099
    rear_overhang: 1.52579
    left_overhang: 0.32358
    right_overhang: 0.34983
    vehicle_height: 3.080
    max_steer_angle: 0.640
"""


class FakeObserver:
    def __init__(self, *args):
        self.args = args
        self.calls = []

    def observe(self, *args):
        self.calls.append(args)

    def inputs(self):
        return {}


def _fake_osp(width):
    return SimpleNamespace(ML_PLANNER_EGO_HISTORY_DIM=width, MlPlannerObserver=FakeObserver)


def _model(width, spec=None):
    model = MlPlannerOnnx.__new__(MlPlannerOnnx)
    model.layout = SimpleNamespace(width=width)
    model.vehicle_spec = spec
    return model


def _state(speed, yaw_rate):
    return {"twist": {"linear_x": speed, "angular_z": yaw_rate}}


def test_vehicle_spec_is_composed_as_the_node_does(tmp_path):
    path = tmp_path / "vehicle_info.param.yaml"
    path.write_text(VEHICLE_INFO)
    np.testing.assert_allclose(vehicle_spec(path), [5.71111, 7.2369, 2.42741, 4.76012], atol=1e-6)


def test_steering_observer_reports_the_kinematic_angle_and_holds_it_when_slow():
    inner = FakeObserver()
    observer = Observer(inner, wheel_base=4.0)
    observer.observe(1, _state(10.0, 0.25))
    observer.observe(1, _state(0.5, 0.0))
    assert inner.calls[0] == (1, pytest.approx(math.atan(0.1)))
    assert inner.calls[1] == (1, pytest.approx(math.atan(0.1)))


def test_observer_without_steering_calls_the_simulator_as_before():
    inner = FakeObserver()
    Observer(inner).observe(2, _state(10.0, 0.25))
    assert inner.calls == [(2,)]


def test_model_passes_its_vehicle_only_to_a_steering_preprocessing():
    spec = [5.71111, 7.2369, 2.42741, 4.76012]
    observer = _model(7, spec).observer(_fake_osp(7), "ego", [1], [0.0, 0.0, 0.0])
    assert observer._observer.args == ("ego", [1], [0.0, 0.0, 0.0], spec)
    observer = _model(6).observer(_fake_osp(6), "ego", [1], [0.0, 0.0, 0.0])
    assert observer._observer.args == ("ego", [1], [0.0, 0.0, 0.0])


def test_model_refuses_an_overlay_built_for_the_other_interface():
    with pytest.raises(ValueError, match="7-column"):
        _model(6).observer(_fake_osp(7), "ego", [1], [0.0, 0.0, 0.0])
    # An overlay older than the attribute builds the 6-column history.
    with pytest.raises(ValueError, match="6-column"):
        _model(7, [1.0] * 4).observer(SimpleNamespace(), "ego", [1], [0.0, 0.0, 0.0])
