"""autoware_ml_planner's road border avoidance and trajectory optimizer, driven through the binding.

Repeats checks from the package's trajectory_optimizer_test and road_border_avoidance_test with its
default ml_planner.param.yaml and both stages enabled.
"""

import os
from pathlib import Path
from typing import NamedTuple

import ml_planner_optimizer
import numpy as np
import pytest
import yaml

N = ml_planner_optimizer.HORIZON
DT = ml_planner_optimizer.DT
SPEED = 8.0
EGO = (0.0, 0.0, 0.0, SPEED)  # x, y, yaw, v
TOL = 1e-6
# road_border_avoidance_test's vehicle: tread 1.6 m + 0.1 m overhangs = 1.8 m wide.
TEST_VEHICLE = {
    "wheel_radius": 0.3,
    "wheel_width": 0.2,
    "wheel_base": 2.75,
    "wheel_tread": 1.6,
    "front_overhang": 1.0,
    "rear_overhang": 1.0,
    "left_overhang": 0.1,
    "right_overhang": 0.1,
    "vehicle_height": 2.0,
    "max_steer_angle": 0.7,
}


class Config(NamedTuple):
    param_yaml: str
    vehicle_yaml: str
    params: dict  # ros__parameters as written to param_yaml


def _write_ros_params(path: Path, params: dict) -> str:
    path.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}))
    return str(path)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    defaults = Path(os.environ["ML_PLANNER_SOURCE_DIR"]) / "config" / "ml_planner.param.yaml"
    params = yaml.safe_load(defaults.read_text())["/**"]["ros__parameters"]
    params["trajectory_optimization"]["enable"] = True
    params["road_border_avoidance"]["enable"] = True
    tmp = tmp_path_factory.mktemp("params")
    return Config(
        _write_ros_params(tmp / "ml_planner.param.yaml", params),
        _write_ros_params(tmp / "vehicle_info.param.yaml", TEST_VEHICLE),
        params,
    )


@pytest.fixture
def optimizer(config):
    return ml_planner_optimizer.Optimizer(config.param_yaml, config.vehicle_yaml)


def _straight(noise):
    """Constant-speed line along +x from one DT ahead, with seeded position noise."""
    rng = np.random.default_rng(42)
    t = DT * np.arange(1, N + 1)
    return np.stack(
        [SPEED * t + rng.normal(0, noise, N), rng.normal(0, noise, N), np.zeros(N)], axis=1
    )


def _border_at(y):
    return np.array([[-10.0, y], [100.0, y]])


def test_optimizes_noisy_trajectory(optimizer, config):
    limits = config.params["trajectory_optimization"]
    out = optimizer.step(_straight(noise=0.15), EGO, 0.0, 0.0)
    assert out["optimized"], out["solver_status"]
    assert out["trajectory"].shape == (N, 7)
    x, y, _, v, steer, accel, _ = out["trajectory"].T
    # The first point follows from base_link at the ego speed through the dynamics.
    assert x[0] == pytest.approx(SPEED * DT, abs=0.3)
    assert y[0] == pytest.approx(0.0, abs=0.3)
    assert np.all(v >= limits["min_velocity_mps"] - TOL)
    assert np.all(np.abs(steer) <= TEST_VEHICLE["max_steer_angle"] + TOL)
    assert np.all(accel >= limits["min_acceleration_mps2"] - TOL)
    assert np.all(accel <= limits["max_acceleration_mps2"] + TOL)
    # The solution stays close to the noise-free line.
    assert np.all(np.abs(y) < 0.5)


def test_leaves_the_reference_when_clear(optimizer):
    optimizer.set_road_borders([_border_at(5.0)])
    out = optimizer.step(_straight(noise=0.0), EGO, 0.0, 0.0)
    assert out["border_shifted_points"] == 0
    assert out["border_max_shift_m"] == 0.0


def test_shifts_away_from_a_close_border(optimizer, config):
    # y = 1.0 is inside half the width (0.9 m) plus footprint_margin_m.
    optimizer.set_road_borders([_border_at(1.0)])
    out = optimizer.step(_straight(noise=0.0), EGO, 0.0, 0.0)
    assert out["border_shifted_points"] > 0
    assert out["border_unresolved_points"] == 0
    max_shift = config.params["road_border_avoidance"]["max_lateral_shift_m"]
    assert 0.0 < out["border_max_shift_m"] <= max_shift
    assert out["optimized"], out["solver_status"]
    # The optimizer follows the shifted reference away from the border.
    assert np.mean(out["trajectory"][N // 2 :, 1]) < 0.0


def test_rejects_a_raw_plan_of_the_wrong_length(optimizer):
    with pytest.raises(ValueError, match="raw must be"):
        optimizer.step(np.zeros((N - 1, 3)), EGO, 0.0, 0.0)
