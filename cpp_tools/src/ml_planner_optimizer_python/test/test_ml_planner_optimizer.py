"""The binding carries arrays to autoware_ml_planner's classes and back without changing them.

Mirrors the package's own trajectory_optimizer_test / road_border_avoidance_test, through Python
and with the launcher parameters this package ships.
"""

import math
from pathlib import Path

import numpy as np
import pytest

ml_planner_optimizer = pytest.importorskip("ml_planner_optimizer")

CONFIG = Path(__file__).resolve().parents[1] / "config"
PARAMS = str(CONFIG / "ml_planner.param.yaml")
VEHICLE = str(CONFIG / "vehicle_info.param.yaml")
# The upstream tests' vehicle: 1.8 m wide, wheelbase 2.75 m.
TEST_VEHICLE = {
    "wheel_base": 2.75,
    "wheel_tread": 1.6,
    "front_overhang": 1.0,
    "rear_overhang": 1.0,
    "left_overhang": 0.1,
    "right_overhang": 0.1,
    "max_steer_angle": 0.7,
}
N = ml_planner_optimizer.HORIZON
DT = ml_planner_optimizer.DT


def _noisy_straight(speed=8.0, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    t = DT * np.arange(1, N + 1)
    return np.stack(
        [speed * t + rng.normal(0, noise, N), rng.normal(0, noise, N), np.zeros(N)], axis=1
    )


def _optimizer(**kw):
    return ml_planner_optimizer.Optimizer(PARAMS, VEHICLE, kw.pop("vehicle", TEST_VEHICLE))


def test_both_stages_follow_the_launcher():
    opt = _optimizer()
    assert opt.optimizes and opt.avoids_road_borders


def test_optimizes_noisy_trajectory():
    opt = _optimizer()
    out = opt.step(_noisy_straight(), (0.0, 0.0, 0.0, 8.0), 0.0, 0.0)
    assert out["optimized"], out["solver_status"]
    traj = out["trajectory"]
    assert traj.shape == (N, 7)
    # Point 0 is t = 0.1 s, reached from base_link at v0 = 8 m/s.
    assert traj[0, 0] == pytest.approx(8.0 * DT, abs=0.3)
    assert traj[0, 1] == pytest.approx(0.0, abs=0.3)
    assert np.all(traj[:, 3] >= -1e-6)
    assert np.all(np.abs(traj[:, 4]) <= TEST_VEHICLE["max_steer_angle"] + 1e-6)
    assert np.all((traj[:, 5] >= -4.0 - 1e-6) & (traj[:, 5] <= 3.0 + 1e-6))
    assert np.all(np.abs(traj[:, 1]) < 0.5)


def test_map_frame_round_trip():
    """The same plan placed elsewhere in the map comes back as the same solution, moved there."""
    local = _noisy_straight()
    at_origin = _optimizer().step(local, (0.0, 0.0, 0.0, 8.0), 0.0, 0.0)["trajectory"]
    yaw, ox, oy = 0.7, 100.0, -50.0
    c, s = math.cos(yaw), math.sin(yaw)
    raw = np.stack(
        [
            ox + c * local[:, 0] - s * local[:, 1],
            oy + s * local[:, 0] + c * local[:, 1],
            local[:, 2] + yaw,
        ],
        axis=1,
    )
    moved = _optimizer().step(raw, (ox, oy, yaw, 8.0), 0.0, 0.0)["trajectory"]
    dx, dy = moved[:, 0] - ox, moved[:, 1] - oy
    back = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
    np.testing.assert_allclose(back, at_origin[:, :2], atol=1e-3)
    np.testing.assert_allclose(moved[:, 2] - yaw, at_origin[:, 2], atol=1e-3)
    np.testing.assert_allclose(moved[:, 3:], at_origin[:, 3:], atol=1e-3)


def test_successive_cycles_stay_solvable():
    opt = _optimizer()
    raw = _noisy_straight()
    first = opt.step(raw, (0.0, 0.0, 0.0, 8.0), 0.0, 10.0)
    second = opt.step(raw + [0.8, 0, 0], (0.8, 0.0, 0.0, 8.0), 0.0, 10.1)
    assert first["optimized"] and second["optimized"]


def test_road_border_shifts_the_reference():
    opt = _optimizer()
    raw = _noisy_straight(noise=0.0)
    clear = opt.step(raw, (0.0, 0.0, 0.0, 8.0), 0.0, 0.0)
    assert clear["border_shifted_points"] == 0
    # A border 1.0 m to the left: closer than half the width (0.9 m) plus the 0.4 m margin.
    opt.set_road_borders([np.array([[-10.0, 1.0], [100.0, 1.0]])])
    out = opt.step(raw, (0.0, 0.0, 0.0, 8.0), 0.0, 1.0)
    assert out["border_shifted_points"] > 0
    assert 0.0 < out["border_max_shift_m"] <= 1.0 + 1e-9  # max_lateral_shift_m in the launcher
    assert clear["border_max_shift_m"] == 0.0
    assert out["optimized"]
    # Points from t = 2.0 s on are pushed away from it.
    assert np.mean(out["trajectory"][30:, 1]) < -0.1


def test_rejects_wrong_shapes():
    opt = _optimizer()
    with pytest.raises(ValueError):
        opt.step(np.zeros((10, 3)), (0.0, 0.0, 0.0, 0.0), 0.0, 0.0)
    with pytest.raises(ValueError):
        ml_planner_optimizer.Optimizer(PARAMS, VEHICLE, {"no_such_key": 1.0})
