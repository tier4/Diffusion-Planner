"""PlanOptimizer: its switch, what it feeds the native optimizer, and how it treats a failed solve.

The native module is replaced by a recorder; the real one is exercised by
cpp_tools/src/ml_planner_optimizer_python/test.
"""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest
import yaml

from scenario_generation import plan_optimizer as po

# The vehicle of autoware_ml_planner's own tests.
VEHICLE = {"wheel_base": 2.75, "wheel_tread": 1.6, "front_overhang": 1.0, "rear_overhang": 1.0}


class _Recorder:
    """Stands in for ml_planner_optimizer.Optimizer; ``results`` are returned in order."""

    def __init__(self, param_yaml, vehicle_yaml, overrides):
        self.overrides = overrides
        self.steers: list[float] = []
        self.results: list[np.ndarray | None] = []

    def set_road_borders(self, borders):
        pass

    def step(self, raw, ego, steer, stamp, goal):
        self.steers.append(steer)
        return {
            "trajectory": self.results.pop(0),
            "solve_time_ms": 2.0,
            "border_shifted_points": 3,
            "border_unresolved_points": 0,
            "border_max_shift_m": 0.25,
        }


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """The module installed by colcon, with its default parameters under share/."""
    prefix = tmp_path / "install"
    config = prefix / "share" / "ml_planner_optimizer_python" / "config"
    config.mkdir(parents=True)
    (config / "ml_planner.param.yaml").write_text("")
    module = types.ModuleType("ml_planner_optimizer")
    module.__file__ = str(prefix / "lib/python3.10/site-packages/ml_planner_optimizer.so")
    module.Optimizer = _Recorder
    monkeypatch.setitem(sys.modules, "ml_planner_optimizer", module)
    return config


@pytest.fixture
def config_dir(tmp_path):
    path = tmp_path / "vehicle"
    path.mkdir()
    doc = {"/**": {"ros__parameters": VEHICLE}}
    (path / "vehicle_info.param.yaml").write_text(yaml.safe_dump(doc))
    return path


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    po.add_plan_optimizer_argument(parser)
    return parser.parse_args(argv)


def test_switch_takes_one_form():
    assert _args().plan_optimizer is False
    assert _args("--plan_optimizer").plan_optimizer is True
    assert _args("--plan_optimizer", "false").plan_optimizer is False
    assert po.PlanOptimizerConfig.from_args(_args()) is None


def _config(config_dir) -> po.PlanOptimizerConfig:
    args = _args("--plan_optimizer", "--plan_optimizer_config_dir", str(config_dir))
    return po.PlanOptimizerConfig.from_args(args)


def test_parameters_default_to_the_installed_ones(installed, config_dir, tmp_path):
    # colcon build --symlink-install: site-packages/<module>.so -> build/<pkg>/<module>.so
    built = tmp_path / "build" / "ml_planner_optimizer_python" / "ml_planner_optimizer.so"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"")
    link = sys.modules["ml_planner_optimizer"].__file__
    (tmp_path / "install/lib/python3.10/site-packages").mkdir(parents=True)
    (tmp_path / link).symlink_to(built)
    assert _config(config_dir).param_yaml == installed / "ml_planner.param.yaml"
    (config_dir / "ml_planner.param.yaml").write_text("")
    assert _config(config_dir).param_yaml == config_dir / "ml_planner.param.yaml"


def _optimizer(config_dir, length=4.0, width=1.8, center_x=1.0) -> po.PlanOptimizer:
    box = {"dimensions": {"x": length, "y": width}, "center": {"x": center_x}}
    return po.PlanOptimizer(_config(config_dir), box, [])


@pytest.mark.usefixtures("installed")
@pytest.mark.parametrize("length", [7.0, 2.5])  # longer and shorter than the wheel base
def test_footprint_spans_the_simulated_box(config_dir, length):
    o = _optimizer(config_dir, length=length, width=2.4, center_x=1.2)._optimizer.overrides
    assert o["rear_overhang"] == pytest.approx(length / 2 - 1.2)
    assert o["rear_overhang"] + VEHICLE["wheel_base"] + o["front_overhang"] == pytest.approx(length)
    assert o["left_overhang"] == o["right_overhang"] == pytest.approx((2.4 - 1.6) / 2)


def _solution(steer_rate: float) -> np.ndarray:
    out = np.zeros((80, 7))
    out[:, 4] = steer_rate * po.DT * np.arange(1, 81)  # front wheel angle
    return out


@pytest.mark.usefixtures("installed")
def test_a_failed_solve_keeps_the_tracked_plan(config_dir):
    optimizer = _optimizer(config_dir)
    recorder = optimizer._optimizer
    plan = np.zeros((80, 3))
    recorder.results = [None, _solution(0.2), None, _solution(0.2)]
    assert optimizer.step(plan, (0, 0, 0), 0.0, 0.0, None) is None
    assert optimizer.step(plan, (0, 0, 0), 0.0, 0.1, None).shape == (80, 3)
    assert optimizer.step(plan, (0, 0, 0), 0.0, 0.2, None) is None
    optimizer.step(plan, (0, 0, 0), 0.0, 0.5, None)
    # Steering is read off the last solution, which the failure did not replace.
    assert recorder.steers == pytest.approx([0.0, 0.0, 0.02, 0.08])
    s = optimizer.summary()
    assert (s["calls"], s["failures"], s["failures_without_plan"]) == (4, 2, 1)
    assert s["border_shifted"] is True and s["border_shift_rate"] == 1.0
    assert s["border_shifted_cycles"] == 4 and s["border_shifted_points"] == 12
    assert s["border_max_shift_m"] == 0.25 and s["border_first_shift_s"] == 0.0
    assert s["solve_ms_p50"] == s["solve_ms_max"] == 2.0
