import dataclasses

import numpy as np
import pytest
from diffusion_planner.config.scenario_open_loop_config import ScenarioOpenLoopConfig

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.geometry import _horizon_s
from scenario_generation.scenario_metrics.progress import DepartureParams, YieldParams
from scenario_generation.scenario_metrics.shared_config import open_loop_parameters
from scenario_generation.scenario_metrics.stop_arrival import ArrivalParams, StopParams
from scenario_generation.scenario_metrics.testing import (
    make_input,
    speed_profile_path,
    straight_path,
)

ANCHOR = 10


def _config(**overrides) -> ScenarioOpenLoopConfig:
    return dataclasses.replace(ScenarioOpenLoopConfig(), **overrides)


def test_defaults_match_the_former_closed_loop_constants():
    for label in ("departure", "traffic_light_go"):
        assert DepartureParams.from_config(label) == DepartureParams(3.0, 2.0)
    for label in ("pedestrian_yield", "vehicle_yield", "temporal_stop"):
        assert YieldParams.from_config(label) == YieldParams(3.0, 0.5)
    for label in ("traffic_light_stop", "obstacle_stop"):
        assert StopParams.from_config(label) == StopParams(0.5, 0.5, 0.5)
    assert ArrivalParams.from_config() == ArrivalParams(2.0, 10.0, 5.0)
    for label in ("simple_turn", "centerline", "lane_change"):
        assert _horizon_s(label, None) == 8.0
    lane_change = open_loop_parameters("lane_change")
    assert lane_change["minimum_lateral_shift_m"] == 1.0
    assert lane_change["chain_tolerance_m"] == 1.0


def _departure_input(label: str):
    # Stands still until the anchor, then 1 m/s: 3 m of progress over the 3 s horizon.
    ego_xy, ego_yaw = speed_profile_path(np.r_[np.zeros(ANCHOR), np.full(40, 1.0)])
    rec_xy, rec_yaw = straight_path(80, 2.0)
    return make_input(
        label=label,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        terminated="max_steps",
    )


def test_departure_threshold_follows_the_open_loop_config():
    config = _config(scenario_departure_minimum_displacement_m=4.0)
    assert registry.score(_departure_input("departure")).passed is True
    r = registry.score(_departure_input("departure"), config)
    assert r.passed is False and r.values["threshold_m"] == 4.0
    # Grouped per label: traffic_light_go keeps its own threshold.
    assert registry.score(_departure_input("traffic_light_go"), config).passed is True


def test_stop_tolerance_follows_the_open_loop_config():
    # Human stops at 25 m; the ego stops 3 m past it.
    rec_xy, rec_yaw = speed_profile_path(np.r_[np.full(50, 5.0), np.zeros(100)])
    ego_xy, ego_yaw = speed_profile_path(np.r_[np.full(56, 5.0), np.zeros(100)])
    inp = make_input(
        label="traffic_light_stop",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
    )
    assert registry.score(inp).passed is False
    r = registry.score(inp, _config(scenario_traffic_light_stop_tolerance_m=3.5))
    assert r.passed is True and r.values["tolerance_m"] == 3.5


def test_simple_turn_horizon_follows_the_open_loop_config():
    rec_xy, rec_yaw = straight_path(200, 5.0)
    inp = make_input(
        label="simple_turn",
        ego_xy=rec_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
    )
    assert registry.score(inp).values["eval_interval_s"] == pytest.approx(8.0)
    r = registry.score(inp, _config(scenario_simple_turn_horizon_seconds=4.0))
    assert r.values["eval_interval_s"] == pytest.approx(4.0)
