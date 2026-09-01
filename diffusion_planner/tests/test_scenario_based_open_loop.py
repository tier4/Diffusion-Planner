"""Tests for open-loop metric-list loading and parameter extraction."""

import json

import numpy as np
import pytest
from diffusion_planner.scenario_based_open_loop.open_loop import (
    METRICS,
    _metric_parameters_from_args,
    load_scenario_based_open_loop_settings,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_scenario_based_open_loop_settings(tmp_path):
    """Load valid metric-to-NPZ lists after checking their files are readable."""
    centerline_path = tmp_path / "centerline.npz"
    departure_path = tmp_path / "departure.npz"
    np.savez(centerline_path)
    np.savez(departure_path)
    list_path = tmp_path / "list.json"
    _write_json(
        list_path,
        {
            "centerline": [str(centerline_path)],
            "departure": [str(departure_path)],
        },
    )
    metric_lists = load_scenario_based_open_loop_settings(str(list_path))

    assert metric_lists["centerline"] == [str(centerline_path)]
    assert metric_lists["departure"] == [str(departure_path)]


def test_load_scenario_based_open_loop_settings_rejects_missing_npz(tmp_path):
    """Reject a metric list that references a missing NPZ file."""
    list_path = tmp_path / "list.json"
    _write_json(list_path, {"centerline": [str(tmp_path / "missing.npz")]})

    with pytest.raises(FileNotFoundError, match="not found"):
        load_scenario_based_open_loop_settings(str(list_path))


def test_scenario_based_open_loop_rejects_unknown_metric(tmp_path):
    """Reject metric names that are not registered in the scorer registry."""
    list_path = tmp_path / "list.json"
    _write_json(list_path, {"unknown": ["/data/scene.npz"]})

    with pytest.raises(ValueError, match="Unsupported"):
        load_scenario_based_open_loop_settings(str(list_path))


def test_scenario_metric_registry_has_initial_metrics():
    """Keep the full registered metric-name set explicit."""
    assert set(METRICS) == {
        "centerline",
        "departure",
        "traffic_light_go",
        "simple_turn",
        "object_avoidance",
        "pedestrian_yield",
        "vehicle_yield",
        "temporal_stop",
        "obstacle_stop",
        "traffic_light_stop",
    }


def test_metric_parameters_are_derived_from_train_config_field_names():
    """Strip metric prefixes when collecting parameters from training args."""

    class Args:
        def __init__(self):
            self.scenario_centerline_horizon_seconds = 8.0
            self.scenario_simple_turn_horizon_seconds = 8.0
            self.scenario_departure_horizon_seconds = 3.0
            self.scenario_departure_minimum_displacement_m = 2.0
            self.scenario_traffic_light_go_horizon_seconds = 3.0
            self.scenario_traffic_light_go_minimum_displacement_m = 2.0
            self.scenario_pedestrian_yield_horizon_seconds = 3.0
            self.scenario_pedestrian_yield_maximum_forward_progress_m = 0.5
            self.scenario_vehicle_yield_horizon_seconds = 3.0
            self.scenario_vehicle_yield_maximum_forward_progress_m = 0.5
            self.scenario_temporal_stop_horizon_seconds = 3.0
            self.scenario_temporal_stop_maximum_forward_progress_m = 0.5
            self.scenario_obstacle_stop_tolerance_m = 0.5
            self.scenario_traffic_light_stop_tolerance_m = 0.5

    assert _metric_parameters_from_args(Args()) == {
        "centerline": {"horizon_seconds": 8.0},
        "simple_turn": {"horizon_seconds": 8.0},
        "departure": {"horizon_seconds": 3.0, "minimum_displacement_m": 2.0},
        "traffic_light_go": {"horizon_seconds": 3.0, "minimum_displacement_m": 2.0},
        "object_avoidance": {},
        "pedestrian_yield": {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5},
        "vehicle_yield": {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5},
        "temporal_stop": {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5},
        "obstacle_stop": {"tolerance_m": 0.5},
        "traffic_light_stop": {"tolerance_m": 0.5},
    }
