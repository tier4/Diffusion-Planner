import json

import pytest
from diffusion_planner.override_validation.open_loop import (
    METRICS,
    _metric_parameters_from_args,
    load_override_open_loop_settings,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_override_open_loop_settings(tmp_path):
    list_path = tmp_path / "list.json"
    _write_json(
        list_path,
        {
            "centerline": ["/data/centerline.npz"],
            "departure": ["/data/departure.npz"],
        },
    )
    metric_lists = load_override_open_loop_settings(str(list_path))

    assert metric_lists["centerline"] == ["/data/centerline.npz"]
    assert metric_lists["departure"] == ["/data/departure.npz"]


def test_override_open_loop_rejects_unknown_metric(tmp_path):
    list_path = tmp_path / "list.json"
    _write_json(list_path, {"unknown": ["/data/scene.npz"]})

    with pytest.raises(ValueError, match="Unsupported"):
        load_override_open_loop_settings(str(list_path))


def test_override_metric_registry_has_initial_metrics():
    assert set(METRICS) == {"centerline", "departure"}


def test_metric_parameters_are_derived_from_train_config_field_names():
    class Args:
        def __init__(self):
            self.override_centerline_horizon_seconds = 8.0
            self.override_departure_horizon_seconds = 3.0
            self.override_departure_minimum_displacement_m = 2.0

    assert _metric_parameters_from_args(Args()) == {
        "centerline": {"horizon_seconds": 8.0},
        "departure": {"horizon_seconds": 3.0, "minimum_displacement_m": 2.0},
    }
