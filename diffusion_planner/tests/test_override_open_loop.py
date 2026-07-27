import json

import pytest
from diffusion_planner.override_validation.metrics import METRICS
from diffusion_planner.override_validation.open_loop import load_override_open_loop_settings


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_override_open_loop_settings(tmp_path):
    list_path = tmp_path / "list.json"
    config_path = tmp_path / "config.json"
    _write_json(
        list_path,
        {
            "centerline": ["/data/centerline.npz"],
            "departure": ["/data/departure.npz"],
        },
    )
    _write_json(
        config_path,
        {
            "interval_epochs": 2,
            "metrics": {"centerline": {}, "departure": {}},
        },
    )

    metric_lists, config = load_override_open_loop_settings(str(list_path), str(config_path))

    assert metric_lists["centerline"] == ["/data/centerline.npz"]
    assert metric_lists["departure"] == ["/data/departure.npz"]
    assert config["interval_epochs"] == 2


def test_override_open_loop_rejects_unknown_metric(tmp_path):
    list_path = tmp_path / "list.json"
    config_path = tmp_path / "config.json"
    _write_json(list_path, {"unknown": ["/data/scene.npz"]})
    _write_json(config_path, {"metrics": {}})

    with pytest.raises(ValueError, match="Unsupported"):
        load_override_open_loop_settings(str(list_path), str(config_path))


def test_override_metric_registry_has_initial_metrics():
    assert set(METRICS) == {"centerline", "departure"}
