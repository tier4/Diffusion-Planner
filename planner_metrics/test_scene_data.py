import torch

from planner_metrics.scene_data import extract_metric_scene_data


def test_extract_metric_scene_data_keeps_only_known_fields():
    source = {
        "ego_current_state": torch.zeros(1, 4),
        "route_lanes": torch.zeros(1, 2, 2, 4),
        "unrelated_field": torch.zeros(1),
        "scenario_tag": "centerline",
    }

    result = extract_metric_scene_data(source)

    assert set(result) == {"ego_current_state", "route_lanes"}
    assert result["ego_current_state"] is source["ego_current_state"]


def test_extract_metric_scene_data_ignores_missing_fields():
    assert extract_metric_scene_data({}) == {}


def test_extract_metric_scene_data_accepts_any_mapping_source():
    """A non-NPZ source that carries the same keys scores the same way."""

    class DictLikeSource:
        def __init__(self, data):
            self._data = data

        def __contains__(self, key):
            return key in self._data

        def __getitem__(self, key):
            return self._data[key]

    source = DictLikeSource({"ego_agent_future": torch.zeros(1, 8, 3)})

    result = extract_metric_scene_data(source)

    assert set(result) == {"ego_agent_future"}
