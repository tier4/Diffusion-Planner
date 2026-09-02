"""Regression tests for the token-analysis helper functions."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    script_path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_neighbor_distance_ignores_history_discarded_by_encoder():
    token_importance = _load_script("token_importance")
    nbr = torch.zeros(1, 2, 10, 11)
    nbr[0, 0, 0, 0] = 3.0  # outside NeighborEncoder's retained window
    nbr[0, 1, -1, :2] = torch.tensor([3.0, 4.0])

    dist = token_importance._neighbor_dist(nbr)

    assert torch.isinf(dist[0, 0])
    assert dist[0, 1].item() == 5.0


def test_onnx_model_returns_prediction_on_requested_device():
    token_importance = _load_script("token_importance")
    model = object.__new__(token_importance.OnnxModel)
    model.device = "cpu"
    model.input_specs = []

    class Session:
        @staticmethod
        def run(_names, _feed):
            return [np.zeros((2, 1, 3, 4), dtype=np.float32)]

    model.sess = Session()
    _, outputs = model({"ego_current_state": torch.zeros(2, 10)})

    assert outputs["prediction"].device.type == "cpu"


def test_token_occupancy_statistics_include_capacity_and_utilization():
    attention_analysis = _load_script("attention_analysis")

    stats = attention_analysis.occupancy_stats([0, 1, 2, 4], slots=8)

    assert stats["mean"] == 1.75
    assert stats["max"] == 4
    assert stats["mean_utilization_pct"] == 21.875
    assert stats["p95_capacity"] == 4
    assert stats["p99_capacity"] == 4


def test_metric_totals_are_converted_to_sample_weighted_means():
    token_importance = _load_script("token_importance")
    totals = {
        "fde_top": 9.0,
        "ade_top": 6.0,
        "min_fde": 3.0,
        "min_ade": 1.5,
    }

    metrics = token_importance.reduce_metric_totals(totals, count=3, device="cpu", world_size=1)

    assert metrics == {
        "fde_top": 3.0,
        "ade_top": 2.0,
        "min_fde": 1.0,
        "min_ade": 0.5,
    }


def test_attention_payload_merge_preserves_rank_local_alignment():
    attention_analysis = _load_script("attention_analysis")

    def payload(value, turning):
        return {
            "turning": [turning],
            "ego_share": [{name: [value] for name, _ in attention_analysis.CLASSES}],
            "all_share": [{name: [value + 1] for name, _ in attention_analysis.CLASSES}],
            "vw_share": [{name: [value + 2] for name, _ in attention_analysis.CLASSES}],
            "count_share": {name: [value + 3] for name, _ in attention_analysis.CLASSES},
            "valid_count": {name: [value + 4] for name, _ in attention_analysis.CLASSES},
            "total_valid_count": [value + 5],
            "lane_bin_share": [[value] for _ in attention_analysis.DIST_BINS],
            "nbr_bin_share": [[value + 1] for _ in attention_analysis.DIST_BINS],
            "route_share_per_sample": [value + 6],
        }

    merged = attention_analysis.merge_payloads([payload(10, False), payload(20, True)], n_layers=1)

    assert merged["turning"] == [False, True]
    assert merged["route_share_per_sample"] == [16, 26]
    assert merged["ego_share"][0]["neighbors"] == [10, 20]
    assert merged["valid_count"]["lanes"] == [14, 24]


def test_all_token_visualization_layout_and_non_spatial_records():
    visualization = _load_script("visualize_all_token_attention")
    sample = {
        "neighbor_agents_past": np.zeros((320, 31, 11), dtype=np.float32),
        "static_objects": np.zeros((5, 10), dtype=np.float32),
        "lanes": np.zeros((140, 20, 33), dtype=np.float32),
        "route_lanes": np.zeros((25, 20, 33), dtype=np.float32),
        "polygons": np.zeros((10, 40, 3), dtype=np.float32),
        "line_strings": np.zeros((60, 20, 4), dtype=np.float32),
        "goal_pose": np.array([3.0, 4.0, 0.0], dtype=np.float32),
    }
    layout = visualization.token_layout(sample)
    scores = np.zeros(564, dtype=np.float32)
    valid = np.zeros(564, dtype=bool)
    selected = [0, 1, 561, 562, 563]
    valid[selected] = True
    scores[selected] = 0.2

    records = visualization.token_records(sample, scores, valid)

    assert layout[-1] == ("turn_indicator", 563, 564)
    assert [record["class"] for record in records] == [
        "ego",
        "neighbors",
        "goal_pose",
        "ego_shape",
        "turn_indicator",
    ]
    goal = next(record for record in records if record["class"] == "goal_pose")
    assert goal["distance_m"] == 5.0
    assert not next(record for record in records if record["class"] == "ego_shape")["spatial"]
