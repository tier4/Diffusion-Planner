from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPECTED_FEATURE_NAMES = [
    "ego_speed",
    "ego_accel",
    "ego_yaw_rate",
    "ego_speed_past_mean",
    "ego_speed_past_std",
    "ego_speed_past_max",
    "ego_accel_past_mean",
    "ego_accel_past_std",
    "travel_distance",
    "endpoint_displacement",
    "heading_change_deg",
    "max_curvature",
    "path_straightness",
    "n_active_neighbors",
    "closest_neighbor_dist",
    "neighbor_vehicle_ratio",
    "neighbor_ped_ratio",
    "neighbor_bike_ratio",
    "mean_lane_curvature",
    "n_traffic_light_segments",
    "speed_limit_mean",
    "speed_limit_std",
    "goal_distance",
    "route_curvature",
    "route_length",
]


def test_extract_features_returns_dict(make_npz):
    from dataset_curation.features import extract_features

    npz_path = make_npz(ego_speed=10.0, heading_change_deg=30.0, n_neighbors=5)
    result = extract_features(str(npz_path))
    assert isinstance(result, dict)
    for name in EXPECTED_FEATURE_NAMES:
        assert name in result, f"Missing feature: {name}"
    for v in result.values():
        assert isinstance(v, float), f"Feature value should be float, got {type(v)}"


def test_extract_features_speed_matches_input(make_npz):
    from dataset_curation.features import extract_features

    npz_path = make_npz(ego_speed=12.0, heading_change_deg=0.0)
    result = extract_features(str(npz_path))
    assert abs(result["ego_speed"] - 12.0) < 1.0


def test_extract_features_heading_change(make_npz):
    from dataset_curation.features import extract_features

    straight = extract_features(str(make_npz(heading_change_deg=0.0)))
    turning = extract_features(str(make_npz(heading_change_deg=45.0)))
    assert turning["heading_change_deg"] > straight["heading_change_deg"]


def test_extract_features_neighbor_count(make_npz):
    from dataset_curation.features import extract_features

    few = extract_features(str(make_npz(n_neighbors=2)))
    many = extract_features(str(make_npz(n_neighbors=10)))
    assert few["n_active_neighbors"] < many["n_active_neighbors"]


def test_extract_features_traffic_light(make_npz):
    from dataset_curation.features import extract_features

    no_tl = extract_features(str(make_npz(has_traffic_light=False)))
    with_tl = extract_features(str(make_npz(has_traffic_light=True)))
    assert no_tl["n_traffic_light_segments"] == 0.0
    assert with_tl["n_traffic_light_segments"] > 0.0


def test_extract_features_batch(make_npz, tmp_path):
    from dataset_curation.features import extract_features_batch

    paths = [str(make_npz(ego_speed=5.0 + i)) for i in range(5)]
    json_path = tmp_path / "list.json"
    json_path.write_text(json.dumps(paths))

    df = extract_features_batch(paths, n_workers=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert set(EXPECTED_FEATURE_NAMES).issubset(set(df.columns))
    assert list(df.index) == paths


def test_extract_features_handles_corrupt_npz(tmp_path):
    from dataset_curation.features import extract_features

    bad_path = tmp_path / "bad.npz"
    bad_path.write_bytes(b"not a npz")
    with pytest.raises(Exception):
        extract_features(str(bad_path))
