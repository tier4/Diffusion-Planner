"""Unit tests for wandb_closed_loop's vehicle_type column and per-vehicle rollup."""

from __future__ import annotations

from scenario_generation.wandb_closed_loop import (
    EPISODE_TABLE_COLUMNS,
    build_combined_episode_table,
    build_sites_aggregate_log,
)

_ROW = {"segment": [0, 1], "route": "route_0", "n_steps_run": 1, "route_completion": 0.9}


def test_episode_table_has_vehicle_type_column():
    assert "vehicle_type" in EPISODE_TABLE_COLUMNS


def test_combined_episode_table_fills_vehicle_type():
    table = build_combined_episode_table([("site_a", [_ROW], None, "vehicle_x")])
    vehicle_col = EPISODE_TABLE_COLUMNS.index("vehicle_type")
    assert table.data[0][vehicle_col] == "vehicle_x"


def test_combined_episode_table_backward_compatible_without_vehicle_type():
    table = build_combined_episode_table([("site_a", [_ROW], None)])
    vehicle_col = EPISODE_TABLE_COLUMNS.index("vehicle_type")
    assert table.data[0][vehicle_col] == ""


def test_sites_aggregate_log_adds_per_vehicle_rollup():
    summaries = {
        "site_a": {"n_segments": 1, "mean_route_completion": 0.9},
        "site_b": {"n_segments": 1, "mean_route_completion": 0.5},
    }
    site_vehicle_types = {"site_a": "vehicle_x", "site_b": "vehicle_y"}
    log = build_sites_aggregate_log(summaries, site_vehicle_types)
    assert log["closed_loop_overview_by_vehicle/vehicle_x/n_sites"] == 1
    assert log["closed_loop_overview_by_vehicle/vehicle_y/n_sites"] == 1
    assert log["closed_loop_overview/n_sites"] == 2


def test_sites_aggregate_log_without_vehicle_types_has_no_per_vehicle_keys():
    summaries = {"site_a": {"n_segments": 1, "mean_route_completion": 0.9}}
    log = build_sites_aggregate_log(summaries)
    assert not any(k.startswith("closed_loop_overview_by_vehicle/") for k in log)


def test_noobj_label_shares_base_site_vehicle_type_in_rollup():
    summaries = {
        "site_a": {"n_segments": 1, "mean_route_completion": 0.9},
        "site_a__noobj": {"n_segments": 1, "mean_route_completion": 1.0},
    }
    log = build_sites_aggregate_log(summaries, {"site_a": "vehicle_x"})
    assert log["closed_loop_overview_by_vehicle/vehicle_x/n_sites"] == 2
