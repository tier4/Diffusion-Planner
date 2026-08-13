"""Unit tests for wandb_closed_loop's vehicle_type column and per-vehicle rollup."""

from __future__ import annotations

import json
from pathlib import Path

from scenario_generation.site_discovery import discover_sites_with_vehicles_from_json
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


def test_no_project_vehicle_map_produces_no_per_vehicle_rollup(tmp_path: Path):
    """Driven through the real caller shape: no map in, no per-vehicle W&B keys out.

    ``test_sites_aggregate_log_without_vehicle_types_has_no_per_vehicle_keys`` above omits
    ``site_vehicle_types`` entirely, which the production callers never do -- they build it
    from ``discover_sites_with_vehicles_from_json`` (see ``closed_loop_validate`` in train.py
    and ``_log_to_wandb`` in run_all_sites_closed_loop.py). Unlike the HTML report, these key
    names cannot be corrected after a run has logged them.
    """
    path_list = tmp_path / "path_list.json"
    path_list.write_text(
        json.dumps(
            [
                "/data/proj_a/site_1/manual/2026-01-01/10-00-00",
                "/data/proj_b/site_2/manual/2026-01-01/10-00-00",
            ]
        )
    )
    sites = discover_sites_with_vehicles_from_json(path_list)  # no --project_vehicle_map
    site_vehicle_types = {
        name: info["vehicle_type"] for name, info in sites.items() if info["vehicle_type"]
    }
    summaries = {name: {"n_segments": 1, "mean_route_completion": 0.9} for name in sites}

    log = build_sites_aggregate_log(summaries, site_vehicle_types)
    per_vehicle = sorted(k for k in log if k.startswith("closed_loop_overview_by_vehicle/"))
    assert not per_vehicle, f"per-vehicle rollup logged without a map: {per_vehicle[:4]}"
    assert log["closed_loop_overview/n_sites"] == 2  # the overall rollup is unaffected
