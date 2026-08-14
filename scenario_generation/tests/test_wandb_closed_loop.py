"""Unit tests for wandb_closed_loop's per-site bar chart, vehicle_type column and rollup."""

from __future__ import annotations

import json
from pathlib import Path

from scenario_generation.site_discovery import discover_sites_with_vehicles_from_json
from scenario_generation.wandb_closed_loop import (
    EPISODE_TABLE_COLUMNS,
    build_combined_episode_table,
    build_full_closed_loop_wandb_log,
    build_sites_aggregate_log,
    build_sites_score_bar_charts,
)


def _summary(collisions=0, curb_hits=0, snaps=0, red_light=0, strong_brake=0, completion=1.0):
    return {
        "mean_route_completion": completion,
        "object": {"collision_count": collisions},
        "road_border": {"collision_count": curb_hits},
        "reproducer": {"snap_count": snaps},
        "red_light_violation": {"count": red_light},
        "strong_brake": {"count": strong_brake},
    }


def test_build_sites_score_bar_charts_one_chart_per_metric_with_all_sites():
    """Every SCORE_KEY gets its own chart; each chart has one bar per site."""
    summaries = {
        "site_a": _summary(collisions=2, curb_hits=1),
        "site_b": _summary(collisions=0, curb_hits=3),
    }

    log = build_sites_score_bar_charts(summaries)

    assert set(log.keys()) == {
        "closed_loop_scores_bar/mean_route_completion",
        "closed_loop_scores_bar/total_curb_hits",
        "closed_loop_scores_bar/total_snaps",
        "closed_loop_scores_bar/total_red_light_violations",
        "closed_loop_scores_bar/total_strong_brakes",
        "closed_loop_scores_bar/total_collision_events",
    }
    curb_hits_rows = log["closed_loop_scores_bar/total_curb_hits"].table.data
    assert dict(curb_hits_rows) == {"site_a": 1, "site_b": 3}
    collision_rows = log["closed_loop_scores_bar/total_collision_events"].table.data
    assert dict(collision_rows) == {"site_a": 2, "site_b": 0}


def test_build_sites_score_bar_charts_excludes_noobj_from_collision_events():
    """The empty-world ablation (__noobj) is always a meaningless 0 for collision events."""
    summaries = {
        "site_a": _summary(collisions=2),
        "site_a__noobj": _summary(collisions=0),
    }

    log = build_sites_score_bar_charts(summaries)

    collision_rows = dict(log["closed_loop_scores_bar/total_collision_events"].table.data)
    assert collision_rows == {"site_a": 2}
    # Comparison keys (not objects-only) still include the noobj label.
    completion_rows = dict(log["closed_loop_scores_bar/mean_route_completion"].table.data)
    assert set(completion_rows) == {"site_a", "site_a__noobj"}


def test_build_sites_score_bar_charts_empty_summaries_returns_empty_log():
    assert build_sites_score_bar_charts({}) == {}


def test_include_score_scalars_false_omits_per_site_score_keys():
    """Sites that only run once per training run skip the (now single-point) scalar trend."""
    log = build_full_closed_loop_wandb_log(
        _summary(collisions=1, curb_hits=2),
        site="site_a",
        render_media=False,
        include_score_scalars=False,
    )

    assert not any(key.startswith("closed_loop_scores/") for key in log)


def test_include_score_scalars_true_by_default_keeps_per_site_score_keys():
    """main (closed_loop_npz_root) still runs every cadence call -- its trend stays intact."""
    log = build_full_closed_loop_wandb_log(
        _summary(collisions=1, curb_hits=2), site="main", render_media=False
    )

    assert log["closed_loop_scores/total_collision_events/main"] == 1
    assert log["closed_loop_scores/total_curb_hits/main"] == 2


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
