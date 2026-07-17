"""Tests for grouped closed-loop metric aggregation."""

from __future__ import annotations

from scenario_generation.metrics.group_report import aggregate_segment_rows
from scenario_generation.wandb_closed_loop import build_grouped_closed_loop_wandb_log


def test_aggregate_segment_rows_totals_and_by_area() -> None:
    rows = [
        {
            "area_name": "area_a",
            "metric_group": "left_turn",
            "n_steps_run": 100,
            "neighbor_violation_steps": 2,
            "rb_violation_steps": 1,
            "collision_steps": 3,
            "n_collision_steps": 3,
            "n_near_miss_steps": 4,
            "turn_match_rate": 0.9,
            "min_clearance_m": 0.5,
        },
        {
            "area_name": "area_a",
            "metric_group": "left_turn",
            "n_steps_run": 50,
            "neighbor_violation_steps": 1,
            "rb_violation_steps": 0,
            "collision_steps": 0,
            "n_collision_steps": 0,
            "n_near_miss_steps": 1,
            "turn_match_rate": 1.0,
            "min_clearance_m": 1.0,
        },
        {
            "area_name": "area_b",
            "metric_group": "right_turn",
            "n_steps_run": 20,
            "neighbor_violation_steps": 0,
            "rb_violation_steps": 0,
            "collision_steps": 1,
            "n_collision_steps": 1,
            "n_near_miss_steps": 0,
            "turn_match_rate": 0.5,
            "min_clearance_m": 2.0,
        },
    ]
    summary = aggregate_segment_rows(rows)

    assert "by_metric_group" not in summary
    assert "n_segments_total" not in summary

    totals = summary["totals"]
    assert totals["n_steps_run"] == 170
    assert totals["neighbor_violation_steps"] == 3
    assert totals["rb_violation_steps"] == 1
    assert totals["collision_steps"] == 4
    assert totals["n_near_miss_steps"] == 5
    assert totals["min_clearance_m"] == 0.5
    # weighted turn match: (0.9*100 + 1.0*50 + 0.5*20) / 170
    assert abs(totals["turn_match_rate"] - (150.0 / 170.0)) < 1e-6

    area_a = summary["by_area_name"]["area_a"]
    assert area_a["n_steps_run"] == 150
    assert area_a["collision_steps"] == 3
    assert "n_segments" not in area_a


def test_build_grouped_closed_loop_wandb_log_shape() -> None:
    summary = {
        "elapsed_sec": 12.5,
        "grouped_summary": {
            "totals": {
                "n_steps_run": 10,
                "collision_steps": 1,
                "turn_match_rate": 0.8,
            },
            "by_area_name": {
                "foo-area": {
                    "n_steps_run": 10,
                    "collision_steps": 1,
                }
            },
        },
        "segments": [
            {
                "area_name": "foo-area",
                "bag": "13-42-45",
                "n_steps_run": 10,
                "collision_steps": 1,
            }
        ],
        "video_mp4s": [],
    }
    log = build_grouped_closed_loop_wandb_log(summary)

    assert log["closed_loop/grouped/total/n_steps_run"] == 10
    assert log["closed_loop/grouped/total/collision_steps"] == 1
    assert log["closed_loop/grouped/area/foo-area/collision_steps"] == 1
    assert "closed_loop/grouped/n_videos" not in log
    assert "closed_loop/grouped/n_episodes" not in log
    assert not any(k.startswith("closed_loop/grouped/metric_group/") for k in log)
