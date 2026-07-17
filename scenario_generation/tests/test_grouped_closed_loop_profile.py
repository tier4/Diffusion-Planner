"""Tests for grouped closed-loop profiling and DDP sharding."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scenario_generation.grouped_closed_loop_ddp import (
    merge_ddp_grouped_shards,
    shard_bag_items,
    write_ddp_shard,
)
from scenario_generation.grouped_closed_loop_eval import run_grouped_closed_loop_eval
from scenario_generation.metrics.cl_route_by_area import RouteRolloutResult, StepRecord
from scenario_generation.perf_timer import Timers


def _fake_rollout(bag_name: str, route_key: str) -> RouteRolloutResult:
    timers = Timers()
    timers.add("draw", 1.5, 2)
    timers.add("model_forward", 0.2, 1)
    return RouteRolloutResult(
        route_key=route_key,
        sequence=bag_name,
        steps=[
            StepRecord(k=0, rec_idx=0, area="area_a", clearance_m=5.0),
        ],
        n_steps_run=1,
        terminated="goal",
        timers=timers,
    )


def _setup_fixture(tmp_path: Path) -> tuple[Path, Path]:
    doc = {
        "areas": {"area_a": {"metric_group": "straight"}},
        "time_series": {
            "bag_a": {
                "route_key": "route_a",
                "episodes": [
                    {
                        "area": "area_a",
                        "labeled_ranges": [[0, 1]],
                        "video_start_idx": 0,
                        "video_end_idx": 1,
                    }
                ],
            }
        },
    }
    classification_json = tmp_path / "2026-01-15.json"
    classification_json.write_text(json.dumps(doc), encoding="utf-8")
    npz_root = tmp_path / "npz"
    bag_dir = npz_root / "bag_a"
    bag_dir.mkdir(parents=True)
    for i in range(2):
        np.savez(
            bag_dir / f"frame_{i:05d}.npz",
            ego_agent_past=np.zeros((31, 4), dtype=np.float32),
            ego_current_state=np.zeros(10, dtype=np.float32),
            neighbor_agents_past=np.zeros((320, 31, 11), dtype=np.float32),
            lanes=np.zeros((140, 20, 33), dtype=np.float32),
            lanes_speed_limit=np.zeros((140, 1), dtype=np.float32),
            lanes_has_speed_limit=np.zeros((140, 1), dtype=bool),
            route_lanes=np.zeros((25, 20, 33), dtype=np.float32),
            route_lanes_speed_limit=np.zeros((25, 1), dtype=np.float32),
            route_lanes_has_speed_limit=np.zeros((25, 1), dtype=bool),
            polygons=np.zeros((10, 40, 3), dtype=np.float32),
            line_strings=np.zeros((60, 20, 4), dtype=np.float32),
            static_objects=np.zeros((5, 10), dtype=np.float32),
            ego_shape=np.array([4.5, 1.8, 2.9], dtype=np.float32),
            turn_indicators=np.zeros(31, dtype=np.int64),
            goal_pose=np.array([10.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )
    return npz_root, classification_json


def test_shard_bag_items() -> None:
    items = [(f"bag_{i}", {}) for i in range(4)]
    assert [b[0] for b in shard_bag_items(items, 0, 4)] == ["bag_0"]
    assert [b[0] for b in shard_bag_items(items, 1, 4)] == ["bag_1"]
    assert len(shard_bag_items(items, 0, 2)) == 2


def test_grouped_eval_writes_timing_json_sequential(tmp_path: Path) -> None:
    npz_root, classification_json = _setup_fixture(tmp_path)
    model = MagicMock()
    model_args = MagicMock()
    rollout = _fake_rollout("bag_a", "route_a")
    bag_result = {
        "bag_name": "bag_a",
        "rollout": rollout,
        "rows": [{"area_name": "area_a", "bag": "bag_a", "n_steps_run": 1}],
        "video_mp4s": [],
        "bag_timing": {
            "bag": "bag_a",
            "n_steps_run": 1,
            "stages": rollout.timers.as_dict(),
        },
    }

    with patch(
        "scenario_generation.grouped_closed_loop_eval.process_grouped_bag",
        return_value=bag_result,
    ):
        summary = run_grouped_closed_loop_eval(
            model,
            model_args,
            npz_root,
            classification_json,
            tmp_path / "out",
            profile=True,
            verbose=False,
        )

    timing_path = tmp_path / "out" / "timing.json"
    assert timing_path.is_file()
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    assert "draw" in timing["stages"]
    assert summary.get("timing") is not None


def test_merge_ddp_grouped_shards(tmp_path: Path) -> None:
    npz_root, classification_json = _setup_fixture(tmp_path)
    out_dir = tmp_path / "out"
    write_ddp_shard(
        out_dir,
        0,
        segments=[{"area_name": "area_a", "bag": "bag_a", "n_steps_run": 1}],
        video_mp4s=[],
        rollout_bags=["bag_a"],
        elapsed_sec=1.0,
        total_steps=10,
        per_bag_timing=[],
        profile=False,
    )
    write_ddp_shard(
        out_dir,
        1,
        segments=[{"area_name": "area_a", "bag": "bag_b", "n_steps_run": 2}],
        video_mp4s=[],
        rollout_bags=["bag_b"],
        elapsed_sec=2.0,
        total_steps=20,
        per_bag_timing=[],
        profile=False,
    )
    summary = merge_ddp_grouped_shards(
        out_dir,
        2,
        classification_json=classification_json,
        npz_root=npz_root,
        near_miss_thresh=0.3,
        profile=False,
    )
    assert summary["n_episodes"] == 2
    assert summary["ddp_world_size"] == 2
    assert (out_dir / "results_table.csv").is_file()
