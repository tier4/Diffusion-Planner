"""Unit tests for closed-loop step scorers and nested aggregate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scenario_generation.closed_loop_eval import aggregate, metrics_for_json
from scenario_generation.metrics.ego_traj import ego_traj_ego_frame
from scenario_generation.metrics.red_light import score_red_light_step
from scenario_generation.metrics.road_border import score_road_border_step
from scenario_generation.metrics.strong_brake import strong_brake_mask
from scenario_generation.reproducer_rollout import _clearance_stats, _event_count


def _default_hist(live: np.ndarray) -> np.ndarray:
    hist = np.zeros((31, 3), dtype=np.float64)
    hist[-1] = live
    hist[-2] = live + np.array([-0.2, 0.0, 0.0])
    return hist


def test_clearance_stats_p5():
    stats = _clearance_stats(np.arange(1, 101, dtype=np.float32))
    assert stats["clearance_min_m"] == 1.0
    assert abs(stats["clearance_mean_m"] - 50.5) < 1e-5
    assert abs(stats["clearance_p5_m"] - 5.95) < 0.1
    assert stats["clearance_finite_steps"] == 100
    assert "_tdigest" in stats
    empty = _clearance_stats(np.full(4, np.inf, dtype=np.float32))
    assert empty["clearance_finite_steps"] == 0
    assert empty["clearance_mean_m"] == float("inf")


def test_pool_clearance_weights_by_finite_steps():
    """Pooled mean must weight by finite clearance samples, not n_steps_run."""
    from scenario_generation.metrics.tdigest import TDIGEST_KEY, tdigest_dict_from_values

    dense = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    sparse = np.array([10.0], dtype=np.float32)
    row_dense = _segment_row(
        route="dense",
        n_steps_run=100,
        object={
            "miss_thresh_m": 0.5,
            "collision_steps": 0,
            "collision_count": 0,
            "miss_steps": 0,
            "miss_count": 0,
            "clearance_min_m": float(dense.min()),
            "clearance_mean_m": float(dense.mean()),
            "clearance_p5_m": float(np.percentile(dense, 5)),
            "clearance_finite_steps": int(dense.size),
            TDIGEST_KEY: tdigest_dict_from_values(dense),
        },
    )
    row_sparse = _segment_row(
        route="sparse",
        n_steps_run=100,
        object={
            "miss_thresh_m": 0.5,
            "collision_steps": 0,
            "collision_count": 0,
            "miss_steps": 0,
            "miss_count": 0,
            "clearance_min_m": float(sparse.min()),
            "clearance_mean_m": float(sparse.mean()),
            "clearance_p5_m": float(np.percentile(sparse, 5)),
            "clearance_finite_steps": int(sparse.size),
            TDIGEST_KEY: tdigest_dict_from_values(sparse),
        },
    )
    summary = aggregate([row_dense, row_sparse], near_miss_thresh=0.5, strong_brake_mps2=-2.5)
    # (1*4 + 10*1) / 5 = 2.8 — not the equal-weight 5.5 from n_steps_run.
    assert abs(summary["object"]["clearance_mean_m"] - 2.8) < 1e-6


def test_ego_traj_ego_frame_uses_real_hist():
    live = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    hist = np.zeros((31, 3), dtype=np.float64)
    hist[-1] = live
    hist[-2] = np.array([9.5, 0.5, 0.2], dtype=np.float64)  # not on -x from speed alone
    traj = ego_traj_ego_frame(live, hist, n_steps=2, device="cpu")
    assert traj.shape == (1, 2, 4)
    assert float(traj[0, 1, 0]) == 0.0
    assert float(traj[0, 1, 1]) == 0.0
    # Past pose relative to live: dx=-0.5, dy=0.5 (yaw=0 → ego frame same)
    assert abs(float(traj[0, 0, 0]) - (-0.5)) < 1e-5
    assert abs(float(traj[0, 0, 1]) - 0.5) < 1e-5
    # Must NOT be the old speed*dt synthetic trail (-0.2 on -x only).
    assert abs(float(traj[0, 0, 0]) - (-0.2)) > 0.1


def test_red_light_step_detects_nearby_red():
    rl = np.zeros((1, 4, 33), dtype=np.float32)
    rl[0, 0, 0] = 1.0
    rl[0, 0, 2] = 1.0  # dx
    rl[0, 0, 10] = 1.0  # red
    live = np.zeros(3, dtype=np.float64)
    hist = _default_hist(live)
    assert score_red_light_step(
        {"route_lanes": rl},
        device="cpu",
        ego_speed_mps=2.0,
        live_pose=live,
        ego_hist=hist,
    )["red_light_violation"]
    assert not score_red_light_step(
        {"route_lanes": rl},
        device="cpu",
        ego_speed_mps=0.1,
        live_pose=live,
        ego_hist=hist,
    )["red_light_violation"]


def test_red_light_step_ignores_white():
    rl = np.zeros((1, 2, 33), dtype=np.float32)
    rl[0, 0, 0] = 1.0
    rl[0, 0, 2] = 1.0
    rl[0, 0, 11] = 1.0  # white
    live = np.zeros(3, dtype=np.float64)
    assert not score_red_light_step(
        {"route_lanes": rl},
        device="cpu",
        ego_speed_mps=2.0,
        live_pose=live,
        ego_hist=_default_hist(live),
    )["red_light_violation"]


def test_road_border_step_no_borders_is_inf():
    ls = np.zeros((2, 4, 4), dtype=np.float32)
    out = score_road_border_step(
        {"line_strings": ls, "ego_shape": np.array([2.7, 4.0, 2.0], dtype=np.float32)},
        device="cpu",
    )
    assert out["rb_dist_m"] == float("inf") or out["rb_dist_m"] > 1e3
    assert set(out.keys()) == {"rb_dist_m"}


def test_road_border_step_near_border_counts_miss():
    ls = np.zeros((1, 4, 4), dtype=np.float32)
    ls[0, 0, :2] = (-2.0, 0.05)
    ls[0, 1, :2] = (2.0, 0.05)
    ls[0, 0, 3] = 1.0
    ls[0, 1, 3] = 1.0
    out = score_road_border_step(
        {"line_strings": ls, "ego_shape": np.array([2.7, 4.0, 2.0], dtype=np.float32)},
        device="cpu",
    )
    assert np.isfinite(out["rb_dist_m"])
    assert out["rb_dist_m"] < 0.3


def test_strong_brake_mask():
    # Isolated spike (-5) does not count; only the 2nd frame of a consecutive pair does.
    # Uses an explicit thresh (not the default) so the fixture values stay readable.
    mask = strong_brake_mask(
        np.array([0.0, -5.0, -4.0, -3.9, -6.0, -5.0], dtype=np.float32),
        thresh_mps2=-4.0,
    )
    assert mask.tolist() == [False, False, True, False, False, True]


def _segment_row(**overrides) -> dict:
    from scenario_generation.metrics.tdigest import TDIGEST_KEY, tdigest_dict_from_values

    obj_vals = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    rb_vals = np.array([0.2, 0.8, 1.0, 2.0], dtype=np.float32)
    row = {
        "route": "a",
        "n_steps_run": 4,
        "terminated": "goal",
        "object": {
            "miss_thresh_m": 0.5,
            "collision_steps": 1,
            "collision_count": 1,
            "miss_steps": 2,
            "miss_count": 1,
            "clearance_min_m": 0.1,
            "clearance_mean_m": float(obj_vals.mean()),
            "clearance_p5_m": float(np.percentile(obj_vals, 5)),
            "clearance_finite_steps": int(obj_vals.size),
            TDIGEST_KEY: tdigest_dict_from_values(obj_vals),
        },
        "road_border": {
            "miss_thresh_m": 0.5,
            "collision_steps": 0,
            "collision_count": 0,
            "miss_steps": 1,
            "miss_count": 1,
            "clearance_min_m": 0.2,
            "clearance_mean_m": float(rb_vals.mean()),
            "clearance_p5_m": float(np.percentile(rb_vals, 5)),
            "clearance_finite_steps": int(rb_vals.size),
            TDIGEST_KEY: tdigest_dict_from_values(rb_vals),
        },
        "red_light_violation": {"steps": 1, "count": 1},
        "strong_brake": {
            "thresh_mps2": -2.5,
            "strongest_mps2": float("inf"),
            "steps": 0,
            "count": 0,
        },
        "reproducer": {
            "expand_count": 1,
            "snap_count": 0,
            "normal_steps": 3,
            "repeat_steps": 1,
        },
    }
    row.update(overrides)
    return row


def test_aggregate_nested_keeps_tdigest_in_json():
    from scenario_generation.metrics.tdigest import TDIGEST_KEY

    rows = [_segment_row()]
    summary = aggregate(rows, near_miss_thresh=0.5, strong_brake_mps2=-2.5)
    assert summary["object"]["collision_steps"] == 1
    assert summary["object"]["miss_count"] == 1
    assert summary["object"]["miss_thresh_m"] == 0.5
    assert abs(summary["object"]["clearance_min_m"] - 0.1) < 1e-6
    assert abs(summary["object"]["clearance_mean_m"] - 0.9) < 1e-5
    assert np.isfinite(summary["object"]["clearance_p5_m"])
    assert summary["road_border"]["miss_steps"] == 1
    assert summary["road_border"]["miss_thresh_m"] == 0.5
    assert summary["red_light_violation"]["count"] == 1
    assert summary["reproducer"]["expand_count"] == 1
    assert abs(summary["reproducer"]["repeat_step_rate"] - 0.25) < 1e-9
    assert summary["strong_brake"]["strongest_mps2"] == float("inf")
    # Human-readable segments.jsonl strips digests; in-memory rows keep them.
    cleaned = metrics_for_json(rows[0])
    assert TDIGEST_KEY not in cleaned["object"]
    assert TDIGEST_KEY in rows[0]["object"]


def test_tdigest_sidecar_roundtrip(tmp_path):
    """Digests leave segments.jsonl but survive via tdigests sidecar for shard merge."""
    from scenario_generation.closed_loop_eval import (
        attach_tdigest_sidecars,
        load_segment_rows_with_tdigests,
        metrics_for_json,
        tdigest_sidecar_row,
    )
    from scenario_generation.metrics.tdigest import TDIGEST_KEY

    row = _segment_row()
    row["route"] = "r0"
    human = metrics_for_json(row)
    assert TDIGEST_KEY not in human["object"]
    side = tdigest_sidecar_row(row)
    assert side is not None and side["route"] == "r0"
    assert "object" in side and "road_border" in side

    (tmp_path / "segments_0.jsonl").write_text(json.dumps(human) + "\n")
    (tmp_path / "tdigests_0.jsonl").write_text(json.dumps(side) + "\n")
    loaded = load_segment_rows_with_tdigests(tmp_path)
    assert len(loaded) == 1
    assert TDIGEST_KEY in loaded[0]["object"]
    assert loaded[0]["object"][TDIGEST_KEY] == row["object"][TDIGEST_KEY]

    # attach helper is idempotent on already-stripped rows
    stripped = [human]
    attach_tdigest_sidecars(stripped, [side])
    assert stripped[0]["object"][TDIGEST_KEY] == row["object"][TDIGEST_KEY]


def test_aggregate_missing_category_fails():
    bad = _segment_row()
    del bad["object"]
    with pytest.raises(KeyError, match="object"):
        aggregate([bad], near_miss_thresh=0.5, strong_brake_mps2=-2.5)


def test_merge_shards_defers_summary_write(tmp_path):
    """Multi-GPU parent must add elapsed_sec before writing summary.json."""
    import importlib.util
    from pathlib import Path

    from scenario_generation.closed_loop_eval import metrics_for_json, tdigest_sidecar_row

    script = (
        Path(__file__).resolve().parents[2] / "diffusion_planner" / "valid_predictor_closed_loop.py"
    )
    spec = importlib.util.spec_from_file_location("valid_predictor_closed_loop", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    row = _segment_row()
    row["route"] = "r0"
    (tmp_path / "segments_0.jsonl").write_text(json.dumps(metrics_for_json(row)) + "\n")
    side = tdigest_sidecar_row(row)
    assert side is not None
    (tmp_path / "tdigests_0.jsonl").write_text(json.dumps(side) + "\n")
    summary = mod._merge_shards(tmp_path, tmp_path, 0.5, strong_brake_mps2=-2.5)
    assert not (tmp_path / "summary.json").exists()
    summary["elapsed_sec"] = 1.25
    summary["model_path"] = "/tmp/model.pth"
    mod._write_summary(tmp_path, summary)
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["elapsed_sec"] == 1.25
    assert written["model_path"] == "/tmp/model.pth"
    assert written["strong_brake"]["thresh_mps2"] == -2.5
    assert np.isfinite(written["object"]["clearance_p5_m"])


def test_segment_row_for_json_strips_tdigest():
    from scenario_generation.closed_loop_eval import segment_row_for_json
    from scenario_generation.metrics.tdigest import TDIGEST_KEY

    row = segment_row_for_json(_segment_row(), route="r0", chunk_id="c1")
    assert row["route"] == "r0"
    assert row["chunk_id"] == "c1"
    assert TDIGEST_KEY not in row["object"]
    assert TDIGEST_KEY not in row["road_border"]


def test_event_count_debounce_unchanged():
    assert _event_count(np.array([1, 0, 0, 0, 1], dtype=bool)) == 2
