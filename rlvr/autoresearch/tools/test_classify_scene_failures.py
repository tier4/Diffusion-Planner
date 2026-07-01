from __future__ import annotations

import json

import numpy as np
import torch

from planner_metrics.aggregate import compute_subscores_batch, compute_subscores_scene_batch
from rlvr.autoresearch.tools.classify_scene_failures import (
    _DEFAULT_THRESHOLD_CONFIG,
    _apply_scene_thresholds,
    _load_npz_data,
    _load_scene_thresholds,
    _merge_output_dirs,
    _prediction_path_for_scene,
    _prepare_scoring_data,
    _saved_prediction_trajectory,
    _stack_scene_data,
    _write_outputs,
    classify_loaded_scene,
    classify_loaded_scenes_batch,
)
from rlvr.reward import RewardConfig

T = 80


def _ego_shape() -> torch.Tensor:
    return torch.tensor([[0.0, 2.0, 2.0]], dtype=torch.float32)


def _moving_collision_data_3col() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = torch.arange(T, dtype=torch.float32)
    ego = torch.stack(
        [
            1.0 + 0.2 * t,
            torch.zeros(T),
            torch.ones(T),
            torch.zeros(T),
        ],
        dim=-1,
    ).unsqueeze(0)

    neighbor = torch.zeros(1, 1, T, 3)
    neighbor[..., 0] = 100.0
    neighbor[..., 2] = 0.0
    neighbor[0, 0, 30, 0] = ego[0, 30, 0]
    neighbor[0, 0, 30, 1] = 0.0

    past = torch.zeros(1, 1, 21, 11)
    past[0, 0, -1, 6] = 2.0
    past[0, 0, -1, 7] = 2.0

    data = {
        "ego_shape": _ego_shape(),
        "neighbor_agents_future": neighbor,
        "neighbor_agents_past": past,
    }
    return ego, data


def _clone_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in data.items()}


def test_classify_scene_failures_converts_3col_future_and_flags_moving_collision():
    ego, data = _moving_collision_data_3col()

    row = classify_loaded_scene(
        "/tmp/moving_collision.npz",
        ego,
        data,
        RewardConfig(),
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert "moving_collision" in row["labels"]
    assert row["moving_collision_step"] == 30
    assert row["moving_argmin_t"] == 30
    assert row["moving_argmin_neighbor"] == 0
    assert row["moving_min_dist"] < 0.0
    assert row["ttc_first_collision_step"] == 30
    assert "static_collision" in row
    assert "static_min_dist" in row
    assert "static_collision_step" in row
    assert "static_neighbor_count" in row


def test_compute_subscores_scene_batch_matches_per_scene_scoring():
    ego, data = _moving_collision_data_3col()
    clean_candidate = ego.clone()
    clean_candidate[..., 1] = 10.0
    candidates = torch.cat([ego, clean_candidate], dim=0)
    datas = [_prepare_scoring_data(data), _prepare_scoring_data(_clone_data(data))]

    batched = compute_subscores_scene_batch(
        candidates.unsqueeze(0).repeat(2, 1, 1, 1),
        _stack_scene_data(datas),
        RewardConfig(),
    )
    single = compute_subscores_batch(candidates, datas[0], RewardConfig())

    assert torch.allclose(batched["safety"][0], single["safety"])
    assert torch.allclose(batched["ttc"][1], single["ttc"])
    assert batched["collision_step"][0] == single["collision_step"]
    assert batched["ttc_first_collision_steps"][1] == single["ttc_first_collision_steps"]


def test_classify_loaded_scenes_batch_handles_multiple_scenes_one_trajectory_each():
    ego, data = _moving_collision_data_3col()
    clean_ego = ego.clone()
    clean_ego[..., 1] = 10.0
    ego_trajs = torch.stack([ego, clean_ego], dim=0)

    rows = classify_loaded_scenes_batch(
        ["/tmp/a.npz", "/tmp/b.npz"],
        ego_trajs,
        [_clone_data(data), _clone_data(data)],
        RewardConfig(),
        moving_near_thresh=1.0,
        static_near_thresh=0.4,
        rb_near_thresh=0.45,
        device=torch.device("cpu"),
    )

    assert len(rows) == 2
    assert [row["candidate_index"] for row in rows] == [0, 0]
    assert "moving_collision" in rows[0]["labels"]
    assert rows[0]["moving_collision_step"] == 30
    assert rows[1]["labels"] == ["clean"]


def test_classify_scene_failures_writes_training_path_lists(tmp_path):
    rows = [
        {"scene_path": "/tmp/a.npz", "labels": ["moving_collision", "road_border_crossing"]},
        {"scene_path": "/tmp/a.npz", "labels": ["clean"]},
        {"scene_path": "/tmp/b.npz", "labels": ["clean"]},
        {"scene_path": "/tmp/c.npz", "labels": ["moving_collision"]},
    ]

    _write_outputs(rows, [], tmp_path, {"moving_near_thresh": 1.0})

    assert json.loads((tmp_path / "lists" / "moving_collision.json").read_text()) == [
        "/tmp/a.npz",
        "/tmp/c.npz",
    ]
    assert json.loads((tmp_path / "lists" / "all_flagged.json").read_text()) == [
        "/tmp/a.npz",
        "/tmp/c.npz",
    ]
    assert json.loads((tmp_path / "lists" / "clean.json").read_text()) == ["/tmp/b.npz"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["label_counts"]["moving_collision"] == 2
    assert summary["label_counts"]["clean"] == 2


def test_scene_failure_threshold_config_uses_requested_defaults():
    thresholds = _load_scene_thresholds(_DEFAULT_THRESHOLD_CONFIG)

    assert thresholds == {
        "moving_near_thresh": 0.7,
        "static_near_thresh": 0.5,
        "rb_near_thresh": 0.2,
        "sc_cross_thresh": 0.2,
        "rb_cross_thresh": 0.2,
    }


def test_scene_failure_thresholds_override_reward_config():
    class Args:
        threshold_config = _DEFAULT_THRESHOLD_CONFIG
        moving_near_thresh = None
        static_near_thresh = None
        rb_near_thresh = None
        sc_cross_thresh = None
        rb_cross_thresh = None

    config = RewardConfig(rb_cross_thresh=0.45, rb_near_thresh=0.45, sc_near_thresh=0.4)
    thresholds = _apply_scene_thresholds(config, Args())

    assert thresholds["moving_near_thresh"] == 0.7
    assert thresholds["static_near_thresh"] == 0.5
    assert thresholds["rb_near_thresh"] == 0.2
    assert thresholds["sc_cross_thresh"] == 0.2
    assert thresholds["rb_cross_thresh"] == 0.2
    assert config.rb_cross_thresh == 0.2
    assert config.rb_near_thresh == 0.2
    assert config.sc_near_thresh == 0.5


def test_merge_output_dirs_rejects_threshold_mismatch(tmp_path):
    shard = tmp_path / "shard"
    _write_outputs(
        [{"scene_path": "/tmp/a.npz", "labels": ["moving_near_miss"]}],
        [],
        shard,
        {"moving_near_thresh": 1.0},
    )

    try:
        _merge_output_dirs([shard], tmp_path / "merged", {"moving_near_thresh": 0.7})
    except ValueError as exc:
        assert "do not match requested merge thresholds" in str(exc)
    else:
        raise AssertionError("expected threshold mismatch to fail")


def test_saved_prediction_trajectory_extracts_ego_from_agent_major_npz(tmp_path):
    pred = torch.zeros(3, T, 4).numpy()
    pred[0, :, 0] = 1.5
    pred[1, :, 0] = 9.0
    pred_path = tmp_path / "prediction00000000.npz"

    np.savez(pred_path, prediction=pred, turn_indicator=0)

    ego = _saved_prediction_trajectory(pred_path, torch.device("cpu"))

    assert ego.shape == (1, T, 4)
    assert torch.allclose(ego[0, :, 0], torch.full((T,), 1.5))


def test_load_npz_data_preserves_nonzero_delay(tmp_path):
    scene_path = tmp_path / "scene.npz"
    np.savez(scene_path, ego_shape=np.array([2.79, 4.34, 1.70]), delay=np.array(4))

    data = _load_npz_data(scene_path, torch.device("cpu"))

    assert data["delay"].dtype == torch.long
    assert data["delay"].shape == (1,)
    assert int(data["delay"].item()) == 4


def test_prediction_path_for_scene_supports_flat_and_mirrored_layouts(tmp_path):
    scene_path = "/data/root/dataset/train/date/time/frame_000123.npz"
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    flat = flat_dir / "prediction00000007.npz"
    flat.write_bytes(b"")
    assert _prediction_path_for_scene(flat_dir, scene_path, 7) == flat

    mirrored_dir = tmp_path / "mirrored"
    mirrored = mirrored_dir / "dataset/train/date/time/frame_000123.npz"
    mirrored.parent.mkdir(parents=True)
    mirrored.write_bytes(b"")
    assert (
        _prediction_path_for_scene(
            mirrored_dir,
            scene_path,
            7,
            prediction_scene_root=tmp_path / "missing",
        )
        == mirrored
    )
