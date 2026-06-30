from __future__ import annotations

import json

import torch

from planner_metrics.aggregate import compute_subscores_batch, compute_subscores_scene_batch
from rlvr.autoresearch.tools.classify_scene_failures import (
    _prepare_scoring_data,
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
