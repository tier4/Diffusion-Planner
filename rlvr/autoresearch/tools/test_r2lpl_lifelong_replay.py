from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import scenario_generation.reproducer_rollout as reproducer_rollout
from rlvr.autoresearch.tools import build_avoiding_target as build_avoiding_target_tool
from rlvr.autoresearch.tools import mine_credit_window_scenes as mine_credit_window_scenes_tool
from rlvr.autoresearch.tools import (
    mine_direct_reproducer_chunks as mine_direct_reproducer_chunks_tool,
)
from rlvr.autoresearch.tools import reproducer_danger_scorer
from rlvr.autoresearch.tools import run_lifelong_r2lpl_rounds as round_runner
from rlvr.autoresearch.tools.build_avoiding_target import (
    _best_safe_candidate,
    _candidate_violation_score,
    _drop_t0_dirty_event_windows,
    _filtered_npz_payload,
    _future4_to_3col,
    _parse_ego_shape,
    _source_scene_t0_moving_overlap,
)
from rlvr.autoresearch.tools.lifelong_replay_memory import build_memory
from rlvr.autoresearch.tools.mine_credit_window_scenes import (
    _resolve_row,
    _select_event_windows,
    _validate_credit_config,
)
from rlvr.autoresearch.tools.mine_direct_reproducer_chunks import (
    Chunk,
    _chunk_row,
    _iter_scene_list,
    _sample_value,
    _validate_timeline_continuity,
    iter_direct_chunks,
    iter_manifest_chunks,
)
from rlvr.autoresearch.tools.reproducer_danger_scorer import build_realized_event_scorer
from rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds import (
    _base_train_invocation,
    _gpu_ids_from_config,
    _load_config,
    _lora_for_policy,
    _perception_mining_cmd,
    _repair_cmd,
    _run_mining_phase,
    _run_repair_phase,
    _union_scene_lists,
)
from rlvr.deviation import rollout_gt_deviation
from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss
from rlvr.reward import RewardConfig
from scenario_generation.conflict_detector import detect_expert_disagreement
from scenario_generation.danger_event_selection import (
    OnlineEventSelector,
    contiguous_index_runs,
    decluster_indices,
    sustained_true_indices,
)
from scenario_generation.tools.classify_replay_steps import _decluster as _decluster_replay_steps


class _IdentityObservationNormalizer:
    def __call__(self, data):
        return data


class _StateNormalizer:
    def __init__(self):
        self.mean = torch.zeros(1, 4)
        self.std = torch.ones(1, 4)


class _ConstantDenoiser(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, inputs):
        sampled = inputs["sampled_trajectories"]
        return None, {"model_output": sampled * 0.0 + self.value}


def _minimal_sft_batch(batch_size: int = 2, future_len: int = 3):
    device = torch.device("cpu")
    model_args = SimpleNamespace(
        predicted_neighbor_num=0,
        future_len=future_len,
        state_normalizer=_StateNormalizer(),
        observation_normalizer=_IdentityObservationNormalizer(),
    )
    data = {
        "ego_current_state": torch.zeros(batch_size, 10, device=device),
        "neighbor_agents_past": torch.zeros(batch_size, 0, 31, 4, device=device),
    }
    neighbor_gt = torch.zeros(batch_size, 0, future_len, 4, device=device)
    neighbor_mask = torch.zeros(batch_size, 0, future_len, dtype=torch.bool, device=device)
    return device, model_args, data, neighbor_gt, neighbor_mask


def test_credit_window_config_rejects_missing_observed_label(tmp_path):
    cfg = tmp_path / "credit.json"
    cfg.write_text(
        json.dumps(
            {
                "_frame_hz": 10,
                "_defaults": {"width_s": 1.5, "gap_s": 1.5},
                "static_collision": {},
            }
        )
    )

    try:
        _validate_credit_config(cfg, {"static_collision", "expert_disagreement"})
    except ValueError as exc:
        assert "expert_disagreement" in str(exc)
    else:
        raise AssertionError("missing label should fail loudly")


def test_credit_window_config_uses_seconds_schema_and_rejects_scalars(tmp_path):
    cfg = tmp_path / "credit.json"
    cfg.write_text(
        json.dumps(
            {
                "_frame_hz": 10,
                "_defaults": {"width_s": 1.5, "gap_s": 1.5},
                "moving_collision": {},
                "road_border_crossing": {"width_s": 2.0},
            }
        )
    )

    parsed = reproducer_danger_scorer.load_credit_windows(cfg)

    assert parsed["moving_collision"]["width_frames"] == 15
    assert parsed["moving_collision"]["gap_frames"] == 15
    assert parsed["road_border_crossing"]["width_frames"] == 20
    assert parsed["road_border_crossing"]["gap_frames"] == 15

    old_cfg = tmp_path / "old_credit.json"
    old_cfg.write_text(
        json.dumps(
            {
                "_frame_hz": 10,
                "_defaults": {"width_s": 1.5, "gap_s": 1.5},
                "moving_collision": 15,
            }
        )
    )
    with pytest.raises(ValueError, match="scalar frame counts are not supported"):
        reproducer_danger_scorer.load_credit_windows(old_cfg)


def _credit_spec(width_frames: int = 15, gap_frames: int = 15):
    return {
        "width_s": width_frames / 10.0,
        "gap_s": gap_frames / 10.0,
        "width_frames": width_frames,
        "gap_frames": gap_frames,
    }


def _write_direct_chunk_scene_list(tmp_path, stems):
    paths = [tmp_path / f"{stem}.npz" for stem in stems]
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps([str(p) for p in paths]))
    return path


def _read_test_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _visible_gpu_count_for_test() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        proc = None
    if proc is not None and proc.returncode == 0:
        return sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))
    return torch.cuda.device_count()


def test_direct_reproducer_chunks_sample_every_80th_scene(tmp_path):
    scene_list = _write_direct_chunk_scene_list(
        tmp_path,
        [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 240)],
    )

    chunks = list(iter_direct_chunks(scene_list))

    assert [chunk.global_start_index for chunk in chunks] == [0, 80, 160]
    assert [chunk.n_frames for chunk in chunks] == [80, 80, 80]
    assert chunks[0].start_frame == 31
    assert chunks[0].end_frame == 110


def test_direct_reproducer_scene_list_parser_handles_pretty_json(tmp_path):
    paths = [tmp_path / f"bagA_00000001_{i:08d}.npz" for i in range(31, 34)]
    scene_list = tmp_path / "pretty_scenes.json"
    scene_list.write_text("[\n" + ",\n".join(f'  "{p}"' for p in paths) + "\n]\n")

    assert list(_iter_scene_list(scene_list)) == paths


def test_direct_reproducer_chunks_discard_short_jump_by_default(tmp_path):
    stems = [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 80)]
    stems += [f"bagA_00000001_{i:08d}" for i in range(200, 200 + 20)]
    stems += [f"bagB_00000001_{i:08d}" for i in range(1, 1 + 60)]
    scene_list = _write_direct_chunk_scene_list(tmp_path, stems)

    chunks = list(iter_direct_chunks(scene_list))
    partial_chunks = list(iter_direct_chunks(scene_list, min_chunk_len=2))

    assert [chunk.global_start_index for chunk in chunks] == [0]
    assert [chunk.global_start_index for chunk in partial_chunks] == [0, 80]
    assert partial_chunks[1].n_frames == 20
    assert partial_chunks[1].end_reason == "lineage_break"


def test_direct_reproducer_chunks_shard_by_global_start(tmp_path):
    scene_list = _write_direct_chunk_scene_list(
        tmp_path,
        [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 320)],
    )

    shard0 = list(iter_direct_chunks(scene_list, num_shards=2, shard_index=0))
    shard1 = list(iter_direct_chunks(scene_list, num_shards=2, shard_index=1))

    assert [chunk.global_start_index for chunk in shard0] == [0, 160]
    assert [chunk.global_start_index for chunk in shard1] == [80, 240]


def test_direct_reproducer_chunks_sample_fraction_is_deterministic(tmp_path):
    scene_list = _write_direct_chunk_scene_list(
        tmp_path,
        [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 800)],
    )

    chunks = list(iter_direct_chunks(scene_list, sample_fraction=0.5, sample_seed=17))
    repeated = list(iter_direct_chunks(scene_list, sample_fraction=0.5, sample_seed=17))
    expected_starts = [start for start in range(0, 800, 80) if _sample_value(start, 17) < 0.5]

    assert [chunk.global_start_index for chunk in chunks] == expected_starts
    assert [chunk.global_start_index for chunk in repeated] == expected_starts
    assert 0 < len(chunks) < 10


def test_direct_reproducer_chunks_read_compact_manifest(tmp_path):
    scene_list = _write_direct_chunk_scene_list(
        tmp_path,
        [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 320)],
    )
    manifest = tmp_path / "chunks.jsonl"
    with open(manifest, "w") as f:
        for chunk in iter_direct_chunks(scene_list):
            f.write(json.dumps(_chunk_row(chunk), sort_keys=True) + "\n")

    shard0 = list(iter_manifest_chunks(manifest, num_shards=2, shard_index=0))
    shard1 = list(iter_manifest_chunks(manifest, num_shards=2, shard_index=1))

    assert [chunk.global_start_index for chunk in shard0] == [0, 160]
    assert [chunk.global_start_index for chunk in shard1] == [80, 240]
    assert [p.name for p in shard0[0].paths[:3]] == [
        "bagA_00000001_00000031.npz",
        "bagA_00000001_00000032.npz",
        "bagA_00000001_00000033.npz",
    ]


def test_direct_reproducer_timeline_guard_rejects_pose_jump(tmp_path):
    chunk = Chunk(
        key="chunk",
        global_start_index=0,
        global_end_index=3,
        paths=tuple(tmp_path / f"bagA_00000001_{i:08d}.npz" for i in range(3)),
        start_frame=0,
        end_frame=2,
        end_reason="chunk_len",
        is_full_length=True,
    )
    timeline = SimpleNamespace(
        frame_indices=np.array([0, 1, 2], dtype=np.int64),
        poses=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
    )

    try:
        _validate_timeline_continuity(
            timeline,
            chunk,
            expected_frame_step=1,
            max_pose_step_m=10.0,
            max_pose_speed_mps=20.0,
            max_yaw_step_rad=1.57,
        )
    except ValueError as exc:
        assert "pose jump" in str(exc)
    else:
        raise AssertionError("pose jump should be rejected")


def test_direct_reproducer_timeline_guard_rejects_pose_speed_spike(tmp_path):
    chunk = Chunk(
        key="chunk",
        global_start_index=0,
        global_end_index=3,
        paths=tuple(tmp_path / f"bagA_00000001_{i:08d}.npz" for i in range(3)),
        start_frame=0,
        end_frame=2,
        end_reason="chunk_len",
        is_full_length=True,
    )
    timeline = SimpleNamespace(
        frame_indices=np.array([0, 1, 2], dtype=np.int64),
        poses=np.array([[0.0, 0.0, 0.0], [4.91, 0.0, 0.0], [5.91, 0.0, 0.0]]),
    )

    try:
        _validate_timeline_continuity(
            timeline,
            chunk,
            expected_frame_step=1,
            max_pose_step_m=10.0,
            max_pose_speed_mps=20.0,
            max_yaw_step_rad=1.57,
        )
    except ValueError as exc:
        assert "pose speed" in str(exc)
        assert "49.1m/s" in str(exc)
    else:
        raise AssertionError("pose speed spike should be rejected")


def test_direct_reproducer_timeline_guard_unwraps_yaw(tmp_path):
    chunk = Chunk(
        key="chunk",
        global_start_index=0,
        global_end_index=3,
        paths=tuple(tmp_path / f"bagA_00000001_{i:08d}.npz" for i in range(3)),
        start_frame=0,
        end_frame=2,
        end_reason="chunk_len",
        is_full_length=True,
    )
    timeline = SimpleNamespace(
        frame_indices=np.array([0, 1, 2], dtype=np.int64),
        poses=np.array([[0.0, 0.0, 3.13], [1.0, 0.0, -3.13], [2.0, 0.0, -3.12]]),
    )

    _validate_timeline_continuity(
        timeline,
        chunk,
        expected_frame_step=1,
        max_pose_step_m=10.0,
        max_pose_speed_mps=20.0,
        max_yaw_step_rad=0.1,
    )


def test_perception_mining_cmd_supports_direct_reproducer_chunks(tmp_path):
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")
    cfg = {
        "scene_pool": str(scene_list),
        "route_scene_list": str(scene_list),
        "reward_config": str(tmp_path / "reward.json"),
        "threshold_config": str(tmp_path / "thresholds.json"),
        "credit_window_config": str(tmp_path / "credit.json"),
        "mine_labels": ["moving_collision", "road_border_crossing"],
        "perception_mining": {
            "tool": "direct_reproducer_chunks",
            "chunk_len": 80,
            "batch_size": 32,
            "max_pose_step_m": 10.0,
            "max_pose_speed_mps": 20.0,
            "sample_fraction": 0.25,
            "sample_seed": 123,
            "allow_existing_out_dir": True,
        },
    }

    cmd, save_dir = _perception_mining_cmd(cfg, tmp_path / "model.pth", tmp_path / "round")

    assert "rlvr.autoresearch.tools.mine_direct_reproducer_chunks" in cmd
    assert "--scene_list" in cmd
    assert str(scene_list) in cmd
    assert "--max_pose_step_m" in cmd
    assert "10.0" in cmd
    assert "--max_pose_speed_mps" in cmd
    assert "20.0" in cmd
    assert "--sample_fraction" in cmd
    assert "0.25" in cmd
    assert "--sample_seed" in cmd
    assert "123" in cmd
    assert "--allow_existing_out_dir" in cmd
    assert save_dir == tmp_path / "round" / "perception_danger_windows"


def test_workflow_contract_preserves_repaired_target_validation_flag(tmp_path):
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "judgement": {
                    "reward_config": "/tmp/reward.json",
                    "threshold_config": "/tmp/thresholds.json",
                    "credit_window_config": "/tmp/credit.json",
                    "enabled_labels": ["moving_collision"],
                },
                "repair_generation": {"ego_shape": "from_npz", "min_margin": 0.3},
                "training": {"val_scenes": "/tmp/valid.json"},
                "validate_on_repaired_targets": True,
            }
        )
    )
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"backend": "base_sft", "train_args": {}}))

    cfg = round_runner._config_from_workflow_contract(
        {
            "model_path": "/tmp/model.pth",
            "scene_list": str(scene_list),
            "workflow_config": str(workflow),
            "training_config": str(training),
            "output_dir": str(tmp_path / "auto_research" / "out"),
        }
    )

    assert cfg["validate_on_repaired_targets"] is True


def test_round_runner_splits_mine_and_repair_labels(tmp_path):
    cfg = {
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "mine_labels": [
            "road_border_crossing",
            "static_collision",
            "moving_collision",
            "road_border_near",
            "static_near_miss",
            "moving_near_miss",
            "moving_ttc",
            "expert_disagreement",
        ],
        "repair_labels": ["road_border_crossing", "static_collision", "moving_collision"],
        "perception_mining": {
            "tool": "direct_reproducer_chunks",
            "scene_list": "/data/scenes.json",
            "batch_size": 4,
        },
        "repair_config": {
            "ego_shape": "from_npz",
            "min_margin": 0.3,
            "K": 8,
        },
        "count_rear_end_collisions": True,
    }

    mine_cmd, _save_dir = _perception_mining_cmd(cfg, tmp_path / "model.pth", tmp_path / "round")
    repair_cmd = _repair_cmd(
        cfg,
        tmp_path / "model.pth",
        tmp_path / "round" / "credit_windows.jsonl",
        tmp_path / "round",
    )

    assert mine_cmd[mine_cmd.index("--labels") + 1] == ",".join(cfg["mine_labels"])
    assert repair_cmd[repair_cmd.index("--labels") + 1] == ",".join(cfg["repair_labels"])


def test_perception_mining_cmd_supports_direct_chunk_manifest(tmp_path):
    manifest = tmp_path / "chunks.jsonl"
    manifest.write_text("")
    cfg = {
        "reward_config": str(tmp_path / "reward.json"),
        "threshold_config": str(tmp_path / "thresholds.json"),
        "credit_window_config": str(tmp_path / "credit.json"),
        "mine_labels": ["road_border_crossing"],
        "perception_mining": {
            "tool": "direct_reproducer_chunks",
            "chunk_manifest": str(manifest),
            "batch_size": 32,
            "max_scenes": 1000,
        },
    }

    cmd, save_dir = _perception_mining_cmd(cfg, tmp_path / "model.pth", tmp_path / "round")

    assert "--chunk_manifest" in cmd
    assert str(manifest) in cmd
    assert "--scene_list" not in cmd
    assert "--max_scenes" not in cmd
    assert save_dir == tmp_path / "round" / "perception_danger_windows"


def test_direct_reproducer_credit_row_propagates_frame_segment(tmp_path):
    from rlvr.autoresearch.tools.mine_direct_reproducer_chunks import (
        _credit_row_from_saved_scene,
    )

    scene_path = tmp_path / "window" / "credit+00000.npz"
    scene_path.parent.mkdir()
    scene_path.write_bytes(b"npz")
    row = _credit_row_from_saved_scene(
        scene_path,
        {
            "offense_step": 12,
            "offense_frame_id": 112,
            "segment": [10, 90],
            "segment_route_indices": [10, 90],
            "segment_frame_ids": [110, 190],
        },
        "moving_collision",
    )

    assert row["segment"] == [10, 90]
    assert row["segment_route_indices"] == [10, 90]
    assert row["segment_frame_ids"] == [110, 190]


def test_mine_direct_main_forwards_realized_hard_event_scorer(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_build_realized_event_scorer(**kwargs):
        captured["realized_allowed_labels"] = kwargs.get("allowed_labels")
        return "realized-scorer"

    def _fake_run_segments_batched(*args, **kwargs):
        captured["realized_event_scorer"] = kwargs.get("realized_event_scorer")
        captured["danger_scorer"] = kwargs.get("danger_scorer")
        return [SimpleNamespace(metrics={"n_collision_steps": 1, "min_clearance": -0.1})]

    chunk = Chunk(
        key="chunkA",
        global_start_index=0,
        global_end_index=1,
        paths=(tmp_path / "scene_00000000.npz",),
        start_frame=0,
        end_frame=0,
        end_reason="chunk_len",
        is_full_length=True,
    )

    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "_load_model",
        lambda *_args, **_kwargs: (object(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "build_reproducer_danger_scorer",
        lambda **_kwargs: "danger-scorer",
    )
    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "build_realized_event_scorer",
        _fake_build_realized_event_scorer,
    )
    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "load_credit_windows",
        lambda _path: {
            "moving_collision": _credit_spec(),
            "road_border_crossing": _credit_spec(),
        },
    )
    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "iter_direct_chunks",
        lambda *_args, **_kwargs: iter([chunk]),
    )
    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "_build_work_unit",
        lambda *_args, **_kwargs: (SimpleNamespace(prefetch=lambda _items: None), 0, 1),
    )
    monkeypatch.setattr(
        mine_direct_reproducer_chunks_tool,
        "run_segments_batched",
        _fake_run_segments_batched,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_direct_reproducer_chunks.py",
            "--scene_list",
            str(tmp_path / "scenes.json"),
            "--model_path",
            str(tmp_path / "model.pth"),
            "--out_dir",
            str(tmp_path / "windows"),
            "--out_jsonl",
            str(tmp_path / "credit_windows.jsonl"),
            "--segments_jsonl",
            str(tmp_path / "segments.jsonl"),
            "--danger_reward_config",
            str(tmp_path / "reward.json"),
            "--danger_threshold_config",
            str(tmp_path / "thresholds.json"),
            "--danger_credit_window_config",
            str(tmp_path / "credit.json"),
            "--labels",
            "road_border_crossing,static_collision,moving_collision",
            "--batch_size",
            "1",
        ],
    )

    mine_direct_reproducer_chunks_tool.main()

    assert captured["danger_scorer"] == "danger-scorer"
    assert captured["realized_allowed_labels"] == {
        "road_border_crossing",
        "static_collision",
        "moving_collision",
    }
    assert captured["realized_event_scorer"] == "realized-scorer"


def test_round_runner_mining_shards_use_private_outputs_and_merge(monkeypatch, tmp_path):
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text(json.dumps(["/data/scene_0.npz"]))
    cfg = {
        "reward_config": str(tmp_path / "reward.json"),
        "threshold_config": str(tmp_path / "thresholds.json"),
        "credit_window_config": str(tmp_path / "credit.json"),
        "mine_labels": ["moving_collision"],
        "perception_mining": {
            "tool": "direct_reproducer_chunks",
            "scene_list": str(scene_list),
            "batch_size": 4,
        },
    }
    rdir = tmp_path / "round"
    seen_jobs = []
    seen_plan_cmds = []

    def _arg(cmd, flag):
        return Path(cmd[cmd.index(flag) + 1])

    def _fake_run_parallel(jobs, *, cwd=None):
        assert cwd is None
        seen_jobs.extend(jobs)
        out_jsonls = [_arg(cmd, "--out_jsonl") for _, cmd, _, _ in jobs]
        segments = [_arg(cmd, "--segments_jsonl") for _, cmd, _, _ in jobs]
        summaries = [_arg(cmd, "--summary_json") for _, cmd, _, _ in jobs]
        logs = [log for _, _, log, _ in jobs]
        private_paths = [*out_jsonls, *segments, *summaries, *logs]
        assert len({path.resolve(strict=False) for path in private_paths}) == len(private_paths)
        assert (rdir / "credit_windows.jsonl") not in out_jsonls
        assert (rdir / "perception_reproducer_hits.jsonl") not in segments
        assert (rdir / "perception_direct_summary.json") not in summaries
        for expected_idx, (label, cmd, _log, env) in enumerate(jobs):
            assert label == f"perception_mine[{expected_idx}]"
            assert env["CUDA_VISIBLE_DEVICES"] == str(expected_idx)
            assert cmd[cmd.index("--num_shards") + 1] == "2"
            assert cmd[cmd.index("--shard_index") + 1] == str(expected_idx)
            assert "--chunk_manifest" in cmd
            assert cmd[cmd.index("--chunk_manifest") + 1] == str(rdir / "planned_chunks.jsonl")
            assert "--scene_list" not in cmd
            _arg(cmd, "--out_jsonl").parent.mkdir(parents=True, exist_ok=True)
            _arg(cmd, "--out_jsonl").write_text(
                json.dumps(
                    {
                        "scene_path": f"/data/repaired_source_{expected_idx}.npz",
                        "label": "moving_collision",
                    }
                )
                + "\n"
            )
            _arg(cmd, "--segments_jsonl").write_text(json.dumps({"segment": expected_idx}) + "\n")
            _arg(cmd, "--summary_json").write_text(
                json.dumps(
                    {
                        "planned_chunks": 3,
                        "simulated_chunks": 2,
                        "skipped_chunks": 1,
                        "credit_rows": 1,
                        "elapsed_sec": float(expected_idx + 1),
                    }
                )
            )
        return 12.0

    def _fake_run(cmd, log_path, *, cwd=None, env=None):
        assert cwd is None
        assert env is None
        seen_plan_cmds.append(cmd)
        assert "--plan_only" in cmd
        assert "--scene_list" in cmd
        assert cmd[cmd.index("--scene_list") + 1] == str(scene_list)
        assert "--segments_jsonl" in cmd
        Path(cmd[cmd.index("--segments_jsonl") + 1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[cmd.index("--segments_jsonl") + 1]).write_text(
            json.dumps(
                {
                    "chunk_key": "chunk0",
                    "global_start_index": 0,
                    "global_end_index": 80,
                    "n_frames": 80,
                    "start_frame": 0,
                    "end_frame": 79,
                    "end_reason": "chunk_len",
                    "is_full_length": True,
                    "start_scene_path": "/data/scene_0.npz",
                    "end_scene_path": "/data/scene_79.npz",
                }
            )
            + "\n"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("planned\n")
        return 1.0

    monkeypatch.setattr(round_runner, "_run", _fake_run)
    monkeypatch.setattr(round_runner, "_run_parallel", _fake_run_parallel)

    elapsed = _run_mining_phase(cfg, tmp_path / "model.pth", rdir, [0, 1])

    assert elapsed == 12.0
    assert len(seen_plan_cmds) == 1
    assert len(seen_jobs) == 2
    assert [row["scene_path"] for row in _read_test_jsonl(rdir / "credit_windows.jsonl")] == [
        "/data/repaired_source_0.npz",
        "/data/repaired_source_1.npz",
    ]
    assert _read_test_jsonl(rdir / "perception_reproducer_hits.jsonl") == [
        {"segment": 0},
        {"segment": 1},
    ]
    summary = json.loads((rdir / "perception_direct_summary.json").read_text())
    assert summary["planned_chunks"] == 6
    assert summary["simulated_chunks"] == 4
    assert summary["skipped_chunks"] == 2
    assert summary["credit_rows"] == 2
    assert json.loads((rdir / "credit_windows_paths.json").read_text()) == [
        "/data/repaired_source_0.npz",
        "/data/repaired_source_1.npz",
    ]


def test_round_runner_repair_shards_use_private_inputs_outputs_and_merge(monkeypatch, tmp_path):
    rdir = tmp_path / "round"
    rows = [
        {"scene_path": f"/data/source_{idx}.npz", "label": "moving_collision"} for idx in range(4)
    ]
    (rdir / "credit_windows.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (rdir / "credit_windows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    cfg = {
        "reward_config": str(tmp_path / "reward.json"),
        "threshold_config": str(tmp_path / "thresholds.json"),
        "mine_labels": ["moving_collision"],
        "repair_config": {
            "ego_shape": "4.76,7.24,2.29",
            "min_margin": 0.3,
            "K": 8,
        },
        "count_rear_end_collisions": True,
    }
    seen_jobs = []

    def _arg(cmd, flag):
        return Path(cmd[cmd.index(flag) + 1])

    def _fake_run_parallel(jobs, *, cwd=None):
        assert cwd is None
        seen_jobs.extend(jobs)
        shard_inputs = [_arg(cmd, "--scene_rows_jsonl") for _, cmd, _, _ in jobs]
        out_lists = [_arg(cmd, "--out_list") for _, cmd, _, _ in jobs]
        out_rows = [_arg(cmd, "--out_rows_jsonl") for _, cmd, _, _ in jobs]
        out_dirs = [_arg(cmd, "--out_dir") for _, cmd, _, _ in jobs]
        logs = [log for _, _, log, _ in jobs]
        private_paths = [*shard_inputs, *out_lists, *out_rows, *out_dirs, *logs]
        assert len({path.resolve(strict=False) for path in private_paths}) == len(private_paths)
        assert (rdir / "credit_windows.jsonl") not in shard_inputs
        assert (rdir / "repaired_targets.json") not in out_lists
        assert (rdir / "repaired_targets.jsonl") not in out_rows
        for expected_idx, (label, cmd, _log, env) in enumerate(jobs):
            assert label == f"repair[{expected_idx}]"
            assert env["CUDA_VISIBLE_DEVICES"] == str(expected_idx)
            assert cmd[cmd.index("--device") + 1] == "cuda"
            assert "--allow_empty" in cmd
            shard_rows = _read_test_jsonl(_arg(cmd, "--scene_rows_jsonl"))
            assert shard_rows
            _arg(cmd, "--out_list").parent.mkdir(parents=True, exist_ok=True)
            repaired_paths = [
                f"/data/repaired_{expected_idx}_{row_idx}.npz"
                for row_idx, _row in enumerate(shard_rows)
            ]
            _arg(cmd, "--out_list").write_text(json.dumps(repaired_paths))
            _arg(cmd, "--out_rows_jsonl").write_text(
                "".join(
                    json.dumps({"scene_path": path, "label": "moving_collision"}) + "\n"
                    for path in repaired_paths
                )
            )
            (_arg(cmd, "--out_list").parent / "repaired_targets_unrepaired.json").write_text(
                json.dumps([{"shard": expected_idx, "count": 0}])
            )
        return 8.0

    monkeypatch.setattr(round_runner, "_run_parallel", _fake_run_parallel)

    elapsed = _run_repair_phase(cfg, tmp_path / "model.pth", rdir, [0, 1])

    assert elapsed == 8.0
    assert len(seen_jobs) == 2
    assert json.loads((rdir / "repaired_targets.json").read_text()) == [
        "/data/repaired_0_0.npz",
        "/data/repaired_0_1.npz",
        "/data/repaired_1_0.npz",
        "/data/repaired_1_1.npz",
    ]
    assert [row["scene_path"] for row in _read_test_jsonl(rdir / "repaired_targets.jsonl")] == [
        "/data/repaired_0_0.npz",
        "/data/repaired_0_1.npz",
        "/data/repaired_1_0.npz",
        "/data/repaired_1_1.npz",
    ]
    assert json.loads((rdir / "repaired_targets_unrepaired.json").read_text()) == [
        {"shard": 0, "count": 0},
        {"shard": 1, "count": 0},
    ]


def test_lineage_resolver_maps_route_frame_and_step(tmp_path):
    route = [tmp_path / f"bagA_{i:04d}.npz" for i in range(100, 121)]
    row = {
        "scene_path": str(tmp_path / "bagA_0105.npz"),
        "labels": ["static_collision"],
        "static_collision_step": 7,
    }

    [resolved] = _resolve_row(row, {"bagA": route}, {"static_collision": _credit_spec()})

    assert resolved["route_key"] == "bagA"
    assert resolved["frame_index"] == 105
    assert resolved["offense_frame"] == 112
    assert resolved["offense_index"] == 12
    assert resolved["start_frame"] == 100
    assert resolved["start_index"] == 0


def test_lineage_resolver_rejects_non_route_path(tmp_path):
    row = {"scene_path": str(tmp_path / "hand_curated_scene.npz"), "labels": ["static_collision"]}

    try:
        _resolve_row(row, {}, {"static_collision": _credit_spec()})
    except ValueError as exc:
        assert "route-lineage" in str(exc) or "frame index" in str(exc)
    else:
        raise AssertionError("non-route scene should fail loudly")


def test_conflict_detector_disabled_is_silent():
    gt = torch.zeros(20, 4)
    rollout = torch.ones(20, 4) * 10.0

    result = detect_expert_disagreement(
        rollout,
        gt,
        enabled=False,
        threshold_m=1.0,
        sustain_steps=10,
    )

    assert not result.expert_disagreement
    assert result.expert_disagreement_step is None


def test_conflict_detector_requires_sustained_mismatch():
    gt = torch.zeros(20, 4)
    rollout = torch.zeros(20, 4)
    rollout[3:8, 0] = 2.0
    short = detect_expert_disagreement(
        rollout,
        gt,
        enabled=True,
        threshold_m=1.0,
        sustain_steps=10,
    )
    rollout[8:14, 0] = 2.0
    sustained = detect_expert_disagreement(
        rollout,
        gt,
        enabled=True,
        threshold_m=1.0,
        sustain_steps=10,
    )

    assert not short.expert_disagreement
    assert sustained.expert_disagreement
    assert sustained.expert_disagreement_step == 3


def test_rollout_gt_deviation_reports_first_sustained_crossing():
    gt = torch.zeros(12, 4)
    rollout = torch.zeros(12, 4)
    rollout[4:8, 1] = 1.5

    max_dev, step = rollout_gt_deviation(
        rollout,
        gt,
        threshold_m=1.0,
        sustain_steps=4,
    )

    assert max_dev == 1.5
    assert step == 4


def test_replay_memory_preserves_rare_buckets_with_capacity():
    rows = [
        {
            "scene_path": f"/tmp/common_{i}.npz",
            "label": "static_collision",
            "route_arc_m": 1.0,
            "variant_kind": "credit",
            "difficulty": float(i),
        }
        for i in range(5)
    ]
    rows.append(
        {
            "scene_path": "/tmp/rare.npz",
            "label": "expert_disagreement",
            "route_arc_m": 200.0,
            "variant_kind": "credit",
            "difficulty": 0.0,
        }
    )

    memory = build_memory(
        rows,
        {"entries": []},
        {},
        capacity=2,
        alpha=1.0,
        beta=0.0,
        arc_bin_m=25.0,
    )

    selected = {row["scene_path"] for row in memory["entries"]}
    assert "/tmp/rare.npz" in selected
    assert len(selected) == 2


def test_round_runner_checkpoint_policy_prefers_latest_lora(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "lora_epoch_001").mkdir()
    (run_dir / "lora_latest").symlink_to("lora_epoch_001")

    assert _lora_for_policy(run_dir, "latest") == run_dir / "lora_latest"


def test_round_runner_checkpoint_policy_resolves_epoch_lora(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "lora_epoch_003").mkdir()

    assert _lora_for_policy(run_dir, "epoch:3") == run_dir / "lora_epoch_003"
    assert _lora_for_policy(run_dir, "epoch:2") is None


def test_round_runner_requires_perception_mining_source(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "rounds": 1,
                "epochs_per_round": 1,
                "model_path": "/tmp/model.pth",
                "val_scenes": "/tmp/val.json",
                "reward_config": "/tmp/reward.json",
                "threshold_config": "/tmp/thresholds.json",
                "credit_window_config": "/tmp/credit.json",
                "replay_memory": {},
                "training_config": "/tmp/train.json",
                "repair_config": {"ego_shape": "4.76,7.24,2.29", "min_margin": 0.3},
                "perception_mining": {"tool": "direct_reproducer_chunks"},
                "output_dir": str(tmp_path / "auto_research" / "run"),
            }
        )
    )

    try:
        _load_config(cfg)
    except ValueError as exc:
        assert "chunk_manifest" in str(exc)
        assert "scene_list" in str(exc)
    else:
        raise AssertionError("a mining source should be required")


def test_round_runner_load_config_translates_single_entry_contract(tmp_path):
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "judgement": {
                    "reward_config": "/tmp/reward.json",
                    "threshold_config": "/tmp/thresholds.json",
                    "credit_window_config": "/tmp/credit.json",
                    "enabled_labels": ["moving_collision"],
                },
                "resources": {"gpu_ids": [0, 1]},
                "perception_reproducer": {
                    "chunk_len": 80,
                    "start_stride": 80,
                    "batch_size": 4,
                    "max_pose_step_m": 10.0,
                    "max_pose_speed_mps": 20.0,
                },
                "repair_generation": {
                    "ego_shape": "4.76,7.24,2.29",
                    "min_margin": 0.3,
                },
                "replay_memory": {"capacity": 32, "alpha": 0.7, "beta": 0.2},
                "rounds": {"rounds": 2, "epochs_per_round": 3},
            }
        )
    )
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"train_args": {"valid_set_list": "/tmp/val.json"}}))
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "model_path": "/tmp/model.pth",
                "scene_list": "/tmp/scenes.json",
                "workflow_config": str(workflow),
                "training_config": str(training),
                "output_dir": str(tmp_path / "auto_research" / "run"),
            }
        )
    )

    cfg = _load_config(contract)

    assert cfg["route_scene_list"] == "/tmp/scenes.json"
    assert cfg["perception_mining"]["tool"] == "direct_reproducer_chunks"
    assert cfg["perception_mining"]["scene_list"] == "/tmp/scenes.json"
    assert cfg["perception_mining"]["chunk_len"] == 80
    assert cfg["perception_mining"]["start_stride"] == 80
    assert cfg["perception_mining"]["timeline_progress_mode"] == "clock"
    assert cfg["perception_mining"]["tracker_mode"] == "mpc"
    assert cfg["perception_mining"]["max_pose_step_m"] == 10.0
    assert cfg["perception_mining"]["max_pose_speed_mps"] == 20.0
    assert cfg["training_backend"] == "base_sft"
    assert cfg["gpu_ids"] == [0, 1]


def test_round_runner_parses_gpu_ids_from_string():
    assert _gpu_ids_from_config({"resources": {"gpu_ids": "0, 2,3"}}) == [0, 2, 3]


def test_round_runner_cli_dry_run_uses_multiple_visible_gpus_or_skips(tmp_path):
    visible_gpus = _visible_gpu_count_for_test()
    if visible_gpus < 2:
        pytest.skip("multi-GPU runner smoke requires at least two visible CUDA devices")
    gpu_ids = list(range(min(2, visible_gpus)))
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text(json.dumps([str(tmp_path / "scene_00000000.npz")]))
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "judgement": {
                    "reward_config": str(tmp_path / "reward.json"),
                    "threshold_config": str(tmp_path / "thresholds.json"),
                    "credit_window_config": str(tmp_path / "credit.json"),
                    "enabled_labels": ["moving_collision"],
                },
                "resources": {"gpu_ids": gpu_ids},
                "event_mining": {
                    "max_scenes": 1,
                    "chunk_len": 80,
                    "start_stride": 80,
                },
                "repair_generation": {
                    "ego_shape": "4.76,7.24,2.29",
                    "min_margin": 0.3,
                    "candidate_count_per_scene": 2,
                },
                "training": {"val_scenes": str(tmp_path / "val.json")},
                "rounds": {"rounds": 1, "epochs_per_round": 1},
            }
        )
    )
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"train_args": {"valid_set_list": str(tmp_path / "val.json")}}))
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo_root),
            str(repo_root / "diffusion_planner"),
            env.get("PYTHONPATH", ""),
        ]
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds",
            "--model_path",
            str(tmp_path / "model.pth"),
            "--scene_list",
            str(scene_list),
            "--workflow_config",
            str(workflow),
            "--training_config",
            str(training),
            "--output_dir",
            str(tmp_path / "auto_research" / "runner_smoke"),
            "--dry_run",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    for shard_index, gpu_id in enumerate(gpu_ids):
        assert f"perception_mine[{shard_index}]" in proc.stdout
        assert f"repair[{shard_index}]" in proc.stdout
        assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in proc.stdout
        assert f"--shard_index {shard_index}" in proc.stdout
    assert f"--num_shards {len(gpu_ids)}" in proc.stdout
    assert "--allow_empty" in proc.stdout
    assert "--generation_mode grpo_temperature" in proc.stdout
    assert "--grpo_noise_scale 3.0" in proc.stdout


def test_round_runner_perception_mining_writes_credit_rows_directly(tmp_path):
    cfg = {
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "mine_labels": ["moving_collision"],
        "perception_mining": {
            "tool": "direct_reproducer_chunks",
            "scene_list": "/data/scenes.json",
            "batch_size": 4,
        },
    }

    cmd, save_dir = _perception_mining_cmd(cfg, tmp_path / "model.pth", tmp_path / "round")

    assert "rlvr.autoresearch.tools.mine_direct_reproducer_chunks" in cmd
    assert "--out_jsonl" in cmd
    assert cmd[cmd.index("--out_jsonl") + 1] == str(tmp_path / "round" / "credit_windows.jsonl")
    assert "--scene_list" in cmd
    assert cmd[cmd.index("--scene_list") + 1] == "/data/scenes.json"
    assert "--labels" in cmd
    assert cmd[cmd.index("--labels") + 1] == "moving_collision"
    assert save_dir == tmp_path / "round" / "perception_danger_windows"


def test_shared_declustering_matches_replay_step_semantics():
    assert decluster_indices([1, 2, 4, 12, 13], window=10) == [1, 12]


def test_replay_step_declustering_zero_window_is_noop():
    assert _decluster_replay_steps([4, 1, 2], window=0) == [1, 2, 4]


def test_contiguous_index_runs_group_single_event():
    assert contiguous_index_runs([423, 424, 425, 433, 434], max_gap=9) == [
        [423, 424, 425, 433, 434]
    ]
    assert contiguous_index_runs([423, 424, 425, 436], max_gap=9) == [[423, 424, 425], [436]]


def test_credit_window_miner_selects_eta_closest_anchor_within_event(tmp_path):
    route = [tmp_path / f"bagA_{i:04d}.npz" for i in range(100, 200)]
    windows = [
        {
            "route_key": "bagA",
            "label": "road_border_crossing",
            "frame_index": 120 + i,
            "source_index": 20 + i,
            "violation_step": eta,
            "offense_index": 20 + i + eta,
            "offense_frame": 120 + i + eta,
            "credit_width": 15,
            "start_frame": 120 + i,
        }
        for i, eta in enumerate([27, 30, 25])
    ]

    [event] = _select_event_windows(
        windows,
        {"bagA": route},
        source_gap_steps=1,
        anchor_horizon_steps=26,
        max_rollout_steps=15,
    )

    assert event["source_index"] == 22
    assert event["frame_index"] == 122
    assert event["violation_step"] == 25
    assert event["event_source_start_index"] == 20
    assert event["event_source_end_index"] == 22
    assert event["event_member_count"] == 3
    assert event["start_index"] == 22
    assert event["end_index"] == 36


def test_credit_window_miner_anchor_uses_anchor_horizon_not_window_width(tmp_path):
    route = [tmp_path / f"bagA_{i:04d}.npz" for i in range(100, 200)]
    windows = [
        {
            "route_key": "bagA",
            "label": "road_border_crossing",
            "frame_index": 120 + i,
            "source_index": 20 + i,
            "violation_step": eta,
            "offense_index": 20 + i + eta,
            "offense_frame": 120 + i + eta,
            "credit_width": 40,
            "start_frame": 120 + i,
        }
        for i, eta in enumerate([14, 19, 25])
    ]

    [event] = _select_event_windows(
        windows,
        {"bagA": route},
        source_gap_steps=1,
        anchor_horizon_steps=20,
        max_rollout_steps=15,
    )

    assert event["source_index"] == 21
    assert event["anchor_horizon_steps"] == 20
    assert event["max_rollout_steps"] == 15
    assert event["end_index"] == 35


def test_credit_window_miner_respects_source_gap_grouping(tmp_path):
    route = [tmp_path / f"bagA_{i:04d}.npz" for i in range(100, 200)]
    windows = [
        {
            "route_key": "bagA",
            "label": "road_border_crossing",
            "frame_index": frame,
            "source_index": idx,
            "violation_step": 15,
            "offense_index": idx + 15,
            "offense_frame": frame + 15,
            "credit_width": 15,
            "start_frame": frame,
        }
        for idx, frame in [(20, 120), (21, 121), (25, 125)]
    ]

    events = _select_event_windows(windows, {"bagA": route}, source_gap_steps=2)
    assert len(events) == 2
    assert [event["source_index"] for event in events] == [20, 25]


def test_shared_sustained_runs_returns_whole_qualified_run():
    mask = torch.tensor([False, True, True, False, True, True, True]).numpy()
    assert sustained_true_indices(mask, min_steps=3) == {4, 5, 6}


def test_online_event_selector_saves_rising_edges_not_adjacent_frames():
    selector = OnlineEventSelector(decluster_steps=3)

    assert selector.update(0, ["static_collision"]) == ["static_collision"]
    assert selector.update(1, ["static_collision"]) == []
    assert selector.update(2, []) == []
    assert selector.update(3, ["static_collision"]) == ["static_collision"]


def test_online_event_selector_suppresses_short_clear_flicker():
    selector = OnlineEventSelector(decluster_steps=5)

    assert selector.update(10, ["moving_near_miss"]) == ["moving_near_miss"]
    assert selector.update(11, []) == []
    assert selector.update(12, ["moving_near_miss"]) == []
    assert selector.update(15, ["moving_near_miss"]) == []
    assert selector.update(16, []) == []
    assert selector.update(17, ["moving_near_miss"]) == ["moving_near_miss"]


def test_online_event_selector_tracks_labels_independently():
    selector = OnlineEventSelector(decluster_steps=10)

    assert selector.update(0, ["road_border_crossing"]) == ["road_border_crossing"]
    assert selector.update(1, ["road_border_crossing", "moving_ttc"]) == ["moving_ttc"]
    assert selector.update(2, ["moving_ttc"]) == []
    assert selector.update(3, ["lane_crossing", "moving_ttc"]) == ["lane_crossing"]


def test_verify_credit_rollout_saves_first_realized_event_window(monkeypatch, tmp_path):
    calls = []

    class _FakeModel:
        def __call__(self, data):
            pred = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
            return None, {
                "prediction": pred,
                "turn_indicator_logit": torch.zeros((1, 3), dtype=torch.float32),
            }

    class _FakeTimeline:
        frame_indices = np.arange(300, dtype=np.int64)

    def _fake_seed_state(*args, **kwargs):
        return SimpleNamespace(
            tl=_FakeTimeline(),
            start=100,
            end=103,
            ego_shape=np.ones(3, dtype=np.float32),
            clearances=np.full(8, np.inf, dtype=np.float32),
            collisions=np.zeros(8, dtype=bool),
            k=0,
            done=False,
            terminated="max_steps",
            max_steps=999,
            live_pose=np.zeros(3, dtype=np.float32),
            save_buf=None,
            last_snap_step=None,
            n_snaps=0,
            turn_hist=None,
            credit_window=None,
            credit_saved=False,
            verified_credit_labels=set(),
            verified_credit_first_step={},
            danger_event_selector=None,
        )

    def _fake_pre_step(s, gpu_transform=False):
        if s.k >= s.max_steps:
            s.done = True
            return None
        return (
            {"ego_shape": np.ones((1, 3), dtype=np.float32), "k": s.k},
            np.zeros((320, 11), dtype=np.float32),
            s.start + s.k,
            None,
            None,
        )

    def _fake_to_torch_batch(np_dicts, model_args, device):
        return {"dummy": torch.zeros((len(np_dicts), 1), dtype=torch.float32)}

    def _fake_score_step_batched(neighbors_list, ego_shapes, device):
        return [(1.0, False, 0, -1) for _ in neighbors_list]

    def _fake_realized_event_scorer(np_dict, *, collided):
        return (
            {"labels": ["road_border_crossing"], "label": "road_border_crossing"}
            if np_dict["k"] == 1
            else {"labels": ["clean"], "label": "clean"}
        )

    def _fake_advance_step(s, pred, idx, device, timers):
        s.k += 1

    def _fake_finalize(s, timers):
        return SimpleNamespace(metrics={"n_steps_run": s.k})

    def _fake_dump_credit_window(*args, **kwargs):
        calls.append(
            {
                "out_dir": args[0],
                "saved_steps": [rec[0] for rec in args[4]],
                "label": args[10],
                "realized_frame": kwargs["extra_manifest"]["realized_frame"],
                "source_label": kwargs["extra_manifest"]["source_label"],
            }
        )
        return {"n_scenes": len(args[4])}

    monkeypatch.setattr(reproducer_rollout, "_seed_state", _fake_seed_state)
    monkeypatch.setattr(reproducer_rollout, "_pre_step", _fake_pre_step)
    monkeypatch.setattr(reproducer_rollout, "_to_torch_batch", _fake_to_torch_batch)
    monkeypatch.setattr(reproducer_rollout, "score_step_batched", _fake_score_step_batched)
    monkeypatch.setattr(reproducer_rollout, "_advance_step", _fake_advance_step)
    monkeypatch.setattr(reproducer_rollout, "_finalize", _fake_finalize)
    monkeypatch.setattr(reproducer_rollout, "_dump_credit_window", _fake_dump_credit_window)

    reproducer_rollout.run_segments_batched(
        _FakeModel(),
        SimpleNamespace(predicted_neighbor_num=0, future_len=2, observation_normalizer=lambda x: x),
        [(_FakeTimeline(), 100, 103)],
        device="cpu",
        batch_size=1,
        n_build_threads=1,
        prefetch_ahead=0,
        verify_credit_windows=[
            {
                "route_key": "bagA",
                "label": "road_border_crossing",
                "start_frame": 423,
                "offense_frame": 438,
                "credit_width": 15,
                "frame_index": 423,
                "source_index": 100,
                "event_source_start_index": 100,
                "event_source_end_index": 101,
                "event_member_count": 2,
            }
        ],
        danger_save_dir=tmp_path,
        realized_event_scorer=_fake_realized_event_scorer,
        danger_credit_windows={"road_border_crossing": _credit_spec()},
    )

    assert len(calls) == 1
    assert calls[0]["saved_steps"] == [0, 1]
    assert calls[0]["label"] == "road_border_crossing"
    assert calls[0]["source_label"] == "road_border_crossing"
    assert calls[0]["realized_frame"] == 101


def test_direct_danger_window_saves_realized_moving_collision(monkeypatch, tmp_path):
    calls = []

    class _FakeTimeline:
        poses = np.zeros((4, 3), dtype=np.float32)
        frame_indices = np.arange(100, 104, dtype=np.int64)

        def prefetch(self, _items):
            return None

    class _FakeModel:
        def __call__(self, data):
            batch = data["dummy"].shape[0]
            return None, {
                "prediction": torch.zeros((batch, 1, 2, 4), dtype=torch.float32),
                "turn_indicator_logit": torch.zeros((batch, 2, 3), dtype=torch.float32),
            }

    def _fake_seed_state(
        tl,
        start,
        end,
        search_radius,
        warmup_steps,
        near_miss_thresh,
        goal_reach_m,
        max_stuck_steps,
        timers,
        *,
        max_steps,
        **_kwargs,
    ):
        return SimpleNamespace(
            tl=tl,
            start=start,
            end=end,
            k=0,
            max_steps=max_steps,
            done=False,
            terminated="running",
            replay_mode="clock",
            clearances=np.full(max_steps + 1, np.inf, dtype=np.float32),
            collisions=np.zeros(max_steps + 1, dtype=bool),
            near_miss_thresh=near_miss_thresh,
            ego_shape=np.ones(3, dtype=np.float32),
            live_pose=np.zeros(3, dtype=np.float32),
            save_buf=None,
            last_snap_step=None,
            n_snaps=0,
            turn_hist=None,
            credit_window=None,
            credit_saved=False,
            verified_credit_labels=set(),
            verified_credit_first_step={},
            danger_event_selector=None,
            output_route_key=None,
        )

    def _fake_pre_step(s, gpu_transform=False):
        if s.k >= 3:
            s.done = True
            return None
        return (
            {"ego_shape": np.ones((1, 3), dtype=np.float32), "k": s.k},
            np.zeros((320, 11), dtype=np.float32),
            s.start + s.k,
            None,
            None,
        )

    def _fake_to_torch_batch(np_dicts, model_args, device):
        return {"dummy": torch.zeros((len(np_dicts), 1), dtype=torch.float32)}

    score_calls = {"n": 0}

    def _fake_score_step_batched(neighbors_list, ego_shapes, device):
        score_calls["n"] += 1
        collided = score_calls["n"] >= 2
        return [(0.0, collided, 1, 0) for _ in neighbors_list]

    def _fake_danger_scorer(built, preds, data, device):
        return [{"labels": ["clean"], "label": "clean"} for _ in built]

    def _fake_realized_event_scorer(np_dict, *, collided):
        return (
            {"labels": ["moving_collision"], "label": "moving_collision"}
            if np_dict["k"] == 1
            else {"labels": ["clean"], "label": "clean"}
        )

    def _fake_advance_step(s, pred, idx, device, timers):
        s.k += 1

    def _fake_finalize(s, timers):
        return SimpleNamespace(metrics={"n_steps_run": s.k})

    def _fake_dump_credit_window(*args, **kwargs):
        calls.append(
            {
                "out_dir": args[0],
                "saved_steps": [rec[0] for rec in args[4]],
                "label": args[10],
            }
        )
        return {"n_scenes": len(args[4])}

    monkeypatch.setattr(reproducer_rollout, "_seed_state", _fake_seed_state)
    monkeypatch.setattr(reproducer_rollout, "_pre_step", _fake_pre_step)
    monkeypatch.setattr(reproducer_rollout, "_to_torch_batch", _fake_to_torch_batch)
    monkeypatch.setattr(reproducer_rollout, "score_step_batched", _fake_score_step_batched)
    monkeypatch.setattr(reproducer_rollout, "_advance_step", _fake_advance_step)
    monkeypatch.setattr(reproducer_rollout, "_finalize", _fake_finalize)
    monkeypatch.setattr(reproducer_rollout, "_dump_credit_window", _fake_dump_credit_window)

    reproducer_rollout.run_segments_batched(
        _FakeModel(),
        SimpleNamespace(predicted_neighbor_num=0, future_len=2, observation_normalizer=lambda x: x),
        [(_FakeTimeline(), 100, 103)],
        device="cpu",
        batch_size=1,
        n_build_threads=1,
        prefetch_ahead=0,
        route_keys=["bagA"],
        danger_save_dir=tmp_path,
        danger_scorer=_fake_danger_scorer,
        realized_event_scorer=_fake_realized_event_scorer,
        danger_credit_windows={"moving_collision": _credit_spec()},
    )

    assert len(calls) == 1
    assert calls[0]["label"] == "moving_collision"
    assert calls[0]["saved_steps"] == [0, 1]
    assert Path(calls[0]["out_dir"]).name == "bagA_100_101_danger_moving_collision"


def test_direct_danger_window_saves_raw_collision_when_scorers_miss(monkeypatch, tmp_path):
    calls = []

    class _FakeTimeline:
        poses = np.zeros((4, 3), dtype=np.float32)
        frame_indices = np.arange(100, 104, dtype=np.int64)

        def prefetch(self, _items):
            return None

    class _FakeModel:
        def __call__(self, data):
            batch = data["dummy"].shape[0]
            return None, {
                "prediction": torch.zeros((batch, 1, 2, 4), dtype=torch.float32),
                "turn_indicator_logit": torch.zeros((batch, 2, 3), dtype=torch.float32),
            }

    def _fake_seed_state(
        tl,
        start,
        end,
        search_radius,
        warmup_steps,
        near_miss_thresh,
        goal_reach_m,
        max_stuck_steps,
        timers,
        *,
        max_steps,
        **_kwargs,
    ):
        return SimpleNamespace(
            tl=tl,
            start=start,
            end=end,
            k=0,
            max_steps=max_steps,
            done=False,
            terminated="running",
            replay_mode="clock",
            clearances=np.full(max_steps + 1, np.inf, dtype=np.float32),
            collisions=np.zeros(max_steps + 1, dtype=bool),
            near_miss_thresh=near_miss_thresh,
            ego_shape=np.ones(3, dtype=np.float32),
            live_pose=np.zeros(3, dtype=np.float32),
            save_buf=None,
            last_snap_step=None,
            n_snaps=0,
            turn_hist=None,
            credit_window=None,
            credit_saved=False,
            verified_credit_labels=set(),
            verified_credit_first_step={},
            danger_event_selector=None,
            output_route_key=None,
        )

    def _fake_pre_step(s, gpu_transform=False):
        if s.k >= 2:
            s.done = True
            return None
        return (
            {"ego_shape": np.ones((1, 3), dtype=np.float32), "k": s.k},
            np.zeros((320, 11), dtype=np.float32),
            s.start + s.k,
            None,
            None,
        )

    def _fake_to_torch_batch(np_dicts, model_args, device):
        return {"dummy": torch.zeros((len(np_dicts), 1), dtype=torch.float32)}

    def _fake_score_step_batched(neighbors_list, ego_shapes, device):
        return [(0.0, True, 1, 0) for _ in neighbors_list]

    def _fake_clean_scorer(*_args, **_kwargs):
        return [{"labels": ["clean"], "label": "clean"}]

    def _fake_clean_realized(*_args, **_kwargs):
        return {"labels": ["clean"], "label": "clean"}

    def _fake_advance_step(s, pred, idx, device, timers):
        s.k += 1

    def _fake_finalize(s, timers):
        return SimpleNamespace(metrics={"n_steps_run": s.k})

    def _fake_dump_credit_window(*args, **kwargs):
        calls.append({"out_dir": args[0], "label": args[10]})
        return {"n_scenes": len(args[4])}

    monkeypatch.setattr(reproducer_rollout, "_seed_state", _fake_seed_state)
    monkeypatch.setattr(reproducer_rollout, "_pre_step", _fake_pre_step)
    monkeypatch.setattr(reproducer_rollout, "_to_torch_batch", _fake_to_torch_batch)
    monkeypatch.setattr(reproducer_rollout, "score_step_batched", _fake_score_step_batched)
    monkeypatch.setattr(reproducer_rollout, "_advance_step", _fake_advance_step)
    monkeypatch.setattr(reproducer_rollout, "_finalize", _fake_finalize)
    monkeypatch.setattr(reproducer_rollout, "_dump_credit_window", _fake_dump_credit_window)

    reproducer_rollout.run_segments_batched(
        _FakeModel(),
        SimpleNamespace(predicted_neighbor_num=0, future_len=2, observation_normalizer=lambda x: x),
        [(_FakeTimeline(), 100, 103)],
        device="cpu",
        batch_size=1,
        n_build_threads=1,
        prefetch_ahead=0,
        route_keys=["bagA"],
        danger_save_dir=tmp_path,
        danger_scorer=_fake_clean_scorer,
        realized_event_scorer=_fake_clean_realized,
        danger_credit_windows={"moving_collision": _credit_spec()},
    )

    assert len(calls) == 1
    assert calls[0]["label"] == "moving_collision"
    assert Path(calls[0]["out_dir"]).name == "bagA_100_100_danger_moving_collision"


def test_dump_full_credit_segment_uses_step_key_for_verified_frame(monkeypatch, tmp_path):
    def _fake_dump_precollision_window(
        out_dir,
        tl,
        model_args,
        offense_step,
        buf,
        last_snap_step,
        credit_width,
        save_thresh,
        seg_start,
        seg_end,
        **kwargs,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "collision00000.npz").write_bytes(b"npz")
        return {}

    def _fake_frame_id(_tl, idx):
        return 1000 + int(idx)

    monkeypatch.setattr(
        reproducer_rollout, "_dump_precollision_window", _fake_dump_precollision_window
    )
    monkeypatch.setattr(reproducer_rollout, "_frame_id", _fake_frame_id)

    out_dir = tmp_path / "credit_segment"
    buf = [
        (5, 20, {}, None, np.zeros((0, 11), dtype=np.float32), np.zeros(3, dtype=np.float32)),
        (6, 21, {}, None, np.zeros((0, 11), dtype=np.float32), np.zeros(3, dtype=np.float32)),
        (7, 22, {}, None, np.zeros((0, 11), dtype=np.float32), np.zeros(3, dtype=np.float32)),
    ]

    manifest = reproducer_rollout._dump_full_credit_segment(
        out_dir,
        SimpleNamespace(),
        SimpleNamespace(),
        buf,
        last_snap_step=None,
        seg_start=0,
        seg_end=3,
        label="moving_collision",
        verified_step=6,
    )

    assert manifest is not None
    saved = json.loads((out_dir / "manifest.json").read_text())
    assert saved["verified_first_step"] == 6
    assert saved["verified_first_frame_id"] == 1021


def test_mine_credit_window_main_forwards_allowed_labels_to_realized_verifier(
    monkeypatch, tmp_path
):
    route_path = tmp_path / "bagA_0100.npz"
    np.savez(route_path, dummy=np.zeros((1,), dtype=np.float32))
    route_scene_list = tmp_path / "route_scene_list.json"
    route_scene_list.write_text(json.dumps([str(route_path)]))
    classified = tmp_path / "classified.jsonl"
    classified.write_text(
        json.dumps(
            {
                "scene_path": str(route_path),
                "labels": ["moving_collision"],
                "moving_collision_step": 0,
            }
        )
        + "\n"
    )
    credit_cfg = tmp_path / "credit.json"
    credit_cfg.write_text(
        json.dumps(
            {
                "_frame_hz": 10,
                "_defaults": {"width_s": 1.5, "gap_s": 1.5},
                "moving_collision": {},
            }
        )
    )
    out_dir = tmp_path / "out"
    out_jsonl = tmp_path / "out.jsonl"
    stale_dir = out_dir / "bagA_100_099_event_road_border_crossing"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scene_frame_ids_saved": [99],
                "realized_label": "road_border_crossing",
            }
        )
    )
    np.savez(stale_dir / "credit+00000.npz", ego_agent_future=np.zeros((80, 4), np.float32))

    captured: dict[str, object] = {}

    class _FakeModel:
        def eval(self):
            return None

    def _fake_build_realized_event_scorer(**kwargs):
        captured["allowed_labels"] = kwargs.get("allowed_labels")
        return lambda _np_dict, *, collided=False: {
            "labels": ["moving_collision"],
            "label": "moving_collision",
            "realized_collision": bool(collided),
        }

    def _fake_run_segments_batched(*args, **kwargs):
        saved_dir = Path(kwargs["danger_save_dir"]) / "bagA_100_100_event_moving_collision"
        saved_dir.mkdir(parents=True, exist_ok=True)
        (saved_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "scene_frame_ids_saved": [100],
                    "realized_label": "moving_collision",
                }
            )
        )
        np.savez(saved_dir / "credit+00000.npz", ego_agent_future=np.zeros((80, 4), np.float32))
        return []

    monkeypatch.setattr(
        mine_credit_window_scenes_tool,
        "RouteTimeline",
        lambda paths, _sidecar=None: paths,
    )
    monkeypatch.setattr(
        mine_credit_window_scenes_tool,
        "load_model",
        lambda _model_path, _device: (_FakeModel(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        mine_credit_window_scenes_tool,
        "build_realized_event_scorer",
        _fake_build_realized_event_scorer,
    )
    monkeypatch.setattr(
        mine_credit_window_scenes_tool,
        "run_segments_batched",
        _fake_run_segments_batched,
    )
    monkeypatch.setattr(
        mine_credit_window_scenes_tool,
        "load_credit_windows",
        lambda _path: {"moving_collision": _credit_spec()},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_credit_window_scenes.py",
            "--classified_scenes_jsonl",
            str(classified),
            "--credit_window_config",
            str(credit_cfg),
            "--route_scene_list",
            str(route_scene_list),
            "--model_path",
            str(tmp_path / "model.pth"),
            "--out_dir",
            str(out_dir),
            "--out_jsonl",
            str(out_jsonl),
            "--reward_config",
            str(tmp_path / "reward.json"),
            "--threshold_config",
            str(tmp_path / "thresholds.json"),
            "--verify_reproduced_issue",
            "--labels",
            "moving_collision",
        ],
    )

    mine_credit_window_scenes_tool.main()

    assert captured["allowed_labels"] == {"moving_collision"}
    assert not stale_dir.exists()
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["label"] == "moving_collision"


def test_repair_candidate_selector_requires_safe_fix():
    source_row = {"repair_labels": ["road_border_crossing", "static_collision"]}
    candidate_rows = [
        {
            "moving_collision_step": None,
            "expert_disagreement": False,
            "labels": ["road_border_crossing"],
        },
        {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
    ]
    reward_rows = [
        SimpleNamespace(
            total=-20.0,
            collision_step=None,
            rb_crossing=True,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=1.0,
            rb_min_dist=0.0,
        ),
        SimpleNamespace(
            total=5.0,
            collision_step=None,
            rb_crossing=False,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=0.8,
            rb_min_dist=0.5,
        ),
    ]

    idx, meta = _best_safe_candidate(
        source_row,
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        require_conflict_clear=True,
    )

    assert idx == 1
    assert meta["selected_total"] == 5.0


def test_candidate_violation_score_does_not_double_count_expert_disagreement():
    row = {
        "labels": ["expert_disagreement", "road_border_near"],
        "expert_disagreement": True,
        "moving_collision_step": None,
    }

    score = _candidate_violation_score(row, SimpleNamespace())

    assert score == 1.5


def test_seed_state_tracker_mode_selects_mpc():
    class _MiniTimeline:
        def __init__(self):
            self.poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
            self.speeds = np.array([0.0, 1.0], dtype=np.float32)
            self._npz = {
                "ego_agent_past": np.zeros((31, 3), dtype=np.float32),
                "ego_shape": np.array([4.76, 7.24, 2.29], dtype=np.float32),
                "neighbor_agents_past": np.zeros((320, 31, 11), dtype=np.float32),
                "turn_indicators": np.zeros((31,), dtype=np.int64),
            }

        def __len__(self):
            return len(self.poses)

        def npz(self, idx):
            return self._npz

    tl = _MiniTimeline()
    timers = reproducer_rollout.Timers()

    state = reproducer_rollout._seed_state(
        tl,
        0,
        2,
        search_radius=1.5,
        warmup_steps=0,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        tracker_mode="mpc",
    )

    assert state.tracker.__class__.__name__ == "MPCTracker"


def test_repair_candidate_selector_breaks_ties_by_lower_deviation():
    source_row = {"repair_labels": ["road_border_crossing"]}
    candidate_rows = [
        {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
        {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
    ]
    reward_rows = [
        SimpleNamespace(
            total=1.0,
            collision_step=None,
            rb_crossing=False,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=1.0,
            rb_min_dist=1.0,
        ),
        SimpleNamespace(
            total=10.0,
            collision_step=None,
            rb_crossing=False,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=1.0,
            rb_min_dist=1.0,
        ),
    ]
    candidate_trajs = [
        torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.1, 0.0, 1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[5.0, 0.0, 1.0, 0.0], [5.1, 0.0, 1.0, 0.0]], dtype=torch.float32),
    ]
    reference_traj = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    idx, meta = _best_safe_candidate(
        source_row,
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        require_conflict_clear=True,
        candidate_trajs=candidate_trajs,
        reference_traj=reference_traj,
    )

    assert idx == 0
    assert meta["selected_deviation_penalty"] < 1.0


def test_source_scene_t0_moving_overlap_rejects_already_collided_scene():
    data = {
        "ego_shape": torch.tensor([[4.76, 7.24, 2.29]], dtype=torch.float32),
        "neighbor_agents_future": torch.tensor(
            [[[[-2.0, 0.0, 1.0, 0.0]]]],
            dtype=torch.float32,
        ),
        "neighbor_agents_past": torch.tensor(
            [[[[0.0] * 11 for _ in range(31)]]],
            dtype=torch.float32,
        ),
    }
    data["neighbor_agents_past"][0, 0, -1, 0] = -2.0
    data["neighbor_agents_past"][0, 0, -1, 2] = 1.0
    data["neighbor_agents_past"][0, 0, -1, 6] = 2.0
    data["neighbor_agents_past"][0, 0, -1, 7] = 4.5

    collided, min_clearance = _source_scene_t0_moving_overlap(
        data,
        RewardConfig(ignore_rear_end_collisions=False),
        device=torch.device("cpu"),
        moving_collision_thresh=0.2,
    )

    assert collided is True
    assert min_clearance < 0.0


def test_t0_dirty_source_discards_whole_event_window(monkeypatch):
    rows = [
        {
            "scene_path": "/tmp/event_a/credit-00030.npz",
            "window_dir": "/tmp/event_a",
            "labels": ["moving_collision"],
            "repair_labels": ["moving_collision"],
        },
        {
            "scene_path": "/tmp/event_a/credit-00029.npz",
            "window_dir": "/tmp/event_a",
            "labels": ["moving_collision"],
            "repair_labels": ["moving_collision"],
        },
        {
            "scene_path": "/tmp/event_b/credit-00030.npz",
            "window_dir": "/tmp/event_b",
            "labels": ["road_border_crossing"],
            "repair_labels": ["road_border_crossing"],
        },
    ]
    checked_paths: list[str] = []

    monkeypatch.setattr(
        build_avoiding_target_tool,
        "load_npz_data",
        lambda path, _device: {"scene_path": str(path)},
    )

    def _fake_t0_collision(data, **_kwargs):
        checked_paths.append(data["scene_path"])
        return data["scene_path"].endswith("credit-00029.npz"), -0.1

    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_source_scene_t0_any_neighbor_overlap",
        _fake_t0_collision,
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_source_scene_t0_road_border_crossing",
        lambda *_args, **_kwargs: (False, 1.0),
    )

    kept, dirty = _drop_t0_dirty_event_windows(
        rows,
        rcfg=RewardConfig(),
        thresholds={"moving_collision_thresh": 0.2, "rb_cross_thresh": 0.2},
        device=torch.device("cpu"),
    )

    assert [row["scene_path"] for row in kept] == ["/tmp/event_b/credit-00030.npz"]
    assert [row["scene_path"] for row in dirty] == [
        "/tmp/event_a/credit-00030.npz",
        "/tmp/event_a/credit-00029.npz",
    ]
    assert {row["reason"] for row in dirty} == {"event_window_t0_already_collided"}
    assert checked_paths == ["/tmp/event_a/credit-00030.npz", "/tmp/event_a/credit-00029.npz"]


def test_r2lpl_workflow_defaults_to_count_rear_end_collisions(tmp_path):
    cfg = {
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/threshold.json",
        "repair_config": {
            "ego_shape": "4.76,7.24,2.29",
            "min_margin": 0.3,
        },
        "count_rear_end_collisions": True,
    }

    repair_cmd = _repair_cmd(
        cfg,
        model_path=tmp_path / "model.pth",
        credit_jsonl=tmp_path / "credit.jsonl",
        rdir=tmp_path,
    )

    assert "--count_rear_end_collisions" in repair_cmd


def test_realized_event_scorer_uses_same_moving_collision_threshold(tmp_path):
    reward_cfg = tmp_path / "reward.json"
    threshold_cfg = tmp_path / "thresholds.json"
    reward_cfg.write_text(
        json.dumps(
            {
                "reward_mode": "gate",
                "w_progress": 2.0,
                "w_centerline": 5.0,
                "w_safety": 5.0,
                "w_smooth": 0.5,
                "w_feasibility": 5.0,
                "stopped_penalty": 50.0,
                "rb_gate_enabled": True,
                "rb_penalty_mode": "frac",
                "rb_cross_thresh": 0.2,
                "rb_near_thresh": 0.2,
                "rb_wide_thresh": 0.6,
                "rb_cont_thresh": 1.0,
                "rb_near_scale": 3.0,
                "rb_wide_scale": 0.2,
                "rb_cont_scale": 0.0,
                "enable_lane_departure": False,
                "lane_gate_enabled": False,
                "lane_cross_thresh": 0.2,
                "lane_near_thresh": 0.25,
                "lane_wide_thresh": 0.4,
                "lane_cont_thresh": 0.8,
                "lane_near_scale": 3.0,
                "lane_wide_scale": 0.2,
                "lane_cont_scale": 0.0,
                "centerline_usage_mode": "baselink",
                "enable_overprogress": False,
                "overprogress_margin": 1.1,
                "overprogress_penalty": 0.3,
                "progress_norm_scale": 20.0,
                "underprogress_penalty": 0.0,
                "underprogress_threshold": 0.5,
                "underprogress_reference": "baseline",
            }
        )
    )
    threshold_cfg.write_text(
        json.dumps(
            {
                "moving_collision_thresh": 0.2,
                "moving_near_thresh": 0.7,
                "static_near_thresh": 0.5,
                "rb_near_thresh": 0.2,
                "expert_disagreement_thresh": 1.0,
                "expert_disagreement_sustain_steps": 10,
                "sc_cross_thresh": 0.2,
                "rb_cross_thresh": 0.2,
            }
        )
    )

    scorer = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"moving_collision"},
    )
    np_dict = {
        "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
        "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
    }
    np_dict["neighbor_agents_past"][0, -1, 0] = 4.52
    np_dict["neighbor_agents_past"][0, -1, 2] = 1.0
    np_dict["neighbor_agents_past"][0, -1, 6] = 2.0
    np_dict["neighbor_agents_past"][0, -1, 7] = 4.5
    np_dict["neighbor_agents_future"][0, :, 0] = 4.52
    np_dict["neighbor_agents_future"][0, :, 2] = 1.0

    row = scorer(np_dict, collided=False)

    assert "moving_collision" in row["labels"]
    assert row["moving_collision_step"] == 0


def test_realized_event_scorer_supports_static_collision(tmp_path):
    reward_cfg = tmp_path / "reward.json"
    threshold_cfg = tmp_path / "thresholds.json"
    reward_cfg.write_text(
        json.dumps(
            {
                "reward_mode": "gate",
                "w_progress": 2.0,
                "w_centerline": 5.0,
                "w_safety": 5.0,
                "w_smooth": 0.5,
                "w_feasibility": 5.0,
                "stopped_penalty": 50.0,
                "rb_gate_enabled": True,
                "rb_penalty_mode": "frac",
                "rb_cross_thresh": 0.2,
                "rb_near_thresh": 0.2,
                "rb_wide_thresh": 0.6,
                "rb_cont_thresh": 1.0,
                "rb_near_scale": 3.0,
                "rb_wide_scale": 0.2,
                "rb_cont_scale": 0.0,
                "enable_lane_departure": False,
                "lane_gate_enabled": False,
                "lane_cross_thresh": 0.2,
                "lane_near_thresh": 0.25,
                "lane_wide_thresh": 0.4,
                "lane_cont_thresh": 0.8,
                "lane_near_scale": 3.0,
                "lane_wide_scale": 0.2,
                "lane_cont_scale": 0.0,
                "centerline_usage_mode": "baselink",
                "enable_overprogress": False,
                "overprogress_margin": 1.1,
                "overprogress_penalty": 0.3,
                "progress_norm_scale": 20.0,
                "underprogress_penalty": 0.0,
                "underprogress_threshold": 0.5,
                "underprogress_reference": "baseline",
            }
        )
    )
    threshold_cfg.write_text(
        json.dumps(
            {
                "moving_collision_thresh": 0.2,
                "moving_near_thresh": 0.7,
                "static_near_thresh": 0.5,
                "rb_near_thresh": 0.2,
                "expert_disagreement_thresh": 1.0,
                "expert_disagreement_sustain_steps": 10,
                "sc_cross_thresh": 0.2,
                "rb_cross_thresh": 0.2,
            }
        )
    )

    scorer = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"static_collision"},
    )
    np_dict = {
        "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
        "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
    }
    np_dict["neighbor_agents_past"][0, -1, 0] = 3.0
    np_dict["neighbor_agents_past"][0, -1, 2] = 1.0
    np_dict["neighbor_agents_past"][0, -1, 6] = 2.0
    np_dict["neighbor_agents_past"][0, -1, 7] = 4.5
    np_dict["neighbor_agents_future"][0, :, 0] = 3.0
    np_dict["neighbor_agents_future"][0, :, 2] = 1.0

    row = scorer(np_dict, collided=True)

    assert "static_collision" in row["labels"]
    assert row["static_collision_step"] == 0
    assert row["stopped_neighbor_count"] == 1


def test_union_scene_lists_dedupes_current_and_replay(tmp_path):
    out = tmp_path / "train.json"
    _union_scene_lists(["/a.npz", "/b.npz"], ["/b.npz", "/c.npz"], out)
    assert json.loads(out.read_text()) == ["/a.npz", "/b.npz", "/c.npz"]


def test_filtered_npz_payload_keeps_only_training_fields():
    payload = _filtered_npz_payload(
        {
            "ego_agent_future": np.zeros((80, 4), dtype=np.float32),
            "neighbor_agents_past": np.zeros((320, 31, 11), dtype=np.float32),
            "version": np.array(1, dtype=np.uint32),
            "origin": np.asarray("map"),
            "token": np.asarray("abc"),
        }
    )

    assert "ego_agent_future" in payload
    assert "neighbor_agents_past" in payload
    assert "version" not in payload
    assert "origin" not in payload
    assert "token" not in payload


def test_future4_to_3col_restores_yaw():
    traj = np.array([[1.0, 2.0, 0.0, 1.0], [3.0, 4.0, 1.0, 0.0]], dtype=np.float32)
    out = _future4_to_3col(traj)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, :2], traj[:, :2])
    assert np.allclose(out[:, 2], np.array([np.pi / 2, 0.0], dtype=np.float32))


def test_build_avoiding_target_accepts_scene_local_ego_shape():
    assert _parse_ego_shape("from_npz") is None
    assert _parse_ego_shape("npz") is None
    assert np.allclose(_parse_ego_shape("4.76,7.24,2.29"), [4.76, 7.24, 2.29])


def test_build_repaired_targets_preserves_simulated_context(monkeypatch, tmp_path):
    source = tmp_path / "sim_windows" / "credit+00000.npz"
    source.parent.mkdir()
    sim_ego_past = np.arange(31 * 3, dtype=np.float32).reshape(31, 3) + 10.0
    sim_current = np.linspace(0.0, 9.0, 10, dtype=np.float32) + 20.0
    sim_neighbors = np.full((2, 31, 11), 3.5, dtype=np.float32)
    sim_ego_shape = np.array([4.99, 10.74, 2.56], dtype=np.float32)
    sim_goal_pose = np.array([42.0, 24.0, 0.5], dtype=np.float32)
    sim_lanes = np.full((70, 20, 13), 1.25, dtype=np.float32)
    sim_lanes_speed_limit = np.ones((70, 1), dtype=np.float32)
    sim_lanes_has_speed_limit = np.ones((70, 1), dtype=np.bool_)
    sim_turn_indicators = np.zeros((31,), dtype=np.int64)
    np.savez(
        source,
        ego_shape=sim_ego_shape,
        ego_agent_past=sim_ego_past,
        ego_current_state=sim_current,
        goal_pose=sim_goal_pose,
        neighbor_agents_past=sim_neighbors,
        lanes=sim_lanes,
        lanes_speed_limit=sim_lanes_speed_limit,
        lanes_has_speed_limit=sim_lanes_has_speed_limit,
        turn_indicators=sim_turn_indicators,
        ego_agent_future=np.full((80, 4), -5.0, dtype=np.float32),
        origin=np.asarray("sim"),
    )

    data = {
        "ego_shape": torch.tensor(sim_ego_shape[None], dtype=torch.float32),
        "ego_agent_future": torch.zeros((80, 4), dtype=torch.float32),
    }
    selected = torch.zeros((80, 4), dtype=torch.float32)
    selected[:, 0] = torch.arange(80, dtype=torch.float32)
    selected[:, 2] = 1.0

    monkeypatch.setattr(
        build_avoiding_target_tool,
        "load_reward_config",
        lambda _path: RewardConfig(ignore_rear_end_collisions=False),
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_load_scene_thresholds",
        lambda _path: {
            "moving_collision_thresh": 0.2,
            "moving_near_thresh": 0.7,
            "static_near_thresh": 0.5,
            "rb_near_thresh": 0.2,
            "expert_disagreement_thresh": 1.0,
            "expert_disagreement_sustain_steps": 10,
        },
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "load_model",
        lambda _model_path, _device: (object(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "load_npz_data",
        lambda _path, _device: data,
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_source_scene_t0_moving_overlap",
        lambda *_args, **_kwargs: (False, 99.0),
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_stack_scene_data",
        lambda _datas, _device: {},
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_normalize_batch",
        lambda _batch, _model_args: {},
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "generate_all_scenes_batched",
        lambda *_args, **_kwargs: [torch.stack([selected])],
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "classify_loaded_scene_candidates_batch",
        lambda *_args, **_kwargs: [[{"labels": ["clean"]}]],
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "compute_reward_batch",
        lambda *_args, **_kwargs: [SimpleNamespace(total=1.0)],
    )
    monkeypatch.setattr(
        build_avoiding_target_tool,
        "_best_safe_candidate",
        lambda *_args, **_kwargs: (
            0,
            {
                "selected_candidate_index": 0,
                "selected_total": 1.0,
                "selected_labels": ["clean"],
            },
        ),
    )

    rows_jsonl = tmp_path / "repaired.jsonl"
    paths, unrepaired = build_avoiding_target_tool.build_repaired_targets(
        model_path="model.pth",
        rows=[
            {
                "scene_path": str(source),
                "labels": ["moving_collision"],
                "repair_labels": ["moving_collision"],
            }
        ],
        reward_config_path="reward.json",
        threshold_config_path="thresholds.json",
        ego_shape=None,
        out_dir=tmp_path / "repaired",
        out_rows_jsonl=rows_jsonl,
        out_list=tmp_path / "repaired.json",
        min_static_margin=0.3,
        K=1,
        variant="rl_cl_soft_sweep_stretch",
        generation_mode="guided_variant",
        grpo_noise_scale=3.0,
        gt_max_speed=9.0,
        scene_batch_size=1,
        noise_low=0.5,
        noise_high=2.0,
        device=torch.device("cpu"),
        require_conflict_clear=True,
        enable_conflict_detector=True,
        use_route_cl_guidance=False,
        count_rear_end_collisions=True,
    )

    assert len(paths) == 1
    assert unrepaired == []
    repaired_rows = _read_test_jsonl(rows_jsonl)
    assert repaired_rows[0]["source_scene_path"] == str(source)
    assert np.allclose(repaired_rows[0]["ego_shape"], sim_ego_shape)
    with np.load(paths[0]) as repaired:
        assert np.allclose(repaired["ego_shape"], sim_ego_shape)
        assert np.allclose(repaired["ego_agent_past"], sim_ego_past)
        assert np.allclose(repaired["ego_current_state"], sim_current)
        assert np.allclose(repaired["goal_pose"], sim_goal_pose)
        assert np.allclose(repaired["neighbor_agents_past"], sim_neighbors)
        assert np.allclose(repaired["lanes"], sim_lanes)
        assert np.allclose(repaired["lanes_speed_limit"], sim_lanes_speed_limit)
        assert np.array_equal(repaired["lanes_has_speed_limit"], sim_lanes_has_speed_limit)
        assert np.array_equal(repaired["turn_indicators"], sim_turn_indicators)
        assert repaired["ego_agent_future"].shape == (80, 3)
        assert np.allclose(repaired["ego_agent_future"][:, 0], np.arange(80))
        assert "origin" not in repaired.files


def test_base_train_invocation_uses_cumulative_epochs_and_train_predictor(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "latest.pth"
    torch.save({"epoch": 5, "model": {}}, model_path)
    (model_dir / "args.json").write_text(
        json.dumps(
            {
                "normalization_file_path": "normalization.json",
                "batch_size": 4,
                "num_workers": 2,
                "ddp": True,
                "pin_mem": True,
                "learning_rate": 1e-4,
                "predicted_neighbor_num": 320,
                "agent_num": 320,
            }
        )
    )
    (model_dir / "normalization.json").write_text(
        json.dumps(
            {
                "ego": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
                "neighbor": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
            }
        )
    )
    train_list = tmp_path / "train.json"
    train_list.write_text(json.dumps(["/a.npz", "/b.npz"]))
    cfg = {
        "training_config": {
            "train_args": {
                "batch_size": 8,
                "num_workers": 1,
            },
            "nproc_per_node": 1,
        },
        "epochs_per_round": 2,
        "val_scenes": "/tmp/val.json",
    }

    cmd, next_model, cwd, env = _base_train_invocation(
        cfg,
        model_path=model_path,
        train_list=train_list,
        rdir=tmp_path / "round",
        round_idx=3,
    )

    assert next_model == tmp_path / "round" / "base_train" / "latest.pth"
    assert cwd == Path(__file__).resolve().parents[3] / "diffusion_planner"
    assert str(Path(__file__).resolve().parents[3]) in env["PYTHONPATH"].split(":")
    assert str(Path(__file__).resolve().parents[3] / "diffusion_planner") in env[
        "PYTHONPATH"
    ].split(":")
    assert "-m" in cmd
    assert "train_predictor" in cmd
    assert cmd[cmd.index("--train_epochs") + 1] == "7"
    assert cmd[cmd.index("--batch_size") + 1] == "2"


def test_torchrun_subprocess_cleanup_removes_stale_file_store(tmp_path, monkeypatch):
    stale_store = tmp_path / "tmp_dist_init"
    stale_store.write_text("stale")
    monkeypatch.setattr(round_runner, "_TORCH_DDP_FILE_STORE", stale_store)

    round_runner._cleanup_torch_dist_file_store(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "1"]
    )

    assert not stale_store.exists()


def test_non_torchrun_subprocess_cleanup_leaves_file_store(tmp_path, monkeypatch):
    stale_store = tmp_path / "tmp_dist_init"
    stale_store.write_text("stale")
    monkeypatch.setattr(round_runner, "_TORCH_DDP_FILE_STORE", stale_store)

    round_runner._cleanup_torch_dist_file_store([sys.executable, "-m", "json.tool"])

    assert stale_store.read_text() == "stale"


def test_sft_replay_weights_apply_per_scene_in_mixed_batch():
    device, model_args, data, neighbor_gt, neighbor_mask = _minimal_sft_batch()
    ego_gt = torch.zeros(2, 3, 4, device=device)
    # Scene 0 has zero loss. Scene 1 has per-step ego loss:
    # lon |0-1| = 1 and heading (0-1)^2 = 1, total = 2.
    ego_gt[1, :, 0] = 1.0
    ego_gt[1, :, 2] = 1.0

    model = _ConstantDenoiser(0.0)
    loss, metrics = _compute_sft_diffusion_loss(
        model=model,
        model_args=model_args,
        data=data,
        ego_gt=ego_gt,
        neighbor_gt=neighbor_gt,
        neighbor_mask=neighbor_mask,
        device=device,
        K=1,
        neighbor_loss_weight=0.0,
        scene_loss_weights=torch.tensor([1.0, 3.0]),
    )

    assert torch.isclose(loss, torch.tensor(3.0), atol=1e-6)
    assert metrics["sft_scene_weight_mean"] == 2.0
    loss.backward()
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)


def test_sft_der_applies_only_to_replay_rows_in_mixed_batch():
    device, model_args, data, neighbor_gt, neighbor_mask = _minimal_sft_batch()
    ego_gt = torch.zeros(2, 3, 4, device=device)

    model = _ConstantDenoiser(0.0)
    loss, metrics = _compute_sft_diffusion_loss(
        model=model,
        model_args=model_args,
        data=data,
        ego_gt=ego_gt,
        neighbor_gt=neighbor_gt,
        neighbor_mask=neighbor_mask,
        device=device,
        K=1,
        neighbor_loss_weight=0.0,
        replay_der_coef=2.0,
        replay_anchor_model=_ConstantDenoiser(1.0),
        replay_der_mask=torch.tensor([False, True]),
    )

    # Only scene 1 is replay. Its DER MSE is 1.0, coef=2.0, then the batch
    # averages over B=2 scenes: (0 + 2) / 2 = 1.
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)
    assert metrics["sft_replay_der_loss"] == 1.0
    assert metrics["sft_replay_count"] == 1.0
    loss.backward()
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)
