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
from rlvr.autoresearch.tools import build_repaired_targets as build_repaired_targets_tool
from rlvr.autoresearch.tools import mine_credit_window_scenes as mine_credit_window_scenes_tool
from rlvr.autoresearch.tools import (
    mine_direct_reproducer_chunks as mine_direct_reproducer_chunks_tool,
)
from rlvr.autoresearch.tools import reproducer_danger_scorer
from rlvr.autoresearch.tools import run_lifelong_r2lpl_rounds as round_runner
from rlvr.autoresearch.tools.build_repaired_targets import (
    _best_safe_candidate,
    _candidate_violation_score,
    _drop_t0_dirty_event_windows,
    _expert_reference_scoring_data,
    _filtered_npz_payload,
    _future4_to_3col,
    _parse_ego_shape,
    _row_is_expert_disagreement,
    _validate_static_collision_config,
)
from rlvr.autoresearch.tools.expert_morph import (
    build_depart_morph_candidate,
    build_expert_morph_candidate,
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
from rlvr.autoresearch.tools.reproducer_danger_scorer import (
    REALIZED_LAG_SUSTAIN_STEPS,
    build_realized_event_scorer,
)
from rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds import (
    _apply_repair_refresh_config,
    _base_train_invocation,
    _gpu_ids_from_config,
    _load_config,
    _lora_for_policy,
    _perception_mining_cmd,
    _refresh_repair,
    _repair_cmd,
    _repair_refresh_chunk_sizes,
    _repair_refresh_chunk_targets,
    _run_mining_phase,
    _run_repair_phase,
    _split_jsonl_by_event,
    _union_scene_lists,
)
from rlvr.deviation import rollout_gt_deviation
from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss
from rlvr.reward import RewardConfig
from scenario_generation.conflict_detector import (
    detect_expert_disagreement,
    detect_expert_disagreement_projected,
)
from scenario_generation.danger_event_selection import (
    OnlineEventSelector,
    contiguous_index_runs,
    decluster_indices,
    sustained_true_indices,
)
from scenario_generation.tools.classify_replay_steps import _decluster as _decluster_replay_steps


def _reward_row(total: float = 1.0, **overrides):
    """Stub of one ``compute_reward_batch`` row.

    The real row carries the per-term subscores that the realized-reward component
    telemetry reads (centerline / feasibility / progress / gate flags), so a stub with
    only ``total`` diverges from the production interface and the telemetry raises
    AttributeError. Keep every field the callers touch here, in ONE place.
    """
    fields = {
        "total": float(total),
        "centerline": 0.0,
        "feasibility": 0.0,
        "progress": 0.0,
        "collision_step": None,
        "rb_crossing": False,
        "kinematic_violated": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


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


def _write_realized_scorer_configs(
    tmp_path: Path,
    *,
    wait_speed_mps: float = 0.5,
    wait_progress_m: float = 1.0,
    forward_progress_gap_m: float = 2.0,
    lag_progress_gap_m: float = 3.0,
    moving_speed_mps: float = 1.0,
) -> tuple[Path, Path]:
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
                "expert_disagreement_wait_speed_mps": wait_speed_mps,
                "expert_disagreement_wait_progress_m": wait_progress_m,
                "expert_disagreement_forward_progress_gap_m": forward_progress_gap_m,
                "expert_disagreement_lag_progress_gap_m": lag_progress_gap_m,
                "expert_disagreement_moving_speed_mps": moving_speed_mps,
                "sc_cross_thresh": 0.2,
                "rb_cross_thresh": 0.2,
            }
        )
    )
    return reward_cfg, threshold_cfg


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
                "replay_memory": {"capacity": 200},
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
            "expert_disagreement": True,
            "expert_disagreement_step": 6,
            "expert_disagreement_max_dev": 2.5,
            "expert_disagreement_reason": "model_lagging_expert",
            "expert_disagreement_model_end_progress": 1.0,
            "expert_disagreement_expert_end_progress": 4.0,
            "expert_disagreement_model_end_speed": 0.2,
            "expert_disagreement_expert_end_speed": 3.0,
        },
        "moving_collision",
    )

    assert row["segment"] == [10, 90]
    assert row["segment_route_indices"] == [10, 90]
    assert row["segment_frame_ids"] == [110, 190]
    assert row["expert_disagreement"] is True
    assert row["expert_disagreement_step"] == 6
    assert row["expert_disagreement_max_dev"] == 2.5
    assert row["expert_disagreement_reason"] == "model_lagging_expert"
    assert row["expert_disagreement_model_end_progress"] == 1.0
    assert row["expert_disagreement_expert_end_progress"] == 4.0
    assert row["expert_disagreement_model_end_speed"] == 0.2
    assert row["expert_disagreement_expert_end_speed"] == 3.0


def test_mine_direct_main_forwards_realized_hard_event_scorer(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_build_realized_event_scorer(**kwargs):
        captured["realized_allowed_labels"] = kwargs.get("allowed_labels")
        return "realized-scorer"

    def _fake_run_segments_batched(*args, **kwargs):
        captured["realized_event_scorer"] = kwargs.get("realized_event_scorer")
        captured["danger_scorer"] = kwargs.get("danger_scorer")
        return [
            {
                "object": {"collision_steps": 1, "clearance_min_m": -0.1},
            }
        ]

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

    # Closed-loop reproducer mining is realized-only: no predicted danger_scorer.
    assert captured["danger_scorer"] is None
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
            # The plan pass is cached at the CAMPAIGN level (rdir.parent) so
            # later rounds reuse it instead of re-planning the full pool.
            assert cmd[cmd.index("--chunk_manifest") + 1] == str(
                rdir.parent / "planned_chunks.jsonl"
            )
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


def test_repair_shard_split_keeps_event_windows_atomic(tmp_path):
    rows = [
        {"scene_path": "/data/event_a/credit+00000.npz", "window_dir": "/data/event_a"},
        {"scene_path": "/data/event_b/credit+00000.npz", "window_dir": "/data/event_b"},
        {"scene_path": "/data/event_a/credit-00001.npz", "window_dir": "/data/event_a"},
        {"scene_path": "/data/event_b/credit-00001.npz", "window_dir": "/data/event_b"},
    ]
    paths = [tmp_path / f"shard_{idx}.jsonl" for idx in range(2)]

    _split_jsonl_by_event(rows, paths)

    for path in paths:
        shard_rows = _read_test_jsonl(path)
        assert len({row["window_dir"] for row in shard_rows}) <= 1
        assert len(shard_rows) == 2


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
    rollout = torch.zeros(20, 4)
    rollout[:, 0] = torch.arange(20, dtype=torch.float32) * 0.2

    result = detect_expert_disagreement(
        rollout,
        gt,
        enabled=False,
        wait_speed_mps=0.5,
        wait_progress_m=1.0,
        forward_progress_gap_m=2.0,
        lag_progress_gap_m=3.0,
        moving_speed_mps=1.0,
    )

    assert not result.expert_disagreement
    assert result.expert_disagreement_step is None


def test_conflict_detector_flags_expert_wait_model_forward():
    gt = torch.zeros(20, 4)
    rollout = torch.zeros(20, 4)
    rollout[:, 0] = torch.arange(20, dtype=torch.float32) * 0.2

    result = detect_expert_disagreement(
        rollout,
        gt,
        enabled=True,
        wait_speed_mps=0.5,
        wait_progress_m=1.0,
        forward_progress_gap_m=2.0,
        lag_progress_gap_m=3.0,
        moving_speed_mps=1.0,
    )

    assert result.expert_disagreement
    # This detector flags from horizon summaries and does not localize a step.
    assert result.expert_disagreement_step is None
    assert result.reason == "expert_wait_model_forward"


def test_conflict_detector_flags_lag_and_ahead():
    gt = torch.zeros(20, 4)
    gt[:, 0] = torch.arange(20, dtype=torch.float32) * 0.25
    rollout = torch.zeros(20, 4)

    lag = detect_expert_disagreement(
        rollout,
        gt,
        enabled=True,
        wait_speed_mps=0.5,
        wait_progress_m=1.0,
        forward_progress_gap_m=2.0,
        lag_progress_gap_m=3.0,
        moving_speed_mps=1.0,
    )
    ahead = detect_expert_disagreement(
        gt,
        rollout,
        enabled=True,
        wait_speed_mps=0.5,
        wait_progress_m=1.0,
        forward_progress_gap_m=2.0,
        lag_progress_gap_m=3.0,
        moving_speed_mps=1.0,
    )

    assert lag.reason == "model_lagging_expert"
    assert ahead.reason in {"expert_wait_model_forward", "model_ahead_expert"}


_PROJECTED_THRESHOLDS = dict(
    wait_speed_mps=0.5,
    wait_progress_m=1.0,
    forward_progress_gap_m=2.0,
    lag_progress_gap_m=3.0,
    moving_speed_mps=1.0,
)


def test_projected_disagreement_flags_expert_wait_model_forward():
    # Expert idle at the origin, model runs forward.
    result = detect_expert_disagreement_projected(
        model_progress=np.array([0.0, 3.0, 6.0, 9.0, 12.0]),
        model_speed=np.zeros(5),
        expert_progress=np.zeros(6),
        expert_speed=np.zeros(6),
        **_PROJECTED_THRESHOLDS,
    )
    assert result.expert_disagreement
    assert result.reason == "expert_wait_model_forward"


def test_projected_disagreement_flags_model_lagging_expert():
    result = detect_expert_disagreement_projected(
        model_progress=np.zeros(5),
        model_speed=np.zeros(5),
        expert_progress=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        expert_speed=np.full(6, 2.0),
        **_PROJECTED_THRESHOLDS,
    )
    assert result.expert_disagreement
    assert result.reason == "model_lagging_expert"


def test_projected_disagreement_flags_model_ahead_expert():
    # Model far ahead of a slow expert (not waiting, not the lagging speed regime).
    result = detect_expert_disagreement_projected(
        model_progress=np.array([10.0, 10.0, 10.0, 10.0]),
        model_speed=np.zeros(4),
        expert_progress=np.array([0.0, 9.0, 9.0, 9.0, 5.0]),
        expert_speed=np.full(5, 0.4),
        **_PROJECTED_THRESHOLDS,
    )
    assert result.expert_disagreement
    assert result.reason == "model_ahead_expert"
    # +1 shift: expert_end reads expert_progress[4]=5.0, NOT the unshifted [3]=9.0
    # (which would make the ahead-gap 1.0 and NOT trip the 2.0 threshold).
    assert result.expert_end_progress == pytest.approx(5.0)


def test_projected_disagreement_none_when_consistent():
    result = detect_expert_disagreement_projected(
        model_progress=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        model_speed=np.full(5, 0.5),
        expert_progress=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5]),
        expert_speed=np.full(6, 1.5),
        **_PROJECTED_THRESHOLDS,
    )
    assert not result.expert_disagreement
    assert result.reason == ""


def test_projected_disagreement_expert_plus_one_shift_is_applied():
    # Inputs already start at 0 (relative), so relativization is a no-op here.
    # With the +1 shift the reported expert_end is expert_progress[horizon]=5.0;
    # without it, it would be expert_progress[:horizon][-1]=30.0.
    result = detect_expert_disagreement_projected(
        model_progress=np.array([0.0, 1.0, 2.0, 3.0]),
        model_speed=np.zeros(4),
        expert_progress=np.array([0.0, 10.0, 20.0, 30.0, 5.0]),
        expert_speed=np.full(5, 0.4),
        **_PROJECTED_THRESHOLDS,
    )
    assert result.expert_end_progress == pytest.approx(5.0)


def test_projected_disagreement_relativizes_absolute_progress():
    # Progress passed as ABSOLUTE route arc (mid-route: large offset) must be
    # relativized to the current pose so the absolute wait/forward thresholds
    # apply to horizon advancement. Expert waits, model drives forward ->
    # expert_wait_model_forward must fire despite the large absolute offset.
    abs_offset = 250.0
    result = detect_expert_disagreement_projected(
        model_progress=abs_offset + np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        model_speed=np.full(5, 2.0),
        expert_progress=abs_offset + np.zeros(5),
        expert_speed=np.zeros(5),
        **_PROJECTED_THRESHOLDS,
    )
    assert result.expert_disagreement
    assert result.reason == "expert_wait_model_forward"
    assert result.expert_end_progress == pytest.approx(0.0, abs=1e-5)


def test_projected_disagreement_rejects_nonpositive_threshold():
    bad = dict(_PROJECTED_THRESHOLDS)
    bad["moving_speed_mps"] = 0.0
    with pytest.raises(ValueError, match="moving_speed_mps"):
        detect_expert_disagreement_projected(
            model_progress=np.zeros(4),
            model_speed=np.zeros(4),
            expert_progress=np.zeros(5),
            expert_speed=np.zeros(5),
            **bad,
        )


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


def test_replay_memory_label_quotas_reserve_capacity():
    # 20 high-priority rb rows vs 4 low-priority expert rows at capacity 8:
    # without quotas the expert rows are crowded out beyond the single
    # rare-bucket pick; a 0.5 expert quota must retain floor(0.5*8)=4 of them.
    rows = [
        {
            "scene_path": f"/tmp/rb_{i}.npz",
            "label": "road_border_crossing",
            "route_arc_m": float(i),  # spread over arc bins
            "variant_kind": "credit",
            "difficulty": 100.0 + i,
        }
        for i in range(20)
    ]
    rows += [
        {
            "scene_path": f"/tmp/expert_{i}.npz",
            "label": "expert_disagreement",
            "route_arc_m": 0.0,
            "variant_kind": "credit",
            "difficulty": float(i),
        }
        for i in range(4)
    ]

    baseline = build_memory(
        rows, {"entries": []}, {}, capacity=8, alpha=1.0, beta=0.0, arc_bin_m=1000.0
    )
    n_expert_baseline = sum(1 for r in baseline["entries"] if r["label"] == "expert_disagreement")

    memory = build_memory(
        rows,
        {"entries": []},
        {},
        capacity=8,
        alpha=1.0,
        beta=0.0,
        arc_bin_m=1000.0,
        label_quotas={"expert_disagreement": 0.5},
    )
    n_expert = sum(1 for r in memory["entries"] if r["label"] == "expert_disagreement")
    assert len(memory["entries"]) == 8
    assert n_expert == 4
    assert n_expert > n_expert_baseline
    assert memory["label_quotas"] == {"expert_disagreement": 0.5}


def test_replay_memory_label_quotas_parse_rejects_bad_spec():
    from rlvr.autoresearch.tools.lifelong_replay_memory import _parse_label_quotas

    assert _parse_label_quotas(None) == {}
    assert _parse_label_quotas("a=0.25,b=0.5") == {"a": 0.25, "b": 0.5}
    with pytest.raises(ValueError):
        _parse_label_quotas("nofraction")
    with pytest.raises(ValueError):
        _parse_label_quotas("a=1.5")
    with pytest.raises(ValueError):
        _parse_label_quotas("a=0.7,b=0.7")
    with pytest.raises(ValueError):
        _parse_label_quotas("a=0.2,a=0.3")  # duplicate label


def test_validate_anchor_config_rejects_empty_section():
    with pytest.raises(ValueError):
        round_runner._validate_anchor_config({"anchor": {}})
    # absent anchor is fine
    round_runner._validate_anchor_config({})


def test_anchor_slice_paths_ratio_and_waits_stratification(tmp_path):
    normal = [f"/tmp/normal_{i}.npz" for i in range(100)]
    waits = [f"/tmp/waits_{i}.npz" for i in range(50)]
    normal_json = tmp_path / "normal.json"
    waits_json = tmp_path / "waits.json"
    normal_json.write_text(json.dumps(normal))
    waits_json.write_text(json.dumps(waits))

    cfg = {
        "scene_list": str(normal_json),
        "ratio": 1.6,
        "waits_scene_list": str(waits_json),
        "waits_fraction": 0.25,
        "seed": 7,
    }
    picked = round_runner._anchor_slice_paths(cfg, n_focus=10, round_idx=1)
    assert len(picked) == 16  # ceil(1.6 * 10)
    n_waits = sum(1 for p in picked if "waits_" in p)
    assert n_waits == 4  # round(0.25 * 16)
    assert len(set(picked)) == len(picked)
    # reproducible per round, redrawn across rounds
    assert picked == round_runner._anchor_slice_paths(cfg, n_focus=10, round_idx=1)
    assert picked != round_runner._anchor_slice_paths(cfg, n_focus=10, round_idx=2)

    # disabled when absent; loud on partial config
    assert round_runner._anchor_slice_paths(None, n_focus=10, round_idx=1) == []
    with pytest.raises(ValueError):
        round_runner._anchor_slice_paths({"scene_list": str(normal_json)}, 10, 1)
    with pytest.raises(ValueError):
        round_runner._anchor_slice_paths(
            {"scene_list": str(normal_json), "ratio": 1.0, "waits_fraction": 0.2}, 10, 1
        )


def test_anchor_slice_paths_takeoff_stratification(tmp_path):
    normal = [f"/tmp/normal_{i}.npz" for i in range(100)]
    waits = [f"/tmp/waits_{i}.npz" for i in range(50)]
    takeoff = [f"/tmp/takeoff_{i}.npz" for i in range(50)]
    normal_json = tmp_path / "normal.json"
    waits_json = tmp_path / "waits.json"
    takeoff_json = tmp_path / "takeoff.json"
    normal_json.write_text(json.dumps(normal))
    waits_json.write_text(json.dumps(waits))
    takeoff_json.write_text(json.dumps(takeoff))

    cfg = {
        "scene_list": str(normal_json),
        "ratio": 2.0,
        "waits_scene_list": str(waits_json),
        "waits_fraction": 0.3,
        "takeoff_scene_list": str(takeoff_json),
        "takeoff_fraction": 0.2,
        "seed": 3,
    }
    picked = round_runner._anchor_slice_paths(cfg, n_focus=10, round_idx=1)
    assert len(picked) == 20
    assert sum(1 for p in picked if "waits_" in p) == 6  # round(0.3 * 20)
    assert sum(1 for p in picked if "takeoff_" in p) == 4  # round(0.2 * 20)
    assert len(set(picked)) == len(picked)

    # loud on partial config and on fractions that leave no normal slice
    with pytest.raises(ValueError):
        round_runner._anchor_slice_paths(
            {"scene_list": str(normal_json), "ratio": 1.0, "takeoff_fraction": 0.2}, 10, 1
        )
    with pytest.raises(ValueError):
        round_runner._anchor_slice_paths(
            {
                "scene_list": str(normal_json),
                "ratio": 1.0,
                "waits_scene_list": str(waits_json),
                "waits_fraction": 0.6,
                "takeoff_scene_list": str(takeoff_json),
                "takeoff_fraction": 0.4,
            },
            10,
            1,
        )


def test_validate_release_bands_config_requires_all_knobs():
    base = {
        "training_backend": "base_sft",
        "perception_mining": {"chunk_manifest": "/tmp/manifest.jsonl"},
    }
    # absent section is fine; empty section is loud
    round_runner._validate_release_bands_config(dict(base))
    with pytest.raises(ValueError):
        round_runner._validate_release_bands_config({**base, "release_bands": {}})
    full = {
        "post_window_s": 10.0,
        "stride_s": 0.5,
        "min_takeoff_travel_m": 10.0,
        "frame_hz": 10.0,
        "ratio": 1.0,
        "seed": 0,
        "workers": 4,
    }
    round_runner._validate_release_bands_config({**base, "release_bands": dict(full)})
    for key in full:
        broken = {k: v for k, v in full.items() if k != key}
        with pytest.raises(ValueError):
            round_runner._validate_release_bands_config({**base, "release_bands": broken})
    # a chunk manifest is mandatory provenance
    with pytest.raises(ValueError):
        round_runner._validate_release_bands_config(
            {"training_backend": "base_sft", "release_bands": dict(full)}
        )


def _release_band_fixture(tmp_path, *, tl_state: str, future_travel_m: float, frame: int):
    """One frame NPZ with a route-lane TL one-hot and a straight-line future."""
    seq = tmp_path / "seq"
    seq.mkdir(exist_ok=True)
    onehot = {"green": 8, "red": 10}[tl_state]
    route = np.zeros((2, 20, 33), dtype=np.float32)
    route[0, :, 0] = 1.0  # mark the lane valid
    route[0, 0, onehot] = 1.0
    step = future_travel_m / 79.0
    future = np.zeros((80, 3), dtype=np.float32)
    future[:, 0] = np.arange(80, dtype=np.float32) * step
    path = seq / f"scene_{frame:016d}.npz"
    np.savez(
        path,
        ego_agent_past=np.zeros((31, 3), dtype=np.float32),
        ego_agent_future=future,
        route_lanes=route,
    )
    return path


def test_build_release_bands_tl_alignment(tmp_path):
    from rlvr.autoresearch.tools.build_release_bands import build_release_bands

    start = _release_band_fixture(tmp_path, tl_state="red", future_travel_m=0.0, frame=100)
    _release_band_fixture(tmp_path, tl_state="green", future_travel_m=2.0, frame=110)  # creep
    kept_go = _release_band_fixture(tmp_path, tl_state="green", future_travel_m=20.0, frame=120)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"chunk_key": "g000_logA_00000000", "start_scene_path": str(start)}) + "\n"
    )
    credit = tmp_path / "credit.jsonl"
    credit.write_text(
        json.dumps(
            {
                "event_key": "g000_logA_00000000_0_16_danger_expert_disagreement",
                "offense_frame_id": 100,
            }
        )
        + "\n"
    )
    out = tmp_path / "bands.json"
    rows = build_release_bands(
        credit,
        manifest,
        out,
        post_window_s=3.0,
        stride_s=1.0,
        min_takeoff_travel_m=10.0,
        frame_hz=10.0,
        workers=1,
    )
    # red frame kept (patience), green take-off kept (release), green creep dropped
    assert str(start) in rows
    assert str(kept_go) in rows
    assert len(rows) == 2
    assert json.loads(out.read_text()) == rows

    # unmatched chunk keys fail loudly
    bad_credit = tmp_path / "bad_credit.jsonl"
    bad_credit.write_text(
        json.dumps({"event_key": "g999_other_00000000_0_1_danger_x", "offense_frame_id": 1}) + "\n"
    )
    with pytest.raises(ValueError):
        build_release_bands(
            bad_credit,
            manifest,
            tmp_path / "x.json",
            post_window_s=1.0,
            stride_s=1.0,
            min_takeoff_travel_m=10.0,
            frame_hz=10.0,
            workers=1,
        )


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
                "replay_memory": {"capacity": 200},
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


def test_round_runner_defaults_enable_expert_disagreement_repair(tmp_path):
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "judgement": {
                    "reward_config": "/tmp/reward.json",
                    "threshold_config": "/tmp/thresholds.json",
                    "credit_window_config": "/tmp/credit.json",
                },
                "repair_generation": {
                    "ego_shape": "from_npz",
                    "min_margin": 0.3,
                },
                "replay_memory": {"capacity": 200},
                "training": {"val_scenes": "/tmp/val.json"},
            }
        )
    )
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"train_args": {"valid_set_list": "/tmp/val.json"}}))
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")

    cfg = round_runner._config_from_workflow_contract(
        {
            "model_path": "/tmp/model.pth",
            "scene_list": str(scene_list),
            "workflow_config": str(workflow),
            "training_config": str(training),
            "output_dir": str(tmp_path / "auto_research" / "run"),
        }
    )

    assert "expert_disagreement" in cfg["mine_labels"]
    assert "expert_disagreement" in cfg["repair_labels"]
    assert cfg["enable_conflict_detector"] is True


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
                "replay_memory": {"capacity": 200},
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
            rb_dists=np.full(8, np.inf, dtype=np.float32),
            red_light=np.zeros(8, dtype=bool),
            accels=np.zeros(8, dtype=np.float32),
            strong_brake_mps2=-2.5,
            dyn=SimpleNamespace(speed=0.0),
            ego_hist=np.zeros((31, 3), dtype=np.float64),
            k=0,
            done=False,
            terminated="max_steps",
            max_steps=999,
            live_pose=np.zeros(3, dtype=np.float32),
            save_buf=None,
            last_snap_step=None,
            snap_count=0,
            turn_hist=np.zeros(31, dtype=np.int64),
            last_turn_indicator=0,
            credit_window=None,
            credit_saved=False,
            verified_credit_labels=set(),
            verified_credit_first_step={},
            danger_event_selector=None,
            output_route_key="bagA",
            realized_lag_streak=0,
            route_arc_s=None,
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

    def _fake_score_object_step_batched(neighbors_list, ego_shapes, device):
        return [(1.0, False, 0, -1) for _ in neighbors_list]

    def _fake_realized_event_scorer(
        np_dict,
        *,
        collided,
        step,
        **_kwargs,
    ):
        return (
            {"labels": ["road_border_crossing"], "label": "road_border_crossing"}
            if np_dict["k"] == 1
            else {"labels": ["clean"], "label": "clean"}
        )

    def _fake_advance_step(s, pred, idx, device, timers, tracked=None):
        s.k += 1

    def _fake_finalize(s, *args, **kwargs):
        return {"n_steps_run": s.k}

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
    monkeypatch.setattr(
        reproducer_rollout, "score_object_step_batched", _fake_score_object_step_batched
    )
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
        # window-saving tests fake _advance_step / seg states: pin the serial
        # tracker so the mpc_batched default's solve precompute stays out.
        tracker_mode="mpc",
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
                "turn_indicator_logit": torch.zeros((batch, 5), dtype=torch.float32),
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
            rb_dists=np.full(max_steps + 1, np.inf, dtype=np.float32),
            red_light=np.zeros(max_steps + 1, dtype=bool),
            accels=np.zeros(max_steps + 1, dtype=np.float32),
            strong_brake_mps2=-2.5,
            dyn=SimpleNamespace(speed=0.0),
            ego_hist=np.zeros((31, 3), dtype=np.float64),
            near_miss_thresh=near_miss_thresh,
            ego_shape=np.ones(3, dtype=np.float32),
            live_pose=np.zeros(3, dtype=np.float32),
            save_buf=None,
            last_snap_step=None,
            snap_count=0,
            turn_hist=np.zeros(31, dtype=np.int64),
            last_turn_indicator=0,
            credit_window=None,
            credit_saved=False,
            verified_credit_labels=set(),
            verified_credit_first_step={},
            danger_event_selector=None,
            output_route_key="bagA",
            realized_lag_streak=0,
            route_arc_s=None,
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

    def _fake_score_object_step_batched(neighbors_list, ego_shapes, device):
        score_calls["n"] += 1
        collided = score_calls["n"] >= 2
        return [(0.0, collided, 1, 0) for _ in neighbors_list]

    def _fake_danger_scorer(built, preds, data, device):
        return [{"labels": ["clean"], "label": "clean"} for _ in built]

    def _fake_realized_event_scorer(
        np_dict,
        *,
        collided,
        step,
        **_kwargs,
    ):
        return (
            {"labels": ["moving_collision"], "label": "moving_collision"}
            if np_dict["k"] == 1
            else {"labels": ["clean"], "label": "clean"}
        )

    def _fake_advance_step(s, pred, idx, device, timers, tracked=None):
        s.k += 1

    def _fake_finalize(s, *args, **kwargs):
        return {"n_steps_run": s.k}

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
    monkeypatch.setattr(
        reproducer_rollout, "score_object_step_batched", _fake_score_object_step_batched
    )
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
        # window-saving tests fake _advance_step / seg states: pin the serial
        # tracker so the mpc_batched default's solve precompute stays out.
        tracker_mode="mpc",
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
                "turn_indicator_logit": torch.zeros((batch, 5), dtype=torch.float32),
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
            rb_dists=np.full(max_steps + 1, np.inf, dtype=np.float32),
            red_light=np.zeros(max_steps + 1, dtype=bool),
            accels=np.zeros(max_steps + 1, dtype=np.float32),
            strong_brake_mps2=-2.5,
            dyn=SimpleNamespace(speed=0.0),
            ego_hist=np.zeros((31, 3), dtype=np.float64),
            near_miss_thresh=near_miss_thresh,
            ego_shape=np.ones(3, dtype=np.float32),
            live_pose=np.zeros(3, dtype=np.float32),
            save_buf=None,
            last_snap_step=None,
            snap_count=0,
            turn_hist=np.zeros(31, dtype=np.int64),
            last_turn_indicator=0,
            credit_window=None,
            credit_saved=False,
            verified_credit_labels=set(),
            verified_credit_first_step={},
            danger_event_selector=None,
            output_route_key="bagA",
            realized_lag_streak=0,
            route_arc_s=None,
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

    def _fake_score_object_step_batched(neighbors_list, ego_shapes, device):
        return [(0.0, True, 1, 0) for _ in neighbors_list]

    def _fake_clean_scorer(*_args, **_kwargs):
        return [{"labels": ["clean"], "label": "clean"}]

    def _fake_clean_realized(*_args, **_kwargs):
        return {"labels": ["clean"], "label": "clean"}

    def _fake_advance_step(s, pred, idx, device, timers, tracked=None):
        s.k += 1

    def _fake_finalize(s, *args, **kwargs):
        return {"n_steps_run": s.k}

    def _fake_dump_credit_window(*args, **kwargs):
        calls.append({"out_dir": args[0], "label": args[10]})
        return {"n_scenes": len(args[4])}

    monkeypatch.setattr(reproducer_rollout, "_seed_state", _fake_seed_state)
    monkeypatch.setattr(reproducer_rollout, "_pre_step", _fake_pre_step)
    monkeypatch.setattr(reproducer_rollout, "_to_torch_batch", _fake_to_torch_batch)
    monkeypatch.setattr(
        reproducer_rollout, "score_object_step_batched", _fake_score_object_step_batched
    )
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
        # window-saving tests fake _advance_step / seg states: pin the serial
        # tracker so the mpc_batched default's solve precompute stays out.
        tracker_mode="mpc",
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

        def _scorer(
            _np_dict,
            *,
            collided=False,
            step=None,
            **_kwargs,
        ):
            return {
                "labels": ["moving_collision"],
                "label": "moving_collision",
                "realized_collision": bool(collided),
            }

        return _scorer

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
            centerline=0.0,
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
            centerline=0.0,
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
        target_gt_disagreement_thresh=2.0,
    )

    assert idx == 1
    assert meta["selected_total"] == 5.0


def test_repair_candidate_selector_does_not_hard_gate_expert_disagreement():
    # Paper-faithful R2 retrieval: expert / GT likeness is NEVER a hard gate. A
    # candidate still flagged expert_disagreement must remain SELECTABLE (recoverability
    # depends only on rule-feasibility q_i>0); expert agreement enters only via the soft,
    # class-weighted r2lpl_score. Here both candidates are rule-feasible; with no
    # candidate/reference trajs the soft expert term ties (exp(0)=1 each), so the higher
    # rule-reward candidate wins even though it is the conflict-flagged one.
    source_row = {"repair_labels": ["expert_disagreement"]}
    candidate_rows = [
        {
            "moving_collision_step": None,
            "expert_disagreement": True,
            "labels": ["expert_disagreement"],
        },
        {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
    ]
    reward_rows = [
        SimpleNamespace(
            centerline=0.0,
            total=10.0,
            collision_step=None,
            rb_crossing=False,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=1.0,
            rb_min_dist=1.0,
        ),
        SimpleNamespace(
            centerline=0.0,
            total=1.0,
            collision_step=None,
            rb_crossing=False,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=1.0,
            rb_min_dist=1.0,
        ),
    ]

    idx, meta = _best_safe_candidate(
        source_row,
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
    )

    # Conflict-flagged candidate is NOT rejected; soft ranking keeps the higher-reward one.
    assert idx == 0
    assert meta["selected_labels"] == ["expert_disagreement"]


def test_expert_disagreement_selection_uses_r2lpl_state_class_weights():
    candidate_rows = [
        {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
        {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
    ]
    reward_rows = [
        SimpleNamespace(
            centerline=0.0,
            total=0.0,
            collision_step=None,
            rb_crossing=False,
            lane_crossing=False,
            static_crossing=False,
            kinematic_violated=False,
            sc_min_dist=1.0,
            rb_min_dist=1.0,
        ),
        SimpleNamespace(
            centerline=0.0,
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
    reference = torch.zeros((4, 4), dtype=torch.float32)
    close_to_expert = torch.zeros((4, 4), dtype=torch.float32)
    high_reward_far = torch.zeros((4, 4), dtype=torch.float32)
    high_reward_far[:, 0] = 10.0
    candidates = [close_to_expert, high_reward_far]

    near_idx, near_meta = _best_safe_candidate(
        {"repair_labels": ["expert_disagreement"], "expert_disagreement_max_dev": 0.5},
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=candidates,
        reference_traj=reference,
    )
    recoverable_idx, recoverable_meta = _best_safe_candidate(
        {"repair_labels": ["expert_disagreement"], "expert_disagreement_max_dev": 2.0},
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=candidates,
        reference_traj=reference,
    )

    assert near_idx == 0
    assert near_meta["selected_r2lpl_state_class"] == "near_log"
    assert recoverable_idx == 1
    assert recoverable_meta["selected_r2lpl_state_class"] == "recoverable"


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


def _clean_reward_row(total: float, centerline: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        total=total,
        centerline=centerline,
        collision_step=None,
        rb_crossing=False,
        lane_crossing=False,
        static_crossing=False,
        kinematic_violated=False,
        sc_min_dist=1.0,
        rb_min_dist=1.0,
    )


_SELECTOR_CANDIDATE_ROWS = [
    {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
    {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]},
]
_SELECTOR_CANDIDATE_TRAJS = [
    torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.1, 0.0, 1.0, 0.0]], dtype=torch.float32),
    torch.tensor([[5.0, 0.0, 1.0, 0.0], [5.1, 0.0, 1.0, 0.0]], dtype=torch.float32),
]
_SELECTOR_REFERENCE = torch.tensor(
    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32
)


def test_repair_candidate_selector_unified_recoverable_prefers_rule_score():
    # Paper Eq. 26, unified across mistake types: a collision/road-border row is a
    # "recoverable" state (0.65 rule / 0.30 ref / 0.05 expert), so the higher-reward
    # candidate wins even though the lower-reward one sits nearer the expert.
    source_row = {"repair_labels": ["road_border_crossing"]}
    reward_rows = [_clean_reward_row(total=1.0), _clean_reward_row(total=10.0)]

    idx, meta = _best_safe_candidate(
        source_row,
        _SELECTOR_CANDIDATE_ROWS,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=_SELECTOR_CANDIDATE_TRAJS,
        reference_traj=_SELECTOR_REFERENCE,
    )

    assert idx == 1
    assert meta["selected_r2lpl_state_class"] == "recoverable"
    assert meta["selected_r2lpl_score"] > 0.0
    # R2LPL hard-example signal is emitted alongside the deviation penalty and is
    # distinct from realized expert_disagreement.
    assert meta["target_gt_disagreement_mean_l2"] == meta["selected_deviation_penalty"]
    assert meta["target_gt_disagreement_max_l2"] >= meta["target_gt_disagreement_mean_l2"]
    assert isinstance(meta["target_gt_disagreement"], bool)


def test_repair_candidate_selector_unified_near_log_prefers_expert():
    # Near-log conflict states weight expert consistency 0.80 and gate out
    # candidates far below the most expert-consistent one, so the expert-nearest
    # candidate wins despite a lower reward total.
    source_row = {
        "repair_labels": ["expert_disagreement"],
        "expert_disagreement_max_dev": 0.5,
    }
    reward_rows = [_clean_reward_row(total=1.0), _clean_reward_row(total=10.0)]

    idx, meta = _best_safe_candidate(
        source_row,
        _SELECTOR_CANDIDATE_ROWS,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=_SELECTOR_CANDIDATE_TRAJS,
        reference_traj=_SELECTOR_REFERENCE,
    )

    assert idx == 0
    assert meta["selected_r2lpl_state_class"] == "near_log"
    assert meta["selected_deviation_penalty"] < 1.0


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
        build_repaired_targets_tool,
        "load_npz_data",
        lambda path, _device: {"scene_path": str(path)},
    )

    def _fake_t0_collision(data, **_kwargs):
        checked_paths.append(data["scene_path"])
        return data["scene_path"].endswith("credit-00029.npz"), -0.1

    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_source_scene_t0_any_neighbor_overlap",
        _fake_t0_collision,
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
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
    assert {row["reason"] for row in dirty} == {"event_window_t0_collision_or_near"}
    assert checked_paths == ["/tmp/event_a/credit-00030.npz", "/tmp/event_a/credit-00029.npz"]


def test_t0_dirty_static_collision_uses_static_cross_threshold(monkeypatch):
    rows = [
        {
            "scene_path": "/tmp/event_a/credit-00030.npz",
            "window_dir": "/tmp/event_a",
            "labels": ["static_collision"],
            "repair_labels": ["static_collision"],
        }
    ]
    seen_thresholds = []

    monkeypatch.setattr(
        build_repaired_targets_tool,
        "load_npz_data",
        lambda path, _device: {"scene_path": str(path)},
    )

    def _fake_t0_collision(_data, **kwargs):
        seen_thresholds.append(kwargs["collision_thresh"])
        return False, 99.0

    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_source_scene_t0_any_neighbor_overlap",
        _fake_t0_collision,
    )

    kept, dirty = _drop_t0_dirty_event_windows(
        rows,
        rcfg=RewardConfig(),
        thresholds={"moving_collision_thresh": 0.2, "sc_cross_thresh": 0.7, "rb_cross_thresh": 0.2},
        device=torch.device("cpu"),
    )

    assert kept == rows
    assert dirty == []
    assert seen_thresholds == [0.7]


def test_static_collision_repair_requires_static_enabled_config():
    rows = [
        {
            "scene_path": "/tmp/event_a/credit-00030.npz",
            "labels": ["static_collision"],
            "repair_labels": ["static_collision"],
        }
    ]

    with pytest.raises(ValueError, match="static_collision_enabled=true"):
        _validate_static_collision_config(rows, RewardConfig(static_collision_enabled=False))

    _validate_static_collision_config(rows, RewardConfig(static_collision_enabled=True))


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


def test_realized_event_scorer_supports_expert_disagreement(tmp_path):
    reward_cfg, threshold_cfg = _write_realized_scorer_configs(tmp_path)
    scorer = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"expert_disagreement"},
    )
    np_dict = {
        "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
        "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
    }
    # Straight route along +x used as the arc-length reference.
    ref_polyline_world = np.stack([np.linspace(0.0, 100.0, 101), np.zeros(101)], axis=1).astype(
        np.float32
    )
    # Expert waits: current pose (index 0) + a still future all at the origin.
    expert_future_world = np.zeros((81, 2), dtype=np.float32)
    expert_future_speed = np.zeros((81,), dtype=np.float32)
    # Model drives forward from ~0 to ~10 m of arc length.
    model_pred_world = np.stack([np.linspace(0.1, 10.0, 80), np.zeros(80)], axis=1).astype(
        np.float32
    )

    row = scorer(
        np_dict,
        collided=False,
        step=4,
        model_pred_world=model_pred_world,
        expert_future_world=expert_future_world,
        expert_future_speed=expert_future_speed,
        ref_polyline_world=ref_polyline_world,
    )

    assert "expert_disagreement" in row["labels"]
    assert row["expert_disagreement_step"] == 4
    assert row["expert_disagreement_reason"] == "expert_wait_model_forward"
    # Progress is relativized to a COMMON origin (expert index 0). The expert
    # sits at the route origin (s0=0), so the model keeps its ~10 m end arc.
    assert row["expert_disagreement_model_end_progress"] == pytest.approx(10.0, abs=0.1)
    assert row["expert_disagreement_expert_end_progress"] == pytest.approx(0.0, abs=1e-4)


def test_realized_event_scorer_realized_lag_flags_on_sustained_streak(tmp_path):
    # Closed-loop fail-to-resume: the model's open-loop proposal keeps pace with the
    # expert schedule (no projected-branch flag — the dithering blindspot), but the
    # rollout loop reports a sustained realized gap -> the realized branch must flag
    # model_lagging_expert so the depart repair can fire.
    reward_cfg, threshold_cfg = _write_realized_scorer_configs(tmp_path)
    scorer = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"expert_disagreement"},
    )
    np_dict = {
        "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
        "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
    }
    ref_polyline_world = np.stack([np.linspace(0.0, 100.0, 101), np.zeros(101)], axis=1).astype(
        np.float32
    )
    # Expert drives 5 -> 15 m; the proposal matches its schedule exactly, so none of
    # the frozen 3 branches fires on its own.
    expert_future_world = np.stack([np.linspace(5.0, 15.0, 81), np.zeros(81)], axis=1).astype(
        np.float32
    )
    expert_future_speed = np.full((81,), 1.25, dtype=np.float32)
    model_pred_world = expert_future_world[1:].copy()

    common = dict(
        collided=False,
        model_pred_world=model_pred_world,
        expert_future_world=expert_future_world,
        expert_future_speed=expert_future_speed,
        ref_polyline_world=ref_polyline_world,
    )
    below = scorer(
        np_dict,
        step=7,
        realized_lag_streak=REALIZED_LAG_SUSTAIN_STEPS - 1,
        realized_lag_gap_m=4.2,
        **common,
    )
    assert "expert_disagreement" not in below["labels"]
    assert below["expert_disagreement_realized_lag"] is False
    assert below["expert_disagreement_realized_gap_m"] == pytest.approx(4.2)

    flagged = scorer(
        np_dict,
        step=8,
        realized_lag_streak=REALIZED_LAG_SUSTAIN_STEPS,
        realized_lag_gap_m=4.5,
        **common,
    )
    assert "expert_disagreement" in flagged["labels"]
    assert flagged["expert_disagreement_reason"] == "model_lagging_expert"
    assert flagged["expert_disagreement_realized_lag"] is True
    assert flagged["expert_disagreement_step"] == 8


def test_realized_event_scorer_exposes_expert_lag_thresholds(tmp_path):
    # The rollout loop maintains the streak using the SAME thresholds the scorer was
    # built with — exposed as an attribute so there is a single source of truth.
    reward_cfg, threshold_cfg = _write_realized_scorer_configs(tmp_path)
    scorer = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"expert_disagreement"},
    )
    thr = scorer.expert_lag_thresholds
    assert thr is not None
    assert set(thr) == {"lag_progress_gap_m", "moving_speed_mps"}
    assert thr["lag_progress_gap_m"] > 0
    assert thr["moving_speed_mps"] > 0

    no_ed = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"moving_collision"},
    )
    assert no_ed.expert_lag_thresholds is None


def test_realized_event_scorer_expert_disagreement_requires_inputs(tmp_path):
    # No silent fallback: the branch is allowed but the open-loop / route inputs
    # are missing -> fail loudly instead of silently skipping.
    reward_cfg, threshold_cfg = _write_realized_scorer_configs(tmp_path)
    scorer = build_realized_event_scorer(
        reward_config=reward_cfg,
        threshold_config=threshold_cfg,
        device="cpu",
        allowed_labels={"expert_disagreement"},
    )
    np_dict = {
        "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
        "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="mining path must pass them"):
        scorer(np_dict, collided=False, step=0)


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
                "expert_disagreement_wait_speed_mps": 0.5,
                "expert_disagreement_wait_progress_m": 1.0,
                "expert_disagreement_forward_progress_gap_m": 2.0,
                "expert_disagreement_lag_progress_gap_m": 3.0,
                "expert_disagreement_moving_speed_mps": 1.0,
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
    # Collision is checked at the neighbor's CURRENT pose (last past frame),
    # overlapping the ego at x=4.52. The future must show real motion so the
    # neighbor is classified as MOVING (not stopped) — a stationary neighbor
    # here is a static_collision, not a moving one.
    np_dict["neighbor_agents_past"][0, -1, 0] = 4.52
    np_dict["neighbor_agents_past"][0, -1, 2] = 1.0
    np_dict["neighbor_agents_past"][0, -1, 6] = 2.0
    np_dict["neighbor_agents_past"][0, -1, 7] = 4.5
    np_dict["neighbor_agents_future"][0, :, 0] = np.linspace(4.52, 6.52, 80)
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
                "expert_disagreement_wait_speed_mps": 0.5,
                "expert_disagreement_wait_progress_m": 1.0,
                "expert_disagreement_forward_progress_gap_m": 2.0,
                "expert_disagreement_lag_progress_gap_m": 3.0,
                "expert_disagreement_moving_speed_mps": 1.0,
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


def test_build_repaired_targets_accepts_scene_local_ego_shape():
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
        ego_expert_future=np.full((80, 4), -5.0, dtype=np.float32),
        origin=np.asarray("sim"),
    )

    data = {
        "ego_shape": torch.tensor(sim_ego_shape[None], dtype=torch.float32),
        "ego_agent_future": torch.zeros((80, 4), dtype=torch.float32),
        "ego_expert_future": torch.zeros((80, 4), dtype=torch.float32),
    }
    selected = torch.zeros((80, 4), dtype=torch.float32)
    selected[:, 0] = torch.arange(80, dtype=torch.float32)
    selected[:, 2] = 1.0

    monkeypatch.setattr(
        build_repaired_targets_tool,
        "load_reward_config",
        lambda _path: RewardConfig(ignore_rear_end_collisions=False),
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_load_scene_thresholds",
        lambda _path: {
            "moving_collision_thresh": 0.2,
            "moving_near_thresh": 0.7,
            "static_near_thresh": 0.5,
            "rb_near_thresh": 0.2,
            "expert_disagreement_wait_speed_mps": 0.5,
            "expert_disagreement_wait_progress_m": 1.0,
            "expert_disagreement_forward_progress_gap_m": 2.0,
            "expert_disagreement_lag_progress_gap_m": 3.0,
            "expert_disagreement_moving_speed_mps": 1.0,
        },
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "load_model",
        lambda _model_path, _device: (object(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "load_npz_data",
        lambda _path, _device: data,
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_source_scene_t0_any_neighbor_overlap",
        lambda *_args, **_kwargs: (False, 99.0),
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_stack_scene_data",
        lambda _datas, _device: {},
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_normalize_batch",
        lambda _batch, _model_args: {},
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "generate_all_scenes_batched",
        lambda *_args, **_kwargs: [torch.stack([selected])],
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "classify_loaded_scene_candidates_batch",
        # return_subscores=True contract: (rows_per_scene, B-major subscores)
        lambda *_args, **_kwargs: ([[{"labels": ["clean"]}]], {}),
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "compute_reward_batch",
        lambda *_args, **_kwargs: [_reward_row(1.0)],
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
        "_shape_reward",
        lambda *_args, **_kwargs: [_reward_row(1.0)],
    )
    monkeypatch.setattr(
        build_repaired_targets_tool,
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
    paths, unrepaired = build_repaired_targets_tool.build_repaired_targets(
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
                "ema_decay": 0.996,
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
    # ema_decay must survive the train_args passthrough — 0.999 is too slow
    # to absorb behavior changes within a short per-round fine-tune.
    assert cmd[cmd.index("--ema_decay") + 1] == "0.996"


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


# ---------------------------------------------------------------------------
# intra-training re-repair (repair_refresh) inner loop
# ---------------------------------------------------------------------------


def _repair_refresh_base_cfg():
    return {
        "epochs_per_round": 4,
        "training_backend": "base_sft",
    }


def test_repair_refresh_rejects_every_greater_than_epochs_per_round():
    cfg = _repair_refresh_base_cfg()
    cfg["repair_refresh_every_epochs"] = 5
    with pytest.raises(ValueError, match="epochs_per_round"):
        _apply_repair_refresh_config(cfg)


def test_repair_refresh_rejects_non_base_sft_backend():
    cfg = _repair_refresh_base_cfg()
    cfg["repair_refresh_every_epochs"] = 2
    cfg["training_backend"] = "rsft"
    with pytest.raises(ValueError, match="base_sft"):
        _apply_repair_refresh_config(cfg)


def test_repair_refresh_rejects_unknown_scope():
    cfg = _repair_refresh_base_cfg()
    cfg["repair_refresh_scope"] = "bogus"
    with pytest.raises(ValueError, match="repair_refresh_scope"):
        _apply_repair_refresh_config(cfg)


def test_repair_refresh_defaults_when_absent():
    cfg = _apply_repair_refresh_config(_repair_refresh_base_cfg())
    assert cfg["repair_refresh_every_epochs"] == 0
    assert cfg["repair_refresh_scope"] == "unrepaired"
    assert cfg["repair_refresh_max_cycles"] == 0


def test_repair_refresh_chunk_sizes_and_targets_split_evenly():
    # epochs_per_round=4, refresh_every=2 => two chunks of 2, one refresh between.
    assert _repair_refresh_chunk_sizes(4, 2, 0) == [2, 2]
    # base epoch 80 -> 82 -> 84 (absolute train_epochs targets).
    assert _repair_refresh_chunk_targets(80, 4, 2, 0) == [82, 84]


def test_repair_refresh_chunk_sizes_handles_remainder_and_off_state():
    # remainder: 5 epochs in chunks of 2 => 2, 2, 1.
    assert _repair_refresh_chunk_sizes(5, 2, 0) == [2, 2, 1]
    # feature OFF (every==0) or every>=epochs_per_round => single full chunk.
    assert _repair_refresh_chunk_sizes(4, 0, 0) == [4]
    assert _repair_refresh_chunk_sizes(4, 4, 0) == [4]


def test_repair_refresh_chunk_sizes_respects_max_cycles():
    # max_cycles=1: one refresh, then finish the rest in a single last chunk.
    assert _repair_refresh_chunk_sizes(6, 2, 1) == [2, 4]
    assert _repair_refresh_chunk_targets(80, 6, 2, 1) == [82, 86]


def test_refresh_repair_strips_repair_added_keys_from_unrepaired_rows(monkeypatch, tmp_path):
    rdir = tmp_path / "round"
    rdir.mkdir(parents=True, exist_ok=True)
    # Credit rows define the legal key set for build_repaired_targets.
    credit_rows = [
        {"scene_path": "/data/a.npz", "label": "moving_collision", "event_key": "ev0"},
        {"scene_path": "/data/b.npz", "label": "static_collision", "event_key": "ev1"},
    ]
    (rdir / "credit_windows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in credit_rows)
    )
    # Unrepaired rows = credit-row keys PLUS repair-added keys that must be stripped.
    unrepaired = [
        {
            "scene_path": "/data/a.npz",
            "label": "moving_collision",
            "event_key": "ev0",
            "reason": "no_safe_candidate",
            "meta": {"min_dist": 0.1},
            "t0_min_dist": 0.05,
        }
    ]
    (rdir / "repaired_targets_unrepaired.json").write_text(json.dumps(unrepaired))
    (rdir / "repaired_targets.json").write_text(json.dumps([]))
    (rdir / "repaired_targets.jsonl").write_text("")

    cfg = {
        "reward_config": str(tmp_path / "reward.json"),
        "threshold_config": str(tmp_path / "thresholds.json"),
        "repair_labels": ["moving_collision", "static_collision"],
        "repair_config": {"ego_shape": "4.76,7.24,2.29", "min_margin": 0.3, "K": 8},
    }

    def _fake_run(cmd, log_path, *, cwd=None, env=None):
        # build_repaired_targets writes: out_list, out_rows_jsonl, and an
        # <out_list>_unrepaired.json. Simulate repairing the single scene.
        out_list = Path(cmd[cmd.index("--out_list") + 1])
        out_rows = Path(cmd[cmd.index("--out_rows_jsonl") + 1])
        out_list.parent.mkdir(parents=True, exist_ok=True)
        out_list.write_text(json.dumps(["/data/repaired_a.npz"]))
        out_rows.write_text(
            json.dumps({"scene_path": "/data/repaired_a.npz", "source_scene_path": "/data/a.npz"})
            + "\n"
        )
        return 1.0

    monkeypatch.setattr(round_runner, "_run", _fake_run)

    newly, _refresh_elapsed = _refresh_repair(
        cfg, tmp_path / "model.pth", rdir, scope="unrepaired", cycle=0, gpu_ids=[]
    )

    # The scene-rows jsonl fed to build_repaired_targets must contain ONLY the
    # credit-row keys (reason/meta/t0_min_dist stripped).
    written = _read_test_jsonl(rdir / "refresh_0" / "credit_windows.jsonl")
    assert written == [
        {"scene_path": "/data/a.npz", "label": "moving_collision", "event_key": "ev0"}
    ]
    assert set(written[0].keys()) == {"scene_path", "label", "event_key"}

    # Harvested repairs fold into the canonical artifacts.
    assert newly == ["/data/repaired_a.npz"]
    assert json.loads((rdir / "repaired_targets.json").read_text()) == ["/data/repaired_a.npz"]
    # Cleared everything => unrepaired list rewritten to empty.
    assert json.loads((rdir / "repaired_targets_unrepaired.json").read_text()) == []


def test_refresh_repair_returns_empty_when_no_unrepaired_rows(monkeypatch, tmp_path):
    rdir = tmp_path / "round"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "credit_windows.jsonl").write_text(
        json.dumps({"scene_path": "/data/a.npz", "label": "moving_collision"}) + "\n"
    )
    # No unrepaired file at all => nothing to re-attempt, repair never runs.
    cfg = {
        "reward_config": str(tmp_path / "reward.json"),
        "threshold_config": str(tmp_path / "thresholds.json"),
        "repair_config": {"ego_shape": "4.76,7.24,2.29", "min_margin": 0.3},
    }

    def _boom(*_args, **_kwargs):
        raise AssertionError("_run should not be called when there is nothing to re-attempt")

    monkeypatch.setattr(round_runner, "_run", _boom)
    # Returns the (paths, elapsed) tuple the caller unpacks — a bare [] here
    # crashed the round loop the first time a refresh found nothing to do.
    assert _refresh_repair(
        cfg, tmp_path / "m.pth", rdir, scope="unrepaired", cycle=0, gpu_ids=[]
    ) == ([], 0.0)


# ---------------------------------------------------------------------------
# det-path re-timing morph candidate (expert_morph.build_expert_morph_candidate)
# ---------------------------------------------------------------------------

_MORPH_T = 80
_MORPH_DT = 0.1


def _straight_traj(xs: np.ndarray) -> np.ndarray:
    xy = np.stack([xs, np.zeros(len(xs))], axis=1)
    return np.concatenate([xy, np.ones((len(xs), 1)), np.zeros((len(xs), 1))], axis=1)


def test_expert_morph_takeoff_advances_toward_expert():
    # det ~stationary, expert ramps forward -> morph must advance to ~expert terminal.
    det = _straight_traj(np.zeros(_MORPH_T))
    expert = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))  # ~4 m/s
    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    assert morph.shape == (_MORPH_T, 4)
    # Monotone non-decreasing longitudinal progress (no backward step).
    assert np.all(np.diff(morph[:, 0]) >= -1e-4)
    # Reaches most of the expert's terminal progress by the horizon end.
    assert morph[-1, 0] > 0.8 * expert[-1, 0]
    # t=0 continuity in xy.
    assert np.allclose(morph[0, :2], det[0, :2], atol=0.2)


def test_expert_morph_stop_holds_position():
    # det ramps forward, expert flat (stationary) -> morph is pulled back toward
    # the stationary expert (terminal progress well below det's) and stays monotone.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))  # ~4 m/s
    expert = _straight_traj(np.zeros(_MORPH_T))
    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    assert np.all(np.diff(morph[:, 0]) >= -1e-4)
    # Pulled back well short of the det terminal progress ("holds position").
    assert morph[-1, 0] < 0.5 * det[-1, 0]
    # t=0 continuity in xy AND initial speed (v0 == det initial speed).
    assert np.allclose(morph[0, :2], det[0, :2], atol=0.2)
    v0_det = np.linalg.norm(det[1, :2] - det[0, :2]) / _MORPH_DT
    v0_morph = np.linalg.norm(morph[1, :2] - morph[0, :2]) / _MORPH_DT
    assert abs(v0_det - v0_morph) < 0.5


def test_expert_morph_rejects_impossible_deceleration():
    # High initial speed, short horizon, expert demands a full stop -> the accel /
    # jerk clamps cannot follow, so the candidate is rejected (None).
    Ts = 20
    det = _straight_traj(np.cumsum(np.full(Ts, 2.0)))  # 20 m/s
    expert = _straight_traj(np.zeros(Ts))
    assert build_expert_morph_candidate(det, expert) is None


def test_expert_morph_rejects_diverged_paths():
    # Expert far off the det path laterally -> no meaningful merge corridor.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))
    expert = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))
    expert[:, 1] = 12.0
    morph, diag = build_expert_morph_candidate(det, expert, return_diag=True)
    assert morph is None
    assert diag["stage"] == "paths_diverged"


def test_expert_morph_never_reverses():
    # A wiggly blend must still yield a monotone longitudinal signal.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.3)))
    expert = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.5)))
    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    assert np.all(np.diff(morph[:, 0]) >= -1e-4)


def test_expert_morph_no_lateral_motion_at_standstill():
    # det moves forward on a laterally-offset line, expert is stopped on the
    # centerline: after the morph brakes to a stop, the position must FREEZE —
    # lateral blending must not keep sliding the point sideways (the
    # "curls onto itself" artifact: crab-walk at standstill).
    det_xy_x = np.cumsum(np.full(_MORPH_T, 0.4))  # ~4 m/s
    det = _straight_traj(det_xy_x)
    det[:, 1] = 1.5  # det path laterally offset from the expert's stop point
    expert = _straight_traj(np.zeros(_MORPH_T))  # stopped at origin (d=0)
    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    step = np.linalg.norm(np.diff(morph[:, :2], axis=0), axis=-1)
    stopped = step / _MORPH_DT < 0.1
    assert stopped.any()
    t_stop = int(np.argmax(stopped))
    # once stopped, every subsequent step must stay stopped (no curl / drift)
    assert np.all(step[t_stop:] / _MORPH_DT < 0.1)
    tail_drift = np.linalg.norm(morph[-1, :2] - morph[t_stop, :2])
    assert tail_drift < 0.15


def test_expert_morph_passes_reward_kinematic_gate():
    # The acceptance criterion that used to reject 32/52 real expert-waits
    # morphs: the reward's own kinematic gate (SG-smoothed yaw-rate vs absolute
    # cap AND bicycle-model curvature bound). Motion-derived heading must pass
    # it without scenario tuning — including the hard case: braking to a stop
    # while the route curves.
    import torch as _torch

    from planner_metrics.config import RewardConfig
    from planner_metrics.subscores import compute_kinematic_gate

    # Curved route: straight 20 m then a gentle 90-degree bend.
    a = np.stack([np.linspace(0.0, 20.0, 80), np.zeros(80)], axis=1)
    ang = np.linspace(-np.pi / 2, 0.0, 120)
    b = np.stack([20.0 + 15.0 * np.cos(ang), 15.0 + 15.0 * np.sin(ang)], axis=1)
    route = np.concatenate([a, b[1:]], axis=0)

    # det: constant 4 m/s along the route; expert: stops early (at ~6 m).
    s_det = np.cumsum(np.full(_MORPH_T, 0.4))
    det = _traj_along_polyline(route, s_det)
    s_exp = np.minimum(np.cumsum(np.full(_MORPH_T, 0.2)), 6.0)
    expert = _traj_along_polyline(route, s_exp)

    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    cfg = RewardConfig()
    gate = compute_kinematic_gate(
        _torch.from_numpy(morph)[None],
        cfg,
        _torch.tensor([3.0, 4.5, 1.9]),
    )
    assert float(gate[0]) == 1.0


def _traj_along_polyline(route: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    seg = np.diff(route, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s_ref = np.concatenate([[0.0], np.cumsum(seg_len)])
    xs = np.interp(s_query, s_ref, route[:, 0])
    ys = np.interp(s_query, s_ref, route[:, 1])
    th = np.arctan2(
        np.gradient(ys),
        np.maximum(np.abs(np.gradient(xs)), 1e-9) * np.sign(np.gradient(xs) + 1e-12),
    )
    return np.stack([xs, ys, np.cos(th), np.sin(th)], axis=1).astype(np.float32)


def test_expert_morph_stop_anchor_overrides_stop_location():
    # recorded-anchor semantics: the caller-supplied stop point wins over the
    # expert's traveled distance. Expert travels to 20 m; anchor at 12 m must
    # pull the stop to ~12.5 (anchor + overshoot margin).
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))  # 4 m/s
    s_exp = np.minimum(np.cumsum(np.full(_MORPH_T, 0.4)), 20.0)
    expert = _straight_traj(s_exp)
    morph = build_expert_morph_candidate(det, expert, stop_anchor_xy=np.array([12.0, 0.0]))
    assert morph is not None
    assert abs(morph[-1, 0] - 12.5) < 1.0
    # without the anchor the stop lands at the expert's terminal (~20.5 bound)
    default = build_expert_morph_candidate(det, expert)
    assert default is not None
    assert default[-1, 0] > 15.0


def test_expert_morph_stop_anchor_floored_by_feasible_stop():
    # An anchor the ego cannot reach (1 m ahead at 5 m/s) must be floored at
    # the tracker's shortest feasible stop, not demanded.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.5)))  # 5 m/s
    expert = _straight_traj(np.zeros(_MORPH_T))
    morph = build_expert_morph_candidate(det, expert, stop_anchor_xy=np.array([1.0, 0.0]))
    assert morph is not None
    step = np.linalg.norm(np.diff(morph[:, :2], axis=0), axis=-1)
    assert (step / _MORPH_DT < 0.5).any()  # it does stop
    assert morph[-1, 0] > 1.5  # but beyond the infeasible 1 m anchor


def test_expert_morph_lateral_slope_cap_honors_w_max():
    # The analytic lateral magnitude limit must include w_max: at w_max=2 the
    # quintic's peak slope doubles, and without the factor the crab-angle
    # bound is violated.
    from rlvr.autoresearch.tools.expert_morph import _DEFAULT_LATERAL_SLOPE_CAP

    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))
    det[:, 1] = 0.0
    expert = _straight_traj(np.minimum(np.cumsum(np.full(_MORPH_T, 0.4)), 4.0))
    expert[:, 1] = 4.0  # large lateral gap, short stop arc
    morph = build_expert_morph_candidate(det, expert, w_max=2.0)
    if morph is None:
        return  # rejected outright is acceptable; the cap claim is about accepted output
    dd = np.abs(np.diff(morph[:, 1]))
    ds = (
        np.linalg.norm(np.diff(morph[:, :2], axis=1), axis=-1)
        if False
        else np.abs(np.diff(morph[:, 0]))
    )
    moving = ds > 1e-3
    assert (dd[moving] / ds[moving]).max() <= _DEFAULT_LATERAL_SLOPE_CAP * 1.2 + 1e-6


def test_preflight_recorded_anchor_raises_on_missing_field(tmp_path):
    from rlvr.autoresearch.tools.build_repaired_targets import _preflight_recorded_anchor

    T = 80
    base = dict(
        ego_agent_future=np.zeros((T, 3), dtype=np.float32),
        ego_expert_future=np.zeros((T, 4), dtype=np.float32),
    )
    old = tmp_path / "old.npz"
    np.savez(old, **base)
    new = tmp_path / "new.npz"
    np.savez(new, **base, ego_recorded_future=np.zeros((T, 4), dtype=np.float32))

    rows_old = [{"scene_path": str(old), "repair_labels": ["expert_disagreement"]}]
    rows_new = [{"scene_path": str(new), "repair_labels": ["expert_disagreement"]}]
    rows_nonexpert = [{"scene_path": str(old), "repair_labels": ["road_border_crossing"]}]

    with pytest.raises(ValueError, match="ego_recorded_future"):
        _preflight_recorded_anchor(rows_old)
    _preflight_recorded_anchor(rows_new)  # passes
    _preflight_recorded_anchor(rows_nonexpert)  # non-expert rows are exempt


def test_repair_cmd_forwards_expert_stop_anchor(tmp_path):
    cfg = {
        "reward_config": "r.json",
        "threshold_config": "t.json",
        "repair_config": {
            "ego_shape": "4.76,7.24,2.29",
            "min_margin": 0.3,
            "expert_stop_anchor": "pseudo",
        },
    }
    cmd = round_runner._repair_cmd(cfg, tmp_path / "m.pth", tmp_path / "c.jsonl", tmp_path)
    idx = cmd.index("--expert_stop_anchor")
    assert cmd[idx + 1] == "pseudo"
    # without the key the flag is omitted (CLI default 'recorded' applies)
    del cfg["repair_config"]["expert_stop_anchor"]
    cmd2 = round_runner._repair_cmd(cfg, tmp_path / "m.pth", tmp_path / "c.jsonl", tmp_path)
    assert "--expert_stop_anchor" not in cmd2


def test_expert_morph_return_diag_reports_stage():
    # ok: diag carries the clamp statistics.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))
    expert = _straight_traj(np.zeros(_MORPH_T))
    morph, diag = build_expert_morph_candidate(det, expert, return_diag=True)
    assert morph is not None
    assert diag["stage"] == "ok"
    assert "terminal_err_m" in diag and "clamp_fire_fraction" in diag and "v0_mps" in diag

    # infeasible deceleration: rejection reason + statistics.
    Ts = 20
    det_fast = _straight_traj(np.cumsum(np.full(Ts, 2.0)))  # 20 m/s
    expert_stop = _straight_traj(np.zeros(Ts))
    morph, diag = build_expert_morph_candidate(det_fast, expert_stop, return_diag=True)
    assert morph is None
    assert diag["stage"] == "infeasible_deceleration"
    assert diag["terminal_err_m"] > 0.0

    # too short horizon.
    morph, diag = build_expert_morph_candidate(det[:1], expert[:1], return_diag=True)
    assert morph is None
    assert diag["stage"] == "too_short_horizon"

    # default (no return_diag) keeps the plain return type.
    assert build_expert_morph_candidate(det, expert) is not None


def _morph_outcome_reward_row(total: float, *, rb_crossing: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        centerline=0.0,
        total=total,
        collision_step=None,
        rb_crossing=rb_crossing,
        lane_crossing=False,
        static_crossing=False,
        kinematic_violated=False,
        sc_min_dist=1.0,
        rb_min_dist=1.0,
    )


def _morph_outcome_candidate_row(labels: list[str]) -> dict:
    return {"moving_collision_step": None, "expert_disagreement": False, "labels": labels}


def test_replay_memory_quota_does_not_double_dip_rare_buckets():
    # A quota label whose reserved rows already cover its bucket must NOT get a
    # second row from the rare-bucket pass (that would squeeze the majority
    # label beyond the quota's promise).
    rows = [
        {
            "scene_path": f"/tmp/rb_{i}.npz",
            "label": "road_border_crossing",
            "route_arc_m": 0.0,
            "variant_kind": "credit",
            "difficulty": 100.0 + i,
        }
        for i in range(6)
    ]
    rows += [
        {
            "scene_path": f"/tmp/expert_{i}.npz",
            "label": "expert_disagreement",
            "route_arc_m": 0.0,
            "variant_kind": "credit",
            "difficulty": float(i),
        }
        for i in range(3)
    ]
    memory = build_memory(
        rows,
        {"entries": []},
        {},
        capacity=4,
        alpha=1.0,
        beta=0.0,
        arc_bin_m=1000.0,
        label_quotas={"expert_disagreement": 0.5},
    )
    n_expert = sum(1 for r in memory["entries"] if r["label"] == "expert_disagreement")
    n_rb = sum(1 for r in memory["entries"] if r["label"] == "road_border_crossing")
    assert n_expert == 2  # exactly the reserve, not reserve + bucket pick
    assert n_rb == 2


def test_replay_memory_label_quotas_rejects_empty_label():
    from rlvr.autoresearch.tools.lifelong_replay_memory import _parse_label_quotas

    with pytest.raises(ValueError):
        _parse_label_quotas("=0.5")


def test_kinematic_gate_standstill_noise_passes_turn_in_place_fails():
    # Pins the low-speed conditioning of compute_kinematic_gate: sub-noise yaw
    # at standstill must PASS (the fixed false positive), while genuine
    # turning-in-place above the clamped bound must still FAIL.
    from planner_metrics.config import RewardConfig
    from planner_metrics.subscores import compute_kinematic_gate

    cfg = RewardConfig()
    ego_shape = torch.tensor([3.0, 4.5, 1.9])
    T = 80
    # Standstill with tiny heading noise (0.002 rad/s scale) on a fixed point.
    h = np.cumsum(np.full(T, 0.0002))
    still = np.stack([np.zeros(T), np.zeros(T), np.cos(h), np.sin(h)], axis=1).astype(np.float32)
    gate = compute_kinematic_gate(torch.from_numpy(still)[None], cfg, ego_shape)
    assert float(gate[0]) == 1.0
    # Standstill turning in place well above kappa_max * 0.1 (but below the
    # absolute max_yaw_rate cap) must still be caught by the curvature check.
    kappa_max = cfg.kinematic_margin * np.tan(cfg.max_steer) / 3.0
    yaw_rate = min(3.0 * kappa_max * 0.1, 0.9 * cfg.max_yaw_rate)
    h2 = np.cumsum(np.full(T, yaw_rate * 0.1))
    spin = np.stack([np.zeros(T), np.zeros(T), np.cos(h2), np.sin(h2)], axis=1).astype(np.float32)
    gate2 = compute_kinematic_gate(torch.from_numpy(spin)[None], cfg, ego_shape)
    assert float(gate2[0]) == 0.0


def test_expert_morph_stops_at_expert_location():
    # The stop-arc cap must stop the morph AT the expert's stop location
    # (projected onto the det path), not merely after the expert's traveled
    # distance. Expert stops at x=20 m, well beyond the braking floor from
    # 4 m/s, so the projection-based cap is the binding constraint.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.4)))  # 4 m/s, 32 m
    s_exp = np.minimum(np.cumsum(np.full(_MORPH_T, 0.4)), 20.0)
    expert = _straight_traj(s_exp)
    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    assert abs(morph[-1, 0] - 20.5) < 1.0  # expert stop arc + overshoot margin


def test_expert_morph_preserves_slow_creep():
    # Sub-centimetre-step snapping must only affect the standstill TAIL —
    # a slow queue creep (~0.09 m/s) must not be converted into a full park.
    det = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.009)))
    expert = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.009)))
    morph = build_expert_morph_candidate(det, expert)
    assert morph is not None
    assert morph[-1, 0] > 0.5  # creep distance preserved (~0.7 m), not 0


def test_depart_morph_synthesizes_departure_from_parked_plan():
    # The stop morph cannot do this (det path has no road ahead) — the depart
    # morph sources geometry from the expert path bridged to the ego pose.
    det = _straight_traj(np.linspace(0.0, 2.0, _MORPH_T))  # parked/creeping plan
    expert = _straight_traj(6.0 + np.cumsum(np.full(_MORPH_T, 0.45)))  # ahead, ~4.5 m/s
    morph, diag = build_depart_morph_candidate(det, expert, return_diag=True)
    assert diag["stage"] == "ok"
    assert morph.shape == (_MORPH_T, 4)
    assert abs(morph[0, 0]) < 0.1  # starts at the ego, not at the expert (no teleport)
    assert np.all(np.diff(morph[:, 0]) >= -1e-4)
    assert morph[-1, 0] > 10.0  # genuinely departs
    steps = np.linalg.norm(np.diff(morph[:, :2], axis=0), axis=1)
    assert steps.max() < 3.0  # accel/jerk-feasible, no jumps


def test_depart_morph_bails_on_stationary_or_behind_expert():
    det = _straight_traj(np.zeros(_MORPH_T))
    stationary = _straight_traj(np.full(_MORPH_T, 6.0))
    _, diag = build_depart_morph_candidate(det, stationary, return_diag=True)
    assert diag["stage"] == "expert_stationary"
    behind = _straight_traj(-8.0 + np.cumsum(np.full(_MORPH_T, 0.3)))
    _, diag = build_depart_morph_candidate(det, behind, return_diag=True)
    assert diag["stage"] == "expert_behind_ego"


def test_depart_morph_rejects_undrivable_geometry():
    det = _straight_traj(np.zeros(_MORPH_T))
    # Near-pure-lateral expert offset: no forward-drivable bridge exists, and
    # a raw chord would emit sideways (crab-motion) headings as supervision.
    lateral = _straight_traj(np.cumsum(np.full(_MORPH_T, 0.5)) - 0.5)
    lateral[:, 1] = 0.4
    _, diag = build_depart_morph_candidate(det, lateral, return_diag=True)
    assert diag["stage"] == "not_forward_from_ego"
    # Reversing expert: arc-length heading would flip ~180 deg at the retrace.
    reversing = _straight_traj(10.0 - np.cumsum(np.full(_MORPH_T, 0.1)))
    _, diag = build_depart_morph_candidate(det, reversing, return_diag=True)
    assert diag["stage"] == "path_reversal"
    # A small forward gap must still synthesize with forward headings (the
    # bridge, not a raw chord, applies at ANY positive gap).
    near = _straight_traj(0.3 + np.cumsum(np.full(_MORPH_T, 0.5)))
    morph, diag = build_depart_morph_candidate(det, near, return_diag=True)
    assert diag["stage"] == "ok"
    headings = np.degrees(np.arctan2(morph[:, 3], morph[:, 2]))
    assert np.abs(headings).max() < 10.0


def test_depart_morph_preserves_recorded_wait_at_origin():
    # An expert waiting at the ego-frame origin is written as [0, 0, cos, sin]
    # (valid unit heading, reproducer_rollout convention) — NOT padding.
    # Treating those rows as leading padding would back-fill them from the
    # first moved pose and synthesize motion before the recorded departure.
    det = _straight_traj(np.zeros(_MORPH_T))
    xs = np.concatenate([np.zeros(10), np.cumsum(np.full(_MORPH_T - 10, 0.5))])
    expert = _straight_traj(xs)  # waits 1 s at origin, then departs at ~5 m/s
    morph, diag = build_depart_morph_candidate(det, expert, return_diag=True)
    assert diag["stage"] == "ok"
    assert np.abs(morph[:10, 0]).max() < 0.05  # initial wait preserved
    assert morph[-1, 0] > 10.0  # then genuinely departs


def test_depart_morph_rejects_padded_or_nonfinite_expert():
    det = _straight_traj(np.zeros(_MORPH_T))
    # A zero-padded LEADING sample must not read as gap=0 and bypass the
    # gap/behind gates — the first valid sample defines the expert start.
    far = _straight_traj(40.0 + np.cumsum(np.full(_MORPH_T, 0.5)))
    far[0] = 0.0
    _, diag = build_depart_morph_candidate(det, far, return_diag=True)
    assert diag["stage"] == "gap_too_large"
    behind = _straight_traj(-8.0 + np.cumsum(np.full(_MORPH_T, 0.5)))
    behind[0] = 0.0
    _, diag = build_depart_morph_candidate(det, behind, return_diag=True)
    assert diag["stage"] == "expert_behind_ego"
    # NaN passes every threshold gate (comparisons are False) so it must be
    # rejected up front, not propagated into the SFT target.
    nan_expert = _straight_traj(6.0 + np.cumsum(np.full(_MORPH_T, 0.5)))
    nan_expert[30, 0] = np.nan
    _, diag = build_depart_morph_candidate(det, nan_expert, return_diag=True)
    assert diag["stage"] == "non_finite_input"


def _patience_npz(path, *, red_route: bool, v_profile: np.ndarray) -> None:
    T = len(v_profile) + 1
    x = np.concatenate([[0.0], np.cumsum(v_profile * 0.1)])
    fut = np.stack([x, np.zeros(T), np.zeros(T)], axis=1).astype(np.float32)
    route = np.zeros((2, 20, 33), dtype=np.float32)
    route[..., 0] = 1.0  # non-zero xy so lanes read as valid
    if red_route:
        route[0, :, 10] = 1.0
    np.savez(
        path,
        route_lanes=route,
        neighbor_agents_past=np.zeros((4, 31, 11), dtype=np.float32),
        ego_agent_future=fut,
    )


def test_build_patience_benchmark_synthetic(tmp_path):
    from rlvr.autoresearch.tools.build_patience_benchmark import build_benchmark

    T = 79
    # Onset offset: moving at t0, stops mid-horizon (red light on route).
    onset = np.concatenate([np.full(30, 5.0), np.zeros(T - 30)])
    _patience_npz(tmp_path / "bag_0000000100.npz", red_route=True, v_profile=onset)
    # Same event 2 s later: already stopped (same bag, same frame bin).
    _patience_npz(tmp_path / "bag_0000000120.npz", red_route=True, v_profile=np.zeros(T))
    # No wait: cruises through (never below the wait speed).
    _patience_npz(tmp_path / "bag_0000005000.npz", red_route=True, v_profile=np.full(T, 5.0))

    scenes = [
        str(tmp_path / n)
        for n in ("bag_0000000100.npz", "bag_0000000120.npz", "bag_0000005000.npz")
    ]
    benchmark, pool, stats = build_benchmark(
        scenes,
        wait_speed_mps=0.5,
        min_wait_steps=10,
        ped_near_m=5.0,
        event_frame_gap=100,
        max_scenes=10,
    )
    assert stats["pool"] == 2  # the two waiting offsets
    assert stats["events"] == 1  # deduped to one event
    assert benchmark == [str(tmp_path / "bag_0000000100.npz")]  # onset preferred
    assert stats["onset_in_benchmark"] == 1


def test_eval_patience_onset_stop_metrics():
    from rlvr.autoresearch.tools.eval_patience_onset import _stop_metrics

    T = 80
    stop_at_3s = np.concatenate([np.full(30, 0.5), np.zeros(T - 1 - 30)])
    xy = np.stack([np.concatenate([[0.0], np.cumsum(stop_at_3s)]), np.zeros(T)], axis=1)
    m = _stop_metrics(xy, stop_speed=0.5)
    assert m["stops"] and abs(m["stop_onset_s"] - 3.0) < 0.11
    never = np.stack([np.arange(T, dtype=float), np.zeros(T)], axis=1)
    m2 = _stop_metrics(never, stop_speed=0.5)
    assert not m2["stops"] and m2["stop_onset_s"] is None


def test_best_safe_candidate_reports_morph_outcome():
    source_row = {"repair_labels": ["expert_disagreement"], "expert_disagreement_max_dev": 0.5}
    T = 20
    expert = np.zeros((T, 4), dtype=np.float32)
    mover = np.zeros((T, 4), dtype=np.float32)
    mover[:, 0] = np.arange(T, dtype=np.float32)  # far from the waiting expert
    morph = np.zeros((T, 4), dtype=np.float32)  # matches the waiting expert

    # morph accepted + wins (near_log -> expert-closeness dominates).
    idx, meta = _best_safe_candidate(
        source_row,
        [_morph_outcome_candidate_row(["clean"]), _morph_outcome_candidate_row(["clean"])],
        [_morph_outcome_reward_row(5.0), _morph_outcome_reward_row(0.0)],
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[mover, morph],
        reference_traj=expert,
        morph_index=1,
    )
    assert idx == 1
    assert meta["morph_outcome"] == "selected"

    # morph accepted but loses -> lost_selection + its r2lpl score recorded.
    far_off = {"repair_labels": ["expert_disagreement"], "expert_disagreement_max_dev": 9.0}
    idx, meta = _best_safe_candidate(
        far_off,  # far_offpolicy -> rule-score weight 1.0, expert weight 0.0
        [_morph_outcome_candidate_row(["clean"]), _morph_outcome_candidate_row(["clean"])],
        [_morph_outcome_reward_row(5.0), _morph_outcome_reward_row(0.0)],
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[mover, morph],
        reference_traj=expert,
        morph_index=1,
    )
    assert idx == 0
    assert meta["morph_outcome"] == "lost_selection"
    assert "morph_r2lpl_score" in meta

    # morph gate-rejected (rb crossing) while a model candidate is safe.
    idx, meta = _best_safe_candidate(
        source_row,
        [
            _morph_outcome_candidate_row(["clean"]),
            _morph_outcome_candidate_row(["road_border_crossing"]),
        ],
        [_morph_outcome_reward_row(5.0), _morph_outcome_reward_row(0.0, rb_crossing=True)],
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[mover, morph],
        reference_traj=expert,
        morph_index=1,
    )
    assert idx == 0
    assert meta["morph_outcome"] == "gate_rejected"
    assert meta["morph_labels"] == ["road_border_crossing"]

    # every candidate gate-rejected -> no_safe_candidate still reports the morph gate.
    idx, meta = _best_safe_candidate(
        source_row,
        [
            _morph_outcome_candidate_row(["road_border_crossing"]),
            _morph_outcome_candidate_row(["road_border_crossing"]),
        ],
        [
            _morph_outcome_reward_row(-5.0, rb_crossing=True),
            _morph_outcome_reward_row(-9.0, rb_crossing=True),
        ],
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[mover, morph],
        reference_traj=expert,
        morph_index=1,
    )
    assert idx is None
    assert meta["reason"] == "no_safe_candidate"
    assert meta["morph_outcome"] == "gate_rejected"


def test_best_safe_candidate_reports_depart_outcome():
    # The depart candidate must get the same outcome taxonomy as the stop
    # morph (selected / lost_selection / gate_rejected), independently keyed.
    source_row = {"repair_labels": ["expert_disagreement"], "state_class": "stopped"}
    expert = torch.zeros(4, 4)
    mover = torch.ones(4, 4)
    depart = torch.full((4, 4), 2.0)

    idx, meta = _best_safe_candidate(
        source_row,
        [
            _morph_outcome_candidate_row(["clean"]),
            _morph_outcome_candidate_row(["clean"]),
            _morph_outcome_candidate_row(["road_border_crossing"]),
        ],
        [
            _morph_outcome_reward_row(5.0),
            _morph_outcome_reward_row(3.0),
            _morph_outcome_reward_row(0.0, rb_crossing=True),
        ],
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[mover, depart, expert],
        reference_traj=expert,
        morph_index=2,
        depart_index=1,
    )
    assert idx is not None
    outcome = meta["depart_outcome"]
    assert outcome == ("selected" if idx == 1 else "lost_selection")
    if outcome == "lost_selection":
        assert "depart_r2lpl_score" in meta
    # Both scripted candidates reported independently.
    assert meta["morph_outcome"] == "gate_rejected"
    assert meta["morph_labels"] == ["road_border_crossing"]


# ---------------------------------------------------------------------------
# Expert-reference scoring fix
# ---------------------------------------------------------------------------


def test_expert_reference_scoring_data_swaps_progress_gt():
    realized = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    expert = torch.zeros(1, 3, 4)
    data = {
        "ego_agent_future": realized,
        "ego_expert_future": expert,
        "route_lanes": torch.ones(1, 2, 5, 4),
    }
    scoring = _expert_reference_scoring_data(data, scene_path="/data/x.npz")
    # Reward progress GT (ego_agent_future) is now the logged expert.
    assert torch.equal(scoring["ego_agent_future"], expert)
    # Original dict is NOT mutated (scoring-only).
    assert torch.equal(data["ego_agent_future"], realized)
    # Other keys are shared through (route unaffected).
    assert torch.equal(scoring["route_lanes"], data["route_lanes"])


def test_expert_reference_scoring_data_fails_loud_when_missing():
    data = {"ego_agent_future": torch.zeros(1, 3, 4)}
    with pytest.raises(ValueError, match="ego_expert_future"):
        _expert_reference_scoring_data(data, scene_path="/data/x.npz")


def test_expert_reference_only_applies_to_expert_disagreement():
    assert _row_is_expert_disagreement({"repair_labels": ["expert_disagreement"]}) is True
    assert _row_is_expert_disagreement({"repair_labels": ["moving_collision"]}) is False
    assert _row_is_expert_disagreement({"repair_labels": ["road_border_crossing"]}) is False


def test_repair_cmd_defaults_expert_knobs_on_and_honors_overrides(tmp_path):
    base_cfg = {
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "repair_config": {"ego_shape": "from_npz", "min_margin": 0.3},
    }
    # Defaults on: no disabling flags emitted.
    cmd = _repair_cmd(
        base_cfg,
        tmp_path / "model.pth",
        tmp_path / "round" / "credit.jsonl",
        tmp_path / "round",
    )
    assert "--no_repair_expert_gt_candidate" not in cmd
    assert "--no_repair_expert_reference" not in cmd

    # Explicit off + morph overrides threaded through.
    cfg2 = {
        **base_cfg,
        "repair_config": {
            "ego_shape": "from_npz",
            "min_margin": 0.3,
            "repair_expert_gt_candidate": False,
            "repair_expert_reference": False,
            "expert_morph_w_max": 0.8,
            "expert_morph_max_accel": 1.5,
            "expert_morph_max_jerk": 3.0,
        },
    }
    cmd2 = _repair_cmd(
        cfg2,
        tmp_path / "model.pth",
        tmp_path / "round" / "credit.jsonl",
        tmp_path / "round",
    )
    assert "--no_repair_expert_gt_candidate" in cmd2
    assert "--no_repair_expert_reference" in cmd2
    assert cmd2[cmd2.index("--expert_morph_w_max") + 1] == "0.8"
    assert cmd2[cmd2.index("--expert_morph_max_accel") + 1] == "1.5"
    assert cmd2[cmd2.index("--expert_morph_max_jerk") + 1] == "3.0"


# ---------------------------------------------------------------------------
# Post-round guard evaluation (report-only; includes removed-gate regression pins)
# ---------------------------------------------------------------------------


def _guard_assets(tmp_path):
    manifest = tmp_path / "frozen_chunks.jsonl"
    manifest.write_text('{"scene_path": "/tmp/a.npz"}\n')
    benchmark = tmp_path / "patience_benchmark.json"
    benchmark.write_text(json.dumps(["/tmp/waits_0.npz"]))
    return manifest, benchmark


def test_validate_guards_config_rejects_empty_and_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="empty 'guards'"):
        round_runner._validate_guards_config({"guards": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        round_runner._validate_guards_config({"guards": {"frozen_manifest": "x"}})
    # absent guards is fine with the default policy
    round_runner._validate_guards_config({})


def test_validate_guards_config_rejects_composite_policy(tmp_path):
    manifest, benchmark = _guard_assets(tmp_path)
    # the removed automatic gate must fail loudly, with or without guards
    with pytest.raises(ValueError, match="report-only"):
        round_runner._validate_guards_config({"checkpoint_policy": "composite"})
    with pytest.raises(ValueError, match="report-only"):
        round_runner._validate_guards_config(
            {
                "checkpoint_policy": "composite",
                "guards": {
                    "frozen_chunk_manifest": str(manifest),
                    "patience_benchmark": str(benchmark),
                },
            }
        )
    # missing frozen assets fail loudly
    with pytest.raises(ValueError, match="does not exist"):
        round_runner._validate_guards_config(
            {"guards": {"frozen_chunk_manifest": str(tmp_path / "missing.jsonl")}}
        )
    # empty benchmark / frozen manifest rejected
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(ValueError, match="is empty"):
        round_runner._validate_guards_config({"guards": {"patience_benchmark": str(empty)}})
    empty_manifest = tmp_path / "empty.jsonl"
    empty_manifest.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        round_runner._validate_guards_config(
            {"guards": {"frozen_chunk_manifest": str(empty_manifest)}}
        )
    # fully-specified report-only config passes
    round_runner._validate_guards_config(
        {
            "checkpoint_policy": "latest",
            "guards": {
                "frozen_chunk_manifest": str(manifest),
                "patience_benchmark": str(benchmark),
            },
        }
    )


def test_validate_guards_config_rejects_unknown_sections(tmp_path):
    manifest, benchmark = _guard_assets(tmp_path)
    # the removed composite tolerance section is now an unknown key
    with pytest.raises(ValueError, match="unknown keys"):
        round_runner._validate_guards_config(
            {
                "guards": {
                    "frozen_chunk_manifest": str(manifest),
                    "patience_benchmark": str(benchmark),
                    "composite": {"event_tolerance_frac": 0.0},
                }
            }
        )
    # closed_loop knobs without the npz root are a misconfiguration
    with pytest.raises(ValueError, match="closed_loop_npz_root"):
        round_runner._validate_guards_config(
            {"guards": {"patience_benchmark": str(benchmark), "closed_loop": {"seg_len": 600}}}
        )


def test_guard_mining_cmd_uses_frozen_manifest_and_neutralizes_subsampling(tmp_path):
    manifest, benchmark = _guard_assets(tmp_path)
    cfg = {
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "mine_labels": ["expert_disagreement"],
        "guards": {
            "frozen_chunk_manifest": str(manifest),
            "patience_benchmark": str(benchmark),
        },
        "perception_mining": {
            "scene_list": "/tmp/campaign_scenes.json",
            "chunk_len": 80,
            "sample_fraction": 0.25,
            "sample_seed": 3,
            "num_shards": 4,
            "shard_index": 1,
            "max_chunks": 100,
            "allow_existing_out_dir": True,
        },
    }
    gdir = tmp_path / "round" / "guards"
    cmd = round_runner._guard_mining_cmd(cfg, tmp_path / "model.pth", gdir, gpu_id=0)
    assert cmd[cmd.index("--chunk_manifest") + 1] == str(manifest)
    # the frozen set is mined in full every round: no campaign subsampling,
    # and stale guard windows must fail loudly even when the campaign allows
    # reusing its own out dirs
    for neutralized in ("--sample_fraction", "--sample_seed", "--num_shards", "--max_chunks"):
        assert neutralized not in cmd
    assert "--scene_list" not in cmd
    assert "--allow_existing_out_dir" not in cmd
    assert cmd[cmd.index("--out_jsonl") + 1] == str(gdir / "credit_windows.jsonl")

    onset_cmd = round_runner._patience_onset_cmd(cfg, tmp_path / "model.pth", gdir)
    assert onset_cmd[onset_cmd.index("--scenes") + 1] == str(benchmark)
    assert onset_cmd[onset_cmd.index("--out") + 1] == str(gdir / "patience_onset.json")
    assert onset_cmd[onset_cmd.index("--stop_speed") + 1] == "0.5"


def test_workflow_contract_parses_guards(tmp_path):
    manifest, benchmark = _guard_assets(tmp_path)
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"backend": "base_sft", "train_args": {}}))

    def _workflow(guards):
        payload = {
            "judgement": {
                "reward_config": "/tmp/reward.json",
                "threshold_config": "/tmp/thresholds.json",
                "credit_window_config": "/tmp/credit.json",
                "enabled_labels": ["moving_collision"],
            },
            "repair_generation": {"ego_shape": "from_npz", "min_margin": 0.3},
            "replay_memory": {"capacity": 200},
            "training": {"val_scenes": "/tmp/valid.json"},
        }
        if guards is not None:
            payload["guards"] = guards
        path = tmp_path / "workflow.json"
        path.write_text(json.dumps(payload))
        return path

    guards = {
        "frozen_chunk_manifest": str(manifest),
        "patience_benchmark": str(benchmark),
        "_comment": "stripped at parse time",
    }
    contract = {
        "model_path": "/tmp/model.pth",
        "scene_list": str(scene_list),
        "workflow_config": str(_workflow(guards)),
        "training_config": str(training),
        "output_dir": str(tmp_path / "auto_research" / "out"),
    }
    cfg = round_runner._config_from_workflow_contract(contract)
    assert cfg["guards"] == {
        "frozen_chunk_manifest": str(manifest),
        "patience_benchmark": str(benchmark),
    }
    round_runner._validate_guards_config(cfg)

    # present-but-empty guards section is rejected at contract parse time
    contract["workflow_config"] = str(_workflow({}))
    with pytest.raises(ValueError, match="guards is present but empty"):
        round_runner._config_from_workflow_contract(contract)


def test_guard_mining_metrics_and_summary_derivation(tmp_path):
    gdir = tmp_path / "guards"
    gdir.mkdir()
    rows = [
        {"scene_path": "/tmp/a.npz", "label": "expert_disagreement", "event_key": "e1"},
        {"scene_path": "/tmp/b.npz", "label": "expert_disagreement", "event_key": "e1"},
        {"scene_path": "/tmp/c.npz", "label": "expert_disagreement", "event_key": "e2"},
        {"scene_path": "/tmp/d.npz", "label": "road_border_crossing", "event_key": "e3"},
    ]
    with open(gdir / "credit_windows.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (gdir / "summary.json").write_text(json.dumps({"simulated_chunks": 500}))

    metrics = round_runner._guard_mining_metrics(gdir)
    assert metrics["event_count_by_label"] == {
        "expert_disagreement": 2,
        "road_border_crossing": 1,
    }
    assert metrics["total_events"] == 3
    assert metrics["events_per_1000_chunks"] == 6.0

    # zero simulated chunks means the frozen set is broken, not clean
    (gdir / "summary.json").write_text(json.dumps({"simulated_chunks": 0}))
    with pytest.raises(RuntimeError, match="simulated 0 chunks"):
        round_runner._guard_mining_metrics(gdir)


def test_shipped_template_guards_pass_validation(tmp_path):
    """Round-trip the actual shipped template's guards/rounds sections.

    Regression pin: the template carries "_"-prefixed JSON-comment keys inside
    guards, which the whitelist validator must tolerate (and the contract
    parser must strip)."""
    template_path = (
        Path(round_runner.__file__).resolve().parents[2]
        / "configs"
        / "r2lpl_workflow_template.json"
    )
    template = json.loads(template_path.read_text())
    manifest, benchmark = _guard_assets(tmp_path)
    guards = dict(template["guards"])
    guards["frozen_chunk_manifest"] = str(manifest)
    guards["patience_benchmark"] = str(benchmark)
    cfg = {
        "checkpoint_policy": template["rounds"]["checkpoint_selection_rule"],
        "guards": guards,
    }
    round_runner._validate_guards_config(cfg)
    assert template["rounds"]["checkpoint_selection_rule"] == "latest"
    # the contract parser strips comment keys from the runtime config
    stripped = round_runner._strip_comment_keys(guards)
    assert not [k for k in stripped if k.startswith("_")]


def test_main_runs_report_only_guards_per_round(tmp_path, monkeypatch):
    """End-to-end main() wiring: guards run on the starting model (reference
    row) and on every round checkpoint, metrics land in the round summaries,
    and checkpoints always advance (no gating, no rollback)."""
    manifest, benchmark = _guard_assets(tmp_path)
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text(json.dumps(["/tmp/a.npz"]))
    out_dir = tmp_path / "auto_research" / "out"
    initial_model = tmp_path / "initial.pth"
    # A real (if tiny) checkpoint: inference phases resolve EMA weights through
    # _ema_inference_model_path, which torch.loads this file. An empty stub only passed
    # while that resolution step did not exist.
    torch.save({"model": {}}, initial_model)
    cfg = {
        "rounds": 2,
        "epochs_per_round": 1,
        "model_path": str(initial_model),
        "val_scenes": "/tmp/valid.json",
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "replay_memory": {"capacity": 10},
        "training_config": {"train_args": {}},
        "training_backend": "base_sft",
        "output_dir": str(out_dir),
        "checkpoint_policy": "latest",
        "repair_config": {"ego_shape": "2.7,4.3,1.7", "min_margin": 0.3},
        "mine_labels": ["expert_disagreement"],
        "perception_mining": {"scene_list": str(scene_list), "chunk_len": 80},
        "guards": {
            "frozen_chunk_manifest": str(manifest),
            "patience_benchmark": str(benchmark),
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    mining_calls: list[str] = []

    def fake_mining(cfg, model_path, rdir, gpu_ids):
        mining_calls.append(str(model_path))
        round_runner._write_jsonl(
            rdir / "credit_windows.jsonl",
            [{"scene_path": "/tmp/a.npz", "label": "expert_disagreement", "event_key": "e1"}],
        )
        round_runner._write_json(rdir / "perception_direct_summary.json", {"simulated_chunks": 1})
        round_runner._write_json(rdir / "credit_windows_paths.json", ["/tmp/a.npz"])
        return 0.0

    def fake_repair(cfg, model_path, rdir, gpu_ids, **kwargs):
        round_runner._write_json(rdir / "repaired_targets.json", ["/tmp/repaired_a.npz"])
        round_runner._write_jsonl(
            rdir / "repaired_targets.jsonl",
            [{"scene_path": "/tmp/repaired_a.npz", "source_scene_path": "/tmp/a.npz"}],
        )
        return 0.0

    def fake_run(cmd, log_path, *, cwd=None, env=None):
        if "--out_memory" in cmd:
            round_runner._write_json(Path(cmd[cmd.index("--out_memory") + 1]), {"entries": []})
            round_runner._write_json(Path(cmd[cmd.index("--out_replay_scenes") + 1]), [])
        return 0.0

    train_warm_starts: list[str] = []

    def fake_training(cfg, *, model_path, train_input_list, rdir, round_idx, gpu_ids):
        train_warm_starts.append(str(model_path))
        ckpt = rdir / "base_train" / "latest.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        # Loadable stub: the next round's inference phases resolve EMA weights from this
        # checkpoint (see _ema_inference_model_path), so it must be a real torch file.
        torch.save({"model": {}}, ckpt)
        return 0.0, ckpt

    guarded: list[tuple[str, str]] = []

    guard_metric_seq = [
        {
            "frozen_chunk_mining": {
                "event_count_by_label": {"expert_disagreement": 5},
                "total_events": 5,
                "simulated_chunks": 40,
            },
            "patience_onset": {"fail_to_stop": 0, "fail_to_resume": 10},
        },
        {
            "frozen_chunk_mining": {
                "event_count_by_label": {"expert_disagreement": 6},
                "total_events": 6,
                "simulated_chunks": 40,
            },
            "patience_onset": {"fail_to_stop": 0, "fail_to_resume": 10},
        },
        {
            "frozen_chunk_mining": {
                "event_count_by_label": {"expert_disagreement": 3},
                "total_events": 3,
                "simulated_chunks": 40,
            },
            "patience_onset": {"fail_to_stop": 0, "fail_to_resume": 8},
        },
    ]

    def fake_guards(cfg, model_path, gdir, gpu_ids, *, tag):
        guarded.append((tag, str(model_path)))
        return {"tag": tag, "model_path": str(model_path), **guard_metric_seq.pop(0)}

    monkeypatch.setattr(round_runner, "_run_mining_phase", fake_mining)
    monkeypatch.setattr(round_runner, "_run_repair_phase", fake_repair)
    monkeypatch.setattr(round_runner, "_run", fake_run)
    monkeypatch.setattr(round_runner, "_run_base_sft_training", fake_training)
    monkeypatch.setattr(round_runner, "_run_guard_phase", fake_guards)
    monkeypatch.setattr(sys, "argv", ["run_lifelong_r2lpl_rounds", "--config", str(cfg_path)])
    round_runner.main()

    round1_ckpt = str(out_dir / "r2lpl_round_001" / "base_train" / "latest.pth")
    round2_ckpt = str(out_dir / "r2lpl_round_002" / "base_train" / "latest.pth")
    # reference row + one guard pass per round checkpoint
    assert guarded == [
        ("round_000", str(initial_model)),
        ("round_001", round1_ckpt),
        ("round_002", round2_ckpt),
    ]
    # checkpoints always advance; each round mines with the previous checkpoint
    assert train_warm_starts == [str(initial_model), round1_ckpt]
    assert mining_calls == [str(initial_model), round1_ckpt]
    round1 = json.loads((out_dir / "r2lpl_round_001" / "round_summary.json").read_text())
    round2 = json.loads((out_dir / "r2lpl_round_002" / "round_summary.json").read_text())
    assert round1["guards"]["tag"] == "round_001"
    assert round1["next_model_path"] == round1_ckpt
    assert "composite_decision" not in round1
    assert round2["next_model_path"] == round2_ckpt
    # advisory selection report: round 1 regressed (5->6 events) and is
    # vetoed; round 2 improved and is recommended — with NO effect on the
    # training chain (warm starts asserted above are still latest-chained)
    report = json.loads((out_dir / "selection_report.json").read_text())
    assert report["recommended"] == {"round_idx": 2, "checkpoint": round2_ckpt}
    assert not [r for r in report["rounds"] if r["round_idx"] == 1][0]["eligible"]


def test_ensure_4col_neighbor_futures_converts_only_3col(tmp_path):
    """Anchor scenes (3-col logged) must be homogenized to the 4-col schema
    before unioning with repaired scenes, or collate crashes on mixed batches."""
    base = {
        "ego_agent_future": np.zeros((80, 3), dtype=np.float32),
        "turn_indicators": np.zeros(31, dtype=np.int64),
    }
    three = dict(base)
    nf3 = np.zeros((4, 80, 3), dtype=np.float32)
    nf3[0, :, 0] = np.arange(1, 81)  # nonzero xy: (0,0) rows are padding by contract
    nf3[0, :, 2] = np.pi / 2  # heading -> cos 0 / sin 1
    # rows 1..3 stay all-zero = padding, must remain zero after conversion
    three["neighbor_agents_future"] = nf3
    p3 = tmp_path / "logged_scene.npz"
    np.savez(p3, **three)

    four = dict(base)
    four["neighbor_agents_future"] = np.zeros((4, 80, 4), dtype=np.float32)
    p4 = tmp_path / "repaired_scene.npz"
    np.savez(p4, **four)

    out_dir = tmp_path / "converted"
    result = round_runner._ensure_4col_neighbor_futures([str(p3), str(p4)], out_dir)

    # 4-col passes through by original path
    assert result[1] == str(p4)
    # 3-col was rewritten to a copy
    assert result[0] != str(p3)
    with np.load(result[0]) as d:
        nf = d["neighbor_agents_future"]
        assert nf.shape == (4, 80, 4)
        np.testing.assert_allclose(nf[0, :, 2], 0.0, atol=1e-6)  # cos(pi/2)
        np.testing.assert_allclose(nf[0, :, 3], 1.0, atol=1e-6)  # sin(pi/2)
        assert np.abs(nf[1:]).sum() == 0.0  # padding rows preserved as zero
        assert d["ego_agent_future"].shape == (80, 3)  # other fields verbatim
    # torch collate over the homogenized pair must now stack
    batch = [dict(np.load(p)) for p in result]
    stacked = torch.stack([torch.from_numpy(b["neighbor_agents_future"]) for b in batch])
    assert stacked.shape == (2, 4, 80, 4)


def test_ensure_4col_neighbor_futures_caches_across_calls(tmp_path):
    """The 4-col rewrite is called with a CAMPAIGN-level out_dir every round; a
    second call over the same pool must reuse the manifest + converted files
    instead of re-decompressing/re-compressing every scene, and a converted
    file deleted out from under the manifest must be rebuilt (a stale manifest
    entry is never trusted blindly)."""
    three = {
        "ego_agent_future": np.zeros((80, 3), dtype=np.float32),
        "neighbor_agents_future": np.zeros((4, 80, 3), dtype=np.float32),
    }
    three["neighbor_agents_future"][0, :, 0] = 1.0
    p3 = tmp_path / "logged_scene.npz"
    np.savez(p3, **three)
    out_dir = tmp_path / "converted"

    first = round_runner._ensure_4col_neighbor_futures([str(p3)], out_dir)
    manifest = json.loads((out_dir / "conversion_manifest.json").read_text())
    assert manifest[str(p3)]["marker"] == Path(first[0]).name
    mtime = Path(first[0]).stat().st_mtime_ns

    second = round_runner._ensure_4col_neighbor_futures([str(p3)], out_dir)
    assert second == first
    assert Path(first[0]).stat().st_mtime_ns == mtime  # cached: not rewritten

    Path(first[0]).unlink()  # simulate a lost/partial conversion
    third = round_runner._ensure_4col_neighbor_futures([str(p3)], out_dir)
    assert third == first
    assert Path(first[0]).exists()  # rebuilt, not blindly trusted

    # A source regenerated IN PLACE (same path, new content/mtime) must be
    # reconverted — the cache is path-keyed but stamped with size+mtime.
    three["neighbor_agents_future"][0, :, 1] = 5.0
    np.savez(p3, **three)
    fourth = round_runner._ensure_4col_neighbor_futures([str(p3)], out_dir)
    assert fourth == first
    with np.load(fourth[0]) as d:
        np.testing.assert_allclose(d["neighbor_agents_future"][0, :, 1], 5.0)


def test_materialize_chunk_manifest_campaign_cache(tmp_path, monkeypatch):
    """The plan_only chunk pass is planned once per CAMPAIGN and reused across
    rounds; a knob change against the cached plan fails loudly (chunking must
    stay constant within a campaign for guard comparability)."""
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")
    out_dir = tmp_path / "campaign"
    r1 = out_dir / "r2lpl_round_001"
    r2 = out_dir / "r2lpl_round_002"
    r1.mkdir(parents=True)
    r2.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(cmd, log_path, env=None):
        calls.append([str(c) for c in cmd])
        Path(out_dir / "planned_chunks.jsonl").write_text("")
        return 0.0

    monkeypatch.setattr(round_runner, "_run", fake_run)
    cfg = {"perception_mining": {"scene_list": str(scene_list), "chunk_len": 160}}

    got1 = round_runner._materialize_chunk_manifest_for_shards(cfg, r1)
    assert len(calls) == 1
    assert got1["perception_mining"]["chunk_manifest"] == str(out_dir / "planned_chunks.jsonl")

    got2 = round_runner._materialize_chunk_manifest_for_shards(cfg, r2)
    assert len(calls) == 1  # reused, no second plan pass
    assert got2["perception_mining"]["chunk_manifest"] == str(out_dir / "planned_chunks.jsonl")

    bad_cfg = {"perception_mining": {"scene_list": str(scene_list), "chunk_len": 80}}
    with pytest.raises(ValueError, match="different knobs"):
        round_runner._materialize_chunk_manifest_for_shards(bad_cfg, r2)

    # A scene list regenerated IN PLACE (same path, new content) must fail
    # loudly too — the plan is keyed by content digest, not pathname.
    scene_list.write_text('["/data/new_scene.npz"]')
    with pytest.raises(ValueError, match="different knobs"):
        round_runner._materialize_chunk_manifest_for_shards(cfg, r2)


def test_replay_capacity_is_required():
    with pytest.raises(ValueError, match="capacity is required"):
        round_runner._required_replay_capacity({})
    with pytest.raises(ValueError, match=">= 1"):
        round_runner._required_replay_capacity({"capacity": 0})
    assert round_runner._required_replay_capacity({"capacity": 5000}) == 5000


def test_validate_normal_scene_list_config(tmp_path):
    """normal_scene_list must fail loudly at startup: it is rsft-only (base_sft
    uses training.anchor), must be a non-empty JSON list of paths, and the
    prob/normal split it enables needs explicit, sane n_prob_scenes /
    n_normal_scenes — run_experiment would otherwise only raise (or silently
    subsample) after the expensive mine+repair phases."""
    scene = tmp_path / "normal_0.npz"
    np.savez(scene, neighbor_agents_future=np.zeros((1, 2, 4), dtype=np.float32))
    normals = tmp_path / "normals.json"
    normals.write_text(json.dumps([str(scene)]))
    tcfg = tmp_path / "train_cfg.json"
    tcfg.write_text(json.dumps({"ranked_sft_mode": "curated"}))
    base = {
        "training_normal_scene_list": str(normals),
        "training_backend": "rsft",
        "training_config": str(tcfg),
    }
    # No normal_scene_list -> no-op.
    round_runner._validate_normal_scene_list_config({"training_backend": "base_sft"})
    with pytest.raises(ValueError, match="ranked-SFT backend"):
        round_runner._validate_normal_scene_list_config({**base, "training_backend": "base_sft"})
    with pytest.raises(ValueError, match="does not exist"):
        round_runner._validate_normal_scene_list_config(
            {**base, "training_normal_scene_list": str(tmp_path / "missing.json")}
        )
    # Empty or malformed list content fails at startup, not at collate time.
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(ValueError, match="non-empty JSON list"):
        round_runner._validate_normal_scene_list_config(
            {**base, "training_normal_scene_list": str(empty)}
        )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        round_runner._validate_normal_scene_list_config(
            {**base, "training_normal_scene_list": str(malformed)}
        )
    not_paths = tmp_path / "not_paths.json"
    not_paths.write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="non-empty JSON list"):
        round_runner._validate_normal_scene_list_config(
            {**base, "training_normal_scene_list": str(not_paths)}
        )
    # A listed scene file that does not exist fails at startup, not in the
    # 4-col conversion after the mine+repair phases.
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([str(scene), "/missing/scene.npz"]))
    with pytest.raises(ValueError, match=r"1/2 missing scene files.*missing/scene\.npz"):
        round_runner._validate_normal_scene_list_config(
            {**base, "training_normal_scene_list": str(broken)}
        )
    with pytest.raises(ValueError, match="n_prob_scenes"):
        round_runner._validate_normal_scene_list_config(base)
    tcfg.write_text(
        json.dumps({"ranked_sft_mode": "curated", "n_prob_scenes": 0, "n_normal_scenes": 230})
    )
    with pytest.raises(ValueError, match="positive integer"):
        round_runner._validate_normal_scene_list_config(base)
    tcfg.write_text(
        json.dumps({"ranked_sft_mode": "curated", "n_prob_scenes": 1000, "n_normal_scenes": -1})
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        round_runner._validate_normal_scene_list_config(base)
    tcfg.write_text(
        json.dumps({"ranked_sft_mode": "curated", "n_prob_scenes": 1000, "n_normal_scenes": 230})
    )
    round_runner._validate_normal_scene_list_config(base)  # valid -> no raise


def test_rsft_scene_args_resolves_normals_and_guards_subsampling(tmp_path):
    """The rsft prob/normal path must (a) homogenize 3-col logged normal scenes
    to the 4-col schema before handing them to run_experiment (a mixed batch
    cannot collate — same incompatibility the base_sft anchor slice already
    handles), and (b) refuse an n_prob_scenes below the repaired count, which
    run_experiment's min(n_prob, len(prob)) sampling would silently truncate."""
    base = {
        "ego_agent_future": np.zeros((80, 3), dtype=np.float32),
        "turn_indicators": np.zeros(31, dtype=np.int64),
    }
    three = dict(base)
    nf3 = np.zeros((4, 80, 3), dtype=np.float32)
    nf3[0, :, 0] = np.arange(1, 81)
    nf3[0, :, 2] = np.pi / 2
    three["neighbor_agents_future"] = nf3
    p3 = tmp_path / "logged_normal.npz"
    np.savez(p3, **three)
    four = dict(base)
    four["neighbor_agents_future"] = np.zeros((4, 80, 4), dtype=np.float32)
    p4 = tmp_path / "converted_normal.npz"
    np.savez(p4, **four)
    normals = tmp_path / "normals.json"
    normals.write_text(json.dumps([str(p3), str(p4)]))
    repaired_list = tmp_path / "repaired.json"
    repaired_paths = ["/data/repaired_0.npz", "/data/repaired_1.npz"]
    repaired_list.write_text(json.dumps(repaired_paths))
    tcfg = tmp_path / "train_cfg.json"
    tcfg.write_text(
        json.dumps({"ranked_sft_mode": "curated", "n_prob_scenes": 10, "n_normal_scenes": 2})
    )
    cfg = {
        "training_normal_scene_list": str(normals),
        "training_backend": "rsft",
        "training_config": str(tcfg),
    }
    rdir = tmp_path / "round"
    rdir.mkdir()

    args = round_runner._rsft_scene_args(
        cfg,
        repaired_paths=repaired_paths,
        repaired_list_json=repaired_list,
        rdir=rdir,
        round_idx=1,
    )
    assert args[args.index("--prob_scenes") + 1] == str(repaired_list)
    resolved = Path(args[args.index("--normal_scenes") + 1])
    assert resolved == rdir / "normal_scenes_resolved.json"
    resolved_paths = json.loads(resolved.read_text())
    # 4-col passes through by original path; 3-col was rewritten to a copy.
    assert resolved_paths[1] == str(p4)
    assert resolved_paths[0] != str(p3)
    batch = [dict(np.load(p)) for p in resolved_paths]
    stacked = torch.stack([torch.from_numpy(b["neighbor_agents_future"]) for b in batch])
    assert stacked.shape == (2, 4, 80, 4)

    # Undersized n_prob_scenes fails before training instead of silently
    # dropping repaired scenes.
    tcfg.write_text(
        json.dumps({"ranked_sft_mode": "curated", "n_prob_scenes": 1, "n_normal_scenes": 2})
    )
    with pytest.raises(ValueError, match="silently subsample"):
        round_runner._rsft_scene_args(
            cfg,
            repaired_paths=repaired_paths,
            repaired_list_json=repaired_list,
            rdir=rdir,
            round_idx=1,
        )

    # Without normal_scene_list the legacy combined-list path is kept.
    args = round_runner._rsft_scene_args(
        {"training_backend": "rsft", "training_config": str(tcfg)},
        repaired_paths=repaired_paths,
        repaired_list_json=repaired_list,
        rdir=rdir,
        round_idx=1,
    )
    assert args == ["--train_scenes", str(repaired_list)]


def test_workflow_contract_forwards_prototypes_path_to_repair_cmd(tmp_path):
    """Regression pin: the contract parser used to silently drop
    repair_generation.prototypes_path, so anchor variants generated without
    their prototype library."""
    protos = tmp_path / "prototypes.npy"
    protos.write_bytes(b"")
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"backend": "base_sft", "train_args": {}}))
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
                "repair_generation": {
                    "ego_shape": "2.7,4.3,1.7",
                    "min_margin": 0.3,
                    "generation_mode": "guided_variant",
                    "variant": "anchor_fan_16",
                    "candidate_count_per_scene": 17,
                    "prototypes_path": str(protos),
                },
                "replay_memory": {"capacity": 200},
                "training": {"val_scenes": "/tmp/valid.json"},
            }
        )
    )
    cfg = round_runner._config_from_workflow_contract(
        {
            "model_path": "/tmp/model.pth",
            "scene_list": str(scene_list),
            "workflow_config": str(workflow),
            "training_config": str(training),
            "output_dir": str(tmp_path / "auto_research" / "out"),
        }
    )
    assert cfg["repair_config"]["prototypes_path"] == str(protos)
    round_runner._validate_repair_generation_config(cfg)
    cmd = round_runner._repair_cmd(
        cfg, tmp_path / "model.pth", tmp_path / "credit.jsonl", tmp_path / "round"
    )
    assert cmd[cmd.index("--prototypes_path") + 1] == str(protos)
    assert cmd[cmd.index("--variant") + 1] == "anchor_fan_16"
    assert cmd[cmd.index("--K") + 1] == "17"


def test_validate_repair_generation_config_is_slot_derived():
    """Anchor-slot variants need prototypes and enough K at STARTUP, for both
    config paths — derived from the variant's slots, not its name."""
    base = {"generation_mode": "guided_variant", "K": 17}
    with pytest.raises(ValueError, match="prototypes_path"):
        round_runner._validate_repair_generation_config(
            {"repair_config": {**base, "variant": "anchor_fan_16"}}
        )
    with pytest.raises(ValueError, match="K >= 17"):
        round_runner._validate_repair_generation_config(
            {
                "repair_config": {
                    "generation_mode": "guided_variant",
                    "K": 8,
                    "variant": "anchor_fan_16",
                    "prototypes_path": "/tmp/p.npy",
                }
            }
        )
    with pytest.raises(ValueError):  # unknown variant names fail at startup too
        round_runner._validate_repair_generation_config(
            {"repair_config": {**base, "variant": "no_such_variant"}}
        )
    # a null/empty variant means "not set", never the literal string "None"
    round_runner._validate_repair_generation_config({"repair_config": {**base, "variant": None}})
    round_runner._validate_repair_generation_config({"repair_config": {**base, "variant": " "}})
    # valid anchor config passes; non-anchor variants need no prototypes
    round_runner._validate_repair_generation_config(
        {"repair_config": {**base, "variant": "anchor_fan_16", "prototypes_path": "/tmp/p.npy"}}
    )
    round_runner._validate_repair_generation_config(
        {"repair_config": {"generation_mode": "grpo_temperature", "K": 8, "variant": "rsft_v2"}}
    )


def test_dry_run_round0_guards_respect_partial_configs(tmp_path, monkeypatch, capsys):
    """A validator-passing guards section with only some probes must dry-run
    without crashing, printing exactly the configured probes."""
    _, benchmark = _guard_assets(tmp_path)
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text(json.dumps(["/tmp/a.npz"]))
    cfg = {
        "rounds": 1,
        "epochs_per_round": 1,
        "model_path": "/tmp/model.pth",
        "val_scenes": "/tmp/valid.json",
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "replay_memory": {"capacity": 10},
        "training_config": {"train_args": {}},
        "training_backend": "base_sft",
        "output_dir": str(tmp_path / "auto_research" / "out"),
        "repair_config": {"ego_shape": "2.7,4.3,1.7", "min_margin": 0.3},
        "mine_labels": ["expert_disagreement"],
        "perception_mining": {"scene_list": str(scene_list), "chunk_len": 80},
        "guards": {"patience_benchmark": str(benchmark)},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        sys, "argv", ["run_lifelong_r2lpl_rounds", "--config", str(cfg_path), "--dry_run"]
    )
    round_runner.main()
    output = capsys.readouterr().out
    assert "[round 0] guard_onset:" in output
    assert "guard_mine" not in output
    assert "guard_closed_loop" not in output


def test_patience_stop_metrics_cover_both_directions():
    """fail_to_stop covers only the stopping direction; the take-off direction
    (GT resumes, plan parks) must be measured too."""
    from rlvr.autoresearch.tools.eval_patience_onset import _stop_metrics

    dt = 0.1

    def _xy(speeds):
        x = np.concatenate([[0.0], np.cumsum(np.asarray(speeds) * dt)])
        return np.column_stack([x, np.zeros_like(x)])

    # stop at 2s, resume at 5s
    gt = _stop_metrics(_xy([5.0] * 20 + [0.0] * 30 + [5.0] * 30), stop_speed=0.5)
    assert gt["stops"] and gt["resumes"]
    assert gt["stop_onset_s"] == pytest.approx(2.0)
    assert gt["resume_onset_s"] == pytest.approx(5.0)
    # stop and park forever
    parker = _stop_metrics(_xy([5.0] * 20 + [0.0] * 60), stop_speed=0.5)
    assert parker["stops"] and not parker["resumes"]
    # never stops
    cruiser = _stop_metrics(_xy([5.0] * 80), stop_speed=0.5)
    assert not cruiser["stops"] and not cruiser["resumes"]


def test_guard_mining_metrics_split_expert_disagreement_by_reason(tmp_path):
    gdir = tmp_path / "guards"
    gdir.mkdir()
    rows = [
        {
            "scene_path": "/tmp/a.npz",
            "label": "expert_disagreement",
            "event_key": "e1",
            "expert_disagreement_reason": "expert_wait_model_forward",
        },
        {
            "scene_path": "/tmp/b.npz",
            "label": "expert_disagreement",
            "event_key": "e1",
            "expert_disagreement_reason": "expert_wait_model_forward",
        },
        {
            "scene_path": "/tmp/c.npz",
            "label": "expert_disagreement",
            "event_key": "e2",
            "expert_disagreement_reason": "model_lagging_expert",
        },
        {"scene_path": "/tmp/d.npz", "label": "road_border_crossing", "event_key": "e3"},
    ]
    round_runner._write_jsonl(gdir / "credit_windows.jsonl", rows)
    (gdir / "summary.json").write_text(json.dumps({"simulated_chunks": 40}))
    metrics = round_runner._guard_mining_metrics(gdir)
    # per-reason counts sum to the label aggregate; fail-to-take-off visible
    assert metrics["expert_disagreement_by_reason"] == {
        "expert_wait_model_forward": 1,
        "model_lagging_expert": 1,
    }
    assert metrics["event_count_by_label"]["expert_disagreement"] == 2


def test_closed_loop_guard_section_tolerates_comment_keys(tmp_path):
    """The '_'-comment convention must hold in nested sections too: validator
    tolerance and no --_comment* leaking into the probe command."""
    manifest, benchmark = _guard_assets(tmp_path)
    npz_root = tmp_path / "cl_root"
    npz_root.mkdir()
    guards = {
        "frozen_chunk_manifest": str(manifest),
        "patience_benchmark": str(benchmark),
        "closed_loop_npz_root": str(npz_root),
        "closed_loop": {"seg_len": 600, "_comment": "throttle rendering"},
    }
    round_runner._validate_guards_config({"guards": guards})
    cmd = round_runner._closed_loop_probe_cmd({"guards": guards}, tmp_path / "model.pth")
    assert "--seg_len" in cmd
    assert not [arg for arg in cmd if arg.startswith("--_")]


def test_ensure_4col_rejects_unexpected_channel_count(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, neighbor_agents_future=np.zeros((2, 80, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="5 channels"):
        round_runner._ensure_4col_neighbor_futures([str(bad)], tmp_path / "out")


def test_selection_report_vetoes_and_ranks():
    def _m(path, events, fail_stop, fail_resume, sim=40):
        return {
            "model_path": path,
            "frozen_chunk_mining": {
                "event_count_by_label": dict(events),
                "total_events": sum(events.values()),
                "simulated_chunks": sim,
            },
            "patience_onset": {"fail_to_stop": fail_stop, "fail_to_resume": fail_resume},
        }

    reference = _m("/m/base.pth", {"moving_collision": 5, "road_border_crossing": 2}, 0, 16)
    candidates = [
        (1, _m("/m/r1.pth", {"moving_collision": 3, "road_border_crossing": 1}, 0, 5)),
        (2, _m("/m/r2.pth", {"moving_collision": 3}, 0, 7)),
        (3, _m("/m/r3.pth", {"moving_collision": 6}, 0, 7)),  # mc regression
        (4, _m("/m/r4.pth", {"moving_collision": 3}, 1, 6)),  # fail_to_stop regression
    ]
    report = round_runner._selection_report(reference, candidates)
    by_round = {r["round_idx"]: r for r in report["rounds"]}
    assert by_round[1]["eligible"] and by_round[2]["eligible"]
    assert not by_round[3]["eligible"] and any(
        "moving_collision" in v for v in by_round[3]["veto_reasons"]
    )
    assert not by_round[4]["eligible"] and any(
        "fail_to_stop" in v for v in by_round[4]["veto_reasons"]
    )
    # fewest total frozen events wins (r2: 3 < r1: 4)
    assert report["recommended"] == {"round_idx": 2, "checkpoint": "/m/r2.pth"}

    # denominator drift vetoes; nothing eligible -> no recommendation
    drifted = [(1, _m("/m/r1.pth", {"moving_collision": 3}, 0, 5, sim=39))]
    report = round_runner._selection_report(reference, drifted)
    assert report["recommended"] is None
    assert any("denominator" in v for v in report["rounds"][0]["veto_reasons"])


def test_road_border_gate_is_absolute():
    # The road-border gate is absolute: ANY crossing candidate is rejected, scripted
    # (depart/morph) or model/fan alike. rb_min_dist is an UNSIGNED distance to the border,
    # so it cannot tell a shoulder skim from a trajectory that crossed to the wrong side and
    # ran far outside — an earlier expert-relative distance floor was removed as unsafe.
    source_row = {
        "repair_labels": ["expert_disagreement", "road_border_crossing"],
        "expert_disagreement_max_dev": 0.5,
    }
    T = 20
    expert = np.zeros((T, 4), dtype=np.float32)
    depart = np.zeros((T, 4), dtype=np.float32)
    model = np.zeros((T, 4), dtype=np.float32)
    xrow = lambda: SimpleNamespace(  # noqa: E731
        **{**vars(_morph_outcome_reward_row(2.0, rb_crossing=True)), "rb_min_dist": 0.12}
    )
    rows = [
        _morph_outcome_candidate_row(["road_border_crossing"]),
        _morph_outcome_candidate_row(["road_border_crossing"]),
    ]

    # Both crossing candidates gate-rejected, including the scripted depart (idx 0).
    idx, meta = _best_safe_candidate(
        source_row,
        rows,
        [xrow(), xrow()],
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[depart, model],
        reference_traj=expert,
        depart_index=0,
    )
    assert idx is None and meta["depart_outcome"] == "gate_rejected"
    assert meta["depart_gate_flags"]["rb_crossing"] is True


def test_direct_reproducer_chunks_support_overlapping_stride(tmp_path):
    # start_stride < chunk_len: consecutive windows overlap (sliding read-ahead
    # buffer). Anchor density and chunk runway become independent knobs — needed
    # for long expert wait cycles where an 80-step chunk has no post-departure
    # runway but a 160 stride would halve the anchors.
    scene_list = _write_direct_chunk_scene_list(
        tmp_path,
        [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 400)],
    )

    chunks = list(iter_direct_chunks(scene_list, chunk_len=160, start_stride=80))

    # Tail window (start 320) is only 80 frames -> dropped by the default
    # min_chunk_len == chunk_len.
    assert [c.global_start_index for c in chunks] == [0, 80, 160, 240]
    assert [c.n_frames for c in chunks] == [160, 160, 160, 160]
    assert chunks[0].start_frame == 31
    assert chunks[0].end_frame == 31 + 159
    assert chunks[1].start_frame == 31 + 80  # overlaps chunk 0 by 80 frames

    with_tail = list(
        iter_direct_chunks(scene_list, chunk_len=160, start_stride=80, min_chunk_len=80)
    )
    assert [c.global_start_index for c in with_tail] == [0, 80, 160, 240, 320]
    assert with_tail[-1].n_frames == 80
    assert with_tail[-1].end_reason == "scene_list_tail"


def test_direct_reproducer_chunks_overlap_preserves_sharding(tmp_path):
    # Sharding/sampling still key off global_start // start_stride with overlap.
    scene_list = _write_direct_chunk_scene_list(
        tmp_path,
        [f"bagA_00000001_{i:08d}" for i in range(31, 31 + 400)],
    )

    shard0 = list(
        iter_direct_chunks(scene_list, chunk_len=160, start_stride=80, num_shards=2, shard_index=0)
    )
    shard1 = list(
        iter_direct_chunks(scene_list, chunk_len=160, start_stride=80, num_shards=2, shard_index=1)
    )

    assert [c.global_start_index for c in shard0] == [0, 160]
    assert [c.global_start_index for c in shard1] == [80, 240]


def test_realized_reward_scorer_per_batch_accumulate_and_teleport_skip(monkeypatch):
    """The realized-reward scorer must: score the driven future per sampled pose,
    accumulate across finalize() calls (one per mining batch) while clearing its
    buffers each time (bounds memory + avoids cross-batch id(s) aliasing), and skip
    any window that crosses an unstick teleport."""
    import rlvr.reward as _reward

    # Deterministic stand-in reward = the future's total forward displacement, so we can
    # predict which poses get scored and that teleport windows are excluded.
    def fake_reward(ego, data, cfg):
        xy = ego[0, :, :2].cpu().numpy()
        return [_reward_row(float(abs(xy[-1, 0] - xy[0, 0])))]

    monkeypatch.setattr(_reward, "compute_reward_batch", fake_reward)

    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        rc, _tc = _write_realized_scorer_configs(_P(td))
        hook, finalize = reproducer_danger_scorer.build_realized_reward_scorer(
            reward_config=rc, device="cpu", horizon=3, sample_step=1
        )

        def _np_dict():
            return {
                "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
                "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
                "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
                "ego_agent_past": np.zeros((21, 3), dtype=np.float32),
            }

        # Batch 1: one segment, 6 steps, straight line, a teleport lands at k=3.
        seg = SimpleNamespace(k=0, live_pose=np.zeros(3), snap_count=0)
        for k in range(6):
            seg.k = k
            seg.live_pose = np.array([float(k), 0.0, 0.0])  # 1 m/step forward
            if k == 3:
                seg.snap_count = 1  # teleport landed here
            hook([(seg, _np_dict())], None, None, "cpu")
        mean1, n1 = finalize()
        # horizon=3 → sampled poses t need t+1..t+3 present AND no teleport in that window.
        # k can be 0,1,2 (k+3<=5). k=0 window {1,2,3} contains teleport@3 -> skip.
        # k=1 window {2,3,4} contains 3 -> skip. k=2 window {3,4,5} contains 3 -> skip.
        assert n1 == 0, f"all windows cross the teleport, expected 0 scored, got {n1}"

        # Batch 2: fresh segment (id reused is fine — buffers were cleared), 5 steps, no teleport.
        seg2 = SimpleNamespace(k=0, live_pose=np.zeros(3), snap_count=0)
        for k in range(5):
            seg2.k = k
            seg2.live_pose = np.array([float(k), 0.0, 0.0])
            hook([(seg2, _np_dict())], None, None, "cpu")
        mean2, n2 = finalize()
        # k=0 {1,2,3}, k=1 {2,3,4} valid; k=2 needs {3,4,5} -> 5 absent -> skip. So 2 scored.
        assert n2 == 2, f"expected 2 scored poses in batch 2, got {n2}"
        # running total accumulates across batches (n2 cumulative); fake reward = intra-window fwd disp
        assert mean2 == pytest.approx(2.0, abs=1e-6)  # fake reward = intra-window forward disp


def test_realized_reward_scorer_buffers_cleared_after_finalize(monkeypatch):
    """finalize() must clear its internal buffers so a second finalize on no new data
    does not re-score the same poses."""
    import rlvr.reward as _reward

    monkeypatch.setattr(_reward, "compute_reward_batch", lambda ego, data, cfg: [_reward_row(1.0)])
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        rc, _ = _write_realized_scorer_configs(_P(td))
        hook, finalize = reproducer_danger_scorer.build_realized_reward_scorer(
            reward_config=rc, device="cpu", horizon=2, sample_step=1
        )
        nd = {
            "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
            "neighbor_agents_past": np.zeros((1, 31, 11), dtype=np.float32),
            "neighbor_agents_future": np.zeros((1, 80, 4), dtype=np.float32),
            "ego_agent_past": np.zeros((21, 3), dtype=np.float32),
        }
        seg = SimpleNamespace(k=0, live_pose=np.zeros(3), snap_count=0)
        for k in range(4):
            seg.k = k
            seg.live_pose = np.array([float(k), 0.0, 0.0])
            hook([(seg, nd)], None, None, "cpu")
        _, n_first = finalize()
        _, n_second = finalize()  # no new data
        assert n_first > 0
        assert n_second == n_first, "second finalize must not re-score cleared buffers"


def test_main_replay_refresh_keeps_max_of_frozen_and_fresh(tmp_path, monkeypatch):
    """End-to-end wiring of replay_refresh through main(), not just the join helper.

    Two replayed scenes seeded from a previous link's memory: for one the current policy
    proposes a better target (must switch to the fresh NPZ), for the other a worse one (must
    keep the frozen NPZ). Asserts the refresh actually ran, in its own subdirectory, and that
    the TRAIN LIST is what changed -- the failure mode this guards against is a refresh that
    computes a list nobody trains on.
    """
    manifest, benchmark = _guard_assets(tmp_path)
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text(json.dumps(["/tmp/a.npz"]))
    out_dir = tmp_path / "auto_research" / "out"
    initial_model = tmp_path / "initial.pth"
    torch.save({"model": {}}, initial_model)

    frozen_better = str(tmp_path / "frozen_better.npz")  # policy regressed here
    frozen_worse = str(tmp_path / "frozen_worse.npz")  # policy improved here
    src_better, src_worse = str(tmp_path / "src_b.npz"), str(tmp_path / "src_w.npz")
    fresh_worse = str(tmp_path / "fresh_w.npz")
    seed_memory = tmp_path / "seed_memory.json"
    seed_memory.write_text(
        json.dumps(
            {
                "capacity": 10,
                "entries": [
                    {
                        "scene_path": frozen_better,
                        "source_scene_path": src_better,
                        "selected_total": -1.0,
                    },
                    {
                        "scene_path": frozen_worse,
                        "source_scene_path": src_worse,
                        "selected_total": -3.0,
                    },
                ],
            }
        )
    )

    refresh_dirs: list[str] = []

    def fake_repair(cfg, model_path, rdir, gpu_ids, **kwargs):
        if rdir.name == "replay_refresh":
            refresh_dirs.append(str(rdir))
            # the fresh pass must have been handed the SOURCE scenes to re-repair
            rows = [json.loads(x) for x in (rdir / "credit_windows.jsonl").read_text().splitlines()]
            assert {r["scene_path"] for r in rows} == {src_better, src_worse}
            round_runner._write_jsonl(
                rdir / "repaired_targets.jsonl",
                [
                    # worse than frozen -1.0 -> frozen must be kept
                    {
                        "scene_path": str(tmp_path / "fresh_b.npz"),
                        "source_scene_path": src_better,
                        "selected_total": -2.5,
                    },
                    # better than frozen -3.0 -> fresh must win
                    {
                        "scene_path": fresh_worse,
                        "source_scene_path": src_worse,
                        "selected_total": -0.5,
                    },
                ],
            )
            return 0.0
        round_runner._write_json(rdir / "repaired_targets.json", ["/tmp/repaired_a.npz"])
        round_runner._write_jsonl(
            rdir / "repaired_targets.jsonl", [{"scene_path": "/tmp/repaired_a.npz"}]
        )
        return 0.0

    def fake_run(cmd, log_path, *, cwd=None, env=None):
        if "--out_memory" in cmd:
            round_runner._write_json(Path(cmd[cmd.index("--out_memory") + 1]), {"entries": []})
            # the memory step hands back the two seeded replay scenes
            round_runner._write_json(
                Path(cmd[cmd.index("--out_replay_scenes") + 1]), [frozen_better, frozen_worse]
            )
        return 0.0

    trained_lists: list[list[str]] = []

    def fake_training(cfg, *, model_path, train_input_list, rdir, round_idx, gpu_ids):
        trained_lists.append([str(p) for p in json.loads(Path(train_input_list).read_text())])
        ckpt = rdir / "base_train" / "latest.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}}, ckpt)
        return 0.0, ckpt

    cfg = {
        "rounds": 1,
        "epochs_per_round": 1,
        "model_path": str(initial_model),
        "val_scenes": "/tmp/valid.json",
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "replay_memory": {"capacity": 10},
        "initial_replay_memory": str(seed_memory),
        "replay_refresh": True,
        "training_config": {"train_args": {}},
        "training_backend": "base_sft",
        "output_dir": str(out_dir),
        "checkpoint_policy": "latest",
        "repair_config": {"ego_shape": "2.7,4.3,1.7", "min_margin": 0.3},
        "mine_labels": ["expert_disagreement"],
        "perception_mining": {"scene_list": str(scene_list), "chunk_len": 80},
        "guards": {"frozen_chunk_manifest": str(manifest), "patience_benchmark": str(benchmark)},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    def fake_mining(cfg, model_path, rdir, gpu_ids):
        # the runner reads this summary to fail loudly on a zero-chunk mine
        round_runner._write_json(
            rdir / "perception_direct_summary.json",
            {"planned_chunks": 1, "simulated_chunks": 1, "skipped_chunks": 0},
        )
        round_runner._write_jsonl(rdir / "credit_windows.jsonl", [{"scene_path": "/tmp/a.npz"}])
        return 0.0

    monkeypatch.setattr(round_runner, "_run_mining_phase", fake_mining)
    monkeypatch.setattr(round_runner, "_run_repair_phase", fake_repair)
    monkeypatch.setattr(round_runner, "_run", fake_run)
    monkeypatch.setattr(round_runner, "_run_base_sft_training", fake_training)
    monkeypatch.setattr(round_runner, "_run_guard_phase", lambda *a, **k: {})
    monkeypatch.setattr(round_runner, "_anchor_slice_paths", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", ["run_lifelong_r2lpl_rounds", "--config", str(cfg_path)])
    round_runner.main()

    assert refresh_dirs, "replay_refresh was configured but the refresh repair never ran"
    stats = json.loads((Path(refresh_dirs[0]) / "refresh_stats.json").read_text())
    assert stats["improved_by_fresh"] == 1 and stats["kept_frozen"] == 1

    assert trained_lists, "training never ran"
    trained = trained_lists[0]
    assert fresh_worse in trained, "the improved fresh target never reached the train list"
    assert frozen_better in trained, "the frozen target must survive where the policy regressed"
    assert frozen_worse not in trained, "the surpassed frozen target must NOT be trained on"


def _mk_reward_row(total=5.0):
    return SimpleNamespace(
        centerline=0.0,
        total=total,
        collision_step=None,
        rb_crossing=False,
        lane_crossing=False,
        static_crossing=False,
        kinematic_violated=False,
        sc_min_dist=1.0,
        rb_min_dist=1.0,
    )


def _mk_candidate_row():
    return {"moving_collision_step": None, "expert_disagreement": False, "labels": ["clean"]}


def test_unified_morph_bit_identical_to_legacy_pair():
    # THE morph: the unified entry must be byte-equal to the two legacy builders
    # on both disagreement classes (also verified bit-identical on real mined rows
    # of both classes before the legacy call sites were removed).
    import numpy as np

    from rlvr.autoresearch.tools.expert_morph import (
        build_depart_morph_candidate,
        build_expert_morph_candidate,
        build_unified_morph_candidates,
    )

    T = 80
    t = np.arange(T) * 0.1
    expert = np.zeros((T, 4), dtype=np.float64)
    expert[:, 0] = 8.0 + np.cumsum(np.minimum(1.5 * t, 7.0) * 0.1)
    expert[:, 2] = 1.0
    det = np.zeros((T, 4), dtype=np.float64)
    det[:, 2] = 1.0
    anchor = expert[-1, :2]

    for reason, expected_kinds in (
        ("model_lagging_expert", ["stay_behind", "depart"]),
        ("expert_wait_model_forward", ["stay_behind"]),
    ):
        uni = build_unified_morph_candidates(det, expert, reason, stop_anchor_xy=anchor)
        assert [k for k, _, _ in uni] == expected_kinds
        stay, stay_d = build_expert_morph_candidate(
            det, expert, stop_anchor_xy=anchor, return_diag=True
        )
        assert (stay is None) == (uni[0][1] is None)
        if stay is not None:
            assert stay.tobytes() == uni[0][1].tobytes()
        assert stay_d == uni[0][2]
        if reason == "model_lagging_expert":
            dep, dep_d = build_depart_morph_candidate(det, expert, return_diag=True)
            assert (dep is None) == (uni[1][1] is None)
            if dep is not None:
                assert dep.tobytes() == uni[1][1].tobytes()
            assert dep_d == uni[1][2]


def test_trust_region_uses_path_deviation_for_depart_only():
    # A delayed catch-up ON the expert's path: pointwise L2 is huge (schedule offset),
    # path deviation ~0. The trust region must pass it as the DEPART candidate and
    # reject the identical trajectory when it is an ordinary generated candidate.
    import numpy as np

    from rlvr.autoresearch.tools.build_repaired_targets import (
        _candidate_deviation_penalty,
        _candidate_path_deviation,
    )

    T = 80
    expert = np.zeros((T, 4), dtype=np.float32)
    expert[:, 0] = np.linspace(0, 60, T)  # cruising expert
    expert[:, 2] = 1.0
    delayed = np.zeros((T, 4), dtype=np.float32)
    delayed[:, 0] = np.concatenate([np.zeros(20), np.linspace(0, 40, 60)])  # waits, then follows
    delayed[:, 2] = 1.0
    assert _candidate_deviation_penalty(delayed, expert) > 4.0
    assert _candidate_path_deviation(delayed, expert) < 0.1

    offset = delayed.copy()
    offset[:, 1] += 6.0  # genuinely divergent geometry
    assert _candidate_path_deviation(offset, expert) > 4.0


def test_trust_region_selection_passes_delayed_depart_rejects_far_generated():
    import numpy as np

    from rlvr.autoresearch.tools.build_repaired_targets import _best_safe_candidate

    T = 80
    expert = np.zeros((T, 4), dtype=np.float32)
    expert[:, 0] = np.linspace(0, 60, T)
    expert[:, 2] = 1.0
    delayed = np.zeros((T, 4), dtype=np.float32)
    delayed[:, 0] = np.concatenate([np.zeros(20), np.linspace(0, 40, 60)])
    delayed[:, 2] = 1.0
    source_row = {
        "repair_labels": ["expert_disagreement"],
        "expert_disagreement_reason": "model_lagging_expert",
        "expert_disagreement_max_dev": 0.5,  # measured: near_log
    }
    candidate_rows = [_mk_candidate_row() for _ in range(2)]
    # The generated twin carries the HIGHER reward: if the trust region ever stops
    # rejecting it, unified selection would pick it and this test fails.
    reward_rows = [_mk_reward_row(5.0), _mk_reward_row(2.0)]
    # candidate 0 = generated with the SAME delayed trajectory; candidate 1 = depart
    trajs = [delayed, delayed]
    idx, meta = _best_safe_candidate(
        source_row,
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=trajs,
        reference_traj=expert,
        max_expert_dev_m=4.0,
        depart_index=1,
    )
    # The depart candidate's path-based deviation passes the trust region while the
    # generated twin's pointwise deviation rejects it — depart is the only survivor
    # and unified selection picks it. The depart_outcome assertion makes trust-region
    # removal detectable even where Eq.26 weights would pick depart anyway.
    assert idx == 1
    assert meta["selected_r2lpl_state_class"] == "near_log"
    assert meta["depart_outcome"] == "selected"


def test_selection_classifies_reason_only_rows_by_measured_deviation():
    # A stall that ends in a collision is labelled moving_collision while the conflict
    # detector still records model_lagging_expert. Selection must classify such rows by
    # their measured deviation (the ed-like predicate), not default them to recoverable.
    import numpy as np

    from rlvr.autoresearch.tools.build_repaired_targets import _best_safe_candidate

    T = 80
    expert = np.zeros((T, 4), dtype=np.float32)
    expert[:, 0] = np.linspace(0, 60, T)
    expert[:, 2] = 1.0
    delayed = np.zeros((T, 4), dtype=np.float32)
    delayed[:, 0] = np.concatenate([np.zeros(20), np.linspace(0, 40, 60)])
    delayed[:, 2] = 1.0
    source_row = {
        "repair_labels": ["moving_collision"],  # NOT expert_disagreement
        "expert_disagreement_reason": "model_lagging_expert",
        # 0.4 -> near_log: discriminates from BOTH failure modes (pre-unification
        # the row was not ed-like -> recoverable; a dropped measurement -> recoverable).
        "expert_disagreement_max_dev": 0.4,
    }
    candidate_rows = [_mk_candidate_row() for _ in range(2)]
    reward_rows = [_mk_reward_row(1.0), _mk_reward_row(2.0)]
    idx, meta = _best_safe_candidate(
        source_row,
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[delayed, delayed],
        reference_traj=expert,
        max_expert_dev_m=4.0,
        depart_index=1,
    )
    assert idx == 1
    assert meta["selected_r2lpl_state_class"] == "near_log"


def test_state_class_handles_missing_and_measured_deviation():
    from rlvr.autoresearch.tools.build_repaired_targets import _r2lpl_state_class

    base = {
        "repair_labels": ["moving_collision"],
        "expert_disagreement_reason": "model_lagging_expert",
    }
    # no measurement -> the middle class, never a phantom near_log
    assert _r2lpl_state_class(dict(base)) == "recoverable"
    assert _r2lpl_state_class({**base, "expert_disagreement_max_dev": 0.4}) == "near_log"
    assert _r2lpl_state_class({**base, "expert_disagreement_max_dev": 3.0}) == "recoverable"
    assert _r2lpl_state_class({**base, "expert_disagreement_max_dev": 9.0}) == "far_offpolicy"
    # pure safety row without any disagreement measurement
    assert _r2lpl_state_class({"repair_labels": ["moving_collision"]}) == "recoverable"


def test_preflight_recorded_anchor_checks_reason_only_rows(tmp_path):
    """Reason-only ed-like rows reach morph synthesis, so the preflight must check
    them too — a strict-label preflight once let a run die hours into generation."""
    import numpy as np

    from rlvr.autoresearch.tools.build_repaired_targets import _preflight_recorded_anchor

    scene = tmp_path / "row.npz"
    np.savez(scene, ego_agent_future=np.zeros((80, 3), dtype=np.float32))
    rows = [
        {
            "scene_path": str(scene),
            "repair_labels": ["moving_collision"],
            "expert_disagreement_reason": "model_lagging_expert",
        }
    ]
    with pytest.raises(ValueError, match="ego_recorded_future"):
        _preflight_recorded_anchor(rows)


def test_selection_breaks_score_ties_by_lower_deviation():
    import numpy as np

    from rlvr.autoresearch.tools.build_repaired_targets import _best_safe_candidate

    T = 80
    expert = np.zeros((T, 4), dtype=np.float32)
    expert[:, 0] = np.linspace(0, 60, T)
    expert[:, 2] = 1.0
    near = expert.copy()
    near[:, 1] += 0.5
    far = expert.copy()
    far[:, 1] += 2.0
    source_row = {"repair_labels": ["moving_collision"]}
    candidate_rows = [_mk_candidate_row() for _ in range(2)]
    # identical rewards -> identical Eq.26 scores -> the deviation tiebreak decides
    reward_rows = [_mk_reward_row(2.0), _mk_reward_row(2.0)]
    idx, _meta = _best_safe_candidate(
        source_row,
        candidate_rows,
        reward_rows,
        min_static_margin=0.3,
        target_gt_disagreement_thresh=2.0,
        candidate_trajs=[far, near],
        reference_traj=expert,
    )
    assert idx == 1


@pytest.mark.parametrize(
    ("section", "knob", "value"),
    [
        ("repair_generation", "lagging_expert_target", "force"),
        ("repair_generation", "expert_target_require_clear", True),
        ("repair_generation", "state_class_mode", "state_dev"),
        ("event_mining", "expert_min_end_progress_m", 1.0),
    ],
)
def test_workflow_contract_rejects_removed_knobs(tmp_path, section, knob, value):
    import json as _json

    workflow_dict = {
        "judgement": {
            "reward_config": "/tmp/reward.json",
            "threshold_config": "/tmp/thresholds.json",
            "credit_window_config": "/tmp/credit.json",
            "enabled_labels": ["moving_collision"],
        },
        "repair_generation": {"ego_shape": "from_npz", "min_margin": 0.3},
        "replay_memory": {"capacity": 200},
        "training": {"val_scenes": "/tmp/valid.json"},
    }
    workflow_dict.setdefault(section, {})[knob] = value
    workflow = tmp_path / "workflow.json"
    workflow.write_text(_json.dumps(workflow_dict))
    training = tmp_path / "training.json"
    training.write_text(_json.dumps({"backend": "base_sft", "train_args": {}}))
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text("[]")
    with pytest.raises(ValueError, match=knob):
        round_runner._config_from_workflow_contract(
            {
                "model_path": "/tmp/model.pth",
                "scene_list": str(scene_list),
                "workflow_config": str(workflow),
                "training_config": str(training),
                "output_dir": str(tmp_path / "auto_research" / "out"),
            }
        )


def test_refresh_join_rejects_negative_min_gain(tmp_path):
    from rlvr.autoresearch.tools.refresh_replay_targets import join

    with pytest.raises(ValueError, match="min_gain"):
        join(
            replay_scenes=tmp_path / "replay.json",
            prev_rows=[tmp_path / "rows.jsonl"],
            fresh_rows=[tmp_path / "fresh.jsonl"],
            out_list=tmp_path / "out.json",
            out_stats=tmp_path / "stats.json",
            min_gain=-0.1,
        )


def test_direct_config_rejects_removed_knobs(tmp_path):
    """The tombstones must also fire on the direct --config path, not only the
    workflow-contract parser."""
    import json as _json

    cfg = {
        "rounds": 1,
        "epochs_per_round": 1,
        "model_path": "/tmp/model.pth",
        "scene_list": "/tmp/scenes.json",
        "val_scenes": "/tmp/valid.json",
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "output_dir": str(tmp_path / "auto_research" / "out"),
        "training_config": "/tmp/training.json",
        "perception_mining": {"batch_size": 1},
        "repair_config": {
            "ego_shape": "from_npz",
            "min_margin": 0.3,
            "state_class_mode": "state_dev",
        },
        "replay_memory": {"capacity": 10},
    }
    path = tmp_path / "cfg.json"
    path.write_text(_json.dumps(cfg))
    with pytest.raises(ValueError, match="state_class_mode"):
        round_runner._load_config(path)
