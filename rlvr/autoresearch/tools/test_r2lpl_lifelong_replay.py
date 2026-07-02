from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rlvr.autoresearch.tools.build_avoiding_target import (
    _best_safe_candidate,
    _filtered_npz_payload,
    _future4_to_3col,
)
from rlvr.autoresearch.tools.lifelong_replay_memory import build_memory
from rlvr.autoresearch.tools.mine_credit_window_scenes import (
    _resolve_row,
    _validate_credit_config,
)
from rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds import (
    _base_train_invocation,
    _load_config,
    _lora_for_policy,
    _perception_mining_cmd,
    _union_scene_lists,
)
from rlvr.deviation import rollout_gt_deviation
from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss
from scenario_generation.conflict_detector import detect_expert_disagreement
from scenario_generation.danger_event_selection import (
    OnlineEventSelector,
    decluster_indices,
    sustained_true_indices,
)


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
    cfg.write_text(json.dumps({"static_collision": 15}))

    try:
        _validate_credit_config(cfg, {"static_collision", "expert_disagreement"})
    except ValueError as exc:
        assert "expert_disagreement" in str(exc)
    else:
        raise AssertionError("missing label should fail loudly")


def test_lineage_resolver_maps_route_frame_and_step(tmp_path):
    route = [tmp_path / f"bagA_{i:04d}.npz" for i in range(100, 121)]
    row = {
        "scene_path": str(tmp_path / "bagA_0105.npz"),
        "labels": ["static_collision"],
        "static_collision_step": 7,
    }

    [resolved] = _resolve_row(row, {"bagA": route}, {"static_collision": 15})

    assert resolved["route_key"] == "bagA"
    assert resolved["frame_index"] == 105
    assert resolved["offense_frame"] == 112
    assert resolved["offense_index"] == 12
    assert resolved["start_frame"] == 100
    assert resolved["start_index"] == 0


def test_lineage_resolver_rejects_non_route_path(tmp_path):
    row = {"scene_path": str(tmp_path / "hand_curated_scene.npz"), "labels": ["static_collision"]}

    try:
        _resolve_row(row, {}, {"static_collision": 15})
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


def test_round_runner_requires_scene_pool_without_perception_mining(tmp_path):
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
                "output_dir": str(tmp_path / "auto_research" / "run"),
            }
        )
    )

    try:
        _load_config(cfg)
    except ValueError as exc:
        assert "scene_pool" in str(exc)
        assert "scene_pool_root" in str(exc)
    else:
        raise AssertionError("scene_pool fields should be required without perception_mining")


def test_round_runner_perception_mining_defaults_video_off(tmp_path):
    cfg = {
        "reward_config": "/tmp/reward.json",
        "threshold_config": "/tmp/thresholds.json",
        "credit_window_config": "/tmp/credit.json",
        "perception_mining": {
            "npz_root": "/data/route",
            "seg_len": 600,
            "max_segments": 1,
        },
    }

    cmd, save_dir = _perception_mining_cmd(cfg, tmp_path / "model.pth", tmp_path / "round")

    assert "--dump_hits" in cmd
    assert cmd[cmd.index("--dump_hits") + 1] == "0"
    assert "--render_webm" not in cmd
    assert "--danger_save_dir" in cmd
    assert save_dir == tmp_path / "round" / "perception_danger_windows"


def test_shared_declustering_matches_replay_step_semantics():
    assert decluster_indices([1, 2, 4, 12, 13], window=10) == [1, 12]


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


def test_union_scene_lists_dedupes_current_and_replay(tmp_path):
    out = tmp_path / "train.json"
    _union_scene_lists(["/a.npz", "/b.npz"], ["/b.npz", "/c.npz"], out)
    assert json.loads(out.read_text()) == ["/a.npz", "/b.npz", "/c.npz"]


def test_filtered_npz_payload_drops_string_fields():
    payload = _filtered_npz_payload(
        {
            "ego_agent_future": np.zeros((80, 4), dtype=np.float32),
            "origin": np.asarray("map"),
            "token": np.asarray("abc"),
        }
    )

    assert "ego_agent_future" in payload
    assert "origin" not in payload
    assert "token" not in payload


def test_future4_to_3col_restores_yaw():
    traj = np.array([[1.0, 2.0, 0.0, 1.0], [3.0, 4.0, 1.0, 0.0]], dtype=np.float32)
    out = _future4_to_3col(traj)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, :2], traj[:, :2])
    assert np.allclose(out[:, 2], np.array([np.pi / 2, 0.0], dtype=np.float32))


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
    assert env["PYTHONPATH"].endswith("/diffusion_planner")
    assert "-m" in cmd
    assert "train_predictor" in cmd
    assert cmd[cmd.index("--train_epochs") + 1] == "7"
    assert cmd[cmd.index("--batch_size") + 1] == "2"


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
