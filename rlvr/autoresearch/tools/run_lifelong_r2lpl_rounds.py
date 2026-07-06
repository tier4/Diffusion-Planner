#!/usr/bin/env python3
"""Run configurable R2LPL-style lifelong replay rounds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

_LEGACY_REQUIRED = {
    "rounds",
    "epochs_per_round",
    "model_path",
    "val_scenes",
    "reward_config",
    "threshold_config",
    "credit_window_config",
    "replay_memory",
    "training_config",
    "output_dir",
}
_DEFAULT_ENABLED_LABELS = ["road_border_crossing", "moving_collision"]
_BASE_TRAINING_KEYS = {
    "train_args",
    "batch_size",
    "learning_rate",
    "num_workers",
    "save_utd",
    "warm_up_epoch",
}
_RSFT_TRAINING_KEYS = {
    "ranked_sft_mode",
    "use_lora",
    "num_generations",
    "generation_variant",
    "replay_loss_weight",
    "replay_der_coef",
}


def _load_any_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_json(path: Path) -> dict[str, Any]:
    raw = _load_any_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _validate_output_dir(path_value: str | os.PathLike[str]) -> Path:
    out = Path(path_value).resolve()
    if "auto_research" not in out.parts:
        raise ValueError(f"output_dir must be under an auto_research area, got {out}")
    return out


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _workflow_count_rear_end_collisions(judgement: dict[str, Any]) -> bool:
    if "count_rear_end_collisions" in judgement:
        return bool(judgement["count_rear_end_collisions"])
    if "ignore_rear_end_collisions" in judgement:
        return not bool(judgement["ignore_rear_end_collisions"])
    return True


def _parse_route_source(contract: dict[str, Any]) -> tuple[str | None, str | None]:
    scene_list = _first_non_null(contract.get("scene_list"), contract.get("scene_pool"))
    route_root = _first_non_null(
        contract.get("route_root"),
        contract.get("scene_pool_root"),
    )
    return scene_list, route_root


def _training_config_payload(path_or_dict: Any) -> dict[str, Any]:
    if isinstance(path_or_dict, (str, os.PathLike)):
        return _load_json(Path(path_or_dict))
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    raise ValueError(f"unsupported training config source: {type(path_or_dict).__name__}")


def _val_scenes_from_training_cfg(training_cfg: dict[str, Any]) -> str | None:
    train_args = training_cfg.get("train_args")
    return _first_non_null(
        training_cfg.get("val_scenes"),
        training_cfg.get("validation_scenes"),
        training_cfg.get("valid_set_list"),
        train_args.get("valid_set_list") if isinstance(train_args, dict) else None,
    )


def _infer_training_backend(training_cfg: dict[str, Any]) -> str:
    explicit = training_cfg.get("backend")
    if explicit in {"base_sft", "rsft"}:
        return str(explicit)
    mode = training_cfg.get("mode")
    if mode in {"full_model", "base_sft"}:
        return "base_sft"
    if mode in {"lora", "rsft"}:
        return "rsft"
    if "train_args" in training_cfg or any(key in training_cfg for key in _BASE_TRAINING_KEYS):
        return "base_sft"
    if any(key in training_cfg for key in _RSFT_TRAINING_KEYS):
        return "rsft"
    return "base_sft"


def _legacy_from_workflow_contract(contract: dict[str, Any]) -> dict[str, Any]:
    workflow_source = contract.get("workflow_config")
    if workflow_source is None:
        raise ValueError("workflow_config is required for the single-entry orchestrator contract")
    workflow = (
        _load_json(Path(workflow_source))
        if isinstance(workflow_source, (str, os.PathLike))
        else dict(workflow_source)
    )
    training_source = contract.get("training_config")
    if training_source is None:
        raise ValueError("training_config is required for the single-entry orchestrator contract")
    training_cfg = _training_config_payload(training_source)

    inference = dict(workflow.get("inference") or {})
    judgement = dict(workflow.get("judgement") or {})
    event_mining = dict(workflow.get("event_mining") or {})
    reproducer = dict(workflow.get("perception_reproducer") or {})
    repair = dict(workflow.get("repair_generation") or {})
    replay = dict(workflow.get("replay_memory") or {})
    rounds = dict(workflow.get("rounds") or {})
    training_section = dict(workflow.get("training") or {})

    scene_list, route_root = _parse_route_source(contract)
    if scene_list is None and route_root is None:
        raise ValueError("one of scene_list or route_root is required")

    reward_config = _first_non_null(
        judgement.get("reward_config"),
        judgement.get("reward_config_path"),
        contract.get("reward_config"),
    )
    threshold_config = _first_non_null(
        judgement.get("threshold_config"),
        judgement.get("threshold_config_path"),
        contract.get("threshold_config"),
    )
    credit_window_config = _first_non_null(
        judgement.get("credit_window_config"),
        judgement.get("credit_window_config_path"),
        contract.get("credit_window_config"),
    )
    if reward_config is None or threshold_config is None or credit_window_config is None:
        raise ValueError(
            "workflow_config.judgement must define reward_config, threshold_config, "
            "and credit_window_config"
        )

    mode = str(_first_non_null(inference.get("mode"), "det"))
    if mode not in {"det", "saved_predictions"}:
        raise ValueError(f"inference.mode must be 'det' or 'saved_predictions', got {mode!r}")
    trajectory = "saved_pred" if mode == "saved_predictions" else "det"

    enabled_labels = judgement.get("enabled_labels") or list(_DEFAULT_ENABLED_LABELS)
    if not isinstance(enabled_labels, list) or not enabled_labels:
        raise ValueError("judgement.enabled_labels must be a non-empty list")

    repair_cfg = {
        "ego_shape": repair.get("ego_shape"),
        "min_margin": _first_non_null(
            repair.get("min_margin"),
            repair.get("static_min_margin"),
            repair.get("acceptance_gates", {}).get("min_static_margin")
            if isinstance(repair.get("acceptance_gates"), dict)
            else None,
        ),
        "K": int(_first_non_null(repair.get("candidate_count_per_scene"), repair.get("K"), 8)),
        "variant": _first_non_null(
            repair.get("variant"),
            repair.get("generation_variant"),
            "rl_cl_soft_sweep_stretch",
        ),
        "gt_max_speed": float(_first_non_null(repair.get("gt_max_speed"), 9.0)),
        "scene_batch_size": int(
            _first_non_null(repair.get("generation_batch_size"), repair.get("scene_batch_size"), 8)
        ),
        "noise_low": float(
            _first_non_null(repair.get("noise_low"), repair.get("guidance_noise_low"), 0.5)
        ),
        "noise_high": float(
            _first_non_null(repair.get("noise_high"), repair.get("guidance_noise_high"), 2.0)
        ),
        "device": str(_first_non_null(repair.get("device"), "cuda")),
        "allow_conflict_candidates": bool(repair.get("allow_conflict_candidates", False)),
        "use_route_cl_guidance": bool(repair.get("use_route_cl_guidance", True)),
    }
    missing_repair = [k for k in ("ego_shape", "min_margin") if not repair_cfg.get(k)]
    if missing_repair:
        raise ValueError(
            f"workflow_config.repair_generation is missing required fields: {missing_repair}"
        )

    output_dir = _validate_output_dir(contract["output_dir"])
    val_scenes = _first_non_null(
        contract.get("val_scenes"),
        training_section.get("val_scenes"),
        _val_scenes_from_training_cfg(training_cfg),
    )
    if val_scenes is None:
        raise ValueError(
            "validation scenes are required; set val_scenes in the contract, "
            "workflow_config.training.val_scenes, or training_config"
        )

    saved_predictions_dir = contract.get("saved_predictions_dir")
    if saved_predictions_dir is None:
        saved_predictions_dir = _first_non_null(
            inference.get("saved_predictions_dir"),
            inference.get("predictions_dir"),
            inference.get("save_predictions_dir"),
        )

    training_backend = str(
        _first_non_null(training_section.get("backend"), _infer_training_backend(training_cfg))
    )

    cfg = {
        "rounds": int(_first_non_null(rounds.get("rounds"), 1)),
        "epochs_per_round": int(_first_non_null(rounds.get("epochs_per_round"), 1)),
        "model_path": str(contract["model_path"]),
        "val_scenes": str(val_scenes),
        "reward_config": str(reward_config),
        "threshold_config": str(threshold_config),
        "credit_window_config": str(credit_window_config),
        "replay_memory": {
            "capacity": int(_first_non_null(replay.get("capacity"), 200)),
            "alpha": float(_first_non_null(replay.get("alpha"), 0.5)),
            "beta": float(_first_non_null(replay.get("beta"), 0.5)),
            "arc_bin_m": float(_first_non_null(replay.get("arc_bin_m"), 25.0)),
        },
        "training_config": str(training_source)
        if isinstance(training_source, (str, os.PathLike))
        else training_source,
        "training_backend": training_backend,
        "output_dir": str(output_dir),
        "repair_config": repair_cfg,
        "trajectory": trajectory,
        "mine_labels": [str(label) for label in enabled_labels],
        "enable_conflict_detector": bool(
            _first_non_null(
                judgement.get("enable_conflict_detector"),
                judgement.get("conflict_detector_enabled"),
                False,
            )
        ),
        "danger_decluster_steps": int(
            _first_non_null(event_mining.get("danger_decluster_steps"), 10)
        ),
        "classified_decluster_steps": int(
            _first_non_null(
                event_mining.get("source_gap_steps"),
                event_mining.get("classified_decluster_steps"),
                10,
            )
        ),
        "verify_reproduced_issue": bool(
            _first_non_null(reproducer.get("verify_reproduced_issue"), True)
        ),
        "anchor_horizon_steps": int(
            _first_non_null(
                reproducer.get("anchor_horizon_steps"),
                reproducer.get("source_anchor_horizon_steps"),
                40,
            )
        ),
        "max_rollout_steps": int(
            _first_non_null(
                reproducer.get("max_rollout_steps"),
                reproducer.get("rollout_length_frames"),
                reproducer.get("rollout_length"),
                80,
            )
        ),
        "repair_window_scene_count": int(
            _first_non_null(
                reproducer.get("repair_window_scene_count"),
                reproducer.get("repair_window_scenes"),
                15,
            )
        ),
        "classify_batch_size": int(_first_non_null(inference.get("batch_size"), 32)),
        "classify_device": str(_first_non_null(inference.get("device"), "cuda")),
        "classify_prediction_scene_root": _first_non_null(
            inference.get("prediction_scene_root"),
            inference.get("source_scene_root"),
        ),
        "checkpoint_policy": str(
            _first_non_null(rounds.get("checkpoint_selection_rule"), "latest")
        ),
        "mine_batch_size": int(_first_non_null(reproducer.get("batch_size"), 16)),
        "mine_device": str(_first_non_null(reproducer.get("device"), "cuda")),
        "neighbor_history_mode": str(
            _first_non_null(reproducer.get("neighbor_history_mode"), "sim")
        ),
        "timeline_progress_mode": str(
            _first_non_null(reproducer.get("timeline_progress_mode"), "clock")
        ),
        "tracker_mode": str(_first_non_null(reproducer.get("tracker_mode"), "mpc")),
        "mine_gpu_transform": bool(_first_non_null(reproducer.get("gpu_transform"), False)),
        "mine_goal_reach_m": float(_first_non_null(reproducer.get("goal_reach_m"), 0.0)),
        "mine_render_webm": bool(_first_non_null(reproducer.get("render_webm"), False)),
        "count_rear_end_collisions": _workflow_count_rear_end_collisions(judgement),
    }
    if scene_list is not None:
        cfg["scene_pool"] = str(scene_list)
        cfg["route_scene_list"] = str(scene_list)
    if route_root is not None:
        cfg["scene_pool_root"] = str(route_root)

    if trajectory == "saved_pred":
        predictions_dir = _first_non_null(
            contract.get("saved_predictions_dir"),
            inference.get("saved_predictions_dir"),
            inference.get("predictions_dir"),
        )
        if predictions_dir is None:
            raise ValueError(
                "saved_predictions_dir is required when inference.mode is saved_predictions"
            )
        cfg["classify_predictions_dir"] = str(predictions_dir)
    else:
        if saved_predictions_dir is not None:
            cfg["classify_save_predictions_dir"] = str(saved_predictions_dir)

    return cfg


def _load_config(path: Path) -> dict[str, Any]:
    cfg = _load_json(path)
    if "workflow_config" in cfg:
        return _legacy_from_workflow_contract(cfg)

    missing = sorted(_LEGACY_REQUIRED - set(cfg))
    if not cfg.get("perception_mining") and not any(
        key in cfg for key in ("scene_pool", "scene_pool_root", "route_scene_list")
    ):
        missing.extend(k for k in ("scene_pool", "scene_pool_root") if k not in cfg)
    if "repair_config" not in cfg:
        missing.append("repair_config")
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    _validate_output_dir(cfg["output_dir"])
    return cfg


def _config_from_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        name
        for name, value in (
            ("model_path", args.model_path),
            ("workflow_config", args.workflow_config),
            ("training_config", args.training_config),
            ("output_dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"missing required CLI arguments: {missing}")
    if args.scene_list is None and args.route_root is None:
        raise ValueError("one of --scene_list or --route_root is required")
    contract = {
        "model_path": str(args.model_path),
        "scene_list": str(args.scene_list) if args.scene_list else None,
        "route_root": str(args.route_root) if args.route_root else None,
        "saved_predictions_dir": str(args.saved_predictions_dir)
        if args.saved_predictions_dir
        else None,
        "workflow_config": str(args.workflow_config),
        "training_config": str(args.training_config),
        "output_dir": str(args.output_dir),
    }
    return _legacy_from_workflow_contract(contract)


def _run(
    cmd: list[str], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=cwd, env=env
        )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}) after {elapsed:.1f}s; "
            f"see {log_path}: {' '.join(cmd)}"
        )
    print(f"  completed in {elapsed:.1f}s; log: {log_path}")
    return elapsed


def _read_json_list(path: Path) -> list[Any]:
    raw = _load_any_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    return list(raw)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_run_dir(output_dir: Path, name: str) -> Path:
    matches = sorted(output_dir.glob(f"*_{name}"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run dir found for experiment {name} under {output_dir}")
    return matches[-1]


def _lora_for_policy(run_dir: Path, policy: str) -> Path | None:
    if policy == "latest":
        latest = run_dir / "lora_latest"
        return latest if latest.exists() else None
    if policy.startswith("epoch:"):
        ep = int(policy.split(":", 1)[1])
        lora = run_dir / f"lora_epoch_{ep:03d}"
        return lora if lora.exists() else None
    if policy != "best":
        raise ValueError(f"Unknown checkpoint policy {policy!r}")
    summary = run_dir / "best_checkpoint.txt"
    if summary.exists():
        candidate = Path(summary.read_text().strip())
        return candidate if candidate.is_dir() else None
    latest = run_dir / "lora_latest"
    return latest if latest.exists() else None


def _checkpoint_for(run_dir: Path, policy: str, current_model_path: Path) -> Path:
    lora_dir = _lora_for_policy(run_dir, policy)
    if lora_dir is None:
        if policy.startswith("epoch:"):
            ep = int(policy.split(":", 1)[1])
            epoch_path = run_dir / f"epoch_{ep:03d}.pth"
            return epoch_path if epoch_path.exists() else run_dir / "latest.pth"
        return run_dir / "latest.pth"
    out = run_dir / f"merged_{policy.replace(':', '_')}.pth"
    _run(
        [
            sys.executable,
            "-m",
            "preference_optimization.merge_lora",
            "--model_path",
            str(current_model_path),
            "--lora_dir",
            str(lora_dir),
            "--output",
            str(out),
        ],
        run_dir / f"merge_{policy.replace(':', '_')}.log",
    )
    if not out.exists():
        raise FileNotFoundError(f"LoRA merge did not create expected checkpoint: {out}")
    return out


def _perception_mining_cmd(
    cfg: dict[str, Any], model_path: Path, rdir: Path
) -> tuple[list[str], Path]:
    mining = dict(cfg.get("perception_mining") or {})
    tool = str(mining.get("tool", mining.get("mode", "mine_collisions_reproducer")))
    hits_jsonl = rdir / "perception_reproducer_hits.jsonl"
    save_dir = rdir / "perception_reproducer_scenes"
    danger_save_dir = rdir / "perception_danger_windows"
    if tool in {"direct_reproducer_chunks", "direct_chunks"}:
        chunk_manifest = mining.get("chunk_manifest")
        scene_list = _first_non_null(
            mining.get("scene_list"),
            cfg.get("route_scene_list"),
            cfg.get("scene_pool"),
        )
        if scene_list is None and chunk_manifest is None:
            raise ValueError(
                "perception_mining.tool=direct_reproducer_chunks requires chunk_manifest, "
                "scene_list, route_scene_list, or scene_pool"
            )
        cmd = [
            sys.executable,
            "-m",
            "rlvr.autoresearch.tools.mine_direct_reproducer_chunks",
            "--model_path",
            str(model_path),
            "--out_dir",
            str(danger_save_dir),
            "--out_jsonl",
            str(rdir / "perception_direct_credit_windows.jsonl"),
            "--segments_jsonl",
            str(hits_jsonl),
            "--summary_json",
            str(rdir / "perception_direct_summary.json"),
            "--chunk_len",
            str(mining.get("chunk_len", 80)),
            "--start_stride",
            str(mining.get("start_stride", mining.get("chunk_len", 80))),
            "--batch_size",
            str(mining.get("batch_size", 64)),
            "--timeline_build_workers",
            str(mining.get("timeline_build_workers", 8)),
            "--n_build_threads",
            str(mining.get("n_build_threads", 16)),
            "--prefetch_ahead",
            str(mining.get("prefetch_ahead", 2)),
            "--danger_reward_config",
            str(cfg["reward_config"]),
            "--danger_threshold_config",
            str(cfg["threshold_config"]),
            "--danger_credit_window_config",
            str(cfg["credit_window_config"]),
            "--labels",
            ",".join(cfg.get("mine_labels") or []),
            "--skip_bad_chunks",
        ]
        if chunk_manifest is not None:
            cmd.extend(["--chunk_manifest", str(chunk_manifest)])
        else:
            cmd.extend(["--scene_list", str(scene_list)])
        optional_keys = (
            "max_scenes",
            "max_chunks",
            "num_shards",
            "shard_index",
            "sample_fraction",
            "sample_seed",
            "expected_frame_step",
            "min_chunk_len",
            "max_pose_step_m",
            "max_pose_speed_mps",
            "max_yaw_step_rad",
            "near_miss_thresh",
            "search_radius",
            "warmup_steps",
            "max_steps_mult",
            "goal_reach_m",
            "unstick_after",
            "unstick_advance_m",
            "device",
            "tracker_mode",
            "timeline_progress_mode",
            "neighbor_history_mode",
            "danger_decluster_steps",
        )
        for key in optional_keys:
            if key in mining and mining[key] is not None:
                cmd.extend([f"--{key}", str(mining[key])])
        if mining.get("sidecar_root"):
            cmd.extend(["--sidecar_root", str(mining["sidecar_root"])])
        if bool(mining.get("gpu_transform", True)):
            cmd.append("--gpu_transform")
        if bool(cfg.get("enable_conflict_detector", False)) or bool(
            mining.get("enable_conflict_detector", False)
        ):
            cmd.append("--enable_conflict_detector")
        if bool(mining.get("prebuild_neighbor_tracks", True)) is False:
            cmd.append("--no_prebuild_neighbor_tracks")
        if bool(mining.get("allow_existing_out_dir", False)):
            cmd.append("--allow_existing_out_dir")
        return cmd, danger_save_dir

    missing = [k for k in ("npz_root",) if k not in mining]
    if missing:
        raise ValueError(f"perception_mining is missing required fields: {missing}")
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.mine_collisions_reproducer",
        "--npz_root",
        str(mining["npz_root"]),
        "--model_path",
        str(model_path),
        "--out",
        str(hits_jsonl),
        "--seg_len",
        str(mining.get("seg_len", 600)),
        "--max_segments",
        str(mining.get("max_segments", 1)),
        "--batch_size",
        str(mining.get("batch_size", 1)),
        "--save_dir",
        str(save_dir),
        "--save_thresh",
        str(mining.get("save_thresh", 0.5)),
        "--save_pre_steps",
        str(mining.get("save_pre_steps", 80)),
        "--save_max_scenes",
        str(mining.get("save_max_scenes", 160)),
        "--save_min_pre_frames",
        str(mining.get("save_min_pre_frames", 30)),
        "--save_min_ego_speed",
        str(mining.get("save_min_ego_speed", 0.5)),
        "--dump_hits",
        str(mining.get("dump_hits", 0)),
    ]
    if bool(mining.get("danger_search", True)):
        cmd.extend(
            [
                "--danger_save_dir",
                str(danger_save_dir),
                "--danger_reward_config",
                str(cfg["reward_config"]),
                "--danger_threshold_config",
                str(cfg["threshold_config"]),
                "--danger_credit_window_config",
                str(cfg["credit_window_config"]),
                "--danger_decluster_steps",
                str(mining.get("danger_decluster_steps", 10)),
            ]
        )
    if mining.get("sidecar_root"):
        cmd.extend(["--sidecar_root", str(mining["sidecar_root"])])
    for key in (
        "near_miss_thresh",
        "search_radius",
        "warmup_steps",
        "device",
        "max_routes",
        "n_build_threads",
        "prefetch_ahead",
        "max_steps_mult",
        "unstick_after",
        "unstick_advance_m",
        "save_pre_arc_m",
        "save_min_post_snap_s",
    ):
        if key in mining:
            cmd.extend([f"--{key}", str(mining[key])])
    if bool(mining.get("preload", False)):
        cmd.append("--preload")
    if bool(mining.get("gpu_transform", False)):
        cmd.append("--gpu_transform")
    if bool(cfg.get("enable_conflict_detector", False)) or bool(
        mining.get("enable_conflict_detector", False)
    ):
        cmd.append("--enable_conflict_detector")
    if bool(mining.get("render_webm", False)):
        if int(mining.get("dump_hits", 0)) <= 0:
            raise ValueError("perception_mining.render_webm requires dump_hits > 0")
        cmd.append("--render_webm")
    if "webm_fps" in mining:
        cmd.extend(["--webm_fps", str(mining["webm_fps"])])
    return cmd, danger_save_dir if bool(mining.get("danger_search", True)) else save_dir


def _event_metadata_from_path(scene_path: str) -> dict[str, Any]:
    parent = Path(scene_path).parent.name
    meta: dict[str, Any] = {"event_key": parent, "window_dir": str(Path(scene_path).parent)}
    if "_danger_" in parent:
        stem, label = parent.rsplit("_danger_", 1)
        meta["label"] = label
    elif "_credit_" in parent:
        stem, label = parent.rsplit("_credit_", 1)
        meta["label"] = label
    else:
        return meta
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        meta["route_key"] = parts[0]
        meta["start_frame"] = int(parts[1])
        meta["event_frame"] = int(parts[2])
    return meta


def _write_scene_list_from_saved_batches(save_dir: Path, out_path: Path) -> list[str]:
    scenes = sorted(str(p) for p in save_dir.rglob("credit*.npz"))
    if not scenes:
        scenes = sorted(str(p) for p in save_dir.rglob("collision*.npz"))
    if not scenes:
        raise FileNotFoundError(f"Perception mining saved no scenes under {save_dir}")
    out_path.write_text(json.dumps(scenes, indent=2))
    return scenes


def _write_credit_rows_from_scene_list(scene_list: list[str], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for scene in scene_list:
            meta = _event_metadata_from_path(scene)
            label = str(meta.get("label", "perception_reproducer"))
            row = {
                "scene_path": scene,
                "label": label,
                "labels": [label],
                "variant_kind": "perception_reproducer",
                "window_dir": meta.get("window_dir"),
                "event_key": meta.get("event_key"),
            }
            for key in ("route_key", "start_frame", "event_frame"):
                if key in meta:
                    row[key] = meta[key]
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _args_json_for_model(model_path: Path) -> Path:
    candidates = [model_path.parent / "args.json", model_path.parent.parent / "args.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find args.json next to {model_path}")


def _checkpoint_epoch(model_path: Path) -> int:
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return 0
    if isinstance(ckpt, dict):
        try:
            return int(ckpt.get("epoch", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _resolve_norm_path(path_value: str, args_json_path: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    candidate = args_json_path.parent / path
    if candidate.exists():
        return str(candidate.resolve())
    repo_candidate = Path(__file__).resolve().parents[3] / "diffusion_planner" / path
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    raise FileNotFoundError(
        f"normalization_file_path {path_value!r} was not found relative to {args_json_path.parent} "
        f"or diffusion_planner/"
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _union_scene_lists(current_scenes: list[str], replay_scenes: list[str], out_path: Path) -> None:
    seen: set[str] = set()
    merged: list[str] = []
    for scene in [*current_scenes, *replay_scenes]:
        if scene not in seen:
            seen.add(scene)
            merged.append(scene)
    _write_json(out_path, merged)


def _classify_cmd(
    cfg: dict[str, Any],
    *,
    scene_pool: Path,
    classify_dir: Path,
    model_path: Path,
) -> list[str]:
    trajectory = str(cfg.get("trajectory", "det"))
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.classify_scene_failures",
        "--scenes",
        str(scene_pool),
        "--config",
        str(cfg["reward_config"]),
        "--threshold_config",
        str(cfg["threshold_config"]),
        "--output_dir",
        str(classify_dir),
        "--trajectory",
        trajectory,
        "--batch_size",
        str(cfg.get("classify_batch_size", 32)),
        "--device",
        str(cfg.get("classify_device", "cuda")),
    ]
    if trajectory == "det":
        cmd.extend(["--model_path", str(model_path)])
        if cfg.get("classify_save_predictions_dir"):
            cmd.extend(["--save_predictions_dir", str(Path(cfg["classify_save_predictions_dir"]))])
    elif trajectory == "saved_pred":
        predictions_dir = cfg.get("classify_predictions_dir")
        if not predictions_dir:
            raise ValueError(
                "trajectory=saved_pred requires classify_predictions_dir in the round config"
            )
        cmd.extend(["--predictions_dir", str(Path(predictions_dir))])
        if cfg.get("classify_prediction_scene_root"):
            cmd.extend(
                ["--prediction_scene_root", str(Path(cfg["classify_prediction_scene_root"]))]
            )
    else:
        raise ValueError(f"Unsupported trajectory mode {trajectory!r}")
    if bool(cfg.get("enable_conflict_detector", False)):
        cmd.append("--enable_conflict_detector")
    if bool(cfg.get("count_rear_end_collisions", False)):
        cmd.append("--count_rear_end_collisions")
    return cmd


def _mine_credit_cmd(
    cfg: dict[str, Any],
    *,
    scene_pool_root: Path | None,
    route_scene_list: Path | None,
    classify_dir: Path,
    model_path: Path,
    credit_dir: Path,
    credit_jsonl: Path,
    events_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.mine_credit_window_scenes",
        "--classified_scenes_jsonl",
        str(classify_dir / "classified_scenes.jsonl"),
        "--credit_window_config",
        str(cfg["credit_window_config"]),
        "--model_path",
        str(model_path),
        "--out_dir",
        str(credit_dir),
        "--out_jsonl",
        str(credit_jsonl),
        "--out_events_json",
        str(events_json),
        "--batch_size",
        str(cfg.get("mine_batch_size", 16)),
        "--device",
        str(cfg.get("mine_device", "cuda")),
        "--neighbor_history_mode",
        str(cfg.get("neighbor_history_mode", "sim")),
        "--timeline_progress_mode",
        str(cfg.get("timeline_progress_mode", "clock")),
        "--tracker_mode",
        str(cfg.get("tracker_mode", "mpc")),
        "--goal_reach_m",
        str(cfg.get("mine_goal_reach_m", 0.0)),
        "--classified_decluster_steps",
        str(cfg.get("classified_decluster_steps", 10)),
        "--anchor_horizon_steps",
        str(cfg.get("anchor_horizon_steps", 40)),
        "--max_rollout_steps",
        str(cfg.get("max_rollout_steps", 80)),
    ]
    if scene_pool_root is not None:
        cmd.extend(["--route_npz_root", str(scene_pool_root)])
    elif route_scene_list is not None:
        cmd.extend(["--route_scene_list", str(route_scene_list)])
    else:
        raise ValueError("either scene_pool_root or route_scene_list is required for event mining")
    mine_labels = cfg.get("mine_labels")
    if mine_labels:
        cmd.extend(["--labels", ",".join(mine_labels)])
    if bool(cfg.get("mine_gpu_transform", False)):
        cmd.append("--gpu_transform")
    if bool(cfg.get("verify_reproduced_issue", True)):
        cmd.extend(
            [
                "--verify_reproduced_issue",
                "--reward_config",
                str(cfg["reward_config"]),
                "--threshold_config",
                str(cfg["threshold_config"]),
                "--danger_decluster_steps",
                str(cfg.get("danger_decluster_steps", 10)),
            ]
        )
        if bool(cfg.get("enable_conflict_detector", False)):
            cmd.append("--enable_conflict_detector")
    return cmd


def _repair_cmd(cfg: dict[str, Any], model_path: Path, credit_jsonl: Path, rdir: Path) -> list[str]:
    repair_cfg = dict(cfg.get("repair_config") or {})
    missing = [k for k in ("ego_shape", "min_margin") if k not in repair_cfg]
    if missing:
        raise ValueError(f"repair_config is missing required fields: {missing}")
    repaired_dir = rdir / "repaired_targets"
    repaired_list = rdir / "repaired_targets.json"
    repaired_rows = rdir / "repaired_targets.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.build_avoiding_target",
        "--model",
        str(model_path),
        "--scene_rows_jsonl",
        str(credit_jsonl),
        "--config",
        str(cfg["reward_config"]),
        "--threshold_config",
        str(cfg["threshold_config"]),
        "--ego_shape",
        str(repair_cfg["ego_shape"]),
        "--min_margin",
        str(repair_cfg["min_margin"]),
        "--out_dir",
        str(repaired_dir),
        "--out_list",
        str(repaired_list),
        "--out_rows_jsonl",
        str(repaired_rows),
        "--K",
        str(repair_cfg.get("K", 8)),
        "--variant",
        str(repair_cfg.get("variant", "rl_cl_soft_sweep_stretch")),
        "--gt_max_speed",
        str(repair_cfg.get("gt_max_speed", 9.0)),
        "--scene_batch_size",
        str(repair_cfg.get("scene_batch_size", 8)),
        "--noise_low",
        str(repair_cfg.get("noise_low", 0.5)),
        "--noise_high",
        str(repair_cfg.get("noise_high", 2.0)),
        "--device",
        str(repair_cfg.get("device", "cuda")),
    ]
    if cfg.get("mine_labels"):
        cmd.extend(["--labels", ",".join(cfg["mine_labels"])])
    if bool(cfg.get("enable_conflict_detector", False)):
        cmd.append("--enable_conflict_detector")
    if bool(repair_cfg.get("allow_conflict_candidates", False)):
        cmd.append("--allow_conflict_candidates")
    if not bool(repair_cfg.get("use_route_cl_guidance", True)):
        cmd.append("--disable_route_cl_guidance")
    if bool(cfg.get("count_rear_end_collisions", False)):
        cmd.append("--count_rear_end_collisions")
    return cmd


def _base_training_cfg(path_or_dict: Any) -> dict[str, Any]:
    raw = _training_config_payload(path_or_dict)
    if "train_args" in raw:
        return dict(raw)
    return {"train_args": raw}


def _append_train_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if key == "pin_mem":
        cmd.append("--pin-mem" if bool(value) else "--no-pin-mem")
        return
    if isinstance(value, bool):
        cmd.extend([f"--{key}", "true" if value else "false"])
        return
    if isinstance(value, list):
        if key == "coeff_timestep" and value == [1.0, 1.0, 1.0, 1.0]:
            return
        raise ValueError(
            f"base trainer wrapper does not support list-valued arg {key!r} via CLI; "
            "set a scalar override or keep the model/default value"
        )
    cmd.extend([f"--{key}", str(value)])


def _base_train_invocation(
    cfg: dict[str, Any],
    *,
    model_path: Path,
    train_list: Path,
    rdir: Path,
    round_idx: int,
) -> tuple[list[str], Path, Path, dict[str, str]]:
    train_cfg = _base_training_cfg(cfg["training_config"])
    overrides = dict(train_cfg.get("train_args", {}))
    args_json = _args_json_for_model(model_path)
    base_args = _load_json(args_json)
    normalization_path = _resolve_norm_path(
        str(
            overrides.get(
                "normalization_file_path",
                base_args.get("normalization_file_path", "normalization.json"),
            )
        ),
        args_json,
    )
    save_dir = rdir / "base_train"
    save_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = _checkpoint_epoch(model_path) + int(cfg["epochs_per_round"])
    nproc = int(train_cfg.get("nproc_per_node", overrides.pop("nproc_per_node", 1)))
    master_port = str(train_cfg.get("master_port", overrides.pop("master_port", 29531 + round_idx)))
    train_scene_count = len(_read_json_list(train_list))
    if train_scene_count < 1:
        raise ValueError(f"{train_list} is empty; refusing to launch base training")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--standalone",
        "--master_port",
        master_port,
        "-m",
        "train_predictor",
        "--exp_name",
        f"r2lpl_round_{round_idx:03d}",
        "--save_dir",
        str(save_dir),
        "--train_set_list",
        str(train_list),
        "--valid_set_list",
        str(cfg["val_scenes"]),
        "--train_epochs",
        str(total_epochs),
        "--resume_model_path",
        str(model_path),
        "--normalization_file_path",
        normalization_path,
    ]

    passthrough = (
        "train_subsample_step",
        "batch_size",
        "save_utd",
        "learning_rate",
        "warm_up_epoch",
        "num_workers",
        "augment_prob",
        "augment_type",
        "num_refine",
        "ego_past_noise_std",
        "use_smoothing_future_trajectory",
        "use_data_augment",
        "seed",
        "device",
        "use_ema",
        "use_ego_history",
        "ego_history_dropout_rate",
        "use_turn_indicators",
        "coeff_position_lat_loss",
        "coeff_position_lon_loss",
        "coeff_heading_l2_loss",
        "coeff_velocity",
        "coeff_road_border_loss",
        "road_border_margin",
        "road_border_n_interp",
        "coeff_neighbor_collision_loss",
        "neighbor_collision_margin_vehicle",
        "neighbor_collision_margin_pedestrian",
        "neighbor_collision_margin_bicycle",
        "alpha_planning_loss",
        "alpha_neighbor_loss",
        "use_velocity_representation",
        "hybrid_loss_omega",
        "hybrid_loss_window",
        "guidance_scale",
        "encoder_mixer_depth",
        "encoder_fusion_depth",
        "decoder_depth",
        "num_heads",
        "hidden_dim",
        "diffusion_model_type",
        "predicted_neighbor_num",
        "agent_num",
        "future_len",
        "time_len",
        "ego_prediction_horizon",
        "agent_state_dim",
        "static_objects_state_dim",
        "static_objects_num",
        "lane_num",
        "lane_len",
        "route_num",
        "route_len",
        "polygon_num",
        "polygon_len",
        "line_string_num",
        "line_string_len",
        "pin_mem",
        "ddp",
        "port",
        "use_wandb",
        "wandb_project_name",
        "notes",
    )
    merged = {k: base_args[k] for k in passthrough if k in base_args}
    merged.update(overrides)
    merged["ddp"] = True
    merged["port"] = master_port
    if "batch_size" in merged:
        merged["batch_size"] = max(1, min(int(merged["batch_size"]), train_scene_count))
    if "warm_up_epoch" in merged:
        merged["warm_up_epoch"] = max(0, min(int(merged["warm_up_epoch"]), total_epochs))
    for key in passthrough:
        if key in merged:
            _append_train_arg(cmd, key, merged[key])

    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath_entries = [str(repo_root), str(repo_root / "diffusion_planner")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return cmd, save_dir / "latest.pth", repo_root / "diffusion_planner", env


def _prepare_scene_pool(cfg: dict[str, Any], rdir: Path) -> Path:
    scene_pool = cfg.get("scene_pool")
    if scene_pool:
        path = Path(scene_pool)
        if not path.exists():
            raise FileNotFoundError(f"scene pool list does not exist: {path}")
        return path
    scene_pool_root = cfg.get("scene_pool_root")
    if scene_pool_root is None:
        raise ValueError("scene_pool or scene_pool_root is required")
    paths = sorted(str(p) for p in Path(scene_pool_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No route NPZ files under {scene_pool_root}")
    scene_pool_path = rdir / "route_scene_pool.json"
    _write_json(scene_pool_path, paths)
    cfg["scene_pool"] = str(scene_pool_path)
    if "route_scene_list" not in cfg:
        cfg["route_scene_list"] = str(scene_pool_path)
    return scene_pool_path


def _derive_event_counts_from_credit_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    events_by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row.get("label") or "unknown")
        event_key = str(
            row.get("event_key") or row.get("window_dir") or Path(str(row["scene_path"])).parent
        )
        events_by_label[label].add(event_key)
    return {label: len(events) for label, events in sorted(events_by_label.items())}


def _summarize_round(
    *,
    cfg: dict[str, Any],
    round_idx: int,
    rdir: Path,
    scene_pool: Path,
    train_input_list: Path,
    phase_times: dict[str, float],
    next_model_path: Path,
) -> None:
    classify_summary = _load_json(rdir / "classified" / "summary.json")
    credit_rows = _read_jsonl(rdir / "credit_windows.jsonl")
    repaired_rows = _read_jsonl(rdir / "repaired_targets.jsonl")
    events_json = rdir / "selected_events.json"
    events = _read_json_list(events_json) if events_json.exists() else []
    memory = _load_json(rdir / f"round_{round_idx}_memory.json")
    unrepaired_path = rdir / "repaired_targets_unrepaired.json"
    unrepaired_rows = _read_json_list(unrepaired_path) if unrepaired_path.exists() else []
    route_scene_count = len(_read_json_list(scene_pool))
    train_scene_count = len(_read_json_list(train_input_list))

    label_counts = {
        label: int(count)
        for label, count in sorted(classify_summary.get("label_counts", {}).items())
        if label != "clean"
    }
    if events:
        event_counts = dict(sorted(Counter(str(event["label"]) for event in events).items()))
        timestamp_counts: dict[str, int] = defaultdict(int)
        for event in events:
            timestamp_counts[str(event["label"])] += int(event.get("event_member_count", 1))
        timestamp_counts = dict(sorted(timestamp_counts.items()))
    else:
        event_counts = _derive_event_counts_from_credit_rows(credit_rows)
        timestamp_counts = label_counts
    reproduced_event_counts = _derive_event_counts_from_credit_rows(credit_rows)

    summary = {
        "round_idx": int(round_idx),
        "route_scene_count": route_scene_count,
        "deterministic_predictions": {
            "loaded": int(classify_summary["n_classified"])
            if str(cfg.get("trajectory", "det")) == "saved_pred"
            else 0,
            "computed": int(classify_summary["n_classified"])
            if str(cfg.get("trajectory", "det")) == "det"
            else 0,
        },
        "violating_timestamps_by_label": timestamp_counts,
        "judged_label_counts": label_counts,
        "open_loop_event_count_by_label": event_counts,
        "distinct_events_by_label": event_counts,
        "simulated_event_count": int(sum(event_counts.values())),
        "reproduced_event_count_by_label": reproduced_event_counts,
        "reproduced_event_count": int(sum(reproduced_event_counts.values())),
        "generated_scene_count": len(credit_rows),
        "accepted_repaired_scene_count": len(repaired_rows),
        "discarded_unrepaired_scene_count": len(unrepaired_rows),
        "replay_memory_size": len(memory.get("entries", [])),
        "final_training_scene_count": train_scene_count,
        "phase_wall_time_sec": {k: round(v, 3) for k, v in sorted(phase_times.items())},
        "phase_peak_memory_mb": {k: None for k in sorted(phase_times)},
        "next_model_path": str(next_model_path),
    }
    _write_json(rdir / "round_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--scene_list", type=Path)
    parser.add_argument("--route_root", type=Path)
    parser.add_argument("--saved_predictions_dir", type=Path)
    parser.add_argument("--workflow_config", type=Path)
    parser.add_argument("--training_config", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config else _config_from_cli_args(args)

    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["model_path"])
    previous_memory: Path | None = None
    checkpoint_policy = str(cfg.get("checkpoint_policy", "latest"))
    training_backend = str(cfg.get("training_backend", "base_sft"))
    if training_backend != "base_sft" and not isinstance(
        cfg["training_config"], (str, os.PathLike)
    ):
        resolved_training_cfg = out / "resolved_training_config.json"
        _write_json(resolved_training_cfg, cfg["training_config"])
        cfg["training_config"] = str(resolved_training_cfg)
    workflow_summary: list[dict[str, Any]] = []

    for round_idx in range(1, int(cfg["rounds"]) + 1):
        rdir = out / f"r2lpl_round_{round_idx:03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        classify_dir = rdir / "classified"
        credit_dir = rdir / "credit_windows"
        credit_jsonl = rdir / "credit_windows.jsonl"
        repaired_rows_jsonl = rdir / "repaired_targets.jsonl"
        repaired_list_json = rdir / "repaired_targets.json"
        memory_json = rdir / f"round_{round_idx}_memory.json"
        replay_json = rdir / f"round_{round_idx}_replay_scenes.json"
        events_json = rdir / "selected_events.json"
        train_input_list = rdir / "train_scenes.json"
        name = f"r2lpl_round_{round_idx:03d}"
        scene_pool = _prepare_scene_pool(cfg, rdir)
        scene_pool_root = Path(cfg["scene_pool_root"]) if "scene_pool_root" in cfg else None
        route_scene_list = Path(cfg["route_scene_list"]) if "route_scene_list" in cfg else None
        use_perception_as_credit = (
            bool(
                (cfg.get("perception_mining") or {}).get("use_saved_scenes_as_credit_windows", True)
            )
            if cfg.get("perception_mining") is not None
            else False
        )

        classify_cmd = _classify_cmd(
            cfg,
            scene_pool=scene_pool,
            classify_dir=classify_dir,
            model_path=model_path,
        )
        mine_cmd = None
        if not use_perception_as_credit:
            mine_cmd = _mine_credit_cmd(
                cfg,
                scene_pool_root=scene_pool_root,
                route_scene_list=route_scene_list,
                classify_dir=classify_dir,
                model_path=model_path,
                credit_dir=credit_dir,
                credit_jsonl=credit_jsonl,
                events_json=events_json,
            )

        mem_cfg = dict(cfg["replay_memory"])
        memory_cmd = [
            sys.executable,
            "-m",
            "rlvr.autoresearch.tools.lifelong_replay_memory",
            "--current_credit_jsonl",
            str(repaired_rows_jsonl),
            "--out_memory",
            str(memory_json),
            "--out_replay_scenes",
            str(replay_json),
            "--capacity",
            str(mem_cfg.get("capacity", 200)),
            "--alpha",
            str(mem_cfg.get("alpha", 0.5)),
            "--beta",
            str(mem_cfg.get("beta", 0.5)),
            "--arc_bin_m",
            str(mem_cfg.get("arc_bin_m", 25.0)),
        ]
        if previous_memory is not None:
            memory_cmd.extend(["--previous_memory", str(previous_memory)])

        phase_times: dict[str, float] = {}
        cmds: list[tuple[str, list[str]]] = []
        perception_save_dir = None
        if cfg.get("perception_mining"):
            perception_cmd, perception_save_dir = _perception_mining_cmd(cfg, model_path, rdir)
            cmds.append(("perception_mine", perception_cmd))
        cmds.append(("classify", classify_cmd))
        if mine_cmd is not None:
            cmds.append(("mine", mine_cmd))
        cmds.append(("repair", _repair_cmd(cfg, model_path, credit_jsonl, rdir)))
        cmds.append(("memory", memory_cmd))

        for label, cmd in cmds:
            print(f"[round {round_idx}] {label}: {' '.join(cmd)}")
            if args.dry_run:
                continue
            phase_times[label] = _run(cmd, rdir / f"{label}.log")
            if label == "perception_mine":
                assert perception_save_dir is not None
                scenes = _write_scene_list_from_saved_batches(perception_save_dir, scene_pool)
                if use_perception_as_credit:
                    _write_credit_rows_from_scene_list(scenes, credit_jsonl)
            if label == "mine":
                rows = _read_jsonl(credit_jsonl)
                _write_json(rdir / "credit_windows_paths.json", [row["scene_path"] for row in rows])
            if label == "classify" and use_perception_as_credit and not credit_jsonl.exists():
                scenes = [str(p) for p in _read_json_list(scene_pool)]
                _write_credit_rows_from_scene_list(scenes, credit_jsonl)

        if args.dry_run:
            continue

        repaired_paths = [str(p) for p in _read_json_list(repaired_list_json)]
        replay_paths = [str(p) for p in _read_json_list(replay_json)]
        _union_scene_lists(repaired_paths, replay_paths, train_input_list)

        if training_backend == "base_sft":
            train_cmd, next_model_path, train_cwd, train_env = _base_train_invocation(
                cfg,
                model_path=model_path,
                train_list=train_input_list,
                rdir=rdir,
                round_idx=round_idx,
            )
            print(f"[round {round_idx}] train: {' '.join(train_cmd)}")
            phase_times["train"] = _run(train_cmd, rdir / "train.log", cwd=train_cwd, env=train_env)
            if not next_model_path.exists():
                raise FileNotFoundError(
                    f"base trainer did not create expected checkpoint: {next_model_path}"
                )
            model_path = next_model_path
        else:
            train_cmd = [
                sys.executable,
                "-m",
                "rlvr.autoresearch.run_experiment",
                "--config",
                str(cfg["training_config"]),
                "--name",
                name,
                "--model_path",
                str(model_path),
                "--train_scenes",
                str(repaired_list_json),
                "--replay_scenes",
                str(replay_json),
                "--val_scenes",
                str(cfg["val_scenes"]),
                "--output_dir",
                str(out),
                "--train_epochs",
                str(cfg["epochs_per_round"]),
                "--skip_baseline",
            ]
            print(f"[round {round_idx}] train: {' '.join(train_cmd)}")
            phase_times["train"] = _run(train_cmd, rdir / "train.log")
            run_dir = _latest_run_dir(out, name)
            model_path = _checkpoint_for(run_dir, checkpoint_policy, model_path)

        previous_memory = memory_json
        _summarize_round(
            cfg=cfg,
            round_idx=round_idx,
            rdir=rdir,
            scene_pool=scene_pool,
            train_input_list=train_input_list,
            phase_times=phase_times,
            next_model_path=model_path,
        )
        workflow_summary.append(_load_json(rdir / "round_summary.json"))
        print(f"[round {round_idx}] next model: {model_path}")

    if not args.dry_run:
        _write_json(out / "workflow_summary.json", {"rounds": workflow_summary})


if __name__ == "__main__":
    main()
