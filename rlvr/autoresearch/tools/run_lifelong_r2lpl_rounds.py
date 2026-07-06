#!/usr/bin/env python3
"""Run configurable R2LPL-style lifelong replay rounds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

_CONFIG_REQUIRED = {
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
_MINING_TOOL = "direct_reproducer_chunks"
_TORCH_DDP_FILE_STORE = Path("/tmp/tmp_dist_init")


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


def _gpu_ids_from_config(config: dict[str, Any]) -> list[int]:
    raw = _first_non_null(config.get("gpu_ids"), config.get("gpus"))
    if raw is None:
        resources = config.get("resources")
        if isinstance(resources, dict):
            raw = _first_non_null(resources.get("gpu_ids"), resources.get("gpus"))
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list) or not raw:
        raise ValueError("resources.gpu_ids must be a non-empty list or comma-separated string")
    return [int(gpu) for gpu in raw]


def _contract_scene_list(contract: dict[str, Any]) -> str | None:
    return _first_non_null(contract.get("scene_list"), contract.get("scene_pool"))


def _validate_mining_tool(mining: dict[str, Any]) -> None:
    tool = mining.get("tool", mining.get("mode"))
    if tool is not None and str(tool) != _MINING_TOOL:
        raise ValueError("perception_mining.tool must be 'direct_reproducer_chunks'")


def _has_mining_source(cfg: dict[str, Any]) -> bool:
    mining = dict(cfg.get("perception_mining") or {})
    return (
        mining.get("chunk_manifest") is not None
        or mining.get("scene_list") is not None
        or cfg.get("route_scene_list") is not None
        or cfg.get("scene_pool") is not None
    )


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


def _config_from_workflow_contract(contract: dict[str, Any]) -> dict[str, Any]:
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

    judgement = dict(workflow.get("judgement") or {})
    resources = dict(workflow.get("resources") or {})
    event_mining = dict(workflow.get("event_mining") or {})
    reproducer = dict(workflow.get("perception_reproducer") or {})
    repair = dict(workflow.get("repair_generation") or {})
    replay = dict(workflow.get("replay_memory") or {})
    rounds = dict(workflow.get("rounds") or {})
    training_section = dict(workflow.get("training") or {})

    scene_list = _contract_scene_list(contract)
    chunk_manifest = _first_non_null(
        contract.get("chunk_manifest"),
        event_mining.get("chunk_manifest"),
        reproducer.get("chunk_manifest"),
    )
    if scene_list is None and chunk_manifest is None:
        raise ValueError("one of scene_list or chunk_manifest is required")

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

    training_backend = str(
        _first_non_null(training_section.get("backend"), _infer_training_backend(training_cfg))
    )
    gpu_ids = _gpu_ids_from_config({"resources": resources})
    chunk_len = int(
        _first_non_null(
            event_mining.get("chunk_len"),
            reproducer.get("chunk_len"),
            reproducer.get("max_rollout_steps"),
            reproducer.get("rollout_length_frames"),
            reproducer.get("rollout_length"),
            80,
        )
    )
    perception_mining = {
        "tool": _MINING_TOOL,
        "chunk_len": chunk_len,
        "start_stride": int(
            _first_non_null(
                event_mining.get("start_stride"),
                reproducer.get("start_stride"),
                chunk_len,
            )
        ),
        "batch_size": int(_first_non_null(reproducer.get("batch_size"), 64)),
        "timeline_build_workers": int(_first_non_null(reproducer.get("timeline_build_workers"), 8)),
        "n_build_threads": int(_first_non_null(reproducer.get("n_build_threads"), 16)),
        "prefetch_ahead": int(_first_non_null(reproducer.get("prefetch_ahead"), 2)),
        "gpu_transform": bool(_first_non_null(reproducer.get("gpu_transform"), True)),
        "neighbor_history_mode": str(
            _first_non_null(reproducer.get("neighbor_history_mode"), "sim")
        ),
        "timeline_progress_mode": str(
            _first_non_null(reproducer.get("timeline_progress_mode"), "clock")
        ),
        "tracker_mode": str(_first_non_null(reproducer.get("tracker_mode"), "mpc")),
        "goal_reach_m": float(_first_non_null(reproducer.get("goal_reach_m"), 0.0)),
        "danger_decluster_steps": int(
            _first_non_null(event_mining.get("danger_decluster_steps"), 10)
        ),
    }
    if scene_list is not None:
        perception_mining["scene_list"] = str(scene_list)
    if chunk_manifest is not None:
        perception_mining["chunk_manifest"] = str(chunk_manifest)
    for key in (
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
        "unstick_after",
        "unstick_advance_m",
        "device",
        "sidecar_root",
        "prebuild_neighbor_tracks",
    ):
        value = _first_non_null(event_mining.get(key), reproducer.get(key))
        if value is not None:
            perception_mining[key] = value

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
        "repair_window_scene_count": int(
            _first_non_null(
                reproducer.get("repair_window_scene_count"),
                reproducer.get("repair_window_scenes"),
                15,
            )
        ),
        "checkpoint_policy": str(
            _first_non_null(rounds.get("checkpoint_selection_rule"), "latest")
        ),
        "count_rear_end_collisions": _workflow_count_rear_end_collisions(judgement),
        "perception_mining": perception_mining,
    }
    if gpu_ids:
        cfg["gpu_ids"] = gpu_ids
    if scene_list is not None:
        cfg["scene_pool"] = str(scene_list)
        cfg["route_scene_list"] = str(scene_list)

    return cfg


def _load_config(path: Path) -> dict[str, Any]:
    cfg = _load_json(path)
    if "workflow_config" in cfg:
        return _config_from_workflow_contract(cfg)

    missing = sorted(_CONFIG_REQUIRED - set(cfg))
    if not cfg.get("perception_mining"):
        missing.append("perception_mining")
    if "repair_config" not in cfg:
        missing.append("repair_config")
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    _validate_output_dir(cfg["output_dir"])
    mining = dict(cfg.get("perception_mining") or {})
    _validate_mining_tool(mining)
    if not _has_mining_source(cfg):
        raise ValueError(
            "perception_mining requires chunk_manifest, scene_list, route_scene_list, or scene_pool"
        )
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
    if args.scene_list is None and args.chunk_manifest is None:
        raise ValueError("one of --scene_list or --chunk_manifest is required")
    contract = {
        "model_path": str(args.model_path),
        "scene_list": str(args.scene_list) if args.scene_list else None,
        "chunk_manifest": str(args.chunk_manifest) if args.chunk_manifest else None,
        "workflow_config": str(args.workflow_config),
        "training_config": str(args.training_config),
        "output_dir": str(args.output_dir),
    }
    return _config_from_workflow_contract(contract)


def _uses_torch_distributed_run(cmd: list[str]) -> bool:
    if any(Path(part).name == "torchrun" for part in cmd):
        return True
    return any(
        part == "-m" and idx + 1 < len(cmd) and cmd[idx + 1] == "torch.distributed.run"
        for idx, part in enumerate(cmd)
    )


def _cleanup_torch_dist_file_store(cmd: list[str]) -> None:
    if not _uses_torch_distributed_run(cmd):
        return
    try:
        _TORCH_DDP_FILE_STORE.unlink()
    except FileNotFoundError:
        pass


def _run(
    cmd: list[str], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_torch_dist_file_store(cmd)
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


def _env_for_gpu(gpu_id: int | None) -> dict[str, str]:
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath_entries = [str(repo_root), str(repo_root / "diffusion_planner")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _run_parallel(
    jobs: list[tuple[str, list[str], Path, dict[str, str] | None]],
    *,
    cwd: Path | None = None,
) -> float:
    if not jobs:
        return 0.0
    t0 = time.perf_counter()
    running = []
    logs = []
    for label, cmd, log_path, env in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w")
        logs.append(log)
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env,
        )
        running.append((label, cmd, log_path, proc))
        print(f"  started {label}: {' '.join(cmd)}; log: {log_path}")
    failures = []
    try:
        for label, cmd, log_path, proc in running:
            rc = proc.wait()
            if rc != 0:
                failures.append((label, rc, cmd, log_path))
    finally:
        for log in logs:
            log.close()
    elapsed = time.perf_counter() - t0
    if failures:
        details = "; ".join(
            f"{label} failed ({rc}), see {log_path}: {' '.join(cmd)}"
            for label, rc, cmd, log_path in failures
        )
        raise RuntimeError(f"Parallel stage failed after {elapsed:.1f}s: {details}")
    print(f"  parallel stage completed in {elapsed:.1f}s")
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _split_jsonl_round_robin(rows: list[dict[str, Any]], paths: list[Path]) -> None:
    shards = [[] for _ in paths]
    for idx, row in enumerate(rows):
        shards[idx % len(paths)].append(row)
    for path, shard_rows in zip(paths, shards, strict=True):
        _write_jsonl(path, shard_rows)


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _require_unique_paths(paths: list[Path], context: str) -> None:
    seen: dict[Path, Path] = {}
    duplicates = []
    for path in paths:
        canonical = _canonical_path(path)
        if canonical in seen:
            duplicates.append(f"{seen[canonical]} and {path}")
        seen[canonical] = path
    if duplicates:
        raise ValueError(f"{context} has duplicate paths: {duplicates}")


def _require_disjoint_paths(left: list[Path], right: list[Path], context: str) -> None:
    right_set = {_canonical_path(path) for path in right}
    overlap = [str(path) for path in left if _canonical_path(path) in right_set]
    if overlap:
        raise ValueError(f"{context} paths overlap with merged outputs: {overlap}")


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
    cfg: dict[str, Any],
    model_path: Path,
    rdir: Path,
    *,
    out_dir: Path | None = None,
    out_jsonl: Path | None = None,
    segments_jsonl: Path | None = None,
    summary_json: Path | None = None,
    mining_overrides: dict[str, Any] | None = None,
) -> tuple[list[str], Path]:
    mining = dict(cfg.get("perception_mining") or {})
    if mining_overrides:
        mining.update(mining_overrides)
    hits_jsonl = segments_jsonl or rdir / "perception_reproducer_hits.jsonl"
    danger_save_dir = out_dir or rdir / "perception_danger_windows"
    _validate_mining_tool(mining)
    chunk_manifest = mining.get("chunk_manifest")
    scene_list = _first_non_null(
        mining.get("scene_list"),
        cfg.get("route_scene_list"),
        cfg.get("scene_pool"),
    )
    if scene_list is None and chunk_manifest is None:
        raise ValueError(
            "perception_mining requires chunk_manifest, scene_list, route_scene_list, or scene_pool"
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
        str(out_jsonl or rdir / "credit_windows.jsonl"),
        "--segments_jsonl",
        str(hits_jsonl),
        "--summary_json",
        str(summary_json or rdir / "perception_direct_summary.json"),
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
        if chunk_manifest is not None and key == "max_scenes":
            continue
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


def _materialize_chunk_manifest_for_shards(
    cfg: dict[str, Any],
    rdir: Path,
) -> dict[str, Any]:
    mining = dict(cfg.get("perception_mining") or {})
    if mining.get("chunk_manifest") is not None:
        return cfg
    scene_list = _first_non_null(
        mining.get("scene_list"),
        cfg.get("route_scene_list"),
        cfg.get("scene_pool"),
    )
    if scene_list is None:
        return cfg

    manifest = rdir / "planned_chunks.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.mine_direct_reproducer_chunks",
        "--scene_list",
        str(scene_list),
        "--segments_jsonl",
        str(manifest),
        "--plan_only",
        "--chunk_len",
        str(mining.get("chunk_len", 80)),
        "--start_stride",
        str(mining.get("start_stride", mining.get("chunk_len", 80))),
    ]
    for key in ("expected_frame_step", "min_chunk_len", "max_scenes"):
        if key in mining and mining[key] is not None:
            cmd.extend([f"--{key}", str(mining[key])])
    _run(cmd, rdir / "plan_chunks.log")

    updated = dict(cfg)
    mining["chunk_manifest"] = str(manifest)
    updated["perception_mining"] = mining
    return updated


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


def _repair_cmd(
    cfg: dict[str, Any],
    model_path: Path,
    credit_jsonl: Path,
    rdir: Path,
    *,
    out_dir: Path | None = None,
    out_list: Path | None = None,
    out_rows_jsonl: Path | None = None,
    repair_overrides: dict[str, Any] | None = None,
    allow_empty: bool = False,
) -> list[str]:
    repair_cfg = dict(cfg.get("repair_config") or {})
    if repair_overrides:
        repair_cfg.update(repair_overrides)
    missing = [k for k in ("ego_shape", "min_margin") if k not in repair_cfg]
    if missing:
        raise ValueError(f"repair_config is missing required fields: {missing}")
    repaired_dir = out_dir or rdir / "repaired_targets"
    repaired_list = out_list or rdir / "repaired_targets.json"
    repaired_rows = out_rows_jsonl or rdir / "repaired_targets.jsonl"
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
    if allow_empty:
        cmd.append("--allow_empty")
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
    gpu_ids = _gpu_ids_from_config(cfg)
    default_nproc = len(gpu_ids) if gpu_ids else 1
    nproc = int(train_cfg.get("nproc_per_node", overrides.pop("nproc_per_node", default_nproc)))
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
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
    return cmd, save_dir / "latest.pth", repo_root / "diffusion_planner", env


def _source_scene_list(cfg: dict[str, Any]) -> Path | None:
    mining = dict(cfg.get("perception_mining") or {})
    source = _first_non_null(
        mining.get("scene_list"),
        cfg.get("route_scene_list"),
        cfg.get("scene_pool"),
    )
    if source is None:
        return None
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"scene list does not exist: {path}")
    cfg["scene_pool"] = str(path)
    cfg["route_scene_list"] = str(path)
    mining["scene_list"] = str(path)
    cfg["perception_mining"] = mining
    return path


def _merge_jsonl_files(inputs: list[Path], output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(_read_jsonl(path))
    _write_jsonl(output, rows)
    return rows


def _merge_json_lists(inputs: list[Path], output: Path) -> list[str]:
    merged: list[str] = []
    for path in inputs:
        if path.exists():
            merged.extend(str(p) for p in _read_json_list(path))
    _write_json(output, merged)
    return merged


def _merge_unrepaired_lists(inputs: list[Path], output: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in inputs:
        if path.exists():
            merged.extend(dict(row) for row in _read_json_list(path))
    if merged:
        _write_json(output, merged)
    return merged


def _merge_mining_summaries(inputs: list[Path], output: Path) -> dict[str, Any]:
    summaries = [_load_json(path) for path in inputs if path.exists()]
    aggregate = {
        "planned_chunks": sum(int(s.get("planned_chunks", 0)) for s in summaries),
        "simulated_chunks": sum(int(s.get("simulated_chunks", 0)) for s in summaries),
        "skipped_chunks": sum(int(s.get("skipped_chunks", 0)) for s in summaries),
        "credit_rows": sum(int(s.get("credit_rows", 0)) for s in summaries),
        "elapsed_sec": max((float(s.get("elapsed_sec", 0.0)) for s in summaries), default=0.0),
        "shards": summaries,
    }
    _write_json(output, aggregate)
    return aggregate


def _run_mining_phase(
    cfg: dict[str, Any],
    model_path: Path,
    rdir: Path,
    gpu_ids: list[int],
) -> float:
    credit_jsonl = rdir / "credit_windows.jsonl"
    segments_jsonl = rdir / "perception_reproducer_hits.jsonl"
    summary_json = rdir / "perception_direct_summary.json"
    if len(gpu_ids) <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        cmd, _ = _perception_mining_cmd(
            cfg,
            model_path,
            rdir,
            mining_overrides={"device": "cuda"} if gpu_id is not None else None,
        )
        elapsed = _run(cmd, rdir / "perception_mine.log", env=_env_for_gpu(gpu_id))
        rows = _read_jsonl(credit_jsonl)
        _write_json(rdir / "credit_windows_paths.json", [row["scene_path"] for row in rows])
        return elapsed

    cfg = _materialize_chunk_manifest_for_shards(cfg, rdir)
    shard_root = rdir / "perception_mine_shards"
    jobs = []
    credit_parts = []
    segment_parts = []
    summary_parts = []
    window_dirs = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        credit_part = shard_dir / "credit_windows.jsonl"
        segment_part = shard_dir / "segments.jsonl"
        summary_part = shard_dir / "summary.json"
        window_dir = shard_dir / "windows"
        credit_parts.append(credit_part)
        segment_parts.append(segment_part)
        summary_parts.append(summary_part)
        window_dirs.append(window_dir)
        cmd, _ = _perception_mining_cmd(
            cfg,
            model_path,
            rdir,
            out_dir=window_dir,
            out_jsonl=credit_part,
            segments_jsonl=segment_part,
            summary_json=summary_part,
            mining_overrides={
                "num_shards": len(gpu_ids),
                "shard_index": shard_index,
                "device": "cuda",
            },
        )
        jobs.append(
            (
                f"perception_mine[{shard_index}]",
                cmd,
                shard_dir / "perception_mine.log",
                _env_for_gpu(gpu_id),
            )
        )
    _require_unique_paths(
        [*credit_parts, *segment_parts, *summary_parts, *window_dirs, *[job[2] for job in jobs]],
        "perception mining shards",
    )
    _require_disjoint_paths(
        [*credit_parts, *segment_parts, *summary_parts],
        [credit_jsonl, segments_jsonl, summary_json],
        "perception mining shard outputs",
    )
    elapsed = _run_parallel(jobs)
    rows = _merge_jsonl_files(credit_parts, credit_jsonl)
    _merge_jsonl_files(segment_parts, segments_jsonl)
    _merge_mining_summaries(summary_parts, summary_json)
    _write_json(rdir / "credit_windows_paths.json", [row["scene_path"] for row in rows])
    return elapsed


def _run_repair_phase(
    cfg: dict[str, Any],
    model_path: Path,
    rdir: Path,
    gpu_ids: list[int],
) -> float:
    credit_jsonl = rdir / "credit_windows.jsonl"
    rows = _read_jsonl(credit_jsonl)
    if not rows:
        raise RuntimeError(f"{credit_jsonl} is empty; no mined scenes to repair")
    if len(gpu_ids) <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        overrides = {"device": "cuda"} if gpu_id is not None else None
        cmd = _repair_cmd(
            cfg,
            model_path,
            credit_jsonl,
            rdir,
            repair_overrides=overrides,
        )
        return _run(cmd, rdir / "repair.log", env=_env_for_gpu(gpu_id))

    shard_root = rdir / "repair_shards"
    shard_inputs = [
        shard_root / f"shard_{idx:02d}" / "credit_windows.jsonl" for idx in range(len(gpu_ids))
    ]
    _split_jsonl_round_robin(rows, shard_inputs)
    jobs = []
    repaired_lists = []
    repaired_rows_jsonls = []
    unrepaired_lists = []
    repaired_dirs = []
    for shard_index, (gpu_id, shard_input) in enumerate(zip(gpu_ids, shard_inputs, strict=True)):
        shard_dir = shard_input.parent
        out_list = shard_dir / "repaired_targets.json"
        out_rows = shard_dir / "repaired_targets.jsonl"
        out_dir = shard_dir / "repaired_targets"
        repaired_lists.append(out_list)
        repaired_rows_jsonls.append(out_rows)
        unrepaired_lists.append(shard_dir / "repaired_targets_unrepaired.json")
        repaired_dirs.append(out_dir)
        cmd = _repair_cmd(
            cfg,
            model_path,
            shard_input,
            rdir,
            out_dir=out_dir,
            out_list=out_list,
            out_rows_jsonl=out_rows,
            repair_overrides={"device": "cuda"},
            allow_empty=True,
        )
        jobs.append(
            (
                f"repair[{shard_index}]",
                cmd,
                shard_dir / "repair.log",
                _env_for_gpu(gpu_id),
            )
        )
    _require_unique_paths(
        [
            *shard_inputs,
            *repaired_lists,
            *repaired_rows_jsonls,
            *unrepaired_lists,
            *repaired_dirs,
            *[job[2] for job in jobs],
        ],
        "repair shards",
    )
    _require_disjoint_paths(
        [*shard_inputs, *repaired_lists, *repaired_rows_jsonls, *unrepaired_lists],
        [
            credit_jsonl,
            rdir / "repaired_targets.json",
            rdir / "repaired_targets.jsonl",
            rdir / "repaired_targets_unrepaired.json",
        ],
        "repair shard outputs",
    )
    elapsed = _run_parallel(jobs)
    repaired_paths = _merge_json_lists(repaired_lists, rdir / "repaired_targets.json")
    _merge_jsonl_files(repaired_rows_jsonls, rdir / "repaired_targets.jsonl")
    _merge_unrepaired_lists(unrepaired_lists, rdir / "repaired_targets_unrepaired.json")
    if not repaired_paths:
        raise RuntimeError("No repaired targets were produced across repair shards")
    return elapsed


def _print_dry_run_plan(
    cfg: dict[str, Any],
    model_path: Path,
    rdir: Path,
    gpu_ids: list[int],
    memory_cmd: list[str],
    round_idx: int,
) -> None:
    if len(gpu_ids) <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        mining_cmd, _ = _perception_mining_cmd(
            cfg,
            model_path,
            rdir,
            mining_overrides={"device": "cuda"} if gpu_id is not None else None,
        )
        repair_cmd = _repair_cmd(
            cfg,
            model_path,
            rdir / "credit_windows.jsonl",
            rdir,
            repair_overrides={"device": "cuda"} if gpu_id is not None else None,
        )
        prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        print(f"[round {round_idx}] perception_mine: {prefix}{' '.join(mining_cmd)}")
        print(f"[round {round_idx}] repair: {prefix}{' '.join(repair_cmd)}")
    else:
        for shard_index, gpu_id in enumerate(gpu_ids):
            shard_dir = rdir / "perception_mine_shards" / f"shard_{shard_index:02d}"
            mining_cmd, _ = _perception_mining_cmd(
                cfg,
                model_path,
                rdir,
                out_dir=shard_dir / "windows",
                out_jsonl=shard_dir / "credit_windows.jsonl",
                segments_jsonl=shard_dir / "segments.jsonl",
                summary_json=shard_dir / "summary.json",
                mining_overrides={
                    "num_shards": len(gpu_ids),
                    "shard_index": shard_index,
                    "device": "cuda",
                },
            )
            print(
                f"[round {round_idx}] perception_mine[{shard_index}]: "
                f"CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(mining_cmd)}"
            )
        for shard_index, gpu_id in enumerate(gpu_ids):
            shard_dir = rdir / "repair_shards" / f"shard_{shard_index:02d}"
            repair_cmd = _repair_cmd(
                cfg,
                model_path,
                shard_dir / "credit_windows.jsonl",
                rdir,
                out_dir=shard_dir / "repaired_targets",
                out_list=shard_dir / "repaired_targets.json",
                out_rows_jsonl=shard_dir / "repaired_targets.jsonl",
                repair_overrides={"device": "cuda"},
                allow_empty=True,
            )
            print(
                f"[round {round_idx}] repair[{shard_index}]: "
                f"CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(repair_cmd)}"
            )
    print(f"[round {round_idx}] memory: {' '.join(memory_cmd)}")


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
    scene_list: Path | None,
    train_input_list: Path,
    phase_times: dict[str, float],
    next_model_path: Path,
) -> None:
    mining_summary = _load_json(rdir / "perception_direct_summary.json")
    credit_rows = _read_jsonl(rdir / "credit_windows.jsonl")
    repaired_rows = _read_jsonl(rdir / "repaired_targets.jsonl")
    memory = _load_json(rdir / f"round_{round_idx}_memory.json")
    unrepaired_path = rdir / "repaired_targets_unrepaired.json"
    unrepaired_rows = _read_json_list(unrepaired_path) if unrepaired_path.exists() else []
    route_scene_count = len(_read_json_list(scene_list)) if scene_list is not None else None
    train_scene_count = len(_read_json_list(train_input_list))

    event_counts = _derive_event_counts_from_credit_rows(credit_rows)

    summary = {
        "round_idx": int(round_idx),
        "route_scene_count": route_scene_count,
        "planned_chunks": int(mining_summary.get("planned_chunks", 0)),
        "simulated_chunks": int(mining_summary.get("simulated_chunks", 0)),
        "skipped_chunks": int(mining_summary.get("skipped_chunks", 0)),
        "mined_event_count_by_label": event_counts,
        "simulated_event_count": int(sum(event_counts.values())),
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
    parser.add_argument("--chunk_manifest", type=Path)
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
        credit_jsonl = rdir / "credit_windows.jsonl"
        repaired_rows_jsonl = rdir / "repaired_targets.jsonl"
        repaired_list_json = rdir / "repaired_targets.json"
        memory_json = rdir / f"round_{round_idx}_memory.json"
        replay_json = rdir / f"round_{round_idx}_replay_scenes.json"
        train_input_list = rdir / "train_scenes.json"
        name = f"r2lpl_round_{round_idx:03d}"
        scene_list = _source_scene_list(cfg)
        gpu_ids = _gpu_ids_from_config(cfg)

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

        if args.dry_run:
            _print_dry_run_plan(cfg, model_path, rdir, gpu_ids, memory_cmd, round_idx)
            continue

        print(f"[round {round_idx}] perception_mine")
        phase_times["perception_mine"] = _run_mining_phase(cfg, model_path, rdir, gpu_ids)
        print(f"[round {round_idx}] repair")
        phase_times["repair"] = _run_repair_phase(cfg, model_path, rdir, gpu_ids)
        print(f"[round {round_idx}] memory: {' '.join(memory_cmd)}")
        phase_times["memory"] = _run(memory_cmd, rdir / "memory.log")

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
            scene_list=scene_list,
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
