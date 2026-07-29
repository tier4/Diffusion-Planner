#!/usr/bin/env python3
"""Run configurable R2LPL-style lifelong replay rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusion_planner.utils.dist_init import dist_init_file_path

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
_DEFAULT_ENABLED_LABELS = ["road_border_crossing", "moving_collision", "expert_disagreement"]
_DEFAULT_REPAIR_LABELS = [
    "road_border_crossing",
    "static_collision",
    "moving_collision",
    "expert_disagreement",
]
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
_TORCH_DDP_FILE_STORE = dist_init_file_path()
_REPAIR_REFRESH_SCOPES = {"unrepaired", "all"}


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


def _required_replay_capacity(replay: dict[str, Any]) -> int:
    """Replay capacity must be sized per campaign — there is no sane default.

    Capacity caps the CARRY-OVER memory between rounds (this round's repaired
    scenes always train regardless); a too-small value silently evicts earlier
    rounds' corrections. Rule of thumb: >= 2-3x the expected repaired scenes
    per round — hundreds for local corpora, thousands+ for server-scale.
    """
    capacity = replay.get("capacity")
    if capacity is None:
        raise ValueError(
            "replay_memory.capacity is required (no default): size it to the campaign "
            "(>= 2-3x expected repaired scenes per round)"
        )
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError(f"replay_memory.capacity must be >= 1, got {capacity}")
    return capacity


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


def _apply_repair_refresh_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize + loudly validate the intra-training re-repair (repair_refresh) knobs.

    ``repair_refresh_every_epochs`` (int, default 0 => OFF). When ``0 < every <
    epochs_per_round`` the base_sft path trains in chunks of ``every`` epochs,
    re-running repair on the mined credit windows with the current checkpoint
    between chunks. Only the base_sft backend supports the inner loop.
    """
    cfg.setdefault("repair_refresh_every_epochs", 0)
    cfg.setdefault("repair_refresh_scope", "unrepaired")
    cfg.setdefault("repair_refresh_max_cycles", 0)
    every = int(cfg["repair_refresh_every_epochs"])
    max_cycles = int(cfg["repair_refresh_max_cycles"])
    scope = str(cfg["repair_refresh_scope"])
    epochs_per_round = int(cfg["epochs_per_round"])
    if every < 0:
        raise ValueError(f"repair_refresh_every_epochs must be >= 0, got {every}")
    if max_cycles < 0:
        raise ValueError(f"repair_refresh_max_cycles must be >= 0, got {max_cycles}")
    if scope not in _REPAIR_REFRESH_SCOPES:
        raise ValueError(
            f"repair_refresh_scope must be one of {sorted(_REPAIR_REFRESH_SCOPES)}, got {scope!r}"
        )
    if every > epochs_per_round:
        raise ValueError(
            f"repair_refresh_every_epochs ({every}) must be <= epochs_per_round "
            f"({epochs_per_round})"
        )
    if every > 0:
        backend = str(cfg.get("training_backend", "base_sft"))
        if backend != "base_sft":
            raise ValueError(
                "repair_refresh_every_epochs requires training_backend='base_sft'; "
                f"the intra-training re-repair loop is not supported for backend {backend!r}"
            )
    cfg["repair_refresh_every_epochs"] = every
    cfg["repair_refresh_max_cycles"] = max_cycles
    cfg["repair_refresh_scope"] = scope
    return cfg


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
    refresh = dict(rounds.get("repair_refresh") or workflow.get("repair_refresh") or {})
    training_section = dict(workflow.get("training") or {})
    if "anchor" in training_section and not training_section["anchor"]:
        raise ValueError(
            "workflow training.anchor is present but empty; remove the section or "
            "fill in scene_list/ratio"
        )
    if "guards" in workflow and not workflow["guards"]:
        raise ValueError("workflow guards is present but empty; remove the section or fill it in")

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
    enabled_labels = [str(label) for label in enabled_labels]
    repair_labels = judgement.get("repair_labels")
    if repair_labels is None:
        repair_labels = [label for label in enabled_labels if label in _DEFAULT_REPAIR_LABELS]
    if not isinstance(repair_labels, list) or not repair_labels:
        raise ValueError(
            "judgement.repair_labels must be a non-empty list, or enabled_labels must include "
            f"at least one repairable label from {_DEFAULT_REPAIR_LABELS}"
        )
    repair_labels = [str(label) for label in repair_labels]
    non_mined_repair = sorted(set(repair_labels) - set(enabled_labels))
    if non_mined_repair:
        raise ValueError(f"judgement.repair_labels are not enabled for mining: {non_mined_repair}")

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
        "generation_mode": str(_first_non_null(repair.get("generation_mode"), "grpo_temperature")),
        "grpo_noise_scale": float(_first_non_null(repair.get("grpo_noise_scale"), 3.0)),
        "variant": _first_non_null(
            repair.get("variant"),
            repair.get("generation_variant"),
            "rl_cl_col_sweep",
        ),
        "gt_max_speed": float(_first_non_null(repair.get("gt_max_speed"), 9.0)),
        "expert_stop_anchor": str(_first_non_null(repair.get("expert_stop_anchor"), "recorded")),
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
        "use_route_cl_guidance": bool(repair.get("use_route_cl_guidance", True)),
    }
    if repair.get("prototypes_path"):
        repair_cfg["prototypes_path"] = str(repair["prototypes_path"])
    if repair.get("enable_depart_morph"):
        repair_cfg["enable_depart_morph"] = bool(repair["enable_depart_morph"])
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
        "training_normal_scene_list": (
            str(training_section["normal_scene_list"])
            if training_section.get("normal_scene_list")
            else None
        ),
        "reward_config": str(reward_config),
        "threshold_config": str(threshold_config),
        "credit_window_config": str(credit_window_config),
        "replay_memory": {
            "capacity": _required_replay_capacity(replay),
            "alpha": float(_first_non_null(replay.get("alpha"), 0.5)),
            "beta": float(_first_non_null(replay.get("beta"), 0.5)),
            "arc_bin_m": float(_first_non_null(replay.get("arc_bin_m"), 25.0)),
            "label_quotas": replay.get("label_quotas"),
        },
        # Set only when actually configured — an ever-present empty dict would
        # make "anchor key present but empty" (a misconfiguration the validator
        # rejects) indistinguishable from "no anchor".
        **({"anchor": dict(training_section["anchor"])} if training_section.get("anchor") else {}),
        # Same present-but-empty rule as anchor (rejected above), so a set
        # guards key here is always non-empty. "_"-prefixed comment keys are
        # stripped so the runtime config carries only real knobs.
        **(
            {"guards": _strip_comment_keys(dict(workflow["guards"]))}
            if workflow.get("guards")
            else {}
        ),
        "training_config": str(training_source)
        if isinstance(training_source, (str, os.PathLike))
        else training_source,
        "training_backend": training_backend,
        "output_dir": str(output_dir),
        "repair_config": repair_cfg,
        "mine_labels": enabled_labels,
        "repair_labels": repair_labels,
        "enable_conflict_detector": bool(
            _first_non_null(
                judgement.get("enable_conflict_detector"),
                judgement.get("conflict_detector_enabled"),
                "expert_disagreement" in enabled_labels,
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
        "validate_on_repaired_targets": bool(workflow.get("validate_on_repaired_targets", False)),
        "count_rear_end_collisions": _workflow_count_rear_end_collisions(judgement),
        "realized_reward": bool(workflow.get("realized_reward", False)),
        "final_round_mining": bool(
            workflow.get("final_round_mining", workflow.get("realized_reward", False))
        ),
        "perception_mining": perception_mining,
        "repair_refresh_every_epochs": int(
            _first_non_null(
                refresh.get("every_epochs"),
                refresh.get("repair_refresh_every_epochs"),
                rounds.get("repair_refresh_every_epochs"),
                0,
            )
        ),
        "repair_refresh_scope": str(
            _first_non_null(
                refresh.get("scope"),
                rounds.get("repair_refresh_scope"),
                "unrepaired",
            )
        ),
        "repair_refresh_max_cycles": int(
            _first_non_null(
                refresh.get("max_cycles"),
                rounds.get("repair_refresh_max_cycles"),
                0,
            )
        ),
    }
    if gpu_ids:
        cfg["gpu_ids"] = gpu_ids
    if scene_list is not None:
        cfg["scene_pool"] = str(scene_list)
        cfg["route_scene_list"] = str(scene_list)

    return _apply_repair_refresh_config(cfg)


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
    _required_replay_capacity(dict(cfg.get("replay_memory") or {}))
    mining = dict(cfg.get("perception_mining") or {})
    _validate_mining_tool(mining)
    if not _has_mining_source(cfg):
        raise ValueError(
            "perception_mining requires chunk_manifest, scene_list, route_scene_list, or scene_pool"
        )
    return _apply_repair_refresh_config(cfg)


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


def _event_group_key(row: dict[str, Any]) -> str:
    return str(row.get("window_dir") or row.get("event_key") or row.get("scene_path"))


def _split_jsonl_by_event(rows: list[dict[str, Any]], paths: list[Path]) -> None:
    shards = [[] for _ in paths]
    group_to_shard: dict[str, int] = {}
    for row in rows:
        key = _event_group_key(row)
        shard_idx = group_to_shard.get(key)
        if shard_idx is None:
            shard_idx = len(group_to_shard) % len(paths)
            group_to_shard[key] = shard_idx
        shards[shard_idx].append(row)
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
    if bool(cfg.get("count_rear_end_collisions", False)):
        # Keep realized-event mining rear-end-consistent with the repair side.
        cmd.append("--count_rear_end_collisions")
    if bool(cfg.get("realized_reward", False)) or bool(mining.get("realized_reward", False)):
        # Compute the realized closed-loop reward on the driven trajectory in the
        # same rollout (no extra sim, no disk save/reload) and write it to the
        # mining summary. The reward reflects the checkpoint THIS mine ran with.
        cmd.append("--realized_reward")
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


def _anchor_slice_paths(
    anchor_cfg: dict[str, Any] | None, n_focus: int, round_idx: int
) -> list[str]:
    """Real logged normal scenes unioned into a round's train list.

    Training only on repaired+replay scenes leaves the regularizer with no reach
    outside the failure distribution — competence drifts on everything the
    rounds never see. The anchor slice counters this with real logged scenes at
    ``ratio`` : 1 (anchor : focus), optionally stratified so ``waits_fraction``
    of the anchors come from a waits/interaction list (scenes where the logged
    ego stops — the slice most eroded by pro-motion post-training).

    Config (``training.anchor`` in the workflow JSON): ``scene_list`` +
    ``ratio`` are required when the section is present (fail loudly, no silent
    defaults); ``waits_scene_list``/``waits_fraction`` must be given together;
    ``seed`` (default 0) is offset by the round index so each round redraws
    reproducibly.
    """
    if not anchor_cfg:
        return []
    missing = [key for key in ("scene_list", "ratio") if not anchor_cfg.get(key)]
    if missing:
        raise ValueError(f"training.anchor is missing required fields: {missing}")
    ratio = float(anchor_cfg["ratio"])
    if ratio <= 0.0:
        raise ValueError(f"training.anchor.ratio must be > 0: {ratio}")
    waits_list = anchor_cfg.get("waits_scene_list")
    waits_fraction = anchor_cfg.get("waits_fraction")
    if (waits_list is None) != (waits_fraction is None):
        raise ValueError("training.anchor.waits_scene_list and waits_fraction must be set together")

    import math as _math
    import random as _random

    n_anchor = int(_math.ceil(ratio * n_focus))
    if n_anchor < 1:
        return []
    rng = _random.Random(int(anchor_cfg.get("seed", 0)) + round_idx)

    def _sample(pool: list[str], n: int) -> list[str]:
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    normal_pool = [str(p) for p in _read_json_list(Path(anchor_cfg["scene_list"]))]
    if not normal_pool:
        raise ValueError(f"training.anchor.scene_list is empty: {anchor_cfg['scene_list']}")
    picked: list[str] = []
    if waits_list is not None:
        frac = float(waits_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"training.anchor.waits_fraction must be in (0, 1): {frac}")
        waits_pool = [str(p) for p in _read_json_list(Path(waits_list))]
        if not waits_pool:
            raise ValueError(f"training.anchor.waits_scene_list is empty: {waits_list}")
        n_waits = int(round(frac * n_anchor))
        picked.extend(_sample(waits_pool, n_waits))
    picked_set = set(picked)
    picked.extend(_sample([p for p in normal_pool if p not in picked_set], n_anchor - len(picked)))
    return picked


def _ensure_4col_neighbor_futures(paths: list[str], out_dir: Path) -> list[str]:
    """Homogenize ``neighbor_agents_future`` to the 4-col training schema.

    Repaired/replay scenes carry 4-col ``[x, y, cos, sin]`` neighbor futures
    while raw logged anchor scenes carry 3-col ``[x, y, heading]``. Each format
    trains fine alone, but torch's default collate cannot stack a mixed batch
    (``[320, 80, 3]`` vs ``[320, 80, 4]``). Convert the 3-col scenes to 4-col
    copies under ``out_dir`` with the canonical converter (zero padding rows
    stay zero, so the trainer's validity mask is preserved); 4-col scenes pass
    through by original path.
    """
    from planner_metrics.scene_format import future_to_4col as _future_to_4col

    out: list[str] = []
    for path in paths:
        src = Path(path)
        with np.load(src) as loaded:
            future = loaded["neighbor_agents_future"]
            if future.shape[-1] == 4:
                out.append(str(src))
                continue
            if future.shape[-1] != 3:
                raise ValueError(
                    f"{src}: neighbor_agents_future has {future.shape[-1]} channels; "
                    "expected 3 [x, y, heading] or 4 [x, y, cos, sin]"
                )
            data = dict(loaded)
        data["neighbor_agents_future"] = _future_to_4col(future)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Anchor pools may mix source dirs with colliding basenames — prefix
        # with a stable hash of the source path.
        digest = hashlib.sha1(str(src).encode()).hexdigest()[:10]
        dst = out_dir / f"{digest}_{src.name}"
        np.savez_compressed(dst, **data)
        out.append(str(dst))
    return out


def _validate_normal_scene_list_config(cfg: dict[str, Any]) -> None:
    """Fail at STARTUP on normal_scene_list misconfiguration, not after mining.

    training.normal_scene_list is only wired into the ranked-SFT backend (the
    base_sft counterpart is training.anchor), and the prob/normal split it
    enables requires explicit n_prob_scenes / n_normal_scenes in the training
    config — run_experiment raises otherwise, but only at the train phase,
    after the expensive mine + repair phases. n_prob_scenes also caps how many
    repaired scenes train, so it must be sized >= the expected repaired count
    per round or repairs are silently subsampled.
    """
    normal_list = cfg.get("training_normal_scene_list")
    if not normal_list:
        return
    backend = str(cfg.get("training_backend", "base_sft"))
    if backend == "base_sft":
        raise ValueError(
            "training.normal_scene_list is only wired into the ranked-SFT backend; "
            "for base_sft use training.anchor instead"
        )
    if not Path(normal_list).exists():
        raise ValueError(f"training.normal_scene_list does not exist: {normal_list}")
    try:
        entries = json.loads(Path(normal_list).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"training.normal_scene_list is not valid JSON: {normal_list}") from exc
    if not isinstance(entries, list) or not entries or not all(isinstance(e, str) for e in entries):
        raise ValueError(
            f"training.normal_scene_list must be a non-empty JSON list of scene "
            f"paths: {normal_list}"
        )
    missing_scenes = [e for e in entries if not Path(e).exists()]
    if missing_scenes:
        preview = ", ".join(missing_scenes[:3])
        raise ValueError(
            f"training.normal_scene_list references {len(missing_scenes)}/{len(entries)} "
            f"missing scene files (first: {preview}); a broken entry would otherwise "
            "only fail in the 4-col conversion after the mine+repair phases"
        )
    payload = _training_config_payload(cfg["training_config"])
    missing = [k for k in ("n_prob_scenes", "n_normal_scenes") if k not in payload]
    if missing:
        raise ValueError(
            "training.normal_scene_list enables the prob/normal split, which requires "
            f"explicit {missing} in the training config (n_prob_scenes must be >= the "
            "expected repaired scenes per round — smaller values silently subsample "
            "the repairs)"
        )
    n_prob = payload["n_prob_scenes"]
    n_normal = payload["n_normal_scenes"]
    if not isinstance(n_prob, int) or n_prob < 1:
        raise ValueError(f"n_prob_scenes must be a positive integer, got {n_prob!r}")
    if not isinstance(n_normal, int) or n_normal < 0:
        raise ValueError(f"n_normal_scenes must be a non-negative integer, got {n_normal!r}")


def _rsft_scene_args(
    cfg: dict[str, Any],
    *,
    repaired_paths: list[str],
    repaired_list_json: Path,
    rdir: Path,
    round_idx: int,
) -> list[str]:
    """Scene args for the ranked-SFT training call of one round.

    With training.normal_scene_list the round trains a prob/normal split
    (repaired = prob, real normals = normal) so curated rounds can mix real
    scenes; without it, the legacy combined-list path is kept. The normal
    scenes are homogenized to 4-col neighbor futures (raw logged scenes may
    carry 3-col ones, and a mixed batch cannot collate — same incompatibility
    as the base_sft anchor slice) into a per-round resolved list.
    """
    normal_list = cfg.get("training_normal_scene_list")
    if not normal_list:
        return ["--train_scenes", str(repaired_list_json)]
    n_prob = _training_config_payload(cfg["training_config"])["n_prob_scenes"]
    if n_prob < len(repaired_paths):
        raise ValueError(
            f"[round {round_idx}] n_prob_scenes={n_prob} < {len(repaired_paths)} "
            "repaired scenes: run_experiment would silently subsample the "
            "repairs (min(n_prob, len(prob)) sampling). Raise n_prob_scenes to "
            "cover the largest expected repaired count per round."
        )
    normal_paths = _ensure_4col_neighbor_futures(
        [str(p) for p in _read_json_list(Path(normal_list))],
        rdir / "normal_scenes_4col",
    )
    resolved_normals = rdir / "normal_scenes_resolved.json"
    resolved_normals.write_text(json.dumps(normal_paths, indent=2))
    return [
        "--prob_scenes",
        str(repaired_list_json),
        "--normal_scenes",
        str(resolved_normals),
    ]


def _validate_anchor_config(cfg: dict[str, Any]) -> None:
    """Fail at STARTUP on anchor misconfiguration, not hours into round 1.

    Checks the required fields, the paired waits fields, that the referenced
    scene lists exist and are non-empty, and that the anchor slice is actually
    consumable: only the base_sft backend unions the anchor into training —
    with any other backend the slice would be silently dropped, which this
    rejects loudly instead.
    """
    anchor_cfg = cfg.get("anchor")
    if "anchor" in cfg and not anchor_cfg:
        raise ValueError(
            "config has an empty 'anchor' section; remove it or fill in scene_list/ratio"
        )
    if not anchor_cfg:
        # Legacy direct configs may carry the template's nested form; a nested
        # training.anchor that is NOT lifted would be silently ignored.
        training_section = cfg.get("training")
        if isinstance(training_section, dict) and training_section.get("anchor"):
            raise ValueError(
                "config has training.anchor but the runner consumes the top-level "
                "'anchor' key on this config path; move the section to the top level"
            )
        return
    if str(cfg.get("training_backend", "base_sft")) != "base_sft":
        raise ValueError(
            "training.anchor is only wired into the base_sft training backend; "
            "remove the anchor section or use backend 'base_sft'"
        )
    missing = [key for key in ("scene_list", "ratio") if not anchor_cfg.get(key)]
    if missing:
        raise ValueError(f"training.anchor is missing required fields: {missing}")
    if float(anchor_cfg["ratio"]) <= 0.0:
        raise ValueError(f"training.anchor.ratio must be > 0: {anchor_cfg['ratio']}")
    if (anchor_cfg.get("waits_scene_list") is None) != (anchor_cfg.get("waits_fraction") is None):
        raise ValueError("training.anchor.waits_scene_list and waits_fraction must be set together")
    waits_fraction = anchor_cfg.get("waits_fraction")
    if waits_fraction is not None and not 0.0 < float(waits_fraction) < 1.0:
        raise ValueError(f"training.anchor.waits_fraction must be in (0, 1): {waits_fraction}")
    for key in ("scene_list", "waits_scene_list"):
        value = anchor_cfg.get(key)
        if value is None:
            continue
        path = Path(value)
        if not path.exists():
            raise ValueError(f"training.anchor.{key} does not exist: {path}")
        if not _read_json_list(path):
            raise ValueError(f"training.anchor.{key} is empty: {path}")


_GUARD_KEYS = {
    "frozen_chunk_manifest",
    "patience_benchmark",
    "patience_stop_speed",
    "closed_loop_npz_root",
    "closed_loop",
}
_GUARD_CLOSED_LOOP_KEYS = {
    "seg_len",
    "replan_interval",
    "draw_every",
    "near_miss_thresh",
    "search_radius",
    "warmup_steps",
    "unstick_after",
    "unstick_advance_m",
    "fps",
}


def _validate_repair_generation_config(cfg: dict[str, Any]) -> None:
    """Fail at STARTUP on repair-generation misconfiguration, not mid-round.

    Runs for BOTH config paths (workflow contract and direct config). Checks
    are slot-derived, not name-derived: any variant containing anchor-mode
    slots needs a prototype library, and K must cover det + every fixed slot —
    otherwise the failure surfaces only at repair time, after a full mining
    phase.
    """
    from rlvr.generation_variants import get_variant

    repair_cfg = dict(cfg.get("repair_config") or {})
    # None / empty mean "not set" — don't turn null into the literal "None".
    variant_value = repair_cfg.get("variant")
    if variant_value is None or not str(variant_value).strip():
        return
    variant_name = str(variant_value)
    variant = get_variant(variant_name)  # unknown variant names raise here
    if any("anchor" in slot for slot in variant.cl_spd_configs) and not repair_cfg.get(
        "prototypes_path"
    ):
        raise ValueError(
            f"repair variant {variant_name!r} has anchor-mode slots and requires "
            "prototypes_path (build with rlvr.autoresearch.tools.build_path_prototypes)"
        )
    min_k = 1 + len(variant.cl_spd_configs) + len(variant.noise_configs)
    k = int(repair_cfg.get("K", 8))
    if str(repair_cfg.get("generation_mode", "")) == "guided_variant" and k < min_k:
        raise ValueError(
            f"repair variant {variant_name!r} needs K >= {min_k} "
            f"(det + {min_k - 1} fixed slots), got K={k}"
        )


def _validate_guards_config(cfg: dict[str, Any]) -> None:
    """Fail at STARTUP on guard misconfiguration, not after a full round.

    ``guards`` drives the post-round guard phase — REPORT-ONLY evaluation:
    frozen-chunk event-rate mining (the cheap closed-loop metric), the
    open-loop patience onset eval, and the optional closed-loop probe. Guards
    never take checkpoint decisions; select checkpoints from the per-round
    tables (see the README's selection recipe).
    """
    guards = cfg.get("guards")
    policy = str(cfg.get("checkpoint_policy", "latest"))
    if policy == "composite":
        raise ValueError(
            "the 'composite' checkpoint selection rule "
            "(rounds.checkpoint_selection_rule in workflow configs, checkpoint_policy "
            "in direct configs) was removed: automatic reject/rollback never recovered "
            "a round (retries just re-roll the same distribution). Guards are "
            "report-only — use 'latest' and select checkpoints from the per-round "
            "guard tables / selection_report.json"
        )
    if "guards" in cfg and not guards:
        raise ValueError("config has an empty 'guards' section; remove it or fill it in")
    if guards is None:
        return
    # "_"-prefixed keys are JSON comments (template convention) — ignored here
    # and stripped from the runtime config at contract-parse time.
    unknown = sorted(k for k in set(guards) - _GUARD_KEYS if not k.startswith("_"))
    if unknown:
        raise ValueError(f"guards has unknown keys {unknown}; allowed: {sorted(_GUARD_KEYS)}")
    for key in ("frozen_chunk_manifest", "closed_loop_npz_root"):
        value = guards.get(key)
        if value is not None and not Path(value).exists():
            raise ValueError(f"guards.{key} does not exist: {value}")
    manifest = guards.get("frozen_chunk_manifest")
    if manifest is not None and not _read_jsonl(Path(manifest)):
        # An empty frozen set yields all-zero event counts — a metric that
        # can only mislead. Reject it up front.
        raise ValueError(f"guards.frozen_chunk_manifest is empty: {manifest}")
    benchmark = guards.get("patience_benchmark")
    if benchmark is not None:
        path = Path(benchmark)
        if not path.exists():
            raise ValueError(f"guards.patience_benchmark does not exist: {path}")
        if not _read_json_list(path):
            raise ValueError(f"guards.patience_benchmark is empty: {path}")
    stop_speed = guards.get("patience_stop_speed")
    if stop_speed is not None and float(stop_speed) <= 0.0:
        raise ValueError(f"guards.patience_stop_speed must be > 0: {stop_speed}")
    closed_loop = guards.get("closed_loop")
    if closed_loop is not None:
        if guards.get("closed_loop_npz_root") is None:
            raise ValueError("guards.closed_loop knobs are set without closed_loop_npz_root")
        unknown_cl = sorted(
            k for k in set(closed_loop) - _GUARD_CLOSED_LOOP_KEYS if not k.startswith("_")
        )
        if unknown_cl:
            raise ValueError(
                f"guards.closed_loop has unknown keys {unknown_cl}; "
                f"allowed: {sorted(_GUARD_CLOSED_LOOP_KEYS)}"
            )


def _strip_comment_keys(section: dict[str, Any]) -> dict[str, Any]:
    """Drop "_"-prefixed JSON-comment keys (template convention), recursively."""
    return {
        k: _strip_comment_keys(v) if isinstance(v, dict) else v
        for k, v in section.items()
        if not k.startswith("_")
    }


def _guard_mining_cmd(
    cfg: dict[str, Any], model_path: Path, gdir: Path, *, gpu_id: int | None
) -> list[str]:
    """Mining command for the FROZEN held-out chunk set (guard event rates).

    The frozen set must be mined in full every round so event counts are
    comparable — campaign-side subsampling/sharding knobs are neutralized.
    """
    overrides: dict[str, Any] = {
        "chunk_manifest": str(cfg["guards"]["frozen_chunk_manifest"]),
        "num_shards": None,
        "shard_index": None,
        "sample_fraction": None,
        "sample_seed": None,
        "max_chunks": None,
        # A stale windows dir from an aborted guard run must fail loudly, even
        # when the campaign mining config allows reusing its own out dirs.
        "allow_existing_out_dir": False,
    }
    if gpu_id is not None:
        overrides["device"] = "cuda"
    cmd, _ = _perception_mining_cmd(
        cfg,
        model_path,
        gdir,
        out_dir=gdir / "windows",
        out_jsonl=gdir / "credit_windows.jsonl",
        segments_jsonl=gdir / "segments.jsonl",
        summary_json=gdir / "summary.json",
        mining_overrides=overrides,
    )
    return cmd


def _patience_onset_cmd(cfg: dict[str, Any], model_path: Path, gdir: Path) -> list[str]:
    guards = cfg["guards"]
    return [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.eval_patience_onset",
        "--model",
        str(model_path),
        "--scenes",
        str(guards["patience_benchmark"]),
        "--out",
        str(gdir / "patience_onset.json"),
        "--stop_speed",
        str(guards.get("patience_stop_speed", 0.5)),
    ]


def _closed_loop_probe_cmd(cfg: dict[str, Any], model_path: Path) -> list[str]:
    guards = cfg["guards"]
    # The probe subprocess runs with cwd=diffusion_planner/, so relative paths
    # that validated against the orchestrator's cwd must be resolved here.
    cmd = [
        sys.executable,
        "-m",
        "valid_predictor_closed_loop",
        "--model_path",
        str(Path(model_path).resolve()),
        "--npz_root",
        str(Path(guards["closed_loop_npz_root"]).resolve()),
    ]
    # Direct configs may carry template-style "_comment*" keys (the workflow
    # path strips them at parse time) — never forward those as CLI flags.
    for key, value in sorted(_strip_comment_keys(dict(guards.get("closed_loop") or {})).items()):
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _expert_disagreement_events_by_reason(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Per-branch event counts for the conflict detector's three directions.

    The aggregate expert_disagreement count hides which way the model fails:
    ``expert_wait_model_forward`` = fail-to-stop, ``model_lagging_expert`` =
    fail-to-take-off, ``model_ahead_expert`` = over-eager progress. Repair
    training can move failures between branches (observed: a campaign flipped
    from stop-dominant to lagging-dominant), so the split must be visible in
    the per-round tables.
    """
    events_by_reason: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if str(row.get("label")) != "expert_disagreement":
            continue
        reason = str(row.get("expert_disagreement_reason") or "unknown")
        # Same event keying as _derive_event_counts_from_credit_rows, so the
        # per-reason counts sum to the label's aggregate count.
        event_key = str(
            row.get("event_key") or row.get("window_dir") or Path(str(row["scene_path"])).parent
        )
        events_by_reason[reason].add(event_key)
    return {reason: len(events) for reason, events in sorted(events_by_reason.items())}


def _guard_mining_metrics(gdir: Path) -> dict[str, Any]:
    rows = _read_jsonl(gdir / "credit_windows.jsonl")
    summary = _load_json(gdir / "summary.json")
    event_counts = _derive_event_counts_from_credit_rows(rows)
    simulated = int(summary.get("simulated_chunks", 0))
    if simulated <= 0:
        # Zero simulated chunks means zero events for every label — a metric
        # that can only mislead. The frozen set is broken (moved NPZs, bad
        # manifest), not clean.
        raise RuntimeError(
            f"guard mining simulated 0 chunks (see {gdir / 'summary.json'}); "
            "the frozen chunk set is not usable as a metric"
        )
    total_events = int(sum(event_counts.values()))
    metrics: dict[str, Any] = {
        "event_count_by_label": event_counts,
        "total_events": total_events,
        "simulated_chunks": simulated,
        "events_per_1000_chunks": round(1000.0 * total_events / simulated, 3),
    }
    by_reason = _expert_disagreement_events_by_reason(rows)
    if by_reason:
        metrics["expert_disagreement_by_reason"] = by_reason
    return metrics


def _selection_report(
    reference: dict[str, Any], candidates: list[tuple[int, dict[str, Any]]]
) -> dict[str, Any]:
    """ADVISORY checkpoint recommendation from the per-round guard tables.

    Applies the README selection recipe mechanically: a round is vetoed on any
    per-label frozen-event increase vs the reference row, any
    ``fail_to_stop`` / ``fail_to_resume`` increase, or an incomparable frozen
    denominator; eligible rounds are ranked by total frozen events (then
    ``fail_to_resume``, then ``fail_to_stop``). This REPORTS a recommendation —
    it never feeds a checkpoint back into training, and the standard val L2
    leg of the recipe still has to be run on the finalists by hand.
    """
    ref_mining = reference.get("frozen_chunk_mining") or {}
    ref_events = dict(ref_mining.get("event_count_by_label") or {})
    ref_onset = reference.get("patience_onset") or {}
    rows: list[dict[str, Any]] = []
    for round_idx, metrics in candidates:
        row: dict[str, Any] = {
            "round_idx": round_idx,
            "checkpoint": metrics.get("model_path"),
            "veto_reasons": [],
        }
        mining = metrics.get("frozen_chunk_mining")
        onset = metrics.get("patience_onset")
        if mining and ref_mining:
            if int(mining.get("simulated_chunks", -1)) != int(
                ref_mining.get("simulated_chunks", -1)
            ):
                row["veto_reasons"].append("frozen denominator differs from reference")
            events = dict(mining.get("event_count_by_label") or {})
            for label in sorted(set(events) | set(ref_events)):
                cur, ref = int(events.get(label, 0)), int(ref_events.get(label, 0))
                if cur > ref:
                    row["veto_reasons"].append(f"{label} events {ref} -> {cur}")
            row["total_frozen_events"] = int(mining.get("total_events", 0))
        else:
            row["veto_reasons"].append("no frozen-chunk metrics")
        if onset and ref_onset:
            for key in ("fail_to_stop", "fail_to_resume"):
                cur, ref = int(onset.get(key, 0)), int(ref_onset.get(key, 0))
                if cur > ref:
                    row["veto_reasons"].append(f"{key} {ref} -> {cur}")
            row["fail_to_stop"] = int(onset.get("fail_to_stop", 0))
            row["fail_to_resume"] = int(onset.get("fail_to_resume", 0))
        row["eligible"] = not row["veto_reasons"]
        rows.append(row)
    eligible = [r for r in rows if r["eligible"] and "total_frozen_events" in r]
    eligible.sort(
        key=lambda r: (
            r["total_frozen_events"],
            r.get("fail_to_resume", 0),
            r.get("fail_to_stop", 0),
            r["round_idx"],
        )
    )
    recommended = eligible[0] if eligible else None
    return {
        "note": (
            "Advisory only — apply the standard val L2 eval to the finalists before "
            "shipping; the runner does not compute L2."
        ),
        "reference_checkpoint": reference.get("model_path"),
        "rounds": rows,
        "recommended": (
            {"round_idx": recommended["round_idx"], "checkpoint": recommended["checkpoint"]}
            if recommended
            else None
        ),
    }


def _run_guard_phase(
    cfg: dict[str, Any],
    model_path: Path,
    gdir: Path,
    gpu_ids: list[int],
    *,
    tag: str,
) -> dict[str, Any]:
    """Report-only guard evaluation of ``model_path``; returns the metrics dict.

    Guards never take checkpoint decisions — they produce the per-round tables
    (frozen-chunk event rates, patience onset, optional closed-loop probe)
    that checkpoint selection reads.
    """
    guards = cfg["guards"]
    gdir.mkdir(parents=True, exist_ok=True)
    gpu_id = gpu_ids[0] if gpu_ids else None
    # The closed-loop probe runs with a different cwd and writes next to the
    # checkpoint — a relative checkpoint path would break both.
    model_path = Path(model_path).resolve()
    metrics: dict[str, Any] = {"tag": tag, "model_path": str(model_path)}

    if guards.get("frozen_chunk_manifest"):
        cmd = _guard_mining_cmd(cfg, model_path, gdir, gpu_id=gpu_id)
        _run(cmd, gdir / "guard_mine.log", env=_env_for_gpu(gpu_id))
        metrics["frozen_chunk_mining"] = _guard_mining_metrics(gdir)

    if guards.get("patience_benchmark"):
        cmd = _patience_onset_cmd(cfg, model_path, gdir)
        _run(cmd, gdir / "patience_onset.log", env=_env_for_gpu(gpu_id))
        report = _load_json(gdir / "patience_onset.json")
        onset = dict(report["summary"]["model"])
        onset["n_scenes"] = report["summary"]["n_scenes"]
        metrics["patience_onset"] = onset

    if guards.get("closed_loop_npz_root"):
        metrics["closed_loop"] = _run_closed_loop_probe(cfg, model_path, gdir, gpu_id)

    _write_json(gdir / "guard_metrics.json", metrics)
    return metrics


def _run_closed_loop_probe(
    cfg: dict[str, Any], model_path: Path, gdir: Path, gpu_id: int | None
) -> dict[str, Any]:
    """Run ``valid_predictor_closed_loop`` on the guard NPZ root (report-only).

    The tool writes to ``<ckpt dir>/closed_loop/<timestamp>/`` and offers no
    --out_dir (diffusion_planner/ is not editable), so the new run dir is
    detected by diffing and its summary copied into the guard dir. It also
    requires args.json adjacent to the checkpoint; when the trainer did not
    leave one there, the nearest one is copied in.
    """
    import shutil

    adjacent_args = model_path.parent / "args.json"
    if not adjacent_args.exists():
        shutil.copy2(_args_json_for_model(model_path), adjacent_args)
    cl_root = model_path.parent / "closed_loop"
    before = set(cl_root.glob("*")) if cl_root.exists() else set()
    repo_root = Path(__file__).resolve().parents[3]
    _run(
        _closed_loop_probe_cmd(cfg, model_path),
        gdir / "closed_loop_probe.log",
        cwd=repo_root / "diffusion_planner",
        env=_env_for_gpu(gpu_id),
    )
    new_dirs = sorted(set(cl_root.glob("*")) - before, key=lambda p: p.name)
    if not new_dirs:
        raise RuntimeError(f"closed-loop probe produced no new run dir under {cl_root}")
    summary_path = new_dirs[-1] / "summary.json"
    summary = _load_json(summary_path)
    shutil.copy2(summary_path, gdir / "closed_loop_summary.json")
    keep = (
        "n_segments",
        "object",
        "road_border",
        "red_light_violation",
        "strong_brake",
        "reproducer",
    )
    picked = {k: summary[k] for k in keep if k in summary}
    picked["run_dir"] = str(new_dirs[-1])
    return picked


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
        "rlvr.autoresearch.tools.build_repaired_targets",
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
        "--generation_mode",
        str(repair_cfg.get("generation_mode", "grpo_temperature")),
        "--grpo_noise_scale",
        str(repair_cfg.get("grpo_noise_scale", 3.0)),
        "--variant",
        str(repair_cfg.get("variant", "rl_cl_col_sweep")),
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
    # Expert-disagreement repair knobs (default on; no config change needed to enable).
    if not bool(repair_cfg.get("repair_expert_gt_candidate", True)):
        cmd.append("--no_repair_expert_gt_candidate")
    if not bool(repair_cfg.get("repair_expert_reference", True)):
        cmd.append("--no_repair_expert_reference")
    if "expert_morph_w_max" in repair_cfg:
        cmd.extend(["--expert_morph_w_max", str(repair_cfg["expert_morph_w_max"])])
    if "expert_morph_max_accel" in repair_cfg:
        cmd.extend(["--expert_morph_max_accel", str(repair_cfg["expert_morph_max_accel"])])
    if "expert_morph_max_jerk" in repair_cfg:
        cmd.extend(["--expert_morph_max_jerk", str(repair_cfg["expert_morph_max_jerk"])])
    if "expert_stop_anchor" in repair_cfg:
        cmd.extend(["--expert_stop_anchor", str(repair_cfg["expert_stop_anchor"])])
    if bool(repair_cfg.get("enable_depart_morph", False)):
        cmd.append("--enable_depart_morph")
    if repair_cfg.get("prototypes_path"):
        cmd.extend(["--prototypes_path", str(repair_cfg["prototypes_path"])])
    if cfg.get("repair_labels"):
        cmd.extend(["--labels", ",".join(cfg["repair_labels"])])
    if bool(cfg.get("enable_conflict_detector", False)):
        cmd.append("--enable_conflict_detector")
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


def _effective_replay_knobs(cfg: dict[str, Any]) -> tuple[float, float]:
    """Resolve (replay_der_coef, replay_loss_weight), workflow keys winning.

    The knobs may be set at the workflow top level (``_WORKFLOW_KEYS``) or inside
    the training config; the workflow-level value takes precedence.
    """
    payload = _training_config_payload(cfg["training_config"])
    train_args = payload.get("train_args", payload)
    der = float(cfg.get("replay_der_coef", train_args.get("replay_der_coef", 0.0)))
    weight = float(cfg.get("replay_loss_weight", train_args.get("replay_loss_weight", 1.0)))
    return der, weight


def _assert_base_sft_supports_replay_knobs(cfg: dict[str, Any]) -> None:
    """Fail loudly if base_sft is asked for DER / replay-loss weighting.

    The base SFT trainer (``train_predictor``) trains on the flat repaired+replay
    union with no role metadata, so it cannot honor these knobs. Silently
    ignoring them would leave the user believing forgetting-protection is active
    when it is not.
    """
    der, weight = _effective_replay_knobs(cfg)
    if der > 0.0 or weight != 1.0:
        raise ValueError(
            "training_backend='base_sft' cannot honor replay_der_coef / "
            f"replay_loss_weight (got replay_der_coef={der}, replay_loss_weight={weight}). "
            "The base SFT trainer trains on the flat repaired+replay union with no role "
            "weighting or DER. Use the ranked-SFT backend for DER / replay-loss weighting, "
            "or remove these knobs."
        )


def _round_training_config(cfg: dict[str, Any], *, rdir: Path, anchor_model_path: Path) -> Path:
    """Write this round's ranked-SFT training config.

    Injects the workflow-level replay knobs and — when DER is active — sets the
    replay anchor to THIS round's warm-start model (the prior round's
    checkpoint), per the R2LPL spec: DER anchors replay outputs to the
    prior-round model. Without this the anchor would stay frozen at round 1.
    """
    payload = _training_config_payload(cfg["training_config"])
    out = dict(payload)
    if "replay_loss_weight" in cfg:
        out["replay_loss_weight"] = cfg["replay_loss_weight"]
    if "replay_der_coef" in cfg:
        out["replay_der_coef"] = cfg["replay_der_coef"]
    if float(out.get("replay_der_coef", 0.0)) > 0.0:
        out["replay_anchor_model_path"] = str(anchor_model_path)
    round_cfg_path = rdir / "training_config.json"
    _write_json(round_cfg_path, out)
    return round_cfg_path


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


def _repair_refresh_chunk_sizes(
    epochs_per_round: int, refresh_every: int, max_cycles: int
) -> list[int]:
    """Split ``epochs_per_round`` into training chunks for the re-repair inner loop.

    Returns per-chunk epoch counts. A re-repair fires between each consecutive
    pair of chunks (i.e. ``len(sizes) - 1`` refreshes). ``max_cycles`` (>0) caps
    the number of refreshes: after that many refreshes the remaining epochs are
    trained in one final chunk with no further re-repair. When the feature is off
    (``refresh_every <= 0`` or ``>= epochs_per_round``) a single full chunk is
    returned, preserving the legacy single-invocation behavior.
    """
    if refresh_every <= 0 or refresh_every >= epochs_per_round:
        return [epochs_per_round]
    sizes: list[int] = []
    remaining = epochs_per_round
    refreshes_done = 0
    while remaining > 0:
        if max_cycles and refreshes_done >= max_cycles:
            sizes.append(remaining)
            break
        chunk = min(refresh_every, remaining)
        sizes.append(chunk)
        remaining -= chunk
        if remaining <= 0:
            break
        refreshes_done += 1
    return sizes


def _repair_refresh_chunk_targets(
    base_epoch: int, epochs_per_round: int, refresh_every: int, max_cycles: int
) -> list[int]:
    """Absolute ``--train_epochs`` target for each chunk given the warm-start epoch."""
    targets: list[int] = []
    acc = base_epoch
    for chunk in _repair_refresh_chunk_sizes(epochs_per_round, refresh_every, max_cycles):
        acc += chunk
        targets.append(acc)
    return targets


def _base_train_invocation(
    cfg: dict[str, Any],
    *,
    model_path: Path,
    train_list: Path,
    rdir: Path,
    round_idx: int,
    epochs_this_chunk: int | None = None,
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
    epochs_to_add = (
        int(cfg["epochs_per_round"]) if epochs_this_chunk is None else int(epochs_this_chunk)
    )
    total_epochs = _checkpoint_epoch(model_path) + epochs_to_add
    gpu_ids = _gpu_ids_from_config(cfg)
    default_nproc = len(gpu_ids) if gpu_ids else 1
    nproc = int(train_cfg.get("nproc_per_node", overrides.pop("nproc_per_node", default_nproc)))
    master_port = str(train_cfg.get("master_port", overrides.pop("master_port", 29531 + round_idx)))
    train_scene_count = len(_read_json_list(train_list))
    if train_scene_count < 1:
        raise ValueError(f"{train_list} is empty; refusing to launch base training")

    # Device-aware launch: on CPU (device=cpu, e.g. a CPU-only host or --device cpu) a
    # torch.distributed.run / DDP launch drives NCCL + torch.cuda.set_device and dies
    # with "No CUDA GPUs are available". Launch train_predictor directly with ddp=false
    # there; keep the DDP launcher on CUDA.
    device = str(overrides.get("device", base_args.get("device", "cuda"))).lower()
    cpu = device == "cpu"
    launcher = (
        [sys.executable, "-m", "train_predictor"]
        if cpu
        else [
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
        ]
    )
    cmd = [
        *launcher,
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
        "ema_decay",
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
    merged["ddp"] = not cpu  # single-process on CPU; DDP on CUDA
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
    # Realized closed-loop reward: pose-count-weighted mean across shards (each shard
    # scored a disjoint set of poses). Without this the sharded/multi-GPU summary would
    # drop the field the single-GPU path emits.
    rr_poses = sum(int(s.get("realized_cl_reward_poses", 0)) for s in summaries)
    if rr_poses > 0:
        rr_sum = sum(
            float(s.get("realized_cl_reward", 0.0)) * int(s.get("realized_cl_reward_poses", 0))
            for s in summaries
            if s.get("realized_cl_reward") is not None
        )
        aggregate["realized_cl_reward"] = rr_sum / rr_poses
        aggregate["realized_cl_reward_poses"] = rr_poses
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
    _split_jsonl_by_event(rows, shard_inputs)
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


def _credit_row_keys(credit_jsonl: Path) -> set[str]:
    """Union of keys seen across the mined credit-window rows.

    Used to strip repair-added keys (``reason``, repair meta) off unrepaired rows
    so a refresh re-attempt feeds ``build_repaired_targets`` clean credit rows.
    """
    keys: set[str] = set()
    for row in _read_jsonl(credit_jsonl):
        keys.update(row.keys())
    return keys


def _refresh_repair(
    cfg: dict[str, Any],
    model_path: Path,
    rdir: Path,
    *,
    scope: str,
    cycle: int,
    gpu_ids: list[int],
) -> tuple[list[str], float]:
    """Re-run repair (K-gen + retrieval, NO re-sim) on mined credit windows.

    Uses the CURRENT checkpoint ``model_path`` to harvest scenes that became
    rule-feasible as the policy improved. Returns (newly-repaired scene NPZ paths,
    repair wall-time seconds). Also folds the harvested repairs into the round's
    canonical ``repaired_targets.json`` / ``.jsonl`` and rewrites
    ``repaired_targets_unrepaired.json`` to the still-unrepaired set so the next
    cycle only re-attempts what remains.
    """
    refresh_dir = rdir / f"refresh_{cycle}"
    refresh_dir.mkdir(parents=True, exist_ok=True)
    credit_jsonl = rdir / "credit_windows.jsonl"
    if scope == "all":
        input_jsonl = credit_jsonl
    elif scope == "unrepaired":
        input_jsonl = refresh_dir / "credit_windows.jsonl"
        unrepaired_path = rdir / "repaired_targets_unrepaired.json"
        unrepaired_rows = _read_json_list(unrepaired_path) if unrepaired_path.exists() else []
        if not unrepaired_rows:
            _write_jsonl(input_jsonl, [])
            return [], 0.0
        keep_keys = _credit_row_keys(credit_jsonl)
        stripped = [{k: v for k, v in row.items() if k in keep_keys} for row in unrepaired_rows]
        _write_jsonl(input_jsonl, stripped)
    else:
        raise ValueError(f"unknown repair_refresh_scope: {scope!r}")

    gpu_id = gpu_ids[0] if gpu_ids else None
    out_list = refresh_dir / "repaired_targets.json"
    cmd = _repair_cmd(
        cfg,
        model_path,
        input_jsonl,
        rdir,
        out_dir=refresh_dir / "repaired_targets",
        out_list=out_list,
        out_rows_jsonl=refresh_dir / "repaired_targets.jsonl",
        repair_overrides={"device": "cuda"} if gpu_id is not None else None,
        allow_empty=True,
    )
    refresh_elapsed = _run(cmd, refresh_dir / "repair.log", env=_env_for_gpu(gpu_id))

    newly = [str(p) for p in _read_json_list(out_list)] if out_list.exists() else []

    # scope="all" re-attempts EVERY credit window, so it re-repairs sources the
    # initial repair already covered. Those land at new refresh output paths, which
    # _union_scene_lists (dedup by output path) can't see as the same source -> the
    # source would train twice with two different targets. Keep only refresh
    # repairs whose SOURCE scene is not already in the canonical set.
    refresh_rows_jsonl = refresh_dir / "repaired_targets.jsonl"
    canonical_rows_jsonl = rdir / "repaired_targets.jsonl"
    if scope == "all" and newly:
        canon_sources = {
            row.get("source_scene_path")
            for row in (_read_jsonl(canonical_rows_jsonl) if canonical_rows_jsonl.exists() else [])
            if row.get("source_scene_path")
        }
        kept = [
            row
            for row in _read_jsonl(refresh_rows_jsonl)
            if row.get("source_scene_path") not in canon_sources
        ]
        _write_jsonl(refresh_rows_jsonl, kept)
        newly = [str(row["scene_path"]) for row in kept if row.get("scene_path")]

    # Fold harvested repairs into the round's canonical artifacts.
    if newly:
        canonical_list = rdir / "repaired_targets.json"
        existing = (
            [str(p) for p in _read_json_list(canonical_list)] if canonical_list.exists() else []
        )
        _union_scene_lists(existing, newly, canonical_list)
        _merge_jsonl_files([canonical_rows_jsonl, refresh_rows_jsonl], canonical_rows_jsonl)

    # The refresh's still-unrepaired output IS the new full unrepaired set: for
    # scope="unrepaired" we re-attempted exactly the prior unrepaired rows, for
    # scope="all" we re-attempted every credit window. Rewrite it (empty when the
    # refresh cleared everything) so the next cycle re-attempts only what remains.
    refresh_unrepaired = out_list.with_name(out_list.stem + "_unrepaired.json")
    new_unrepaired = _read_json_list(refresh_unrepaired) if refresh_unrepaired.exists() else []
    _write_json(rdir / "repaired_targets_unrepaired.json", new_unrepaired)
    return newly, refresh_elapsed


def _run_base_sft_training(
    cfg: dict[str, Any],
    *,
    model_path: Path,
    train_input_list: Path,
    rdir: Path,
    round_idx: int,
    gpu_ids: list[int],
) -> tuple[float, Path]:
    """Run the round's base_sft training, optionally with intra-training re-repair.

    When ``repair_refresh_every_epochs`` is off (0 or >= epochs_per_round) this is
    the legacy single ``_base_train_invocation`` + ``_run``. When enabled, training
    proceeds in chunks; between chunks a re-repair harvests newly-recoverable
    scenes (no re-sim) which are merged into the train set before the next chunk.
    Returns (total_train_wall_time_sec, produced_checkpoint_path).
    """
    epochs_per_round = int(cfg["epochs_per_round"])
    refresh_every = int(cfg.get("repair_refresh_every_epochs", 0))
    max_cycles = int(cfg.get("repair_refresh_max_cycles", 0))
    scope = str(cfg.get("repair_refresh_scope", "unrepaired"))
    chunk_sizes = _repair_refresh_chunk_sizes(epochs_per_round, refresh_every, max_cycles)

    if len(chunk_sizes) == 1:
        train_cmd, next_model_path, cwd, env = _base_train_invocation(
            cfg,
            model_path=model_path,
            train_list=train_input_list,
            rdir=rdir,
            round_idx=round_idx,
        )
        print(f"[round {round_idx}] train: {' '.join(train_cmd)}")
        elapsed = _run(train_cmd, rdir / "train.log", cwd=cwd, env=env)
        if not next_model_path.exists():
            raise FileNotFoundError(
                f"base trainer did not create expected checkpoint: {next_model_path}"
            )
        return elapsed, next_model_path

    total_elapsed = 0.0
    current_model = model_path
    last_cycle = len(chunk_sizes) - 1
    chunk_targets = _repair_refresh_chunk_targets(
        _checkpoint_epoch(model_path), epochs_per_round, refresh_every, max_cycles
    )
    for cycle, chunk in enumerate(chunk_sizes):
        train_cmd, next_model_path, cwd, env = _base_train_invocation(
            cfg,
            model_path=current_model,
            train_list=train_input_list,
            rdir=rdir,
            round_idx=round_idx,
            epochs_this_chunk=chunk,
        )
        print(
            f"[round {round_idx}] train chunk {cycle} "
            f"({chunk} epochs -> target epoch {chunk_targets[cycle]}): {' '.join(train_cmd)}"
        )
        total_elapsed += _run(train_cmd, rdir / f"train_chunk{cycle}.log", cwd=cwd, env=env)
        if not next_model_path.exists():
            raise FileNotFoundError(
                f"base trainer did not create expected checkpoint: {next_model_path}"
            )
        current_model = next_model_path
        if cycle == last_cycle:
            break
        newly, refresh_elapsed = _refresh_repair(
            cfg, current_model, rdir, scope=scope, cycle=cycle, gpu_ids=gpu_ids
        )
        total_elapsed += refresh_elapsed
        if newly:
            existing = [str(p) for p in _read_json_list(train_input_list)]
            _union_scene_lists(existing, newly, train_input_list)
            print(
                f"[round {round_idx}] repair_refresh cycle {cycle}: "
                f"merged {len(newly)} newly-repaired scene(s) into train set"
            )
        else:
            print(f"[round {round_idx}] repair_refresh cycle {cycle}: no new repairs")
    return total_elapsed, current_model


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
    guards = cfg.get("guards")
    if guards:
        gdir = rdir / "guards"
        gpu_id = gpu_ids[0] if gpu_ids else None
        if guards.get("frozen_chunk_manifest"):
            cmd = _guard_mining_cmd(cfg, model_path, gdir, gpu_id=gpu_id)
            print(f"[round {round_idx}] guard_mine: {' '.join(cmd)}")
        if guards.get("patience_benchmark"):
            print(
                f"[round {round_idx}] guard_onset: {' '.join(_patience_onset_cmd(cfg, model_path, gdir))}"
            )
        if guards.get("closed_loop_npz_root"):
            print(
                f"[round {round_idx}] guard_closed_loop: "
                f"{' '.join(_closed_loop_probe_cmd(cfg, model_path))}"
            )
    anchor_cfg = cfg.get("anchor")
    if anchor_cfg:
        print(
            f"[round {round_idx}] anchor slice: ratio {anchor_cfg['ratio']}:1 from "
            f"{anchor_cfg['scene_list']}"
            + (
                f" (waits {anchor_cfg['waits_fraction']} from {anchor_cfg['waits_scene_list']})"
                if anchor_cfg.get("waits_scene_list")
                else ""
            )
        )


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
    guard_metrics: dict[str, Any] | None = None,
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

    # R2LPL hard-example signal: disagreement between the selected repaired target
    # and the original logged GT target (distinct from realized expert_disagreement).
    tgt_flags = [bool(r.get("target_gt_disagreement", False)) for r in repaired_rows]
    tgt_max_l2 = [
        float(r["target_gt_disagreement_max_l2"])
        for r in repaired_rows
        if r.get("target_gt_disagreement_max_l2") is not None
    ]
    tgt_mean_l2 = [
        float(r["target_gt_disagreement_mean_l2"])
        for r in repaired_rows
        if r.get("target_gt_disagreement_mean_l2") is not None
    ]

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
        "target_gt_disagreement_count": int(sum(tgt_flags)),
        "target_gt_disagreement_mean_l2": (
            round(sum(tgt_mean_l2) / len(tgt_mean_l2), 4) if tgt_mean_l2 else None
        ),
        "target_gt_disagreement_max_l2": (round(max(tgt_max_l2), 4) if tgt_max_l2 else None),
        "discarded_unrepaired_scene_count": len(unrepaired_rows),
        "replay_memory_size": len(memory.get("entries", [])),
        "final_training_scene_count": train_scene_count,
        "phase_wall_time_sec": {k: round(v, 3) for k, v in sorted(phase_times.items())},
        "phase_peak_memory_mb": {k: None for k in sorted(phase_times)},
        "next_model_path": str(next_model_path),
    }
    # Propagate the realized closed-loop reward (from this round's mining, i.e. the
    # incoming checkpoint) into the round summary so workflow_summary.json can compare
    # per-round checkpoints. None when --realized_reward was not enabled.
    if mining_summary.get("realized_cl_reward") is not None:
        summary["realized_cl_reward"] = mining_summary["realized_cl_reward"]
        summary["realized_cl_reward_poses"] = int(mining_summary.get("realized_cl_reward_poses", 0))
    if guard_metrics is not None:
        summary["guards"] = guard_metrics
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
    _validate_anchor_config(cfg)
    _validate_guards_config(cfg)
    _validate_repair_generation_config(cfg)
    _validate_normal_scene_list_config(cfg)

    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["model_path"])
    previous_memory: Path | None = None
    checkpoint_policy = str(cfg.get("checkpoint_policy", "latest"))
    guards_cfg = cfg.get("guards")
    reference_metrics: dict[str, Any] | None = None
    if guards_cfg:
        # Round-0 reference row of the per-round guard tables: the starting
        # model measured on the same frozen assets as every round checkpoint.
        gdir0 = out / "guard_round_000"
        gpu_ids0 = _gpu_ids_from_config(cfg)
        if args.dry_run:
            gpu_id0 = gpu_ids0[0] if gpu_ids0 else None
            # Every probe is individually optional — gate each print exactly
            # like _run_guard_phase gates the execution.
            if guards_cfg.get("frozen_chunk_manifest"):
                print(
                    "[round 0] guard_mine: "
                    f"{' '.join(_guard_mining_cmd(cfg, model_path, gdir0, gpu_id=gpu_id0))}"
                )
            if guards_cfg.get("patience_benchmark"):
                print(
                    "[round 0] guard_onset: "
                    f"{' '.join(_patience_onset_cmd(cfg, model_path, gdir0))}"
                )
            if guards_cfg.get("closed_loop_npz_root"):
                print(
                    "[round 0] guard_closed_loop: "
                    f"{' '.join(_closed_loop_probe_cmd(cfg, model_path))}"
                )
        else:
            print("[round 0] guards on the starting model (reference row)")
            reference_metrics = _run_guard_phase(cfg, model_path, gdir0, gpu_ids0, tag="round_000")
    guard_rows: list[tuple[int, dict[str, Any]]] = []
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
            str(_required_replay_capacity(mem_cfg)),
            "--alpha",
            str(mem_cfg.get("alpha", 0.5)),
            "--beta",
            str(mem_cfg.get("beta", 0.5)),
            "--arc_bin_m",
            str(mem_cfg.get("arc_bin_m", 25.0)),
        ]
        if mem_cfg.get("label_quotas"):
            quotas = mem_cfg["label_quotas"]
            if isinstance(quotas, dict):
                quotas = ",".join(f"{k}={v}" for k, v in sorted(quotas.items()))
            memory_cmd.extend(["--label_quotas", str(quotas)])
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
        anchor_paths = _anchor_slice_paths(
            cfg.get("anchor"), len(set(repaired_paths) | set(replay_paths)), round_idx
        )
        if anchor_paths:
            anchor_paths = _ensure_4col_neighbor_futures(anchor_paths, rdir / "anchor_scenes_4col")
            print(
                f"[round {round_idx}] anchor slice: {len(anchor_paths)} real scenes "
                f"({len(repaired_paths)} repaired + {len(replay_paths)} replay)"
            )
        _union_scene_lists(repaired_paths, [*replay_paths, *anchor_paths], train_input_list)
        if bool(cfg.get("validate_on_repaired_targets", False)):
            cfg["val_scenes"] = str(train_input_list)

        if training_backend == "base_sft":
            _assert_base_sft_supports_replay_knobs(cfg)
            phase_times["train"], model_path = _run_base_sft_training(
                cfg,
                model_path=model_path,
                train_input_list=train_input_list,
                rdir=rdir,
                round_idx=round_idx,
                gpu_ids=gpu_ids,
            )
        else:
            # H2: anchor DER replay to THIS round's warm-start model (the prior
            # round's checkpoint), not a frozen round-1 anchor.
            round_training_config = _round_training_config(
                cfg, rdir=rdir, anchor_model_path=model_path
            )
            scene_args = _rsft_scene_args(
                cfg,
                repaired_paths=repaired_paths,
                repaired_list_json=repaired_list_json,
                rdir=rdir,
                round_idx=round_idx,
            )
            train_cmd = [
                sys.executable,
                "-m",
                "rlvr.autoresearch.run_experiment",
                "--config",
                str(round_training_config),
                "--name",
                name,
                "--model_path",
                str(model_path),
                *scene_args,
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

        guard_metrics: dict[str, Any] | None = None
        if guards_cfg:
            print(f"[round {round_idx}] guards")
            t0 = time.perf_counter()
            guard_metrics = _run_guard_phase(
                cfg, model_path, rdir / "guards", gpu_ids, tag=f"round_{round_idx:03d}"
            )
            phase_times["guards"] = time.perf_counter() - t0
            guard_rows.append((round_idx, guard_metrics))

        previous_memory = memory_json
        _summarize_round(
            cfg=cfg,
            round_idx=round_idx,
            rdir=rdir,
            scene_list=scene_list,
            train_input_list=train_input_list,
            phase_times=phase_times,
            next_model_path=model_path,
            guard_metrics=guard_metrics,
        )
        workflow_summary.append(_load_json(rdir / "round_summary.json"))
        print(f"[round {round_idx}] next model: {model_path}")

    # Final-round mining: one closed-loop mine with the LAST round's checkpoint so
    # its residual problem-scene count AND realized closed-loop reward are recorded
    # (every earlier round's final model is already covered by the next round's mine;
    # the last round has no successor, so it needs this pass). Enabled whenever
    # realized-reward is on, or explicitly via final_round_mining.
    want_final_mine = bool(cfg.get("final_round_mining", cfg.get("realized_reward", False)))
    final_round = None
    if not args.dry_run and want_final_mine and not str(cfg.get("training_backend")) == "none":
        fdir = out / "final_round_mine"
        fdir.mkdir(parents=True, exist_ok=True)
        print(f"[final] mining residual problems + realized reward with {model_path}")
        _run_mining_phase(cfg, model_path, fdir, _gpu_ids_from_config(cfg))
        fsum = _load_json(fdir / "perception_direct_summary.json")
        # Represent the final-round mine in the operator-facing summary contract, not
        # just stdout: the last round's checkpoint has no successor mine, so this is the
        # only place its residual counts + realized reward appear.
        final_round = {
            "checkpoint": str(model_path),
            "residual_credit_rows": int(fsum.get("credit_rows", 0)),
            "residual_event_count_by_label": _derive_event_counts_from_credit_rows(
                _read_jsonl(fdir / "credit_windows.jsonl")
            ),
            "realized_cl_reward": fsum.get("realized_cl_reward"),
            "realized_cl_reward_poses": int(fsum.get("realized_cl_reward_poses", 0)),
        }
        print(
            f"[final] residual credit_rows={final_round['residual_credit_rows']} "
            f"realized_cl_reward={final_round['realized_cl_reward']}"
        )

    if not args.dry_run:
        workflow_out = {"rounds": workflow_summary}
        if final_round is not None:
            workflow_out["final_round_mine"] = final_round
        _write_json(out / "workflow_summary.json", workflow_out)
        if reference_metrics is not None and guard_rows:
            report = _selection_report(reference_metrics, guard_rows)
            _write_json(out / "selection_report.json", report)
            rec = report["recommended"]
            if rec is None:
                print(
                    "[selection] no round improved without a regression — "
                    "keep the starting model (see selection_report.json)"
                )
            else:
                print(
                    f"[selection] recommended checkpoint (advisory, L2 pending): "
                    f"round {rec['round_idx']} -> {rec['checkpoint']}"
                )


if __name__ == "__main__":
    main()
