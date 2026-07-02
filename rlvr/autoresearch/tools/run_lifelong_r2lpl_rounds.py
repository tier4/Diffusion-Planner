#!/usr/bin/env python3
"""Run configurable R2LPL-style lifelong replay rounds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REQUIRED = {
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


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = json.load(f)
    missing = sorted(_REQUIRED - set(cfg))
    if not cfg.get("perception_mining"):
        missing.extend(k for k in ("scene_pool", "scene_pool_root") if k not in cfg)
    if "repair_config" not in cfg:
        missing.append("repair_config")
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    out = Path(cfg["output_dir"]).resolve()
    if "auto_research" not in out.parts:
        raise ValueError(f"output_dir must be under an auto_research area, got {out}")
    return cfg


def _run(
    cmd: list[str], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
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


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


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
    # run_experiment prints best but does not persist a single pointer. Fall back
    # to latest trained adapter when LoRA is active.
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
    missing = [k for k in ("npz_root",) if k not in mining]
    if missing:
        raise ValueError(f"perception_mining is missing required fields: {missing}")
    hits_jsonl = rdir / "perception_reproducer_hits.jsonl"
    save_dir = rdir / "perception_reproducer_scenes"
    danger_save_dir = rdir / "perception_danger_windows"
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
            label = "perception_reproducer"
            for part in Path(scene).parts:
                if "_danger_" in part:
                    label = part.rsplit("_danger_", 1)[1]
                    break
            f.write(
                json.dumps(
                    {
                        "scene_path": scene,
                        "label": label,
                        "labels": [label],
                        "variant_kind": "perception_reproducer",
                    },
                    sort_keys=True,
                )
                + "\n"
            )


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
        str(repair_cfg.get("K", 16)),
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
    return cmd


def _base_training_cfg(path_or_dict: Any) -> dict[str, Any]:
    raw = (
        _load_json(Path(path_or_dict))
        if isinstance(path_or_dict, (str, os.PathLike))
        else dict(path_or_dict)
    )
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
    train_scene_count = len(json.loads(train_list.read_text()))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    cfg = _load_config(args.config)

    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["model_path"])
    previous_memory: Path | None = None
    checkpoint_policy = str(cfg.get("checkpoint_policy", "latest"))
    training_backend = str(cfg.get("training_backend", "rsft"))

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
        name = f"r2lpl_round_{round_idx:03d}"
        scene_pool = (
            Path(cfg["scene_pool"]) if "scene_pool" in cfg else rdir / "mined_scene_pool.json"
        )
        scene_pool_root = Path(cfg["scene_pool_root"]) if "scene_pool_root" in cfg else None
        if cfg.get("perception_mining") is not None:
            use_perception_as_credit = bool(
                (cfg.get("perception_mining") or {}).get("use_saved_scenes_as_credit_windows", True)
            )
        else:
            use_perception_as_credit = False
        mine_labels = ",".join(cfg.get("mine_labels", []))

        classify_cmd = [
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
            str(cfg.get("trajectory", "det")),
            "--model_path",
            str(model_path),
        ]
        if bool(cfg.get("enable_conflict_detector", False)):
            classify_cmd.append("--enable_conflict_detector")
        mine_cmd = None
        if not use_perception_as_credit:
            if scene_pool_root is None:
                raise ValueError(
                    "scene_pool_root is required when not using perception saved scenes as credit"
                )
            mine_cmd = [
                sys.executable,
                "-m",
                "rlvr.autoresearch.tools.mine_credit_window_scenes",
                "--classified_scenes_jsonl",
                str(classify_dir / "classified_scenes.jsonl"),
                "--credit_window_config",
                str(cfg["credit_window_config"]),
                "--route_npz_root",
                str(scene_pool_root),
                "--model_path",
                str(model_path),
                "--out_dir",
                str(credit_dir),
                "--out_jsonl",
                str(credit_jsonl),
            ]
            if mine_labels:
                mine_cmd.extend(["--labels", mine_labels])
            if bool(cfg.get("verify_reproduced_issue", False)):
                mine_cmd.extend(
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
                    mine_cmd.append("--enable_conflict_detector")
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
        ]
        if previous_memory is not None:
            memory_cmd.extend(["--previous_memory", str(previous_memory)])
        train_input_list = rdir / "train_scenes.json"

        cmds = []
        perception_save_dir = None
        if cfg.get("perception_mining"):
            perception_cmd, perception_save_dir = _perception_mining_cmd(cfg, model_path, rdir)
            cmds.append(("perception_mine", perception_cmd))
        cmds.append(("classify", classify_cmd))
        if mine_cmd is not None:
            cmds.append(("mine", mine_cmd))
        cmds.extend(
            [("repair", _repair_cmd(cfg, model_path, credit_jsonl, rdir)), ("memory", memory_cmd)]
        )
        for label, cmd in cmds:
            print(f"[round {round_idx}] {label}: {' '.join(cmd)}")
            if not args.dry_run:
                _run(cmd, rdir / f"{label}.log")
                if label == "perception_mine":
                    assert perception_save_dir is not None
                    scenes = _write_scene_list_from_saved_batches(perception_save_dir, scene_pool)
                    if use_perception_as_credit:
                        _write_credit_rows_from_scene_list(scenes, credit_jsonl)
                if label == "mine":
                    with open(credit_jsonl) as f:
                        rows = [json.loads(line)["scene_path"] for line in f if line.strip()]
                    _write_json(rdir / "credit_windows_paths.json", rows)
                if label == "classify" and use_perception_as_credit and not credit_jsonl.exists():
                    scenes = json.loads(scene_pool.read_text())
                    _write_credit_rows_from_scene_list(scenes, credit_jsonl)
        if args.dry_run:
            continue

        repaired_paths = json.loads(repaired_list_json.read_text())
        replay_paths = json.loads(replay_json.read_text())
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
            _run(train_cmd, rdir / "train.log", cwd=train_cwd, env=train_env)
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
            _run(train_cmd, rdir / "train.log")
            run_dir = _latest_run_dir(out, name)
            model_path = _checkpoint_for(run_dir, checkpoint_policy, model_path)
        previous_memory = memory_json
        print(f"[round {round_idx}] next model: {model_path}")


if __name__ == "__main__":
    main()
