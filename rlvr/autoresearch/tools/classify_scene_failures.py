#!/usr/bin/env python3
"""Classify NPZ scenes by reward/safety failure modes and write training lists.

This is a geometry/reward based classifier, not the explorer-policy classifier.
It scores either the GT future already stored in each NPZ or a deterministic
model prediction, then emits:

* ``classified_scenes.jsonl``: one rich diagnostics row per scene.
* ``summary.json``: label counts and thresholds used.
* ``lists/*.json``: plain NPZ path lists per label, plus ``all_flagged.json``
  and ``clean.json``. These lists are directly usable by RSFT / IL loaders.

Usage:
    python -m rlvr.autoresearch.tools.classify_scene_failures \\
        --scenes scenes.json --config reward_config.json \\
        --threshold_config rlvr/configs/scene_failure_thresholds.json \\
        --output_dir /tmp/scene_flags --trajectory gt

    python -m rlvr.autoresearch.tools.classify_scene_failures \\
        --scenes scenes.json --config reward_config.json \\
        --threshold_config rlvr/configs/scene_failure_thresholds.json \\
        --output_dir /tmp/scene_flags_det --trajectory det \\
        --model_path /path/to/best_model.pth --batch_size 32

    python -m rlvr.autoresearch.tools.classify_scene_failures \\
        --scenes scenes.json --config reward_config.json \\
        --threshold_config rlvr/configs/scene_failure_thresholds.json \\
        --output_dir /tmp/scene_flags_saved --trajectory saved_pred \\
        --predictions_dir /path/to/validation_result/predictions --batch_size 32

    python -m rlvr.autoresearch.tools.classify_scene_failures \\
        --config reward_config.json \\
        --threshold_config rlvr/configs/scene_failure_thresholds.json \\
        --output_dir /tmp/scene_flags_saved \\
        --trajectory saved_pred \\
        --predictions_dir /path/to/validation_result/predictions \\
        --source_scene_root /path/to/dataset_root --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner_metrics.aggregate import compute_subscores_scene_batch
from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
    compute_safety_score_batch,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import RewardConfig

_ALWAYS_WRITE_LISTS = (
    "all_flagged",
    "clean",
    "road_border_crossing",
    "road_border_near",
    "lane_crossing",
    "static_collision",
    "static_near_miss",
    "moving_collision",
    "moving_near_miss",
    "moving_ttc",
)

_DEFAULT_THRESHOLD_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "scene_failure_thresholds.json"
)
_REQUIRED_THRESHOLD_FIELDS = (
    "moving_near_thresh",
    "static_near_thresh",
    "rb_near_thresh",
    "sc_cross_thresh",
    "rb_cross_thresh",
)
_UNKNOWN_NEIGHBOR_SHAPE_M = 2.0
_DEFAULT_NEIGHBOR_WIDTH_M = 2.0
_DEFAULT_NEIGHBOR_LENGTH_M = 4.5
_NEIGHBOR_COORD_EPS_M = 1e-6
_NEIGHBOR_SHAPE_EPS_M = 1e-3
_NO_MOVING_NEIGHBOR_DISTANCE_M = float("inf")
_THRESHOLD_MATCH_TOL = 1e-9
_MISSING_SOURCE_SAMPLE_LIMIT = 5


def _load_scene_thresholds(path: str | Path) -> dict[str, float]:
    with open(path) as f:
        raw = json.load(f)
    missing = [k for k in _REQUIRED_THRESHOLD_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"Threshold config {path} is missing required fields: {missing}")
    return {k: float(raw[k]) for k in _REQUIRED_THRESHOLD_FIELDS}


def _apply_scene_thresholds(config: RewardConfig, args) -> dict[str, float]:
    threshold_config = _load_scene_thresholds(args.threshold_config)

    def resolve(name: str) -> float:
        cli_value = getattr(args, name)
        return float(cli_value) if cli_value is not None else threshold_config[name]

    args.moving_near_thresh = resolve("moving_near_thresh")
    args.static_near_thresh = resolve("static_near_thresh")
    args.rb_near_thresh = resolve("rb_near_thresh")
    config.sc_cross_thresh = resolve("sc_cross_thresh")
    config.rb_cross_thresh = resolve("rb_cross_thresh")
    config.sc_near_thresh = args.static_near_thresh
    config.rb_near_thresh = args.rb_near_thresh

    return {
        "moving_near_thresh": float(args.moving_near_thresh),
        "static_near_thresh": float(args.static_near_thresh),
        "rb_near_thresh": float(args.rb_near_thresh),
        "static_collision_thresh": float(config.sc_cross_thresh),
        "sc_cross_thresh": float(config.sc_cross_thresh),
        "rb_cross_thresh": float(config.rb_cross_thresh),
    }


def _thresholds_match(a: dict[str, Any], b: dict[str, float]) -> bool:
    return all(
        abs(float(a.get(k, float("nan"))) - float(v)) < _THRESHOLD_MATCH_TOL for k, v in b.items()
    )


def _load_scene_paths(path: str | Path) -> list[str]:
    with open(path) as f:
        paths = json.load(f)
    if not isinstance(paths, list):
        raise ValueError(f"{path} must contain a JSON list of NPZ paths")
    return [str(p) for p in paths]


def _iter_prediction_npzs(
    predictions_dir: Path,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
):
    """Yield saved prediction NPZs in deterministic path order."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    global_idx = 0
    for dirpath, dirnames, filenames in os.walk(predictions_dir):
        dirnames[:] = sorted(d for d in dirnames if d != "scene_mining")
        for filename in sorted(filenames):
            if filename.endswith(".npz") and not filename.startswith("loss"):
                if global_idx % num_shards == shard_index:
                    yield global_idx, Path(dirpath) / filename
                global_idx += 1


def _resolve_source_scene_for_prediction(
    prediction_path: Path,
    predictions_dir: Path,
    candidate_source_roots: list[Path],
) -> Path:
    """Resolve a mirrored prediction path back to its source scene NPZ.

    The saved predictions in some validation runs are stored as:
        predictions/<scene_relpath>.npz

    while source datasets may be:
        source_root/<scene_relpath>.npz
    or:
        source_root/<dataset_group>/<scene_relpath>.npz
    """
    rel = prediction_path.relative_to(predictions_dir)
    checked: list[Path] = []
    for root in candidate_source_roots:
        candidate = root / rel
        checked.append(candidate)
        if candidate.exists():
            return candidate

    sample = ", ".join(str(p) for p in checked[:_MISSING_SOURCE_SAMPLE_LIMIT])
    if len(checked) > _MISSING_SOURCE_SAMPLE_LIMIT:
        sample += ", ..."
    raise FileNotFoundError(
        f"source scene not found for prediction {prediction_path}; checked {sample}"
    )


def _candidate_source_roots(source_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in source_roots:
        candidates.append(root)
        if root.is_dir():
            candidates.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return candidates


def _prediction_scene_pairs_from_dir(
    predictions_dir: Path,
    source_roots: list[Path],
    *,
    max_pairs: int | None = None,
) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    candidate_roots = _candidate_source_roots(source_roots)
    for _, prediction_path in _iter_prediction_npzs(predictions_dir):
        source_scene = _resolve_source_scene_for_prediction(
            prediction_path,
            predictions_dir,
            candidate_roots,
        )
        pairs.append((str(source_scene), prediction_path))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def _future_heading_to_cos_sin(fut: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., T, 3)`` x/y/yaw futures to ``(..., T, 4)`` x/y/cos/sin."""
    if fut.shape[-1] >= 4:
        return fut
    if fut.shape[-1] != 3:
        raise ValueError(f"future tensor last dim must be 3 or >=4, got {tuple(fut.shape)}")
    return torch.cat([fut[..., :2], torch.cos(fut[..., 2:3]), torch.sin(fut[..., 2:3])], dim=-1)


def _load_npz_data(npz_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(str(npz_path)) as loaded:
        data: dict[str, torch.Tensor] = {}
        for key, value in loaded.items():
            if key in {"map_name", "token", "origin"}:
                continue
            array = np.asarray(value)
            if array.dtype.kind in ("U", "S", "O"):
                continue
            if key == "delay":
                data[key] = torch.as_tensor(array.reshape(-1), dtype=torch.long, device=device)
                continue
            if array.dtype.kind == "u" and array.dtype != np.uint8:
                array = array.astype(np.int64)
            data[key] = torch.as_tensor(np.expand_dims(array, axis=0), device=device)

    if "goal_pose" in data:
        data["goal_pose"] = _future_heading_to_cos_sin(data["goal_pose"])
    if "ego_agent_past" in data:
        data["ego_agent_past"] = _future_heading_to_cos_sin(data["ego_agent_past"])
    if "ego_shape" not in data:
        raise ValueError(f"'{npz_path}' is missing required ego_shape")
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data


def _prepare_scoring_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone data and normalize future tensors for reward/scoring code.

    Scene-editor exports may store futures as x/y/yaw. The reward geometry
    expects x/y/cos/sin, so convert in memory. This leaves source NPZs intact.
    """
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    if "ego_agent_future" in out:
        out["ego_agent_future"] = _future_heading_to_cos_sin(out["ego_agent_future"])
    if "neighbor_agents_future" in out:
        out["neighbor_agents_future"] = _future_heading_to_cos_sin(out["neighbor_agents_future"])
    return out


def _stack_scene_data(datas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack loaded scene dicts into one B-major dict for batched scoring."""
    if not datas:
        raise ValueError("cannot stack an empty scene batch")

    expected_keys = set(datas[0])
    for idx, data in enumerate(datas[1:], start=1):
        if set(data) != expected_keys:
            missing = sorted(expected_keys - set(data))
            extra = sorted(set(data) - expected_keys)
            raise ValueError(
                f"scene batch has inconsistent tensor keys at index {idx}; "
                f"missing={missing}, extra={extra}"
            )

    stacked: dict[str, torch.Tensor] = {}
    for key in datas[0]:
        values = [data[key] for data in datas]
        first = values[0]
        if not torch.is_tensor(first):
            stacked[key] = first
            continue
        if all(torch.is_tensor(v) and v.shape == first.shape for v in values):
            if first.dim() > 0 and first.shape[0] == 1:
                stacked[key] = torch.cat(values, dim=0)
            else:
                stacked[key] = torch.stack(values, dim=0)
        else:
            shapes = [tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for v in values]
            raise ValueError(f"cannot batch key {key}: inconsistent shapes {shapes}")
    return stacked


def _slice_scene_data(
    data: dict[str, torch.Tensor], scene_idx: int, batch_size: int
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
            out[key] = value[scene_idx : scene_idx + 1]
        else:
            out[key] = value
    return out


def _ego_shape_from_data(data: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if "ego_shape" not in data:
        raise ValueError("scene is missing required ego_shape")
    ego_shape = data["ego_shape"]
    if ego_shape.dim() == 2:
        ego_shape = ego_shape[0]
    if ego_shape.numel() < 3:
        raise ValueError(f"ego_shape has shape {tuple(ego_shape.shape)}, expected >=3 values")
    return ego_shape[:3].to(device)


def _gt_trajectory(data: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if "ego_agent_future" not in data:
        raise ValueError("scene is missing ego_agent_future; cannot use --trajectory gt")
    traj = data["ego_agent_future"]
    if traj.dim() == 3:
        traj = traj[0]
    traj = _future_heading_to_cos_sin(traj)
    return traj[:, :4].to(device).unsqueeze(0)


def _prediction_path_for_scene(
    predictions_dir: Path,
    scene_path: str,
    scene_index: int,
    *,
    prediction_scene_root: Path | None = None,
) -> Path:
    """Resolve saved predictions from flat or source-path-mirrored layouts."""
    flat = predictions_dir / f"prediction{scene_index:08d}.npz"
    if flat.exists():
        return flat

    scene = Path(scene_path)
    direct = predictions_dir / scene.name
    if direct.exists():
        return direct

    if prediction_scene_root is not None:
        try:
            rooted = predictions_dir / scene.relative_to(prediction_scene_root)
        except ValueError:
            rooted = None
        if rooted is not None and rooted.exists():
            return rooted

    parts = scene.parts
    first_relative_part = 1 if scene.is_absolute() else 0
    for start in range(first_relative_part, len(parts)):
        candidate = predictions_dir.joinpath(*parts[start:])
        if candidate.exists():
            return candidate

    return flat


def _saved_prediction_trajectory(
    prediction_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    with np.load(prediction_path) as pred_npz:
        if "prediction" not in pred_npz:
            raise ValueError(f"{prediction_path} is missing 'prediction'")
        pred = torch.as_tensor(pred_npz["prediction"], dtype=torch.float32, device=device)

    if pred.dim() == 3:
        pred = pred[0]
    elif pred.dim() != 2:
        raise ValueError(
            f"{prediction_path} prediction must be shaped (A,T,4) or (T,4), got {tuple(pred.shape)}"
        )
    if pred.shape[0] == 0 or pred.shape[-1] < 4:
        raise ValueError(
            f"{prediction_path} prediction must have non-empty T and >=4 channels, got {tuple(pred.shape)}"
        )
    return pred[:, :4].unsqueeze(0)


def _neighbor_inputs(
    data: dict[str, torch.Tensor],
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbor_futures = torch.zeros(0, T, 4, device=device)
    neighbor_shapes = torch.zeros(0, 2, device=device)
    neighbor_valid = torch.zeros(0, T, dtype=torch.bool, device=device)

    if "neighbor_agents_future" not in data:
        return neighbor_futures, neighbor_shapes, neighbor_valid

    nf = data["neighbor_agents_future"]
    if nf.dim() == 4:
        nf = nf[0]
    if nf.shape[0] == 0:
        return neighbor_futures, neighbor_shapes, neighbor_valid
    if nf.shape[1] < T or (nf.shape[2] != 3 and nf.shape[2] < 4):
        raise ValueError(
            "neighbor_agents_future must be shaped (N,T,3|4) with x,y,yaw or x,y,cos,sin; "
            f"got {tuple(nf.shape)} for trajectory length {T}"
        )

    nf_data = _future_heading_to_cos_sin(nf[:, :T])
    slot_valid = nf_data[:, :, :2].abs().sum(dim=(1, 2)) > _NEIGHBOR_COORD_EPS_M
    if not slot_valid.any():
        return neighbor_futures, neighbor_shapes, neighbor_valid

    neighbor_futures = nf_data[slot_valid].to(device)
    neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > _NEIGHBOR_COORD_EPS_M

    if "neighbor_agents_past" in data:
        nap = data["neighbor_agents_past"]
        if nap.dim() == 4:
            nap = nap[0]
        ns = nap[slot_valid, -1, :]
        if ns.shape[-1] >= 8:
            neighbor_shapes = ns[:, [6, 7]].to(device)  # width, length
        else:
            neighbor_shapes = torch.full(
                (neighbor_futures.shape[0], 2),
                _UNKNOWN_NEIGHBOR_SHAPE_M,
                device=device,
            )
    else:
        neighbor_shapes = torch.full(
            (neighbor_futures.shape[0], 2),
            _UNKNOWN_NEIGHBOR_SHAPE_M,
            device=device,
        )

    zero_shapes = neighbor_shapes.abs().sum(dim=-1) < _NEIGHBOR_SHAPE_EPS_M
    if zero_shapes.any():
        neighbor_shapes = neighbor_shapes.clone()
        neighbor_shapes[zero_shapes] = torch.tensor(
            [_DEFAULT_NEIGHBOR_WIDTH_M, _DEFAULT_NEIGHBOR_LENGTH_M],
            device=device,
        )

    return neighbor_futures, neighbor_shapes, neighbor_valid


def _stopped_neighbor_mask(
    neighbor_futures: torch.Tensor,
    neighbor_valid: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    N_nb = neighbor_futures.shape[0]
    device = neighbor_futures.device
    if N_nb == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    nb_xy = neighbor_futures[:, :, :2]
    if nb_xy.shape[1] < 2:
        return torch.zeros(N_nb, dtype=torch.bool, device=device)

    both_valid_01 = neighbor_valid[:, 0] & neighbor_valid[:, 1]
    v0 = torch.zeros(N_nb, device=device)
    if both_valid_01.any():
        v0[both_valid_01] = (nb_xy[both_valid_01, 1] - nb_xy[both_valid_01, 0]).norm(
            dim=-1
        ) / config.dt

    disp_all = (nb_xy - nb_xy[:, 0:1]).norm(dim=-1)
    max_disp = disp_all.masked_fill(~neighbor_valid, 0.0).max(dim=1).values
    has_any_valid = neighbor_valid.any(dim=1)

    return (
        has_any_valid
        & both_valid_01
        & (v0 < config.sc_neighbor_vel_thresh)
        & (max_disp < config.sc_neighbor_disp_thresh)
    )


def _first_step(steps: list[int | None]) -> int | None:
    values = [s for s in steps if s is not None]
    return min(values) if values else None


def _moving_diagnostics(
    ego_traj: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
    moving_near_thresh: float,
    device: torch.device,
) -> dict[str, Any]:
    T = ego_traj.shape[1]
    ego_shape = _ego_shape_from_data(data, device)
    neighbor_futures, neighbor_shapes, neighbor_valid = _neighbor_inputs(data, T, device)
    stopped_mask = _stopped_neighbor_mask(neighbor_futures, neighbor_valid, config)
    moving_mask = ~stopped_mask
    moving_count = int(moving_mask.sum().item())

    empty = {
        "moving_neighbor_count": moving_count,
        "stopped_neighbor_count": int(stopped_mask.sum().item()),
        "moving_min_dist": _NO_MOVING_NEIGHBOR_DISTANCE_M,
        "moving_argmin_neighbor": None,
        "moving_argmin_t": None,
        "moving_collision_step": None,
        "moving_near_miss": False,
    }
    if moving_count == 0:
        return empty

    mf = neighbor_futures[moving_mask]
    ms = neighbor_shapes[moving_mask]
    mv = neighbor_valid[moving_mask]
    moving_global_idx = moving_mask.nonzero(as_tuple=True)[0]

    distances = compute_ego_neighbor_signed_clearance(ego_traj, ego_shape, mf, ms, mv)
    flat_idx = int(distances.reshape(-1).argmin().item())
    _, M, T = distances.shape
    argmin_m = (flat_idx // T) % M
    argmin_t = flat_idx % T
    min_dist = float(distances.reshape(-1)[flat_idx].item())

    _, moving_collision_steps = compute_safety_score_batch(
        ego_traj,
        ego_shape,
        mf,
        ms,
        mv,
        config,
    )
    moving_collision_step = _first_step(moving_collision_steps)

    return {
        "moving_neighbor_count": moving_count,
        "stopped_neighbor_count": int(stopped_mask.sum().item()),
        "moving_min_dist": min_dist,
        "moving_argmin_neighbor": int(moving_global_idx[argmin_m].item()),
        "moving_argmin_t": int(argmin_t),
        "moving_collision_step": moving_collision_step,
        "moving_near_miss": moving_collision_step is None and min_dist < moving_near_thresh,
    }


def classify_loaded_scene(
    scene_path: str,
    ego_traj: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
    *,
    moving_near_thresh: float,
    static_near_thresh: float,
    rb_near_thresh: float,
    device: torch.device,
) -> dict[str, Any]:
    rows = classify_loaded_scenes_batch(
        [scene_path],
        ego_traj.unsqueeze(0),
        [_prepare_scoring_data(data)],
        config,
        moving_near_thresh=moving_near_thresh,
        static_near_thresh=static_near_thresh,
        rb_near_thresh=rb_near_thresh,
        device=device,
    )
    return rows[0]


def _build_candidate_row(
    scene_path: str,
    candidate_idx: int,
    subs: dict[str, torch.Tensor | list[list[int | None]]],
    moving: dict[str, Any],
    *,
    bidx: int,
    static_near_thresh: float,
    rb_near_thresh: float,
) -> dict[str, Any]:
    labels: list[str] = []
    rb_crossing = bool(subs["rb_crossing_gate"][bidx, candidate_idx].item() < 0.5)
    rb_min_dist = float(subs["rb_min_dist"][bidx, candidate_idx].item())
    if rb_crossing:
        labels.append("road_border_crossing")
    elif rb_min_dist < rb_near_thresh:
        labels.append("road_border_near")

    lane_crossing = bool(subs["lane_crossing_gate"][bidx, candidate_idx].item() < 0.5)
    if lane_crossing:
        labels.append("lane_crossing")

    static_collision = bool(subs["sc_crossing_gate"][bidx, candidate_idx].item() < 0.5)
    static_min_dist = float(subs["sc_min_dist"][bidx, candidate_idx].item())
    static_neighbor_count = int(subs["sc_n_stopped"][bidx, candidate_idx].item())
    static_collision_step = subs["sc_crossing_steps"][bidx][candidate_idx]
    if static_collision:
        labels.append("static_collision")
    elif static_neighbor_count > 0 and static_min_dist < static_near_thresh:
        labels.append("static_near_miss")

    if moving["moving_collision_step"] is not None:
        labels.append("moving_collision")
    elif moving["moving_near_miss"]:
        labels.append("moving_near_miss")

    ttc_first_unsafe = subs["ttc_first_unsafe_steps"][bidx][candidate_idx]
    if ttc_first_unsafe is not None and moving["moving_collision_step"] is None:
        labels.append("moving_ttc")

    if not labels:
        labels.append("clean")

    return {
        "scene_path": scene_path,
        "candidate_index": candidate_idx,
        "labels": labels,
        "trajectory_source": None,  # filled by caller
        "rb_min_dist": rb_min_dist,
        "rb_crossing": rb_crossing,
        "rb_crossing_step": subs["rb_crossing_steps"][bidx][candidate_idx],
        "lane_crossing": lane_crossing,
        "lane_crossing_step": subs["lane_crossing_steps"][bidx][candidate_idx],
        "static_collision": static_collision,
        "static_min_dist": static_min_dist,
        "static_collision_step": static_collision_step,
        "static_neighbor_count": static_neighbor_count,
        # Backward-compatible aliases for older consumers.
        "static_crossing": static_collision,
        "sc_min_dist": static_min_dist,
        "sc_crossing_step": static_collision_step,
        "sc_n_stopped": static_neighbor_count,
        "collision_step": subs["collision_step"][bidx][candidate_idx],
        "ttc_score": float(subs["ttc"][bidx, candidate_idx].item()),
        "ttc_first_unsafe_step": ttc_first_unsafe,
        "ttc_first_collision_step": subs["ttc_first_collision_steps"][bidx][candidate_idx],
        **moving,
    }


def classify_loaded_scenes_batch(
    scene_paths: list[str],
    ego_trajs: torch.Tensor,
    datas: list[dict[str, torch.Tensor]],
    config: RewardConfig,
    *,
    moving_near_thresh: float,
    static_near_thresh: float,
    rb_near_thresh: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Classify a B-scene batch with one scored trajectory per scene."""
    if ego_trajs.dim() != 4:
        raise ValueError(
            "classify_loaded_scenes_batch expects ego_trajs shaped (B,N,T,4); "
            f"got {tuple(ego_trajs.shape)}"
        )
    if ego_trajs.shape[1] != 1:
        raise ValueError(
            "dangerous scene classification expects exactly one trajectory per scene; "
            f"got N={ego_trajs.shape[1]}"
        )
    if len(scene_paths) != ego_trajs.shape[0] or len(datas) != ego_trajs.shape[0]:
        raise ValueError(
            "scene_paths, ego_trajs, and datas must have the same scene batch size; "
            f"got {len(scene_paths)}, {ego_trajs.shape[0]}, {len(datas)}"
        )

    prepared_datas = [_prepare_scoring_data(data) for data in datas]
    batched_data = _stack_scene_data(prepared_datas)
    subs = compute_subscores_scene_batch(ego_trajs, batched_data, config)

    rows: list[dict[str, Any]] = []
    B = ego_trajs.shape[0]
    for bidx, scene_path in enumerate(scene_paths):
        scene_data = _slice_scene_data(batched_data, bidx, B)
        moving = _moving_diagnostics(
            ego_trajs[bidx, 0:1],
            scene_data,
            config,
            moving_near_thresh,
            device,
        )
        rows.append(
            _build_candidate_row(
                scene_path,
                0,
                subs,
                moving,
                bidx=bidx,
                static_near_thresh=static_near_thresh,
                rb_near_thresh=rb_near_thresh,
            )
        )
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_outputs(
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    output_dir: Path,
    thresholds: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "classified_scenes.jsonl"
    with open(jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    path_labels: dict[str, set[str]] = defaultdict(set)
    path_order: list[str] = []
    for row in rows:
        path = row["scene_path"]
        if path not in path_labels:
            path_order.append(path)
        path_labels[path].update(row["labels"])

    by_label: dict[str, list[str]] = defaultdict(list)
    all_flagged: list[str] = []
    clean: list[str] = []
    for path in path_order:
        labels = path_labels[path]
        if labels == {"clean"}:
            clean.append(path)
        else:
            all_flagged.append(path)
            for label in sorted(labels):
                if label != "clean":
                    by_label[label].append(path)

    by_label["all_flagged"] = all_flagged
    by_label["clean"] = clean
    lists_dir = output_dir / "lists"
    for label in sorted(set(_ALWAYS_WRITE_LISTS) | set(by_label)):
        _write_json(lists_dir / f"{label}.json", by_label.get(label, []))

    counts = Counter(label for row in rows for label in row["labels"])
    summary = {
        "n_input": len(rows) + len(errors),
        "n_classified": len(rows),
        "n_errors": len(errors),
        "label_counts": dict(sorted(counts.items())),
        "thresholds": thresholds,
        "outputs": {
            "classified_scenes_jsonl": str(jsonl),
            "lists_dir": str(lists_dir),
        },
        "errors": errors,
    }
    _write_json(output_dir / "summary.json", summary)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_output_dirs(
    input_dirs: list[Path], output_dir: Path, thresholds: dict[str, float]
) -> None:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for input_dir in input_dirs:
        rows.extend(_read_jsonl(input_dir / "classified_scenes.jsonl"))
        summary_path = input_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            shard_thresholds = summary.get("thresholds")
            if shard_thresholds is not None and not _thresholds_match(shard_thresholds, thresholds):
                raise ValueError(
                    f"{summary_path} thresholds {shard_thresholds} do not match requested "
                    f"merge thresholds {thresholds}"
                )
            errors.extend(summary.get("errors", []))
    rows.sort(key=lambda row: str(row.get("prediction_path", row["scene_path"])))
    _write_outputs(rows, errors, output_dir, thresholds)


def _classify_gt(
    scene_paths: list[str],
    config: RewardConfig,
    args,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for start in range(0, len(scene_paths), args.batch_size):
        batch_paths = scene_paths[start : start + args.batch_size]
        datas: list[dict[str, torch.Tensor]] = []
        ego_trajs: list[torch.Tensor] = []
        valid_paths: list[str] = []
        for offset, scene_path in enumerate(batch_paths):
            idx = start + offset
            try:
                data = _load_npz_data(scene_path, device)
                datas.append(data)
                ego_trajs.append(_gt_trajectory(_prepare_scoring_data(data), device))
                valid_paths.append(scene_path)
            except Exception as exc:  # noqa: BLE001
                errors.append({"scene_path": scene_path, "error": str(exc)})
                print(f"  [{idx:4d}] ERROR {Path(scene_path).name}: {exc}")
        if not datas:
            continue
        try:
            batch_rows = classify_loaded_scenes_batch(
                valid_paths,
                torch.stack(ego_trajs, dim=0),
                datas,
                config,
                moving_near_thresh=args.moving_near_thresh,
                static_near_thresh=args.static_near_thresh,
                rb_near_thresh=args.rb_near_thresh,
                device=device,
            )
            for bi, row in enumerate(batch_rows):
                row["trajectory_source"] = "gt"
                rows.append(row)
                print(
                    f"  [{start + bi:4d}] {Path(row['scene_path']).name}: {','.join(row['labels'])}"
                )
        except Exception as exc:  # noqa: BLE001
            for scene_path in valid_paths:
                errors.append({"scene_path": scene_path, "error": str(exc)})
            print(f"  [{start:4d}] ERROR batch {len(valid_paths)} scenes: {exc}")
    return rows, errors


@torch.no_grad()
def _classify_det(
    scene_paths: list[str],
    config: RewardConfig,
    args,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not args.model_path:
        raise ValueError("--trajectory det requires --model_path")
    from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model

    model, model_args = load_model(args.model_path, device)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for start in range(0, len(scene_paths), args.batch_size):
        batch_paths = scene_paths[start : start + args.batch_size]
        datas: list[dict[str, torch.Tensor]] = []
        valid_paths: list[str] = []
        for scene_path in batch_paths:
            try:
                datas.append(_load_npz_data(scene_path, device))
                valid_paths.append(scene_path)
            except Exception as exc:  # noqa: BLE001
                errors.append({"scene_path": scene_path, "error": str(exc)})
                print(f"  [err ] {Path(scene_path).name}: {exc}")
        if not datas:
            continue
        det_trajs = det_inference_batched(model, model_args, datas, device)
        try:
            batch_rows = classify_loaded_scenes_batch(
                valid_paths,
                det_trajs.unsqueeze(1),
                datas,
                config,
                moving_near_thresh=args.moving_near_thresh,
                static_near_thresh=args.static_near_thresh,
                rb_near_thresh=args.rb_near_thresh,
                device=device,
            )
            for bi, row in enumerate(batch_rows):
                row["trajectory_source"] = "det"
                rows.append(row)
                print(
                    f"  [{start + bi:4d}] {Path(row['scene_path']).name}: {','.join(row['labels'])}"
                )
        except Exception as exc:  # noqa: BLE001
            for scene_path in valid_paths:
                errors.append({"scene_path": scene_path, "error": str(exc)})
            print(f"  [{start:4d}] ERROR batch {len(valid_paths)} scenes: {exc}")
    return rows, errors


def _classify_saved_predictions(
    scene_paths: list[str],
    config: RewardConfig,
    args,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not args.predictions_dir:
        raise ValueError("--trajectory saved_pred requires --predictions_dir")
    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.is_dir():
        raise ValueError(f"--predictions_dir is not a directory: {predictions_dir}")
    prediction_scene_root = (
        Path(args.prediction_scene_root) if args.prediction_scene_root is not None else None
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for start in range(0, len(scene_paths), args.batch_size):
        batch_paths = scene_paths[start : start + args.batch_size]
        datas: list[dict[str, torch.Tensor]] = []
        ego_trajs: list[torch.Tensor] = []
        valid_paths: list[str] = []
        for offset, scene_path in enumerate(batch_paths):
            idx = start + offset
            try:
                prediction_path = _prediction_path_for_scene(
                    predictions_dir,
                    scene_path,
                    idx,
                    prediction_scene_root=prediction_scene_root,
                )
                if not prediction_path.exists():
                    raise FileNotFoundError(f"saved prediction not found: {prediction_path}")
                ego_trajs.append(_saved_prediction_trajectory(prediction_path, device))
                datas.append(_load_npz_data(scene_path, device))
                valid_paths.append(scene_path)
            except Exception as exc:  # noqa: BLE001
                errors.append({"scene_path": scene_path, "error": str(exc)})
                print(f"  [{idx:4d}] ERROR {Path(scene_path).name}: {exc}")
        if not datas:
            continue
        try:
            batch_rows = classify_loaded_scenes_batch(
                valid_paths,
                torch.stack(ego_trajs, dim=0),
                datas,
                config,
                moving_near_thresh=args.moving_near_thresh,
                static_near_thresh=args.static_near_thresh,
                rb_near_thresh=args.rb_near_thresh,
                device=device,
            )
            for bi, row in enumerate(batch_rows):
                row["trajectory_source"] = "saved_pred"
                rows.append(row)
                print(
                    f"  [{start + bi:4d}] {Path(row['scene_path']).name}: {','.join(row['labels'])}"
                )
        except Exception as exc:  # noqa: BLE001
            for scene_path in valid_paths:
                errors.append({"scene_path": scene_path, "error": str(exc)})
            print(f"  [{start:4d}] ERROR batch {len(valid_paths)} scenes: {exc}")
    return rows, errors


def _classify_saved_prediction_pairs(
    scene_prediction_pairs: list[tuple[str, Path]],
    config: RewardConfig,
    args,
    device: torch.device,
    *,
    start_index_base: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for start in range(0, len(scene_prediction_pairs), args.batch_size):
        batch_pairs = scene_prediction_pairs[start : start + args.batch_size]
        datas: list[dict[str, torch.Tensor]] = []
        ego_trajs: list[torch.Tensor] = []
        valid_paths: list[str] = []
        valid_prediction_paths: list[Path] = []
        for offset, (scene_path, prediction_path) in enumerate(batch_pairs):
            idx = start_index_base + start + offset
            try:
                ego_trajs.append(_saved_prediction_trajectory(prediction_path, device))
                datas.append(_load_npz_data(scene_path, device))
                valid_paths.append(scene_path)
                valid_prediction_paths.append(prediction_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "scene_path": scene_path,
                        "prediction_path": str(prediction_path),
                        "error": str(exc),
                    }
                )
                print(f"  [{idx:4d}] ERROR {Path(scene_path).name}: {exc}")
        if not datas:
            continue
        try:
            batch_rows = classify_loaded_scenes_batch(
                valid_paths,
                torch.stack(ego_trajs, dim=0),
                datas,
                config,
                moving_near_thresh=args.moving_near_thresh,
                static_near_thresh=args.static_near_thresh,
                rb_near_thresh=args.rb_near_thresh,
                device=device,
            )
            for bi, row in enumerate(batch_rows):
                row["trajectory_source"] = "saved_pred"
                row["prediction_path"] = str(valid_prediction_paths[bi])
                rows.append(row)
                print(
                    f"  [{start_index_base + start + bi:4d}] "
                    f"{Path(row['scene_path']).name}: {','.join(row['labels'])}"
                )
        except Exception as exc:  # noqa: BLE001
            for scene_path, prediction_path in zip(valid_paths, valid_prediction_paths):
                errors.append(
                    {
                        "scene_path": scene_path,
                        "prediction_path": str(prediction_path),
                        "error": str(exc),
                    }
                )
            print(f"  [{start:4d}] ERROR batch {len(valid_paths)} scenes: {exc}")
    return rows, errors


def _classify_saved_prediction_dir(
    predictions_dir: Path,
    source_roots: list[Path],
    config: RewardConfig,
    args,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    candidate_roots = _candidate_source_roots(source_roots)
    batch_pairs: list[tuple[str, Path]] = []
    n_seen = 0

    def flush_batch(start_index: int) -> None:
        nonlocal rows, errors, batch_pairs
        if not batch_pairs:
            return
        batch_rows, batch_errors = _classify_saved_prediction_pairs(
            batch_pairs,
            config,
            args,
            device,
            start_index_base=start_index,
        )
        rows.extend(batch_rows)
        errors.extend(batch_errors)
        batch_pairs = []

    batch_start = 0
    for global_idx, prediction_path in _iter_prediction_npzs(
        predictions_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ):
        if args.max_scenes is not None and global_idx >= args.max_scenes:
            break
        try:
            scene_path = _resolve_source_scene_for_prediction(
                prediction_path,
                predictions_dir,
                candidate_roots,
            )
            if not batch_pairs:
                batch_start = global_idx
            batch_pairs.append((str(scene_path), prediction_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "scene_path": "",
                    "prediction_path": str(prediction_path),
                    "error": str(exc),
                }
            )
            print(f"  [{global_idx:4d}] ERROR {prediction_path.name}: {exc}")
        n_seen += 1
        if len(batch_pairs) >= args.batch_size:
            flush_batch(batch_start)

    flush_batch(batch_start)
    return rows, errors, n_seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help=(
            "JSON list of NPZ scene paths. Required for gt/det. Optional for saved_pred "
            "when --source_scene_root is provided."
        ),
    )
    parser.add_argument("--config", required=True, help="Reward config JSON")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trajectory", choices=("gt", "det", "saved_pred"), default="gt")
    parser.add_argument("--model_path", default=None, help="Required when --trajectory det")
    parser.add_argument(
        "--predictions_dir",
        default=None,
        help="Required when --trajectory saved_pred; valid_predictor saved predictions directory",
    )
    parser.add_argument(
        "--prediction_scene_root",
        default=None,
        help=(
            "Optional source scene root to strip when resolving mirrored saved predictions. "
            "Flat predictionNNNNNNNN.npz resolution does not need this."
        ),
    )
    parser.add_argument(
        "--source_scene_root",
        action="append",
        default=None,
        help=(
            "Source dataset root for mirrored saved predictions. May be repeated. "
            "When --trajectory saved_pred and --scenes is omitted, every prediction NPZ "
            "under --predictions_dir is paired to this root by relative path."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--threshold_config",
        type=Path,
        required=True,
        help=(
            "Scene-mining threshold JSON, for example "
            f"{_DEFAULT_THRESHOLD_CONFIG}. CLI threshold flags override values in this file."
        ),
    )
    parser.add_argument("--moving_near_thresh", type=float, default=None)
    parser.add_argument("--static_near_thresh", type=float, default=None)
    parser.add_argument("--rb_near_thresh", type=float, default=None)
    parser.add_argument("--sc_cross_thresh", type=float, default=None)
    parser.add_argument("--rb_cross_thresh", type=float, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--merge_output_dirs",
        nargs="+",
        default=None,
        help="Merge previously written shard output dirs into --output_dir and exit.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_reward_config(args.config)
    thresholds = _apply_scene_thresholds(config, args)
    if args.merge_output_dirs is not None:
        _merge_output_dirs(
            [Path(p) for p in args.merge_output_dirs],
            Path(args.output_dir),
            thresholds,
        )
        print(f"Merged {len(args.merge_output_dirs)} dirs into {args.output_dir}")
        return

    scene_paths: list[str] = []
    discovered_prediction_count: int | None = None
    if args.trajectory == "saved_pred" and args.scenes is None:
        if not args.predictions_dir:
            raise ValueError("--trajectory saved_pred requires --predictions_dir")
        if not args.source_scene_root:
            raise ValueError(
                "--trajectory saved_pred without --scenes requires at least one --source_scene_root"
            )
        predictions_dir = Path(args.predictions_dir)
        source_roots = [Path(p) for p in args.source_scene_root]
        rows, errors, discovered_prediction_count = _classify_saved_prediction_dir(
            predictions_dir,
            source_roots,
            config,
            args,
            device,
        )
    else:
        if args.scenes is None:
            raise ValueError(f"--trajectory {args.trajectory} requires --scenes")
        scene_paths = _load_scene_paths(args.scenes)
        if args.max_scenes is not None:
            scene_paths = scene_paths[: args.max_scenes]

    if args.trajectory == "gt":
        rows, errors = _classify_gt(scene_paths, config, args, device)
    elif args.trajectory == "det":
        rows, errors = _classify_det(scene_paths, config, args, device)
    elif discovered_prediction_count is not None:
        scene_paths = [str(row["scene_path"]) for row in rows] + [
            str(err.get("scene_path", "")) for err in errors
        ]
    else:
        rows, errors = _classify_saved_predictions(scene_paths, config, args, device)

    _write_outputs(rows, errors, Path(args.output_dir), thresholds)
    total = (
        discovered_prediction_count if discovered_prediction_count is not None else len(scene_paths)
    )
    print(f"\nClassified {len(rows)}/{total} scenes; errors={len(errors)}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
