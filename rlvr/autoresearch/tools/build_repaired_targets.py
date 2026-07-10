#!/usr/bin/env python3
"""Build repaired curated targets for mined dangerous scenes.

For each scene, generate K guided candidates under the source model, classify
every candidate with the same dangerous-scene logic used in the mining step,
and keep only candidates that actually repair the mined issue while also
passing the global safety / lane / kinematic gates. The selected candidate is
written back into ``ego_agent_future`` for base SFT / IL training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner_metrics.subscores import compute_road_border_penalty
from rlvr.autoresearch.tools.classify_scene_failures import (
    _ego_shape_from_data,
    _load_scene_thresholds,
    classify_loaded_scene_candidates_batch,
    current_ego_neighbor_clearance,
)
from rlvr.autoresearch.tools.eval_det_avoidance import load_model, load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.deviation import rollout_gt_deviation
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
)
from rlvr.reward import compute_reward_batch

_GENERATION_MODE_GUIDED_VARIANT = "guided_variant"
_GENERATION_MODE_GRPO_TEMPERATURE = "grpo_temperature"

_REPAIRABLE_LABELS = {
    "road_border_crossing",
    "static_collision",
    "moving_collision",
    "expert_disagreement",
}
_VALIDITY_LABEL_WEIGHTS = {
    "moving_ttc": 4.0,
    "moving_near_miss": 3.0,
    "static_near_miss": 2.0,
    "road_border_near": 1.0,
    "expert_disagreement": 0.5,
}
_NEIGHBOR_COORD_EPS_M = 1e-6
_TRAINING_NPZ_KEYS = {
    "ego_agent_past",
    "ego_agent_future",
    "ego_current_state",
    "goal_pose",
    "neighbor_agents_past",
    "neighbor_agents_future",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "polygons",
    "line_strings",
    "ego_shape",
    "turn_indicators",
}


def _parse_ego_shape(text: str) -> np.ndarray | None:
    if text.strip().lower() in {"from_npz", "npz", "scene"}:
        return None
    vals = np.array([float(x) for x in text.split(",")], dtype=np.float32)
    if vals.shape != (3,):
        raise ValueError(f"--ego_shape must be WB,L,W, got {text!r}")
    return vals


def _output_name_for_scene(scene_path: str) -> str:
    path = Path(scene_path)
    parent = path.parent.name
    if not parent:
        return path.name
    return f"{parent}__{path.name}"


def _filtered_npz_payload(loaded) -> dict[str, Any]:
    return {
        key: value
        for key, value in loaded.items()
        if key in _TRAINING_NPZ_KEYS and np.asarray(value).dtype.kind not in ("U", "S", "O")
    }


def _future4_to_3col(traj: np.ndarray) -> np.ndarray:
    if traj.ndim != 2 or traj.shape[1] != 4:
        raise ValueError(f"expected (T,4) future, got {traj.shape}")
    yaw = np.arctan2(traj[:, 3], traj[:, 2])
    return np.column_stack([traj[:, 0], traj[:, 1], yaw]).astype(np.float32)


def _future_to_4col(traj: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected future shaped (T,C) or (1,T,C), got {arr.shape}")
    if arr.shape[1] >= 4:
        return arr[:, :4].astype(np.float32)
    if arr.shape[1] != 3:
        raise ValueError(f"expected 3 or 4 future channels, got {arr.shape}")
    yaw = arr[:, 2]
    return np.column_stack([arr[:, 0], arr[:, 1], np.cos(yaw), np.sin(yaw)]).astype(np.float32)


@torch.no_grad()
def _generate_grpo_temperature_scenes(
    model,
    norm_batch: dict[str, torch.Tensor],
    *,
    K: int,
    grpo_noise_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate K candidates per scene using the same sampler as train_grpo_predictor.py."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if grpo_noise_scale < 0:
        raise ValueError(f"grpo_noise_scale must be >= 0, got {grpo_noise_scale}")

    from diffusion_planner.grpo_utils import expand_batch, sample_group

    n_scenes = int(norm_batch["ego_current_state"].shape[0])
    expanded = expand_batch(norm_batch, K)
    trajs = sample_group(model, expanded, grpo_noise_scale, device)
    if trajs.shape[0] != n_scenes * K:
        raise RuntimeError(
            f"GRPO temperature sampler returned {trajs.shape[0]} trajectories for "
            f"{n_scenes} scenes and K={K}"
        )
    return trajs.reshape(n_scenes, K, trajs.shape[1], trajs.shape[2])


def _candidate_violation_score(label_row: dict[str, Any], reward_row) -> float:
    labels = {str(label) for label in label_row.get("labels", []) if str(label) != "clean"}
    score = sum(_VALIDITY_LABEL_WEIGHTS.get(label, 1.0) for label in labels)
    if bool(label_row.get("expert_disagreement", False)) and "expert_disagreement" not in labels:
        score += _VALIDITY_LABEL_WEIGHTS["expert_disagreement"]
    moving_step = label_row.get("moving_collision_step")
    if moving_step is not None:
        score += 10.0
    if getattr(reward_row, "lane_crossing", False):
        score += 10.0
    if getattr(reward_row, "kinematic_violated", False):
        score += 10.0
    return float(score)


def _candidate_deviation_penalty(
    candidate_traj: torch.Tensor | np.ndarray,
    reference_traj: torch.Tensor | np.ndarray | None,
) -> float:
    if reference_traj is None or candidate_traj is None:
        return 0.0
    if isinstance(candidate_traj, torch.Tensor):
        cand = candidate_traj.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        cand = np.asarray(candidate_traj, dtype=np.float32)
    ref = _future_to_4col(reference_traj)
    if cand.ndim != 2 or cand.shape[1] < 2:
        raise ValueError(f"candidate trajectory must be shaped (T,C), got {cand.shape}")
    T = min(cand.shape[0], ref.shape[0])
    if T < 1:
        return 0.0
    return float(np.linalg.norm(cand[:T, :2] - ref[:T, :2], axis=1).mean())


def _candidate_deviation_max_l2(
    candidate_traj: torch.Tensor | np.ndarray | None,
    reference_traj: torch.Tensor | np.ndarray | None,
) -> float:
    """Max per-step xy L2 between a candidate and the reference GT trajectory.

    Reuses ``rollout_gt_deviation`` (the canonical trajectory-deviation helper)
    with no threshold, so it stays consistent with the realized
    ``expert_disagreement`` deviation metric. Mirrors
    ``_candidate_deviation_penalty``'s null-handling.
    """
    if reference_traj is None or candidate_traj is None:
        return 0.0
    if isinstance(candidate_traj, torch.Tensor):
        cand = candidate_traj.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        cand = np.asarray(candidate_traj, dtype=np.float32)
    ref = _future_to_4col(reference_traj)
    if cand.ndim != 2 or cand.shape[1] < 2:
        raise ValueError(f"candidate trajectory must be shaped (T,C), got {cand.shape}")
    max_dev, _ = rollout_gt_deviation(
        torch.from_numpy(cand[:, :2]), torch.from_numpy(ref[:, :2]), threshold_m=None
    )
    return float(max_dev)


def _normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _r2lpl_state_class(source_row: dict[str, Any]) -> str:
    dev = float(source_row.get("expert_disagreement_max_dev", 0.0))
    if dev <= 1.0:
        return "near_log"
    if dev <= 5.0:
        return "recoverable"
    return "far_offpolicy"


def _r2lpl_weights(state_class: str) -> tuple[float, float]:
    if state_class == "near_log":
        return 0.20, 0.80
    if state_class == "recoverable":
        return 0.95, 0.05
    return 1.00, 0.00


def _apply_rear_end_collision_mode(rcfg, *, count_rear_end_collisions: bool) -> None:
    if count_rear_end_collisions:
        rcfg.ignore_rear_end_collisions = False


def _source_scene_t0_any_neighbor_overlap(
    data: dict[str, torch.Tensor],
    *,
    rcfg,
    device: torch.device,
    collision_thresh: float,
) -> tuple[bool, float]:
    clearance = current_ego_neighbor_clearance(
        data,
        rcfg,
        device=device,
        neighbor_kind="any",
    )
    distances = clearance["distances"]
    if distances.numel() == 0:
        return False, math.inf
    min_clearance = float(clearance["min_clearance"])
    return min_clearance <= collision_thresh, min_clearance


def _source_scene_t0_road_border_crossing(
    data: dict[str, torch.Tensor],
    rcfg,
    *,
    device: torch.device,
    rb_cross_thresh: float,
) -> tuple[bool, float]:
    ego_shape = _ego_shape_from_data(data, device)
    ego_now = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32, device=device)
    *_unused, per_timestep_min = compute_road_border_penalty(
        ego_now,
        ego_shape,
        data,
        config=rcfg,
    )
    min_dist = float(per_timestep_min[0, 0].item())
    return min_dist < rb_cross_thresh, min_dist


def _source_row_labels(row: dict[str, Any]) -> set[str]:
    return {str(label) for label in row.get("repair_labels") or row.get("labels") or []}


def _rows_require_static_collision(rows: list[dict[str, Any]]) -> bool:
    return any("static_collision" in _source_row_labels(row) for row in rows)


def _validate_static_collision_config(rows: list[dict[str, Any]], rcfg) -> None:
    if _rows_require_static_collision(rows) and not bool(
        getattr(rcfg, "static_collision_enabled", False)
    ):
        raise ValueError(
            "static_collision repair rows require reward_config.static_collision_enabled=true"
        )


def _event_group_key(row: dict[str, Any]) -> str:
    return str(row.get("window_dir") or row.get("event_key") or row.get("scene_path"))


def _drop_t0_dirty_event_windows(
    rows: list[dict[str, Any]],
    *,
    rcfg,
    thresholds: dict[str, float],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        key = _event_group_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    kept: list[dict[str, Any]] = []
    dirty: list[dict[str, Any]] = []
    for key in order:
        group = grouped[key]
        dirty_reason = None
        dirty_value = math.inf
        for row in group:
            labels = _source_row_labels(row)
            data = load_npz_data(row["scene_path"], device)
            if labels & {"moving_collision", "static_collision"}:
                collision_thresh = (
                    float(thresholds["moving_collision_thresh"])
                    if "moving_collision" in labels
                    else float(
                        thresholds.get("sc_cross_thresh", thresholds["moving_collision_thresh"])
                    )
                )
                hit, value = _source_scene_t0_any_neighbor_overlap(
                    data,
                    rcfg=rcfg,
                    device=device,
                    collision_thresh=collision_thresh,
                )
                if hit:
                    dirty_reason = "event_window_t0_already_collided"
                    dirty_value = value
                    break
            if "road_border_crossing" in labels:
                hit, value = _source_scene_t0_road_border_crossing(
                    data,
                    rcfg,
                    device=device,
                    rb_cross_thresh=float(thresholds["rb_cross_thresh"]),
                )
                if hit:
                    dirty_reason = "event_window_t0_already_road_border_crossing"
                    dirty_value = value
                    break
        if dirty_reason is None:
            kept.extend(group)
            continue
        for row in group:
            dirty.append({**row, "reason": dirty_reason, "t0_min_dist": float(dirty_value)})
        print(
            f"  DISCARD event_window={key}: reason={dirty_reason} "
            f"t0_min_dist={dirty_value:+.3f} rows={len(group)}"
        )
    return kept, dirty


def _load_rows(
    *,
    scene_rows_jsonl: Path | None,
    scenes_json: Path | None,
    allowed_labels: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if scene_rows_jsonl is not None:
        with open(scene_rows_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                labels = row.get("labels")
                if not labels:
                    label = row.get("label")
                    labels = [label] if label else []
                repair_labels = [str(label) for label in labels if str(label) in _REPAIRABLE_LABELS]
                if allowed_labels is not None:
                    repair_labels = [label for label in repair_labels if label in allowed_labels]
                if not repair_labels:
                    continue
                row = dict(row)
                row["repair_labels"] = repair_labels
                if "label" not in row:
                    row["label"] = repair_labels[0]
                rows.append(row)
        return rows

    if scenes_json is None:
        raise ValueError("one of --scene_rows_jsonl or --scenes is required")

    with open(scenes_json) as f:
        scene_paths = json.load(f)
    if not isinstance(scene_paths, list):
        raise ValueError(f"{scenes_json} must contain a JSON list of NPZ paths")
    for path in scene_paths:
        row = {
            "scene_path": str(path),
            "label": "static_collision",
            "labels": ["static_collision"],
            "repair_labels": ["static_collision"],
        }
        if allowed_labels is not None and "static_collision" not in allowed_labels:
            continue
        rows.append(row)
    return rows


def _passes_global_gates(label_row: dict[str, Any], reward_row) -> bool:
    return (
        reward_row.collision_step is None
        and not reward_row.rb_crossing
        and not reward_row.lane_crossing
        and not reward_row.static_crossing
        and not reward_row.kinematic_violated
        and label_row["moving_collision_step"] is None
    )


def _repairs_source_labels(
    source_labels: list[str],
    label_row: dict[str, Any],
    reward_row,
    *,
    min_static_margin: float,
) -> bool:
    # Paper-faithful recoverability (R2LPL Eq. recoverability): a candidate is valid iff
    # it is RULE-FEASIBLE (q_i > 0) — safety / drivable / route / static-margin. Expert /
    # GT likeness is NEVER a hard gate: the reference expresses expert agreement only as a
    # soft, deviation-class-weighted term (c_E, dropped entirely for far-off states,
    # w_E=0). So expert_disagreement is handled by the soft r2lpl_score ranking in
    # _best_safe_candidate, not by rejecting conflict-flagged candidates here.
    if not _passes_global_gates(label_row, reward_row):
        return False
    for label in source_labels:
        if label == "road_border_crossing" and reward_row.rb_crossing:
            return False
        if label == "static_collision":
            if reward_row.static_crossing or float(reward_row.sc_min_dist) < min_static_margin:
                return False
        if label == "moving_collision":
            if (
                reward_row.collision_step is not None
                or label_row["moving_collision_step"] is not None
            ):
                return False
    return True


def _best_safe_candidate(
    source_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    reward_rows,
    *,
    min_static_margin: float,
    target_gt_disagreement_thresh: float,
    candidate_trajs: list[torch.Tensor] | torch.Tensor | np.ndarray | None = None,
    reference_traj: torch.Tensor | np.ndarray | None = None,
) -> tuple[int | None, dict[str, Any]]:
    accepted: list[tuple[float, float, float, int]] = []
    candidate_traj_list = None
    if candidate_trajs is not None:
        candidate_traj_list = list(candidate_trajs)
    for idx, (label_row, reward_row) in enumerate(zip(candidate_rows, reward_rows, strict=True)):
        if _repairs_source_labels(
            source_row["repair_labels"],
            label_row,
            reward_row,
            min_static_margin=min_static_margin,
        ):
            violation_score = _candidate_violation_score(label_row, reward_row)
            deviation_penalty = _candidate_deviation_penalty(
                candidate_traj_list[idx] if candidate_traj_list is not None else None,
                reference_traj,
            )
            accepted.append((violation_score, deviation_penalty, float(reward_row.total), idx))

    if not accepted:
        best_total = max(float(r.total) for r in reward_rows)
        best_sc = max(float(getattr(r, "sc_min_dist", -99.0)) for r in reward_rows)
        return None, {
            "reason": "no_safe_candidate",
            "best_total": best_total,
            "best_sc_min_dist": best_sc,
        }

    uses_r2lpl_conflict_selection = "expert_disagreement" in set(
        source_row.get("repair_labels", [])
    )
    if uses_r2lpl_conflict_selection:
        rule_scores = _normalize_values([item[2] for item in accepted])
        state_class = _r2lpl_state_class(source_row)
        rule_weight, expert_weight = _r2lpl_weights(state_class)
        ranked: list[tuple[float, float, float, int]] = []
        for item, rule_score in zip(accepted, rule_scores, strict=True):
            violation_score, deviation_penalty, _reward_total, idx = item
            expert_score = math.exp(-max(deviation_penalty, 0.0) / 4.0)
            r2lpl_score = rule_weight * rule_score + expert_weight * expert_score
            ranked.append((-r2lpl_score, violation_score, deviation_penalty, idx))
        ranked.sort()
        neg_r2lpl_score, violation_score, deviation_penalty, idx = ranked[0]
        selected_r2lpl_score = -float(neg_r2lpl_score)
        selected_state_class = state_class
    else:
        accepted.sort(key=lambda item: (item[0], item[1], item[3]))
        violation_score, deviation_penalty, _reward_total, idx = accepted[0]
        selected_r2lpl_score = None
        selected_state_class = None
    reward_row = reward_rows[idx]
    label_row = candidate_rows[idx]

    # R2LPL hard-example signal: how far the SELECTED repaired target strays from
    # the ORIGINAL logged GT target. Distinct from realized `expert_disagreement`
    # (model rollout vs logged expert during simulation). deviation_penalty is
    # already mean-L2(selected, GT); reuse it and add the max-L2.
    selected_traj = candidate_traj_list[idx] if candidate_traj_list is not None else None
    target_gt_mean_l2 = float(deviation_penalty)
    target_gt_max_l2 = _candidate_deviation_max_l2(selected_traj, reference_traj)

    meta = {
        "selected_total": float(reward_row.total),
        "selected_sc_min_dist": float(getattr(reward_row, "sc_min_dist", 99.0)),
        "selected_rb_min_dist": float(getattr(reward_row, "rb_min_dist", 99.0)),
        "selected_labels": list(label_row["labels"]),
        "selected_violation_score": float(violation_score),
        "selected_deviation_penalty": float(deviation_penalty),
        "selected_candidate_index": int(idx),
        "target_gt_disagreement": bool(target_gt_max_l2 >= target_gt_disagreement_thresh),
        "target_gt_disagreement_mean_l2": target_gt_mean_l2,
        "target_gt_disagreement_max_l2": target_gt_max_l2,
    }
    if selected_r2lpl_score is not None:
        meta["selected_r2lpl_score"] = float(selected_r2lpl_score)
        meta["selected_r2lpl_state_class"] = selected_state_class
    return idx, meta


@torch.no_grad()
def build_repaired_targets(
    *,
    model_path: str,
    rows: list[dict[str, Any]],
    reward_config_path: str,
    threshold_config_path: str,
    ego_shape: np.ndarray | None,
    out_dir: Path,
    out_rows_jsonl: Path | None,
    out_list: Path,
    min_static_margin: float,
    K: int,
    variant: str,
    generation_mode: str,
    grpo_noise_scale: float,
    gt_max_speed: float,
    scene_batch_size: int,
    noise_low: float,
    noise_high: float,
    device: torch.device,
    enable_conflict_detector: bool,
    use_route_cl_guidance: bool,
    count_rear_end_collisions: bool,
    allow_empty: bool = False,
    target_gt_disagreement_thresh: float = 2.0,
) -> tuple[list[str], list[dict[str, Any]]]:
    rcfg = load_reward_config(reward_config_path)
    _apply_rear_end_collision_mode(
        rcfg,
        count_rear_end_collisions=count_rear_end_collisions,
    )
    _validate_static_collision_config(rows, rcfg)
    thresholds = _load_scene_thresholds(threshold_config_path)
    model, model_args = load_model(model_path, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    repaired_rows: list[dict[str, Any]] = []
    unrepaired_rows: list[dict[str, Any]] = []
    rows, dirty_rows = _drop_t0_dirty_event_windows(
        rows,
        rcfg=rcfg,
        thresholds=thresholds,
        device=device,
    )
    unrepaired_rows.extend(dirty_rows)

    class _Args:
        pass

    cls_args = _Args()
    cls_args.enable_conflict_detector = enable_conflict_detector
    cls_args.expert_disagreement_wait_speed_mps = thresholds["expert_disagreement_wait_speed_mps"]
    cls_args.expert_disagreement_wait_progress_m = thresholds["expert_disagreement_wait_progress_m"]
    cls_args.expert_disagreement_forward_progress_gap_m = thresholds[
        "expert_disagreement_forward_progress_gap_m"
    ]
    cls_args.expert_disagreement_lag_progress_gap_m = thresholds[
        "expert_disagreement_lag_progress_gap_m"
    ]
    cls_args.expert_disagreement_moving_speed_mps = thresholds[
        "expert_disagreement_moving_speed_mps"
    ]

    for start in range(0, len(rows), scene_batch_size):
        batch_rows = rows[start : start + scene_batch_size]
        datas = []
        kept_rows = []
        for row in batch_rows:
            p = row["scene_path"]
            data = load_npz_data(p, device)
            npz_es = data["ego_shape"].detach().cpu().numpy().reshape(-1)[:3]
            if ego_shape is not None and not np.allclose(npz_es, ego_shape, atol=1e-2):
                raise ValueError(
                    f"{p}: --ego_shape {ego_shape.tolist()} != NPZ ego_shape "
                    f"{npz_es.tolist()} (platform mismatch)"
                )
            datas.append(data)
            kept_rows.append(row)

        if not datas:
            continue

        norm_batch = _normalize_batch(_stack_scene_data(datas, device), model_args)
        if generation_mode == _GENERATION_MODE_GRPO_TEMPERATURE:
            trajs = _generate_grpo_temperature_scenes(
                model,
                norm_batch,
                K=K,
                grpo_noise_scale=grpo_noise_scale,
                device=device,
            )
        elif generation_mode == _GENERATION_MODE_GUIDED_VARIANT:
            trajs = generate_all_scenes_batched(
                model,
                model_args,
                norm_batch,
                K=K,
                noise_range=(noise_low, noise_high),
                device=device,
                gen_chunk_size=K,
                gt_max_speed=gt_max_speed,
                generation_variant=variant,
                use_route_cl_guidance=use_route_cl_guidance,
            )
        else:
            raise ValueError(
                f"unknown generation_mode {generation_mode!r}; expected "
                f"{_GENERATION_MODE_GRPO_TEMPERATURE!r} or {_GENERATION_MODE_GUIDED_VARIANT!r}"
            )
        scene_paths = [str(row["scene_path"]) for row in kept_rows]
        candidate_rows_per_scene = classify_loaded_scene_candidates_batch(
            scene_paths,
            trajs,
            datas,
            rcfg,
            moving_collision_thresh=thresholds["moving_collision_thresh"],
            moving_near_thresh=thresholds["moving_near_thresh"],
            static_near_thresh=thresholds["static_near_thresh"],
            rb_near_thresh=thresholds["rb_near_thresh"],
            device=device,
            args=cls_args,
        )

        for row, data, scene_trajs, candidate_rows in zip(
            kept_rows, datas, trajs, candidate_rows_per_scene, strict=True
        ):
            reward_rows = compute_reward_batch(scene_trajs, data, rcfg)
            reference_traj = _future_to_4col(data["ego_agent_future"].detach().cpu().numpy())
            best_idx, meta = _best_safe_candidate(
                row,
                candidate_rows,
                reward_rows,
                min_static_margin=min_static_margin,
                target_gt_disagreement_thresh=target_gt_disagreement_thresh,
                candidate_trajs=scene_trajs,
                reference_traj=reference_traj,
            )
            name = _output_name_for_scene(row["scene_path"])
            if best_idx is None:
                unrepaired_rows.append({**row, **meta})
                print(
                    f"  UNREPAIRED {name}: labels={','.join(row['repair_labels'])} "
                    f"best_total={meta['best_total']:+.1f}"
                )
                continue

            with np.load(row["scene_path"], allow_pickle=True) as loaded:
                raw = _filtered_npz_payload(loaded)
            raw["ego_agent_future"] = _future4_to_3col(
                scene_trajs[best_idx].detach().cpu().numpy().astype(np.float32)
            )
            out_path = out_dir / name
            np.savez(out_path, **raw)

            repaired = dict(row)
            repaired["source_scene_path"] = str(row["scene_path"])
            repaired["scene_path"] = str(out_path)
            # Per-scene ego_shape from THIS scene's data — not the batch-loop
            # leftover npz_es (which is the last validation-loop scene).
            scene_es = data["ego_shape"].detach().cpu().numpy().reshape(-1)[:3]
            repaired["ego_shape"] = [float(x) for x in scene_es.tolist()]
            repaired.update(meta)
            repaired_rows.append(repaired)
            print(
                f"  repaired {name}: labels={','.join(row['repair_labels'])} "
                f"slot={meta['selected_candidate_index']} total={meta['selected_total']:+.1f}"
            )

    if not repaired_rows and not allow_empty:
        raise RuntimeError(
            "No repaired targets were produced; refusing to emit an empty training set"
        )

    out_list.parent.mkdir(parents=True, exist_ok=True)
    repaired_paths = [row["scene_path"] for row in repaired_rows]
    out_list.write_text(json.dumps(repaired_paths, indent=2))
    if out_rows_jsonl is not None:
        out_rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_rows_jsonl, "w") as f:
            for row in repaired_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if unrepaired_rows:
        unrepaired_path = out_list.with_name(out_list.stem + "_unrepaired.json")
        unrepaired_path.write_text(json.dumps(unrepaired_rows, indent=2))
        print(f"  unrepaired list -> {unrepaired_path}")

    print(f"repaired {len(repaired_rows)}/{len(rows)} -> {out_dir}")
    return repaired_paths, unrepaired_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="generation source model")
    ap.add_argument("--config", required=True, help="reward config JSON")
    ap.add_argument("--threshold_config", help="scene threshold config JSON")
    ap.add_argument(
        "--ego_shape",
        required=True,
        help="WB,L,W to validate every NPZ, or 'from_npz' for mixed-platform scene lists",
    )
    ap.add_argument("--scene_rows_jsonl", help="JSONL with mined scene rows and labels")
    ap.add_argument("--scenes", help="JSON list of NPZs to repair")
    ap.add_argument("--labels", help="comma-separated subset of labels to repair")
    ap.add_argument("--min_margin", type=float, required=True, help="required static clearance")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--out_rows_jsonl", help="write repaired metadata rows")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--variant", default="rl_cl_soft_sweep_stretch")
    ap.add_argument(
        "--generation_mode",
        choices=[_GENERATION_MODE_GRPO_TEMPERATURE, _GENERATION_MODE_GUIDED_VARIANT],
        default=_GENERATION_MODE_GRPO_TEMPERATURE,
        help="candidate generator: exact GRPO temperature sampler, or guided variant registry",
    )
    ap.add_argument(
        "--grpo_noise_scale",
        type=float,
        default=3.0,
        help="max initial-noise temperature for --generation_mode grpo_temperature; "
        "matches train_grpo_predictor.py default",
    )
    ap.add_argument("--gt_max_speed", type=float, default=9.0)
    ap.add_argument("--scene_batch_size", type=int, default=8)
    ap.add_argument("--noise_low", type=float, default=0.5)
    ap.add_argument("--noise_high", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enable_conflict_detector", action="store_true")
    ap.add_argument("--disable_route_cl_guidance", action="store_true")
    ap.add_argument(
        "--count_rear_end_collisions",
        action="store_true",
        help="Count moving collisions where the ego is struck from behind.",
    )
    ap.add_argument(
        "--allow_empty",
        action="store_true",
        help="Write empty outputs instead of failing when this shard has no accepted targets.",
    )
    ap.add_argument(
        "--target_gt_disagreement_thresh",
        type=float,
        default=2.0,
        help="max-L2 (m) between the selected repaired target and the logged GT target above "
        "which target_gt_disagreement is flagged (R2LPL hard-example signal).",
    )
    args = ap.parse_args()

    if not args.scene_rows_jsonl and not args.scenes:
        raise ValueError("one of --scene_rows_jsonl or --scenes is required")
    if args.scene_rows_jsonl and not args.threshold_config:
        raise ValueError("--threshold_config is required with --scene_rows_jsonl")
    if args.scenes and not args.threshold_config:
        raise ValueError("--threshold_config is required")

    allowed_labels = None
    if args.labels:
        allowed_labels = {label.strip() for label in args.labels.split(",") if label.strip()}
        unknown = allowed_labels - _REPAIRABLE_LABELS
        if unknown:
            raise ValueError(f"unsupported repair labels: {sorted(unknown)}")

    rows = _load_rows(
        scene_rows_jsonl=Path(args.scene_rows_jsonl) if args.scene_rows_jsonl else None,
        scenes_json=Path(args.scenes) if args.scenes else None,
        allowed_labels=allowed_labels,
    )
    if not rows:
        if args.allow_empty:
            out_list = Path(args.out_list)
            out_list.parent.mkdir(parents=True, exist_ok=True)
            out_list.write_text("[]")
            if args.out_rows_jsonl:
                out_rows = Path(args.out_rows_jsonl)
                out_rows.parent.mkdir(parents=True, exist_ok=True)
                out_rows.write_text("")
            return
        raise RuntimeError("No repairable scene rows remained after label filtering")

    build_repaired_targets(
        model_path=args.model,
        rows=rows,
        reward_config_path=args.config,
        threshold_config_path=args.threshold_config,
        ego_shape=_parse_ego_shape(args.ego_shape),
        out_dir=Path(args.out_dir),
        out_rows_jsonl=Path(args.out_rows_jsonl) if args.out_rows_jsonl else None,
        out_list=Path(args.out_list),
        min_static_margin=float(args.min_margin),
        K=int(args.K),
        variant=str(args.variant),
        generation_mode=str(args.generation_mode),
        grpo_noise_scale=float(args.grpo_noise_scale),
        gt_max_speed=float(args.gt_max_speed),
        scene_batch_size=int(args.scene_batch_size),
        noise_low=float(args.noise_low),
        noise_high=float(args.noise_high),
        device=torch.device(args.device),
        enable_conflict_detector=bool(args.enable_conflict_detector),
        use_route_cl_guidance=not bool(args.disable_route_cl_guidance),
        count_rear_end_collisions=bool(args.count_rear_end_collisions),
        allow_empty=bool(args.allow_empty),
        target_gt_disagreement_thresh=float(args.target_gt_disagreement_thresh),
    )


if __name__ == "__main__":
    main()
