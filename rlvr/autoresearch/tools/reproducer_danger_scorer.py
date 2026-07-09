from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from planner_metrics.subscores import (
    compute_road_border_penalty,
)
from rlvr.autoresearch.tools.classify_scene_failures import (
    _apply_scene_thresholds,
    _ego_shape_from_data,
    _moving_collision_all_rear_end,
    _moving_collision_step_gated,
    classify_loaded_scenes_batch,
    current_ego_neighbor_clearance,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from scenario_generation.conflict_detector import detect_expert_disagreement
from scenario_generation.reproducer_rollout import _route_key

_SUPPORTED_REALIZED_EVENT_LABELS = frozenset(
    {"moving_collision", "static_collision", "road_border_crossing", "expert_disagreement"}
)


def load_credit_windows(path: Path | None) -> dict[str, dict[str, int | float]] | None:
    if path is None:
        return None
    with open(path) as f:
        raw = json.load(f)
    frame_hz = raw.get("_frame_hz")
    defaults = raw.get("_defaults")
    if not isinstance(frame_hz, int | float) or float(frame_hz) <= 0:
        raise ValueError(f"{path}: _frame_hz must be a positive number")
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: _defaults must define width_s and gap_s")

    def _seconds_to_frames(label: str, field: str, value: object) -> tuple[float, int]:
        if not isinstance(value, int | float):
            raise ValueError(f"{path}: {label}.{field} must be a number of seconds")
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"{path}: {label}.{field} must be >= 0, got {seconds}")
        frames = int(round(seconds * float(frame_hz)))
        return seconds, frames

    out: dict[str, dict[str, int | float]] = {}
    for key, value in raw.items():
        if str(key).startswith("_"):
            continue
        label = str(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"{path}: {label} must be an object with width_s/gap_s; scalar "
                "frame counts are not supported"
            )
        width_s, width_frames = _seconds_to_frames(
            label,
            "width_s",
            value.get("width_s", defaults.get("width_s")),
        )
        gap_s, gap_frames = _seconds_to_frames(
            label,
            "gap_s",
            value.get("gap_s", defaults.get("gap_s")),
        )
        if width_frames < 1:
            raise ValueError(f"{path}: {label}.width_s must round to at least 1 frame")
        out[label] = {
            "width_s": width_s,
            "gap_s": gap_s,
            "width_frames": width_frames,
            "gap_frames": gap_frames,
        }
    return out


def _np_dict_to_scoring_tensors(
    np_dict: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in np_dict.items():
        array = np.asarray(value)
        if key in {"lanes_has_speed_limit", "route_lanes_has_speed_limit"}:
            out[key] = torch.as_tensor(array, dtype=torch.bool, device=device)
        elif key in {"turn_indicators", "delay"}:
            out[key] = torch.as_tensor(array, dtype=torch.long, device=device)
        else:
            out[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
    if "delay" not in out:
        out["delay"] = torch.zeros((1,), dtype=torch.long, device=device)
    return out


def build_reproducer_danger_scorer(
    *,
    reward_config: Path,
    threshold_config: Path,
    device: str,
    enable_conflict_detector: bool = False,
    allowed_labels: set[str] | None = None,
    count_rear_end_collisions: bool | None = None,
):
    reward_cfg = load_reward_config(reward_config)
    if count_rear_end_collisions is not None:
        # Keep mining rear-end consistent with the repair side and the RAW-col
        # event trigger: --count_rear_end_collisions -> do NOT ignore them.
        reward_cfg.ignore_rear_end_collisions = not count_rear_end_collisions
    scorer_args = SimpleNamespace(
        threshold_config=threshold_config,
        moving_near_thresh=None,
        static_near_thresh=None,
        rb_near_thresh=None,
        sc_cross_thresh=None,
        rb_cross_thresh=None,
        enable_conflict_detector=bool(enable_conflict_detector),
    )
    thresholds = _apply_scene_thresholds(reward_cfg, scorer_args)
    torch_device = torch.device(device)
    allowed = set(allowed_labels) if allowed_labels else None

    def _scorer(built, preds, data, _device) -> list[dict[str, Any]]:
        datas = [
            _np_dict_to_scoring_tensors(np_dict, device=torch_device)
            for _s, np_dict, *_rest in built
        ]
        scene_paths = [f"{_route_key(s.tl)}_{idx:08d}" for s, _np, _nb, idx, *_ in built]
        ego = torch.as_tensor(preds[:, None, :, :4], dtype=torch.float32, device=torch_device)
        rows = classify_loaded_scenes_batch(
            scene_paths,
            ego,
            datas,
            reward_cfg,
            moving_collision_thresh=float(thresholds["moving_collision_thresh"]),
            moving_near_thresh=float(thresholds["moving_near_thresh"]),
            static_near_thresh=float(thresholds["static_near_thresh"]),
            rb_near_thresh=float(thresholds["rb_near_thresh"]),
            device=torch_device,
            args=scorer_args,
        )
        for row in rows:
            row["trajectory_source"] = "reproducer_det"
            if allowed is not None:
                labels = [label for label in row.get("labels", []) if label in allowed]
                row["labels"] = labels or ["clean"]
        return rows

    return _scorer


def build_realized_event_scorer(
    *,
    reward_config: Path,
    threshold_config: Path,
    device: str,
    allowed_labels: set[str] | None = None,
    count_rear_end_collisions: bool | None = None,
):
    allowed = set(allowed_labels) if allowed_labels else set(_SUPPORTED_REALIZED_EVENT_LABELS)
    unsupported = sorted(allowed - _SUPPORTED_REALIZED_EVENT_LABELS)
    if unsupported:
        raise ValueError(
            "realized-event verification currently supports only "
            f"{sorted(_SUPPORTED_REALIZED_EVENT_LABELS)}; got unsupported labels {unsupported}"
        )

    reward_cfg = load_reward_config(reward_config)
    if count_rear_end_collisions is not None:
        # See build_reproducer_danger_scorer: keep realized-event moving-collision
        # detection consistent with the repair side / RAW-col trigger.
        reward_cfg.ignore_rear_end_collisions = not count_rear_end_collisions
    scorer_args = SimpleNamespace(
        threshold_config=threshold_config,
        moving_near_thresh=None,
        static_near_thresh=None,
        rb_near_thresh=None,
        sc_cross_thresh=None,
        rb_cross_thresh=None,
        enable_conflict_detector=False,
        expert_disagreement_wait_speed_mps=None,
        expert_disagreement_wait_progress_m=None,
        expert_disagreement_forward_progress_gap_m=None,
        expert_disagreement_lag_progress_gap_m=None,
        expert_disagreement_moving_speed_mps=None,
    )
    thresholds = _apply_scene_thresholds(reward_cfg, scorer_args)
    torch_device = torch.device(device)
    rb_cross_thresh = float(thresholds["rb_cross_thresh"])
    static_collision_thresh = float(thresholds["sc_cross_thresh"])
    moving_collision_thresh = float(thresholds["moving_collision_thresh"])
    expert_state_by_segment: dict[Any, dict[str, Any]] = {}

    def _scorer(
        np_dict: dict[str, Any],
        *,
        collided: bool,
        live_pose: np.ndarray | None = None,
        gt_pose: np.ndarray | None = None,
        segment_key: object | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        labels: list[str] = []
        row: dict[str, Any] = {
            "trajectory_source": "reproducer_realized",
            "realized_collision": bool(collided),
        }
        tensors = _np_dict_to_scoring_tensors(np_dict, device=torch_device)

        if "moving_collision" in allowed:
            moving_clearance = current_ego_neighbor_clearance(
                tensors,
                reward_cfg,
                device=torch_device,
                neighbor_kind="moving",
            )
            row["moving_collision_step"] = None
            row["moving_min_dist"] = float("inf")
            row["rear_end_collision"] = False
            distances = moving_clearance["distances"]
            if distances.numel():
                row["moving_min_dist"] = float(moving_clearance["min_clearance"])
                # Unified gated collision definition (overlap + rear-end
                # suppression per reward_cfg), NOT a raw proximity threshold.
                row["moving_collision_step"] = _moving_collision_step_gated(
                    moving_clearance["ego_now"],
                    moving_clearance["ego_shape"],
                    moving_clearance["neighbors"],
                    moving_clearance["neighbor_shapes"],
                    moving_clearance["neighbor_valid"],
                    reward_cfg,
                )
                if row["moving_collision_step"] is not None:
                    labels.append("moving_collision")
                    # Keep rear-ends but TAG them, so downstream can drop them if
                    # desired instead of losing the collision at detection time.
                    row["rear_end_collision"] = _moving_collision_all_rear_end(
                        moving_clearance["ego_now"],
                        moving_clearance["ego_shape"],
                        moving_clearance["neighbors"],
                        moving_clearance["neighbor_shapes"],
                        moving_clearance["neighbor_valid"],
                    )

        if "static_collision" in allowed:
            static_clearance = current_ego_neighbor_clearance(
                tensors,
                reward_cfg,
                device=torch_device,
                neighbor_kind="static",
            )
            row["static_collision_step"] = None
            row["static_min_dist"] = 99.0
            row["stopped_neighbor_count"] = int(static_clearance["stopped_mask"].sum().item())
            if static_clearance["distances"].numel():
                row["static_min_dist"] = float(static_clearance["min_clearance"])
                if row["static_min_dist"] <= static_collision_thresh:
                    row["static_collision_step"] = 0
                    labels.append("static_collision")

        if "road_border_crossing" in allowed:
            ego_shape = tensors["ego_shape"].reshape(-1)[:3]
            ego_now = torch.tensor(
                [[[0.0, 0.0, 1.0, 0.0]]],
                dtype=torch.float32,
                device=torch_device,
            )
            _gate, _near, _wide, _steps, _cont, per_timestep_min = compute_road_border_penalty(
                ego_now,
                ego_shape,
                tensors,
                reward_cfg,
            )
            rb_min_dist = float(per_timestep_min[0, 0].item())
            row["rb_min_dist"] = rb_min_dist
            if rb_min_dist < rb_cross_thresh:
                labels.append("road_border_crossing")

        if "expert_disagreement" in allowed:
            row["expert_disagreement"] = False
            row["expert_disagreement_step"] = None
            row["expert_disagreement_max_dev"] = 0.0
            row["expert_disagreement_reason"] = ""
            row["expert_disagreement_model_end_progress"] = 0.0
            row["expert_disagreement_expert_end_progress"] = 0.0
            row["expert_disagreement_model_end_speed"] = 0.0
            row["expert_disagreement_expert_end_speed"] = 0.0
            if live_pose is not None and gt_pose is not None:
                key = segment_key if segment_key is not None else "__single_segment__"
                state = expert_state_by_segment.setdefault(
                    key,
                    {
                        "max_dev": 0.0,
                        "live_xy": [],
                        "gt_xy": [],
                    },
                )
                live_xy = np.asarray(live_pose, dtype=np.float32)[:2]
                gt_xy = np.asarray(gt_pose, dtype=np.float32)[:2]
                state["live_xy"].append(live_xy)
                state["gt_xy"].append(gt_xy)
                dev = float(np.linalg.norm(live_xy - gt_xy))
                state["max_dev"] = max(float(state["max_dev"]), dev)
                rollout = torch.as_tensor(
                    np.asarray(state["live_xy"], dtype=np.float32),
                    dtype=torch.float32,
                    device=torch_device,
                )
                expert = torch.as_tensor(
                    np.asarray(state["gt_xy"], dtype=np.float32),
                    dtype=torch.float32,
                    device=torch_device,
                )
                result = detect_expert_disagreement(
                    rollout,
                    expert,
                    enabled=True,
                    wait_speed_mps=float(thresholds["expert_disagreement_wait_speed_mps"]),
                    wait_progress_m=float(thresholds["expert_disagreement_wait_progress_m"]),
                    forward_progress_gap_m=float(
                        thresholds["expert_disagreement_forward_progress_gap_m"]
                    ),
                    lag_progress_gap_m=float(thresholds["expert_disagreement_lag_progress_gap_m"]),
                    moving_speed_mps=float(thresholds["expert_disagreement_moving_speed_mps"]),
                )
                row["expert_disagreement_max_dev"] = float(state["max_dev"])
                row["expert_disagreement_reason"] = result.reason
                row["expert_disagreement_model_end_progress"] = result.model_end_progress
                row["expert_disagreement_expert_end_progress"] = result.expert_end_progress
                row["expert_disagreement_model_end_speed"] = result.model_end_speed
                row["expert_disagreement_expert_end_speed"] = result.expert_end_speed
                if result.expert_disagreement:
                    row["expert_disagreement"] = True
                    row["expert_disagreement_step"] = step
                    labels.append("expert_disagreement")

        row["labels"] = labels or ["clean"]
        row["label"] = labels[0] if labels else "clean"
        return row

    return _scorer
