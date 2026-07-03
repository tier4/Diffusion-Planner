from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
    compute_road_border_penalty,
)
from rlvr.autoresearch.tools.classify_scene_failures import (
    _apply_scene_thresholds,
    _ego_shape_from_data,
    _first_moving_collision_step,
    _neighbor_inputs,
    _stopped_neighbor_mask,
    classify_loaded_scenes_batch,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from scenario_generation.reproducer_rollout import _route_key

_SUPPORTED_REALIZED_EVENT_LABELS = frozenset({"moving_collision", "road_border_crossing"})


def load_credit_windows(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    with open(path) as f:
        raw = json.load(f)
    out = {str(k): int(v) for k, v in raw.items() if not str(k).startswith("_")}
    negative = [k for k, v in out.items() if v < 0]
    if negative:
        raise ValueError(f"credit-window widths must be >=0 for labels: {negative}")
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
):
    reward_cfg = load_reward_config(reward_config)
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
):
    allowed = set(allowed_labels) if allowed_labels else set(_SUPPORTED_REALIZED_EVENT_LABELS)
    unsupported = sorted(allowed - _SUPPORTED_REALIZED_EVENT_LABELS)
    if unsupported:
        raise ValueError(
            "realized-event verification currently supports only "
            f"{sorted(_SUPPORTED_REALIZED_EVENT_LABELS)}; got unsupported labels {unsupported}"
        )

    reward_cfg = load_reward_config(reward_config)
    scorer_args = SimpleNamespace(
        threshold_config=threshold_config,
        moving_near_thresh=None,
        static_near_thresh=None,
        rb_near_thresh=None,
        sc_cross_thresh=None,
        rb_cross_thresh=None,
        enable_conflict_detector=False,
    )
    thresholds = _apply_scene_thresholds(reward_cfg, scorer_args)
    torch_device = torch.device(device)
    rb_cross_thresh = float(thresholds["rb_cross_thresh"])
    moving_collision_thresh = float(thresholds["moving_collision_thresh"])

    def _scorer(np_dict: dict[str, Any], *, collided: bool) -> dict[str, Any]:
        labels: list[str] = []
        row: dict[str, Any] = {
            "trajectory_source": "reproducer_realized",
            "realized_collision": bool(collided),
        }
        tensors = _np_dict_to_scoring_tensors(np_dict, device=torch_device)

        if "moving_collision" in allowed:
            ego_now = torch.tensor(
                [[[0.0, 0.0, 1.0, 0.0]]],
                dtype=torch.float32,
                device=torch_device,
            )
            ego_shape = _ego_shape_from_data(tensors, torch_device)
            neighbor_futures, neighbor_shapes, neighbor_valid = _neighbor_inputs(
                tensors, 1, torch_device
            )
            stopped_mask = _stopped_neighbor_mask(neighbor_futures, neighbor_valid, reward_cfg)
            moving_mask = ~stopped_mask
            row["moving_collision_step"] = None
            row["moving_min_dist"] = float("inf")
            if bool(moving_mask.any().item()):
                current_neighbors = neighbor_futures[moving_mask, :1, :4].clone()
                neighbor_past = tensors.get("neighbor_agents_past")
                if neighbor_past is not None:
                    if neighbor_past.dim() == 4:
                        neighbor_past = neighbor_past[0]
                    future_all = tensors.get("neighbor_agents_future")
                    if future_all is not None:
                        if future_all.dim() == 4:
                            future_all = future_all[0]
                        slot_valid = future_all[:, :1, :2].abs().sum(dim=(1, 2)) > 1e-6
                        filtered_past = neighbor_past[slot_valid]
                    else:
                        filtered_past = neighbor_past
                    current_neighbors[:, 0, :4] = filtered_past[moving_mask, -1, :4]
                current_valid = (current_neighbors[:, :, :2].abs().sum(dim=-1) > 1e-6).to(
                    torch.bool
                )
                distances = compute_ego_neighbor_signed_clearance(
                    ego_now,
                    ego_shape,
                    current_neighbors,
                    neighbor_shapes[moving_mask],
                    current_valid,
                )
                if distances.numel():
                    row["moving_min_dist"] = float(distances.min().item())
                    row["moving_collision_step"] = _first_moving_collision_step(
                        distances,
                        moving_collision_thresh=moving_collision_thresh,
                    )
                    if row["moving_collision_step"] is not None:
                        labels.append("moving_collision")

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

        row["labels"] = labels or ["clean"]
        row["label"] = labels[0] if labels else "clean"
        return row

    return _scorer
