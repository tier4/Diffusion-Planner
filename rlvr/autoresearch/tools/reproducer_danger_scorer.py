from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from rlvr.autoresearch.tools.classify_scene_failures import (
    _apply_scene_thresholds,
    _slice_scene_data,
    classify_loaded_scenes_batch,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from scenario_generation.reproducer_rollout import _route_key


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
        B = len(built)
        datas = [_slice_scene_data(data, i, B) for i in range(B)]
        scene_paths = [f"{_route_key(s.tl)}_{idx:08d}" for s, _np, _nb, idx, *_ in built]
        ego = torch.as_tensor(preds[:, None, :, :4], dtype=torch.float32, device=torch_device)
        rows = classify_loaded_scenes_batch(
            scene_paths,
            ego,
            datas,
            reward_cfg,
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
