"""Build wandb log payloads for grouped closed-loop validation."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import wandb

from scenario_generation.metrics.group_report import WANDB_EXCLUDED_SCALAR_KEYS

RESULTS_TABLE_COLUMNS = [
    "area_name",
    "bag",
    "span_index",
    "segment",
    "n_steps_run",
    "centerline_mean_m",
    "centerline_p95_m",
    "turn_match_rate",
    "neighbor_violation_steps",
    "rb_violation_steps",
    "collision_steps",
    "n_collision_steps",
    "n_near_miss_steps",
    "min_clearance_m",
    "mean_clearance",
    "video_path",
]


def build_grouped_closed_loop_wandb_log(summary: dict) -> dict:
    """Per-area scalars, validation totals, results table, and episode videos."""
    log: dict = {"closed_loop/mode": "grouped"}
    log["closed_loop/grouped/elapsed_sec"] = float(summary.get("elapsed_sec", 0.0))

    timing = summary.get("timing")
    if timing:
        stages = timing.get("stages") or {}
        for key, wandb_key in (
            ("model_forward", "model_forward_total_s"),
            ("draw", "draw_total_s"),
            ("timeline_load_npz", "timeline_load_npz_total_s"),
        ):
            stage = stages.get(key)
            if stage:
                log[f"closed_loop/profile/{wandb_key}"] = float(stage["total_s"])
        log["closed_loop/profile/model_forward_calls"] = int(timing.get("model_forward_calls", 0))
        log["closed_loop/profile/total_sim_steps"] = int(timing.get("total_sim_steps", 0))
        log["closed_loop/profile/model_forward_rate"] = float(timing.get("model_forward_rate", 0.0))

    agg = summary.get("grouped_summary") or {}
    _log_scalar_group(log, "closed_loop/grouped/total", agg.get("totals") or {})

    for area, stats in (agg.get("by_area_name") or {}).items():
        safe_area = area.replace("/", "_")
        _log_scalar_group(log, f"closed_loop/grouped/area/{safe_area}", stats)

    rows = summary.get("segments") or []
    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in RESULTS_TABLE_COLUMNS if c in df.columns]
        extra = [
            c for c in df.columns if c not in cols and c not in ("labeled_ranges", "metric_group")
        ]
        log["closed_loop/grouped/results_table"] = wandb.Table(dataframe=df[cols + extra])

    for mp4 in summary.get("video_mp4s") or []:
        mp4_path = Path(mp4)
        key = _video_wandb_key(mp4_path)
        log[key] = wandb.Video(str(mp4_path), format="mp4")

    return log


def _log_scalar_group(log: dict, prefix: str, stats: dict) -> None:
    for key, val in stats.items():
        if key in WANDB_EXCLUDED_SCALAR_KEYS:
            continue
        if not _wandb_scalar(val):
            continue
        log[f"{prefix}/{key}"] = val


def build_full_closed_loop_wandb_log(summary: dict) -> dict:
    """Legacy full-route closed-loop wandb payload."""
    scalar_keys = [
        "collision_segment_rate",
        "collision_step_rate",
        "near_miss_segment_rate",
        "near_miss_step_rate",
        "global_min_clearance",
        "mean_segment_min_clearance",
        "mean_segment_mean_clearance",
        "total_collision_steps",
        "total_near_miss_steps",
        "total_snaps",
        "total_steps",
    ]
    log = {"closed_loop/mode": "full"}
    for key in scalar_keys:
        val = summary.get(key)
        if _wandb_scalar(val):
            log[f"closed_loop/{key}"] = val
    for mp4 in summary.get("video_mp4s") or []:
        log[f"closed_loop/video/{Path(mp4).stem}"] = wandb.Video(str(mp4), format="mp4")
    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, bool)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False


def _video_wandb_key(mp4_path: Path) -> str:
    parts = mp4_path.parts
    if "videos" in parts:
        idx = parts.index("videos")
        rel_parts = list(parts[idx + 1 :])
        # videos/<metric_group>/<area>/file.mp4 -> area/file for stable area-level keys
        if len(rel_parts) >= 2:
            rel_parts = rel_parts[1:]
        rel = "/".join(rel_parts)
        return f"closed_loop/grouped/video/{rel.replace('/', '_')}"
    return f"closed_loop/grouped/video/{mp4_path.stem}"
