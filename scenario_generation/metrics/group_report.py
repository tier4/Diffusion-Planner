"""Aggregate per-segment metric rows into summary tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

# Keys rolled up with sum() across segment rows (validation / per-area totals).
_SUM_KEYS = (
    "n_steps_run",
    "neighbor_violation_steps",
    "rb_violation_steps",
    "collision_steps",
    "n_collision_steps",
    "n_near_miss_steps",
    "n_snaps",
)

# Keys excluded from wandb scalar comparison (episode/video counts vary by dataset).
WANDB_EXCLUDED_SCALAR_KEYS = frozenset({"n_segments", "n_segments_total", "n_episodes", "n_videos"})


def aggregate_segment_rows(rows: list[dict]) -> dict:
    """Build per-area and validation-wide totals from segment rows."""
    by_area: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_area[row["area_name"]].append(row)

    return {
        "by_area_name": {k: _agg_segment_stats(v) for k, v in sorted(by_area.items())},
        "totals": _agg_segment_stats(rows),
    }


def _agg_segment_stats(items: list[dict]) -> dict:
    if not items:
        return {}

    def _sum(key: str) -> int:
        return int(sum(r.get(key, 0) or 0 for r in items))

    def _weighted_mean(rate_key: str, weight_key: str = "n_steps_run") -> float | None:
        num = 0.0
        den = 0.0
        for row in items:
            rate = row.get(rate_key)
            if rate is None or rate != rate:
                continue
            weight = float(row.get(weight_key) or 0)
            if weight <= 0:
                continue
            num += float(rate) * weight
            den += weight
        return float(num / den) if den > 0 else None

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in items if r.get(key) is not None and r[key] == r[key]]
        return float(sum(vals) / len(vals)) if vals else None

    finite_min_cl = [
        r["min_clearance_m"]
        for r in items
        if r.get("min_clearance_m") is not None and r["min_clearance_m"] == r["min_clearance_m"]
    ]

    out: dict = {key: _sum(key) for key in _SUM_KEYS}
    out["centerline_mean_m"] = _weighted_mean("centerline_mean_m")
    out["centerline_p95_m"] = _mean("centerline_p95_m")
    out["turn_match_rate"] = _weighted_mean("turn_match_rate")
    out["min_clearance_m"] = float(min(finite_min_cl)) if finite_min_cl else None
    return out


RESULTS_TABLE_COLUMNS = [
    "metric_group",
    "area_name",
    "sequence",
    "bag",
    "span_index",
    "list_start_idx",
    "list_end_idx",
    "segment",
    "n_steps_run",
    "terminated",
    "centerline_mean_m",
    "centerline_p95_m",
    "centerline_max_m",
    "turn_match_rate",
    "neighbor_violation_steps",
    "rb_violation_steps",
    "collision_steps",
    "n_collision_steps",
    "n_near_miss_steps",
    "min_clearance",
    "min_clearance_m",
    "mean_clearance",
    "min_rb_dist_m",
    "n_snaps",
    "video_path",
]


def write_results_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    extra = {k for row in rows for k in row} - set(RESULTS_TABLE_COLUMNS)
    fieldnames = RESULTS_TABLE_COLUMNS + sorted(extra)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
