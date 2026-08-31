"""W&B panels for a Scenario Sim run: scalars, one video per failure category, and a summary
table over a series of checkpoints. The taxonomy is ``classify_failure``'s.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import wandb

from scenario_generation.scenario_sim_viewer_export import (
    FAILURE_CATEGORIES,
    ViewerTree,
    classify_failure,
)

_PASS_CATEGORY = "PASS"


def load_case_rows(run_dir: Path) -> list[dict[str, Any]]:
    """The case rows a viewer export wrote.

    Not a closed-loop ``segments.jsonl``: those rows carry no verdict, so every case would read
    as failed. A line that does not parse is skipped -- a run in flight leaves a partial one."""
    path = run_dir / "cases.jsonl"
    if not path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _block(row: dict[str, Any], name: str) -> dict[str, Any]:
    block = row.get(name)
    return block if isinstance(block, dict) else {}


class _Case(NamedTuple):
    """``row`` is kept because the selectors return the caller's own dicts."""

    row: dict[str, Any]
    category: str
    obj_cols: int
    rb_cols: int
    sb_count: int
    clearance: float | None
    progress: float | None

    @property
    def is_pass(self) -> bool:
        return self.category == _PASS_CATEGORY

    def by_clearance(self, sign: float = 1.0) -> tuple[bool, float]:
        """Ascending by room kept, unmeasured last: no measurement is not evidence of danger."""
        return (self.clearance is None, sign * (self.clearance or 0.0))


def _view(r: dict[str, Any]) -> _Case:
    obj = _block(r, "object")
    clearance = obj.get("clearance_min_m")
    progress = r.get("progress_m")
    return _Case(
        row=r,
        category=classify_failure(r, r.get("verdict")),
        obj_cols=int(obj.get("collision_count") or 0),
        rb_cols=int(_block(r, "road_border").get("collision_count") or 0),
        sb_count=int(_block(r, "strong_brake").get("count") or 0),
        clearance=float(clearance) if isinstance(clearance, (int, float)) else None,
        progress=float(progress) if isinstance(progress, (int, float)) else None,
    )


def _case_name(case_dict: dict[str, Any]) -> str:
    return str(case_dict.get("case_key") or "")


def _bucket_name(category: str) -> str:
    return "best_pass" if category == _PASS_CATEGORY else f"worst_{category.lower()}"


def _worst_first(v: _Case) -> tuple[Any, ...]:
    return (v.by_clearance(), v.progress or 0.0)


# A category defined by a count ranks by it first.
_RANKERS = {
    _PASS_CATEGORY: lambda v: (v.by_clearance(-1.0), -(v.progress or 0.0)),
    "ROAD_DEPARTURE": lambda v: (-v.rb_cols, v.by_clearance()),
}


def pick_diverse_worst_cases(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One case per failure category, plus the best pass.

    ``worst_strong_brake`` sits on top of them: braking hard is not itself a failure."""
    by_category: dict[str, list[_Case]] = {}
    strong_brakes: list[_Case] = []

    for v in map(_view, rows):
        by_category.setdefault(v.category, []).append(v)
        if v.sb_count > 0 and not v.is_pass:
            strong_brakes.append(v)

    selected = {
        _bucket_name(category): min(cases, key=_RANKERS.get(category, _worst_first)).row
        for category, cases in by_category.items()
    }
    if strong_brakes:
        selected["worst_strong_brake"] = min(
            strong_brakes, key=lambda v: (-v.sb_count, v.by_clearance())
        ).row

    return selected


def _ratio(part: int, whole: int) -> float:
    return round(part / whole * 100.0, 2) if whole else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


# (scalar name, series-table header, format, value). No format keeps the raw number so W&B can
# sort it. ``*_cases`` counts cases; wandb_closed_loop's ``total_*`` counts events.
_METRICS: tuple[tuple[str, str, str | None, Any], ...] = (
    ("total_cases", "Cases", None, len),
    ("passed_cases", "Passed", None, lambda vs: sum(1 for v in vs if v.is_pass)),
    (
        "pass_rate",
        "Pass rate",
        "{:.2f}%",
        lambda vs: _ratio(sum(1 for v in vs if v.is_pass), len(vs)),
    ),
    (
        "collision_cases",
        "Cases with an object collision",
        None,
        lambda vs: sum(1 for v in vs if v.obj_cols > 0),
    ),
    (
        "road_border_cases",
        "Cases touching a road border",
        None,
        lambda vs: sum(1 for v in vs if v.rb_cols > 0),
    ),
    (
        "strong_brake_cases",
        "Cases braking hard",
        None,
        lambda vs: sum(1 for v in vs if v.sb_count > 0),
    ),
    (
        "mean_case_min_clearance_m",
        "Mean of per-case min clearance (m)",
        "{:.2f}",
        lambda vs: _mean([v.clearance for v in vs if v.clearance is not None and v.clearance >= 0]),
    ),
    (
        "mean_progress_m",
        "Mean progress (m)",
        None,
        lambda vs: _mean([v.progress for v in vs if v.progress is not None]),
    ),
)


def build_scenario_sim_scalars(
    rows: list[dict[str, Any]], prefix: str = "scenario_sim"
) -> dict[str, float | int]:
    """Aggregate scalars for one run, including a per-category breakdown that partitions it."""
    views = [_view(r) for r in rows]

    # Zeros included: a key missing from some steps reads as a gap. These partition the run.
    seen = Counter(v.category for v in views)

    return {
        **{f"{prefix}/{name}": value(views) for name, _, _, value in _METRICS},
        **{f"{prefix}/category/{c.lower()}": seen[c] for c in FAILURE_CATEGORIES},
    }


def find_video_for_case(case_dict: dict[str, Any], media_root: Path) -> Path | None:
    scenario, case = case_dict.get("scenario"), _case_name(case_dict)
    if not (scenario and case):
        return None
    path = ViewerTree(media_root).video(scenario, case)
    return path if path.is_file() else None


def _build_video_entries(
    rows: list[dict[str, Any]], media_root: Path, prefix: str
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for bucket_name, case_info in pick_diverse_worst_cases(rows).items():
        video_path = find_video_for_case(case_info, media_root)
        if video_path is None:
            continue
        caption = f"{bucket_name}: {_case_name(case_info)} ({case_info['result_kind']})"
        try:
            entries[f"{prefix}/videos/{bucket_name}"] = wandb.Video(
                str(video_path), format="mp4", caption=caption
            )
        except Exception as exc:
            print(
                f"wandb_scenario_sim: failed to create wandb.Video for {video_path}: {exc}",
                file=sys.stderr,
            )
    return entries


_TABLE_COLUMNS = [
    "Case",
    "Verdict",
    "Category",
    "Collision",
    "Road Border",
    "Strong Brake",
    "Min Clearance (m)",
    "Progress (m)",
]


def build_scenario_sim_wandb_payload(
    rows: list[dict[str, Any]],
    media_root: Path | None = None,
    prefix: str = "scenario_sim",
) -> dict[str, Any]:
    """Scalars, the per-case table and one video per bucket, ready for ``wandb.log``."""
    payload: dict[str, Any] = dict(build_scenario_sim_scalars(rows, prefix=prefix))

    if media_root:
        payload.update(_build_video_entries(rows, media_root, prefix))

    table_rows = []
    for v in map(_view, rows):
        table_rows.append(
            [
                _case_name(v.row),
                v.row["result_kind"],
                v.category,
                v.obj_cols > 0,
                v.rb_cols > 0,
                v.sb_count > 0,
                f"{v.clearance:.2f}" if v.clearance is not None else "-",
                f"{v.progress:.1f}" if v.progress is not None else "-",
            ]
        )

    payload[f"{prefix}/cases_table"] = wandb.Table(columns=_TABLE_COLUMNS, data=table_rows)
    return payload


_SERIES_HEAD = ("Checkpoint", "Epoch")


class SeriesPoint(NamedTuple):
    label: str
    epoch: int | None
    scalars: dict[str, float | int]


def series_point(label: str, epoch: int | None, payload: dict[str, Any]) -> SeriesPoint:
    """A point read off what was logged, so it cannot disagree with its step. Numbers only."""
    numbers = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    return SeriesPoint(label, epoch, numbers)


def build_scenario_sim_series_panels(
    points: list[SeriesPoint], prefix: str = "scenario_sim"
) -> dict[str, Any]:
    """A summary table over ``points``, in order.

    Nothing else: the per-category scalars already carry the composition, which W&B stacks."""
    if not points:
        return {}

    table_rows = []
    for p in points:
        cells: list[Any] = [p.label, p.epoch if p.epoch is not None else "-"]
        for name, _, fmt, _compute in _METRICS:
            value = p.scalars[f"{prefix}/{name}"]
            cells.append(fmt.format(value) if fmt else value)
        table_rows.append(cells)

    return {
        f"{prefix}/series/summary": wandb.Table(
            columns=[*_SERIES_HEAD, *(header for _, header, _, _ in _METRICS)], data=table_rows
        ),
    }
