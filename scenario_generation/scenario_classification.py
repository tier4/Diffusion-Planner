"""Load and resolve portable scenario classification JSON (Meta-Repository format).

Layout convention (mirrors on-disk NPZ valid trees)::

    NPZ valid:   {corpus}/{project}/{map}/valid/{date}/{bag}/
    Class JSON:  {root}/{project}/{map}/{date}.json

Example::

    .../x2_dev/2231_odaiba_shinagawa_copied_from_xx1/valid/2026-01-15/13-42-45/
    .../scenario_classification_json/x2_dev/2231_odaiba_shinagawa_copied_from_xx1/2026-01-15.json
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

AreaEpisode = dict
# keys: area, metric_group, video_start_idx, video_end_idx, labeled_ranges

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BAG_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class NpzValidLayout:
    """Parsed NPZ path under ``.../{project}/{map}/valid/...``."""

    dataset_key: str
    valid_root: Path
    dates: tuple[str, ...]
    npz_roots_by_date: dict[str, Path]


@dataclass(frozen=True)
class ClassificationEvalJob:
    """One grouped closed-loop eval: one valid date folder + one classification JSON."""

    npz_root: Path
    classification_json: Path
    dataset_key: str
    date: str


def load_classification_json(path: Path) -> dict:
    """Load scenario_classification_json document."""
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def time_series_from_doc(doc: dict) -> dict[str, dict]:
    """Per-bag ``time_series`` entries.

    Legacy docs may use top-level ``sequences``; they are normalized to the same shape.
    """
    if "time_series" in doc:
        return dict(doc["time_series"])
    legacy = doc.get("sequences", {})
    return {k: {"name": k, **v} for k, v in legacy.items()}


def load_area_metric_groups(doc: dict) -> dict[str, str]:
    """area_name -> metric_group."""
    if "area_catalog" in doc:
        return {str(k): str(v["metric_group"]) for k, v in doc["area_catalog"].items()}
    if "areas" in doc:
        return {str(k): str(v["metric_group"]) for k, v in doc["areas"].items()}
    out: dict[str, str] = {}
    for entry in time_series_from_doc(doc).values():
        for ep in episodes_from_entry(entry):
            out[str(ep["area"])] = str(ep["metric_group"])
    return out


def _normalize_labeled_ranges(seg: dict) -> list[list[int]]:
    if "labeled_ranges" in seg:
        return [[int(r[0]), int(r[1])] for r in seg["labeled_ranges"]]
    return [[int(seg["start_idx"]), int(seg["end_idx"])]]


def episodes_from_entry(entry: dict) -> list[AreaEpisode]:
    """Return normalized episode list for one bag entry.

    v2.2 episodes bridge unlabeled gaps via ``video_*``; metrics use ``labeled_ranges``.
    v2.1 ``start_idx``/``end_idx`` and legacy ``area_by_idx`` are upgraded on read.
    """
    if "segments" in entry:
        episodes: list[AreaEpisode] = []
        for seg in entry["segments"]:
            if "labeled_ranges" in seg:
                episodes.append(
                    {
                        "area": str(seg["area"]),
                        "metric_group": str(seg["metric_group"]),
                        "video_start_idx": int(seg["video_start_idx"]),
                        "video_end_idx": int(seg["video_end_idx"]),
                        "labeled_ranges": _normalize_labeled_ranges(seg),
                    }
                )
            else:
                start = int(seg["start_idx"])
                end = int(seg["end_idx"])
                episodes.append(
                    {
                        "area": str(seg["area"]),
                        "metric_group": str(seg["metric_group"]),
                        "video_start_idx": start,
                        "video_end_idx": end,
                        "labeled_ranges": [[start, end]],
                    }
                )
        return _merge_v21_episodes(episodes)

    area_by_idx = entry.get("area_by_idx") or []
    groups = entry.get("metric_group_by_idx") or [None] * len(area_by_idx)
    chunks: list[AreaEpisode] = []
    for area, start_idx, end_idx in iter_area_spans(area_by_idx):
        mg = groups[start_idx] if start_idx < len(groups) else None
        chunks.append(
            {
                "area": area,
                "metric_group": str(mg) if mg else "",
                "video_start_idx": start_idx,
                "video_end_idx": end_idx,
                "labeled_ranges": [[start_idx, end_idx]],
            }
        )
    return _merge_v21_episodes(chunks)


def _merge_v21_episodes(episodes: list[AreaEpisode]) -> list[AreaEpisode]:
    """Merge consecutive same-area v2.1-style episodes (bridge unlabeled gaps)."""
    merged: list[AreaEpisode] = []
    for ep in episodes:
        if merged and merged[-1]["area"] == ep["area"]:
            merged[-1]["video_end_idx"] = int(ep["video_end_idx"])
            merged[-1]["labeled_ranges"].extend(ep["labeled_ranges"])
        else:
            merged.append(
                {
                    "area": ep["area"],
                    "metric_group": ep["metric_group"],
                    "video_start_idx": int(ep["video_start_idx"]),
                    "video_end_idx": int(ep["video_end_idx"]),
                    "labeled_ranges": [list(r) for r in ep["labeled_ranges"]],
                }
            )
    return merged


def segments_from_entry(entry: dict) -> list[AreaEpisode]:
    """Alias for :func:`episodes_from_entry` (kept for older call sites)."""
    return episodes_from_entry(entry)


def iter_episodes(entry: dict) -> Iterator[AreaEpisode]:
    """Yield normalized episodes for one bag."""
    yield from episodes_from_entry(entry)


def iter_segments(entry: dict) -> Iterator[tuple[str, int, int]]:
    """Yield ``(area_name, start_idx, end_idx)`` per labeled range (legacy helper)."""
    for ep in episodes_from_entry(entry):
        for start, end in ep["labeled_ranges"]:
            yield str(ep["area"]), int(start), int(end)


def idx_in_labeled_ranges(idx: int, labeled_ranges: list[list[int]]) -> bool:
    return any(int(start) <= idx < int(end) for start, end in labeled_ranges)


def area_at_idx(episodes: list[AreaEpisode], idx: int) -> str | None:
    """Map reproducer frame index to area name on labeled frames only."""
    for ep in episodes:
        if idx_in_labeled_ranges(idx, ep["labeled_ranges"]):
            return str(ep["area"])
    return None


def iter_area_spans(area_by_idx: list[str | None]) -> Iterator[tuple[str, int, int]]:
    """Yield ``(area_name, start_idx, end_idx)`` for each contiguous labeled span (legacy)."""
    i, n = 0, len(area_by_idx)
    while i < n:
        area = area_by_idx[i]
        if area is None:
            i += 1
            continue
        start = i
        i += 1
        while i < n and area_by_idx[i] == area:
            i += 1
        yield str(area), start, i


def validate_classification_npz_root(doc: dict, npz_root: Path) -> list[str]:
    """Return warning strings if JSON date disagrees with ``npz_root`` basename."""
    warnings: list[str] = []
    date = doc.get("date")
    if date and npz_root.name != date:
        warnings.append(f"npz_root basename {npz_root.name!r} != classification date {date!r}")
    return warnings


def classification_json_search_roots() -> list[Path]:
    """Default roots for ``scenario_classification_json/{dataset_key}/{date}.json``."""
    roots: list[Path] = []
    env = os.environ.get("SCENARIO_CLASSIFICATION_JSON_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    repo_root = Path(__file__).resolve().parents[1]
    roots.append(
        repo_root.parent
        / "Diffusion-Planner-Meta-Repository"
        / "dataset"
        / "scenario_classification_json"
    )
    return roots


def parse_npz_valid_layout(
    npz_root: Path | str,
    *,
    dataset_name: str | None = None,
) -> NpzValidLayout | None:
    """Infer ``{project}/{map}``, valid root, and date(s) from an NPZ path.

    Supported ``npz_root`` shapes::

        .../{project}/{map}/valid
        .../{project}/{map}/valid/{date}
        .../{project}/{map}/valid/{date}/{bag}
    """
    path = Path(npz_root).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    # Keep symlink components (e.g. .../diffusion_planner_local_dataset/x2_dev/.../valid)
    # so dataset_key inference matches classification JSON layout; do not .resolve().
    parts = path.parts
    if "valid" not in parts:
        return None

    valid_idx = parts.index("valid")
    if valid_idx < 2:
        return None

    inferred_key = "/".join(parts[valid_idx - 2 : valid_idx])
    dataset_key = (dataset_name or inferred_key).strip("/")
    valid_root = Path(*parts[: valid_idx + 1])

    def _npz_date_dir(date: str) -> Path:
        logical = valid_root / date
        if logical.is_dir():
            return logical
        return (valid_root / date).resolve()

    after_valid = parts[valid_idx + 1 :]
    if not after_valid:
        dates = sorted(d.name for d in valid_root.iterdir() if d.is_dir() and DATE_RE.match(d.name))
        return NpzValidLayout(
            dataset_key=dataset_key,
            valid_root=valid_root,
            dates=tuple(dates),
            npz_roots_by_date={d: _npz_date_dir(d) for d in dates},
        )

    if not DATE_RE.match(after_valid[0]):
        return None

    date = after_valid[0]
    return NpzValidLayout(
        dataset_key=dataset_key,
        valid_root=valid_root,
        dates=(date,),
        npz_roots_by_date={date: _npz_date_dir(date)},
    )


def _classification_search_roots(
    classification_root: Path | str | None,
    search_roots: list[Path] | None,
) -> list[Path]:
    roots: list[Path] = []
    if classification_root:
        root = Path(classification_root).expanduser()
        if root.is_dir():
            roots.append(root.resolve())
    for root in search_roots or classification_json_search_roots():
        resolved = root.expanduser().resolve()
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def _npz_date_has_bags(npz_date_root: Path) -> bool:
    """True when the date folder exists and has at least one bag subdir or NPZ files."""
    if not npz_date_root.is_dir():
        return False
    for child in npz_date_root.iterdir():
        if child.is_dir() and BAG_RE.match(child.name):
            return True
        if child.suffix == ".npz":
            return True
    return False


def discover_classification_eval_jobs(
    npz_root: Path | str,
    classification_root: Path | str | None = None,
    *,
    explicit_path: str | Path | None = None,
    dataset_name: str | None = None,
    search_roots: list[Path] | None = None,
) -> list[ClassificationEvalJob]:
    """Match classification JSON files under ``classification_root`` to NPZ valid dates.

    Resolution order:
    1. ``explicit_path`` when it is an existing ``.json`` file (legacy single-file override).
    2. ``{root}/{dataset_key}/{date}.json`` for each valid date under ``npz_root``.
    """
    layout = parse_npz_valid_layout(npz_root, dataset_name=dataset_name)
    if layout is None:
        return []

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file():
            date = path.stem
            npz_date_root = layout.npz_roots_by_date.get(date)
            if npz_date_root is None:
                # Legacy: npz_root may already be the date folder while JSON path is explicit.
                npz_date_root = Path(npz_root).expanduser().resolve()
                if DATE_RE.match(npz_date_root.name):
                    date = npz_date_root.name
                else:
                    return []
            return [
                ClassificationEvalJob(
                    npz_root=npz_date_root,
                    classification_json=path.resolve(),
                    dataset_key=layout.dataset_key,
                    date=date,
                )
            ]
        if path.is_dir():
            classification_root = path

    roots = _classification_search_roots(classification_root, search_roots)
    if not roots:
        return []

    jobs: list[ClassificationEvalJob] = []
    for date in layout.dates:
        npz_date_root = layout.npz_roots_by_date[date]
        if not _npz_date_has_bags(npz_date_root):
            continue
        json_path: Path | None = None
        for root in roots:
            candidate = root / layout.dataset_key / f"{date}.json"
            if candidate.is_file():
                json_path = candidate.resolve()
                break
        if json_path is None:
            continue
        jobs.append(
            ClassificationEvalJob(
                npz_root=npz_date_root,
                classification_json=json_path,
                dataset_key=layout.dataset_key,
                date=date,
            )
        )
    return jobs


def resolve_classification_json(
    npz_root: Path | str,
    explicit: str | Path | None = None,
    *,
    dataset_name: str | None = None,
    classification_root: str | Path | None = None,
    search_roots: list[Path] | None = None,
) -> Path | None:
    """Resolve a single classification JSON for grouped closed-loop eval.

    When multiple dates match, returns the JSON for the first matched date (sorted).
    Prefer :func:`discover_classification_eval_jobs` for multi-date validation.
    """
    root = classification_root
    explicit_path = explicit
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_dir():
            root = path
            explicit_path = None

    jobs = discover_classification_eval_jobs(
        npz_root,
        classification_root=root,
        explicit_path=explicit_path,
        dataset_name=dataset_name,
        search_roots=search_roots,
    )
    if not jobs:
        return None
    return jobs[0].classification_json


def merge_grouped_eval_summaries(summaries: list[dict]) -> dict:
    """Merge multiple grouped closed-loop summaries (e.g. several valid dates)."""
    from scenario_generation.metrics.group_report import aggregate_segment_rows

    if not summaries:
        return {"mode": "grouped", "grouped_summary": {"by_area_name": {}, "totals": {}}}
    if len(summaries) == 1:
        return summaries[0]

    all_segments: list[dict] = []
    video_mp4s: list = []
    elapsed_sec = 0.0
    n_bags = 0
    jobs_meta: list[dict] = []

    for summary in summaries:
        all_segments.extend(summary.get("segments") or [])
        video_mp4s.extend(summary.get("video_mp4s") or [])
        elapsed_sec += float(summary.get("elapsed_sec", 0.0))
        n_bags += int(summary.get("n_bags", 0))
        jobs_meta.append(
            {
                "date": summary.get("date"),
                "dataset_key": summary.get("dataset_key"),
                "classification_json": summary.get("classification_json"),
                "npz_root": summary.get("npz_root"),
            }
        )

    grouped_summary = aggregate_segment_rows(all_segments)
    base = summaries[0]
    timing = None
    if any(s.get("timing") for s in summaries):
        from scenario_generation.perf_timer import Timers

        global_timers = Timers()
        per_bag: list[dict] = []
        total_steps = 0
        for summary in summaries:
            t = summary.get("timing") or {}
            total_steps += int(t.get("total_sim_steps", 0))
            per_bag.extend(t.get("per_bag") or [])
            for stage, info in (t.get("stages") or {}).items():
                global_timers.add(stage, float(info["total_s"]), int(info["calls"]))
        from scenario_generation.grouped_closed_loop_eval import _build_timing_summary

        timing = _build_timing_summary(global_timers, per_bag, total_steps, elapsed_sec)

    merged = {
        "mode": "grouped",
        "classification_json": [m["classification_json"] for m in jobs_meta],
        "classification_jobs": jobs_meta,
        "npz_root": base.get("npz_root"),
        "near_miss_thresh": base.get("near_miss_thresh"),
        "n_episodes": len(all_segments),
        "n_bags": n_bags,
        "elapsed_sec": elapsed_sec,
        "grouped_summary": grouped_summary,
        "segments": all_segments,
        "video_mp4s": video_mp4s,
    }
    if timing is not None:
        merged["timing"] = timing
    return merged
