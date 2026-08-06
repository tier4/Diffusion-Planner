"""Build wandb log payloads for full-route (per-group) closed-loop validation."""

from __future__ import annotations

import math
from pathlib import Path

import wandb

from scenario_generation.closed_loop_score_keys import (
    COMPARISON_OVERVIEW_SUM_KEYS,
    OBJECTS_ONLY_OVERVIEW_SUM_KEYS,
    SCORE_KEYS,
    extract_score,
)
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps


def _is_noobj_label(label: str) -> bool:
    """True for the empty-world-ablation group label convention.

    Supports both old format ``{group}__noobj`` and new format ``{json_name}__noobj/{group}``.
    """
    return "__noobj" in label


def episode_stem(out_dir: str | Path, row: dict) -> str:
    """Base filename stem for one segments.jsonl row's video/png-dir/colormap files.

    FullRouteClosedLoopEvaluation (PR2, train-time closed-loop validation) names these
    ``{route}_{start}_{end}`` (segment-suffixed). PR1's ``run_closed_loop_eval`` (the
    ``valid_predictor_closed_loop.py`` / ``run_all_groups_closed_loop.py`` CLI path) names
    them just ``{route}`` -- one route = one whole-route rollout, no sub-segmenting.
    Prefer the segment-suffixed form and fall back to the bare route name when that
    file/dir doesn't exist, so callers resolve videos correctly for either pipeline.
    """
    out_dir = Path(out_dir)
    start, end = row["segment"]
    segmented_stem = f"{row['route']}_{start}_{end}"
    if (out_dir / f"{segmented_stem}.mp4").is_file() or (out_dir / segmented_stem).is_dir():
        return segmented_stem
    return row["route"]


def _segment_paths(out_dir: str | Path, row: dict) -> tuple[Path, Path]:
    """(png_dir, mp4_path) for one segments.jsonl row -- see :func:`episode_stem`."""
    out_dir = Path(out_dir)
    stem = episode_stem(out_dir, row)
    return out_dir / stem, out_dir / f"{stem}.mp4"


def pick_representative_row(rows: list[dict], mode: str = "worst") -> dict | None:
    """Pick one segment row to represent a group's whole run (for the 1 video/image W&B keeps).

    ``mode``: ``"worst"`` (default) = most collision steps, tie-broken by smallest
    min_clearance — the case most worth a human's attention. ``"first"`` = first
    discovered route/segment (stable, arbitrary). ``"longest"`` = most steps run.
    """
    if not rows:
        return None
    if mode == "first":
        return rows[0]
    if mode == "longest":
        return max(rows, key=lambda r: r.get("n_steps_run", 0))

    def _worst_key(r: dict) -> tuple[int, float]:
        obj = r.get("object", {})
        coll = obj.get("collision_steps", 0)
        cl = obj.get("clearance_min_m", float("inf"))
        cl = cl if math.isfinite(cl) else 1e9
        return (coll, -cl)  # more collisions first, then smaller clearance first

    return max(rows, key=_worst_key)


EPISODE_TABLE_COLUMNS = [
    "group",
    "route",
    "segment",
    "n_steps_run",
    "terminated",
    "route_completion",
    "n_collision_events",
    "n_curb_hits",
    "n_snaps",
    "n_red_light_violations",
    "n_strong_brakes",
    "progress_m",
    "video_path",
]


def _episode_row(table: wandb.Table, group: str, r: dict, out_dir: str | Path | None) -> None:
    seg = r.get("segment")
    seg_str = f"[{seg[0]},{seg[1]}]" if seg else ""
    video_path = str(_segment_paths(out_dir, r)[1]) if out_dir is not None else ""
    comp = r.get("route_completion")
    table.add_data(
        group,
        r.get("route", ""),
        seg_str,
        int(r.get("n_steps_run", 0)),
        r.get("terminated", ""),
        float(comp) if comp is not None and math.isfinite(comp) else None,
        int(extract_score(r, "total_collision_events") or 0),
        int(extract_score(r, "total_curb_hits") or 0),
        int(extract_score(r, "total_snaps") or 0),
        int(extract_score(r, "total_red_light_violations") or 0),
        int(extract_score(r, "total_strong_brakes") or 0),
        float(r.get("progress_m", 0.0)),
        video_path,
    )


def build_combined_episode_table(
    group_episodes: list[tuple[str, list[dict], str | Path | None]],
) -> wandb.Table:
    """ONE episode table across every group (``group`` column filled per row), so the W&B UI's
    native sort/filter/group-by works across the whole run — group by ``group``, sort by
    ``n_collision_events`` desc, etc. — in a single interactive panel instead of one table
    per group. ``group_episodes`` is ``[(group_name, rows, out_dir), ...]``.
    """
    table = wandb.Table(columns=EPISODE_TABLE_COLUMNS)
    for group, rows, out_dir in group_episodes:
        for r in rows:
            _episode_row(table, group, r, out_dir)
    return table


def resolve_report_link(out_dir: str | Path, report_base_url: str | None = None) -> str:
    """Where the rich local report (all videos + HTML gallery) for this run lives.

    Returns a clickable ``http(s)://...`` URL if ``report_base_url`` is set (the run's
    ``out_dir`` is being served over HTTP from that base, e.g. on a training server), else
    the plain local filesystem path (informational only — W&B's web UI cannot open
    ``file://`` links, so on a local dev machine this is for the human to copy/open by hand).
    """
    out_dir = Path(out_dir)
    if report_base_url:
        return report_base_url.rstrip("/") + "/" + out_dir.name
    return str(out_dir.resolve())


def build_groups_aggregate_log(
    summaries: dict[str, dict],
    *,
    prefix: str = "closed_loop_overview",
) -> dict:
    """Cross-group rollup under ``<prefix>/`` (default: ``closed_loop_overview/``): the segment-
    weighted mean route-completion (so long routes aren't under-weighted), plus the plain
    cross-group SUM of each event count. Deliberately just the small non-saturating set — no
    segment-rates / min-clearances / means (those stay in each group's summary.json only).

    ``summaries`` is keyed by group LABEL — a ``__noobj`` label is excluded from
    collision-style sums (``OBJECTS_ONLY_OVERVIEW_SUM_KEYS``), since those are 0 by
    construction in the empty-world ablation and would just dilute the objects-mode number
    with zeros.
    """
    log: dict = {}
    if not summaries:
        return log
    values = list(summaries.values())
    objects_values = [s for label, s in summaries.items() if not _is_noobj_label(label)]
    n_groups = len(values)
    total_segments = sum(int(s.get("n_segments", 0)) for s in values)

    log[f"{prefix}/n_groups"] = n_groups
    log[f"{prefix}/n_segments"] = total_segments

    comp_num = sum(
        float(s.get("mean_route_completion", 0.0)) * int(s.get("n_segments", 0)) for s in values
    )
    log[f"{prefix}/route_completion"] = comp_num / total_segments if total_segments else 0.0

    for key in COMPARISON_OVERVIEW_SUM_KEYS:
        log[f"{prefix}/{key}"] = sum(int(extract_score(s, key) or 0) for s in values)
    for key in OBJECTS_ONLY_OVERVIEW_SUM_KEYS:
        log[f"{prefix}/{key}"] = sum(int(extract_score(s, key) or 0) for s in objects_values)

    return {k: v for k, v in log.items() if _wandb_scalar(v) or isinstance(v, int)}


def _group_label(group: str | None) -> str:
    """W&B-key-safe group token; ``None`` (single-npz_root mode) -> ``"main"``."""
    return (group or "main").replace("/", "_")


def build_full_closed_loop_wandb_log(
    summary: dict,
    *,
    out_dir: str | Path | None = None,
    group: str | None = None,
    video_pick: str = "worst",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    near_miss_thresh: float = 0.5,
    report_base_url: str | None = None,
    render_media: bool = True,
    wandb_key_prefix: str | None = None,
) -> dict:
    """Per-group full-route closed-loop wandb payload, keyed into role-based sections so the
    workspace stays navigable (one collapsible section each) instead of one flat ``closed_loop``
    blob of 100+ panels:

    - ``closed_loop_scores/{metric}/{group}`` — scalar trends (metric-first so the same metric's
      groups sort adjacently; the W&B panel-search box filters by either metric or group token).
    - ``closed_loop_media/{group}`` — ONE gallery panel holding every ``colormap_metrics`` image
      for the representative episode (captioned by metric), mirroring the HTML report's
      per-card metric dropdown; ``closed_loop_media/{group}__video`` — that episode's video.
    - ``closed_loop_links/{group}`` — where the full report (all videos + HTML) lives.

    If ``wandb_key_prefix`` is set (e.g. ``override/departure``), it replaces the
    ``closed_loop_scores/``, ``closed_loop_media/``, ``closed_loop_links/`` prefix, allowing
    arbitrary nesting like ``closed_loop/<metric>/override/departure``.

    ``render_media=False`` skips the video + colormap-image block entirely (scores/links are
    unaffected) -- for a caller that already skipped rendering (e.g. train.py's RolloutParams
    ``draw=False`` on most epochs), so there's no colormap image to render from anyway.

    The per-episode table is built once across ALL groups by :func:`build_combined_episode_table`
    at the caller (so it's a single filterable/groupable panel), not here.
    """
    label = _group_label(group)
    log: dict = {}

    def _key(suffix: str) -> str:
        """Build wandb key, optionally prefixed.

        suffix format: "scores/{metric}/{label}" or "media/{label}" or "links/{label}"

        Without prefix: "closed_loop_scores/collision/label" (original format)
        With prefix "override/departure": "closed_loop/collision/override/departure/label"
        """
        if wandb_key_prefix is None:
            return f"closed_loop_{suffix}"
        # Extract metric from "scores/{metric}/..." or put everything after prefix
        parts = suffix.split("/", 3)
        if parts[0] == "scores" and len(parts) >= 3:
            # "scores/collision/label" -> metric=collision, rest=label
            metric = parts[1]
            label = parts[2]
            return f"closed_loop/{metric}/{wandb_key_prefix}/{label}"
        # media, links: "media/label__video" or "links/label"
        rest = "/".join(parts[1:])
        return f"closed_loop/{wandb_key_prefix}/{rest}"

    for key in SCORE_KEYS:
        val = extract_score(summary, key)
        if _wandb_scalar(val):
            log[_key(f"scores/{key}/{label}")] = val

    rows = summary.get("segments") or []
    rep = pick_representative_row(rows, mode=video_pick)
    if render_media and rep is not None and out_dir is not None:
        png_dir, mp4_path = _segment_paths(out_dir, rep)
        if mp4_path.is_file():
            log[_key(f"media/{label}__video")] = wandb.Video(str(mp4_path), format="mp4")
        try:
            rendered = render_trajectory_colormaps(
                png_dir,
                out_dir,
                mp4_path.stem,
                metrics=colormap_metrics,
                near_miss_thresh=near_miss_thresh,
                strong_brake_mps2=summary.get("strong_brake", {}).get("thresh_mps2", -2.5),
                title=f"{group or ''} {mp4_path.stem}".strip(),
            )
        except Exception as e:  # pragma: no cover - rendering must never break training
            print(f"closed_loop: trajectory colormap failed for {mp4_path.stem}: {e}")
            rendered = {}
        gallery = [
            wandb.Image(str(rendered[m]), caption=m) for m in colormap_metrics if m in rendered
        ]
        if gallery:
            log[_key(f"media/{label}")] = gallery

    if out_dir is not None:
        log[_key(f"links/{label}")] = resolve_report_link(out_dir, report_base_url)
    elif summary.get("npz_root"):
        log[_key(f"links/{label}")] = str(summary["npz_root"])
    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, bool)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False
