"""Build wandb log payloads for full-route (per-site) closed-loop validation."""

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
    """True for the empty-world-ablation site label convention (``{site}__noobj``). Used to
    exclude ablation labels from collision-style aggregates that are 0 by construction in that
    mode."""
    return label.endswith("__noobj")


def episode_stem(out_dir: str | Path, row: dict) -> str:
    """Base filename stem for one segments.jsonl row's video/png-dir/colormap files.

    FullRouteClosedLoopEvaluation (PR2, train-time closed-loop validation) names these
    ``{route}_{start}_{end}`` (segment-suffixed). PR1's ``run_closed_loop_eval`` (the
    ``valid_predictor_closed_loop.py`` / ``run_all_sites_closed_loop.py`` CLI path) names
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
    """Pick one segment row to represent a site's whole run (for the 1 video/image W&B keeps).

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
    "site",
    "vehicle_type",
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


def _episode_row(
    table: wandb.Table,
    site: str,
    r: dict,
    out_dir: str | Path | None,
    vehicle_type: str | None = None,
) -> None:
    seg = r.get("segment")
    seg_str = f"[{seg[0]},{seg[1]}]" if seg else ""
    video_path = str(_segment_paths(out_dir, r)[1]) if out_dir is not None else ""
    comp = r.get("route_completion")
    table.add_data(
        site,
        vehicle_type or "",
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
    site_episodes: list[tuple[str, list[dict], str | Path | None]]
    | list[tuple[str, list[dict], str | Path | None, str | None]],
) -> wandb.Table:
    """ONE episode table across every site, so the W&B UI's native sort/filter/group-by
    works across the whole run in a single panel. ``site_episodes`` is
    ``[(site_name, rows, out_dir), ...]``, optionally with a 4th ``vehicle_type`` element.
    """
    table = wandb.Table(columns=EPISODE_TABLE_COLUMNS)
    for entry in site_episodes:
        site, rows, out_dir, *rest = entry
        vehicle_type = rest[0] if rest else None
        for r in rows:
            _episode_row(table, site, r, out_dir, vehicle_type)
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


def _aggregate_rollup(prefix: str, values: list[dict], objects_values: list[dict]) -> dict:
    """Rollup under ``{prefix}/...``, shared by the overall and per-vehicle logs."""
    log: dict = {}
    n_sites = len(values)
    total_segments = sum(int(s.get("n_segments", 0)) for s in values)

    log[f"{prefix}/n_sites"] = n_sites
    log[f"{prefix}/n_segments"] = total_segments

    comp_num = sum(
        float(s.get("mean_route_completion", 0.0)) * int(s.get("n_segments", 0)) for s in values
    )
    log[f"{prefix}/route_completion"] = comp_num / total_segments if total_segments else 0.0

    for key in COMPARISON_OVERVIEW_SUM_KEYS:
        log[f"{prefix}/{key}"] = sum(int(extract_score(s, key) or 0) for s in values)
    for key in OBJECTS_ONLY_OVERVIEW_SUM_KEYS:
        log[f"{prefix}/{key}"] = sum(int(extract_score(s, key) or 0) for s in objects_values)

    return log


def build_sites_aggregate_log(
    summaries: dict[str, dict],
    site_vehicle_types: dict[str, str] | None = None,
) -> dict:
    """Cross-site rollup under ``closed_loop_overview/`` (the at-a-glance block): the segment-
    weighted mean route-completion (so long routes aren't under-weighted), plus the plain
    cross-site SUM of each event count. Deliberately just the small non-saturating set — no
    segment-rates / min-clearances / means (those stay in each site's summary.json only).

    ``summaries`` is keyed by site LABEL — a ``{site}__noobj`` label is excluded from
    collision-style sums (``OBJECTS_ONLY_OVERVIEW_SUM_KEYS``), since those are 0 by
    construction in the empty-world ablation and would just dilute the objects-mode number
    with zeros.

    If ``site_vehicle_types`` (``{site_label: vehicle_type}``) is given, the same rollup is
    also computed per vehicle type under ``closed_loop_overview_by_vehicle/{vehicle_type}/...``.
    """
    log: dict = {}
    if not summaries:
        return log
    values = list(summaries.values())
    objects_values = [s for label, s in summaries.items() if not _is_noobj_label(label)]
    log.update(_aggregate_rollup("closed_loop_overview", values, objects_values))

    if site_vehicle_types:
        by_vehicle: dict[str, list[str]] = {}
        for label in summaries:
            base_label = label[: -len("__noobj")] if _is_noobj_label(label) else label
            vehicle_type = site_vehicle_types.get(base_label) or site_vehicle_types.get(label)
            if vehicle_type is None:
                continue
            by_vehicle.setdefault(vehicle_type, []).append(label)
        for vehicle_type, labels in by_vehicle.items():
            v_values = [summaries[label] for label in labels]
            v_objects_values = [summaries[label] for label in labels if not _is_noobj_label(label)]
            log.update(
                _aggregate_rollup(
                    f"closed_loop_overview_by_vehicle/{vehicle_type}", v_values, v_objects_values
                )
            )

    return {k: v for k, v in log.items() if _wandb_scalar(v) or isinstance(v, int)}


def _site_label(site: str | None) -> str:
    """W&B-key-safe site token; ``None`` (single-npz_root mode) -> ``"main"``."""
    return (site or "main").replace("/", "_")


def build_full_closed_loop_wandb_log(
    summary: dict,
    *,
    out_dir: str | Path | None = None,
    site: str | None = None,
    video_pick: str = "worst",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    near_miss_thresh: float = 0.5,
    report_base_url: str | None = None,
    render_media: bool = True,
) -> dict:
    """Per-site full-route closed-loop wandb payload, keyed into role-based sections so the
    workspace stays navigable (one collapsible section each) instead of one flat ``closed_loop``
    blob of 100+ panels:

    - ``closed_loop_scores/{metric}/{site}`` — scalar trends (metric-first so the same metric's
      sites sort adjacently; the W&B panel-search box filters by either metric or site token).
    - ``closed_loop_media/{site}`` — ONE gallery panel holding every ``colormap_metrics`` image
      for the representative episode (captioned by metric), mirroring the HTML report's
      per-card metric dropdown; ``closed_loop_media/{site}__video`` — that episode's video.
    - ``closed_loop_links/{site}`` — where the full report (all videos + HTML) lives.

    ``render_media=False`` skips the video + colormap-image block entirely (scores/links are
    unaffected) -- for a caller that already skipped rendering (e.g. train.py's RolloutParams
    ``draw=False`` on most epochs), so there's no colormap image to render from anyway.

    The per-episode table is built once across ALL sites by :func:`build_combined_episode_table`
    at the caller (so it's a single filterable/groupable panel), not here.
    """
    label = _site_label(site)
    log: dict = {}
    for key in SCORE_KEYS:
        val = extract_score(summary, key)
        if _wandb_scalar(val):
            log[f"closed_loop_scores/{key}/{label}"] = val

    rows = summary.get("segments") or []
    rep = pick_representative_row(rows, mode=video_pick)
    if render_media and rep is not None and out_dir is not None:
        png_dir, mp4_path = _segment_paths(out_dir, rep)
        if mp4_path.is_file():
            log[f"closed_loop_media/{label}__video"] = wandb.Video(str(mp4_path), format="mp4")
        try:
            rendered = render_trajectory_colormaps(
                png_dir,
                out_dir,
                mp4_path.stem,
                metrics=colormap_metrics,
                near_miss_thresh=near_miss_thresh,
                strong_brake_mps2=summary.get("strong_brake", {}).get("thresh_mps2", -2.5),
                title=f"{site or ''} {mp4_path.stem}".strip(),
            )
        except Exception as e:  # pragma: no cover - rendering must never break training
            print(f"closed_loop: trajectory colormap failed for {mp4_path.stem}: {e}")
            rendered = {}
        # One gallery panel per site: a list of images under a single key (captioned by metric)
        # -> the metric becomes an in-panel selector, not N separate panels. Ordered by the
        # requested colormap_metrics so the gallery is stable across epochs/sites.
        gallery = [
            wandb.Image(str(rendered[m]), caption=m) for m in colormap_metrics if m in rendered
        ]
        if gallery:
            log[f"closed_loop_media/{label}"] = gallery

    if out_dir is not None:
        log[f"closed_loop_links/{label}"] = resolve_report_link(out_dir, report_base_url)
    elif summary.get("npz_root"):
        log[f"closed_loop_links/{label}"] = str(summary["npz_root"])
    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, bool)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False
