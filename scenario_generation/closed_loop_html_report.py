"""Build the self-contained local HTML gallery for a multi-group closed-loop run.

One page: a per-group summary table + a searchable/sortable/filterable grid of episode
cards (video + metrics), reading each group's ``summary.json``/``segments.jsonl`` from
``<out_root>/<group_name>/`` (the layout ``run_all_groups_closed_loop.py`` produces). No
external assets — safe to open directly from disk, and this is the "rich" artifact the
W&B side just links to (see ``scenario_generation.wandb_closed_loop.resolve_report_link``)
rather than duplicating (videos stay local/on the training server, not uploaded to W&B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_score_keys import extract_score
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps
from scenario_generation.wandb_closed_loop import episode_stem

# Shown first in each card's metric dropdown when available (the rest follow METRIC_CHOICES
# order) — clearance is the most broadly useful default view.
_DEFAULT_METRIC = "clearance"

# Cycled per group in discovery order — enough distinct hues for a handful of groups before
# repeating; exact color isn't semantically meaningful, just needs to be stable per-group
# within one report so the summary table and card tags visually match.
_PALETTE = ["#1a73e8", "#d68a1f", "#8a4ad6", "#2a8a6d", "#d63a3a", "#2aa5d6"]


def collect_group_data(
    out_root: str | Path,
    group_names: list[str],
    *,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> tuple[list[dict], list[dict]]:
    """Read each group's ``summary.json`` + ``segments.jsonl`` into (items, summaries) for
    :func:`build_html_report`. ``items`` is one dict per episode (with a resolved, relative
    ``video_path`` when the mp4 exists, and a ``colormap_paths`` dict of ``{metric: relative
    path}`` — one rendered image per metric in ``colormap_metrics`` that actually produced
    something, so the report's per-card dropdown can switch between them); ``summaries`` is
    one aggregate dict per group.
    """
    out_root = Path(out_root)
    items: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for group_name in group_names:
        group_dir = out_root / group_name
        summary_path = group_dir / "summary.json"
        segments_path = group_dir / "segments.jsonl"
        if not summary_path.is_file():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        near_miss_thresh = s.get("near_miss_thresh", 0.5)
        strong_brake_mps2 = s.get("strong_brake", {}).get("thresh_mps2", -2.5)
        summaries.append(
            {
                "group": group_name,
                "n_segments": s.get("n_segments", 0),
                "total_steps": s.get("total_steps", 0),
                "route_completion": s.get("mean_route_completion", 0.0),
                "total_collision_events": extract_score(s, "total_collision_events") or 0,
                "total_curb_hits": extract_score(s, "total_curb_hits") or 0,
                "total_snaps": extract_score(s, "total_snaps") or 0,
                "total_red_light_violations": extract_score(s, "total_red_light_violations") or 0,
                "total_strong_brakes": extract_score(s, "total_strong_brakes") or 0,
                "n_segments_diverged": s.get("n_segments_diverged", 0),
            }
        )
        if not segments_path.is_file():
            continue
        with segments_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                start, end = r["segment"]
                stem = episode_stem(group_dir, r)
                video_name = f"{stem}.mp4"
                video_path = group_dir / video_name

                # Skip metrics already rendered on a prior call for this out_root (report
                # rebuilds are common — e.g. re-running --only_groups) rather than re-drawing
                # every episode's every metric from scratch each time.
                missing_metrics = tuple(
                    m
                    for m in colormap_metrics
                    if not (group_dir / f"{stem}_trajcolormap_{m}.png").is_file()
                )
                if missing_metrics:
                    try:
                        render_trajectory_colormaps(
                            group_dir / stem,
                            group_dir,
                            stem,
                            metrics=missing_metrics,
                            near_miss_thresh=near_miss_thresh,
                            strong_brake_mps2=strong_brake_mps2,
                            title=f"{group_name} {stem}",
                        )
                    except (
                        Exception
                    ) as e:  # pragma: no cover - a bad episode must not break the report
                        print(f"closed_loop_html_report: colormap failed for {stem}: {e}")
                colormap_paths = {
                    m: f"{group_name}/{stem}_trajcolormap_{m}.png"
                    for m in colormap_metrics
                    if (group_dir / f"{stem}_trajcolormap_{m}.png").is_file()
                }

                items.append(
                    {
                        "group": group_name,
                        "route": r["route"],
                        "segment": f"[{start},{end}]",
                        "n_steps_run": r.get("n_steps_run", 0),
                        "terminated": r.get("terminated", ""),
                        "route_completion": round(r.get("route_completion", 0.0), 3),
                        "n_collision_events": extract_score(r, "total_collision_events") or 0,
                        "n_curb_hits": extract_score(r, "total_curb_hits") or 0,
                        "n_snaps": extract_score(r, "total_snaps") or 0,
                        "n_red_light_violations": extract_score(r, "total_red_light_violations")
                        or 0,
                        "n_strong_brakes": extract_score(r, "total_strong_brakes") or 0,
                        "progress_m": round(r.get("progress_m", 0.0), 1),
                        "video_path": f"{group_name}/{video_name}"
                        if video_path.is_file()
                        else None,
                        "colormap_paths": colormap_paths,
                    }
                )
    return items, summaries


_TEMPLATE_PATH = Path(__file__).with_name("closed_loop_report_template.html")


def build_html_report(
    out_root: str | Path,
    group_names: list[str],
    *,
    title: str = "Per-Group Closed-Loop Evaluation",
    subtitle: str = "",
    report_filename: str = "report.html",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> Path | None:
    """Write the self-contained gallery HTML to ``<out_root>/<report_filename>``.

    Returns the written path, or ``None`` if no group had a readable ``summary.json``
    (nothing to report yet). ``colormap_metrics`` picks which per-step metrics get a
    trajectory colormap image rendered (default: all of them) — each episode card shows
    one image at a time with a dropdown to switch between the metrics that rendered for
    it, see :mod:`scenario_generation.trajectory_colormap`.
    """
    out_root = Path(out_root)
    items, summaries = collect_group_data(out_root, group_names, colormap_metrics=colormap_metrics)
    if not summaries:
        return None

    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__TITLE__", title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__DEFAULT_METRIC_JSON__", json.dumps(_DEFAULT_METRIC))
    html = html.replace("__ITEMS_JSON__", json.dumps(items, ensure_ascii=False))
    html = html.replace("__SUMMARY_JSON__", json.dumps(summaries, ensure_ascii=False))
    html = html.replace("__PALETTE_JSON__", json.dumps(_PALETTE))

    out_path = out_root / report_filename
    out_path.write_text(html, encoding="utf-8")
    return out_path
