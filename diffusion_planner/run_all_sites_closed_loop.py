"""Run ``valid_predictor_closed_loop.py`` once per site in a curated JSON manifest.

``--sites_npz_root`` is a curated ``.json`` path-list file (the same format as
``--npz_root``'s JSON-list convention, e.g. ``path_list_closed_loop.json``) --
entries are grouped into per-site route pools by
``site_discovery.discover_sites_with_vehicles_from_json``. A "site" is the path
component immediately before the first recognized split dir
 in each entry, matching the existing
``{project}/{area_map_id}_{area_map_name}/{split}/...``. Each site's routes are never
grouped across sites — this avoids the cross-directory filename-collision failure
mode of pointing --npz_root at a shared parent that mixes multiple sites' npz
together.

If ``--project_vehicle_map`` is given (a JSON file of ``{project_code_name:
vehicle_type_label}``), each site also gets a ``vehicle_type`` label for
reporting/filtering -- it doesn't change what gets simulated. Without it sites are
left unlabelled, and ``--only_vehicle_types`` has nothing to filter on.

Per-site outputs land in ``<out_root>/<site_name>/`` (summary.json,
segments.jsonl, videos), exactly as a standalone single-site run would
produce. A ``sites_summary.json`` at ``<out_root>`` records which npz_root and
vehicle_type were used for which site name, for downstream reporting.

Example::

    python diffusion_planner/run_all_sites_closed_loop.py \\
        --sites_npz_root /media/.../path_list_closed_loop.json \\
        --model_path /media/.../best_model.pth \\
        --out_root /media/.../cl_results/all_sites \\
        --near_miss_thresh 0.3 --replan_interval 10 --draw_every 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scenario_generation.closed_loop_html_report import build_html_report
from scenario_generation.site_discovery import discover_sites_with_vehicles_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites_npz_root",
        required=True,
        type=Path,
        help="curated .json path-list manifest, grouped into per-site route pools by "
        "site_discovery.discover_sites_from_json",
    )
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--only_sites", nargs="*", default=None, help="run only these site names")
    parser.add_argument(
        "--project_vehicle_map",
        type=Path,
        default=None,
        help="optional JSON file of {project_code_name: vehicle_type_label} for "
        "labeling/filtering sites by vehicle type",
    )
    parser.add_argument(
        "--only_vehicle_types",
        nargs="*",
        default=None,
        help="run only sites whose resolved vehicle type is in this list. Requires "
        "--project_vehicle_map: without it sites carry no vehicle label at all",
    )
    parser.add_argument(
        "--object_modes",
        nargs="+",
        choices=("objects", "noobj"),
        default=["objects", "noobj"],
        help="run each site once per mode: 'objects'=normal, 'noobj'=empty-world ablation "
        "(--drop_objects, no dynamic/static objects, map kept). Output/site label for noobj "
        "gets a '__noobj' suffix so both show up as separate sites — overlaid per metric in "
        "W&B, side-by-side rows in the local report.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="passed through verbatim to valid_predictor_closed_loop.py (e.g. --near_miss_thresh 0.3)",
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="skip writing the local HTML gallery (report.html) at --out_root",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="optional: log this evaluation run to wandb (one run, all sites as "
        "closed_loop/<site>/... + cross-site aggregate + a link to the local report.html), "
        "so separate model/dataset iterations can be tracked and compared as wandb runs "
        "the same way training epochs are.",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_video_pick",
        choices=("worst", "first", "longest"),
        default="worst",
        help="which single episode per site gets its video + trajectory colormap uploaded",
    )
    parser.add_argument(
        "--wandb_colormap_metrics",
        nargs="*",
        choices=(
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ),
        default=[
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ],
        help="per-step metrics rendered as trajectory-colormap images for the wandb "
        "representative episode (default: all — cheap images, unlike video/episode picking)",
    )
    parser.add_argument(
        "--report_colormap_metrics",
        nargs="*",
        choices=(
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ),
        default=[
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ],
        help="per-step metrics rendered as trajectory-colormap images in the local HTML "
        "report (every episode, not just the wandb representative one)",
    )
    parser.add_argument(
        "--report_base_url",
        type=str,
        default=None,
        help="if --out_root is served over HTTP from this base, wandb records a clickable "
        "report URL instead of just the local path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.only_vehicle_types and not args.project_vehicle_map:
        # Without a map no site carries a vehicle label, so the filter would drop everything
        # and report it as "no sites found" -- say what actually went wrong instead.
        print(
            "--only_vehicle_types requires --project_vehicle_map (sites are unlabelled without it)",
            file=sys.stderr,
        )
        return 2
    project_vehicle_map = None
    if args.project_vehicle_map:
        project_vehicle_map = json.loads(args.project_vehicle_map.read_text())
    sites = discover_sites_with_vehicles_from_json(args.sites_npz_root, project_vehicle_map)
    if args.only_sites:
        sites = {name: info for name, info in sites.items() if name in args.only_sites}
    if args.only_vehicle_types:
        sites = {
            name: info
            for name, info in sites.items()
            if info["vehicle_type"] in args.only_vehicle_types
        }
    if not sites:
        print(f"No sites with npz found under {args.sites_npz_root}", file=sys.stderr)
        return 1

    cli_path = Path(__file__).resolve().parent / "valid_predictor_closed_loop.py"
    args.out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for site_name, info in sites.items():
        npz_root = info["npz_roots"]
        vehicle_type = info["vehicle_type"]
        for mode in args.object_modes:
            # "noobj" gets a distinct site label (suffix) rather than a separate axis — this
            # lets it ride the existing per-site machinery (HTML rows/cards, W&B per-site keys,
            # metric_regex overlay) completely unchanged.
            label = site_name if mode == "objects" else f"{site_name}__noobj"
            site_out_dir = args.out_root / label
            site_out_dir.mkdir(parents=True, exist_ok=True)
            # A site may span several curated roots (e.g. multiple date/time entries) --
            # valid_predictor_closed_loop.py's --npz_root takes one path, so a multi-root site
            # is handed through as a small JSON path-list file (the same convention
            # resolve_npz_roots already reads for a single --npz_root).
            npz_root_arg = site_out_dir / "npz_roots.json"
            npz_root_arg.write_text(json.dumps([str(p) for p in npz_root]))
            print(
                f"=== [{label}] vehicle_type={vehicle_type} npz_root={npz_root} "
                f"-> out_dir={site_out_dir} ==="
            )
            cmd = [
                sys.executable,
                str(cli_path),
                "--model_path",
                str(args.model_path),
                "--npz_root",
                str(npz_root_arg),
                "--out_dir",
                str(site_out_dir),
                *args.extra_args,
            ]
            if mode == "noobj":
                cmd.append("--drop_objects")
            # One site's failure (e.g. a transient disk I/O error partway through a long route
            # sweep) must not abort the remaining sites — each site is an independent evaluation.
            error = None
            for attempt in range(1, 3):
                try:
                    subprocess.run(cmd, check=True)
                    error = None
                    break
                except subprocess.CalledProcessError as e:
                    error = e
                    print(f"  [{label}] attempt {attempt}/2 failed: {e}", file=sys.stderr)
            summary_path = site_out_dir / "summary.json"
            results[label] = {
                "npz_root": [str(p) for p in npz_root],
                "vehicle_type": vehicle_type,
                "out_dir": str(site_out_dir),
                "error": str(error) if error else None,
                "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
            }

    # Merge into any existing manifest instead of overwriting it outright, so a targeted
    # --only_sites re-run doesn't drop the other sites' entries from a prior full run.
    manifest_path = args.out_root / "sites_summary.json"
    merged = {}
    if manifest_path.is_file():
        merged = json.loads(manifest_path.read_text())
    merged.update(results)
    manifest_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\nWrote {manifest_path}")
    for site_name, r in results.items():
        s = r["summary"] or {}
        obj = s.get("object") or {}
        print(
            f"  {site_name} [{r.get('vehicle_type')}]: n_routes={s.get('n_routes')} "
            f"collision_segment_rate={obj.get('collision_segment_rate')} "
            f"near_miss_segment_rate={obj.get('miss_segment_rate')}"
        )

    # All site names with a summary (from THIS run's `merged`, so a targeted --only_sites
    # re-run's report still covers every previously-completed site, not just the ones just run).
    all_site_names = sorted(name for name, r in merged.items() if r.get("summary") is not None)
    site_vehicle_types = {name: r.get("vehicle_type") for name, r in merged.items()}
    report_path = None
    if not args.no_report and all_site_names:
        report_path = build_html_report(
            args.out_root,
            all_site_names,
            title="Per-Site Closed-Loop Evaluation",
            subtitle=f"sites_npz_root={args.sites_npz_root}",
            colormap_metrics=tuple(args.report_colormap_metrics),
            site_vehicle_types=site_vehicle_types,
        )
        if report_path:
            print(f"Wrote {report_path}")

    if args.wandb_project:
        _log_to_wandb(args, all_site_names, merged, report_path)
    return 0


def _log_to_wandb(
    args: argparse.Namespace,
    site_names: list[str],
    merged: dict,
    report_path: Path | None,
) -> None:
    """One evaluation-only wandb run covering every site in this call — lets separate
    model/dataset iterations (e.g. this vs. a previous checkpoint) be tracked and compared
    as wandb runs, the same way training epochs are (see closed_loop_validate in train.py,
    which this mirrors: per-site scalars/table/representative video, cross-site aggregate,
    a link to the local report instead of uploading every video).
    """
    import wandb

    from scenario_generation.wandb_closed_loop import (
        build_combined_episode_table,
        build_full_closed_loop_wandb_log,
        build_sites_aggregate_log,
        resolve_report_link,
    )

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    try:
        log: dict = {}
        site_summaries: dict[str, dict] = {}
        site_vehicle_types: dict[str, str] = {}
        episode_data: list = []  # (site, rows, out_dir, vehicle_type) for the ONE combined table
        for site_name in site_names:
            r = merged.get(site_name) or {}
            summary = r.get("summary")
            if not summary:
                continue
            vehicle_type = r.get("vehicle_type")
            site_out_dir = r.get("out_dir") or str(args.out_root / site_name)
            segments_path = Path(site_out_dir) / "segments.jsonl"
            rows = []
            if segments_path.is_file():
                with segments_path.open(encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            summary_with_rows = {**summary, "segments": rows}
            # build_full returns final section-based keys (closed_loop_scores/... etc.) — merge as-is.
            log.update(
                build_full_closed_loop_wandb_log(
                    summary_with_rows,
                    out_dir=site_out_dir,
                    site=site_name,
                    video_pick=args.wandb_video_pick,
                    colormap_metrics=tuple(args.wandb_colormap_metrics),
                    near_miss_thresh=summary.get("near_miss_thresh", 0.5),
                    report_base_url=args.report_base_url,
                )
            )
            site_summaries[site_name] = summary_with_rows
            if vehicle_type:
                site_vehicle_types[site_name] = vehicle_type
            episode_data.append((site_name, rows, site_out_dir, vehicle_type))
        if episode_data:
            log["closed_loop_episodes/all"] = build_combined_episode_table(episode_data)
        if len(site_summaries) > 1:
            log.update(build_sites_aggregate_log(site_summaries, site_vehicle_types))
        if report_path is not None:
            log["closed_loop_links/report"] = resolve_report_link(
                args.out_root, args.report_base_url
            )
        wandb.log(log)
        print(f"wandb: logged {len(site_summaries)} site(s) to run {run.id}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    raise SystemExit(main())
