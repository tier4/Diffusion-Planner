"""Run ``valid_predictor_closed_loop.py`` once per group.

``--groups_npz_root`` accepts multiple input formats:
- Folder path: treated as one route directory
- Flat JSON path-list: content is `["/path/to/route1", "/path/to/route2"]` (uses site discovery for group inference)
- Grouped JSON path: content is `{"g1": ["/path/to/route1"], "g2": ["/path/to/route2", ...]}`
- Any combination of the above

Outputs land in ``<out_root>/<group_name>/`` (summary.json,
segments.jsonl, videos). A ``groups_summary.json`` at ``<out_root>``
records which npz_root was used for which group name, for downstream reporting.

Example::

    python diffusion_planner/run_all_groups_closed_loop.py \\
        --groups_npz_root /data/folder \\
        --model_path /media/.../best_model.pth \\
        --out_root /media/.../cl_results/all_groups \\
        --near_miss_thresh 0.3 --replan_interval 10 --draw_every 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scenario_generation.closed_loop_html_report import build_html_report


def resolve_closed_loop_inputs(inputs: str | list[str]) -> dict[str, list[str]]:
    """Resolve multiple input paths to a standardized {group_name: [route_dirs]} dict.

    Supports:
    - Folder path: treated as one route directory
    - Flat JSON path-list: `["/path/to/route1", ...]` -> use discover_sites_from_json for site inference
    - Grouped JSON: `{"g1": ["/path/to/route1"], "g2": ["/path/to/route2"]}` -> keys become group names
    - Mixed inputs: each becomes its own group
    - Single input: auto-wrapped to list

    The returned paths are route directories (containing .npz files), not individual .npz files.
    The actual .npz enumeration is done by closed_loop_eval.enumerate_routes().

    Args:
        inputs: Single path or list of paths (folders, JSON files, or grouped JSONs)

    Returns:
        Dict mapping group_name to list of route directory paths
    """
    from scenario_generation.site_discovery import discover_sites_from_json

    # Normalize to list
    if isinstance(inputs, str):
        inputs = [inputs]

    result: dict[str, list[str]] = {}

    for input_path in inputs:
        p = Path(input_path)

        if not p.exists():
            print(f"Warning: {input_path} does not exist, skipping", file=sys.stderr)
            continue

        if p.is_dir():
            # Folder: treat as one route directory
            name = p.name
            if name not in result:
                result[name] = []
            result[name].append(str(p))

        elif p.suffix == ".json":
            with open(p, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Grouped JSON: {"g1": ["/path/to/route1"], "g2": ["/path/to/route2"]}
                for group_name, paths in data.items():
                    if group_name not in result:
                        result[group_name] = []
                    for item in paths:
                        result[group_name].append(str(Path(item)))

            elif isinstance(data, list):
                # Flat JSON: use discover_sites_from_json for site inference (legacy compatibility)
                # discover_sites_from_json returns {site_name: [route_dir_paths]}
                try:
                    sites = discover_sites_from_json(p)
                    for group_name, paths in sites.items():
                        if group_name not in result:
                            result[group_name] = []
                        for path in paths:
                            result[group_name].append(str(path))
                except Exception:
                    # Fallback: treat as plain list of route directories
                    for item in data:
                        item_p = Path(item)
                        # Infer group name from path structure (parent of split dir)
                        parts = item_p.parts
                        group_name = None
                        for i, part in enumerate(parts):
                            if i > 0 and part in ("valid", "manual", "auto"):
                                group_name = parts[i - 1]
                                break
                        if not group_name:
                            group_name = item_p.stem

                        if group_name not in result:
                            result[group_name] = []
                        result[group_name].append(str(item_p))

    return result


def run_one_group(
    model,  # PyTorch model (for train.py direct call)
    npz_root_list: list[str],
    out_dir: str | Path,
    args: argparse.Namespace,
    group_name: str | None = None,
    mode: str = "objects",  # Single mode: "objects" or "noobj"
) -> tuple[dict, dict, str]:
    """Run closed-loop evaluation for a single group and a single mode.

    This function can be called in two ways:
    1. From train.py: model is a PyTorch nn.Module, uses FullRouteClosedLoopEvaluation directly
    2. From CLI subprocess: model is None, calls valid_predictor_closed_loop.py via subprocess

    Args:
        model: PyTorch model (None for subprocess mode)
        npz_root_list: List of route directory paths (already parsed by resolve_closed_loop_inputs)
        out_dir: Output directory for this group
        args: CLI args or Namespace with device, fps, search_radius, etc.
        group_name: Label for this group (used in logs)
        mode: Single mode to run: "objects" or "noobj"

    Returns:
        (log, summary, label) tuple
    """
    from scenario_generation.closed_loop_evaluation import (
        ClosedLoopEvalConfig,
        FullRouteClosedLoopEvaluation,
        RolloutParams,
    )
    from scenario_generation.wandb_closed_loop import build_full_closed_loop_wandb_log

    # Normalize inputs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine label for this mode
    label = group_name if mode == "objects" else f"{group_name}__noobj"
    mode_out_dir = out_dir / label
    mode_out_dir.mkdir(parents=True, exist_ok=True)

    # npz_root_list is already parsed by resolve_closed_loop_inputs
    # Write npz roots to JSON for valid_predictor_closed_loop.py
    if len(npz_root_list) > 1:
        npz_root_arg = mode_out_dir / "npz_roots.json"
        npz_root_arg.write_text(json.dumps([str(p) for p in npz_root_list]))
    elif len(npz_root_list) == 1 and Path(npz_root_list[0]).is_dir():
        npz_root_arg = npz_root_list[0]
    else:
        npz_root_arg = npz_root_list[0]

    # Determine drop_objects flag
    drop_objects = mode == "noobj"

    if model is not None:
        # Direct call from train.py: use FullRouteClosedLoopEvaluation directly
        seg_len = getattr(args, "closed_loop_seg_len", 100000)
        fps = float(getattr(args, "closed_loop_fps", 10))

        evaluator = FullRouteClosedLoopEvaluation(
            model,
            args,
            ClosedLoopEvalConfig(
                out_dir=mode_out_dir,
                params=RolloutParams(
                    device=args.device,
                    near_miss_thresh=getattr(args, "closed_loop_near_miss_thresh", 0.5),
                    search_radius=getattr(args, "closed_loop_search_radius", 1.5),
                    warmup_steps=getattr(args, "closed_loop_warmup_steps", 0),
                    unstick_after=getattr(args, "closed_loop_unstick_after", 300),
                    unstick_advance_m=getattr(args, "closed_loop_unstick_advance_m", 5.0),
                    unstick_radius_mult=getattr(args, "closed_loop_unstick_radius_mult", 10.0),
                    unstick_teleport_after=getattr(args, "closed_loop_unstick_teleport_after", 300),
                    draw_every=getattr(args, "closed_loop_draw_every", 4) if getattr(args, "render_media", False) else None,
                    replan_interval=getattr(args, "closed_loop_replan_interval", 4),
                    abort_deviation_m=getattr(args, "closed_loop_abort_deviation_m", 50.0),
                    abort_after=getattr(args, "closed_loop_abort_after", 30),
                    abort_max_snaps=getattr(args, "closed_loop_abort_max_snaps", 0),
                    drop_objects=drop_objects,
                ),
                fps=fps,
                verbose=False,
            ),
            npz_root_arg,
            seg_len=seg_len,
        )
        summary = evaluator.run()

        if not summary:
            return {}, {}, label

        render_media = getattr(args, "render_media", True)
        group_log = build_full_closed_loop_wandb_log(
            summary,
            out_dir=str(mode_out_dir),
            group=label,
            video_pick=getattr(args, "closed_loop_wandb_video_pick", "worst"),
            colormap_metrics=tuple(getattr(args, "closed_loop_colormap_metrics", [])),
            near_miss_thresh=getattr(args, "closed_loop_near_miss_thresh", 0.5),
            report_base_url=getattr(args, "closed_loop_report_base_url", None),
            render_media=render_media,
        )

        return group_log, summary, label
    else:
        # Subprocess call from CLI
        import os

        print(f"=== [{label}] npz_root={npz_root_arg} -> out_dir={mode_out_dir} ===")

        cli_path = Path(__file__).resolve().parent / "valid_predictor_closed_loop.py"
        cmd = [
            sys.executable,
            str(cli_path),
            "--model_path",
            str(args.model_path),
            "--npz_root",
            str(npz_root_arg),
            "--out_dir",
            str(mode_out_dir),
        ]
        if hasattr(args, "extra_args") and args.extra_args:
            cmd.extend(args.extra_args)
        if mode == "noobj":
            cmd.append("--drop_objects")

        error = None
        for attempt in range(1, 3):
            try:
                subprocess.run(cmd, check=True)
                error = None
                break
            except subprocess.CalledProcessError as e:
                error = e
                print(f"  [{label}] attempt {attempt}/2 failed: {e}", file=sys.stderr)

        summary_path = mode_out_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None

        return {}, summary, label


def update_groups_summary(out_dir: Path | str, label: str, npz_root: list[str], mode_out_dir: Path | str, summary: dict | None) -> None:
    """Update groups_summary.json with results from a single mode run."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "groups_summary.json"

    merged = {}
    if manifest_path.is_file():
        merged = json.loads(manifest_path.read_text())

    merged[label] = {
        "npz_root": npz_root,
        "out_dir": str(mode_out_dir),
        "summary": summary,
    }

    manifest_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups_npz_root",
        required=True,
        nargs="+",
        help="Input paths: folder(s) (route dirs containing .npz files), flat JSON(s), grouped JSON(s), "
        "or any combination. Each path becomes its own group.",
    )
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--only_groups", nargs="*", default=None, help="run only these group names")
    parser.add_argument(
        "--object_modes",
        nargs="+",
        choices=("objects", "noobj"),
        default=["objects", "noobj"],
        help="run each group once per mode: 'objects'=normal, 'noobj'=empty-world ablation "
        "(--drop_objects, no dynamic/static objects, map kept). Output/group label for noobj "
        "gets a '__noobj' suffix so both show up as separate groups — overlaid per metric in "
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
        help="optional: log this evaluation run to wandb (one run, all groups as "
        "closed_loop/<group>/... + cross-group aggregate + a link to the local report.html), "
        "so separate model/dataset iterations can be tracked and compared as wandb runs "
        "the same way training epochs are.",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_video_pick",
        choices=("worst", "first", "longest"),
        default="worst",
        help="which single episode per group gets its video + trajectory colormap uploaded",
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
    groups = resolve_closed_loop_inputs(args.groups_npz_root)
    if args.only_groups:
        groups = {name: paths for name, paths in groups.items() if name in args.only_groups}
    if not groups:
        print(f"No groups with npz found under {args.groups_npz_root}", file=sys.stderr)
        return 1

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Run each group and each mode separately
    for group_name, npz_paths in groups.items():
        group_out_dir = args.out_root / group_name if group_name else args.out_root

        for mode in args.object_modes:
            mode_label = group_name if mode == "objects" else f"{group_name}__noobj"
            mode_out_dir = group_out_dir / mode_label

            # run_one_group: single group, single mode
            _, summary, returned_label = run_one_group(
                None,  # CLI mode: use subprocess
                npz_paths,
                group_out_dir,
                args,
                group_name=group_name,
                mode=mode,
            )

            # Update groups_summary.json with this mode's result
            update_groups_summary(
                group_out_dir,
                returned_label,
                npz_paths,
                mode_out_dir,
                summary,
            )

    # Read merged results for reporting
    manifest_path = args.out_root / "groups_summary.json"
    merged = {}
    if manifest_path.is_file():
        merged = json.loads(manifest_path.read_text())

    all_group_names = sorted(name for name, r in merged.items() if r.get("summary") is not None)

    # Build HTML report
    report_path = None
    if not args.no_report and all_group_names:
        from scenario_generation.closed_loop_html_report import build_html_report
        report_path = build_html_report(
            args.out_root,
            all_group_names,
            title="Per-Group Closed-Loop Evaluation",
            subtitle=f"groups_npz_root={args.groups_npz_root}",
            colormap_metrics=tuple(args.report_colormap_metrics),
        )
        if report_path:
            print(f"Wrote {report_path}")

    # Log to wandb
    if args.wandb_project and all_group_names:
        _log_to_wandb(args, all_group_names, merged, report_path)

    return 0


def _log_to_wandb(
    args: argparse.Namespace,
    group_names: list[str],
    merged: dict,
    report_path: Path | None,
) -> None:
    """One evaluation-only wandb run covering every group in this call."""
    import wandb

    from scenario_generation.wandb_closed_loop import (
        build_combined_episode_table,
        build_full_closed_loop_wandb_log,
        build_groups_aggregate_log,
        resolve_report_link,
    )

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    try:
        log: dict = {}
        group_summaries: dict[str, dict] = {}
        episode_data: list = []  # (group, rows, out_dir) for the ONE combined table

        for group_name in group_names:
            r = merged.get(group_name) or {}
            summary = r.get("summary")
            if not summary:
                continue
            group_out_dir = r.get("out_dir") or str(args.out_root / group_name)
            segments_path = Path(group_out_dir) / "segments.jsonl"
            rows = []
            if segments_path.is_file():
                with segments_path.open(encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            summary_with_rows = {**summary, "segments": rows}
            log.update(
                build_full_closed_loop_wandb_log(
                    summary_with_rows,
                    out_dir=group_out_dir,
                    group=group_name,
                    video_pick=args.wandb_video_pick,
                    colormap_metrics=tuple(args.wandb_colormap_metrics),
                    near_miss_thresh=summary.get("near_miss_thresh", 0.5),
                    report_base_url=args.report_base_url,
                )
            )
            group_summaries[group_name] = summary_with_rows
            episode_data.append((group_name, rows, group_out_dir))

        if episode_data:
            log["closed_loop_episodes/all"] = build_combined_episode_table(episode_data)
        if len(group_summaries) > 1:
            log.update(build_groups_aggregate_log(group_summaries))
        if report_path is not None:
            log["closed_loop_links/report"] = resolve_report_link(
                args.out_root, args.report_base_url
            )
        wandb.log(log)
        print(f"wandb: logged {len(group_summaries)} group(s) to run {run.id}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    raise SystemExit(main())
