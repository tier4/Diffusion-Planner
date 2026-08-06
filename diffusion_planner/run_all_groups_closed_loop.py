"""Run ``valid_predictor_closed_loop.py`` once per group.

``--groups_npz_root`` accepts multiple input formats:
- Folder path: treated as one route directory -> `{folder_name: {"all": [paths]}}`
- Flat JSON (list): `["/path/to/route1", ...]` -> `{json_stem: {"all": [paths]}}`
- Grouped JSON (dict): `{"g1": [...], "g2": [...]}` -> `{json_stem: {g1: [paths], g2: [paths]}}`

Outputs land in ``<out_root>/<json_name>/<group_name>/`` (summary.json,
segments.jsonl, videos). A ``groups_summary.json`` at each ``<out_root>/<json_name>/``
records results for all groups in that JSON, keyed as ``<json_name>/<group_name>``.
The root ``<out_root>/groups_summary.json`` aggregates all JSONs for reporting.

Object mode suffix: ``objects`` = normal, ``noobj`` = empty-world.
- Directory: ``<out_root>/<json_name>/<group_name>/``
- WANDB key: ``closed_loop/<metric>/<json_name>/<group_name>/``

Example::

    python diffusion_planner/run_all_groups_closed_loop.py \\
        --groups_npz_root override.json site.json \\
        --model_path /media/.../best_model.pth \\
        --out_root /media/.../cl_results \\
        --near_miss_thresh 0.3 --replan_interval 10 --draw_every 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from scenario_generation.closed_loop_html_report import build_html_report


class GroupEntry(TypedDict):
    npz_root: list[str]
    out_dir: str
    summary: dict | None


class GroupsSummary(TypedDict):
    """Keyed as '<json_name>/<group_name>' (mode suffix applied at out_dir level)."""

    npz_root: list[str]
    out_dir: str
    summary: dict | None


def resolve_closed_loop_inputs(inputs: str | list[str]) -> dict[str, dict[str, list[str]]]:
    """Resolve multiple input paths to ``{<json_name>: {<group_name>: [route_dirs]}}``.

    The top-level key is the JSON filename (without extension) or folder name.
    For folder or flat JSON (list), the inner key is always ``"all"``.
    For grouped JSON (dict), the inner keys are the group names from the JSON.

    Args:
        inputs: Single path or list of paths (folders, JSON files).

    Returns:
        Dict mapping ``{json_name: {group_name: [route_dir_paths]}}``.
        Route dirs are the leaves; .npz enumeration is done by closed_loop_eval.enumerate_routes().
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    result: dict[str, dict[str, list[str]]] = {}

    for input_path in inputs:
        p = Path(input_path)

        if not p.exists():
            print(f"Warning: {input_path} does not exist, skipping", file=sys.stderr)
            continue

        if p.is_dir():
            # Folder: treat as one route directory
            name = p.name
            if name not in result:
                result[name] = {}
            if "all" not in result[name]:
                result[name]["all"] = []
            result[name]["all"].append(str(p))

        elif p.suffix == ".json":
            name = p.stem
            if name not in result:
                result[name] = {}

            with open(p, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Grouped JSON: {"g1": [...], "g2": [...]}
                for group_name, paths in data.items():
                    if group_name not in result[name]:
                        result[name][group_name] = []
                    for item in paths:
                        result[name][group_name].append(str(Path(item)))

            elif isinstance(data, list):
                # Flat JSON (list): ["path1", "path2", ...]
                if "all" not in result[name]:
                    result[name]["all"] = []
                for item in data:
                    result[name]["all"].append(str(Path(item)))

    return result


def run_one_group(
    model,  # PyTorch model (for train.py direct call) or None for subprocess
    npz_root_list: list[str],
    out_dir: str | Path,
    args: argparse.Namespace,
    group_name: str | None = None,
) -> tuple[dict, dict | None, str]:
    """Run closed-loop evaluation for a single group (mode handled at caller level).

    Args:
        model: PyTorch model (None for subprocess mode).
        npz_root_list: List of route directory paths.
        out_dir: Output directory for this group (already includes mode suffix).
        args: CLI args or Namespace.
        group_name: Label for this group (used in logs).

    Returns:
        (log, summary, label) tuple.
    """
    from scenario_generation.closed_loop_evaluation import (
        ClosedLoopEvalConfig,
        FullRouteClosedLoopEvaluation,
        RolloutParams,
    )
    from scenario_generation.wandb_closed_loop import build_full_closed_loop_wandb_log

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = group_name or out_dir.name

    # Write npz roots to JSON if multiple, otherwise use as-is
    if len(npz_root_list) > 1:
        npz_root_arg = out_dir / "_npz_roots.json"
        npz_root_arg.write_text(json.dumps([str(p) for p in npz_root_list]))
    else:
        npz_root_arg = npz_root_list[0]

    if model is not None:
        seg_len = getattr(args, "closed_loop_seg_len", 100000)
        fps = float(getattr(args, "closed_loop_fps", 10))
        drop_objects = "__noobj" in str(out_dir)

        evaluator = FullRouteClosedLoopEvaluation(
            model,
            args,
            ClosedLoopEvalConfig(
                out_dir=out_dir,
                params=RolloutParams(
                    device=args.device,
                    near_miss_thresh=getattr(args, "closed_loop_near_miss_thresh", 0.5),
                    search_radius=getattr(args, "closed_loop_search_radius", 1.5),
                    warmup_steps=getattr(args, "closed_loop_warmup_steps", 0),
                    unstick_after=getattr(args, "closed_loop_unstick_after", 300),
                    unstick_advance_m=getattr(args, "closed_loop_unstick_advance_m", 5.0),
                    unstick_radius_mult=getattr(args, "closed_loop_unstick_radius_mult", 10.0),
                    unstick_teleport_after=getattr(args, "closed_loop_unstick_teleport_after", 300),
                    draw_every=getattr(args, "closed_loop_draw_every", 4)
                    if getattr(args, "render_media", False)
                    else None,
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
            return {}, None, label

        render_media = getattr(args, "render_media", True)
        group_log = build_full_closed_loop_wandb_log(
            summary,
            out_dir=str(out_dir),
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
        cli_path = Path(__file__).resolve().parent / "valid_predictor_closed_loop.py"
        cmd = [
            sys.executable,
            str(cli_path),
            "--model_path",
            str(args.model_path),
            "--npz_root",
            str(npz_root_arg),
            "--out_dir",
            str(out_dir),
        ]
        if hasattr(args, "extra_args") and args.extra_args:
            cmd.extend(args.extra_args)
        if "__noobj" in str(out_dir):
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

        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        return {}, summary, label


def _make_summary_key(json_name: str, group_name: str, mode: str) -> str:
    """Build the summary key, e.g. 'override/departure' or 'site/all'."""
    return f"{json_name}/{group_name}"


def update_groups_summary(
    out_dir: Path | str,
    summary_key: str,
    npz_root: list[str],
    mode_out_dir: Path | str,
    summary: dict | None,
) -> None:
    """Update groups_summary.json at <out_dir>/ with one mode's result.

    summary_key is e.g. 'override/departure' or 'site/all'.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "groups_summary.json"

    merged: dict[str, dict] = {}
    if manifest_path.is_file():
        merged = json.loads(manifest_path.read_text())

    merged[summary_key] = {
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
        help="Input paths: folder(s), flat JSON(s) (list of paths), or grouped JSON(s) (dict). "
        "Each JSON/folder becomes its own top-level namespace.",
    )
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument(
        "--only_json",
        nargs="*",
        default=None,
        help="run only these JSON/folder names (e.g. override site)",
    )
    parser.add_argument(
        "--object_modes",
        nargs="+",
        choices=("objects", "noobj"),
        default=["objects", "noobj"],
        help="'objects'=normal, 'noobj'=empty-world ablation (--drop_objects). "
        "Each group runs once per mode; noobj gets a '__noobj' suffix on the out_dir.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="passed through verbatim to valid_predictor_closed_loop.py",
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="skip writing the local HTML gallery at --out_root",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="optional: log to wandb (one run, all groups + per-json aggregates)",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_video_pick",
        choices=("worst", "first", "longest"),
        default="worst",
        help="which episode gets its video uploaded per group",
    )
    parser.add_argument(
        "--wandb_colormap_metrics",
        nargs="*",
        default=[
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ],
        help="per-step metrics rendered as trajectory-colormap images for wandb",
    )
    parser.add_argument(
        "--report_colormap_metrics",
        nargs="*",
        default=[
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ],
        help="per-step metrics rendered in the local HTML report",
    )
    parser.add_argument(
        "--report_base_url",
        type=str,
        default=None,
        help="if --out_root is served over HTTP, wandb records a clickable report URL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = resolve_closed_loop_inputs(args.groups_npz_root)

    if args.only_json:
        resolved = {k: v for k, v in resolved.items() if k in args.only_json}
    if not resolved:
        print(f"No inputs found under {args.groups_npz_root}", file=sys.stderr)
        return 1

    args.out_root.mkdir(parents=True, exist_ok=True)

    all_summary_keys: list[str] = []  # for reporting

    # Triple loop: json_name -> group_name -> mode
    for json_name, groups in resolved.items():
        json_out_dir = args.out_root / json_name
        json_out_dir.mkdir(parents=True, exist_ok=True)

        for group_name, npz_paths in groups.items():
            for mode in args.object_modes:
                # Label and out_dir
                if mode == "objects":
                    mode_out_dir = json_out_dir / group_name
                    summary_key = _make_summary_key(json_name, group_name, mode)
                    label = summary_key  # e.g. "override/departure"
                else:  # noobj
                    json_mode_name = f"{json_name}__noobj"
                    mode_out_dir = args.out_root / json_mode_name / group_name
                    summary_key = _make_summary_key(json_mode_name, group_name, mode)
                    label = summary_key  # e.g. "override__noobj/departure"

                print(f"=== [{label}] npz={npz_paths} -> out={mode_out_dir} ===")

                _, summary, _ = run_one_group(
                    None,  # CLI mode: subprocess
                    npz_paths,
                    mode_out_dir,
                    args,
                    group_name=label,
                )

                # Update groups_summary.json at json_out_dir (or args.out_root for __noobj)
                summary_out_dir = json_out_dir if mode == "objects" else args.out_root / f"{json_name}__noobj"
                update_groups_summary(
                    summary_out_dir,
                    summary_key,
                    npz_paths,
                    mode_out_dir,
                    summary,
                )
                all_summary_keys.append(summary_key)

    # Merge all per-json groups_summary.json into root groups_summary.json
    root_manifest: dict[str, dict] = {}
    for json_name in resolved:
        for mode in args.object_modes:
            if mode == "objects":
                manifest_path = args.out_root / json_name / "groups_summary.json"
            else:
                manifest_path = args.out_root / f"{json_name}__noobj" / "groups_summary.json"

            if manifest_path.is_file():
                partial = json.loads(manifest_path.read_text())
                root_manifest.update(partial)

    root_manifest_path = args.out_root / "groups_summary.json"
    root_manifest_path.write_text(json.dumps(root_manifest, indent=2, ensure_ascii=False))

    all_group_names = sorted(k for k, v in root_manifest.items() if v.get("summary") is not None)

    # Build HTML report
    report_path = None
    if not args.no_report and all_group_names:
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
        _log_to_wandb(args, all_group_names, root_manifest, report_path)

    return 0


def _log_to_wandb(
    args: argparse.Namespace,
    group_names: list[str],
    merged: dict,
    report_path: Path | None,
) -> None:
    """One wandb run: closed_loop/<metric>/<json_name>/<group_name> + per-json aggregates."""
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
        episode_data: list = []

        for group_name in group_names:
            r = merged.get(group_name) or {}
            summary = r.get("summary")
            if not summary:
                continue
            group_out_dir = r.get("out_dir") or ""
            segments_path = Path(group_out_dir) / "segments.jsonl"
            rows = []
            if segments_path.is_file():
                with segments_path.open(encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            summary_with_rows = {**summary, "segments": rows}

            wandb_key_prefix = group_name  # e.g. "override/departure"
            log.update(
                build_full_closed_loop_wandb_log(
                    summary_with_rows,
                    out_dir=group_out_dir,
                    group=group_name,
                    video_pick=args.wandb_video_pick,
                    colormap_metrics=tuple(args.wandb_colormap_metrics),
                    near_miss_thresh=summary.get("near_miss_thresh", 0.5),
                    report_base_url=args.report_base_url,
                    wandb_key_prefix=wandb_key_prefix,
                )
            )
            group_summaries[group_name] = summary_with_rows
            episode_data.append((group_name, rows, group_out_dir))

        if episode_data:
            log["closed_loop_episodes/all"] = build_combined_episode_table(episode_data)

        # Per-json aggregates: group by json_name (before __noobj or /)
        json_aggregates: dict[str, dict[str, dict]] = {}
        for name, summary in group_summaries.items():
            # name is "json/group" or "json__noobj/group"
            # extract json_name
            if "/" in name:
                json_name = name.split("/")[0]
            else:
                json_name = name.replace("__noobj", "")
            if json_name not in json_aggregates:
                json_aggregates[json_name] = {}
            json_aggregates[json_name][name] = summary

        for json_name, sub in json_aggregates.items():
            if len(sub) > 1:
                prefix = f"closed_loop_overview/{json_name}"
                log.update(build_groups_aggregate_log(sub, prefix=prefix))

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
