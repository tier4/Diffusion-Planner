"""Shared CLI arguments and runners for closed-loop validation.

Used by:
  - ``diffusion_planner/valid_predictor_closed_loop.py`` (full-route mode)
  - ``scenario_generation/tools/eval_scenario_groups_closed_loop.py`` (grouped-by-area mode)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from scenario_generation.closed_loop_eval import run_closed_loop_eval
from scenario_generation.grouped_closed_loop_eval import run_grouped_closed_loop_eval
from scenario_generation.simulate import load_model, load_onnx_model


def load_checkpoint(model_path: Path, device: str):
    """Load a torch checkpoint or ONNX export (args.json must sit next to the file)."""
    if model_path.suffix == ".onnx":
        return load_onnx_model(model_path, device)
    return load_model(model_path, device)


def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    """Rollout knobs shared by full-route and grouped-by-area validation."""
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="checkpoint .pth; args.json must sit next to it",
    )
    parser.add_argument(
        "--npz_root",
        type=Path,
        required=True,
        help="dir tree of route NPZ frames (recursively globbed into routes)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--near_miss_thresh",
        "--margin_m",
        type=float,
        default=0.5,
        help="clearance threshold (m) for near-miss / safety violation counting",
    )
    parser.add_argument(
        "--search_radius",
        type=float,
        default=1.5,
        help="PerceptionReproducer cursor search radius (m)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="steps driven by recorded GT pose before model control",
    )
    parser.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap ego to GT pose after this many no-progress steps (0=off)",
    )
    parser.add_argument(
        "--unstick_advance_m",
        type=float,
        default=2.5,
        help="how far ahead to snap when unsticking (m)",
    )
    parser.add_argument(
        "--unstick_radius_mult",
        type=float,
        default=3.0,
        help="widen cursor search radius by this factor before hard teleport",
    )
    parser.add_argument(
        "--unstick_teleport_after",
        type=int,
        default=300,
        help="steps after radius widen before hard teleport",
    )
    parser.add_argument("--fps", type=int, default=10, help="output video frame rate")
    parser.add_argument(
        "--draw_every",
        type=int,
        default=8,
        help="render PNG every N steps (matplotlib cost throttle)",
    )
    parser.add_argument(
        "--replan_interval",
        type=int,
        default=10,
        help="re-run the model every N sim steps (1=every step); between replans execute the "
        "cached world-frame plan open-loop",
    )
    parser.add_argument(
        "--tracker_mode",
        choices=("mpc", "perfect"),
        default="perfect",
        help="ego advance: mpc=bicycle MPC tracker; perfect=place ego on predicted polyline",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out_dir",
        "--output_dir",
        type=Path,
        default=None,
        help="output directory (full: default <model_dir>/closed_loop/<ts>; grouped: required)",
    )


def add_full_route_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seg_len",
        type=int,
        default=6000,
        help="frames per route segment in full-route mode (~60s @10Hz)",
    )
    add_output_args(parser)


def add_grouped_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--classification_json_root",
        type=Path,
        default=None,
        help="Root of scenario_classification_json/ ({root}/{project}/{map}/{date}.json). "
        "Auto-matched to --npz_root valid layout.",
    )
    parser.add_argument(
        "--classification_json",
        type=Path,
        default=None,
        help="Optional legacy single {date}.json override. Prefer --classification_json_root.",
    )
    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help="optional subset of area names in the classification JSON",
    )


def _load_area_mapping(path: Path) -> dict[str, str]:
    """Legacy helper for group->list mapping files (classify-time config only)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "areas" in raw:
        return {str(k): str(v["metric_group"]) for k, v in raw["areas"].items()}
    out: dict[str, str] = {}
    for group, areas in raw.items():
        if group == "version" or not isinstance(areas, list):
            continue
        for area in areas:
            out[str(area)] = str(group)
    return out


def run_full_route_eval(args: argparse.Namespace) -> dict:
    model, model_args = load_checkpoint(args.model_path, args.device)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir)
    print(f"device: {args.device} | model: {args.model_path} | out: {out_dir}")

    summary = run_closed_loop_eval(
        model,
        model_args,
        args.npz_root,
        out_dir,
        seg_len=args.seg_len,
        device=args.device,
        near_miss_thresh=args.near_miss_thresh,
        search_radius=args.search_radius,
        warmup_steps=args.warmup_steps,
        unstick_after=args.unstick_after,
        unstick_advance_m=args.unstick_advance_m,
        unstick_radius_mult=args.unstick_radius_mult,
        unstick_teleport_after=args.unstick_teleport_after,
        fps=args.fps,
        draw_every=args.draw_every,
        replan_interval=args.replan_interval,
        tracker_mode=args.tracker_mode,
        neighbor_history_mode="recorded",
        verbose=True,
    )
    summary["model_path"] = str(args.model_path)
    return summary


def run_grouped_eval(args: argparse.Namespace) -> int:
    if args.out_dir is None:
        raise SystemExit("grouped mode requires --out_dir")

    from scenario_generation.scenario_classification import (
        discover_classification_eval_jobs,
        merge_grouped_eval_summaries,
    )

    cl_root = args.classification_json_root
    cl_explicit = args.classification_json
    if cl_explicit is not None and cl_explicit.is_dir():
        cl_root = cl_root or cl_explicit
        cl_explicit = None

    jobs = discover_classification_eval_jobs(
        args.npz_root,
        classification_root=cl_root,
        explicit_path=cl_explicit,
        dataset_name=getattr(args, "scenario_dataset_name", None),
    )
    if not jobs:
        raise SystemExit(
            "grouped mode: no classification JSON matched npz_root. Set "
            "--classification_json_root (scenario_classification_json/) and ensure "
            "npz path follows .../{project}/{map}/valid/{date}/..."
        )

    model, model_args = load_checkpoint(args.model_path, args.device)
    summaries: list[dict] = []
    for job in jobs:
        job_out = args.out_dir if len(jobs) == 1 else args.out_dir / job.date
        print(
            f"Grouped job: {job.dataset_key} {job.date} "
            f"npz={job.npz_root} json={job.classification_json}"
        )
        job_summary = run_grouped_closed_loop_eval(
            model,
            model_args,
            job.npz_root,
            job.classification_json,
            job_out,
            device=args.device,
            near_miss_thresh=args.near_miss_thresh,
            search_radius=args.search_radius,
            warmup_steps=args.warmup_steps,
            unstick_after=args.unstick_after,
            unstick_advance_m=args.unstick_advance_m,
            draw_every=args.draw_every,
            replan_interval=args.replan_interval,
            tracker_mode=args.tracker_mode,
            fps=args.fps,
            areas=args.areas,
            verbose=True,
        )
        job_summary["date"] = job.date
        job_summary["dataset_key"] = job.dataset_key
        summaries.append(job_summary)

    summary = merge_grouped_eval_summaries(summaries)
    out_dir = Path(args.out_dir)
    from scenario_generation.metrics.group_report import write_metrics_summary, write_results_table

    write_results_table(summary.get("segments") or [], out_dir / "results_table.csv")
    write_metrics_summary(summary.get("grouped_summary") or {}, out_dir / "metrics_summary.json")
    print(
        f"Wrote {summary['n_episodes']} episode rows ({len(jobs)} date(s)) -> "
        f"{out_dir / 'results_table.csv'}"
    )
    return 0


def print_full_summary(summary: dict, near_miss_thresh: float, out_dir: Path) -> None:
    n_seg = summary["n_segments"]
    print(f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ===")
    print(
        f"collision: {summary['n_segments_with_collision']}/{n_seg} segments "
        f"(rate {summary['collision_segment_rate']:.4f}), "
        f"{summary['total_collision_steps']} steps (rate {summary['collision_step_rate']:.6f})"
    )
    print(
        f"near-miss (<= {near_miss_thresh} m): "
        f"{summary['n_segments_with_near_miss']}/{n_seg} segments "
        f"(rate {summary['near_miss_segment_rate']:.4f}), {summary['total_near_miss_steps']} steps"
    )
    print(
        f"global_min_clearance={summary['global_min_clearance']:.3f} m  "
        f"mean_segment_min_clearance={summary['mean_segment_min_clearance']:.3f} m  "
        f"mean_segment_mean_clearance={summary['mean_segment_mean_clearance']:.3f} m"
    )
    print(f"total_snaps={summary['total_snaps']}  terminated={summary['terminated_counts']}")
    print(f"videos: per-segment <route>_<start>_<end>.mp4 in {out_dir}")
