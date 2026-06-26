"""Closed-loop validation of a Diffusion-Planner checkpoint with the PerfectTracker.

Open-loop counterpart: ``valid_predictor.py`` runs ONE forward per sample and scores
the prediction against the recorded GT future. This script instead drives the ego in
CLOSED LOOP through ``scenario_generation``: each tick the model predicts the ego
trajectory, ``PerfectTracker`` advances the ego one step along it, the recorded
neighbors are replayed from the log via the Perception-Reproducer cursor, and the
realized ego footprint is scored against those neighbors with the canonical OBB
(``score_step`` -> collision / near-miss / min clearance).

A *route* = one bag-prefix group of consecutive 10 Hz NPZ frames (``RouteTimeline``);
each route is sliced into ``--seg_len`` segments and rolled out with ``render_segment``
(one GPU forward per tick), which BOTH returns the segment metrics AND writes a per-step
PNG of the live-ego scene. Every run therefore always produces video: per segment a PNG
dir + an MP4, and per route a single concatenated ``<route>_full.mp4`` covering the whole
route. Per-segment metrics are streamed to ``segments.jsonl`` and aggregated into
``summary.json`` (both next to the checkpoint).

Only ``--model_path`` and ``--npz_root`` are required; all outputs are written next to
the checkpoint (``<model_path dir>/closed_loop/``) and the rollout knobs default to the
closed-loop mining config. Example (1st epoch of a GRPO run)::

    NPZ=/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/x2_dev/2231_odaiba_shinagawa_copied_from_xx1/valid/2026-01-15/13-42-45

    python3 valid_predictor_closed_loop.py \
        --model_path /mnt/nvme/training_result/20260622-083517_per_sample_noise_grpo/epoch0001/best_model.pth \
        --npz_root ${NPZ}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes


def parse_args() -> argparse.Namespace:
    # Only the checkpoint and the NPZ dir are required; everything else is a tunable
    # knob with the closed-loop mining default. Outputs land next to the checkpoint.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="checkpoint .pth; args.json must sit next to it (e.g. epoch0001/best_model.pth)",
    )
    p.add_argument(
        "--npz_root",
        type=Path,
        required=True,
        help="dir tree of route NPZ frames (recursively globbed, grouped into routes). "
        "Pose JSON sidecars are read from next to each .npz, falling back to this same tree.",
    )
    # --- tunable knobs (default to the closed-loop mining config) ---
    p.add_argument("--seg_len", type=int, default=600, help="frames per segment (~60s @10Hz)")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--near_miss_thresh", type=float, default=0.5, help="near-miss clearance (m)")
    p.add_argument(
        "--search_radius", type=float, default=1.5, help="PerceptionReproducer cursor search (m)"
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="steps driven by the recorded GT pose before handing control to the model",
    )
    p.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap the ego to the GT pose ahead after this many no-progress steps (0=off)",
    )
    p.add_argument(
        "--unstick_advance_m", type=float, default=5.0, help="how far ahead to snap when unsticking"
    )
    p.add_argument("--fps", type=int, default=10, help="output video frame rate (10 = realtime)")
    p.add_argument(
        "--replan_interval",
        type=int,
        default=1,
        help="re-run the model every N steps (1 = every step). Between inferences the cached plan "
        "is executed, re-expressed in the current ego frame each step; the ego still steps at 10Hz",
    )
    p.add_argument(
        "--draw_every",
        type=int,
        default=1,
        help="render a PNG only every N steps (1 = every step). PNG rendering (matplotlib) is the "
        "dominant cost; this throttles it without touching the rollout. The video frame rate is "
        "scaled to fps/draw_every so playback stays real-time (fewer, longer-held frames)",
    )
    return p.parse_args()


def _enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    paths = sorted(npz_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def _aggregate(rows: list[dict], near_miss_thresh: float) -> dict:
    """Aggregate per-segment metric rows into a single closed-loop summary."""
    n_seg = len(rows)
    total_steps = sum(r["n_steps_run"] for r in rows)
    total_collision_steps = sum(r["n_collision_steps"] for r in rows)
    total_near_miss_steps = sum(r["n_near_miss_steps"] for r in rows)
    total_snaps = sum(r["n_snaps"] for r in rows)

    n_seg_collision = sum(1 for r in rows if r["n_collision_steps"] > 0)
    n_seg_near_miss = sum(1 for r in rows if r["n_near_miss_steps"] > 0)

    # min_clearance is +inf for a segment that never saw a valid neighbor; exclude those.
    finite_min_cl = [r["min_clearance"] for r in rows if np.isfinite(r["min_clearance"])]
    finite_mean_cl = [r["mean_clearance"] for r in rows if np.isfinite(r["mean_clearance"])]

    term_counts: dict[str, int] = {}
    for r in rows:
        term_counts[r["terminated"]] = term_counts.get(r["terminated"], 0) + 1

    return {
        "near_miss_thresh": near_miss_thresh,
        "n_segments": n_seg,
        "total_steps": total_steps,
        # collision: at the segment level (any hit) and at the step level (hit fraction).
        "n_segments_with_collision": n_seg_collision,
        "collision_segment_rate": n_seg_collision / n_seg if n_seg else 0.0,
        "total_collision_steps": total_collision_steps,
        "collision_step_rate": total_collision_steps / total_steps if total_steps else 0.0,
        "n_segments_with_near_miss": n_seg_near_miss,
        "near_miss_segment_rate": n_seg_near_miss / n_seg if n_seg else 0.0,
        "total_near_miss_steps": total_near_miss_steps,
        "near_miss_step_rate": total_near_miss_steps / total_steps if total_steps else 0.0,
        # clearance distribution across segments (m); inf-only segments excluded.
        "global_min_clearance": float(min(finite_min_cl)) if finite_min_cl else float("inf"),
        "mean_segment_min_clearance": float(np.mean(finite_min_cl))
        if finite_min_cl
        else float("inf"),
        "mean_segment_mean_clearance": float(np.mean(finite_mean_cl))
        if finite_mean_cl
        else float("inf"),
        "total_snaps": total_snaps,
        "terminated_counts": term_counts,
    }


def _build_mp4(png_dir: Path, mp4_path: Path, fps: float) -> None:
    """Encode the PNG sequence in ``png_dir`` to an MP4.

    PNGs are named by step ``k`` and may be sparse (``--draw_every`` skips frames), so glob the
    directory (gap-tolerant, name-sorted) instead of a contiguous ``%05d`` counter. ``fps`` is the
    already-scaled frame rate (``--fps / --draw_every``) so playback stays real-time.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(png_dir / "*.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(mp4_path),
        ],
        check=True,
    )


def _concat_mp4(mp4_paths: list[Path], out_path: Path, work_dir: Path) -> None:
    """Concatenate per-segment MP4s (same codec/size) into one route-level MP4."""
    if not mp4_paths:
        return
    list_file = work_dir / (out_path.stem + ".ffconcat.txt")
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in mp4_paths))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out_path),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()

    from scenario_generation.simulate import load_model

    model, model_args = load_model(args.model_path, args.device)

    routes = _enumerate_routes(args.npz_root)
    route_keys = sorted(routes)
    print(f"routes: {len(route_keys)} | device: {args.device} | model: {args.model_path}")

    # Outputs land next to the checkpoint: <model_path dir>/closed_loop/.
    out_dir = args.model_path.parent / "closed_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segments.jsonl"
    summary_path = out_dir / "summary.json"

    timers = Timers()
    rows: list[dict] = []
    n_seg = 0
    t0 = time.perf_counter()

    fout = open(out_path, "w")
    try:
        for ri, key in enumerate(route_keys):
            tl = RouteTimeline(routes[key], sidecar_dir=args.npz_root, timers=timers)
            seg_mp4s: list[Path] = []
            for start, end in tl.iter_segments(args.seg_len):
                # render_segment runs the closed-loop rollout single-stepped, writing a
                # per-step PNG AND returning the same metrics dict as run_segment.
                png_dir = out_dir / f"{key}_{start}_{end}"
                metrics = render_segment(
                    model,
                    model_args,
                    tl,
                    start,
                    end,
                    png_dir,
                    device=args.device,
                    near_miss_thresh=args.near_miss_thresh,
                    search_radius=args.search_radius,
                    warmup_steps=args.warmup_steps,
                    unstick_after=args.unstick_after,
                    unstick_advance_m=args.unstick_advance_m,
                    replan_interval=args.replan_interval,
                    draw_every=args.draw_every,
                )
                row = {"route": key, **metrics}
                fout.write(json.dumps(row, default=float) + "\n")
                fout.flush()
                rows.append(row)
                n_seg += 1

                seg_mp4 = out_dir / f"{key}_{start}_{end}.mp4"
                # Fold draw_every into the frame rate so the sparser PNGs still play in real time.
                _build_mp4(png_dir, seg_mp4, args.fps / args.draw_every)
                seg_mp4s.append(seg_mp4)
                print(
                    f"  [{key}] segment [{start},{end}] -> {seg_mp4.name}  "
                    f"coll={metrics['n_collision_steps']} "
                    f"near={metrics['n_near_miss_steps']} "
                    f"min_clr={metrics['min_clearance']:.3f}"
                )

            # Concatenate this route's segment MP4s into one full-route video.
            full_mp4 = out_dir / f"{key}_full.mp4"
            _concat_mp4(seg_mp4s, full_mp4, out_dir)
            print(
                f"[{ri + 1}/{len(route_keys)}] {key}: {len(seg_mp4s)} segments -> {full_mp4.name}"
            )
    finally:
        fout.close()

    elapsed = time.perf_counter() - t0
    summary = _aggregate(rows, args.near_miss_thresh)
    summary["model_path"] = str(args.model_path)
    summary["npz_root"] = str(args.npz_root)
    summary["n_routes"] = len(route_keys)
    summary["elapsed_sec"] = elapsed

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n=== closed-loop validation: {n_seg} segments in {elapsed:.1f}s ===")
    print(
        f"collision: {summary['n_segments_with_collision']}/{n_seg} segments "
        f"(rate {summary['collision_segment_rate']:.4f}), "
        f"{summary['total_collision_steps']} steps "
        f"(rate {summary['collision_step_rate']:.6f})"
    )
    print(
        f"near-miss (<= {args.near_miss_thresh} m): "
        f"{summary['n_segments_with_near_miss']}/{n_seg} segments "
        f"(rate {summary['near_miss_segment_rate']:.4f}), "
        f"{summary['total_near_miss_steps']} steps"
    )
    print(
        f"global_min_clearance={summary['global_min_clearance']:.3f} m  "
        f"mean_segment_min_clearance={summary['mean_segment_min_clearance']:.3f} m  "
        f"mean_segment_mean_clearance={summary['mean_segment_mean_clearance']:.3f} m"
    )
    print(f"total_snaps={summary['total_snaps']}  terminated={summary['terminated_counts']}")
    print(f"\nwrote {n_seg} rows -> {out_path}\nwrote summary -> {summary_path}")
    print(f"videos: per-segment <route>_<s>_<e>.mp4 + per-route <route>_full.mp4 in {out_dir}")


if __name__ == "__main__":
    main()
