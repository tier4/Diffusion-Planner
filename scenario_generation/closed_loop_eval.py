"""Reusable closed-loop rollout + render + metric aggregation.

Shared by the standalone CLI (``diffusion_planner/valid_predictor_closed_loop.py``) and the
per-epoch training validation (``diffusion_planner/diffusion_planner/train.py``): both drive the
ego in CLOSED LOOP through ``reproducer_rollout.render_segment`` over the route NPZ frames under
``npz_root``, write a per-step PNG, build one MP4 per segment, and aggregate the per-segment
metrics into a single summary.

``run_closed_loop_eval`` takes an already-loaded ``(model, model_args)`` (so training can pass its
live model + ``TrainConfig`` straight in, no checkpoint reload) and returns the summary dict plus
the per-segment MP4 paths (for wandb upload).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes


def enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    """Group all .npz under ``npz_root`` into routes (bag-prefix groups)."""
    paths = sorted(Path(npz_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def aggregate(rows: list[dict], near_miss_thresh: float) -> dict:
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
        "n_segments_with_collision": n_seg_collision,
        "collision_segment_rate": n_seg_collision / n_seg if n_seg else 0.0,
        "total_collision_steps": total_collision_steps,
        "collision_step_rate": total_collision_steps / total_steps if total_steps else 0.0,
        "n_segments_with_near_miss": n_seg_near_miss,
        "near_miss_segment_rate": n_seg_near_miss / n_seg if n_seg else 0.0,
        "total_near_miss_steps": total_near_miss_steps,
        "near_miss_step_rate": total_near_miss_steps / total_steps if total_steps else 0.0,
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


def build_mp4(png_dir: Path, mp4_path: Path, fps: float) -> None:
    """Encode the PNG sequence in ``png_dir`` to an MP4.

    PNGs are named by step ``k`` and may be sparse (``draw_every`` skips frames), so glob the
    directory (gap-tolerant, name-sorted) instead of a contiguous ``%05d`` counter. ``fps`` is the
    raw frame rate, so a sparse sequence plays faster than real time (shorter video).
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


def run_closed_loop_eval(
    model,
    model_args,
    npz_root,
    out_dir,
    *,
    seg_len: int,
    device: str,
    near_miss_thresh: float,
    search_radius: float,
    warmup_steps: int,
    unstick_after: int,
    unstick_advance_m: float,
    fps: float,
    replan_interval: int,
    draw_every: int,
    neighbor_history_mode: str,
    unstick_radius_mult: float = 10.0,
    unstick_teleport_after: int = 300,
    verbose: bool = True,
) -> dict:
    """Render closed-loop rollouts over every route under ``npz_root`` and aggregate metrics.

    ``model`` must be an eval-mode Diffusion-Planner (callable ``model(data) -> (_, outputs)`` with
    ``outputs["prediction"]``); ``model_args`` provides ``observation_normalizer`` /
    ``predicted_neighbor_num`` / ``future_len`` (a ``Config`` or ``TrainConfig``). Per segment a PNG
    dir + an MP4 (``<route>_<start>_<end>.mp4``) are written. ``segments.jsonl`` and
    ``summary.json`` are written into ``out_dir``.

    Returns the summary dict with extra keys ``video_mp4s`` (list[Path] of every per-segment MP4),
    ``segments`` (list[row]), and ``elapsed_sec``.
    """
    npz_root = Path(npz_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    routes = enumerate_routes(npz_root)
    route_keys = sorted(routes)

    timers = Timers()
    rows: list[dict] = []
    video_mp4s: list[Path] = []
    t0 = time.perf_counter()

    fout = open(out_dir / "segments.jsonl", "w")
    try:
        for ri, key in enumerate(route_keys):
            tl = RouteTimeline(routes[key], sidecar_dir=npz_root, timers=timers)
            n_seg_videos = 0
            for start, end in tl.iter_segments(seg_len):
                png_dir = out_dir / f"{key}_{start}_{end}"
                metrics = render_segment(
                    model,
                    model_args,
                    tl,
                    start,
                    end,
                    png_dir,
                    device=device,
                    near_miss_thresh=near_miss_thresh,
                    search_radius=search_radius,
                    warmup_steps=warmup_steps,
                    unstick_after=unstick_after,
                    unstick_advance_m=unstick_advance_m,
                    unstick_radius_mult=unstick_radius_mult,
                    unstick_teleport_after=unstick_teleport_after,
                    replan_interval=replan_interval,
                    draw_every=draw_every,
                    neighbor_history_mode=neighbor_history_mode,
                )
                row = {"route": key, **metrics}
                fout.write(json.dumps(row, default=float) + "\n")
                fout.flush()
                rows.append(row)

                # A segment that terminates at step 0 (e.g. ego starts within goal_reach_m) draws
                # no PNG; skip the empty ffmpeg call (its glob would error on an empty dir).
                if not any(png_dir.glob("*.png")):
                    if verbose:
                        print(f"  [{key}] segment [{start},{end}] -> 0 frames, no video")
                    continue
                seg_mp4 = out_dir / f"{key}_{start}_{end}.mp4"
                # Raw fps: with only every draw_every-th frame drawn, the video plays
                # draw_every x faster than real time. For real time use fps = 10 / draw_every.
                build_mp4(png_dir, seg_mp4, fps)
                video_mp4s.append(seg_mp4)
                n_seg_videos += 1
                if verbose:
                    print(
                        f"  [{key}] segment [{start},{end}] -> {seg_mp4.name}  "
                        f"coll={metrics['n_collision_steps']} near={metrics['n_near_miss_steps']} "
                        f"min_clr={metrics['min_clearance']:.3f}"
                    )

            if verbose:
                print(f"[{ri + 1}/{len(route_keys)}] {key}: {n_seg_videos} segment video(s)")
    finally:
        fout.close()

    summary = aggregate(rows, near_miss_thresh)
    summary["npz_root"] = str(npz_root)
    summary["n_routes"] = len(route_keys)
    summary["elapsed_sec"] = time.perf_counter() - t0
    summary["video_mp4s"] = video_mp4s
    summary["segments"] = rows

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {k: v for k, v in summary.items() if k not in ("video_mp4s", "segments")}, f, indent=4
        )

    return summary
