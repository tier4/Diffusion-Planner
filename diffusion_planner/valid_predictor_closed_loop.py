"""Closed-loop validation of a Diffusion-Planner checkpoint with the PerfectTracker.

Open-loop counterpart: ``valid_predictor.py`` runs ONE forward per sample and scores
the prediction against the recorded GT future. This script instead drives the ego in
CLOSED LOOP through ``scenario_generation``: each tick the model predicts the ego
trajectory, ``PerfectTracker`` advances the ego one step along it, the recorded
neighbors are replayed from the log via the Perception-Reproducer cursor, and the
realized ego footprint is scored against those neighbors with the canonical OBB
(``score_object_step`` -> collision / near-miss / min clearance).

A *route* = one bag-prefix group of consecutive 10 Hz NPZ frames (``RouteTimeline``);
each route is rolled out whole with ``render_segment`` (one GPU forward per tick), which BOTH
returns the route metrics AND writes a per-step PNG of the live-ego scene. Every run therefore
always produces video: one MP4 per route (``<route>.mp4``). Per-route metrics are streamed to
``segments.jsonl`` and aggregated into ``summary.json`` (both next to the checkpoint).
Clearance t-digest sketches used for multi-GPU p5 merge are written beside them as
``tdigests.jsonl`` / ``tdigests_{rank}.jsonl`` so ``segments.jsonl`` stays human-readable.

Only ``--model_path`` and ``--npz_root`` are required; outputs default to next to the checkpoint
(``<model_path dir>/closed_loop/``, override with ``--out_dir``) and the rollout knobs default to
the closed-loop mining config. ``--npz_root`` is either one NPZ dir tree or a .json path list of
route dirs; given a path list of multiple routes, the rollout automatically fans out one worker
process per visible GPU (each its own model copy), which also parallelizes the matplotlib render.
Example (1st epoch of a GRPO run)::

    NPZ=/path/to/closed_loop_npz_dir   # or /path/to/path_list.json

    python3 valid_predictor_closed_loop.py \
        --model_path /mnt/nvme/training_result/20260622-083517_per_sample_noise_grpo/epoch0001/best_model.pth \
        --npz_root ${NPZ}
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def _negative_mps2(value: str) -> float:
    """argparse type: strong-brake threshold must be negative (accel <= thresh)."""
    v = float(value)
    if v > 0.0:
        raise argparse.ArgumentTypeError(
            f"strong_brake_mps2 must be <= 0 (got {v}); a positive threshold matches nearly every step"
        )
    return v


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
        help="dir tree of route NPZ frames (recursively globbed, grouped into routes), OR a .json "
        "path list of such dirs (one route dir per entry, like --train_set_list). Pose JSON "
        "sidecars are read from next to each .npz, falling back to its own source tree.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="output dir for segments.jsonl/summary.json/videos. Default: "
        "<model_path dir>/closed_loop/<timestamp>/",
    )
    # --- tunable knobs (default to the closed-loop mining config) ---
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--near_miss_thresh", type=float, default=0.5, help="near-miss clearance (m)")
    p.add_argument(
        "--strong_brake_mps2",
        type=_negative_mps2,
        default=-2.5,
        help="strong-brake threshold (m/s^2, negative); a step counts when this and the previous frame both have tangential accel <= this",
    )
    p.add_argument(
        "--search_radius", type=float, default=1.5, help="PerceptionReproducer cursor search (m)"
    )
    p.add_argument(
        "--yaw_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pose mode only: drop recorded frames whose heading differs from live ego by "
        "more than pi/2 (threshold fixed at pi/2; --no-yaw-gate disables). No-op in clock "
        "mode, which never calls cursor.step (closed-loop/R2LPL default)",
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
        "--unstick_advance_m", type=float, default=2.5, help="how far ahead to snap when unsticking"
    )
    p.add_argument(
        "--unstick_radius_mult",
        type=float,
        default=3.0,
        help="when stuck, first widen the cursor search_radius to this x nominal so it reaches "
        "frames further ahead (model proceeds on its own); restored to nominal once the ego moves. "
        "<=1 disables this gentle stage (teleport straight away at --unstick_after)",
    )
    p.add_argument(
        "--unstick_teleport_after",
        type=int,
        default=300,
        help="if still stuck this many steps AFTER the radius was widened, fall back to the hard "
        "teleport onto the GT pose ahead (last resort)",
    )
    p.add_argument("--fps", type=int, default=10, help="output video frame rate (10 = realtime)")
    p.add_argument(
        "--replan_interval",
        type=int,
        default=4,
        help="re-run the model every N steps (1 = every step). Between inferences the cached plan "
        "is executed, re-expressed in the current ego frame each step; the ego still steps at 10Hz",
    )
    p.add_argument(
        "--draw_every",
        type=int,
        default=8,
        help="render a PNG only every N steps (1 = every step). PNG rendering (matplotlib) is the "
        "dominant cost; this throttles it without touching the rollout. Frames are encoded at --fps "
        "regardless, so the video also plays N x faster (shorter). For real-time use --fps 10/N",
    )
    p.add_argument(
        "--draw_workers",
        type=int,
        default=4,
        help="render the PNGs on this many worker processes (minimum 1). Output is "
        "byte-identical whatever the count; costs ~780 MB of RSS per worker, per GPU shard",
    )
    p.add_argument(
        "--abort_deviation_m",
        type=float,
        default=0.0,
        help="early-abort a segment (terminated='diverged') once GT deviation exceeds this "
        "(m) for --abort_after steps (0=disabled)",
    )
    p.add_argument("--abort_after", type=int, default=30)
    p.add_argument(
        "--abort_max_snaps",
        type=int,
        default=0,
        help="early-abort a segment after this many unstick teleports (0=disabled)",
    )
    p.add_argument(
        "--drop_objects",
        action="store_true",
        help="empty-world ablation: zero out dynamic/static objects each step (map kept)",
    )
    return p.parse_args()


def _load_model(model_path: Path, device: str):
    """Load either a torch checkpoint or an exported .onnx into the same (model, model_args)
    contract. args.json must sit next to whichever file is given."""
    from scenario_generation.simulate import load_model, load_onnx_model

    if model_path.suffix == ".onnx":
        return load_onnx_model(model_path, device)
    return load_model(model_path, device)


def _eval_knobs(args: argparse.Namespace) -> dict:
    """The rollout tunables forwarded to run_closed_loop_eval (everything except model/npz/out/
    device/shard), gathered once so the sequential and per-worker calls stay in lockstep."""
    return dict(
        near_miss_thresh=args.near_miss_thresh,
        search_radius=args.search_radius,
        yaw_gate=args.yaw_gate,
        warmup_steps=args.warmup_steps,
        unstick_after=args.unstick_after,
        unstick_advance_m=args.unstick_advance_m,
        unstick_radius_mult=args.unstick_radius_mult,
        unstick_teleport_after=args.unstick_teleport_after,
        fps=args.fps,
        replan_interval=args.replan_interval,
        draw_every=args.draw_every,
        draw_workers=args.draw_workers,
        neighbor_history_mode="recorded",
        tracker_mode="perfect",
        strong_brake_mps2=args.strong_brake_mps2,
        abort_deviation_m=args.abort_deviation_m,
        abort_after=args.abort_after,
        abort_max_snaps=args.abort_max_snaps,
        drop_objects=args.drop_objects,
    )


def _run_shard(rank, num_workers, gpu_ids, model_path, npz_root, out_dir, knobs) -> None:
    """Worker entrypoint (spawned): load the model on this worker's GPU and roll out the rank-th
    route slice, writing segments_{rank}.jsonl plus its PNG dirs/MP4s into the shared out_dir."""
    from scenario_generation.closed_loop_eval import run_closed_loop_eval

    device = f"cuda:{gpu_ids[rank % len(gpu_ids)]}"
    model, model_args = _load_model(Path(model_path), device)
    run_closed_loop_eval(
        model,
        model_args,
        npz_root,
        out_dir,
        device=device,
        verbose=(rank == 0),
        shard=(rank, num_workers),
        **knobs,
    )


def _merge_shards(
    out_dir: Path, npz_root, near_miss_thresh: float, *, strong_brake_mps2: float
) -> dict:
    """Aggregate every worker's segments_{rank}.jsonl (+ tdigests sidecars) into one summary.

    Also writes a merged, human-readable ``segments.jsonl`` (same tdigest-stripped shape the
    sequential path writes) -- downstream consumers like closed_loop_html_report only look for
    the unsharded filename, so without this a parallel-run site's videos never show up there.
    """
    from scenario_generation.closed_loop_eval import (
        aggregate,
        load_segment_rows_with_tdigests,
        segment_row_for_json,
    )

    rows = load_segment_rows_with_tdigests(out_dir)
    with open(out_dir / "segments.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: r["route"]):
            f.write(json.dumps(segment_row_for_json(r, route=r["route"]), default=float) + "\n")
    summary = aggregate(rows, near_miss_thresh, strong_brake_mps2=strong_brake_mps2)
    summary["npz_root"] = str(npz_root)
    summary["n_routes"] = len({r["route"] for r in rows})
    summary["video_mp4s"] = sorted(str(p) for p in out_dir.glob("*.mp4"))
    return summary


def _write_summary(out_dir: Path, summary: dict) -> None:
    with open(out_dir / "summary.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "video_mp4s"}, f, indent=4)


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or (
        args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    knobs = _eval_knobs(args)

    import torch

    from scenario_generation.closed_loop_eval import enumerate_routes, resolve_npz_roots

    # One worker process per visible GPU (respecting CUDA_VISIBLE_DEVICES), capped at the route
    # count so no worker gets an empty shard. Enumerate up front -- cheap globbing, no CUDA -- so the
    # parent never initializes a CUDA context before spawn. 1 GPU or 1 route => plain sequential.
    n_gpu = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    n_routes = sum(len(enumerate_routes(r)) for r in resolve_npz_roots(args.npz_root))
    nproc = max(1, min(n_gpu, n_routes))

    if nproc <= 1:
        from scenario_generation.closed_loop_eval import run_closed_loop_eval

        model, model_args = _load_model(args.model_path, args.device)
        print(f"device: {args.device} | model: {args.model_path} | out: {out_dir}")
        summary = run_closed_loop_eval(
            model, model_args, args.npz_root, out_dir, device=args.device, verbose=True, **knobs
        )
    else:
        import torch.multiprocessing as mp

        gpu_ids = list(range(n_gpu))
        print(
            f"parallel closed-loop: {nproc} workers ({n_routes} routes) over GPUs {gpu_ids} | "
            f"out: {out_dir}"
        )
        t0 = time.perf_counter()
        mp.spawn(
            _run_shard,
            args=(nproc, gpu_ids, str(args.model_path), str(args.npz_root), str(out_dir), knobs),
            nprocs=nproc,
            join=True,
        )
        summary = _merge_shards(
            out_dir,
            args.npz_root,
            args.near_miss_thresh,
            strong_brake_mps2=args.strong_brake_mps2,
        )
        summary["elapsed_sec"] = time.perf_counter() - t0

    summary["model_path"] = str(args.model_path)
    if nproc > 1:
        _write_summary(out_dir, summary)

    from scenario_generation.closed_loop_eval import format_summary_lines

    n_seg = summary["n_segments"]
    print(f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ===")
    for line in format_summary_lines(summary):
        print(line)
    print(f"videos: one <route>.mp4 per route in {out_dir}")


if __name__ == "__main__":
    main()
