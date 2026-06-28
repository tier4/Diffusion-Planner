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
PNG of the live-ego scene. Every run therefore always produces video: one MP4 per segment
(``<route>_<start>_<end>.mp4``). Per-segment metrics are streamed to ``segments.jsonl`` and
aggregated into ``summary.json`` (both next to the checkpoint).

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
from datetime import datetime
from pathlib import Path


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
    p.add_argument("--seg_len", type=int, default=6000, help="frames per segment (~60s @10Hz)")
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
        "--unstick_advance_m",
        type=float,
        default=2.5,
        help="how far ahead to snap when unsticking",
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
        default=40,
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from scenario_generation.closed_loop_eval import run_closed_loop_eval
    from scenario_generation.simulate import load_model

    model, model_args = load_model(args.model_path, args.device)
    out_dir = args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
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
        replan_interval=args.replan_interval,
        draw_every=args.draw_every,
    )
    summary["model_path"] = str(args.model_path)

    n_seg = summary["n_segments"]
    print(f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ===")
    print(
        f"collision: {summary['n_segments_with_collision']}/{n_seg} segments "
        f"(rate {summary['collision_segment_rate']:.4f}), "
        f"{summary['total_collision_steps']} steps (rate {summary['collision_step_rate']:.6f})"
    )
    print(
        f"near-miss (<= {args.near_miss_thresh} m): "
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


if __name__ == "__main__":
    main()
