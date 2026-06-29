"""Render a Perception Reproducer route segment to PNGs and optional WebM."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes


def _make_webm(frames_dir: Path, out_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        "32",
        "-row-mt",
        "1",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _model_lora_title(model_path: Path, lora_path: Path | None) -> str:
    model_label = f"{model_path.parent.name}/{model_path.name}"
    lora_label = f"{lora_path.parent.name}/{lora_path.name}" if lora_path else "none"
    return f"model: {model_label}  lora: {lora_label}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_root", type=Path, required=True, help="One registered route dataset")
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--lora_path", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--start", type=int, default=0, help="Start index within the route timeline")
    p.add_argument(
        "--end", type=int, default=0, help="End index within the route timeline; 0 = start+steps"
    )
    p.add_argument("--steps", type=int, default=120, help="Used when --end is 0")
    p.add_argument("--near_miss_thresh", type=float, default=0.5)
    p.add_argument("--search_radius", type=float, default=1.5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--unstick_after", type=int, default=300)
    p.add_argument("--unstick_advance_m", type=float, default=5.0)
    p.add_argument(
        "--view_half_m",
        type=float,
        default=50.0,
        help="Half-width of the bird's-eye camera window around ego, in metres",
    )
    p.add_argument(
        "--distance_label_offset_m",
        type=float,
        default=1.2,
        help="Offset the distance badges away from the line, in metres",
    )
    p.add_argument("--make_webm", action="store_true")
    p.add_argument("--webm_fps", type=int, default=10)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from rlvr.autoresearch.tools.ghost_sim_common import load_model

    model, model_args = load_model(
        str(args.model_path), str(args.lora_path) if args.lora_path else None, device
    )

    routes = group_routes(sorted(args.npz_root.rglob("*.npz")))
    if len(routes) != 1:
        raise SystemExit(
            f"{args.npz_root} should contain exactly one route; found {list(routes)}. "
            "Pick one route dataset from the Workspace route list."
        )
    route_key, paths = next(iter(routes.items()))
    tl = RouteTimeline(paths)
    start = max(0, args.start)
    end = args.end if args.end > 0 else start + args.steps
    end = min(end, len(paths))
    if end <= start:
        raise SystemExit(f"Invalid segment start/end: {start}/{end} for {len(paths)} frames")

    out_dir = args.output_dir / f"{route_key}_{start:05d}_{end:05d}"
    title_prefix = f"{route_key}\n{_model_lora_title(args.model_path, args.lora_path)}"
    print(f"Rendering {route_key}[{start}:{end}] -> {out_dir}")
    metrics = render_segment(
        model,
        model_args,
        tl,
        start,
        end,
        out_dir,
        device=device,
        near_miss_thresh=args.near_miss_thresh,
        search_radius=args.search_radius,
        warmup_steps=args.warmup_steps,
        unstick_after=args.unstick_after,
        unstick_advance_m=args.unstick_advance_m,
        title_prefix=title_prefix,
        distance_label_offset_m=args.distance_label_offset_m,
        view_half_m=args.view_half_m,
    )
    print(metrics)
    if args.make_webm:
        webm = out_dir / "reproducer_segment.webm"
        _make_webm(out_dir, webm, args.webm_fps)
        print(f"WebM: {webm}")


if __name__ == "__main__":
    main()
