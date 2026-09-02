"""Create an MP4 of neighbor-token Attention over consecutive dataset frames."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
from diffusion_planner.utils.dataset import DiffusionPlannerData
from visualize_neighbor_attention import (
    attention_for_sample,
    draw_report,
    find_fusion,
    load_encoder,
    movement_and_turn,
    patch_fusion,
    sample_to_batch,
    token_records,
)

matplotlib.use("Agg", force=True)


def sequence_indices(dataset, center: int, before: int, after: int, step: int) -> list[int]:
    """Return a centered sequence without crossing into another log directory."""
    if not 0 <= center < len(dataset):
        raise IndexError(f"--center_index must be in 0..{len(dataset) - 1}")
    parent = Path(dataset.data_list[center]).parent

    left = []
    index = center
    lower = max(0, center - before)
    while index >= lower and Path(dataset.data_list[index]).parent == parent:
        left.append(index)
        index -= step

    right = []
    index = center + step
    upper = min(len(dataset) - 1, center + after)
    while index <= upper and Path(dataset.data_list[index]).parent == parent:
        right.append(index)
        index += step
    return list(reversed(left)) + right


def encode_video(
    frames_dir: Path,
    fps: float,
    width: int,
    height: int,
    out_mp4: Path,
):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white,"
        "setsar=1,format=yuv420p"
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-profile:v",
        "high",
        "-tag:v",
        "avc1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render consecutive neighbor-token Attention frames as an MP4."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--center_index", type=int, required=True)
    parser.add_argument("--frames_before", type=int, default=20)
    parser.add_argument("--frames_after", type=int, default=40)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--video_width", type=int, default=1920)
    parser.add_argument("--video_height", type=int, default=1080)
    parser.add_argument("--layer", default="mean")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--marker_size_min", type=float, default=45.0)
    parser.add_argument("--marker_size_max", type=float, default=950.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_mp4", default="neighbor_attention.mp4")
    parser.add_argument("--out_json", default="")
    parser.add_argument(
        "--keep_frames",
        action="store_true",
        help="retain rendered PNG files next to the MP4",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frames_before < 0 or args.frames_after < 0:
        raise ValueError("--frames_before and --frames_after must be non-negative")
    if args.step <= 0 or args.fps <= 0:
        raise ValueError("--step and --fps must be positive")
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("--video_width and --video_height must be positive")
    if args.video_width % 2 or args.video_height % 2:
        raise ValueError("--video_width and --video_height must be even for yuv420p")
    if args.colormap not in matplotlib.colormaps:
        raise ValueError(f"unknown Matplotlib colormap: {args.colormap}")
    if args.marker_size_min <= 0 or args.marker_size_max < args.marker_size_min:
        raise ValueError("--marker_size_max must be >= --marker_size_min > 0")

    dataset = DiffusionPlannerData(args.valid_set_list)
    indices = sequence_indices(
        dataset,
        args.center_index,
        args.frames_before,
        args.frames_after,
        args.step,
    )
    print(
        f"sequence: {indices[0]}..{indices[-1]} ({len(indices)} frames, step={args.step})",
        flush=True,
    )

    encoder, cfg = load_encoder(Path(args.run_dir).resolve(), args.device)
    fusion_store = []
    patch_fusion(find_fusion(encoder), fusion_store)

    analyzed = []
    global_vmax = 0.0
    for frame_number, index in enumerate(indices, start=1):
        sample = dataset[index]
        _, scores, per_layer = attention_for_sample(
            encoder, cfg, fusion_store, sample, args.layer, args.device
        )
        records = token_records(sample, scores, per_layer)
        movement_m, turn_angle_deg = movement_and_turn(sample)
        if records:
            global_vmax = max(
                global_vmax,
                max(record["attention_pct_within_neighbors"] for record in records),
            )
        analyzed.append(
            {
                "dataset_index": index,
                "sample_path": dataset.data_list[index],
                "movement_m": movement_m,
                "turn_angle_deg": turn_angle_deg,
                "records": records,
            }
        )
        print(f"analyzed {frame_number}/{len(indices)}: index={index}", flush=True)

    out_mp4 = Path(args.out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames_target = out_mp4.with_suffix("").with_name(f"{out_mp4.stem}_frames")
    temporary = None
    if args.keep_frames:
        frames_target.mkdir(parents=True, exist_ok=True)
        frames_dir = frames_target
    else:
        temporary = tempfile.TemporaryDirectory(
            prefix=f".{out_mp4.stem}_frames_", dir=out_mp4.parent
        )
        frames_dir = Path(temporary.name)

    try:
        for frame_number, item in enumerate(analyzed):
            sample = dataset[item["dataset_index"]]
            frame_path = frames_dir / f"frame_{frame_number:06d}.png"
            draw_report(
                sample_to_batch(sample),
                item["records"],
                item["dataset_index"],
                item["sample_path"],
                item["movement_m"],
                item["turn_angle_deg"],
                args.layer,
                args.top_k,
                args.view_range,
                args.colormap,
                args.marker_size_min,
                args.marker_size_max,
                frame_path,
                attention_vmax=global_vmax,
                tight_bbox=False,
            )
            print(f"rendered {frame_number + 1}/{len(analyzed)}", flush=True)
        encode_video(
            frames_dir,
            args.fps,
            args.video_width,
            args.video_height,
            out_mp4,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()

    out_json = Path(args.out_json).resolve() if args.out_json else out_mp4.with_suffix(".json")
    report = {
        "center_index": args.center_index,
        "indices": indices,
        "fps": args.fps,
        "video_width": args.video_width,
        "video_height": args.video_height,
        "video_color_space": "BT.709 limited-range yuv420p",
        "step": args.step,
        "layer": args.layer,
        "global_attention_vmax_pct_within_neighbors": global_vmax,
        "visualization": {
            "colormap": args.colormap,
            "marker_size_min": args.marker_size_min,
            "marker_size_max": args.marker_size_max,
            "top_k": args.top_k,
            "view_range_m": args.view_range,
        },
        "frames": [
            {
                "dataset_index": item["dataset_index"],
                "sample_path": item["sample_path"],
                "movement_m": item["movement_m"],
                "turn_angle_deg": item["turn_angle_deg"],
                "valid_neighbor_count": len(item["records"]),
                "top_tokens": item["records"][: args.top_k],
            }
            for item in analyzed
        ],
    }
    with out_json.open("w") as file:
        json.dump(report, file, indent=2)
    print(f"wrote {out_mp4}", flush=True)
    print(f"wrote {out_json}", flush=True)
    if args.keep_frames:
        print(f"kept frames in {frames_dir}", flush=True)


if __name__ == "__main__":
    main()
