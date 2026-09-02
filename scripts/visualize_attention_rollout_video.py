"""Create all-token and neighbor-only Decoder rollout videos."""

import argparse
import json
from pathlib import Path

import matplotlib
import torch
from diffusion_planner.utils.dataset import DiffusionPlannerData
from token_analysis_common import load_model, patch_fusion, prepare_inputs
from visualize_all_token_attention import draw_report, token_records
from visualize_attention_rollout import attention_rollout, neighbor_records, patch_decoder
from visualize_neighbor_attention import (
    draw_report as draw_neighbor_report,
)
from visualize_neighbor_attention import (
    find_fusion,
    movement_and_turn,
    sample_to_batch,
)
from visualize_neighbor_attention_video import encode_video, sequence_indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument("--colormap", default="plasma")
    parser.add_argument("--marker_size_min", type=float, default=25.0)
    parser.add_argument("--marker_size_max", type=float, default=700.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_all_mp4", required=True)
    parser.add_argument("--out_neighbor_mp4", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--frames_dir", default="")
    parser.add_argument("--overwrite_frames", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frames_before < 0 or args.frames_after < 0:
        raise ValueError("--frames_before and --frames_after must be non-negative")
    if args.step <= 0 or args.fps <= 0 or args.top_k <= 0:
        raise ValueError("--step, --fps, and --top_k must be positive")
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("--video_width and --video_height must be positive")
    if args.video_width % 2 or args.video_height % 2:
        raise ValueError("--video_width and --video_height must be even for yuv420p")
    if args.colormap not in matplotlib.colormaps:
        raise ValueError(f"unknown Matplotlib colormap: {args.colormap}")

    dataset = DiffusionPlannerData(args.valid_set_list)
    indices = sequence_indices(
        dataset, args.center_index, args.frames_before, args.frames_after, args.step
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.frames_dir) if args.frames_dir else Path(f"{args.out_json}.frames")
    all_dir, neighbor_dir = root / "all", root / "neighbor"
    all_dir.mkdir(parents=True, exist_ok=True)
    neighbor_dir.mkdir(parents=True, exist_ok=True)
    for frame_dir in (all_dir, neighbor_dir):
        old_frames = list(frame_dir.glob("frame_*.png"))
        if old_frames and not args.overwrite_frames:
            raise FileExistsError(
                f"{frame_dir} already contains frame PNGs; use --overwrite_frames "
                "or choose a new --frames_dir"
            )
        if args.overwrite_frames:
            for old_frame in old_frames:
                old_frame.unlink()

    model, cfg, _ = load_model(Path(args.run_dir).resolve(), args.device)
    fusion_store, decoder_store = [], []
    patch_fusion(find_fusion(model.encoder), fusion_store)
    patch_decoder(model.decoder, decoder_store)
    frame_meta = []
    with torch.no_grad():
        for frame_number, index in enumerate(indices):
            sample = dataset[index]
            batch = sample_to_batch(sample)
            fusion_store.clear()
            decoder_store.clear()
            inputs = prepare_inputs(dict(batch), cfg, args.device)
            encoding = model.encoder(inputs)
            model.decoder(encoding, inputs)
            fusion_scores = torch.stack([record["weights"][:, 0] for record in fusion_store]).mean(
                dim=0
            )[0]
            rollout_scores, valid = attention_rollout(fusion_store, decoder_store)
            rollout_records = token_records(
                sample, rollout_scores.cpu().numpy(), valid.cpu().numpy()
            )
            neighbors = neighbor_records(rollout_records)
            movement, turn = movement_and_turn(sample)
            common = dict(
                batch=batch,
                sample_index=index,
                sample_path=dataset.data_list[index],
                movement_m=movement,
                turn_angle_deg=turn,
                layer=args.layer,
                top_k=args.top_k,
                view_range=args.view_range,
                colormap=args.colormap,
                marker_size_min=args.marker_size_min,
                marker_size_max=args.marker_size_max,
            )
            draw_report(
                records=rollout_records,
                out_png=all_dir / f"frame_{frame_number:06d}.png",
                attention_label="Decoder-to-input rollout",
                title_prefix="Decoder-to-Input Attention Rollout",
                attention_layer_label="Decoder cross-attention averaged over layers/steps",
                **common,
            )
            draw_neighbor_report(
                records=neighbors,
                out_png=neighbor_dir / f"frame_{frame_number:06d}.png",
                attention_label="Decoder-to-input rollout",
                title_prefix="Neighbor Decoder-to-Input Rollout",
                attention_layer_label="Decoder cross-attention averaged over layers/steps",
                **common,
            )
            frame_meta.append({"frame": frame_number, "dataset_index": index})
            with out_json.open("w") as file:
                json.dump(
                    {
                        "center_index": args.center_index,
                        "indices": indices,
                        "fps": args.fps,
                        "frames": frame_meta,
                        "frames_dir": str(root),
                        "complete": False,
                    },
                    file,
                    indent=2,
                )
            print(f"saved frame {frame_number + 1}/{len(indices)}", flush=True)

    encode_video(all_dir, args.fps, args.video_width, args.video_height, Path(args.out_all_mp4))
    encode_video(
        neighbor_dir, args.fps, args.video_width, args.video_height, Path(args.out_neighbor_mp4)
    )
    with out_json.open("w") as file:
        json.dump(
            {
                "center_index": args.center_index,
                "indices": indices,
                "fps": args.fps,
                "all_token_video": str(Path(args.out_all_mp4)),
                "neighbor_video": str(Path(args.out_neighbor_mp4)),
                "frames": frame_meta,
                "frames_dir": str(root),
                "complete": True,
            },
            file,
            indent=2,
        )
    print(f"wrote {args.out_all_mp4}")
    print(f"wrote {args.out_neighbor_mp4}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
