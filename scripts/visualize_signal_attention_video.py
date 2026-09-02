"""Create Fusion and Decoder-rollout videos focused on traffic-light tokens."""

import argparse
import json
from pathlib import Path

import torch
from diffusion_planner.utils.dataset import DiffusionPlannerData
from token_analysis_common import load_model, patch_fusion, prepare_inputs
from visualize_all_token_attention import all_token_attention, token_records
from visualize_attention_rollout import attention_rollout, patch_decoder
from visualize_neighbor_attention import find_fusion, movement_and_turn, sample_to_batch
from visualize_neighbor_attention_video import encode_video, sequence_indices
from visualize_signal_attention import draw_signal_report


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
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_fusion_mp4", required=True)
    parser.add_argument("--out_rollout_mp4", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--frames_dir",
        default="",
        help="persistent directory for per-frame PNGs; defaults to <out_json>.frames",
    )
    parser.add_argument(
        "--overwrite_frames",
        action="store_true",
        help="remove existing frame PNGs before rendering",
    )
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
    dataset = DiffusionPlannerData(args.valid_set_list)
    indices = sequence_indices(
        dataset, args.center_index, args.frames_before, args.frames_after, args.step
    )
    model, cfg, _ = load_model(Path(args.run_dir).resolve(), args.device)
    fusion_store, decoder_store = [], []
    patch_fusion(find_fusion(model.encoder), fusion_store)
    patch_decoder(model.decoder, decoder_store)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.frames_dir) if args.frames_dir else Path(f"{args.out_json}.frames")
    fusion_dir, rollout_dir = root / "fusion", root / "rollout"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for frame_dir in (fusion_dir, rollout_dir):
        old_frames = list(frame_dir.glob("frame_*.png"))
        if old_frames and not args.overwrite_frames:
            raise FileExistsError(
                f"{frame_dir} already contains frame PNGs; use --overwrite_frames "
                "or choose a new --frames_dir"
            )
        if args.overwrite_frames:
            for old_frame in old_frames:
                old_frame.unlink()
    metadata = []
    with torch.no_grad():
        for frame_no, index in enumerate(indices):
            sample = dataset[index]
            batch = sample_to_batch(sample)
            fusion_store.clear()
            decoder_store.clear()
            inputs = prepare_inputs(dict(batch), cfg, args.device)
            encoding = model.encoder(inputs)
            model.decoder(encoding, inputs)
            fusion_scores = all_token_attention(fusion_store, args.layer)[0]
            rollout_scores, valid = attention_rollout(fusion_store, decoder_store)
            fusion_records = token_records(sample, fusion_scores.cpu().numpy(), valid.cpu().numpy())
            rollout_records = token_records(
                sample, rollout_scores.cpu().numpy(), valid.cpu().numpy()
            )
            movement, turn = movement_and_turn(sample)
            common = dict(
                batch=batch,
                sample=sample,
                sample_index=index,
                sample_path=dataset.data_list[index],
                movement_m=movement,
                turn_angle_deg=turn,
                layer=args.layer,
                top_k=args.top_k,
                view_range=args.view_range,
            )
            draw_signal_report(
                records=fusion_records,
                out_png=fusion_dir / f"frame_{frame_no:06d}.png",
                attention_label="Fusion attention",
                attention_layer_label="Fusion ego-query attention",
                **common,
            )
            draw_signal_report(
                records=rollout_records,
                out_png=rollout_dir / f"frame_{frame_no:06d}.png",
                attention_label="Decoder-to-input rollout",
                attention_layer_label="Decoder cross-attention rolled through Fusion",
                **common,
            )
            metadata.append({"frame": frame_no, "dataset_index": index})
            with out_json.open("w") as file:
                json.dump(
                    {
                        "center_index": args.center_index,
                        "indices": indices,
                        "fps": args.fps,
                        "frames": metadata,
                        "frames_dir": str(root),
                        "complete": False,
                    },
                    file,
                    indent=2,
                )
            print(f"saved frame {frame_no + 1}/{len(indices)}", flush=True)

    encode_video(
        fusion_dir, args.fps, args.video_width, args.video_height, Path(args.out_fusion_mp4)
    )
    encode_video(
        rollout_dir, args.fps, args.video_width, args.video_height, Path(args.out_rollout_mp4)
    )
    with out_json.open("w") as file:
        json.dump(
            {
                "center_index": args.center_index,
                "indices": indices,
                "fps": args.fps,
                "frames": metadata,
                "frames_dir": str(root),
                "complete": True,
            },
            file,
            indent=2,
        )
    print(f"wrote {args.out_fusion_mp4}")
    print(f"wrote {args.out_rollout_mp4}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
