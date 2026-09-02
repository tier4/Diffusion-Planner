"""Render long attention videos with model-predicted trajectories overlaid.

Three scene views are written: signal-bearing lanes/routes, all encoder tokens,
and neighbor-agent tokens. Ground-truth future trajectories are intentionally
not drawn.
"""

import argparse
import json
from pathlib import Path

import torch

from token_analysis_common import load_model, patch_fusion, prepare_inputs
from visualize_all_token_attention import draw_report as draw_all_report, token_records
from visualize_attention_rollout import attention_rollout, patch_decoder
from visualize_neighbor_attention import (
    draw_report as draw_neighbor_report,
    find_fusion,
    movement_and_turn,
    token_records as neighbor_token_records,
    sample_to_batch,
)
from diffusion_planner.dimensions import MAX_NUM_NEIGHBORS
from visualize_neighbor_attention_video import encode_video, sequence_indices
from visualize_signal_attention import draw_signal_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--valid_set_list", required=True)
    p.add_argument("--center_index", type=int, required=True)
    p.add_argument("--frames_before", type=int, default=110)
    p.add_argument("--frames_after", type=int, default=110)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--video_width", type=int, default=1920)
    p.add_argument("--video_height", type=int, default=1080)
    p.add_argument("--layer", default="mean")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--view_range", type=float, default=80.0)
    p.add_argument("--colormap", default="viridis")
    p.add_argument("--marker_size_min", type=float, default=45.0)
    p.add_argument("--marker_size_max", type=float, default=950.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--output_name", default="attention_prediction")
    p.add_argument("--overwrite_frames", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if min(args.frames_before, args.frames_after) < 0 or args.step <= 0 or args.fps <= 0:
        raise ValueError("frame counts must be non-negative; step and fps must be positive")
    if min(args.video_width, args.video_height, args.top_k) <= 0:
        raise ValueError("video dimensions and top_k must be positive")
    if args.video_width % 2 or args.video_height % 2:
        raise ValueError("video dimensions must be even for yuv420p")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from diffusion_planner.utils.dataset import DiffusionPlannerData

    dataset = DiffusionPlannerData(args.valid_set_list)
    indices = sequence_indices(dataset, args.center_index, args.frames_before, args.frames_after, args.step)
    model, cfg, _ = load_model(Path(args.run_dir).resolve(), args.device)
    fusion_store, decoder_store = [], []
    patch_fusion(find_fusion(model.encoder), fusion_store)
    patch_decoder(model.decoder, decoder_store)
    view_dirs = {name: out_dir / f"{args.output_name}_{name}.frames" for name in ("signal", "alltoken", "neighbor")}
    for path in view_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
        old = list(path.glob("frame_*.png"))
        if old and not args.overwrite_frames:
            raise FileExistsError(f"{path} contains frames; use --overwrite_frames")
        if args.overwrite_frames:
            for frame in old:
                frame.unlink()
    metadata = []
    out_json = out_dir / f"{args.output_name}.json"
    with torch.no_grad():
        for frame_no, index in enumerate(indices):
            sample = dataset[index]
            batch = sample_to_batch(sample)
            fusion_store.clear(); decoder_store.clear()
            inputs = prepare_inputs(dict(batch), cfg, args.device)
            encoding = model.encoder(inputs)
            output = model.decoder(encoding, inputs)
            prediction = output["prediction"][0].detach().cpu().numpy()
            rollout_scores, valid = attention_rollout(fusion_store, decoder_store)
            rollout_records = token_records(sample, rollout_scores.cpu().numpy(), valid.cpu().numpy())
            neighbor_scores = rollout_scores[1 : 1 + MAX_NUM_NEIGHBORS].cpu().numpy()
            neighbor_records = neighbor_token_records(
                sample, neighbor_scores, neighbor_scores[None, :]
            )
            movement, turn = movement_and_turn(sample)
            common = dict(batch=batch, sample_index=index,
                          sample_path=dataset.data_list[index], movement_m=movement,
                          turn_angle_deg=turn, layer=args.layer, top_k=args.top_k,
                          view_range=args.view_range, prediction=prediction, show_prediction=True)
            suffix = f"frame_{frame_no:06d}.png"
            # Rollout is used for the attention coloring; prediction paths are orange/teal.
            draw_signal_report(sample=sample, records=rollout_records, out_png=view_dirs["signal"] / suffix,
                               attention_label="Decoder-to-input rollout",
                               attention_layer_label="Decoder cross-attention rolled through Fusion", **common)
            draw_all_report(records=rollout_records, out_png=view_dirs["alltoken"] / suffix,
                            colormap=args.colormap, marker_size_min=args.marker_size_min,
                            marker_size_max=args.marker_size_max,
                            attention_label="Decoder-to-input rollout",
                            title_prefix="All-token Decoder Rollout", **common)
            draw_neighbor_report(records=neighbor_records,
                                 out_png=view_dirs["neighbor"] / suffix,
                                 colormap=args.colormap, marker_size_min=args.marker_size_min,
                                 marker_size_max=args.marker_size_max,
                                 attention_label="Decoder-to-input rollout",
                                 title_prefix="Neighbor-token Decoder Rollout", **common)
            metadata.append({"frame": frame_no, "dataset_index": index})
            with out_json.open("w") as file:
                json.dump({"indices": indices, "fps": args.fps, "frames": metadata,
                           "frames_dir": {k: str(v) for k, v in view_dirs.items()},
                           "top_k": args.top_k, "trajectory": "model_prediction_only",
                           "complete": False}, file, indent=2)
            print(f"saved frame {frame_no + 1}/{len(indices)}", flush=True)
    outputs = {}
    for name, frame_dir in view_dirs.items():
        mp4 = out_dir / f"{args.output_name}_{name}.mp4"
        encode_video(frame_dir, args.fps, args.video_width, args.video_height, mp4)
        outputs[name] = str(mp4)
    with out_json.open("w") as file:
        json.dump({"indices": indices, "fps": args.fps, "frames": metadata,
                   "frames_dir": {k: str(v) for k, v in view_dirs.items()},
                   "videos": outputs, "top_k": args.top_k,
                   "trajectory": "model_prediction_only", "complete": True}, file, indent=2)
    for path in outputs.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
