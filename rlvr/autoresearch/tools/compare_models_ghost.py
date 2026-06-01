#!/usr/bin/env python3
"""Ghost-overlay closed-loop sim comparing two models on the same scene.

Runs both models from identical initial conditions, renders per-step PNGs
with both ego footprints (different colors), planned trajectories, stopped
neighbor OBBs, and assembles a WebM clip.

Usage:
    python -m rlvr.autoresearch.tools.compare_models_ghost \
        --model_a <baseline.pth> --label_a baseline \
        --model_b <trained.pth>  --label_b "trained (ep27)" \
        --scenes scene_0038.npz scene_0037.npz ... \
        --output_dir /path/out --steps 80 --make_webm

    # Single scene:
    python -m rlvr.autoresearch.tools.compare_models_ghost \
        --model_a <baseline.pth> --model_b <trained.pth> \
        --scenes scene_0038.npz --output_dir /path/out --make_webm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from preference_optimization.utils import load_npz_data

from rlvr.autoresearch.tools.ghost_sim_common import (
    GhostSimConfig,
    extract_stopped_neighbors,
    load_model,
    run_ghost_sim,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_a", required=True, help="First model (e.g. baseline)")
    parser.add_argument("--lora_a", default=None)
    parser.add_argument("--label_a", default="baseline")
    parser.add_argument("--model_b", required=True, help="Second model (e.g. trained)")
    parser.add_argument("--lora_b", default=None)
    parser.add_argument("--label_b", default="trained")
    parser.add_argument("--scenes", nargs="+", required=True, help="NPZ scene paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--advance_k", type=int, default=0)
    parser.add_argument("--view_half_m", type=float, default=30.0)
    parser.add_argument("--ego_wheelbase", type=float, default=4.76,
                        help="Ego wheelbase (m); ego footprint is rear-axle offset by (length-wheelbase)/2")
    parser.add_argument("--make_webm", action="store_true")
    parser.add_argument("--webm_fps", type=int, default=10)
    parser.add_argument("--show_lateral", action="store_true",
                        help="Show lateral offset to route centerline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare] loading model A: {args.label_a}")
    model_a, args_a = load_model(args.model_a, args.lora_a, device)
    print(f"[compare] loading model B: {args.label_b}")
    model_b, args_b = load_model(args.model_b, args.lora_b, device)

    cfg = GhostSimConfig(
        model_a_label=args.label_a,
        model_b_label=args.label_b,
        view_half_m=args.view_half_m,
        steps=args.steps,
        advance_k=args.advance_k,
        webm_fps=args.webm_fps,
        show_lateral=args.show_lateral,
        ego_wheelbase=args.ego_wheelbase,
    )

    out_root = Path(args.output_dir)

    for scene_path in args.scenes:
        scene_name = Path(scene_path).stem
        print(f"\n=== {scene_name} ===")

        data = load_npz_data(scene_path, device)
        nb_boxes = extract_stopped_neighbors(scene_path)
        if nb_boxes:
            print(f"  {len(nb_boxes)} stopped neighbor(s)")

        scene_out = out_root / scene_name if len(args.scenes) > 1 else out_root
        cfg.subtitle = scene_name

        run_ghost_sim(
            scene_path=scene_path,
            model_a=model_a, model_a_args=args_a,
            model_b=model_b, model_b_args=args_b,
            scene_data=data,
            output_dir=scene_out,
            cfg=cfg,
            neighbor_boxes=nb_boxes,
            make_webm=args.make_webm,
        )


if __name__ == "__main__":
    main()
