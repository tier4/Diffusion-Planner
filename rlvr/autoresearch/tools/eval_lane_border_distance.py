#!/usr/bin/env python3
"""Evaluate lane departure + road border distance metrics.

Reports per-epoch summary table with both lane and road border metrics.

Usage:
    python -m rlvr.autoresearch.tools.eval_lane_border_distance \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        [--lora_path /path/to/lora_epoch_NNN] \
        [--tag "rw_ep5"]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from guidance_gui.generate_samples import generate_samples
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.reward import (
    compute_lane_departure_penalty,
    compute_road_border_penalty,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, lora_path=None):
    model_dir = Path(model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    args = Config(str(args_path))
    model = Diffusion_Planner(args)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    if lora_path:
        model = load_lora_checkpoint(model, lora_path)
    model.eval()
    return model, args


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    model, model_args = load_model(args.model_path, args.lora_path)

    with open(args.scenes) as f:
        scene_paths = json.load(f)
        from diffusion_planner.utils.scene_skip import filter_scene_list

        scene_paths = filter_scene_list(scene_paths, label="eval_lane_border_distance")

    print(f"Evaluating {len(scene_paths)} scenes [{args.tag}]")

    rb_crossings = 0
    rb_min_dists = []
    lane_crossings = 0
    lane_min_clearances = []

    for i, npz_path in enumerate(scene_paths):
        data = load_npz_data(npz_path, device)
        norm = {k: v.clone() for k, v in data.items()}
        norm = model_args.observation_normalizer(norm)
        traj = generate_samples(model, model_args, norm, 0.0, 1, None, device)[0]
        traj_t = torch.tensor(traj[None], device=device, dtype=torch.float32)

        es = data.get("ego_shape")
        ego_shape = es[0] if es is not None and es.dim() > 1 else es

        # Road border
        rb_gate, rb_near, rb_wide, rb_steps, rb_cont, _ = compute_road_border_penalty(
            traj_t, ego_shape, data
        )
        if rb_gate.item() < 0.5:
            rb_crossings += 1

        # Compute actual min distance (reuse road border internals)
        # We need the per_timestep_min from road border — extract it
        ls = data["line_strings"]
        if ls.dim() == 4:
            ls = ls[0]
        if ls.shape[-1] >= 4:
            border_flag = ls[..., 3]
            border_xy = ls[..., :2]
            valid = (border_flag > 0.5) & (border_xy.norm(dim=-1) > 1e-3)
            border_pts = border_xy[valid]
            if border_pts.shape[0] > 0:
                # Quick min dist from trajectory center to border
                traj_xy = torch.tensor(traj[:, :2], device=device).unsqueeze(1)
                d = (traj_xy - border_pts.unsqueeze(0)).norm(dim=-1).min(dim=1).values
                # Subtract half-width for perimeter distance approximation
                hw = ego_shape[2].item() / 2 if ego_shape is not None else 0.85
                rb_min_dists.append(float((d - hw).min().clamp(min=0).item()))
            else:
                rb_min_dists.append(float("inf"))
        else:
            rb_min_dists.append(float("inf"))

        # Lane departure
        lane_gate, lane_near, lane_wide, _, lane_cont = compute_lane_departure_penalty(
            traj_t, ego_shape, data
        )
        if lane_gate.item() < 0.5:
            lane_crossings += 1

        # Lane min clearance from the penalty internals
        # We approximate: if near_frac=0, clearance > 25cm; if wide_frac=0, clearance > 40cm
        if lane_near.item() > 0:
            lane_min_clearances.append(0.10)  # approximate: near border
        elif lane_wide.item() > 0:
            lane_min_clearances.append(0.30)  # between near and wide thresholds
        else:
            lane_min_clearances.append(0.50)  # safe

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(scene_paths)} scenes...")

    n = len(scene_paths)
    print(f"\n{'=' * 70}")
    print(f"LANE + BORDER METRICS — {args.tag} ({n} scenes)")
    print(f"{'=' * 70}")
    print(f"  Road border crossings:  {rb_crossings}/{n}")
    print(f"  Lane departures:        {lane_crossings}/{n}")
    print(
        f"  RB min dist (approx):   mean={np.mean(rb_min_dists):.3f}m  min={np.min(rb_min_dists):.3f}m"
    )
    print(f"  Lane safe scenes:       {sum(1 for c in lane_min_clearances if c > 0.40)}/{n}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "tag": args.tag,
            "n_scenes": n,
            "rb_crossings": rb_crossings,
            "lane_crossings": lane_crossings,
        }
        with open(out_dir / f"{args.tag}_lane_border_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: {out_dir / f'{args.tag}_lane_border_summary.json'}")


if __name__ == "__main__":
    main()
