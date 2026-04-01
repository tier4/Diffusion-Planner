#!/usr/bin/env python3
"""Compare model reward vs GT reward on specified scenes.

Reports per-scene and aggregate reward breakdown (total, progress, smoothness,
rb_near, rb_crossing) for both the model's deterministic trajectory and GT.

Usage:
    python -m rlvr.autoresearch.tools.eval_reward_vs_gt \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        [--lora_path /path/to/lora_epoch_NNN] \
        [--tag "rw_ep5"] \
        [--reward_config w_progress=3.0 near_edge_scale=20.0 wide_edge_scale=5.0]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from preference_optimization.lora_utils import load_lora_checkpoint
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from guidance_gui.generate_samples import generate_samples
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Scene indices (default: all)")
    parser.add_argument("--worst_n", type=int, default=None,
                        help="Show only N worst scenes by reward gap")
    # Reward config overrides
    parser.add_argument("--w_progress", type=float, default=None)
    parser.add_argument("--near_edge_scale", type=float, default=None)
    parser.add_argument("--wide_edge_scale", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # Load model
    model_dir = Path(args.model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    model_args = Config(str(args_path))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()

    # Reward config
    rcfg = RewardConfig()
    if args.w_progress is not None:
        rcfg.w_progress = args.w_progress
    if args.near_edge_scale is not None:
        rcfg.near_edge_scale = args.near_edge_scale
    if args.wide_edge_scale is not None:
        rcfg.wide_edge_scale = args.wide_edge_scale

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        scene_indices = args.indices
    else:
        scene_indices = list(range(len(scenes)))

    results = []
    for si in scene_indices:
        data = load_npz_data(scenes[si], device)
        norm = {k: v.clone() for k, v in data.items()}
        norm = model_args.observation_normalizer(norm)

        # Model trajectory
        traj_m = generate_samples(model, model_args, norm, 0.0, 1, None, device)[0]
        traj_mt = torch.tensor(traj_m[None], device=device, dtype=torch.float32)
        r_m = compute_reward_batch(traj_mt, data, rcfg)[0]

        # GT trajectory
        gt = data['ego_agent_future']
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        if gt.shape[-1] == 3:
            x, y, h = gt[..., 0], gt[..., 1], gt[..., 2]
            gt = torch.stack([x, y, torch.cos(h), torch.sin(h)], dim=-1)
        r_gt = compute_reward_batch(gt, data, rcfg)[0]

        results.append({
            "scene_idx": si,
            "scene_name": Path(scenes[si]).stem[-30:],
            "model_total": r_m.total,
            "gt_total": r_gt.total,
            "gap": r_m.total - r_gt.total,
            "model_progress": r_m.progress,
            "gt_progress": r_gt.progress,
            "model_smoothness": r_m.smoothness,
            "gt_smoothness": r_gt.smoothness,
            "model_rb_cross": r_m.rb_crossing,
            "gt_rb_cross": r_gt.rb_crossing,
            "model_rb_near": r_m.rb_near_frac,
            "gt_rb_near": r_gt.rb_near_frac,
        })

    # Sort by gap (worst first)
    results.sort(key=lambda x: x["gap"])

    if args.worst_n:
        results = results[:args.worst_n]

    print(f"\n{'='*100}")
    print(f"Reward vs GT — {args.tag} ({len(scene_indices)} scenes)")
    print(f"{'='*100}")
    print(f"{'Sc':>4} | {'M_rwd':>7} | {'GT_rwd':>7} | {'Gap':>7} | {'M_rb':>4} | {'GT_rb':>5} | {'M_near':>6} | {'GT_near':>7} | {'M_prog':>7} | {'GT_prog':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['scene_idx']:>4} | {r['model_total']:>+7.1f} | {r['gt_total']:>+7.1f} | {r['gap']:>+7.1f} | {r['model_rb_cross']:>4} | {r['gt_rb_cross']:>5} | {r['model_rb_near']:>6.3f} | {r['gt_rb_near']:>7.3f} | {r['model_progress']:>7.1f} | {r['gt_progress']:>8.1f}")

    # Aggregates
    m_totals = [r["model_total"] for r in results]
    gt_totals = [r["gt_total"] for r in results]
    gaps = [r["gap"] for r in results]
    m_rb = sum(r["model_rb_cross"] for r in results)
    gt_rb = sum(r["gt_rb_cross"] for r in results)
    print(f"\nAggregate ({len(results)} scenes):")
    print(f"  Model mean reward: {np.mean(m_totals):+.2f}")
    print(f"  GT mean reward:    {np.mean(gt_totals):+.2f}")
    print(f"  Mean gap:          {np.mean(gaps):+.2f}")
    print(f"  Model rb_cross:    {m_rb}/{len(results)}")
    print(f"  GT rb_cross:       {gt_rb}/{len(results)}")


if __name__ == "__main__":
    main()
