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
        [--reward_config w_progress=3.0 rb_near_scale=20.0 rb_wide_scale=5.0]
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
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None, help="Scene indices (default: all)"
    )
    parser.add_argument(
        "--worst_n", type=int, default=None, help="Show only N worst scenes by reward gap"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="gap",
        choices=["gap", "centerline", "lane_dep", "model_total"],
        help="Sort order for results (worst first)",
    )
    parser.add_argument(
        "--dump_json", type=str, default=None, help="Write per-scene results (sorted) to JSON file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="GRPO training config JSON. Reward thresholds and "
        "weights start from here; the per-field overrides "
        "below still win on top.",
    )
    # Reward config overrides
    parser.add_argument("--w_progress", type=float, default=None)
    parser.add_argument("--rb_near_scale", type=float, default=None)
    parser.add_argument("--rb_wide_scale", type=float, default=None)
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

    # Reward config: baseline from training config, then per-field overrides.
    rcfg = load_reward_config(args.config)
    if args.w_progress is not None:
        rcfg.w_progress = args.w_progress
    if args.rb_near_scale is not None:
        rcfg.rb_near_scale = args.rb_near_scale
    if args.rb_wide_scale is not None:
        rcfg.rb_wide_scale = args.rb_wide_scale

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
        gt = data["ego_agent_future"]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        if gt.shape[-1] == 3:
            x, y, h = gt[..., 0], gt[..., 1], gt[..., 2]
            gt = torch.stack([x, y, torch.cos(h), torch.sin(h)], dim=-1)
        r_gt = compute_reward_batch(gt, data, rcfg)[0]

        results.append(
            {
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
                "model_rb_near": r_m.rb_near_penalty,
                "gt_rb_near": r_gt.rb_near_penalty,
                "model_centerline": r_m.centerline,
                "gt_centerline": r_gt.centerline,
                "model_lane_dep": bool(r_m.lane_crossing),
                "model_lane_near_frac": r_m.lane_near_frac,
            }
        )

    sort_keys = {
        "gap": lambda x: x["gap"],
        "model_total": lambda x: x["model_total"],
        "centerline": lambda x: x["model_centerline"],
        "lane_dep": lambda x: (not x["model_lane_dep"], x["model_centerline"]),
    }
    results.sort(key=sort_keys[args.sort_by])

    if args.worst_n:
        results = results[: args.worst_n]

    print(f"\n{'=' * 100}")
    print(f"Reward vs GT — {args.tag} ({len(scene_indices)} scenes)")
    print(f"{'=' * 100}")
    print(
        f"{'Sc':>4} | {'M_rwd':>7} | {'GT_rwd':>7} | {'Gap':>7} | {'M_rb':>4} | {'M_cl':>6} | {'GT_cl':>6} | {'LD':>3} | {'M_prog':>7} | {'GT_prog':>8}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['scene_idx']:>4} | {r['model_total']:>+7.1f} | {r['gt_total']:>+7.1f} | {r['gap']:>+7.1f} | {r['model_rb_cross']:>4} | {r['model_centerline']:>+6.2f} | {r['gt_centerline']:>+6.2f} | {'Y' if r['model_lane_dep'] else ' ':>3} | {r['model_progress']:>7.1f} | {r['gt_progress']:>8.1f}"
        )

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
    m_cl = [r["model_centerline"] for r in results]
    gt_cl = [r["gt_centerline"] for r in results]
    m_ld = sum(r["model_lane_dep"] for r in results)
    print(f"  Model centerline:  mean={np.mean(m_cl):+.3f}  min={np.min(m_cl):+.3f}")
    print(f"  GT centerline:     mean={np.mean(gt_cl):+.3f}  min={np.min(gt_cl):+.3f}")
    print(f"  Model lane_dep:    {m_ld}/{len(results)}")

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDumped per-scene results to {args.dump_json}")


if __name__ == "__main__":
    main()
