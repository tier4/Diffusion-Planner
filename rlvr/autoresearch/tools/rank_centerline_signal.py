#!/usr/bin/env python3
"""Diagnose centerline signal in GRPO K-trajectory ranking.

Generates K trajectories per scene using the training generation_variant,
scores each with the full reward breakdown, and reports:

  1. Per-scene: top-1 by total reward vs top-1 by centerline alone
     (same traj? different? how far apart?)
  2. Per-scene: best centerline achieved across the K trajs vs top-1's centerline
  3. Aggregate: rank-1 centerline distribution, agreement between total-rank and
     centerline-rank

Usage:
    python -m rlvr.autoresearch.tools.rank_centerline_signal \
        --model_path /path/to/base.pth \
        --lora_path /path/to/lora_dir \
        --scenes /path/to/scenes.json \
        --config /path/to/grpo_config.json \
        [--K 16] [--generation_variant rsft_v2] \
        [--dump_json /path/out.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
    get_generation_config_labels_for_variant,
)
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True,
                        help="GRPO training config JSON (reward weights + generation_variant)")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--generation_variant", type=str, default=None,
                        help="Override variant from config (default: use config's setting)")
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=2.0)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--dump_json", type=str, default=None)
    parser.add_argument("--per_scene", action="store_true",
                        help="Print per-scene breakdown in addition to aggregate")
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

    # Reward config (required)
    rcfg = load_reward_config(args.config)

    # Generation variant: prefer CLI, then config
    variant = args.generation_variant
    if variant is None:
        with open(args.config) as f:
            cfg_json = json.load(f)
        variant = cfg_json.get("generation_variant", "default")

    slot_labels = get_generation_config_labels_for_variant(variant, args.K)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"[{args.tag}] {len(scene_paths)} scenes, K={args.K}, variant={variant}")
    print(f"w_centerline={rcfg.w_centerline}, reward_mode={rcfg.reward_mode}")

    per_scene_results = []
    for si, path in enumerate(scene_paths):
        data = load_npz_data(path, device)
        # load_npz_data returns tensors already shaped [1, ...]; wrap into a
        # batch of 1 via the trainer helpers so normalization matches training.
        batch = _stack_scene_data([data], device)
        norm_batch = _normalize_batch(batch, model_args)

        # Compute GT max speed for speed guidance
        if "ego_agent_future" in data:
            gt = data["ego_agent_future"]
            if gt.dim() == 3:
                gt = gt[0]
            gt_np = gt.cpu().numpy()
            valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            if valid.sum() >= 5:
                gt_vel = np.diff(gt_np[valid][:, :2], axis=0) / 0.1
                gt_v_high = float(np.linalg.norm(gt_vel, axis=-1).max())
            else:
                gt_v_high = 3.0
        else:
            gt_v_high = 3.0

        # Generate K trajectories [1, K, T, 4]
        trajs = generate_all_scenes_batched(
            model, model_args, norm_batch,
            K=args.K, noise_range=(args.noise_min, args.noise_max),
            device=device, gen_chunk_size=args.K,
            gt_max_speed=gt_v_high,
            generation_variant=variant,
        )
        K_trajs = trajs[0]  # [K, T, 4]

        # Score each traj
        rewards = []
        for k_i in range(K_trajs.shape[0]):
            r = compute_reward_batch(K_trajs[k_i:k_i+1], data, rcfg)[0]
            rewards.append({
                "k": k_i, "slot": slot_labels[k_i] if k_i < len(slot_labels) else f"slot_{k_i}",
                "total": r.total,
                "progress": r.progress, "smoothness": r.smoothness,
                "safety": r.safety, "feasibility": r.feasibility,
                "centerline": r.centerline,
                "lane_crossing": bool(r.lane_crossing),
                "rb_crossing": bool(r.rb_crossing),
                "rb_near": r.rb_near_penalty, "rb_wide": r.rb_wide_penalty,
            })

        # Rank-1 by total
        by_total = sorted(rewards, key=lambda x: -x["total"])
        top1 = by_total[0]
        # Rank-1 by centerline (ignores ties; if tied picks first)
        by_cl = sorted(rewards, key=lambda x: -x["centerline"])
        best_cl_traj = by_cl[0]
        # Rank of best-cl traj in total ranking
        rank_of_best_cl = next(i for i, r in enumerate(by_total) if r["k"] == best_cl_traj["k"])

        cl_values = [r["centerline"] for r in rewards]
        per_scene_results.append({
            "scene_idx": si,
            "scene_name": Path(path).stem[-30:],
            "top1_k": top1["k"],
            "top1_slot": top1["slot"],
            "top1_total": top1["total"],
            "top1_centerline": top1["centerline"],
            "best_centerline_k": best_cl_traj["k"],
            "best_centerline": best_cl_traj["centerline"],
            "best_cl_rank_by_total": rank_of_best_cl,
            "cl_spread": max(cl_values) - min(cl_values),
            "cl_mean": float(np.mean(cl_values)),
            "cl_median": float(np.median(cl_values)),
            "all_centerlines": cl_values,
            "w_cl_contribution_top1": rcfg.w_centerline * top1["centerline"],
        })

        if args.per_scene:
            print(f"\n--- Scene {si} [{Path(path).stem[-24:]}] ---")
            print(f"  CL spread across K: [{min(cl_values):+.3f}, {max(cl_values):+.3f}]  mean={np.mean(cl_values):+.3f}")
            print(f"  Top-1 (rank by total): k={top1['k']} slot={top1['slot']}  total={top1['total']:+.1f}  cl={top1['centerline']:+.3f}")
            print(f"  Best CL:               k={best_cl_traj['k']} slot={best_cl_traj['slot']}  cl={best_cl_traj['centerline']:+.3f}  rank_by_total={rank_of_best_cl+1}/{args.K}")

        if (si + 1) % 10 == 0:
            print(f"  processed {si+1}/{len(scene_paths)}")

    # ----- Aggregate -----
    n = len(per_scene_results)
    top1_cl = [r["top1_centerline"] for r in per_scene_results]
    best_cl = [r["best_centerline"] for r in per_scene_results]
    gaps = [b - t for b, t in zip(best_cl, top1_cl)]
    cl_spreads = [r["cl_spread"] for r in per_scene_results]
    agree = sum(1 for r in per_scene_results if r["top1_k"] == r["best_centerline_k"])
    top1_is_floored = sum(1 for v in top1_cl if v <= -0.99)
    best_cl_is_floored = sum(1 for v in best_cl if v <= -0.99)
    top1_cl_over_05 = sum(1 for v in top1_cl if v >= -0.5)
    best_cl_over_05 = sum(1 for v in best_cl if v >= -0.5)
    w_cl_contribution = [r["w_cl_contribution_top1"] for r in per_scene_results]

    print(f"\n{'='*70}")
    print(f"Rank-centerline signal — {args.tag} ({n} scenes, K={args.K}, variant={variant})")
    print(f"{'='*70}")
    print(f"\n--- Top-1 (by total reward) centerline scores ---")
    print(f"  mean:   {np.mean(top1_cl):+.3f}")
    print(f"  median: {np.median(top1_cl):+.3f}")
    print(f"  min:    {np.min(top1_cl):+.3f}")
    print(f"  floored (<=-0.99): {top1_is_floored}/{n}")
    print(f"  >= -0.5 (decent):  {top1_cl_over_05}/{n}")

    print(f"\n--- Best centerline available among K trajs ---")
    print(f"  mean:   {np.mean(best_cl):+.3f}")
    print(f"  median: {np.median(best_cl):+.3f}")
    print(f"  min:    {np.min(best_cl):+.3f}")
    print(f"  floored (<=-0.99): {best_cl_is_floored}/{n}")
    print(f"  >= -0.5 (decent):  {best_cl_over_05}/{n}")

    print(f"\n--- Gap: best_cl - top1_cl (how much better CL could be picked) ---")
    print(f"  mean:   +{np.mean(gaps):.3f}")
    print(f"  median: +{np.median(gaps):.3f}")
    print(f"  max:    +{np.max(gaps):.3f}  (scene {int(np.argmax(gaps))})")

    print(f"\n--- Rank agreement ---")
    print(f"  top1_by_total == top1_by_centerline: {agree}/{n} ({100*agree/n:.0f}%)")
    avg_rank_of_best_cl = np.mean([r["best_cl_rank_by_total"] for r in per_scene_results])
    print(f"  avg rank of best-CL traj in total ordering: {avg_rank_of_best_cl+1:.1f}/{args.K}")

    print(f"\n--- CL spread across K trajs (signal strength within a scene) ---")
    print(f"  mean spread:   {np.mean(cl_spreads):.3f}")
    print(f"  median spread: {np.median(cl_spreads):.3f}")

    print(f"\n--- Centerline contribution to top-1 total reward ---")
    print(f"  w_centerline * top1_cl: mean={np.mean(w_cl_contribution):+.2f}, "
          f"median={np.median(w_cl_contribution):+.2f}  "
          f"(w_centerline={rcfg.w_centerline})")

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump({
                "tag": args.tag,
                "variant": variant,
                "K": args.K,
                "n_scenes": n,
                "reward_config": {
                    "w_centerline": rcfg.w_centerline,
                    "reward_mode": rcfg.reward_mode,
                },
                "aggregate": {
                    "top1_cl_mean": float(np.mean(top1_cl)),
                    "top1_cl_median": float(np.median(top1_cl)),
                    "best_cl_mean": float(np.mean(best_cl)),
                    "best_cl_median": float(np.median(best_cl)),
                    "gap_mean": float(np.mean(gaps)),
                    "gap_median": float(np.median(gaps)),
                    "agreement_count": int(agree),
                    "top1_floored_count": int(top1_is_floored),
                    "best_cl_floored_count": int(best_cl_is_floored),
                    "avg_rank_of_best_cl": float(avg_rank_of_best_cl),
                    "cl_spread_mean": float(np.mean(cl_spreads)),
                },
                "per_scene": per_scene_results,
            }, f, indent=2)
        print(f"\nDumped results to {args.dump_json}")


if __name__ == "__main__":
    main()
