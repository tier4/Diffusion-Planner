"""Analyze problem areas in a dataset by running the baseline model and scoring.

Runs deterministic inference on all scenes, computes per-scene reward metrics,
and identifies clusters of problem scenes (high centerline loss, collisions,
road border crossings, lane departures).

Produces a single multi-panel summary PNG (reward distribution, RB-distance
distribution, path-length distribution, per-bag problem-count bar chart,
heading-vs-border scatter, problem-scene reward overlay) plus a top-K
problem-scene JSON and a full per-scene results JSON.

Usage:
    python -m rlvr.autoresearch.tools.analyze_problem_areas \
        --model_path <base_model.pth> \
        --scenes <path_list.json> \
        --output_dir <dir> \
        [--lora_path <lora_dir>] \
        [--tag <name>] \
        [--top_k 50]
"""

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from preference_optimization.model_utils import load_model
from rlvr.autoresearch.run_experiment import DEVICE, load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.closed_loop.batched_rollout import _batched_generate
from rlvr.reward import RewardConfig, compute_reward_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze problem areas in dataset")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None,
                        help="GRPO training config JSON. When given, reward thresholds "
                             "and weights match the live run (enable_lane_departure "
                             "is always forced on).")
    parser.add_argument("--tag", type=str, default="analysis")
    parser.add_argument("--top_k", type=int, default=50, help="Number of problem scenes to output")
    parser.add_argument("--batch_size", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load scenes
    with open(args.scenes) as f:
        scene_paths = json.load(f)
    print(f"Loaded {len(scene_paths)} scenes")

    # Load model
    model, model_args = load_model(args.model_path, device=DEVICE)
    if args.lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint
        model = load_lora_checkpoint(model, args.lora_path)
    model.eval()

    if args.config is not None:
        eval_config = load_reward_config(args.config)
        eval_config.enable_lane_departure = True
        print(f"Using reward thresholds from {args.config}")
    else:
        eval_config = RewardConfig(enable_lane_departure=True)

    # Evaluate all scenes in batches
    results = []
    for chunk_start in range(0, len(scene_paths), args.batch_size):
        chunk_paths = scene_paths[chunk_start:chunk_start + args.batch_size]
        chunk_data = []
        for sp in chunk_paths:
            try:
                chunk_data.append(load_npz_data(sp, DEVICE))
            except Exception as e:
                print(f"  Skip {Path(sp).name}: {e}")
                chunk_data.append(None)

        # Filter valid
        valid_indices = [i for i, d in enumerate(chunk_data) if d is not None]
        if not valid_indices:
            continue

        valid_data = [chunk_data[i] for i in valid_indices]

        # Batch inference
        batch = {}
        for k in valid_data[0]:
            vals = [d[k] for d in valid_data]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.cat(vals, dim=0)
            else:
                batch[k] = vals[0]

        normalizer = copy.deepcopy(model_args.observation_normalizer)
        norm_batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items()}
        norm_batch = normalizer(norm_batch)

        with torch.no_grad():
            det_trajs = _batched_generate(
                model, model_args, norm_batch,
                noise_scale=0.0, composer=None, device=DEVICE)

        # Score each scene
        for local_i, global_i in enumerate(valid_indices):
            sp = chunk_paths[global_i]
            data_i = valid_data[local_i]
            traj = det_trajs[local_i:local_i + 1]
            r = compute_reward_batch(traj, data_i, eval_config)[0]

            traj_np = det_trajs[local_i].cpu().numpy()
            pl = np.linalg.norm(np.diff(traj_np[:, :2], axis=0), axis=1).sum()

            # traj_np is [T, 4] = (x, y, cos_yaw, sin_yaw). Recover yaws via
            # atan2 and take a wrapped angle difference so the result is in
            # radians (not a difference of cosines).
            start_yaw = np.arctan2(traj_np[0, 3], traj_np[0, 2])
            end_yaw = np.arctan2(traj_np[-1, 3], traj_np[-1, 2])
            dh = end_yaw - start_yaw
            heading_chg = float(np.arctan2(np.sin(dh), np.cos(dh)))

            results.append({
                "path": sp,
                "name": Path(sp).stem,
                "bag": Path(sp).parent.name,
                "total_reward": float(r.total),
                "rb_crossing": bool(r.rb_crossing),
                "rb_min_dist": float(r.rb_min_dist),
                "lane_crossing": bool(r.lane_crossing),
                "lane_near_frac": float(r.lane_near_frac),
                "collision_step": int(r.collision_step) if r.collision_step is not None else None,
                "off_road_frac": float(r.off_road_fraction),
                "centerline_score": float(r.centerline),
                "path_length": float(pl),
                "heading_change": heading_chg,
                "stopped": bool(pl < 1.0),
            })

        print(f"  Processed {chunk_start + len(chunk_paths)}/{len(scene_paths)} scenes...")

    print(f"\nTotal scored: {len(results)} scenes")

    n = len(results)
    if n == 0:
        print("No scenes scored — nothing to summarize.")
        return

    # Compute statistics
    rb_cross = sum(1 for r in results if r["rb_crossing"])
    lane_dep = sum(1 for r in results if r["lane_crossing"])
    collisions = sum(1 for r in results if r["collision_step"] is not None)
    stopped = sum(1 for r in results if r["stopped"])
    rb_dists = [r["rb_min_dist"] for r in results]

    print(f"\n{'='*60}")
    print(f"PROBLEM AREA ANALYSIS — {args.tag}")
    print(f"{'='*60}")
    print(f"  Total scenes: {n}")
    print(f"  rb_crossings: {rb_cross}/{n} ({rb_cross/n*100:.1f}%)")
    print(f"  lane_departures: {lane_dep}/{n} ({lane_dep/n*100:.1f}%)")
    print(f"  collisions: {collisions}/{n} ({collisions/n*100:.1f}%)")
    print(f"  stopped: {stopped}/{n} ({stopped/n*100:.1f}%)")
    print(f"  rb_dist: min={min(rb_dists):.3f} p5={np.percentile(rb_dists,5):.3f} "
          f"med={np.median(rb_dists):.3f}")
    print(f"  reward: mean={np.mean([r['total_reward'] for r in results]):.1f}")

    # Sort by severity (lowest reward = worst)
    results.sort(key=lambda r: r["total_reward"])

    # Group by bag
    by_bag = defaultdict(list)
    for r in results:
        by_bag[r["bag"]].append(r)

    print(f"\n--- Per-Bag Summary ---")
    for bag, scenes_in_bag in sorted(by_bag.items()):
        n_bag = len(scenes_in_bag)
        rb_bag = sum(1 for r in scenes_in_bag if r["rb_crossing"])
        ld_bag = sum(1 for r in scenes_in_bag if r["lane_crossing"])
        col_bag = sum(1 for r in scenes_in_bag if r["collision_step"] is not None)
        avg_reward = np.mean([r["total_reward"] for r in scenes_in_bag])
        avg_rb_dist = np.mean([r["rb_min_dist"] for r in scenes_in_bag])
        print(f"  {bag}: {n_bag} scenes, rb_cross={rb_bag}, ld={ld_bag}, "
              f"collision={col_bag}, reward={avg_reward:.1f}, rb_dist={avg_rb_dist:.2f}")

    # Generate summary figure (distributions + per-bag counts + scatter)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Problem Area Analysis — {args.tag} ({n} scenes)", fontsize=14)

    # 1. Reward distribution
    rewards = [r["total_reward"] for r in results]
    axes[0, 0].hist(rewards, bins=50, color="steelblue", edgecolor="black")
    axes[0, 0].axvline(np.mean(rewards), color="red", linestyle="--", label=f"mean={np.mean(rewards):.1f}")
    axes[0, 0].set_title("Total Reward Distribution")
    axes[0, 0].set_xlabel("Reward")
    axes[0, 0].legend()

    # 2. RB min distance distribution
    axes[0, 1].hist(rb_dists, bins=50, color="coral", edgecolor="black")
    axes[0, 1].axvline(eval_config.rb_near_thresh, color="orange", linestyle="--",
                       label=f"near ({eval_config.rb_near_thresh:.2f}m)")
    axes[0, 1].axvline(eval_config.rb_cross_thresh, color="red", linestyle="--",
                       label=f"cross ({eval_config.rb_cross_thresh:.2f}m)")
    axes[0, 1].set_title("Road Border Min Distance")
    axes[0, 1].set_xlabel("Distance (m)")
    axes[0, 1].legend()

    # 3. Path length distribution
    path_lens = [r["path_length"] for r in results]
    axes[0, 2].hist(path_lens, bins=50, color="forestgreen", edgecolor="black")
    axes[0, 2].axvline(1.0, color="red", linestyle="--", label="stopped (<1m)")
    axes[0, 2].set_title("Path Length Distribution")
    axes[0, 2].set_xlabel("Path (m)")
    axes[0, 2].legend()

    # 4. Per-bag problem counts
    bag_names = sorted(by_bag.keys())
    bag_rb = [sum(1 for r in by_bag[b] if r["rb_crossing"]) for b in bag_names]
    bag_ld = [sum(1 for r in by_bag[b] if r["lane_crossing"]) for b in bag_names]
    bag_col = [sum(1 for r in by_bag[b] if r["collision_step"] is not None) for b in bag_names]
    x = range(len(bag_names))
    w = 0.25
    axes[1, 0].bar([i - w for i in x], bag_rb, w, label="RB cross", color="red")
    axes[1, 0].bar(x, bag_ld, w, label="Lane dep", color="orange")
    axes[1, 0].bar([i + w for i in x], bag_col, w, label="Collision", color="purple")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(bag_names, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_title("Problems per Bag")
    axes[1, 0].legend()

    # 5. Heading change vs rb_dist scatter
    headings = [r["heading_change"] for r in results]
    colors = ["red" if r["rb_crossing"] else ("orange" if r["lane_crossing"] else "blue")
              for r in results]
    axes[1, 1].scatter(headings, rb_dists, c=colors, alpha=0.5, s=10)
    axes[1, 1].set_xlabel("Heading Change (rad)")
    axes[1, 1].set_ylabel("RB Min Dist (m)")
    axes[1, 1].set_title("Heading vs Border Distance")
    axes[1, 1].axhline(0.45, color="orange", linestyle="--", alpha=0.5)

    # 6. Problem scene reward distribution
    reward_at_rb = [r["total_reward"] for r in results if r["rb_crossing"]]
    if reward_at_rb:
        axes[1, 2].hist(reward_at_rb, bins=30, color="red", alpha=0.5, label="RB cross")
    reward_at_ld = [r["total_reward"] for r in results if r["lane_crossing"]]
    if reward_at_ld:
        axes[1, 2].hist(reward_at_ld, bins=30, color="orange", alpha=0.5, label="Lane dep")
    reward_at_col = [r["total_reward"] for r in results if r["collision_step"] is not None]
    if reward_at_col:
        axes[1, 2].hist(reward_at_col, bins=30, color="purple", alpha=0.5, label="Collision")
    axes[1, 2].set_title("Problem Scene Reward Distribution")
    axes[1, 2].set_xlabel("Reward")
    axes[1, 2].legend()

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, f"{args.tag}_histograms.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved summary figure: {fig_path}")

    # Save top-K problem scenes
    problem_scenes = [r["path"] for r in results[:args.top_k]]
    prob_path = os.path.join(args.output_dir, f"{args.tag}_problem_scenes.json")
    with open(prob_path, "w") as f:
        json.dump(problem_scenes, f, indent=2)
    print(f"Saved top-{args.top_k} problem scenes: {prob_path}")

    # Save full results
    full_path = os.path.join(args.output_dir, f"{args.tag}_full_results.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results: {full_path}")

    # Print worst scenes
    print(f"\n--- Top 20 Worst Scenes ---")
    for r in results[:20]:
        flags = []
        if r["rb_crossing"]:
            flags.append("RB")
        if r["lane_crossing"]:
            flags.append("LD")
        if r["collision_step"] is not None:
            flags.append("COL")
        if r["stopped"]:
            flags.append("STOP")
        flag_str = " ".join(flags) if flags else "ok"
        print(f"  {r['name']}: reward={r['total_reward']:.1f}, "
              f"rb_dist={r['rb_min_dist']:.3f}, path={r['path_length']:.1f}m "
              f"[{flag_str}]")


if __name__ == "__main__":
    main()
