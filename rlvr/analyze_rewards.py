"""Reward distribution analyzer for GRPO training diagnostics.

Samples N scenes from a dataset, generates trajectories, scores them, and
reports reward statistics to help diagnose GRPO training issues.

Usage:
    source .venv/bin/activate
    python rlvr/analyze_rewards.py \
        --model_path /path/to/model.pth \
        --npz_list /path/to/train.json \
        --n_scenes 20 --n_trajectories 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.grpo_sampler import SamplerConfig, generate_diverse_group
from rlvr.reward import RewardConfig, compute_group_advantages, compute_reward_batch


def analyze(
    model,
    model_args,
    npz_paths: list[str],
    device: torch.device,
    n_scenes: int = 20,
    n_trajectories: int = 8,
    verbose_scenes: int = 3,
    seed: int = 42,
):
    # Fix all random seeds for reproducible results across runs.
    # Same seed as GRPOTrainer.evaluate_rewards() so results are comparable.
    import random as _random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)

    rng = np.random.default_rng(seed)
    sample_paths = rng.choice(npz_paths, size=min(n_scenes, len(npz_paths)), replace=False)

    config = SamplerConfig(n_trajectories=n_trajectories)
    reward_config = RewardConfig(enable_overprogress=True)

    all_totals = []
    all_collisions = 0
    all_offroad = []
    scene_spreads = []
    scene_advantage_stds = []
    zero_advantage_scenes = 0
    components = {k: [] for k in ["safety", "progress", "smoothness", "feasibility", "centerline"]}

    model.eval()
    for i, path in enumerate(sample_paths):
        try:
            data = load_npz_data(path, device)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

        with torch.no_grad():
            sampled = generate_diverse_group(model, model_args, data, config, device)

        trajs = torch.tensor(
            np.stack([s.trajectory for s in sampled]), device=device, dtype=torch.float32
        )
        rewards = compute_reward_batch(trajs, data, reward_config)
        advantages = compute_group_advantages(rewards)

        totals = [r.total for r in rewards]
        all_totals.extend(totals)
        all_collisions += sum(1 for r in rewards if r.collision_step is not None)
        all_offroad.extend([r.off_road_fraction for r in rewards])
        scene_spreads.append(max(totals) - min(totals))
        scene_advantage_stds.append(float(np.std(advantages)))
        if np.all(advantages == 0):
            zero_advantage_scenes += 1

        for r in rewards:
            components["safety"].append(r.safety)
            components["progress"].append(r.progress)
            components["smoothness"].append(r.smoothness)
            components["feasibility"].append(r.feasibility)
            components["centerline"].append(r.centerline)

        if i < verbose_scenes:
            print(f"\nScene {i} ({Path(path).name}):")
            for j, (r, a, s) in enumerate(zip(rewards, advantages, sampled)):
                tag = "[DET]" if s.is_deterministic else f"ns={s.noise_scale:.1f}"
                coll_str = f"coll@{r.collision_step}" if r.collision_step is not None else "safe"
                print(
                    f"  {j}: total={r.total:8.1f}  adv={a:+5.2f}  "
                    f"safe={r.safety:6.1f} prog={r.progress:6.1f} "
                    f"feas={r.feasibility:6.1f} offrd={r.off_road_fraction:4.0%} "
                    f"{coll_str:>8}  {tag}"
                )

    n_trajs = len(all_totals)
    n_valid_scenes = len(scene_spreads)
    totals_arr = np.array(all_totals)
    offroad_arr = np.array(all_offroad)
    spreads_arr = np.array(scene_spreads)
    adv_stds_arr = np.array(scene_advantage_stds)

    print(f"\n{'='*70}")
    print(f"REWARD ANALYSIS: {n_valid_scenes} scenes, {n_trajs} trajectories (N={n_trajectories})")
    print(f"{'='*70}")

    print(f"\nTotal reward:")
    print(f"  mean={totals_arr.mean():8.1f}  std={totals_arr.std():7.1f}")
    print(f"  min ={totals_arr.min():8.1f}  max={totals_arr.max():7.1f}")
    print(f"  median={np.median(totals_arr):7.1f}  IQR=[{np.percentile(totals_arr,25):.1f}, {np.percentile(totals_arr,75):.1f}]")

    print(f"\nPer-scene spread (max - min reward within each group):")
    print(f"  mean={spreads_arr.mean():7.1f}  std={spreads_arr.std():6.1f}")
    print(f"  min ={spreads_arr.min():7.1f}  max={spreads_arr.max():6.1f}")

    print(f"\nPer-scene advantage std:")
    print(f"  mean={adv_stds_arr.mean():.3f}  (should be ~1.0 if rewards are diverse)")
    print(f"  zero-advantage scenes: {zero_advantage_scenes}/{n_valid_scenes}")

    print(f"\nSafety:")
    print(f"  collision rate: {all_collisions}/{n_trajs} ({all_collisions/n_trajs:.1%})")
    print(f"  off-road: mean={offroad_arr.mean():.1%}  >10%: {(offroad_arr>0.1).sum()}/{n_trajs}")

    print(f"\nWeighted component breakdown:")
    cfg = reward_config
    weights = {"safety": cfg.w_safety, "progress": cfg.w_progress, "smoothness": cfg.w_smooth,
               "feasibility": cfg.w_feasibility, "centerline": cfg.w_centerline}
    for name in ["safety", "progress", "smoothness", "feasibility", "centerline"]:
        vals = np.array(components[name]) * weights[name]
        print(f"  w*{name:12s}: mean={vals.mean():8.1f}  std={vals.std():7.1f}  [{vals.min():8.1f}, {vals.max():7.1f}]")

    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    issues = []
    if spreads_arr.mean() < 5:
        issues.append(
            f"  WEAK SIGNAL: per-scene spread={spreads_arr.mean():.1f}. "
            f"Trajectories score too similarly. GRPO has little gradient signal.\n"
            f"    -> Increase noise range, enable more guidance types, or increase N."
        )
    elif spreads_arr.mean() > 50:
        print(f"  STRONG SIGNAL: per-scene spread={spreads_arr.mean():.1f}. Clear winners/losers.")
    else:
        print(f"  MODERATE SIGNAL: per-scene spread={spreads_arr.mean():.1f}.")

    if all_collisions / n_trajs > 0.5:
        issues.append(
            f"  HIGH COLLISION RATE: {all_collisions/n_trajs:.0%}. "
            f"Most trajectories collide, dominating the reward.\n"
            f"    -> Check if the model is fundamentally poor on these scenes."
        )

    if (offroad_arr > 0.1).sum() / n_trajs > 0.3:
        issues.append(
            f"  HIGH OFF-ROAD RATE: {(offroad_arr>0.1).sum()/n_trajs:.0%} of trajectories >10% off-road.\n"
            f"    -> Model may be producing poor lane-following trajectories."
        )

    if zero_advantage_scenes > n_valid_scenes * 0.3:
        issues.append(
            f"  MANY ZERO-ADVANTAGE SCENES: {zero_advantage_scenes}/{n_valid_scenes}.\n"
            f"    -> These scenes contribute zero gradient. Check reward diversity."
        )

    if not issues:
        print("  No issues detected. Reward distribution looks healthy for GRPO.")
    else:
        for issue in issues:
            print(issue)
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="GRPO Reward Distribution Analyzer")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--npz_list", type=Path, required=True)
    parser.add_argument("--lora_path", type=Path, default=None,
                        help="Path to LoRA adapter directory (e.g. lora_epoch_002/)")
    parser.add_argument("--n_scenes", type=int, default=20,
                        help="Number of scenes to sample for analysis")
    parser.add_argument("--n_trajectories", type=int, default=8,
                        help="Trajectories per scene")
    parser.add_argument("--verbose_scenes", type=int, default=3,
                        help="Number of scenes to print in detail")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)

    if args.lora_path is not None:
        from preference_optimization.lora_utils import load_lora_checkpoint
        model = load_lora_checkpoint(model, str(args.lora_path), is_trainable=False)
        print(f"Loaded LoRA adapter from {args.lora_path}")

    model.eval()

    with open(args.npz_list) as f:
        npz_paths = json.load(f)

    print(f"Dataset: {len(npz_paths)} scenes, sampling {args.n_scenes}")
    analyze(model, model_args, npz_paths, device,
            n_scenes=args.n_scenes, n_trajectories=args.n_trajectories,
            verbose_scenes=args.verbose_scenes)


if __name__ == "__main__":
    main()
