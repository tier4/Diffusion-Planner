#!/usr/bin/env python3
"""Diagnose GRPO training signal: generate K trajectories per scene and show reward ranking.

Uses the batched CL sampler (1 det + 7 CL sweep + 8 random) to match
what the training loop generates.

Usage:
    python -m rlvr.autoresearch.tools.diagnose_grpo_signal \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        [--lora_path /path/to/lora_epoch_NNN] \
        [--indices 5 7 8 10] \
        [--K 16] [--tag "ep4"] [--survival]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from preference_optimization.lora_utils import load_lora_checkpoint
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from rlvr.grpo_sampler import SamplerConfig
from rlvr.grpo_sampler_batched import generate_diverse_group_batched
from rlvr.reward import RewardConfig, compute_reward_batch, compute_lane_departure_penalty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def diagnose_scene(model, model_args, npz_path, K, reward_config, sampler_config):
    device = torch.device(DEVICE)
    data = load_npz_data(npz_path, device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es

    # Use batched CL sampler (same as training)
    trajs = generate_diverse_group_batched(model, model_args, data, sampler_config, device)  # [K, T, 4]

    results = []
    for k_i in range(trajs.shape[0]):
        traj_t = trajs[k_i:k_i+1]
        r = compute_reward_batch(traj_t, data, reward_config)[0]
        gate, near, _, _, _ = compute_lane_departure_penalty(traj_t, ego_shape, data)
        results.append({
            "k": k_i, "total": r.total, "progress": r.progress,
            "smoothness": r.smoothness, "lane_crossing": r.lane_crossing,
            "lane_near": r.lane_near_frac, "rb_crossing": r.rb_crossing,
            "rb_near": r.rb_near_frac,
        })

    results.sort(key=lambda x: -x["total"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--survival", action="store_true")
    # Reward config
    parser.add_argument("--enable_lane", action="store_true")
    parser.add_argument("--lane_gate", action="store_true")
    parser.add_argument("--w_progress", type=float, default=3.0)
    parser.add_argument("--near_edge_scale", type=float, default=20.0)
    parser.add_argument("--wide_edge_scale", type=float, default=5.0)
    parser.add_argument("--lane_near_scale", type=float, default=30.0)
    parser.add_argument("--lane_wide_scale", type=float, default=10.0)
    parser.add_argument("--lane_cont_scale", type=float, default=5.0)
    # Run on all scenes and just report summary
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)
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
    model.to(device)
    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
    model.eval()

    reward_mode = "survival" if args.survival else "gate"
    rcfg = RewardConfig(
        enable_lane_departure=args.enable_lane,
        lane_gate_enabled=args.lane_gate,
        w_progress=args.w_progress,
        near_edge_scale=args.near_edge_scale,
        wide_edge_scale=args.wide_edge_scale,
        lane_near_scale=args.lane_near_scale,
        lane_wide_scale=args.lane_wide_scale,
        lane_cont_scale=args.lane_cont_scale,
        reward_mode=reward_mode,
    )

    sampler_cfg = SamplerConfig(
        n_trajectories=args.K,
        enable_guidance=True,
        enable_centerline=True,
        enable_lane_keeping=True,
        enable_road_border=True,
        enable_speed=True,
        enable_lateral=True,
        enable_longitudinal=True,
        guidance_prob=0.7,
    )

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        scene_indices = args.indices
    else:
        scene_indices = list(range(len(scenes)))

    n_good, n_bad, n_easy, n_none = 0, 0, 0, 0

    for si in scene_indices:
        results = diagnose_scene(model, model_args, scenes[si], args.K, rcfg, sampler_cfg)

        n_in = sum(1 for r in results if not r["lane_crossing"])
        n_rb = sum(1 for r in results if r["rb_crossing"])
        mean_total = np.mean([r["total"] for r in results])
        spread = max(r["total"] for r in results) - min(r["total"] for r in results)

        # Classify signal
        if n_in == args.K:
            n_easy += 1
            signal = "EASY"
        elif n_in == 0:
            if spread > 5:
                n_none += 1
                signal = "RANKED"
            else:
                n_bad += 1
                signal = "BAD"
        else:
            in_mean = np.mean([r["total"] for r in results if not r["lane_crossing"]])
            out_vals = [r["total"] for r in results if r["lane_crossing"] and not r["rb_crossing"]]
            out_mean = np.mean(out_vals) if out_vals else -50.0
            if in_mean > out_mean:
                n_good += 1
                signal = "GOOD"
            else:
                n_bad += 1
                signal = "BAD"

        if not args.summary_only:
            print(f"\n{'='*80}")
            print(f"Scene {si} [{args.tag}] — {n_in}/{args.K} in-lane, {n_rb}/{args.K} rb_cross, "
                  f"mean={mean_total:+.1f}, spread={spread:.1f}, signal={signal}")
            print(f"{'='*80}")
            print(f" Rk | total   | prog  | smth  | lane | l_near | rb  ")
            print(f"----|---------|-------|-------|------|--------|-----")
            for rank, r in enumerate(results):
                lc = "OUT" if r["lane_crossing"] else " IN"
                rb = "OUT" if r["rb_crossing"] else " ok"
                print(f" {rank+1:>2} | {r['total']:>+7.1f} | {r['progress']:>5.1f} | {r['smoothness']:>5.2f} | "
                      f"{lc:>4} | {r['lane_near']:>6.3f} | {rb:>3}")

            if n_in > 0 and n_in < args.K:
                in_mean = np.mean([r["total"] for r in results if not r["lane_crossing"]])
                out_vals = [r["total"] for r in results if r["lane_crossing"] and not r["rb_crossing"]]
                out_mean = np.mean(out_vals) if out_vals else -50.0
                print(f"\n  IN={in_mean:+.1f} vs OUT={out_mean:+.1f}, gap={in_mean - out_mean:+.1f}")

        if (si + 1) % 10 == 0 and args.summary_only:
            print(f"  Processed {si+1}/{len(scene_indices)}...")

    total_usable = n_good + n_easy + n_none
    print(f"\n{'='*60}")
    print(f"Signal summary [{args.tag}] — {reward_mode} mode, {len(scene_indices)} scenes")
    print(f"{'='*60}")
    print(f"  GOOD (IN > OUT):        {n_good}/{len(scene_indices)}")
    print(f"  EASY (all in-lane):     {n_easy}/{len(scene_indices)}")
    print(f"  RANKED (survival):      {n_none}/{len(scene_indices)}")
    print(f"  BAD (no signal):        {n_bad}/{len(scene_indices)}")
    print(f"  Total usable:           {total_usable}/{len(scene_indices)}")


if __name__ == "__main__":
    main()
