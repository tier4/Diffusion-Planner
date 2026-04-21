#!/usr/bin/env python3
"""Sweep the centerline usage cap and see how ranking changes.

Current reward.py caps `lane_usage` at 1.0 in `compute_centerline_score_batch`
(line 1008), so per-step penalty cannot exceed 1.0 even for trajs that cross
the lane boundary. This tool re-scores K=16 GRPO samples under configurable
caps ∈ {1.0, 1.5, ...} to see whether a higher cap:
  a) changes the top-1 pick (better differentiation of at-edge vs past-edge?)
  b) improves agreement between reward-top1 and CL-top1
  c) flags trajectories that go beyond the boundary differently from those
     that merely ride it.

Uses a self-contained copy of the centerline computation — reward.py is NOT
modified.

Usage:
    python -m rlvr.autoresearch.tools.rank_centerline_cap_sweep \
        --model_path /path/to/base.pth --lora_path /path/to/lora \
        --scenes /path/to/scenes.json --config /path/to/grpo_config.json \
        --K 16 --caps 1.0 1.5 2.0
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
)
from rlvr.reward import RewardConfig, compute_centerline_score_batch, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_centerline_with_cap(ego_trajs, ego_shape, data, usage_cap=1.0):
    """Compatibility wrapper; reward.compute_centerline_score_batch now accepts usage_cap."""
    return compute_centerline_score_batch(ego_trajs, ego_shape, data, usage_cap=usage_cap)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--generation_variant", type=str, default=None)
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=2.0)
    parser.add_argument("--caps", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--dump_json", type=str, default=None)
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
    model.to(device).eval()
    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()

    rcfg = load_reward_config(args.config) if args.config else RewardConfig(enable_overprogress=True)
    variant = args.generation_variant
    if variant is None and args.config is not None:
        with open(args.config) as f:
            variant = json.load(f).get("generation_variant", "default")
    if variant is None:
        variant = "default"

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    print(f"[{args.tag}] {len(scene_paths)} scenes, K={args.K}, variant={variant}")
    print(f"caps: {args.caps}")
    print(f"w_centerline={rcfg.w_centerline}, reward_mode={rcfg.reward_mode}\n")

    # Accumulators per cap
    per_cap = {c: {"top1_cl": [], "best_cl": [], "agree": 0, "top1_k_changed_vs_base": 0,
                   "top1_total": []} for c in args.caps}
    per_scene = []  # list of {scene_idx, t0_cl, per_cap: {cap: {top1_k, top1_cl, best_k, best_cl}}}
    base_cap = args.caps[0]

    # ---- Load all scenes and stack into one batch ----
    print(f"  loading {len(scene_paths)} scenes...")
    all_data = []
    for path in scene_paths:
        all_data.append(load_npz_data(path, device))
    batch = _stack_scene_data(all_data, device)
    norm_batch = _normalize_batch(batch, model_args)
    N = norm_batch["ego_current_state"].shape[0]

    # Per-scene gt_max_speed
    gt_max_speeds = []
    for d in all_data:
        gt = d.get("ego_agent_future")
        if gt is not None:
            if gt.dim() == 3: gt = gt[0]
            gt_np = gt.cpu().numpy()
            valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            gt_max_speeds.append(
                float(np.linalg.norm(np.diff(gt_np[valid][:, :2], axis=0) / 0.1, axis=-1).max())
                if valid.sum() >= 5 else 3.0
            )
        else:
            gt_max_speeds.append(3.0)
    # generate_all_scenes_batched takes a single scalar — use mean
    gt_v_high_batch = float(np.mean(gt_max_speeds))

    print(f"  generating K={args.K} trajectories for all {N} scenes in one batched pass (mean_v={gt_v_high_batch:.2f})...")
    trajs_all = generate_all_scenes_batched(
        model, model_args, norm_batch,
        K=args.K, noise_range=(args.noise_min, args.noise_max),
        device=device, gen_chunk_size=64,
        gt_max_speed=gt_v_high_batch, generation_variant=variant,
    )  # [N, K, T, 4]

    print(f"  scoring rewards + centerline at each cap per scene...")
    for si, path in enumerate(scene_paths):
        data = all_data[si]
        es = data.get("ego_shape")
        ego_shape = es[0] if es is not None and es.dim() > 1 else es
        trajs = trajs_all[si]  # [K, T, 4]

        base_breakdowns = []
        for k_i in range(trajs.shape[0]):
            r = compute_reward_batch(trajs[k_i:k_i+1], data, rcfg)[0]
            base_breakdowns.append(r)

        traj0 = torch.zeros(1, 1, 4, device=device); traj0[0, 0, 2] = 1.0
        t0_cl = float(compute_centerline_with_cap(traj0, ego_shape, data, usage_cap=1.0)[0].item())

        scene_record = {"scene_idx": si, "scene_name": Path(path).stem[-30:], "t0_cl": t0_cl, "per_cap": {}}
        base_top1_k = None
        for cap in args.caps:
            cl_scores = compute_centerline_with_cap(trajs, ego_shape, data, usage_cap=cap).cpu().tolist()
            per_k_totals = []
            for k_i, r in enumerate(base_breakdowns):
                total_no_cl = r.total - rcfg.w_centerline * r.centerline
                new_total = total_no_cl + rcfg.w_centerline * cl_scores[k_i]
                per_k_totals.append(new_total)
            top1_k = int(np.argmax(per_k_totals))
            best_cl_k = int(np.argmax(cl_scores))
            per_cap[cap]["top1_cl"].append(cl_scores[top1_k])
            per_cap[cap]["best_cl"].append(cl_scores[best_cl_k])
            per_cap[cap]["top1_total"].append(per_k_totals[top1_k])
            if top1_k == best_cl_k:
                per_cap[cap]["agree"] += 1
            if cap == base_cap:
                base_top1_k = top1_k
            else:
                if top1_k != base_top1_k:
                    per_cap[cap]["top1_k_changed_vs_base"] += 1
            scene_record["per_cap"][str(cap)] = {
                "top1_k": top1_k, "top1_cl": cl_scores[top1_k],
                "best_k": best_cl_k, "best_cl": cl_scores[best_cl_k],
                "top1_total": per_k_totals[top1_k],
            }
        per_scene.append(scene_record)

    n = len(scene_paths)
    print(f"\n{'='*70}")
    print(f"Cap sweep summary — {args.tag} ({n} scenes, K={args.K})")
    print(f"{'='*70}")
    print(f"\n{'cap':>6}  {'top1_cl_mean':>12} {'top1_cl_med':>11} {'best_cl_mean':>12} "
          f"{'best_cl_med':>11} {'gap_mean':>8} {'agree%':>7} {'changed_vs_base%':>16}")
    for cap in args.caps:
        t = per_cap[cap]
        top1 = np.array(t["top1_cl"])
        best = np.array(t["best_cl"])
        gaps = best - top1
        print(f"{cap:>6.2f}  {top1.mean():>+12.3f} {float(np.median(top1)):>+11.3f} "
              f"{best.mean():>+12.3f} {float(np.median(best)):>+11.3f} "
              f"{gaps.mean():>+8.3f} {100*t['agree']/n:>6.1f}% "
              f"{100*t['top1_k_changed_vs_base']/n:>15.1f}%")

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump({
                "aggregate_per_cap": {str(c): {k: v if not isinstance(v, list) else v for k, v in d.items()}
                                      for c, d in per_cap.items()},
                "per_scene": per_scene,
            }, f, indent=2)
        print(f"\nDumped per-cap arrays + per-scene to {args.dump_json}")


if __name__ == "__main__":
    main()
