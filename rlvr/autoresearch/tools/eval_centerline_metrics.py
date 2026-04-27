#!/usr/bin/env python3
"""Report the centerline-reward distribution over scenes + lat-offset-to-nearest-route-lane (m).

Uses rlvr.reward.compute_centerline_score_batch — the exact function the training
reward calls — so numbers here are what the training signal actually sees.

Optional --sanity_compare runs a parallel naive reimplementation (same formula,
no caps, no time-weight, no route-deviation branch) and reports where the two
diverge. Useful for surfacing cases the reward is hiding (saturation, time-decay).

Outputs per tag:
  * centerline score distribution: mean, p5, p25, p50, p75, p95  (score is negative; p5 = worst 5%)
  * |lat_offset| (baselink → nearest route lane centerline point, meters): mean, p25, p50, p75, p95
  * fraction of scenes where score hits the cap floor (-1.0)
  * fraction of scenes with any off-route drift

Usage:
    source .venv/bin/activate
    python -m rlvr.autoresearch.tools.eval_centerline_metrics \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        [--lora_path /path/to/lora_epoch_NNN] \
        [--tag ep4] [--out_json out.json] [--sanity_compare]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.reward import RewardConfig, compute_centerline_score_batch


@torch.no_grad()
def generate_trajectory(model, model_args, data, device):
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    norm_data = {k: v.clone() for k, v in data.items()}
    norm_data = model_args.observation_normalizer(norm_data)
    norm_data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4, device=device)
    _orig = model.decoder._guidance_fn
    model.decoder._guidance_fn = None
    _, outputs = model(norm_data)
    model.decoder._guidance_fn = _orig
    return outputs["prediction"][0, 0]  # [T, 4] on device


@torch.no_grad()
def lat_offset_and_naive_score(traj: torch.Tensor, data: dict, ego_half_w: float):
    """Per-scene per-timestep quantities used for sanity-compare + lat_offset stats.

    Mirrors compute_centerline_score_batch lines 985-1017 to get |lat_offset|
    (m) from nearest route-lane centerline point, plus a NAIVE score:
    -mean(lane_usage²) with no cap, no time-weight, no route-deviation branch.
    """
    device = traj.device
    lanes = data.get("route_lanes", data.get("lanes"))
    if lanes is None:
        return None
    if lanes.dim() == 4:
        lanes = lanes[0]
    S_P = lanes.shape[0] * lanes.shape[1]
    centers = lanes[..., 0:2].reshape(S_P, 2)
    dirs = lanes[..., 2:4].reshape(S_P, 2)
    left = lanes[..., 4:6].reshape(S_P, 2)
    right = lanes[..., 6:8].reshape(S_P, 2)
    valid = centers.norm(dim=-1) > 1e-3
    dirs_n = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lat_dir = torch.stack([-dirs_n[..., 1], dirs_n[..., 0]], dim=-1)
    left_hw = (left * lat_dir).sum(dim=-1)
    right_hw = (right * lat_dir).sum(dim=-1)

    ego_pos = traj[:, :2]
    T = ego_pos.shape[0]
    diff = ego_pos.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.norm(dim=-1)
    dist = dist.masked_fill(~valid.view(1, -1).expand(T, -1), 1e6)
    min_dist, nearest = dist.min(dim=-1)

    c = centers[nearest]
    lat_vec = lat_dir[nearest]
    ego_lat = ((ego_pos - c) * lat_vec).sum(dim=-1)
    lhw = left_hw[nearest]
    rhw = right_hw[nearest]
    side_hw = torch.where(ego_lat >= 0, lhw.clamp(min=0.5), (-rhw).clamp(min=0.5))
    lane_usage = (ego_lat.abs() + ego_half_w) / side_hw  # body mode, uncapped

    naive_score = -(lane_usage ** 2).mean().item()  # uniform-weight, no cap, no route-dev
    any_off_route = bool(((min_dist > 5.0).cummax(dim=0).values & (min_dist > 5.0)).any().item())
    return {
        "lat_offset_m": ego_lat.abs().cpu().numpy(),  # [T]
        "lane_usage_uncapped": lane_usage.cpu().numpy(),  # [T]
        "naive_score": naive_score,
        "any_off_route": any_off_route,
    }


def _q(a, p): return float(np.percentile(a, p))


def report(tag, cl_scores, lat_offset_means, lat_offset_maxes, cap_floor_count, off_route_count):
    n = len(cl_scores)
    cl = np.asarray(cl_scores)
    lom = np.asarray(lat_offset_means)
    lomx = np.asarray(lat_offset_maxes)

    print(f"\n=== Centerline — {tag} ({n} scenes) ===")
    print(f"centerline_score (from compute_centerline_score_batch; negative, 0=best):")
    print(f"  mean={cl.mean():+.3f}  p5={_q(cl,5):+.3f}  p25={_q(cl,25):+.3f}  "
          f"p50={_q(cl,50):+.3f}  p75={_q(cl,75):+.3f}  p95={_q(cl,95):+.3f}  min={cl.min():+.3f}")
    print(f"  (p5 and p25 are worst-case tails; min = worst scene)")
    print(f"  hit cap floor (score ≤ -0.999): {cap_floor_count}/{n} "
          f"({100*cap_floor_count/n:.1f}%) — reward is saturating")

    print(f"|lat_offset|_mean-per-scene (baselink → nearest route-lane centerline point, m):")
    print(f"  mean={lom.mean():.2f}  p25={_q(lom,25):.2f}  p50={_q(lom,50):.2f}  "
          f"p75={_q(lom,75):.2f}  p95={_q(lom,95):.2f}  max={lom.max():.2f}")
    print(f"|lat_offset|_max-per-scene (worst timestep in each scene, m):")
    print(f"  mean={lomx.mean():.2f}  p25={_q(lomx,25):.2f}  p50={_q(lomx,50):.2f}  "
          f"p75={_q(lomx,75):.2f}  p95={_q(lomx,95):.2f}  max={lomx.max():.2f}")
    print(f"off-route scenes: {off_route_count}/{n} ({100*off_route_count/n:.1f}%)")

    return {
        "tag": tag, "n_scenes": n,
        "cl_score_mean": float(cl.mean()),
        "cl_score_p5": _q(cl, 5), "cl_score_p25": _q(cl, 25), "cl_score_p50": _q(cl, 50),
        "cl_score_p75": _q(cl, 75), "cl_score_p95": _q(cl, 95), "cl_score_min": float(cl.min()),
        "cl_score_capped_frac": float(cap_floor_count / n),
        "lat_off_mean_mean_m": float(lom.mean()),
        "lat_off_mean_p25_m": _q(lom, 25), "lat_off_mean_p50_m": _q(lom, 50),
        "lat_off_mean_p75_m": _q(lom, 75), "lat_off_mean_p95_m": _q(lom, 95),
        "lat_off_max_mean_m": float(lomx.mean()), "lat_off_max_p95_m": _q(lomx, 95),
        "off_route_frac": float(off_route_count / n),
    }


def sanity_compare(cl_scores_reward, naive_scores):
    """Compare reward-function score vs a naive uniform-weight no-cap no-route-dev score.

    Large gaps reveal where the reward is hiding signal (cap saturation, time-decay,
    route-deviation overriding raw offset).
    """
    reward = np.asarray(cl_scores_reward)
    naive = np.asarray(naive_scores)
    diff = reward - naive  # reward is typically less negative than naive (cap + time-weight)
    print(f"\n=== Sanity: reward vs. naive (uniform-weight, uncapped, no route-dev) ===")
    print(f"reward (from compute_centerline_score_batch): mean={reward.mean():+.3f}  min={reward.min():+.3f}")
    print(f"naive  (uniform-mean(lane_usage²), uncapped): mean={naive.mean():+.3f}  min={naive.min():+.3f}")
    print(f"diff=(reward - naive): mean={diff.mean():+.3f}  p5={_q(diff, 5):+.3f}  p95={_q(diff, 95):+.3f}")
    # scenes where reward is much LESS negative than naive → reward is hiding the pain
    hidden = int((diff > 1.0).sum())
    print(f"scenes where reward >> naive (diff > 1.0, reward hiding pain): {hidden}/{len(reward)}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--sanity_compare", action="store_true",
                        help="Also compute naive score and report divergence vs reward")
    parser.add_argument("--usage_cap", type=float, default=None,
                        help="Override centerline usage cap (default: RewardConfig's 1.0)")
    parser.add_argument("--time_weight_min", type=float, default=None,
                        help="Override centerline time_weight_min (1.0 = flat; default: 0.3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(Path(args.model_path), device)
    model.eval()
    if args.lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint
        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()
        print(f"Loaded LoRA from {args.lora_path}")

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    print(f"Evaluating centerline on {len(scene_paths)} scenes [{args.tag}]")

    cfg = RewardConfig()  # defaults: usage_cap=1.0, usage_mode="body", time_weight_min=0.3
    if args.usage_cap is not None:
        cfg.centerline_usage_cap = args.usage_cap
    if args.time_weight_min is not None:
        cfg.centerline_time_weight_min = args.time_weight_min
    print(f"Reward: usage_cap={cfg.centerline_usage_cap}, usage_mode={cfg.centerline_usage_mode}, "
          f"time_weight_min={cfg.centerline_time_weight_min}")
    cl_scores, naive_scores = [], []
    lat_off_means, lat_off_maxes = [], []
    cap_floor = 0
    off_route = 0

    for i, p in enumerate(scene_paths):
        data = load_npz_data(p, device)
        traj = generate_trajectory(model, model_args, data, device)  # [T, 4]
        ego_shape = data["ego_shape"][0] if data["ego_shape"].dim() > 1 else data["ego_shape"]
        ego_half_w = float(ego_shape[2]) / 2

        # OFFICIAL: the reward function the training sees.
        score = compute_centerline_score_batch(
            traj.unsqueeze(0),              # (N=1, T, 4)
            ego_shape,                      # (3,)
            data,
            usage_cap=cfg.centerline_usage_cap,
            usage_mode=cfg.centerline_usage_mode,
            time_weight_min=cfg.centerline_time_weight_min,
        )[0].item()
        cl_scores.append(score)
        # "capped" = clamped at the current cap → no gradient past this
        if score <= -(cfg.centerline_usage_cap ** 2) * 0.999:
            cap_floor += 1

        # SIDE-CHANNEL: lat_offset_m distribution + naive score for sanity_compare.
        aux = lat_offset_and_naive_score(traj, data, ego_half_w)
        if aux is not None:
            lat_off_means.append(float(aux["lat_offset_m"].mean()))
            lat_off_maxes.append(float(aux["lat_offset_m"].max()))
            naive_scores.append(aux["naive_score"])
            if aux["any_off_route"]:
                off_route += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(scene_paths)}")

    summary = report(args.tag, cl_scores, lat_off_means, lat_off_maxes, cap_floor, off_route)

    if args.sanity_compare:
        sanity_compare(cl_scores, naive_scores)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.out_json, "w"), indent=2)
        print(f"\nSaved summary to {args.out_json}")


if __name__ == "__main__":
    main()
