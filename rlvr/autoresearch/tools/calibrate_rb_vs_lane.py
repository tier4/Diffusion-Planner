"""Calibrate RB scale bump when removing lane reward terms.

Replicates the exact rsft_v2 training generation (same scenes, same variant,
same base model) and dumps per-trajectory reward breakdowns with full
lane_near_frac / lane_wide_frac / rb_near_pen / rb_wide_pen values.

Reports:
  * Average lane vs RB penalty contribution per trajectory
  * Top-1 ranking overlap between (current config) vs (lane-off + boosted RB)
    across a grid of RB scale multipliers
  * Suggested RB scales that preserve top-1 rankings best

Usage:
    python -m rlvr.autoresearch.tools.calibrate_rb_vs_lane \
        --model_path /path/to/v4.0/best_model.pth \
        --scenes /path/to/j6_train_mixed75.json \
        --reference_config /path/to/20260417-105600_j6_rsft_mixed75/grpo_config.json \
        --output /tmp/calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from preference_optimization.utils import load_npz_data
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_sft_trainer import _normalize_batch, _stack_scene_data
from rlvr.grpo_trainer_batched import (
    generate_all_scenes_batched,
    get_generation_config_labels_for_variant,
)
from rlvr.reward import RewardConfig, compute_reward_batch


def build_reward_config_from_grpo(cfg: GRPOConfig) -> RewardConfig:
    """Build a RewardConfig from a GRPOConfig (copies fields by name)."""
    rcfg = RewardConfig()
    for field in rcfg.__dataclass_fields__:
        if hasattr(cfg, field):
            setattr(rcfg, field, getattr(cfg, field))
    return rcfg


@torch.no_grad()
def generate_for_all_scenes(model, model_args, scene_paths, config, device):
    all_data = []
    valid = []
    for p in scene_paths:
        try:
            d = load_npz_data(p, device)
            all_data.append(d)
            valid.append(p)
        except Exception as e:
            print(f"  [skip] {Path(p).name}: {e}")
    print(f"  Loaded {len(all_data)} scenes")
    batch = _stack_scene_data(all_data, device)
    norm = _normalize_batch(batch, model_args)

    gt_speeds = []
    for d in all_data:
        gt = d.get("ego_agent_future")
        if gt is not None:
            if gt.dim() == 3:
                gt = gt[0]
            gt_np = gt.cpu().numpy()
            valid_mask = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            if valid_mask.sum() >= 5:
                vel = np.diff(gt_np[valid_mask][:, :2], axis=0) / 0.1
                gt_speeds.append(float(np.linalg.norm(vel, axis=-1).max()))
            else:
                gt_speeds.append(3.0)
        else:
            gt_speeds.append(3.0)
    median_gt_speed = float(np.median(gt_speeds))

    print(f"  Generating K={config.num_generations} trajs x {len(all_data)} scenes (variant={config.generation_variant})...")
    trajs = generate_all_scenes_batched(
        model, model_args, norm, config.num_generations,
        config.noise_scale_range, device,
        gt_max_speed=median_gt_speed,
        speed_stretch=1.0,
        generation_variant=config.generation_variant,
    )
    return trajs, all_data, valid


def score_all(trajs, all_data, reward_cfg):
    N, K = trajs.shape[0], trajs.shape[1]
    breakdowns = []
    for i in range(N):
        rs = compute_reward_batch(trajs[i], all_data[i], reward_cfg)
        breakdowns.append(rs)
    return breakdowns


def summarize(breakdowns, reward_cfg, label: str) -> dict:
    """Summary stats: mean per-traj lane_pen, rb_pen, and their ranges per scene."""
    K = len(breakdowns[0])
    N = len(breakdowns)

    lane_pen = np.zeros((N, K))
    rb_pen = np.zeros((N, K))
    totals = np.zeros((N, K))
    progress = np.zeros((N, K))
    lane_near = np.zeros((N, K))
    lane_wide = np.zeros((N, K))
    rb_near = np.zeros((N, K))
    rb_wide = np.zeros((N, K))
    lane_cross = np.zeros((N, K))
    rb_cross = np.zeros((N, K))
    for i, bs in enumerate(breakdowns):
        for k, r in enumerate(bs):
            lane_near[i, k] = r.lane_near_frac
            lane_wide[i, k] = r.lane_wide_frac
            rb_near[i, k] = r.rb_near_penalty
            rb_wide[i, k] = r.rb_wide_penalty
            lane_pen[i, k] = (reward_cfg.lane_near_scale * r.lane_near_frac
                              + reward_cfg.lane_wide_scale * r.lane_wide_frac)
            rb_pen[i, k] = (reward_cfg.rb_near_scale * r.rb_near_penalty
                            + reward_cfg.rb_wide_scale * r.rb_wide_penalty)
            totals[i, k] = r.total
            progress[i, k] = r.progress
            lane_cross[i, k] = float(r.lane_crossing)
            rb_cross[i, k] = float(r.rb_crossing)

    # Per-scene winner analysis
    winner_idx = totals.argmax(axis=1)
    mean_total = totals.mean(axis=1)
    winner_total = totals[np.arange(N), winner_idx]
    winner_lane_pen = lane_pen[np.arange(N), winner_idx]
    winner_rb_pen = rb_pen[np.arange(N), winner_idx]

    # Ranking variance within each scene (what matters for top-1 selection)
    total_spread = totals.max(axis=1) - totals.min(axis=1)
    lane_pen_spread = lane_pen.max(axis=1) - lane_pen.min(axis=1)
    rb_pen_spread = rb_pen.max(axis=1) - rb_pen.min(axis=1)

    return {
        "label": label,
        "N_scenes": N, "K_trajs": K,
        "scales_used": {
            "lane_near_scale": reward_cfg.lane_near_scale,
            "lane_wide_scale": reward_cfg.lane_wide_scale,
            "rb_near_scale": reward_cfg.rb_near_scale,
            "rb_wide_scale": reward_cfg.rb_wide_scale,
            "rb_cont_scale": reward_cfg.rb_cont_scale,
            "enable_lane_departure": reward_cfg.enable_lane_departure,
        },
        "mean_per_traj": {
            "lane_near_frac": float(lane_near.mean()),
            "lane_wide_frac": float(lane_wide.mean()),
            "rb_near_pen_unscaled": float(rb_near.mean()),
            "rb_wide_pen_unscaled": float(rb_wide.mean()),
            "lane_penalty_scaled": float(lane_pen.mean()),
            "rb_penalty_scaled": float(rb_pen.mean()),
            "lane_crossing_rate": float(lane_cross.mean()),
            "rb_crossing_rate": float(rb_cross.mean()),
            "progress": float(progress.mean()),
            "total": float(totals.mean()),
        },
        "winner": {
            "mean_winner_total": float(winner_total.mean()),
            "mean_winner_lane_penalty": float(winner_lane_pen.mean()),
            "mean_winner_rb_penalty": float(winner_rb_pen.mean()),
        },
        "per_scene_spread": {
            "mean_total_spread": float(total_spread.mean()),
            "mean_lane_pen_spread": float(lane_pen_spread.mean()),
            "mean_rb_pen_spread": float(rb_pen_spread.mean()),
        },
        "winner_idx_per_scene": winner_idx.tolist(),
    }, {
        "totals": totals,
        "lane_near": lane_near, "lane_wide": lane_wide,
        "rb_near": rb_near, "rb_wide": rb_wide,
        "progress": progress,
        "lane_cross": lane_cross, "rb_cross": rb_cross,
        "breakdowns": breakdowns,
    }


def score_with_variant_config(trajs, all_data, base_cfg: RewardConfig, **overrides) -> tuple[np.ndarray, list]:
    """Re-score pre-generated trajectories with a modified reward config.

    This re-runs compute_reward_batch with a fresh config (enable_lane_departure,
    lane/RB scales modified), so survival_frac is correctly recomputed.

    Returns: (totals [N, K], list of list of RewardBreakdown).
    """
    import copy
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)

    N, K = trajs.shape[0], trajs.shape[1]
    totals = np.zeros((N, K))
    all_bds = []
    for i in range(N):
        rs = compute_reward_batch(trajs[i], all_data[i], cfg)
        all_bds.append(rs)
        for k, r in enumerate(rs):
            totals[i, k] = r.total
    return totals, all_bds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True, help="JSON list of scene NPZ paths")
    parser.add_argument("--reference_config", required=True,
                        help="grpo_config.json from the run we want to replicate")
    parser.add_argument("--n_scenes", type=int, default=None,
                        help="Limit to first N scenes for faster iteration")
    parser.add_argument("--output", default="/tmp/calibration.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load reference GRPOConfig
    with open(args.reference_config) as f:
        cfg_dict = json.load(f)
    cfg = GRPOConfig.from_json(args.reference_config)
    print(f"Reference config: variant={cfg.generation_variant}, K={cfg.num_generations}")
    print(f"  lane: enable={cfg.enable_lane_departure}, near_scale={cfg.lane_near_scale}, "
          f"wide_scale={cfg.lane_wide_scale}")
    print(f"  rb:   gate={cfg.rb_gate_enabled}, near_scale={cfg.rb_near_scale}, "
          f"wide_scale={cfg.rb_wide_scale}, cont_scale={cfg.rb_cont_scale}")

    # Build model
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

    # Load scenes
    with open(args.scenes) as f:
        scene_paths = json.load(f)
    if args.n_scenes:
        scene_paths = scene_paths[:args.n_scenes]
    print(f"Running on {len(scene_paths)} scenes")

    # Generate trajectories (single pass)
    trajs, all_data, valid_paths = generate_for_all_scenes(model, model_args, scene_paths, cfg, device)
    print(f"Trajectories shape: {trajs.shape}")  # [N, K, T, 4]

    # Score with reference config (survival + lane on + ep5 scales)
    ref_reward_cfg = build_reward_config_from_grpo(cfg)
    print(f"\nScoring with reference config...")
    ref_breakdowns = score_all(trajs, all_data, ref_reward_cfg)
    ref_summary, ref_data = summarize(ref_breakdowns, ref_reward_cfg, "reference (ep5_no_blk2 config)")

    # Score with lane fully off + various RB boosts.
    # Each variant re-runs compute_reward_batch, so survival_frac is correctly recomputed.
    rb_grid = [
        (3.0, 0.2, 0.0, "lane_off_keepRB"),
        (6.0, 0.5, 0.3, "lane_off_RB_2x_cont"),
        (8.0, 0.8, 0.4, "lane_off_RB_2.7x_cont"),
        (10.0, 1.0, 0.5, "lane_off_RB_3.3x_cont"),
        (15.0, 1.5, 0.5, "lane_off_RB_5x_cont"),
    ]
    grid_results = []
    ref_totals = ref_data["totals"]
    ref_winners = ref_totals.argmax(axis=1)
    for near_s, wide_s, cont_s, tag in rb_grid:
        new_totals, new_bds = score_with_variant_config(
            trajs, all_data, ref_reward_cfg,
            enable_lane_departure=False, lane_gate_enabled=False,
            lane_near_scale=0.0, lane_wide_scale=0.0, lane_cont_scale=0.0,
            rb_near_scale=near_s, rb_wide_scale=wide_s, rb_cont_scale=cont_s,
        )
        new_winners = new_totals.argmax(axis=1)
        top1_agreement = float((new_winners == ref_winners).mean())
        winner_total_delta = float((new_totals[np.arange(len(ref_winners)), new_winners]
                                    - ref_totals[np.arange(len(ref_winners)), ref_winners]).mean())
        # Variance preservation: how close are the top-1 new totals to reference top-1 totals
        # in absolute magnitude — we want RB penalty to contribute enough to discriminate.
        new_rb_pens = np.array([[near_s * b.rb_near_penalty + wide_s * b.rb_wide_penalty
                                  for b in scene] for scene in new_bds])
        mean_rb_spread = float((new_rb_pens.max(axis=1) - new_rb_pens.min(axis=1)).mean())
        # Safety of new winners: rb_crossing rate and lane_crossing rate among new winners
        new_winner_rb_cross = float(np.mean([
            new_bds[i][new_winners[i]].rb_crossing for i in range(len(new_winners))
        ]))
        new_winner_lane_cross = float(np.mean([
            new_bds[i][new_winners[i]].lane_crossing for i in range(len(new_winners))
        ]))
        new_winner_rb_min_dist = float(np.mean([
            new_bds[i][new_winners[i]].rb_min_dist for i in range(len(new_winners))
        ]))
        grid_results.append({
            "tag": tag,
            "rb_near_scale": near_s, "rb_wide_scale": wide_s, "rb_cont_scale": cont_s,
            "top1_agreement_vs_reference": top1_agreement,
            "mean_new_winner_total_delta": winner_total_delta,
            "mean_new_rb_pen_spread": mean_rb_spread,
            "new_winner_rb_cross_rate": new_winner_rb_cross,
            "new_winner_lane_cross_rate": new_winner_lane_cross,
            "new_winner_mean_rb_min_dist": new_winner_rb_min_dist,
            "new_winner_idx_per_scene": new_winners.tolist(),
        })
        print(f"  [{tag}] rb_near={near_s}, rb_wide={wide_s}, rb_cont={cont_s} → "
              f"top-1 agreement={top1_agreement:.2%}, winner Δ={winner_total_delta:+.2f}, "
              f"RB spread={mean_rb_spread:.2f}, new-winner rb_cross={new_winner_rb_cross:.2%}, "
              f"lane_cross={new_winner_lane_cross:.2%}, rb_min_dist={new_winner_rb_min_dist:.3f}")

    # Also score reference winners for safety baseline
    ref_winner_rb_cross = float(np.mean([
        ref_breakdowns[i][ref_winners[i]].rb_crossing for i in range(len(ref_winners))
    ]))
    ref_winner_lane_cross = float(np.mean([
        ref_breakdowns[i][ref_winners[i]].lane_crossing for i in range(len(ref_winners))
    ]))
    ref_winner_rb_min_dist = float(np.mean([
        ref_breakdowns[i][ref_winners[i]].rb_min_dist for i in range(len(ref_winners))
    ]))
    ref_summary["winner"]["rb_cross_rate"] = ref_winner_rb_cross
    ref_summary["winner"]["lane_cross_rate"] = ref_winner_lane_cross
    ref_summary["winner"]["mean_rb_min_dist"] = ref_winner_rb_min_dist
    print(f"  [REFERENCE] winner rb_cross={ref_winner_rb_cross:.2%}, lane_cross={ref_winner_lane_cross:.2%}, "
          f"rb_min_dist={ref_winner_rb_min_dist:.3f}")

    # Save
    out = {
        "reference_config_path": args.reference_config,
        "n_scenes": len(valid_paths),
        "K": cfg.num_generations,
        "reference_summary": ref_summary,
        "rb_boost_grid": grid_results,
        "notes": [
            "Top-1 agreement measures how often the new reward picks the same winner as the reference.",
            "Ranking is what matters for ranked-SFT training (we train on the winner).",
            "This analysis holds survival_frac fixed — removing enable_lane_departure would increase",
            "survival_frac for lane-crossing trajectories, potentially changing rankings further.",
        ],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved: {args.output}")
    print(f"{'='*60}")
    print(f"\nReference summary:")
    ref_s = ref_summary
    print(f"  Mean per-traj lane_penalty (scaled): {ref_s['mean_per_traj']['lane_penalty_scaled']:.3f}")
    print(f"  Mean per-traj rb_penalty (scaled):   {ref_s['mean_per_traj']['rb_penalty_scaled']:.3f}")
    print(f"  Mean per-traj lane_near_frac:        {ref_s['mean_per_traj']['lane_near_frac']:.4f}")
    print(f"  Mean per-traj lane_wide_frac:        {ref_s['mean_per_traj']['lane_wide_frac']:.4f}")
    print(f"  Mean per-traj rb_near_pen_unscaled:  {ref_s['mean_per_traj']['rb_near_pen_unscaled']:.4f}")
    print(f"  Mean per-traj rb_wide_pen_unscaled:  {ref_s['mean_per_traj']['rb_wide_pen_unscaled']:.4f}")
    print(f"  Per-scene lane_pen spread (max-min): {ref_s['per_scene_spread']['mean_lane_pen_spread']:.3f}")
    print(f"  Per-scene rb_pen spread   (max-min): {ref_s['per_scene_spread']['mean_rb_pen_spread']:.3f}")
    print(f"  Per-scene total spread:              {ref_s['per_scene_spread']['mean_total_spread']:.3f}")


if __name__ == "__main__":
    main()
