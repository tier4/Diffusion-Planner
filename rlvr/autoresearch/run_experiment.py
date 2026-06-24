"""Single GRPO experiment runner for autoresearch.

Takes a config JSON, runs training + evaluation, prints results summary.
Designed to be called by an autonomous agent in a loop.

Usage:
    source .venv/bin/activate
    python rlvr/run_experiment.py --config rlvr/configs/autoresearch/exp001.json --name exp001
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch

from guidance_gui.generate_samples import generate_samples
from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data as _load_npz_data_raw


def load_npz_data(npz_path, device):
    """Wrapper that adds v4 delay key."""
    data = _load_npz_data_raw(npz_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data


from diffusion_planner.utils.scene_skip import filter_scene_list

from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_trainer import GRPOTrainer
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These are set from CLI args in main() — no hardcoded paths.
BASE_MODEL: Path = Path(".")
PROB_SCENES_PATH: Path = Path(".")
NORMAL_POOL_PATH: Path = Path(".")
VALID_SCENES_PATH: Path = Path(".")
OUTPUT_DIR: Path = Path(".")


def load_scene_lists():
    with open(PROB_SCENES_PATH) as f:
        prob_all = json.load(f)
    with open(NORMAL_POOL_PATH) as f:
        normal_pool = json.load(f)
    with open(VALID_SCENES_PATH) as f:
        val_scenes = json.load(f)
    return prob_all, normal_pool, val_scenes


def create_training_set(prob_100, normal_pool, n_prob, n_normal, seed=42):
    rng_prob = np.random.default_rng(seed)
    rng_norm = np.random.default_rng(seed + 1000)
    prob_idx = rng_prob.choice(len(prob_100), size=min(n_prob, len(prob_100)), replace=False)
    prob_scenes = [prob_100[i] for i in prob_idx]

    if n_normal == 0:
        random.Random(seed).shuffle(prob_scenes)
        return prob_scenes

    # Deduplicate: remove prob scenes from normal pool to avoid training on duplicates
    prob_set = set(prob_scenes)
    normal_pool_deduped = [s for s in normal_pool if s not in prob_set]
    if len(normal_pool_deduped) < len(normal_pool):
        n_removed = len(normal_pool) - len(normal_pool_deduped)
        print(
            f"  [WARNING] Removed {n_removed} prob-scene duplicates from normal pool "
            f"({len(normal_pool)} -> {len(normal_pool_deduped)})"
        )
    if len(normal_pool_deduped) == 0 and n_normal > 0:
        raise ValueError(
            f"Normal pool is empty after deduplication! "
            f"All {len(normal_pool)} normal scenes overlap with prob scenes. "
            f"Set n_normal_scenes=0 in config or use a different normal_scenes file."
        )
    else:
        n_actual = min(n_normal, len(normal_pool_deduped))
        norm_idx = rng_norm.choice(len(normal_pool_deduped), size=n_actual, replace=False)
        normal_scenes = [normal_pool_deduped[i] for i in norm_idx]

    combined = prob_scenes + normal_scenes
    random.Random(seed).shuffle(combined)
    return combined


@torch.no_grad()
def evaluate_checkpoint(
    model, model_args, scene_paths, reward_config, label="", batch_size=150, baseline_cache=None
):
    """Evaluate model on scenes. Uses batched inference when batch_size > 1.

    Args:
        baseline_cache: dict mapping scene_path -> {"baseline_path": float, "gt_path": float}.
            If provided, computes progress ratios vs GT and vs baseline.
    """
    model.eval()
    totals, offroads, collisions, path_lengths = [], [], 0, []
    rb_crossings, rb_nears, rb_wides = 0, [], []
    rb_min_dists = []  # per-scene min distance to road border (metres)
    gt_progress_ratios = []  # model_path / gt_path per scene
    base_progress_ratios = []  # model_path / baseline_path per scene
    lane_departures, lane_nears, lane_wides = 0, [], []
    centerlines = []  # per-scene raw centerline score (see compute_centerline_score_batch)
    sc_crossings = 0
    sc_min_dists = []  # per-scene min OBB clearance to stopped neighbors

    if batch_size > 1:
        # Batched evaluation
        from rlvr.closed_loop.batched_rollout import _batched_generate

        # Load all scenes
        all_data = []
        all_paths = []
        for path in scene_paths:
            try:
                data = load_npz_data(path, DEVICE)
                all_data.append(data)
                all_paths.append(path)
            except Exception as e:
                print(f"  [eval] skipping {Path(path).name}: {e}")

        # Process in batches
        for chunk_start in range(0, len(all_data), batch_size):
            chunk_data = all_data[chunk_start : chunk_start + batch_size]
            chunk_paths = all_paths[chunk_start : chunk_start + batch_size]
            B_chunk = len(chunk_data)

            # Stack into batch
            import copy

            batch = {}
            for k in chunk_data[0]:
                vals = [d[k] for d in chunk_data]
                if isinstance(vals[0], torch.Tensor):
                    batch[k] = torch.cat(vals, dim=0)
                else:
                    batch[k] = vals[0]

            # Normalize
            normalizer = copy.deepcopy(model_args.observation_normalizer)
            norm_batch = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }
            norm_batch = normalizer(norm_batch)

            # Batched deterministic trajectory generation
            with torch.no_grad():
                det_trajs = _batched_generate(
                    model,
                    model_args,
                    norm_batch,
                    noise_scale=0.0,
                    composer=None,
                    device=DEVICE,
                )  # [B_chunk, T, 4]

            # Per-scene reward scoring (uses per-scene neighbor data)
            for local_i in range(B_chunk):
                traj_t = det_trajs[local_i : local_i + 1]  # [1, T, 4]
                data_i = chunk_data[local_i]
                reward = compute_reward_batch(traj_t, data_i, reward_config)[0]
                totals.append(reward.total)
                offroads.append(reward.off_road_fraction)
                if reward.collision_step is not None:
                    collisions += 1
                if reward.rb_crossing:
                    rb_crossings += 1
                rb_nears.append(reward.rb_near_penalty)
                rb_wides.append(reward.rb_wide_penalty)
                rb_min_dists.append(reward.rb_min_dist)
                if reward.lane_crossing:
                    lane_departures += 1
                lane_nears.append(reward.lane_near_frac)
                lane_wides.append(reward.lane_wide_frac)
                centerlines.append(reward.centerline)
                if reward.static_crossing:
                    sc_crossings += 1
                sc_min_dists.append(reward.sc_min_dist)
                traj_np = det_trajs[local_i].cpu().numpy()
                pl = np.linalg.norm(np.diff(traj_np[:, :2], axis=0), axis=1).sum()
                path_lengths.append(pl)
                sp = chunk_paths[local_i]
                if baseline_cache and sp in baseline_cache:
                    bc = baseline_cache[sp]
                    gt_progress_ratios.append(pl / max(bc["gt_path"], 1e-3))
                    base_progress_ratios.append(pl / max(bc["baseline_path"], 1e-3))
    else:
        # Sequential evaluation (original)
        for path in scene_paths:
            try:
                data = load_npz_data(path, DEVICE)
                norm_data = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
                }
                norm_data = model_args.observation_normalizer(norm_data)
                det_traj = generate_samples(
                    model,
                    model_args,
                    norm_data,
                    noise_scale=0.0,
                    n_samples=1,
                    composer=None,
                    device=DEVICE,
                )
                det_traj_t = torch.tensor(det_traj, device=DEVICE, dtype=torch.float32)
                reward = compute_reward_batch(det_traj_t, data, reward_config)[0]
                totals.append(reward.total)
                offroads.append(reward.off_road_fraction)
                if reward.collision_step is not None:
                    collisions += 1
                if reward.rb_crossing:
                    rb_crossings += 1
                rb_nears.append(reward.rb_near_penalty)
                rb_wides.append(reward.rb_wide_penalty)
                rb_min_dists.append(reward.rb_min_dist)
                if reward.lane_crossing:
                    lane_departures += 1
                lane_nears.append(reward.lane_near_frac)
                lane_wides.append(reward.lane_wide_frac)
                centerlines.append(reward.centerline)
                if reward.static_crossing:
                    sc_crossings += 1
                sc_min_dists.append(reward.sc_min_dist)
                pl = np.linalg.norm(np.diff(det_traj[0, :, :2], axis=0), axis=1).sum()
                path_lengths.append(pl)
                if baseline_cache and path in baseline_cache:
                    bc = baseline_cache[path]
                    gt_progress_ratios.append(pl / max(bc["gt_path"], 1e-3))
                    base_progress_ratios.append(pl / max(bc["baseline_path"], 1e-3))
            except Exception as e:
                print(f"  [eval] skipping {Path(path).name}: {e}")

    n = len(totals)
    if n == 0:
        return {
            "n_scenes": 0,
            "reward_mean": 0,
            "offroad_mean": 0,
            "collision_rate": 0,
            "path_length_mean": 0,
            "stopped_count": 0,
            "rb_crossings": 0,
            "rb_near_mean": 0,
        }

    pl_arr = np.array(path_lengths)
    rb_nears_arr = np.array(rb_nears) if rb_nears else np.zeros(1)
    rb_wides_arr = np.array(rb_wides) if rb_wides else np.zeros(1)
    rb_dists_arr = np.array(rb_min_dists) if rb_min_dists else np.full(1, 99.0)
    result = {
        "n_scenes": n,
        "reward_mean": float(np.mean(totals)),
        "offroad_mean": float(np.mean(offroads)),
        "collision_rate": collisions / n,
        "path_length_mean": float(pl_arr.mean()),
        "stopped_count": int((pl_arr < 1.0).sum()),
        "rb_crossings": rb_crossings,
        "rb_near_mean": float(rb_nears_arr.mean()),
        "rb_wide_mean": float(rb_wides_arr.mean()),
        "rb_dist_min": float(rb_dists_arr.min()),
        "rb_dist_p5": float(np.percentile(rb_dists_arr, 5)),
        "rb_dist_p25": float(np.percentile(rb_dists_arr, 25)),
        "rb_dist_median": float(np.median(rb_dists_arr)),
        "lane_departures": lane_departures,
        "lane_near_mean": float(np.mean(lane_nears)) if lane_nears else 0.0,
        "lane_wide_mean": float(np.mean(lane_wides)) if lane_wides else 0.0,
        "centerline_mean": float(np.mean(centerlines)) if centerlines else 0.0,
        "centerline_min": float(np.min(centerlines)) if centerlines else 0.0,
        "sc_crossings": sc_crossings,
        "sc_min_dist_min": float(np.min(sc_min_dists)) if sc_min_dists else 99.0,
        "sc_min_dist_p5": float(np.percentile(sc_min_dists, 5)) if sc_min_dists else 99.0,
        "sc_min_dist_mean": float(np.mean(sc_min_dists)) if sc_min_dists else 99.0,
    }
    if centerlines:
        cl_arr = np.array(centerlines)
        for p in (5, 25, 50, 75, 95):
            result[f"centerline_p{p}"] = float(np.percentile(cl_arr, p))
        # "lane-keep" cohort = scenes whose det traj CL score is NOT saturated
        # (proxy: > -2.0). Lane-change scenes saturate CL; separating them out
        # prevents a few lane-change penalties from hiding mean-CL improvements
        # on lane-keeping scenes. Uses the reward.py cl value, no re-derivation.
        lk_mask = cl_arr > -2.0
        if lk_mask.any():
            lk = cl_arr[lk_mask]
            result["centerline_lk_n"] = int(lk_mask.sum())
            result["centerline_lk_mean"] = float(lk.mean())
            for p in (5, 25, 50, 75, 95):
                result[f"centerline_lk_p{p}"] = float(np.percentile(lk, p))
        else:
            result["centerline_lk_n"] = 0
        # Flip-side: "likely lane-change / saturated" cohort.
        lc_mask = ~lk_mask
        result["centerline_lc_n"] = int(lc_mask.sum())
    # Progress ratios (only if baseline cache was provided)
    gt_pr_arr = np.array(gt_progress_ratios) if gt_progress_ratios else None
    base_pr_arr = np.array(base_progress_ratios) if base_progress_ratios else None
    if gt_pr_arr is not None:
        result["progress_vs_gt_p5"] = float(np.percentile(gt_pr_arr, 5))
        result["progress_vs_gt_p25"] = float(np.percentile(gt_pr_arr, 25))
        result["progress_vs_gt_median"] = float(np.median(gt_pr_arr))
        result["progress_vs_base_p5"] = float(np.percentile(base_pr_arr, 5))
        result["progress_vs_base_median"] = float(np.median(base_pr_arr))

    tag = f" [{label}]" if label else ""
    progress_str = ""
    if gt_pr_arr is not None:
        progress_str = (
            f"prog_vs_gt=[p5={np.percentile(gt_pr_arr, 5):.2f} med={np.median(gt_pr_arr):.2f}], "
            f"prog_vs_base=[p5={np.percentile(base_pr_arr, 5):.2f} med={np.median(base_pr_arr):.2f}], "
        )
    cl_pct_str = ""
    cl_lk_str = ""
    if centerlines:
        cl_pct_str = (
            f"cl_all=[mean={result['centerline_mean']:+.3f} "
            f"p5={result['centerline_p5']:+.3f} "
            f"p25={result['centerline_p25']:+.3f} "
            f"p50={result['centerline_p50']:+.3f} "
            f"p75={result['centerline_p75']:+.3f} "
            f"p95={result['centerline_p95']:+.3f} "
            f"min={result['centerline_min']:+.3f}]"
        )
        lk_n = result.get("centerline_lk_n", 0)
        lc_n = result.get("centerline_lc_n", 0)
        if lk_n > 0:
            cl_lk_str = (
                f", cl_lanekeep[n={lk_n}]=[mean={result['centerline_lk_mean']:+.3f} "
                f"p5={result['centerline_lk_p5']:+.3f} "
                f"p25={result['centerline_lk_p25']:+.3f} "
                f"p50={result['centerline_lk_p50']:+.3f} "
                f"p75={result['centerline_lk_p75']:+.3f} "
                f"p95={result['centerline_lk_p95']:+.3f}]"
                f", cl_sat_n={lc_n}"
            )
    rb_dist_str = (
        f"rb_dist=[min={rb_dists_arr.min():.2f} "
        f"p5={np.percentile(rb_dists_arr, 5):.2f} "
        f"p25={np.percentile(rb_dists_arr, 25):.2f} "
        f"p50={np.median(rb_dists_arr):.2f} "
        f"p75={np.percentile(rb_dists_arr, 75):.2f}]"
    )
    print(
        f"  Eval{tag}: {n} scenes, reward={result['reward_mean']:+.2f}, "
        f"rb_cross={rb_crossings}/{n}, lane_dep={lane_departures}/{n}, "
        f"rb_near={rb_nears_arr.mean():.2f}, rb_wide={rb_wides_arr.mean():.2f}, "
        f"{rb_dist_str}, "
        f"{progress_str}"
        f"lane_near={result['lane_near_mean']:.2f}, lane_wide={result['lane_wide_mean']:.2f}, "
        f"{cl_pct_str}{cl_lk_str}, "
        f"collision={result['collision_rate']:.1%}, "
        f"sc_cross={result['sc_crossings']}/{n}, "
        f"sc_dist=[min={result['sc_min_dist_min']:.2f} p5={result['sc_min_dist_p5']:.2f} mean={result['sc_min_dist_mean']:.2f}], "
        f"path={result['path_length_mean']:.1f}m, stopped={result['stopped_count']}"
    )
    return result


def _visualize_trajectories(model, model_args, prob_scenes, reward_config, run_dir, name):
    """Generate trajectory comparison images for the worst offroad scenes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    # Find scenes where base model goes offroad (scan all scenes — don't
    # subsample, that would hide worst-offroad scenes past index 100).
    offroad_indices = []
    for i, path in enumerate(prob_scenes):
        try:
            data = load_npz_data(path, DEVICE)
            norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            norm = model_args.observation_normalizer(norm)
            traj = generate_samples(model, model_args, norm, 0.0, 1, None, DEVICE)
            traj_t = torch.tensor(traj, device=DEVICE, dtype=torch.float32)
            data2 = load_npz_data(path, DEVICE)
            r = compute_reward_batch(traj_t, data2, reward_config)[0]
            offroad_indices.append(
                (
                    i,
                    r.off_road_fraction,
                    r.total,
                    np.linalg.norm(np.diff(traj[0, :, :2], axis=0), axis=1).sum(),
                )
            )
        except Exception:
            pass

    # Sort by offroad fraction and pick 6 worst + 3 best for comparison
    offroad_indices.sort(key=lambda x: -x[1])
    selected = offroad_indices[:6]

    if not selected:
        print("  No scenes to visualize")
        return

    n = len(selected)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for idx, (scene_i, offroad_frac, total, path_len) in enumerate(selected):
        ax = axes[idx // cols][idx % cols]
        path = prob_scenes[scene_i]
        data = load_npz_data(path, DEVICE)
        norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        norm = model_args.observation_normalizer(norm)
        traj = generate_samples(model, model_args, norm, 0.0, 1, None, DEVICE)[0]

        # Plot route lanes
        if "route_lanes" in data:
            rl = data["route_lanes"]
            if rl.dim() == 4:
                rl = rl[0]
            for seg_idx in range(rl.shape[0]):
                pts = rl[seg_idx, :, :2].cpu().numpy()
                valid = np.abs(pts).sum(axis=1) > 0.1
                if valid.sum() > 1:
                    ax.plot(pts[valid, 0], pts[valid, 1], "g-", alpha=0.4, linewidth=1)

        # Plot lane boundaries
        if "lanes" in data:
            lanes = data["lanes"]
            if lanes.dim() == 4:
                lanes = lanes[0]
            for seg_idx in range(min(lanes.shape[0], 60)):
                pts = lanes[seg_idx, :, :2].cpu().numpy()
                valid = np.abs(pts).sum(axis=1) > 0.1
                if valid.sum() > 1:
                    lb = lanes[seg_idx, :, 4:6].cpu().numpy()
                    rb = lanes[seg_idx, :, 6:8].cpu().numpy()
                    ax.plot(
                        (pts + lb)[valid, 0], (pts + lb)[valid, 1], "k-", alpha=0.15, linewidth=0.5
                    )
                    ax.plot(
                        (pts + rb)[valid, 0], (pts + rb)[valid, 1], "k-", alpha=0.15, linewidth=0.5
                    )

        ax.plot(traj[:, 0], traj[:, 1], "r-", linewidth=2.5)
        ax.plot(0, 0, "go", markersize=8)
        ax.set_title(
            f"Scene {scene_i}: offroad={offroad_frac:.0%} path={path_len:.1f}m rew={total:.1f}"
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{name}: trajectory visualization (6 worst offroad scenes)", fontsize=12)
    plt.tight_layout()
    out_path = run_dir / "trajectory_vis.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Visualization saved to {out_path}")


def run(
    config_path: Path,
    name: str,
    skip_baseline: bool = False,
    baseline_cache_path: Path | None = None,
    train_scenes_override: Path | None = None,
    train_epochs_override: int | None = None,
):
    # Load baseline cache (precomputed baseline/GT paths per scene)
    # Auto-detect if not specified: look for baseline_cache_val50.json in output dir
    baseline_cache = None
    if baseline_cache_path and baseline_cache_path.exists():
        with open(baseline_cache_path) as f:
            baseline_cache = json.load(f)
        print(f"Loaded baseline cache: {len(baseline_cache)} scenes from {baseline_cache_path}")
    elif not baseline_cache_path:
        auto_cache = OUTPUT_DIR / "baseline_cache_val50.json"
        if auto_cache.exists():
            with open(auto_cache) as f:
                baseline_cache = json.load(f)
            print(f"Auto-loaded baseline cache: {len(baseline_cache)} scenes from {auto_cache}")

    # Fix seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # Load config
    with open(config_path) as f:
        config_data = json.load(f)

    config_data_raw = dict(config_data)  # save before popping
    curated_normal_path = config_data.pop("curated_normal_path", None)
    prob_scenes_path = config_data.pop("prob_scenes_path", None)
    seed_lora_path = config_data.pop("seed_lora_path", None)
    train_scenes_path = config_data.pop("train_scenes_path", None)
    # --train_scenes (a single, already-assembled training set) overrides the config field.
    if train_scenes_override is not None:
        train_scenes_path = str(train_scenes_override)
    # When an exact training set is supplied, prob/normal sampling is bypassed entirely, so
    # n_prob_scenes / n_normal_scenes are not required. Otherwise they MUST be explicit (the
    # sampler silently duplicates scenes without them).
    using_exact_train = bool(train_scenes_path and Path(train_scenes_path).exists())
    if not using_exact_train and (
        "n_prob_scenes" not in config_data or "n_normal_scenes" not in config_data
    ):
        raise ValueError(
            "Config MUST explicitly set 'n_prob_scenes' and 'n_normal_scenes'. "
            "Omitting these leads to silent scene duplication bugs. "
            "Use n_prob_scenes=50, n_normal_scenes=0 for prob-only, "
            "or n_prob_scenes=50, n_normal_scenes=450 for 500-scene training. "
            "(Not needed when a single --train_scenes set is supplied.)"
        )
    n_prob = config_data.pop("n_prob_scenes", 0)
    n_normal = config_data.pop("n_normal_scenes", 0)

    grpo_config = GRPOConfig()
    for k, v in config_data.items():
        if hasattr(grpo_config, k):
            setattr(grpo_config, k, v)
    # CLI --train_epochs overrides whatever the config declares.
    if train_epochs_override is not None:
        grpo_config.train_epochs = train_epochs_override
    # Re-run __post_init__ to normalize legacy loss type names
    grpo_config.__post_init__()

    print(f"Experiment: {name}")
    print(
        f"Config: lr={grpo_config.learning_rate}, kl={grpo_config.kl_coef}, "
        f"rank={grpo_config.lora_rank}, epochs={grpo_config.train_epochs}, "
        f"scenes={n_prob}p+{n_normal}n, N={grpo_config.num_generations}"
    )
    print(f"Scene files: prob={PROB_SCENES_PATH}, normal={NORMAL_POOL_PATH}")

    # Load scenes
    prob_100, normal_pool, val_50 = load_scene_lists()
    if prob_scenes_path and Path(prob_scenes_path).exists():
        with open(prob_scenes_path) as f:
            prob_100 = json.load(f)
        print(f"Using custom prob scenes: {len(prob_100)} from {prob_scenes_path}")
    # Subsample prob scenes for eval (keep all for training, eval on 50)
    eval_rng = np.random.default_rng(42)
    prob_eval = [
        prob_100[i]
        for i in eval_rng.choice(len(prob_100), size=min(50, len(prob_100)), replace=False)
    ]
    if train_scenes_path and Path(train_scenes_path).exists():
        with open(train_scenes_path) as f:
            train_paths = json.load(f)
        print(f"Using exact training scenes: {len(train_paths)} from {train_scenes_path}")
    elif curated_normal_path and Path(curated_normal_path).exists():
        with open(curated_normal_path) as f:
            curated_pool = json.load(f)
        print(f"Using curated normal pool: {len(curated_pool)} scenes from {curated_normal_path}")
        train_paths = create_training_set(prob_100, curated_pool, n_prob, n_normal)
    else:
        train_paths = create_training_set(prob_100, normal_pool, n_prob, n_normal)
    # Drop converter-flagged skip_for_training frames from everything we train/eval on
    # (default on; reproducer-only frames are not valid supervision). Single chokepoint
    # so every scene source above is covered.
    _sk = grpo_config.skip_filtered_scenes
    _sr = grpo_config.sidecar_root
    train_paths = filter_scene_list(train_paths, sidecar_root=_sr, enabled=_sk, label="train")
    prob_eval = filter_scene_list(prob_eval, sidecar_root=_sr, enabled=_sk, label="prob_eval")
    val_50 = filter_scene_list(val_50, sidecar_root=_sr, enabled=_sk, label="val")
    prob_100 = filter_scene_list(prob_100, sidecar_root=_sr, enabled=_sk, label="prob_viz")
    n_unique = len(set(train_paths))
    n_total = len(train_paths)
    dup_msg = f" ({n_total - n_unique} DUPLICATES!)" if n_unique < n_total else ""
    print(f"Training set: {n_total} scenes ({n_unique} unique){dup_msg}")
    if n_unique < n_total:
        print(
            f"  [WARNING] Duplicate scenes detected! Config: n_prob={n_prob}, n_normal={n_normal}. "
            f"Check that --prob_scenes and --normal_scenes point to DIFFERENT files."
        )

    # Setup experiment dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_DIR / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "latest.pth"
    shutil.copy2(BASE_MODEL, checkpoint_path)
    shutil.copy2(BASE_MODEL.parent / "args.json", run_dir / "args.json")
    grpo_config.to_json(run_dir / "grpo_config.json")
    with open(run_dir / "train_scenes.json", "w") as f:
        json.dump(train_paths, f)

    # Initialize wandb logging (no-op if disabled in config)
    from rlvr.wandb_logger import WandbLogger

    wandb_log = WandbLogger.from_config(
        grpo_config,
        run_dir=str(run_dir),
        run_name=name,
        extra_tags=["autoresearch"],
        extra_config={
            "n_prob_scenes": n_prob,
            "n_normal_scenes": n_normal,
            "n_train_scenes": len(train_paths),
        },
    )

    # Initialize before try so finally block never hits UnboundLocalError
    start_time = time.time()
    best_epoch = 0
    best_prob_reward = float("-inf")
    best_prob_rb_crossings = 999
    best_val_reward = float("-inf")
    best_val_collision = 1.0
    duration = 0.0
    best_checkpoint = ""

    try:
        # Load model
        policy_model, model_args = load_model(checkpoint_path, DEVICE)

        if grpo_config.use_lora:
            if seed_lora_path and Path(seed_lora_path).exists():
                from preference_optimization.lora_utils import load_lora_checkpoint

                policy_model = load_lora_checkpoint(policy_model, seed_lora_path, is_trainable=True)
                print(f"Seeded from LoRA: {seed_lora_path}")
            else:
                from preference_optimization.lora_utils import (
                    LORA_TARGET_BLOCKS_01_REGEX,
                    LORA_TARGET_BLOCKS_02_REGEX,
                    LORA_TARGET_BLOCKS_12_REGEX,
                    LORA_TARGET_FIRST_BLOCK_REGEX,
                    LORA_TARGET_LAST_BLOCK_REGEX,
                    apply_lora,
                )

                target = {
                    "last": LORA_TARGET_LAST_BLOCK_REGEX,
                    "first": LORA_TARGET_FIRST_BLOCK_REGEX,
                    "blocks01": LORA_TARGET_BLOCKS_01_REGEX,
                    "blocks02": LORA_TARGET_BLOCKS_02_REGEX,
                    "blocks12": LORA_TARGET_BLOCKS_12_REGEX,
                }.get(grpo_config.lora_target)
                kwargs = dict(
                    r=grpo_config.lora_rank,
                    lora_alpha=grpo_config.lora_alpha,
                    lora_dropout=grpo_config.lora_dropout,
                )
                if target:
                    kwargs["target_modules"] = target
                policy_model = apply_lora(policy_model, **kwargs)

        if not grpo_config.train_dit:
            # Frozen-DiT policy-only training (exploration trainer): freeze the
            # whole planner; only the exploration policy gets an optimizer
            # (created inside GRPOExplorationTrainer).
            for p in policy_model.parameters():
                p.requires_grad_(False)
            policy_model.eval()
            optimizer = None
            print("train_dit=False: DiT frozen, training exploration policy only")
        else:
            trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=grpo_config.learning_rate)

        # Load frozen base model for full-model (non-LoRA) training when features
        # need a base reference: KL reg or baseline ego IL.
        _frozen_base_model = None
        _needs_base = grpo_config.kl_coef > 0.0 or (
            grpo_config.ego_il_weight > 0.0 and grpo_config.ego_il_mode == "baseline"
        )
        if grpo_config.neighbor_reg_anchor == "baseline":
            # The external anchor is consumed ONLY by the ranked-SFT trainer (it is
            # passed as base_model + prefer_external_base to train_epoch_ranked_sft).
            # Fail loudly rather than load-and-ignore for the other trainers.
            if (
                grpo_config.use_closed_loop
                or grpo_config.use_exploration_policy
                or grpo_config.ranked_sft_mode == "none"
            ):
                raise ValueError(
                    "neighbor_reg_anchor='baseline' is only supported by the ranked-SFT "
                    "trainer (ranked_sft_mode in gt_neighbor/baseline_neighbor/curated, with "
                    "use_closed_loop=False and use_exploration_policy=False); got "
                    f"ranked_sft_mode={grpo_config.ranked_sft_mode!r}, "
                    f"use_closed_loop={grpo_config.use_closed_loop}, "
                    f"use_exploration_policy={grpo_config.use_exploration_policy}. "
                    "Loading the anchor for another trainer would silently have no effect."
                )
            anchor_path = grpo_config.neighbor_reg_anchor_path
            if not anchor_path or not Path(anchor_path).is_file():
                raise ValueError(
                    "neighbor_reg_anchor='baseline' requires neighbor_reg_anchor_path "
                    f"to point to an existing .pth file; got {anchor_path!r}"
                )
            print(
                f"Loading EXTERNAL frozen baseline anchor from {anchor_path} "
                f"(neighbor_reg_anchor='baseline')..."
            )
            _frozen_base_model, _ = load_model(Path(anchor_path), DEVICE)
            _frozen_base_model.eval()
            for p in _frozen_base_model.parameters():
                p.requires_grad_(False)
        elif _needs_base and not grpo_config.use_lora:
            print(
                f"Loading frozen base model (kl_coef={grpo_config.kl_coef}, "
                f"ego_il={grpo_config.ego_il_weight}/{grpo_config.ego_il_mode})..."
            )
            _frozen_base_model, _ = load_model(checkpoint_path, DEVICE)
            _frozen_base_model.eval()
            for p in _frozen_base_model.parameters():
                p.requires_grad_(False)

        # Training reward config uses the configured weights (may boost w_progress
        # to prevent reward hacking where the model learns to stop instead of drive)
        train_reward_config = RewardConfig(
            w_safety=grpo_config.w_safety,
            w_progress=grpo_config.w_progress,
            w_smooth=grpo_config.w_smooth,
            w_feasibility=grpo_config.w_feasibility,
            w_centerline=grpo_config.w_centerline,
            centerline_usage_mode=grpo_config.centerline_usage_mode,
            centerline_time_weight_min=grpo_config.centerline_time_weight_min,
            rb_near_scale=grpo_config.rb_near_scale,
            rb_wide_scale=grpo_config.rb_wide_scale,
            rb_cont_scale=grpo_config.rb_cont_scale,
            rb_gate_enabled=grpo_config.rb_gate_enabled,
            rb_penalty_mode=grpo_config.rb_penalty_mode,
            rb_cross_thresh=grpo_config.rb_cross_thresh,
            rb_near_thresh=grpo_config.rb_near_thresh,
            rb_wide_thresh=grpo_config.rb_wide_thresh,
            rb_cont_thresh=grpo_config.rb_cont_thresh,
            max_lat_accel=grpo_config.max_lat_accel,
            lat_accel_scale=grpo_config.lat_accel_scale,
            enable_overprogress=grpo_config.enable_overprogress,
            overprogress_margin=grpo_config.overprogress_margin,
            overprogress_penalty=grpo_config.overprogress_penalty,
            stopped_penalty=grpo_config.stopped_penalty,
            underprogress_penalty=grpo_config.underprogress_penalty,
            underprogress_threshold=grpo_config.underprogress_threshold,
            underprogress_reference=grpo_config.underprogress_reference,
            progress_norm_scale=grpo_config.progress_norm_scale,
            enable_lane_departure=grpo_config.enable_lane_departure,
            lane_gate_enabled=grpo_config.lane_gate_enabled,
            lane_near_scale=grpo_config.lane_near_scale,
            lane_wide_scale=grpo_config.lane_wide_scale,
            lane_cont_scale=grpo_config.lane_cont_scale,
            lane_cross_thresh=grpo_config.lane_cross_thresh,
            lane_near_thresh=grpo_config.lane_near_thresh,
            lane_wide_thresh=grpo_config.lane_wide_thresh,
            lane_cont_thresh=grpo_config.lane_cont_thresh,
            static_collision_enabled=grpo_config.static_collision_enabled,
            sc_gate_enabled=grpo_config.sc_gate_enabled,
            sc_penalty_mode=grpo_config.sc_penalty_mode,
            sc_near_scale=grpo_config.sc_near_scale,
            sc_wide_scale=grpo_config.sc_wide_scale,
            sc_cont_scale=grpo_config.sc_cont_scale,
            sc_cross_thresh=grpo_config.sc_cross_thresh,
            sc_near_thresh=grpo_config.sc_near_thresh,
            sc_wide_thresh=grpo_config.sc_wide_thresh,
            sc_cont_thresh=grpo_config.sc_cont_thresh,
            sc_neighbor_vel_thresh=grpo_config.sc_neighbor_vel_thresh,
            sc_neighbor_disp_thresh=grpo_config.sc_neighbor_disp_thresh,
            sc_ego_min_speed=grpo_config.sc_ego_min_speed,
            max_yaw_rate=grpo_config.max_yaw_rate,
            max_steer=grpo_config.max_steer,
            kinematic_margin=grpo_config.kinematic_margin,
            reward_mode=grpo_config.reward_mode,
        )
        # Eval reward config: mirrors the training reward so printed reward
        # is directly comparable to training. Turns on lane-departure check
        # so metrics (lane_dep count, lane_near/wide fracs) are populated
        # regardless of whether training has the lane gate enabled.
        eval_reward_config = dc_replace(train_reward_config, enable_lane_departure=True)

        if grpo_config.use_closed_loop:
            from rlvr.closed_loop.closed_loop_trainer import ClosedLoopExplorationTrainer

            trainer = ClosedLoopExplorationTrainer(
                policy_model=policy_model,
                model_args=model_args,
                dit_optimizer=optimizer,
                device=DEVICE,
                run_dir=run_dir,
                config=grpo_config,
                use_lora=grpo_config.use_lora,
            )
        elif grpo_config.use_exploration_policy:
            from rlvr.grpo_exploration_trainer import GRPOExplorationTrainer

            trainer = GRPOExplorationTrainer(
                policy_model=policy_model,
                model_args=model_args,
                dit_optimizer=optimizer,
                device=DEVICE,
                run_dir=run_dir,
                config=grpo_config,
                use_lora=grpo_config.use_lora,
            )
        else:
            trainer = GRPOTrainer(
                policy_model=policy_model,
                model_args=model_args,
                optimizer=optimizer,
                device=DEVICE,
                run_dir=run_dir,
                config=grpo_config,
                use_lora=grpo_config.use_lora,
            )

        # Evaluate base model (can skip if baseline numbers are already known)
        if skip_baseline:
            print("\nSkipping base model evaluation (--skip_baseline)")
            base_prob = {"reward_mean": float("-inf"), "rb_crossings": 999, "collision_rate": 1.0}
            base_val = {"reward_mean": float("-inf"), "rb_crossings": 999, "collision_rate": 1.0}
        else:
            print("\nBase model evaluation:")
            base_prob = evaluate_checkpoint(
                policy_model, model_args, prob_eval, eval_reward_config, "base-prob"
            )
            base_val = evaluate_checkpoint(
                policy_model, model_args, val_50, eval_reward_config, "base-val"
            )

        trainer._eval_scene_paths = val_50

        # Update best trackers from baseline eval
        best_prob_reward = base_prob["reward_mean"]
        best_prob_rb_crossings = base_prob["rb_crossings"]
        best_val_reward = base_val["reward_mean"]
        best_val_collision = base_val["collision_rate"]

        args_dict = {"exp_name": name}

        for epoch in range(1, grpo_config.train_epochs + 1):
            print(f"\n--- Epoch {epoch}/{grpo_config.train_epochs} ---")

            if epoch == 1 and hasattr(trainer, "save_epoch1_baselines"):
                trainer.save_epoch1_baselines(train_paths)

            if not grpo_config.use_exploration_policy and not grpo_config.use_closed_loop:
                if grpo_config.ranked_sft_mode != "none":
                    # Ranked SFT: generate N trajs, pick best, SFT on it
                    from rlvr.grpo_sft_trainer import train_epoch_ranked_sft

                    # Optionally load a pre-trained exploration policy for guided generation
                    _explorer = None
                    if (
                        grpo_config.ranked_sft_use_explorer
                        and grpo_config.exploration_checkpoint_path
                    ):
                        from pathlib import Path as _P

                        from exploration_policy.model import (
                            ExplorationPolicy,
                            ExplorationPolicyConfig,
                        )

                        _ckpt = _P(grpo_config.exploration_checkpoint_path)
                        if not _ckpt.exists():
                            print(f"  WARNING: exploration_checkpoint_path not found: {_ckpt}")
                            print(f"  Falling back to standard generation (no explorer)")
                        elif not hasattr(run, "_cached_explorer") or getattr(
                            run, "_cached_explorer_path", None
                        ) != str(_ckpt):
                            _ep_cfg = ExplorationPolicyConfig(
                                hidden_dim=grpo_config.exploration_hidden_dim,
                                n_mixer_layers=grpo_config.exploration_n_mixer_layers,
                                n_attn_heads=grpo_config.exploration_n_attn_heads,
                                dropout=grpo_config.exploration_dropout,
                                encoder_hidden_dim=model_args.hidden_dim,
                                head_init=grpo_config.exploration_head_init,
                                head_raw_scale=grpo_config.exploration_head_raw_scale,
                                heads=list(grpo_config.exploration_heads),
                            )
                            run._cached_explorer = ExplorationPolicy(
                                _ep_cfg,
                                ref_seq_len=model_args.future_len,
                            ).to(DEVICE)
                            _state = torch.load(_ckpt, map_location=DEVICE, weights_only=False)
                            run._cached_explorer.load_state_dict(_state, strict=False)
                            run._cached_explorer.eval()
                            run._cached_explorer_path = str(_ckpt)
                            print(f"  Loaded frozen explorer from {_ckpt}")
                        _explorer = getattr(run, "_cached_explorer", None)

                    # Create explorer optimizer if training jointly
                    _explorer_opt = None
                    if _explorer is not None and not grpo_config.ranked_sft_freeze_explorer:
                        if not hasattr(run, "_cached_explorer_opt"):
                            run._cached_explorer_opt = torch.optim.AdamW(
                                _explorer.parameters(),
                                lr=grpo_config.exploration_lr,
                            )
                        _explorer_opt = run._cached_explorer_opt

                    metrics = train_epoch_ranked_sft(
                        model=policy_model,
                        model_args=model_args,
                        optimizer=optimizer,
                        scene_paths=train_paths,
                        config=grpo_config,
                        reward_config=train_reward_config,
                        device=DEVICE,
                        epoch=epoch,
                        exploration_policy=_explorer,
                        exploration_optimizer=_explorer_opt,
                        run_dir=run_dir,
                        base_model=_frozen_base_model,
                    )
                else:
                    # Fully batched training: all scenes in ~5 forward passes
                    from rlvr.grpo_trainer_batched import train_epoch_batched

                    metrics = train_epoch_batched(
                        model=policy_model,
                        model_args=model_args,
                        optimizer=optimizer,
                        scene_paths=train_paths,
                        config=grpo_config,
                        reward_config=train_reward_config,
                        device=DEVICE,
                        epoch=epoch,
                    )
            else:
                metrics = trainer.train_epoch(train_paths, epoch)
            trainer.log_metrics(epoch, metrics)
            trainer.save_checkpoint(epoch, args_dict)

            if grpo_config.use_exploration_policy and not grpo_config.train_dit:
                # Frozen DiT: evaluate_checkpoint ignores the policy and would
                # return identical numbers every epoch. Score what the explorer
                # actually does: deterministic policy-guided trajectories.
                prob_result = trainer.evaluate_policy_guided(
                    prob_eval, eval_reward_config, f"epoch{epoch}-prob"
                )
                _dump_rows = getattr(trainer, "last_eval_rows", None)
                if _dump_rows is not None:
                    with open(run_dir / f"policy_eval_prob_epoch_{epoch:03d}.json", "w") as _pf:
                        json.dump(_dump_rows, _pf, indent=1)
                val_eval = trainer.evaluate_policy_guided(
                    val_50, eval_reward_config, f"epoch{epoch}-val"
                )
                _dump_rows = getattr(trainer, "last_eval_rows", None)
                if _dump_rows is not None:
                    with open(run_dir / f"policy_eval_val_epoch_{epoch:03d}.json", "w") as _pf:
                        json.dump(_dump_rows, _pf, indent=1)
            else:
                prob_result = evaluate_checkpoint(
                    policy_model,
                    model_args,
                    prob_eval,
                    eval_reward_config,
                    f"epoch{epoch}-prob",
                    baseline_cache=baseline_cache,
                )
                val_eval = evaluate_checkpoint(
                    policy_model,
                    model_args,
                    val_50,
                    eval_reward_config,
                    f"epoch{epoch}-val",
                    baseline_cache=baseline_cache,
                )

            # Log to wandb
            wandb_log.log_training(epoch, metrics)
            wandb_log.log_eval(epoch, prob_result=prob_result, val_result=val_eval)
            _ra_path = run_dir / f"rank_analytics_epoch_{epoch:03d}.json"
            if _ra_path.exists():
                with open(_ra_path) as _f:
                    wandb_log.log_rank_analytics(epoch, json.load(_f))

            # Track best: highest prob deterministic reward (with val sanity check > -5)
            is_better = (
                prob_result["reward_mean"] > best_prob_reward and val_eval["reward_mean"] > -5
            )

            if is_better:
                best_prob_reward = prob_result["reward_mean"]
                best_prob_rb_crossings = prob_result["rb_crossings"]
                best_val_reward = val_eval["reward_mean"]
                best_val_collision = val_eval["collision_rate"]
                best_epoch = epoch
                if not grpo_config.train_dit:
                    best_checkpoint = str(run_dir / f"exploration_policy_epoch_{epoch:03d}.pth")
                elif grpo_config.use_lora:
                    best_checkpoint = str(run_dir / f"lora_epoch_{epoch:03d}")
                else:
                    best_checkpoint = str(run_dir / "latest.pth")

            # Early stopping on genuine safety collapse. Uses per-scene safety
            # rates rather than total reward, so it is invariant to reward
            # weight choices (e.g. raising w_centerline can push total reward
            # deeply negative without the model actually becoming unsafe, which
            # the old total-reward threshold would false-trigger on).
            _n_val = max(val_eval["n_scenes"], 1)
            _rb_cross_rate = val_eval["rb_crossings"] / _n_val
            _collision_rate = val_eval["collision_rate"]
            if grpo_config.early_stop_on_collapse and (
                _rb_cross_rate > grpo_config.collapse_rb_threshold
                or _collision_rate > grpo_config.collapse_collision_threshold
            ):
                print(
                    f"  Val collapsed (rb_cross={_rb_cross_rate:.1%}, "
                    f"collision={_collision_rate:.1%}), stopping early"
                )
                break

        # Keep ALL checkpoints — don't delete anything. Disk space is cheap,
        # losing the best checkpoint is not.

        # Generate trajectory visualization for the best checkpoint
        if best_checkpoint and Path(best_checkpoint).exists():
            _visualize_trajectories(
                policy_model, model_args, prob_100, eval_reward_config, run_dir, name
            )

        # Cross-epoch rank analytics summary
        try:
            from rlvr.rank_analytics import save_cross_epoch_summary

            save_cross_epoch_summary(run_dir)
        except Exception as e:
            print(f"  [rank_analytics] Cross-epoch summary failed: {e}")
    finally:
        duration = (time.time() - start_time) / 60
        wandb_log.finish(
            {
                "best_epoch": best_epoch,
                "best_prob_reward": best_prob_reward,
                "best_prob_rb_crossings": best_prob_rb_crossings,
                "best_val_reward": best_val_reward,
                "duration_min": duration,
            }
        )

    # Print final summary (machine-parseable)
    print("\n---")
    print(f"name:             {name}")
    print(f"prob_rb_crossings: {best_prob_rb_crossings}")
    print(f"prob_reward:      {best_prob_reward:.2f}")
    print(f"val_reward:       {best_val_reward:.2f}")
    print(f"val_collision:    {best_val_collision:.2f}")
    print(f"best_epoch:       {best_epoch}")
    print(f"duration_min:     {duration:.1f}")
    print(f"checkpoint:       {best_checkpoint}")
    print(f"run_dir:          {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run a single GRPO experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config JSON")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to base model .pth")
    parser.add_argument(
        "--train_scenes",
        type=Path,
        default=None,
        help="JSON list of training scene NPZ paths (single combined set). When given, this is "
        "used verbatim as the SFT training set and --prob_scenes/--normal_scenes are not needed.",
    )
    parser.add_argument(
        "--prob_scenes",
        type=Path,
        default=None,
        help="JSON list of problem scene NPZ paths (advanced: prob/normal split sampling). "
        "Not needed when --train_scenes is given.",
    )
    parser.add_argument(
        "--normal_scenes",
        type=Path,
        default=None,
        help="JSON list of normal scene NPZ paths (advanced: prob/normal split sampling). "
        "Not needed when --train_scenes is given.",
    )
    parser.add_argument(
        "--val_scenes", type=Path, required=True, help="JSON list of validation scene NPZ paths"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory for experiment results"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip base model evaluation (reuse known baseline numbers)",
    )
    parser.add_argument(
        "--baseline_cache",
        type=Path,
        default=None,
        help="JSON file with precomputed baseline/GT paths per scene. "
        "If not provided, progress ratios are not reported. "
        "Generate with: python -m rlvr.autoresearch.tools.compute_baseline_cache",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=None,
        help="Override the config's train_epochs (otherwise the value from --config is used).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    # Either a single --train_scenes set, or the advanced --prob_scenes/--normal_scenes pair.
    if args.train_scenes is None and (args.prob_scenes is None or args.normal_scenes is None):
        parser.error(
            "provide either --train_scenes <list.json> OR both "
            "--prob_scenes <list.json> and --normal_scenes <list.json>"
        )

    # Set module-level paths from CLI args. In single-set mode the prob/normal globals point at
    # the training set so eval/viz subsampling derives from it (sampling itself is bypassed).
    global BASE_MODEL, PROB_SCENES_PATH, NORMAL_POOL_PATH, VALID_SCENES_PATH, OUTPUT_DIR
    BASE_MODEL = args.model_path
    PROB_SCENES_PATH = args.prob_scenes or args.train_scenes
    NORMAL_POOL_PATH = args.normal_scenes or args.train_scenes
    VALID_SCENES_PATH = args.val_scenes
    OUTPUT_DIR = args.output_dir

    try:
        run(
            args.config,
            args.name,
            skip_baseline=args.skip_baseline,
            baseline_cache_path=args.baseline_cache,
            train_scenes_override=args.train_scenes,
            train_epochs_override=args.train_epochs,
        )
    except Exception as e:
        print(f"\n---")
        print(f"name:             {args.name}")
        print(f"prob_offroad:     0.000")
        print(f"prob_reward:      0.00")
        print(f"val_reward:       0.00")
        print(f"val_collision:    0.00")
        print(f"best_epoch:       0")
        print(f"duration_min:     0.0")
        print(f"checkpoint:       CRASH")
        print(f"error:            {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
