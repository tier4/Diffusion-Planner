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
from datetime import datetime
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data as _load_npz_data_raw
from guidance_gui.generate_samples import generate_samples


def load_npz_data(npz_path, device):
    """Wrapper that adds v4 delay key."""
    data = _load_npz_data_raw(npz_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data
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
    norm_idx = rng_norm.choice(len(normal_pool), size=min(n_normal, len(normal_pool)), replace=False)
    normal_scenes = [normal_pool[i] for i in norm_idx]
    combined = prob_scenes + normal_scenes
    random.Random(seed).shuffle(combined)
    return combined


@torch.no_grad()
def evaluate_checkpoint(model, model_args, scene_paths, reward_config, label=""):
    model.eval()
    totals, offroads, collisions, path_lengths = [], [], 0, []
    rb_crossings, rb_nears = 0, []
    for path in scene_paths:
        try:
            data = load_npz_data(path, DEVICE)
            norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            norm_data = model_args.observation_normalizer(norm_data)
            det_traj = generate_samples(model, model_args, norm_data, noise_scale=0.0,
                                        n_samples=1, composer=None, device=DEVICE)
            det_traj_t = torch.tensor(det_traj, device=DEVICE, dtype=torch.float32)
            reward = compute_reward_batch(det_traj_t, data, reward_config)[0]
            totals.append(reward.total)
            offroads.append(reward.off_road_fraction)
            if reward.collision_step is not None:
                collisions += 1
            if reward.rb_crossing:
                rb_crossings += 1
            rb_nears.append(reward.rb_near_frac)
            pl = np.linalg.norm(np.diff(det_traj[0, :, :2], axis=0), axis=1).sum()
            path_lengths.append(pl)
        except Exception as e:
            print(f"  [eval] skipping {Path(path).name}: {e}")

    n = len(totals)
    if n == 0:
        return {"n_scenes": 0, "reward_mean": 0, "offroad_mean": 0, "collision_rate": 0,
                "path_length_mean": 0, "stopped_count": 0, "rb_crossings": 0, "rb_near_mean": 0}

    pl_arr = np.array(path_lengths)
    result = {
        "n_scenes": n,
        "reward_mean": float(np.mean(totals)),
        "offroad_mean": float(np.mean(offroads)),
        "collision_rate": collisions / n,
        "path_length_mean": float(pl_arr.mean()),
        "stopped_count": int((pl_arr < 1.0).sum()),
        "rb_crossings": rb_crossings,
        "rb_near_mean": float(np.mean(rb_nears)),
    }
    tag = f" [{label}]" if label else ""
    print(f"  Eval{tag}: {n} scenes, reward={result['reward_mean']:+.2f}, "
          f"rb_cross={rb_crossings}/{n}, rb_near={np.mean(rb_nears):.2f}, "
          f"collision={result['collision_rate']:.1%}, "
          f"path={result['path_length_mean']:.1f}m, stopped={result['stopped_count']}")
    return result


def _visualize_trajectories(model, model_args, prob_scenes, reward_config, run_dir, name):
    """Generate trajectory comparison images for the worst offroad scenes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()

    # Find scenes where base model goes offroad
    offroad_indices = []
    for i, path in enumerate(prob_scenes[:100]):
        try:
            data = load_npz_data(path, DEVICE)
            norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            norm = model_args.observation_normalizer(norm)
            traj = generate_samples(model, model_args, norm, 0.0, 1, None, DEVICE)
            traj_t = torch.tensor(traj, device=DEVICE, dtype=torch.float32)
            data2 = load_npz_data(path, DEVICE)
            r = compute_reward_batch(traj_t, data2, reward_config)[0]
            offroad_indices.append((i, r.off_road_fraction, r.total,
                                    np.linalg.norm(np.diff(traj[0,:,:2], axis=0), axis=1).sum()))
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
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
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
        if 'route_lanes' in data:
            rl = data['route_lanes']
            if rl.dim() == 4:
                rl = rl[0]
            for seg_idx in range(rl.shape[0]):
                pts = rl[seg_idx, :, :2].cpu().numpy()
                valid = np.abs(pts).sum(axis=1) > 0.1
                if valid.sum() > 1:
                    ax.plot(pts[valid, 0], pts[valid, 1], 'g-', alpha=0.4, linewidth=1)

        # Plot lane boundaries
        if 'lanes' in data:
            lanes = data['lanes']
            if lanes.dim() == 4:
                lanes = lanes[0]
            for seg_idx in range(min(lanes.shape[0], 60)):
                pts = lanes[seg_idx, :, :2].cpu().numpy()
                valid = np.abs(pts).sum(axis=1) > 0.1
                if valid.sum() > 1:
                    lb = lanes[seg_idx, :, 4:6].cpu().numpy()
                    rb = lanes[seg_idx, :, 6:8].cpu().numpy()
                    ax.plot((pts+lb)[valid, 0], (pts+lb)[valid, 1], 'k-', alpha=0.15, linewidth=0.5)
                    ax.plot((pts+rb)[valid, 0], (pts+rb)[valid, 1], 'k-', alpha=0.15, linewidth=0.5)

        ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2.5)
        ax.plot(0, 0, 'go', markersize=8)
        ax.set_title(f'Scene {scene_i}: offroad={offroad_frac:.0%} path={path_len:.1f}m rew={total:.1f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{name}: trajectory visualization (6 worst offroad scenes)', fontsize=12)
    plt.tight_layout()
    out_path = run_dir / "trajectory_vis.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Visualization saved to {out_path}")


def run(config_path: Path, name: str, skip_baseline: bool = False):
    # Fix seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # Load config
    with open(config_path) as f:
        config_data = json.load(f)

    config_data_raw = dict(config_data)  # save before popping
    n_prob = config_data.pop("n_prob_scenes", 50)
    n_normal = config_data.pop("n_normal_scenes", 100)
    curated_normal_path = config_data.pop("curated_normal_path", None)
    prob_scenes_path = config_data.pop("prob_scenes_path", None)
    seed_lora_path = config_data.pop("seed_lora_path", None)

    grpo_config = GRPOConfig()
    for k, v in config_data.items():
        if hasattr(grpo_config, k):
            setattr(grpo_config, k, v)

    print(f"Experiment: {name}")
    print(f"Config: lr={grpo_config.learning_rate}, kl={grpo_config.kl_coef}, "
          f"rank={grpo_config.lora_rank}, epochs={grpo_config.train_epochs}, "
          f"scenes={n_prob}p+{n_normal}n, N={grpo_config.num_generations}")

    # Load scenes
    prob_100, normal_pool, val_50 = load_scene_lists()
    if prob_scenes_path and Path(prob_scenes_path).exists():
        with open(prob_scenes_path) as f:
            prob_100 = json.load(f)
        print(f"Using custom prob scenes: {len(prob_100)} from {prob_scenes_path}")
    # Subsample prob scenes for eval (keep all for training, eval on 50)
    eval_rng = np.random.default_rng(42)
    prob_eval = [prob_100[i] for i in eval_rng.choice(len(prob_100), size=min(50, len(prob_100)), replace=False)]
    if curated_normal_path and Path(curated_normal_path).exists():
        with open(curated_normal_path) as f:
            curated_pool = json.load(f)
        print(f"Using curated normal pool: {len(curated_pool)} scenes from {curated_normal_path}")
        train_paths = create_training_set(prob_100, curated_pool, n_prob, n_normal)
    else:
        train_paths = create_training_set(prob_100, normal_pool, n_prob, n_normal)
    print(f"Training set: {len(train_paths)} scenes")

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

    # Load model
    policy_model, model_args = load_model(checkpoint_path, DEVICE)

    if grpo_config.use_lora:
        if seed_lora_path and Path(seed_lora_path).exists():
            from preference_optimization.lora_utils import load_lora_checkpoint
            policy_model = load_lora_checkpoint(policy_model, seed_lora_path, is_trainable=True)
            print(f"Seeded from LoRA: {seed_lora_path}")
        else:
            from preference_optimization.lora_utils import apply_lora
            policy_model = apply_lora(policy_model, r=grpo_config.lora_rank,
                                       lora_alpha=grpo_config.lora_alpha,
                                       lora_dropout=grpo_config.lora_dropout)

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=grpo_config.learning_rate)

    # Training reward config uses the configured weights (may boost w_progress
    # to prevent reward hacking where the model learns to stop instead of drive)
    train_reward_config = RewardConfig(
        w_safety=grpo_config.w_safety, w_progress=grpo_config.w_progress,
        w_smooth=grpo_config.w_smooth, w_feasibility=grpo_config.w_feasibility,
        w_centerline=grpo_config.w_centerline,
        near_edge_scale=grpo_config.near_edge_scale,
        wide_edge_scale=grpo_config.wide_edge_scale,
        max_lat_accel=grpo_config.max_lat_accel,
        lat_accel_scale=grpo_config.lat_accel_scale,
        enable_overprogress=grpo_config.enable_overprogress,
        overprogress_margin=grpo_config.overprogress_margin,
        overprogress_penalty=grpo_config.overprogress_penalty,
        stopped_penalty=grpo_config.stopped_penalty,
    )
    # Eval reward config always uses STANDARD weights for cross-experiment comparability
    eval_reward_config = RewardConfig()

    if grpo_config.use_exploration_policy:
        from rlvr.grpo_exploration_trainer import GRPOExplorationTrainer
        trainer = GRPOExplorationTrainer(
            policy_model=policy_model, model_args=model_args,
            dit_optimizer=optimizer, device=DEVICE, run_dir=run_dir,
            config=grpo_config, use_lora=grpo_config.use_lora,
        )
    else:
        trainer = GRPOTrainer(
            policy_model=policy_model, model_args=model_args,
            optimizer=optimizer, device=DEVICE, run_dir=run_dir,
            config=grpo_config, use_lora=grpo_config.use_lora,
        )

    # Evaluate base model (can skip if baseline numbers are already known)
    if skip_baseline:
        print("\nSkipping base model evaluation (--skip_baseline)")
        base_prob = {"reward_mean": 0.0, "rb_crossings": 0, "collision_rate": 0.0}
        base_val = {"reward_mean": 0.0, "rb_crossings": 0, "collision_rate": 0.0}
    else:
        print("\nBase model evaluation:")
        base_prob = evaluate_checkpoint(policy_model, model_args, prob_eval, eval_reward_config, "base-prob")
        base_val = evaluate_checkpoint(policy_model, model_args, val_50, eval_reward_config, "base-val")

    trainer._eval_scene_paths = val_50

    # Training loop
    start_time = time.time()
    best_epoch = 0
    best_prob_reward = base_prob["reward_mean"]
    best_prob_rb_crossings = base_prob["rb_crossings"]
    best_val_reward = base_val["reward_mean"]
    best_val_collision = base_val["collision_rate"]
    best_checkpoint = ""

    args_dict = {"exp_name": name}

    # Reward scheduling: epoch 1 uses softer w_feasibility to let model explore,
    # epoch 2+ uses full strength to lock in on-road behavior.
    reward_schedule = config_data_raw.get("reward_schedule", None)

    for epoch in range(1, grpo_config.train_epochs + 1):
        print(f"\n--- Epoch {epoch}/{grpo_config.train_epochs} ---")

        # Apply reward schedule if provided
        if reward_schedule and str(epoch) in reward_schedule:
            sched = reward_schedule[str(epoch)]
            for k, v in sched.items():
                if hasattr(trainer.reward_config, k):
                    setattr(trainer.reward_config, k, v)
                    print(f"  [schedule] epoch {epoch}: {k}={v}")

        if epoch == 1 and hasattr(trainer, 'save_epoch1_baselines'):
            trainer.save_epoch1_baselines(train_paths)

        metrics = trainer.train_epoch(train_paths, epoch)
        trainer.log_metrics(epoch, metrics)
        trainer.save_checkpoint(epoch, args_dict)

        prob_result = evaluate_checkpoint(policy_model, model_args, prob_eval, eval_reward_config, f"epoch{epoch}-prob")
        val_eval = evaluate_checkpoint(policy_model, model_args, val_50, eval_reward_config, f"epoch{epoch}-val")

        # Track best: highest prob deterministic reward (with val sanity check > -5)
        is_better = (prob_result["reward_mean"] > best_prob_reward
                     and val_eval["reward_mean"] > -5)

        if is_better:
            best_prob_reward = prob_result["reward_mean"]
            best_prob_rb_crossings = prob_result["rb_crossings"]
            best_val_reward = val_eval["reward_mean"]
            best_val_collision = val_eval["collision_rate"]
            best_epoch = epoch
            if grpo_config.use_lora:
                best_checkpoint = str(run_dir / f"lora_epoch_{epoch:03d}")
            else:
                best_checkpoint = str(run_dir / "latest.pth")

        # Early stopping: val collapsed
        if val_eval["reward_mean"] < -20:
            print(f"  Val collapsed ({val_eval['reward_mean']:.1f}), stopping early")
            break

    duration = (time.time() - start_time) / 60

    # Keep ALL checkpoints — don't delete anything. Disk space is cheap,
    # losing the best checkpoint is not.

    # Generate trajectory visualization for the best checkpoint
    if best_checkpoint and Path(best_checkpoint).exists():
        _visualize_trajectories(policy_model, model_args, prob_100, eval_reward_config, run_dir, name)

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
    parser.add_argument("--prob_scenes", type=Path, required=True, help="JSON list of problem scene NPZ paths")
    parser.add_argument("--normal_scenes", type=Path, required=True, help="JSON list of normal scene NPZ paths")
    parser.add_argument("--val_scenes", type=Path, required=True, help="JSON list of validation scene NPZ paths")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for experiment results")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip base model evaluation (reuse known baseline numbers)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    # Set module-level paths from CLI args
    global BASE_MODEL, PROB_SCENES_PATH, NORMAL_POOL_PATH, VALID_SCENES_PATH, OUTPUT_DIR
    BASE_MODEL = args.model_path
    PROB_SCENES_PATH = args.prob_scenes
    NORMAL_POOL_PATH = args.normal_scenes
    VALID_SCENES_PATH = args.val_scenes
    OUTPUT_DIR = args.output_dir

    try:
        run(args.config, args.name, skip_baseline=args.skip_baseline)
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
