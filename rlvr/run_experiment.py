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
from preference_optimization.utils import load_npz_data
from guidance_gui.generate_samples import generate_samples
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_trainer import GRPOTrainer
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
SSD = Path("/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207")
BASE_MODEL = SSD / "xx1-best-model/v3.0/best_model.pth"
PROB_SCENES_PATH = SSD / "path_lists/merged_20260216_20260224/path_list.json"
NORMAL_POOL_PATH = SSD / "xx1_grpo_cleansed_data/path_list.json"
VALID_SCENES_PATH = SSD / "xx1_validation_data/xx1_real_valid/path_list.json"
OUTPUT_DIR = SSD / "auto_research"


def load_scene_lists():
    with open(PROB_SCENES_PATH) as f:
        prob_all = json.load(f)
    prob_100 = prob_all[:100]
    with open(NORMAL_POOL_PATH) as f:
        normal_pool = json.load(f)
    with open(VALID_SCENES_PATH) as f:
        valid_all = json.load(f)
    # Fixed 50 val scenes (seeded)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(valid_all), size=50, replace=False)
    val_50 = [valid_all[i] for i in val_idx]
    return prob_100, normal_pool, val_50


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
    totals, offroads, collisions = [], [], 0
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
        except Exception as e:
            print(f"  [eval] skipping {Path(path).name}: {e}")

    n = len(totals)
    if n == 0:
        return {"n_scenes": 0, "reward_mean": 0, "offroad_mean": 0, "collision_rate": 0}

    result = {
        "n_scenes": n,
        "reward_mean": float(np.mean(totals)),
        "offroad_mean": float(np.mean(offroads)),
        "collision_rate": collisions / n,
    }
    tag = f" [{label}]" if label else ""
    print(f"  Eval{tag}: {n} scenes, reward={result['reward_mean']:+.2f}, "
          f"offroad={result['offroad_mean']:.1%}, collision={result['collision_rate']:.1%}")
    return result


def run(config_path: Path, name: str):
    # Fix seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # Load config
    with open(config_path) as f:
        config_data = json.load(f)

    n_prob = config_data.pop("n_prob_scenes", 50)
    n_normal = config_data.pop("n_normal_scenes", 100)

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
        from preference_optimization.lora_utils import apply_lora
        policy_model = apply_lora(policy_model, r=grpo_config.lora_rank,
                                   lora_alpha=grpo_config.lora_alpha,
                                   lora_dropout=grpo_config.lora_dropout)

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=grpo_config.learning_rate)

    reward_config = RewardConfig(
        w_safety=grpo_config.w_safety, w_progress=grpo_config.w_progress,
        w_smooth=grpo_config.w_smooth, w_feasibility=grpo_config.w_feasibility,
        w_centerline=grpo_config.w_centerline,
    )

    trainer = GRPOTrainer(
        policy_model=policy_model, model_args=model_args,
        optimizer=optimizer, device=DEVICE, run_dir=run_dir,
        config=grpo_config, use_lora=grpo_config.use_lora,
    )

    # Evaluate base model
    print("\nBase model evaluation:")
    base_prob = evaluate_checkpoint(policy_model, model_args, prob_100, reward_config, "base-prob")
    base_val = evaluate_checkpoint(policy_model, model_args, val_50, reward_config, "base-val")

    trainer._eval_scene_paths = val_50

    # Training loop
    start_time = time.time()
    best_prob_offroad = base_prob["offroad_mean"]
    best_epoch = 0
    best_prob_reward = base_prob["reward_mean"]
    best_val_reward = base_val["reward_mean"]
    best_val_collision = base_val["collision_rate"]
    best_checkpoint = ""

    args_dict = {"exp_name": name}

    for epoch in range(1, grpo_config.train_epochs + 1):
        print(f"\n--- Epoch {epoch}/{grpo_config.train_epochs} ---")

        if epoch == 1:
            trainer.save_epoch1_baselines(train_paths)

        metrics = trainer.train_epoch(train_paths, epoch)
        trainer.log_metrics(epoch, metrics)
        trainer.save_checkpoint(epoch, args_dict)

        prob_eval = evaluate_checkpoint(policy_model, model_args, prob_100, reward_config, f"epoch{epoch}-prob")
        val_eval = evaluate_checkpoint(policy_model, model_args, val_50, reward_config, f"epoch{epoch}-val")

        # Track best by prob_reward (but only if val is acceptable)
        if prob_eval["reward_mean"] > best_prob_reward and val_eval["reward_mean"] > 0:
            best_prob_offroad = prob_eval["offroad_mean"]
            best_prob_reward = prob_eval["reward_mean"]
            best_val_reward = val_eval["reward_mean"]
            best_val_collision = val_eval["collision_rate"]
            best_epoch = epoch
            if grpo_config.use_lora:
                best_checkpoint = str(run_dir / f"lora_epoch_{epoch:03d}")
            else:
                best_checkpoint = str(run_dir / "latest.pth")

        # Also track if offroad is better even with worse reward
        if prob_eval["offroad_mean"] < best_prob_offroad and val_eval["reward_mean"] > -5:
            best_prob_offroad = prob_eval["offroad_mean"]

        # Early stopping: val collapsed
        if val_eval["reward_mean"] < -20:
            print(f"  Val collapsed ({val_eval['reward_mean']:.1f}), stopping early")
            break

    duration = (time.time() - start_time) / 60

    # Cleanup: remove per-epoch checkpoints except best
    best_dir_name = Path(best_checkpoint).name if best_checkpoint else ""
    for d in run_dir.glob("lora_epoch_*"):
        if d.name != best_dir_name:
            shutil.rmtree(d, ignore_errors=True)

    # Print final summary (machine-parseable)
    print("\n---")
    print(f"name:             {name}")
    print(f"prob_offroad:     {best_prob_offroad:.4f}")
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
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    try:
        run(args.config, args.name)
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
