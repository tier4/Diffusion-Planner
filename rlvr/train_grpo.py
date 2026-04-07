"""GRPO Training for Diffusion Planner.

Supports two modes via JSON config:
- On-policy (M=1): Single gradient step per rollout, simplest and most stable.
- Multi-epoch (M>1): Reuse rollouts with PPO-clipped importance sampling.

Usage:
    source .venv/bin/activate

    # Automatic with on-policy config:
    python rlvr/train_grpo.py \\
        --model_path /path/to/model.pth \\
        --train_npz_list /path/to/train.json \\
        --valid_npz_list /path/to/valid.json \\
        --config rlvr/configs/grpo_onpolicy.json

    # Automatic with multi-epoch config:
    python rlvr/train_grpo.py \\
        --model_path /path/to/model.pth \\
        --train_npz_list /path/to/train.json \\
        --valid_npz_list /path/to/valid.json \\
        --config rlvr/configs/grpo_multi_epoch.json

    # GUI mode:
    python rlvr/train_grpo.py \\
        --model_path /path/to/model.pth \\
        --train_npz_list /path/to/train.json \\
        --valid_npz_list /path/to/valid.json \\
        --config rlvr/configs/grpo_onpolicy.json \\
        --mode gui
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch import optim

from preference_optimization.model_utils import load_model
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_trainer import GRPOTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO Training for Diffusion Planner"
    )

    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--train_npz_list", type=Path, required=True,
                        help="JSON list of training .npz paths")
    parser.add_argument("--valid_npz_list", type=Path, required=True,
                        help="JSON list of validation .npz paths")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to GRPO config JSON (default: on-policy M=1)")
    parser.add_argument("--exp_name", type=str, default="grpo_experiment")
    parser.add_argument("--mode", type=str, choices=["rule", "gui"], default="rule")
    parser.add_argument("--port", type=int, default=7863)

    return parser.parse_args()


def _find_lora_dir(search_dir: Path) -> Path | None:
    lora_latest = search_dir / "lora_latest"
    if lora_latest.exists() and (lora_latest / "adapter_config.json").exists():
        return lora_latest.resolve()
    for d in reversed(sorted(search_dir.glob("lora_epoch_*"))):
        if (d / "adapter_config.json").exists():
            return d
    return None


def setup_experiment(args: argparse.Namespace, config: GRPOConfig) -> tuple[Path, Path, Path | None]:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args_json_path = args.model_path.parent / "args.json"
    if not args_json_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")

    save_dir = args.train_npz_list.parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = save_dir / f"{timestamp}_{args.exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "latest.pth"
    shutil.copy2(args.model_path, checkpoint_path)
    shutil.copy2(args_json_path, run_dir / "args.json")
    config.to_json(run_dir / "grpo_config.json")

    seed_lora_dir: Path | None = None
    if config.use_lora:
        src_lora = _find_lora_dir(args.model_path.parent)
        if src_lora is not None:
            seed_lora_dir = run_dir / "lora_seed"
            shutil.copytree(src_lora, seed_lora_dir)
            print(f"Found previous LoRA adapter: {src_lora}")
            print(f"  -> copied to {seed_lora_dir}")

    print(f"Experiment directory: {run_dir}")
    return run_dir, checkpoint_path, seed_lora_dir


def main():
    args = parse_args()

    # Load config
    if args.config and args.config.exists():
        config = GRPOConfig.from_json(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = GRPOConfig()
        print("Using default GRPOConfig (on-policy: M=1)")

    mode_str = "multi-epoch" if config.uses_importance_sampling else "on-policy"
    print(f"Mode: {mode_str} (N={config.num_generations}, M={config.inner_epochs}, "
          f"clip={config.ppo_clip_epsilon}, kl={config.kl_coef})")

    run_dir, checkpoint_path, seed_lora_dir = setup_experiment(args, config)

    # Load model
    policy_model, model_args = load_model(checkpoint_path, DEVICE)

    # Apply LoRA
    if config.use_lora:
        if seed_lora_dir is not None:
            from preference_optimization.lora_utils import load_lora_checkpoint
            policy_model = load_lora_checkpoint(
                policy_model, str(seed_lora_dir), is_trainable=True,
            )
            policy_model.print_trainable_parameters()
        else:
            from preference_optimization.lora_utils import apply_lora
            policy_model = apply_lora(
                policy_model,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

    # Optimizer
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate)

    if config.use_lora and seed_lora_dir is not None:
        opt_path = seed_lora_dir / "optimizer.pth"
        if opt_path.exists():
            saved = torch.load(opt_path, map_location=DEVICE)
            optimizer.load_state_dict(saved["optimizer"])
            print(f"Restored optimizer from {opt_path} (epoch {saved['epoch']})")

    # Create trainer
    trainer = GRPOTrainer(
        policy_model=policy_model,
        model_args=model_args,
        optimizer=optimizer,
        device=DEVICE,
        run_dir=run_dir,
        config=config,
        use_lora=config.use_lora,
    )

    # Load data
    with open(args.train_npz_list) as f:
        train_npz_paths = json.load(f)
    with open(args.valid_npz_list) as f:
        valid_npz_paths = json.load(f)

    print(f"Training: {len(train_npz_paths)} scenes")
    print(f"Validation: {len(valid_npz_paths)} scenes")

    if args.mode == "gui":
        _run_gui_mode(args, trainer, train_npz_paths, config)
    else:
        _run_rule_mode(args, trainer, train_npz_paths, valid_npz_paths)


def _run_rule_mode(
    args: argparse.Namespace,
    trainer: GRPOTrainer,
    train_npz_paths: list[str],
    valid_npz_paths: list[str],
):
    args_dict = {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()}
    drift_info = ""

    # Fix evaluation scenes from validation set (sampled once, reused every epoch)
    trainer.setup_eval_scenes(valid_npz_paths, n_scenes=50)

    # Evaluate base model before any training (epoch 0)
    print("\nEvaluating base model (epoch 0)...")
    trainer.evaluate_rewards(epoch=0)

    print(f"\nStarting GRPO training for {trainer.config.train_epochs} epochs...")
    print("=" * 60)

    total_epochs = trainer.config.train_epochs
    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        print("-" * 60)

        if drift_info:
            print(f"  {drift_info}")

        if epoch == 1:
            trainer.save_epoch1_baselines(train_npz_paths)

        metrics = trainer.train_epoch(train_npz_paths, epoch)

        drift_info = trainer.compute_trajectory_drift()
        trainer.log_metrics(epoch, metrics)
        trainer.save_checkpoint(epoch, args_dict)
        trainer.evaluate_rewards(epoch)

        print("-" * 60)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {trainer.run_dir}")
    print(f"Eval log: {trainer.run_dir / 'grpo_eval_log.tsv'}")


def _run_gui_mode(
    args: argparse.Namespace,
    trainer: GRPOTrainer,
    train_npz_paths: list[str],
    config: GRPOConfig,
):
    from rlvr.trajectory_ranker_gui import (
        TrajectoryRanker,
        build_interface,
        ensure_prototypes,
        _DEFAULT_PROTOTYPES_PATH,
    )

    prototypes_path = config.prototypes_path or _DEFAULT_PROTOTYPES_PATH
    prototypes_path = ensure_prototypes(
        npz_list_path=str(args.train_npz_list),
        prototypes_path=prototypes_path,
        force=False,
    )

    ranker = TrajectoryRanker(
        policy_model=trainer.policy_model,
        model_args=trainer.model_args,
        npz_paths=train_npz_paths,
        npz_list_path=str(args.train_npz_list),
        prototypes_path=prototypes_path,
    )
    ranker.sampler_config = trainer.sampler_config
    ranker.reward_config = trainer.reward_config

    demo = build_interface(ranker, trainer=trainer)
    demo.launch(server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
