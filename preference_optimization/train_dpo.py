"""Direct Preference Optimization (DPO) Training for Diffusion Planner.

This script trains a trajectory planning model using human preferences via DPO.

Usage:
    cd /path/to/Diffusion-Planner
    python3 -m preference_optimization.train_dpo [args]

Or:
    cd /path/to/Diffusion-Planner/preference_optimization
    python3 train_dpo.py [args]
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure parent directory is in path for diffusion_planner imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import torch
from torch import optim

from preference_optimization.annotation_gui import collect_preferences
from preference_optimization.annotation_ros_node import AnnotationRosServer
from preference_optimization.model_utils import load_model
from preference_optimization.preference_collection import generate_rule_based_preferences
from preference_optimization.trainer import DPOTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Diffusion Planner with Direct Preference Optimization"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to initial model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--train_npz_list",
        type=Path,
        required=True,
        help="Path to JSON file containing training NPZ paths",
    )
    parser.add_argument(
        "--valid_npz_list",
        type=Path,
        required=True,
        help="Path to JSON file containing validation NPZ paths",
    )

    # Training configuration
    parser.add_argument(
        "--exp_name",
        type=str,
        default="dpo_experiment",
        help="Experiment name for organizing outputs",
    )
    parser.add_argument(
        "--preference_mode",
        type=str,
        choices=["rule", "gui", "lichtblick"],
        default="rule",
        help="Preference collection mode: 'rule' for automatic, 'gui' for manual annotation",
    )
    parser.add_argument("--lichtblick_host", type=str, default="127.0.0.1", help="Lichtblick websocket host")
    parser.add_argument("--lichtblick_port", type=int, default=8765, help="Lichtblick websocket port")
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO regularization parameter",
    )

    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Apply LoRA adapters to DiT attention layers. Standard checkpoints load "
             "automatically without any manual migration step.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank r. Lower rank reduces capacity but mitigates catastrophic forgetting.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling. Effective weight delta scale = alpha / r.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability on LoRA activations.",
    )

    return parser.parse_args()


def _find_lora_dir(search_dir: Path) -> Path | None:
    """Return the most recent LoRA adapter directory inside search_dir, or None.

    Checks in order: lora_latest symlink, then highest-numbered lora_epoch_NNN/.
    """
    lora_latest = search_dir / "lora_latest"
    if lora_latest.exists() and (lora_latest / "adapter_config.json").exists():
        return lora_latest.resolve()
    for d in reversed(sorted(search_dir.glob("lora_epoch_*"))):
        if (d / "adapter_config.json").exists():
            return d
    return None


def setup_experiment(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    """Setup experiment directory and copy initial model.

    If the model directory already contains LoRA adapter weights (from a previous
    DPO run) and --use_lora is set, the adapter is copied into the new run dir so
    that training resumes from the previously trained adapter rather than starting
    from a fresh (zero-delta) initialisation.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (run_dir, checkpoint_path, seed_lora_dir)
        seed_lora_dir is the copied adapter directory, or None if not applicable.

    Raises:
        FileNotFoundError: If model or args.json not found
    """
    # Validate paths
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    args_json_path = args.model_path.parent / "args.json"
    if not args_json_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")

    # Create experiment directory
    save_dir = args.train_npz_list.parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = save_dir / f"{timestamp}_{args.exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy model and config to experiment directory
    checkpoint_path = run_dir / "latest.pth"
    shutil.copy2(args.model_path, checkpoint_path)
    shutil.copy2(args_json_path, run_dir / "args.json")

    # If a LoRA adapter exists next to model_path and --use_lora is requested,
    # copy it into the new run dir so we can resume from the trained adapter.
    seed_lora_dir: Path | None = None
    if getattr(args, "use_lora", False):
        src_lora = _find_lora_dir(args.model_path.parent)
        if src_lora is not None:
            seed_lora_dir = run_dir / "lora_seed"
            shutil.copytree(src_lora, seed_lora_dir)
            print(f"Found previous LoRA adapter: {src_lora}")
            print(f"  → copied to {seed_lora_dir} (will resume from these weights)")

    # Save training arguments
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(run_dir / "dpo_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Experiment directory: {run_dir}")

    return run_dir, checkpoint_path, seed_lora_dir


def collect_epoch_preferences(
    args: argparse.Namespace,
    policy_model,
    model_args,
    train_npz_paths: list[str],
    ros_node: AnnotationRosServer | None = None,
    drift_info: str = "",
) -> list[dict]:
    """Collect preferences for current epoch.

    Args:
        args: Parsed arguments
        policy_model: Policy model for trajectory generation
        model_args: Model configuration
        train_npz_paths: List of training data paths
        drift_info: Optional drift summary string to display in the annotation GUI.

    Returns:
        List of preference annotations
    """
    if args.preference_mode == "gui":
        print("Launching GUI for preference annotation...")
        preferences = collect_preferences(
            policy_model,
            model_args,
            args.train_npz_list,
            target_count=len(train_npz_paths),
            drift_info=drift_info,
        )
    elif args.preference_mode == "lichtblick":
        if ros_node is None:
            raise RuntimeError("Lichtblick mode requires a running AnnotationRosServer.")
        print("Launching Lichtblick websocket annotation...")
        ros_node.reset_annotation_round(target_count=len(train_npz_paths))
        ros_node.update_training_status(
            phase="annotation",
            message="Annotate trajectories. If both look poor, click Regenerate first, then select winner and click Launch Training.",
            epoch=0,
            total_epochs=args.train_epochs,
        )
        preferences = ros_node.wait_for_annotation_complete()
    else:
        print("Generating rule-based preferences...")
        preferences = generate_rule_based_preferences(
            policy_model,
            model_args,
            args.train_npz_list,
            DEVICE,
        )

    return preferences


def main():
    """Main training loop."""
    # Parse arguments
    args = parse_args()

    # Setup experiment
    run_dir, checkpoint_path, seed_lora_dir = setup_experiment(args)

    # Load model
    policy_model, model_args = load_model(checkpoint_path, DEVICE)

    # Apply LoRA before creating the optimizer so the optimizer captures only the
    # trainable LoRA parameters (A and B matrices), not the frozen base weights.
    if args.use_lora:
        if seed_lora_dir is not None:
            # Resume from a previously trained LoRA adapter.  PeftModel.from_pretrained
            # reads rank/alpha/dropout from the saved adapter_config.json so the CLI
            # flags --lora_rank/--lora_alpha/--lora_dropout are intentionally ignored
            # in this path to avoid silently changing the adapter architecture.
            from preference_optimization.lora_utils import load_lora_checkpoint
            policy_model = load_lora_checkpoint(policy_model, str(seed_lora_dir), is_trainable=True)
            policy_model.print_trainable_parameters()
            print(f"Resumed LoRA adapter from {seed_lora_dir}")
        else:
            from preference_optimization.lora_utils import apply_lora
            policy_model = apply_lora(
                policy_model,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

    # Create optimizer over trainable parameters only
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # When resuming a LoRA run, restore AdamW moments from the saved optimizer state
    # so training continues with the same first/second moment estimates rather than
    # resetting to zero (which would transiently change the effective learning rate).
    if args.use_lora and seed_lora_dir is not None:
        opt_path = seed_lora_dir / "optimizer.pth"
        if opt_path.exists():
            saved = torch.load(opt_path, map_location=DEVICE)
            optimizer.load_state_dict(saved["optimizer"])
            print(f"Resumed optimizer state from {opt_path} (epoch {saved['epoch']})")

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        model_args=model_args,
        optimizer=optimizer,
        device=DEVICE,
        run_dir=run_dir,
        batch_size=args.batch_size,
        beta=args.beta,
        use_lora=args.use_lora,
    )

    ros_node: AnnotationRosServer | None = None
    if args.preference_mode == "lichtblick":
        ros_node = AnnotationRosServer(
            model_path=checkpoint_path,
            npz_list=args.train_npz_list,
            target_count=len(train_npz_paths) if "train_npz_paths" in locals() else None,
            device=str(DEVICE),
        )
        ros_node.start_background()

    # Load validation data
    with open(args.valid_npz_list, "r") as f:
        valid_npz_paths = json.load(f)
    valid_loader = trainer.create_validation_loader(valid_npz_paths)

    # Load training data paths
    with open(args.train_npz_list, "r") as f:
        train_npz_paths = json.load(f)

    print(f"Training data: {len(train_npz_paths)} samples")
    print(f"Validation data: {len(valid_npz_paths)} samples")

    # Initial visualization
    print("Generating initial validation visualizations...")
    trainer.visualize_epoch(valid_loader, epoch=0)

    # Training loop
    print(f"\nStarting training for {args.train_epochs} epochs...")
    print("=" * 60)

    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    # Drift summary from trainer; populated after epoch 1 baselines are saved.
    drift_info: str = ""

    for epoch in range(1, args.train_epochs + 1):
        print(f"\nEpoch {epoch}/{args.train_epochs}")
        print("-" * 60)

        if drift_info:
            print(f"  {drift_info}")

        # Collect preferences
        preferences = collect_epoch_preferences(
            args, policy_model, model_args, train_npz_paths, ros_node=ros_node,
            drift_info=drift_info,
        )

        if not preferences:
            print("No preferences collected. Skipping this epoch.")
            continue

        # Snapshot the pre-training deterministic outputs for the annotated samples.
        # Must be done BEFORE train_epoch so the baseline captures the current model
        # state; compute_trajectory_drift (called after training) then measures the
        # actual weight change produced by this epoch.
        if epoch == 1:
            print("Saving pre-training deterministic baselines...")
            trainer.save_epoch1_baselines(preferences)

        # Train on preferences
        if ros_node is not None:
            ros_node.update_training_status(
                phase="training",
                message=f"Training epoch {epoch}/{args.train_epochs}...",
                epoch=epoch,
                total_epochs=args.train_epochs,
            )

        def _progress_cb(progress: dict[str, float]) -> None:
            if ros_node is None:
                return
            ros_node.update_training_status(
                phase="training",
                message=f"Training epoch {epoch}/{args.train_epochs}",
                epoch=epoch,
                total_epochs=args.train_epochs,
                batch=int(progress["batch"]),
                total_batches=int(progress["total_batches"]),
                metrics={
                    "loss": float(progress["loss"]),
                    "accuracy": float(progress["accuracy"]),
                    "reward_margin": float(progress["reward_margin"]),
                },
            )

        metrics = trainer.train_epoch(preferences, epoch, progress_callback=_progress_cb)

        # Compute drift: compare current (post-training) model to the pre-training
        # baselines saved above.  Displayed in the next annotation round.
        drift_info = trainer.compute_trajectory_drift()

        # Visualize
        trainer.visualize_epoch(valid_loader, epoch)

        # Log and save
        trainer.log_metrics(epoch, metrics)
        trainer.save_checkpoint(epoch, args_dict)

        if ros_node is not None:
            ros_node.update_training_status(
                phase="training",
                message=f"Epoch {epoch} complete. Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}",
                epoch=epoch,
                total_epochs=args.train_epochs,
                metrics=metrics,
            )
            if epoch < args.train_epochs:
                ros_node.update_training_status(
                    phase="annotation",
                    message=(
                        f"Epoch {epoch} training finished. Next annotation round is ready. "
                        "Please click Regenerate when needed, then select winner and Launch Training."
                    ),
                    epoch=epoch,
                    total_epochs=args.train_epochs,
                    metrics=metrics,
                )

        print("-" * 60)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)

    if ros_node is not None:
        ros_node.update_training_status(
            phase="complete",
            message=f"Training complete. Results: {run_dir}",
            epoch=args.train_epochs,
            total_epochs=args.train_epochs,
        )


if __name__ == "__main__":
    main()
