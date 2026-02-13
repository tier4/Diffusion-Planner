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

    return parser.parse_args()


def setup_experiment(args: argparse.Namespace) -> tuple[Path, Path]:
    """Setup experiment directory and copy initial model.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (run_dir, checkpoint_path)

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

    # Save training arguments
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(run_dir / "dpo_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Experiment directory: {run_dir}")

    return run_dir, checkpoint_path


def collect_epoch_preferences(
    args: argparse.Namespace,
    policy_model,
    model_args,
    train_npz_paths: list[str],
    ros_node: AnnotationRosServer | None = None,
) -> list[dict]:
    """Collect preferences for current epoch.

    Args:
        args: Parsed arguments
        policy_model: Policy model for trajectory generation
        model_args: Model configuration
        train_npz_paths: List of training data paths

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
    run_dir, checkpoint_path = setup_experiment(args)

    # Load model
    policy_model, model_args = load_model(checkpoint_path, DEVICE)

    # Create optimizer
    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        model_args=model_args,
        optimizer=optimizer,
        device=DEVICE,
        run_dir=run_dir,
        batch_size=args.batch_size,
        beta=args.beta,
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

    for epoch in range(1, args.train_epochs + 1):
        print(f"\nEpoch {epoch}/{args.train_epochs}")
        print("-" * 60)

        # Collect preferences
        preferences = collect_epoch_preferences(
            args, policy_model, model_args, train_npz_paths, ros_node=ros_node
        )

        if not preferences:
            print("No preferences collected. Skipping this epoch.")
            continue

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
