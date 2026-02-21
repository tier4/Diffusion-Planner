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
        model: The diffusion model
        data: Input data dictionary
        trajectory: Ground truth trajectory [T, 4]
        model_args: Model arguments
        noise: Pre-generated noise [1, P, T, 4]
        t: Diffusion time [1]
    """
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    # Convert trajectory to tensor and normalize
    gt_trajectory = torch.tensor(trajectory).float().to(DEVICE).unsqueeze(0)  # [1, T, 4]

    # Normalize using ego stats only (StateNormalizer broadcasts to [P, T, 4] which causes shape mismatch)
    ego_mean = model_args.state_normalizer.mean[0].to(DEVICE)
    ego_std = model_args.state_normalizer.std[0].to(DEVICE)
    gt_trajectory_norm = (gt_trajectory - ego_mean) / ego_std  # [1, T, 4]

    # Create full gt with ego + neighbors (neighbors are zeros)
    gt_future = torch.zeros(B, P, future_len, 4, device=DEVICE)
    gt_future[:, 0, :, :] = gt_trajectory_norm  # Only ego has ground truth

    # Current states
    ego_current = data["ego_current_state"][:, :4]
    neighbors_current = (
        data["neighbor_agents_past"][:, : P - 1, -1, :4]
        if P > 1
        else torch.zeros(B, 0, 4, device=DEVICE)
    )
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    # Concatenate current state with future
    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [B, P, 1+T, 4]

    # Add noise to future part only
    mean, std = model.sde.marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))

    if model_args.diffusion_model_type == "flow_matching":
        t_expanded = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t_expanded) * noise + t_expanded * all_gt[:, :, 1:, :]  # [B, P, T, 4]
    else:
        xT = mean + std * noise

    # Concatenate current state with noisy future
    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)  # [B, P, 1+T, 4]

    # Clone data before normalization to avoid inplace modifications
    data_for_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    # Normalize observations (important: this should not modify data inplace)
    data_normalized = model_args.observation_normalizer(data_for_norm)

    # Prepare model inputs - create new dict to avoid modifying original
    merged_inputs = {}
    for k, v in data_normalized.items():
        merged_inputs[k] = v
    merged_inputs["gt_trajectories"] = all_gt
    merged_inputs["sampled_trajectories"] = xT_full
    merged_inputs["diffusion_time"] = t

    # Run model
    _, outputs = model(merged_inputs)
    if "model_output" in outputs:
        outputs = outputs["model_output"]
        outputs = outputs[:, 0, 1:, :]  # [B, T, 4] - ego only
    else:
        outputs = outputs["prediction"]
        outputs = outputs[:, 0, :, :]  # [B, T, 4] - ego only
    model_output = outputs  # [B, T, 4]

    # Compute loss based on model type
    if model_args.diffusion_model_type == "score":
        # Score matching loss
        mse_loss = F.mse_loss(
            (model_output * std[:, 0] + noise[:, 0]),
            torch.zeros_like(model_output),
            reduction="mean",
        )
    elif model_args.diffusion_model_type == "x_start":
        # Direct prediction loss
        mse_loss = F.mse_loss(model_output, gt_trajectory_norm, reduction="mean")
    elif model_args.diffusion_model_type == "flow_matching":
        # Flow matching loss
        target_v = all_gt[:, 0, 1:, :] - noise[:, 0]
        mse_loss = F.mse_loss(model_output, target_v, reduction="mean")
    else:
        raise ValueError(f"Unknown model type: {model_args.diffusion_model_type}")

    return mse_loss


def compute_dpo_loss(
    policy_model: Diffusion_Planner,
    reference_model: Diffusion_Planner,
    batch: list[dict],
    args,
    model_args,
) -> tuple[torch.Tensor, dict]:
    """
    Compute DPO loss for a batch.

    DPO Loss (adapted for MSE loss where lower is better):
        L = -log(sigma(-beta * ((l_w - l_ref_w) - (l_l - l_ref_l))))

    where:
        - l_w: MSE loss for winning trajectory
        - l_l: MSE loss for losing trajectory
        - l_ref_w: reference MSE loss for winning trajectory
        - l_ref_l: reference MSE loss for losing trajectory
        - beta: regularization parameter
        - sigma: sigmoid function
    """
    total_loss = 0.0
    metrics = {
        "accuracy": 0.0,
        "avg_log_ratio": 0.0,
        "avg_reward_margin": 0.0,
    }

    for sample in batch:
        # Determine which is preferred
        data_raw = sample["data"]
        traj_wi = sample["trajectory_w"]
        traj_lo = sample["trajectory_l"]

        # Sample shared diffusion time but separate noise for winner and loser
        B = data_raw["ego_current_state"].shape[0]
        P = 1 + model_args.predicted_neighbor_num
        future_len = model_args.future_len

        # Separate noise for winner and loser
        noise_w = torch.randn(B, P, future_len, 4, device=DEVICE)
        noise_l = torch.randn(B, P, future_len, 4, device=DEVICE)
        eps = 1e-3
        t = torch.rand(B, device=DEVICE) * (1 - eps) + eps

        # Compute losses under policy model (with different noise)
        # Deep copy data to avoid inplace modifications affecting gradient computation
        data_w = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
        data_l = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}

        data_w["delay"] = torch.zeros(B, device=DEVICE)  # Add dummy delay if model expects it
        data_l["delay"] = torch.zeros(B, device=DEVICE)

        l_w = compute_trajectory_loss(policy_model, data_w, traj_wi, model_args, noise_w, t)
        l_l = compute_trajectory_loss(policy_model, data_l, traj_lo, model_args, noise_l, t)

        # Compute losses under reference model (with same noise as policy)
        with torch.no_grad():
            data_ref_w = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()
            }
            data_ref_l = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()
            }

            l_ref_w = compute_trajectory_loss(
                reference_model, data_ref_w, traj_wi, model_args, noise_w.clone(), t
            )
            l_ref_l = compute_trajectory_loss(
                reference_model, data_ref_l, traj_lo, model_args, noise_l.clone(), t
            )

        # Compute DPO loss
        # User's formula: -log(sigma(-beta * ((l_w - l_ref_w) - (l_l - l_ref_l))))
        # Since l(x) is MSE loss (lower is better), we want:
        # - l_w to decrease more than l_l (relative to reference)
        # - (l_w - l_ref_w) < (l_l - l_ref_l)
        # - (l_w - l_ref_w) - (l_l - l_ref_l) < 0
        # So we use -beta * ((l_w - l_ref_w) - (l_l - l_ref_l)) to make it positive when doing well

        loss_diff = (l_w - l_ref_w) - (l_l - l_ref_l)
        loss = -F.logsigmoid(-args.beta * loss_diff)
        total_loss += loss

        # Metrics
        with torch.no_grad():
            reward_margin = -args.beta * loss_diff
            metrics["avg_reward_margin"] += reward_margin.item()
            metrics["avg_log_ratio"] += (-loss_diff).item()
            metrics["accuracy"] += (loss_diff < 0).float().item()

    # Average over batch
    batch_size = len(batch)
    total_loss = total_loss / batch_size
    metrics = {k: v / batch_size for k, v in metrics.items()}

    return total_loss, metrics


@torch.no_grad()
def visualize_validation(
    policy_model: Diffusion_Planner,
    valid_loader: DataLoader,
    model_args,
    save_dir: Path,
    epoch: int,
):
    """
    Visualize validation predictions and save as images.
    """
    policy_model.eval()

    vis_dir = save_dir / "validation_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    for batch in valid_loader:
        for sample in batch:
            # Use preloaded input data
            data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

            B = data["ego_current_state"].shape[0]
            P = 1 + model_args.predicted_neighbor_num
            future_len = model_args.future_len

            # Generate random noise
            data["sampled_trajectories"] = 0.0 * torch.randn(B, P, future_len + 1, 4).to(DEVICE)

            data["delay"] = torch.zeros(B, device=DEVICE)  # Add dummy delay if model expects it

            # Normalize inputs
            data = model_args.observation_normalizer(data)

            # Run model
            _, outputs = policy_model(data)
            prediction = outputs["prediction"][0].cpu().numpy()  # [P+1, T, 4]

            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # Visualize input (map, past trajectories, etc.)
            # Convert data back to unnormalized for visualization
            vis_data = model_args.observation_normalizer.inverse(data)
            for k, v in vis_data.items():
                if isinstance(v, torch.Tensor):
                    vis_data[k] = v.cpu()
            visualize_inputs(vis_data, save_path=None, ax=ax)

            # Plot ego prediction
            ax.plot(
                prediction[0, :, 0],
                prediction[0, :, 1],
                color="orange",
                label="Ego Prediction",
                linewidth=2,
            )

            # Plot neighbor predictions
            for i in range(1, prediction.shape[0]):
                ax.plot(
                    prediction[i, :, 0],
                    prediction[i, :, 1],
                    color="teal",
                    alpha=0.5,
                    linewidth=1,
                )

            ax.legend()
            ax.set_title(f"Epoch {epoch} - Sample {sample_count + 1}")

            # Save figure
            plt.savefig(
                vis_dir / f"sample_{sample_count:03d}_{epoch:04d}.png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()

            sample_count += 1

    print(f"Saved {sample_count} validation visualizations to {vis_dir}")


def train_epoch(
    policy_model: Diffusion_Planner,
    reference_model: Diffusion_Planner,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    args,
    model_args,
) -> dict:
    """Train for one epoch."""
    policy_model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total_reward_margin = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        # Compute DPO loss
        loss, metrics = compute_dpo_loss(policy_model, reference_model, batch, args, model_args)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]
        total_reward_margin += metrics["avg_reward_margin"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
        "reward_margin": total_reward_margin / num_batches,
    }


def main():
    args = parse_args()

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
