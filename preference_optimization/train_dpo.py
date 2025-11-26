"""
Direct Preference Optimization (DPO) Training for Diffusion Planner

This program trains the Diffusion Planner model using DPO based on human preference annotations.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils import ddp
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.train_utils import set_seed
from generate_dpo_data_rule_based import load_model, load_npz_data
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("."))
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--reference_model_path", type=Path, default=None)
    parser.add_argument("--preference_json", type=Path, required=True)
    parser.add_argument("--valid_split", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--early_stop_tolerance", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", type=boolean, default=True)
    parser.add_argument("--use_wandb", type=boolean, default=False)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", type=boolean, default=False)
    parser.add_argument("--port", type=str, default="22324")
    return parser.parse_args()


class DPODataset(Dataset):
    """Dataset for DPO training."""

    def __init__(self, preferences: list[dict], device: str = "cuda"):
        """
        Args:
            preferences: List of preference dictionaries from annotation
            device: Device to load data on
        """
        self.preferences = preferences
        self.device = device

        # Filter out equal preferences (if any)
        self.valid_preferences = [p for p in preferences if p.get("score_w") != p.get("score_l")]

        print(
            f"Loaded {len(self.valid_preferences)} preferences (filtered from {len(preferences)})"
        )

    def __len__(self):
        return len(self.valid_preferences)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - npz_path: path to input data
                - trajectory_1_info: dict with trajectory and intermediate steps
                - trajectory_2_info: dict with trajectory and intermediate steps
                - preference: 0 for trajectory_1, 1 for trajectory_2
        """
        pref = self.valid_preferences[idx]

        return {
            "npz_path": pref["npz_path"],
            "trajectory_w": pref["trajectory_w"],
            "trajectory_l": pref["trajectory_l"],
        }


@torch.no_grad()
def compute_trajectory_loss(
    model: Diffusion_Planner,
    data: dict[str, torch.Tensor],
    trajectory: np.ndarray,
    model_args,
) -> torch.Tensor:
    """
    Compute MSE loss of a trajectory under the model.
    """
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Generate random noise
    data["sampled_trajectories"] = 0.5 * torch.randn(B, P, future_len + 1, 4).to(
        data["ego_current_state"].device
    )

    # Normalize inputs
    data = model_args.observation_normalizer(data)

    # Run model
    _, outputs = model(data)
    prediction = outputs["prediction"][:, 0]  # [B, T, 4] - ego only

    # Target trajectory
    target = torch.tensor(trajectory).float().to(prediction.device).unsqueeze(0)  # [1, T, 4]

    # Compute MSE loss
    mse_loss = F.mse_loss(prediction, target, reduction="mean")

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

    DPO Loss:
        L = -log(sigma(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_pi_ref(y_w|x) + log_pi_ref(y_l|x))))

    where:
        - y_w: preferred (winning) trajectory
        - y_l: dis-preferred (losing) trajectory
        - pi: policy model
        - pi_ref: reference model
        - beta: regularization parameter
        - sigma: sigmoid function
    """
    device = next(policy_model.parameters()).device
    total_loss = 0.0
    metrics = {
        "accuracy": 0.0,
        "avg_log_ratio": 0.0,
        "avg_reward_margin": 0.0,
    }

    for sample in batch:
        # Load input data
        data = load_npz_data(sample["npz_path"], device)

        # Determine which is preferred
        traj_wi = sample["trajectory_w"]
        traj_lo = sample["trajectory_l"]

        # Compute losses under policy model
        l_w = compute_trajectory_loss(policy_model, data, traj_wi, model_args)
        l_l = compute_trajectory_loss(policy_model, data, traj_lo, model_args)

        # Compute losses under reference model
        with torch.no_grad():
            l_ref_w = compute_trajectory_loss(reference_model, data, traj_wi, model_args)
            l_ref_l = compute_trajectory_loss(reference_model, data, traj_lo, model_args)

        # Compute DPO loss
        # Loss = -log(sigmoid( -beta * ( (l_w - l_ref_w) - (l_l - l_ref_l) ) ))
        # Note: The user specified formula: log sigma - coeff( l(x_w) - l_ref(x_w) - (l(x_l) - l_ref(x_l)) )
        # This corresponds to maximizing the margin. Since we minimize loss, we take negative.
        # Also, l(x) is loss (MSE), so lower is better.
        # Original DPO maximizes log_prob (higher is better).
        # Here we want to minimize l_w relative to l_l.
        # So the "reward" is -l(x).
        # reward_w = -l_w, reward_l = -l_l
        # log_ratio = (reward_w - reward_ref_w) - (reward_l - reward_ref_l)
        #           = (-l_w - (-l_ref_w)) - (-l_l - (-l_ref_l))
        #           = -(l_w - l_ref_w) + (l_l - l_ref_l)
        #           = - ( (l_w - l_ref_w) - (l_l - l_ref_l) )

        diff_w = l_w - l_ref_w
        diff_l = l_l - l_ref_l

        # We want diff_w to be smaller than diff_l (i.e. policy improves on winner more than loser)
        # The term inside sigmoid should be positive when we are doing well.
        # - (diff_w - diff_l) should be positive => diff_l > diff_w

        loss_diff = -(diff_w - diff_l)

        loss = -F.logsigmoid(args.beta * loss_diff)
        total_loss += loss

        # Metrics
        with torch.no_grad():
            reward_margin = args.beta * loss_diff
            metrics["avg_reward_margin"] += reward_margin.item()
            metrics["avg_log_ratio"] += loss_diff.item()
            metrics["accuracy"] += (loss_diff > 0).float().item()

    # Average over batch
    batch_size = len(batch)
    total_loss = total_loss / batch_size
    metrics = {k: v / batch_size for k, v in metrics.items()}

    return total_loss, metrics


@torch.no_grad()
def validate_model(
    policy_model: Diffusion_Planner,
    reference_model: Diffusion_Planner,
    valid_loader: DataLoader,
    args,
    model_args,
) -> dict:
    """Validate the model on the validation set."""
    policy_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_reward_margin = 0.0
    num_batches = 0

    for batch in tqdm(valid_loader, desc="Validation"):
        loss, metrics = compute_dpo_loss(policy_model, reference_model, batch, args, model_args)

        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]
        total_reward_margin += metrics["avg_reward_margin"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
        "reward_margin": total_reward_margin / num_batches,
    }


def train_epoch(
    policy_model: Diffusion_Planner,
    reference_model: Diffusion_Planner,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    args,
    model_args,
    model_ema=None,
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

        # Update EMA
        if model_ema is not None:
            model_ema.update(policy_model)

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
    args = get_args()

    # Setup DDP
    global_rank, rank, _ = ddp.ddp_setup_universal(args.ddp, args)
    print(f"{global_rank=}, {rank=}")

    device = torch.device(f"cuda:{rank}" if args.device == "cuda" else args.device)

    # Set seed
    set_seed(args.seed + global_rank)

    # Create save directory
    if global_rank == 0:
        from datetime import datetime

        time = datetime.now()
        time = time.strftime("%Y%m%d-%H%M%S")
        save_path = args.save_dir / f"{time}_{args.exp_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_path}")
    else:
        save_path = None

    # Load preference data
    print(f"Loading preferences from {args.preference_json}")
    with open(args.preference_json, "r") as f:
        preferences = json.load(f)

    print(f"Loaded {len(preferences)} preference annotations")

    # Split into train/valid
    num_valid = int(len(preferences) * args.valid_split)
    num_train = len(preferences) - num_valid

    # Shuffle
    random.seed(args.seed)
    random.shuffle(preferences)

    train_preferences = preferences[:num_train]
    valid_preferences = preferences[num_train:]

    # Create datasets
    train_dataset = DPODataset(train_preferences, device)
    valid_dataset = DPODataset(valid_preferences, device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simplicity with custom data loading
        collate_fn=lambda x: x,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    # Load policy model
    policy_model, model_args = load_model(args.model_path, device)

    # Load reference model (frozen)
    if args.reference_model_path is not None:
        reference_model, _ = load_model(args.reference_model_path, device)
    else:
        print("Using initial policy model as reference model")
        reference_model = Diffusion_Planner(model_args)
        reference_model.load_state_dict(policy_model.state_dict())

    reference_model.to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Setup DDP
    if args.ddp:
        policy_model = DDP(policy_model, device_ids=[rank])

    # EMA
    model_ema = None
    if args.use_ema:
        model_ema = ModelEma(policy_model, decay=0.999, device=device)

    # Optimizer
    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)

    # Logging
    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        wandb.init(
            project="Diffusion-Planner-DPO",
            name=args.exp_name,
            notes=args.notes,
            dir=str(save_path),
        )
        wandb.config.update(vars(args))

        # Save args (convert Path to str for JSON serialization)
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with open(save_path / "dpo_args.json", "w") as f:
            json.dump(args_dict, f, indent=4)

    if args.ddp:
        torch.distributed.barrier()

    # Training loop
    best_accuracy = 0.0
    no_improvement_count = 0
    train_log = []

    for epoch in range(args.train_epochs):
        if args.ddp:
            torch.distributed.barrier()

        # Train
        train_metrics = train_epoch(
            policy_model, reference_model, train_loader, optimizer, args, model_args, model_ema
        )

        # Validate
        if global_rank == 0:
            # Use EMA model for validation if available
            eval_model = model_ema.ema if model_ema is not None else policy_model
            valid_metrics = validate_model(
                eval_model, reference_model, valid_loader, args, model_args
            )

            print(
                f"Epoch {epoch + 1}/{args.train_epochs}\n"
                f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}\n"
                f"  Valid Loss: {valid_metrics['loss']:.4f}, Acc: {valid_metrics['accuracy']:.4f}"
            )

            # Log to wandb
            wandb.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/reward_margin": train_metrics["reward_margin"],
                    "valid/loss": valid_metrics["loss"],
                    "valid/accuracy": valid_metrics["accuracy"],
                    "valid/reward_margin": valid_metrics["reward_margin"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )

            # Save checkpoint
            checkpoint_data = {
                "epoch": epoch + 1,
                "model": policy_model.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict() if model_ema else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }

            torch.save(checkpoint_data, os.path.join(save_path, "latest.pth"))

            # Save best model
            if valid_metrics["accuracy"] > best_accuracy:
                best_accuracy = valid_metrics["accuracy"]
                torch.save(checkpoint_data, os.path.join(save_path, "best_model.pth"))
                print(f"  New best accuracy: {best_accuracy:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= args.early_stop_tolerance:
                print(f"No improvement for {args.early_stop_tolerance} epochs. Stopping.")
                break

            # Save training log
            train_log.append(
                {
                    "epoch": epoch + 1,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"valid_{k}": v for k, v in valid_metrics.items()},
                }
            )
            df = pd.DataFrame(train_log)
            df.to_csv(os.path.join(save_path, "dpo_train_log.tsv"), sep="\t", index=False)

        scheduler.step()

    if args.ddp:
        torch.distributed.destroy_process_group()

    if global_rank == 0:
        print(f"\nTraining complete! Best validation accuracy: {best_accuracy:.4f}")
        wandb.finish()


if __name__ == "__main__":
    main()
