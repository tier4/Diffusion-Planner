"""
Direct Preference Optimization (DPO) Training for Diffusion Planner

This program trains the Diffusion Planner model using DPO based on human preference annotations.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.visualize_input import visualize_inputs
from generate_dpo_data_rule_based import load_model, load_npz_data
from torch import optim
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
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--preference_json", type=Path, required=True)
    parser.add_argument("--valid_split", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
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
                - trajectory_w: winning trajectory (ground truth)
                - trajectory_l: losing trajectory (ground truth)
        """
        pref = self.valid_preferences[idx]

        return {
            "npz_path": pref["npz_path"],
            "trajectory_w": pref["trajectory_w"],
            "trajectory_l": pref["trajectory_l"],
        }


def compute_trajectory_loss(
    model: Diffusion_Planner,
    data: dict[str, torch.Tensor],
    trajectory: np.ndarray,
    model_args,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE loss of a trajectory under the model using diffusion training.

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
    device = data["ego_current_state"].device

    # Convert trajectory to tensor and normalize
    gt_trajectory = torch.tensor(trajectory).float().to(device).unsqueeze(0)  # [1, T, 4]

    # Normalize using ego stats only (StateNormalizer broadcasts to [P, T, 4] which causes shape mismatch)
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    gt_trajectory_norm = (gt_trajectory - ego_mean) / ego_std  # [1, T, 4]

    # Create full gt with ego + neighbors (neighbors are zeros)
    gt_future = torch.zeros(B, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = gt_trajectory_norm  # Only ego has ground truth

    # Current states
    ego_current = data["ego_current_state"][:, :4]
    neighbors_current = (
        data["neighbor_agents_past"][:, : P - 1, -1, :4]
        if P > 1
        else torch.zeros(B, 0, 4, device=device)
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
    device = next(policy_model.parameters()).device
    total_loss = 0.0
    metrics = {
        "accuracy": 0.0,
        "avg_log_ratio": 0.0,
        "avg_reward_margin": 0.0,
    }

    for sample in batch:
        # Load input data
        data_raw = load_npz_data(sample["npz_path"], device)

        # Determine which is preferred
        traj_wi = np.array(sample["trajectory_w"])
        traj_lo = np.array(sample["trajectory_l"])

        # Sample shared diffusion time but separate noise for winner and loser
        B = data_raw["ego_current_state"].shape[0]
        P = 1 + model_args.predicted_neighbor_num
        future_len = model_args.future_len

        # Separate noise for winner and loser
        noise_w = torch.randn(B, P, future_len, 4, device=device)
        noise_l = torch.randn(B, P, future_len, 4, device=device)
        eps = 1e-3
        t = torch.rand(B, device=device) * (1 - eps) + eps

        # Compute losses under policy model (with different noise)
        # Deep copy data to avoid inplace modifications affecting gradient computation
        data_w = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
        data_l = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}

        l_w = compute_trajectory_loss(policy_model, data_w, traj_wi, model_args, noise_w, t)
        l_l = compute_trajectory_loss(policy_model, data_l, traj_lo, model_args, noise_l, t)

        # Compute losses under reference model (with same noise as policy)
        with torch.no_grad():
            data_ref_w = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
            data_ref_l = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}

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

    vis_dir = save_dir / "validation_vis" / f"epoch_{epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    for batch in valid_loader:
        for sample in batch:
            # Load input data
            data = load_npz_data(sample["npz_path"], next(policy_model.parameters()).device)

            B = data["ego_current_state"].shape[0]
            P = 1 + model_args.predicted_neighbor_num
            future_len = model_args.future_len

            # Generate random noise
            data["sampled_trajectories"] = 0.0 * torch.randn(B, P, future_len + 1, 4).to(
                data["ego_current_state"].device
            )

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
            npz_stem = Path(sample["npz_path"]).stem
            plt.savefig(
                vis_dir / f"sample_{sample_count:03d}_{npz_stem}.png", dpi=100, bbox_inches="tight"
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
    args = get_args()

    device = torch.device(args.device)

    # Create save directory
    time = datetime.now()
    time = time.strftime("%Y%m%d-%H%M%S")
    save_dir = args.preference_json.parent
    save_path = save_dir / f"{time}_{args.exp_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_path}")

    # Load preference data
    print(f"Loading preferences from {args.preference_json}")
    with open(args.preference_json, "r") as f:
        preferences = json.load(f)

    print(f"Loaded {len(preferences)} preference annotations")

    # Split into train/valid
    num_valid = int(len(preferences) * args.valid_split)
    num_train = len(preferences) - num_valid

    # Shuffle
    random.shuffle(preferences)

    train_preferences = preferences[:num_train]
    valid_preferences = preferences[num_train:]

    preferences = [preferences[0] for _ in range(100)]  # Use only train prefs for dataset

    # Create datasets
    train_dataset = DPODataset(preferences, device)
    valid_dataset = DPODataset(preferences[0:1], device)

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

    # Load reference model (frozen copy of policy model)
    print("Using initial policy model as reference model")
    reference_model, _ = load_model(args.model_path, device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # Save args (convert Path to str for JSON serialization)
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(save_path / "dpo_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    # Training loop
    train_log = []

    for epoch in range(args.train_epochs):
        # Train
        train_metrics = train_epoch(
            policy_model, reference_model, train_loader, optimizer, args, model_args
        )

        # Visualize validation samples
        visualize_validation(policy_model, valid_loader, model_args, save_path, epoch)

        print(
            f"Epoch {epoch + 1}/{args.train_epochs}\n"
            f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
        )

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch + 1,
            "model": policy_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }

        torch.save(checkpoint_data, os.path.join(save_path, "latest.pth"))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint_data, os.path.join(save_path, f"epoch_{epoch + 1:03d}.pth"))

        # Save training log
        train_log.append(
            {
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
        )
        df = pd.DataFrame(train_log)
        df.to_csv(os.path.join(save_path, "dpo_train_log.tsv"), sep="\t", index=False)

    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
