"""
Direct Preference Optimization (DPO) Training for Diffusion Planner

This program trains the Diffusion Planner model using DPO based on human preference annotations.
"""

import argparse
import copy
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from annotation_gui import collect_preferences_gui
from annotation_gui_gradio import collect_preferences_gui_gradio
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.visualize_input import visualize_inputs
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import calculate_path_length, generate_trajectory_pair, load_npz_data

DEVICE = torch.device("cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--train_npz_list", type=Path, required=True)
    parser.add_argument("--valid_npz_list", type=Path, required=True)
    parser.add_argument(
        "--preference_mode",
        type=str,
        choices=["rule", "gui"],
        default="rule",
        help="Use rule-based scoring or GUI annotation to collect preferences.",
    )
    parser.add_argument(
        "--ui_framework",
        type=str,
        choices=["tkinter", "gradio"],
        default="tkinter",
        help="UI framework for preference annotation.",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    return parser.parse_args()


def load_model(model_path: Path) -> tuple[Diffusion_Planner, Config]:
    """Load Diffusion Planner model and its configuration."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model_dir = model_path.parent
    args_path = model_dir / "args.json"
    model_args = Config(str(args_path), guidance_fn=None)

    model = Diffusion_Planner(model_args)

    if "model" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state_dict, strict=False)
    elif "ema_state_dict" in checkpoint:
        print("Loading EMA weights")
        model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    return model, model_args


def generate_rule_based_preferences(
    policy_model: Diffusion_Planner, model_args, npz_list: Path
) -> list[dict]:
    """Generate preference annotations using the provided policy model."""
    seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    print(f"Random seed: {seed}")

    with open(npz_list, "r") as f:
        npz_paths = json.load(f)

    preferences: list[dict] = []

    print(f"Total NPZ files to annotate: {len(npz_paths)}")

    was_training = policy_model.training
    policy_model.eval()

    print("Starting rule-based annotation...")
    for npz_path in tqdm(npz_paths):
        data = load_npz_data(npz_path, DEVICE)
        traj_1, traj_2 = generate_trajectory_pair(policy_model, model_args, data, device=DEVICE)
        score_1 = calculate_path_length(traj_1)
        score_2 = calculate_path_length(traj_2)

        if score_1 > score_2:
            traj_w, traj_l = traj_1, traj_2
            score_w, score_l = score_1, score_2
        else:
            traj_w, traj_l = traj_2, traj_1
            score_w, score_l = score_2, score_1

        preference_data = {
            "npz_path": npz_path,
            "trajectory_w": traj_w.tolist(),
            "trajectory_l": traj_l.tolist(),
            "score_w": score_w,
            "score_l": score_l,
        }
        preferences.append(preference_data)

    print(f"Annotation complete! Generated {len(preferences)} preferences")

    if was_training:
        policy_model.train()

    return preferences



class DPODataset(Dataset):
    def __init__(self, preferences: list[dict]):
        self.preferences = preferences

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        """Return tensors and trajectories."""
        pref = self.preferences[idx]
        return {
            "data": load_npz_data(pref["npz_path"], DEVICE),
            "trajectory_w": np.asarray(pref["trajectory_w"], dtype=np.float32),
            "trajectory_l": np.asarray(pref["trajectory_l"], dtype=np.float32),
        }


class NPZDataset(Dataset):
    """Dataset that loads NPZ inputs for visualization/evaluation."""

    def __init__(self, npz_paths: list[str]):
        self.npz_paths = npz_paths

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        return load_npz_data(self.npz_paths[idx], DEVICE)


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
    latest_ckpt = run_dir / "latest.pth"

    initial_model_path = Path(args.model_path)
    if not initial_model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {initial_model_path}")

    initial_args_path = initial_model_path.parent / "args.json"
    if not initial_args_path.exists():
        raise FileNotFoundError(f"args.json not found next to model: {initial_args_path}")

    shutil.copy2(initial_model_path, latest_ckpt)
    shutil.copy2(initial_args_path, run_dir / "args.json")

    print(f"Saving artifacts to {run_dir}")

    policy_model, model_args = load_model(latest_ckpt)
    policy_model.train()
    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(run_dir / "dpo_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    with open(args.train_npz_list, "r") as f:
        train_npz_paths = json.load(f)
    with open(args.valid_npz_list, "r") as f:
        valid_npz_paths = json.load(f)
    valid_dataset = NPZDataset(valid_npz_paths)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    train_log: list[dict] = []

    visualize_validation(policy_model, valid_loader, model_args, run_dir, 0)

    for epoch in range(1, args.train_epochs + 1):
        if args.preference_mode == "gui":
            if args.ui_framework == "gradio":
                preferences = collect_preferences_gui_gradio(
                    policy_model,
                    model_args,
                    args.train_npz_list,
                    target_count=len(train_npz_paths),
                )
            else:
                preferences = collect_preferences_gui(
                    policy_model,
                    model_args,
                    args.train_npz_list,
                    target_count=len(train_npz_paths),
                )
        else:
            preferences = generate_rule_based_preferences(
                policy_model,
                model_args,
                args.train_npz_list,
            )

        if not preferences:
            print("No preferences collected this epoch. Skipping training step.")
            continue

        print(f"Loaded {len(preferences)} preference annotations")

        random.shuffle(preferences)
        train_dataset = DPODataset(preferences)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: x,
        )

        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

        policy_model.train()
        train_metrics = train_epoch(
            policy_model, reference_model, train_loader, optimizer, args, model_args
        )

        visualize_validation(policy_model, valid_loader, model_args, run_dir, epoch)

        print(
            f"Epoch {epoch}/{args.train_epochs}\n"
            f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
        )

        checkpoint_data = {
            "epoch": epoch,
            "model": policy_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args_dict,
        }

        torch.save(checkpoint_data, latest_ckpt)

        if epoch % 10 == 0:
            torch.save(checkpoint_data, os.path.join(run_dir, f"epoch_{epoch:03d}.pth"))

        train_log.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
        )
        df = pd.DataFrame(train_log)
        df.to_csv(os.path.join(run_dir, "dpo_train_log.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
