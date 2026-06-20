"""DPO Trainer class for managing training loop."""

import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.scene_skip import is_skipped
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from preference_optimization.datasets import DPODataset, NPZDataset
from preference_optimization.dpo_loss import compute_dpo_loss
from preference_optimization.utils import (
    calculate_ade,
    generate_deterministic_trajectory,
    load_npz_data,
)
from preference_optimization.visualization import visualize_validation


class DPOTrainer:
    """Trainer for Direct Preference Optimization.

    Manages the DPO training loop, including:
    - Dataset creation
    - Reference model management
    - Training epoch execution
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        policy_model: Diffusion_Planner,
        model_args,
        optimizer: optim.Optimizer,
        device: torch.device,
        run_dir: Path,
        batch_size: int = 32,
        beta: float = 0.1,
        use_lora: bool = False,
    ):
        """Initialize DPO trainer.

        Args:
            policy_model: The policy model to train (already LoRA-wrapped if use_lora=True)
            model_args: Model configuration arguments
            optimizer: Optimizer for training (must be created after LoRA is applied)
            device: Computation device
            run_dir: Directory for saving checkpoints and logs
            batch_size: Batch size for training
            beta: DPO regularization parameter
            use_lora: If True, reference model passes use disable_adapter() instead of deepcopy.
        """
        self.policy_model = policy_model
        self.model_args = model_args
        self.optimizer = optimizer
        self.device = device
        self.run_dir = run_dir
        self.batch_size = batch_size
        self.beta = beta
        self.use_lora = use_lora

        self.train_log: list[dict] = []

    def train_epoch(
        self, preferences: list[dict], epoch: int, progress_callback=None
    ) -> dict[str, float]:
        """Train for one epoch on preference data.

        Args:
            preferences: List of preference annotations
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        if not preferences:
            print("No preferences provided. Skipping training.")
            return {"loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0}

        print(f"Training on {len(preferences)} preferences")

        # Shuffle preferences
        random.shuffle(preferences)

        # Create dataset and dataloader
        train_dataset = DPODataset(preferences, self.device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: x,
        )

        # When using LoRA, the reference model is the same model with adapters
        # disabled via PEFT's disable_adapter() context (no deepcopy needed).
        # When using full fine-tuning, deepcopy the current policy as before.
        if self.use_lora:
            reference_model = None
        else:
            reference_model = copy.deepcopy(self.policy_model)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False

        # Train
        self.policy_model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0
        num_batches = 0

        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
            self.optimizer.zero_grad()

            # Compute DPO loss
            loss, metrics = compute_dpo_loss(
                self.policy_model,
                reference_model,
                batch,
                self.beta,
                self.model_args,
                self.device,
            )

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += metrics["accuracy"]
            total_reward_margin += metrics["avg_reward_margin"]
            num_batches += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": total_batches,
                        "loss": float(loss.item()),
                        "accuracy": float(metrics["accuracy"]),
                        "reward_margin": float(metrics["avg_reward_margin"]),
                    }
                )

        # Average metrics
        avg_metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "accuracy": total_accuracy / num_batches if num_batches > 0 else 0.0,
            "reward_margin": total_reward_margin / num_batches if num_batches > 0 else 0.0,
        }

        return avg_metrics

    def save_checkpoint(self, epoch: int, args_dict: dict) -> None:
        """Save model checkpoint.

        When LoRA is active, only the adapter weights are saved (adapter_model.bin +
        adapter_config.json) in a timestamped sub-directory. The base model weights
        are not duplicated — combine them via lora_utils.load_lora_checkpoint() at
        inference time.

        When LoRA is inactive (full fine-tuning), the full model state dict is saved
        as before.

        Args:
            epoch: Current epoch number
            args_dict: Training arguments dictionary
        """
        if self.use_lora:
            from preference_optimization.lora_utils import save_lora_checkpoint

            lora_dir = str(self.run_dir / f"lora_epoch_{epoch:03d}")
            save_lora_checkpoint(self.policy_model, lora_dir)
            # Save optimizer state so resumed runs continue with warm AdamW moments.
            torch.save(
                {"epoch": epoch, "optimizer": self.optimizer.state_dict()},
                Path(lora_dir) / "optimizer.pth",
            )
            # Also keep a pointer to "latest" lora directory
            latest_link = self.run_dir / "lora_latest"
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)
            latest_link.symlink_to(f"lora_epoch_{epoch:03d}")
        else:
            checkpoint_data = {
                "epoch": epoch,
                "model": self.policy_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "args": args_dict,
            }

            latest_path = self.run_dir / "latest.pth"
            torch.save(checkpoint_data, latest_path)

            if epoch % 10 == 0:
                epoch_path = self.run_dir / f"epoch_{epoch:03d}.pth"
                torch.save(checkpoint_data, epoch_path)
                print(f"Saved checkpoint: {epoch_path}")

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log training metrics.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        # Add to log
        log_entry = {"epoch": epoch, **{f"train_{k}": v for k, v in metrics.items()}}
        self.train_log.append(log_entry)

        # Save log to file
        df = pd.DataFrame(self.train_log)
        log_path = self.run_dir / "dpo_train_log.tsv"
        df.to_csv(log_path, sep="\t", index=False)

        # Print metrics
        print(
            f"Epoch {epoch}: "
            f"Loss={metrics['loss']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Reward Margin={metrics['reward_margin']:.4f}"
        )

    def visualize_epoch(self, valid_loader: DataLoader, epoch: int, max_samples: int = 50) -> None:
        """Visualize validation predictions for current epoch.

        Args:
            valid_loader: Validation data loader
            epoch: Current epoch number
            max_samples: Maximum number of samples to visualize
        """
        visualize_validation(
            self.policy_model,
            valid_loader,
            self.model_args,
            self.run_dir,
            epoch,
            self.device,
            max_samples=max_samples,
        )

    def create_validation_loader(self, npz_paths: list[str]) -> DataLoader:
        """Create validation data loader.

        Args:
            npz_paths: List of paths to validation NPZ files

        Returns:
            Validation DataLoader
        """
        valid_dataset = NPZDataset(npz_paths, self.device)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x,
        )
        return valid_loader

    def save_epoch1_baselines(self, preferences: list[dict]) -> None:
        """Save deterministic trajectories for each annotated NPZ as the epoch-1 reference.

        These baselines are used by compute_trajectory_drift() at subsequent epochs
        to verify that the model weights are actually changing under DPO training.
        The file is written only once; subsequent calls are no-ops.

        Args:
            preferences: List of preference dicts with at least an 'npz_path' key.
        """
        baseline_path = self.run_dir / "epoch1_baselines.npz"
        if baseline_path.exists():
            return

        self.policy_model.eval()
        paths_list: list[str] = []
        trajs_list: list[np.ndarray] = []
        seen: set[str] = set()

        for pref in preferences:
            npz_path = pref.get("npz_path")
            # Skip converter-flagged frames (consistent with the DPODataset training filter).
            if npz_path is None or npz_path in seen or is_skipped(npz_path):
                continue
            try:
                obs = load_npz_data(npz_path, self.device)
                traj = generate_deterministic_trajectory(
                    self.policy_model, self.model_args, obs, self.device
                )
                paths_list.append(str(npz_path))
                trajs_list.append(traj)
                seen.add(npz_path)
            except Exception as e:
                print(f"  [baseline] skipping {npz_path}: {e}")

        if not paths_list:
            return

        np.savez(
            baseline_path,
            paths=np.array(paths_list),
            trajectories=np.stack(trajs_list),
        )
        print(f"Saved epoch-1 baselines for {len(paths_list)} samples → {baseline_path}")

    def compute_trajectory_drift(self) -> str:
        """Compute ADE between the current model's deterministic output and epoch-1 baselines.

        Returns a human-readable summary string, or an empty string if no baselines exist.
        """
        baseline_path = self.run_dir / "epoch1_baselines.npz"
        if not baseline_path.exists():
            return ""

        saved = np.load(baseline_path, allow_pickle=True)
        paths_list: list[str] = saved["paths"].tolist()
        baselines: np.ndarray = saved["trajectories"]  # [N, T, 4]

        self.policy_model.eval()
        ades: list[float] = []
        for npz_path, baseline_traj in zip(paths_list, baselines):
            try:
                obs = load_npz_data(npz_path, self.device)
                current_traj = generate_deterministic_trajectory(
                    self.policy_model, self.model_args, obs, self.device
                )
                ades.append(calculate_ade(current_traj, baseline_traj))
            except Exception as e:
                print(f"  [drift] skipping {npz_path}: {e}")

        if not ades:
            return "Drift vs epoch 1: N/A"

        mean_ade = float(np.mean(ades))
        std_ade = float(np.std(ades))
        max_ade = float(np.max(ades))
        print(
            f"  [drift] mean_ADE={mean_ade:.4f}m  std={std_ade:.4f}m  "
            f"max={max_ade:.4f}m  n={len(ades)}"
        )
        return (
            f"Drift vs epoch 1: mean={mean_ade:.3f}m  "
            f"std={std_ade:.3f}m  max={max_ade:.3f}m  (n={len(ades)})"
        )
