"""DPO Trainer class for managing training loop."""

import copy
import json
import random
from pathlib import Path

import pandas as pd
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from preference_optimization.datasets import DPODataset, NPZDataset
from preference_optimization.dpo_loss import compute_dpo_loss
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
    ):
        """Initialize DPO trainer.

        Args:
            policy_model: The policy model to train
            model_args: Model configuration arguments
            optimizer: Optimizer for training
            device: Computation device
            run_dir: Directory for saving checkpoints and logs
            batch_size: Batch size for training
            beta: DPO regularization parameter
        """
        self.policy_model = policy_model
        self.model_args = model_args
        self.optimizer = optimizer
        self.device = device
        self.run_dir = run_dir
        self.batch_size = batch_size
        self.beta = beta

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

        # Create reference model (frozen copy of current policy)
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

        Args:
            epoch: Current epoch number
            args_dict: Training arguments dictionary
        """
        checkpoint_data = {
            "epoch": epoch,
            "model": self.policy_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": args_dict,
        }

        # Save latest checkpoint
        latest_path = self.run_dir / "latest.pth"
        torch.save(checkpoint_data, latest_path)

        # Save periodic checkpoint
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

    def visualize_epoch(
        self, valid_loader: DataLoader, epoch: int, max_samples: int = 50
    ) -> None:
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
