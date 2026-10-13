"""Visualization utilities for DPO training."""

from pathlib import Path

import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.visualize_input import visualize_inputs
from torch.utils.data import DataLoader

from scenario_generation.canvas import Figure


@torch.no_grad()
def visualize_validation(
    policy_model: Diffusion_Planner,
    valid_loader: DataLoader,
    model_args,
    save_dir: Path,
    epoch: int,
    device: torch.device,
    max_samples: int = 50,
) -> None:
    """Visualize validation predictions and save as images.

    Args:
        policy_model: The policy model to evaluate
        valid_loader: DataLoader for validation data
        model_args: Model configuration arguments
        save_dir: Directory to save visualizations
        epoch: Current epoch number
        device: Computation device
        max_samples: Maximum number of samples to visualize
    """
    policy_model.eval()

    vis_dir = save_dir / "validation_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0

    for batch in valid_loader:
        if sample_count >= max_samples:
            break

        for sample in batch:
            if sample_count >= max_samples:
                break

            # Clone and prepare data
            data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

            B = data["ego_current_state"].shape[0]
            P = 1 + model_args.predicted_neighbor_num
            future_len = model_args.future_len

            # Generate prediction with deterministic sampling
            data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4, device=device)

            # Normalize inputs
            data = model_args.observation_normalizer(data)

            # Run model
            _, outputs = policy_model(data)
            prediction = outputs["prediction"][0].cpu().numpy()  # [P, T, 4]

            # Create visualization
            fig = Figure((8, 8), 100)
            ax = fig.panel(equal=True)

            # Visualize input context (map, past trajectories, etc.)
            vis_data = model_args.observation_normalizer.inverse(data)
            for k, v in vis_data.items():
                if isinstance(v, torch.Tensor):
                    vis_data[k] = v.cpu()
            visualize_inputs(vis_data, save_path=None, ax=ax)

            # Plot ego prediction
            ax.lines([prediction[0, :, :2]], "#ffa500", width=2, label="Ego Prediction")

            # Plot neighbor predictions
            ax.lines(
                [prediction[i, :, :2] for i in range(1, prediction.shape[0])],
                "#008080",
                width=1,
                alpha=0.5,
            )

            ax.legend("upper right")
            ax.title(f"Epoch {epoch} - Sample {sample_count + 1}")

            # Save figure
            save_path = vis_dir / f"sample_{sample_count:03d}_epoch_{epoch:04d}.png"
            fig.save(save_path)

            sample_count += 1

    print(f"Saved {sample_count} validation visualizations to {vis_dir}")
