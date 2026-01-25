"""Utility functions for preference optimization."""

from pathlib import Path

import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin


def load_npz_data(npz_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load NPZ file into tensors on the specified device.

    Args:
        npz_path: Path to the NPZ file
        device: Device to load tensors onto

    Returns:
        Dictionary of tensors with observation data
    """
    loaded = np.load(str(npz_path))
    data: dict[str, torch.Tensor] = {}

    for key, value in loaded.items():
        if key in {"map_name", "token"}:
            continue
        data[key] = torch.tensor(np.expand_dims(value, axis=0)).to(device)

    if "goal_pose" in data:
        data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    if "ego_agent_past" in data:
        data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

    if "ego_shape" not in data:
        wheel_base = 2.79
        ego_length = 4.34
        ego_width = 1.70
        data["ego_shape"] = torch.tensor(
            [[wheel_base, ego_length, ego_width]], dtype=torch.float32, device=device
        )

    return data


def calculate_path_length(trajectory: np.ndarray) -> float:
    """Calculate negative path length (longer paths = smaller values for preference ranking).

    Args:
        trajectory: Trajectory array [T, 4] (x, y, heading, velocity)

    Returns:
        Negative sum of distances between consecutive points
    """
    xy = trajectory[:, :2]
    diffs = np.diff(xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(-np.sum(dists))


def calculate_fde(trajectory_1: np.ndarray, trajectory_2: np.ndarray) -> float:
    """Calculate Final Displacement Error between two trajectory endpoints.

    Args:
        trajectory_1: First trajectory [T, 4] (x, y, heading, velocity)
        trajectory_2: Second trajectory [T, 4]

    Returns:
        Euclidean distance between final positions (in meters)
    """
    final_pos_1 = trajectory_1[-1, :2]
    final_pos_2 = trajectory_2[-1, :2]
    fde = np.linalg.norm(final_pos_1 - final_pos_2)
    return float(fde)


@torch.no_grad()
def generate_trajectory_pair(
    policy_model,
    model_args,
    data: dict[str, torch.Tensor],
    noise_scale: float = 2.5,
    fde_threshold: float = 2.0,
    max_retries: int = 50,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Generate two diverse trajectories with FDE-based retry logic.

    Generates pairs of trajectories:
    - First trajectory: deterministic (temperature=0)
    - Second trajectory: stochastic (with noise)

    Retries generation until FDE threshold is met or max retries reached.

    Args:
        policy_model: The diffusion planner model
        model_args: Model configuration arguments
        data: Input observation data
        noise_scale: Noise scale for second trajectory (default: 2.5)
        fde_threshold: Minimum FDE required between trajectories (default: 2.0m)
        max_retries: Maximum number of generation attempts (default: 50)
        device: Computation device (default: model's device)

    Returns:
        Tuple of (trajectory_1, trajectory_2, final_fde, attempts_used)
        - trajectory_1: Deterministic trajectory [T, 4]
        - trajectory_2: Stochastic trajectory [T, 4]
        - final_fde: Final displacement error achieved
        - attempts_used: Number of attempts used
    """
    device = device or next(policy_model.parameters()).device
    data = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data = model_args.observation_normalizer(data)

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    best_fde = 0.0
    best_pair = None

    for attempt in range(max_retries):
        trajectories = []

        # Generate deterministic trajectory (temperature=0)
        data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4).to(device)
        _, outputs = policy_model(data)
        traj_1 = outputs["prediction"][0, 0].cpu().numpy()
        trajectories.append(traj_1)

        # Generate stochastic trajectory (with noise)
        data["sampled_trajectories"] = noise_scale * torch.randn(B, P, future_len + 1, 4).to(
            device
        )
        _, outputs = policy_model(data)
        traj_2 = outputs["prediction"][0, 0].cpu().numpy()
        trajectories.append(traj_2)

        # Calculate FDE
        fde = calculate_fde(traj_1, traj_2)

        # Update best pair
        if fde > best_fde:
            best_fde = fde
            best_pair = (traj_1, traj_2)

        # Check if threshold is met
        if fde >= fde_threshold:
            return traj_1, traj_2, fde, attempt + 1

    # Max retries reached, return best pair found
    return best_pair[0], best_pair[1], best_fde, max_retries
