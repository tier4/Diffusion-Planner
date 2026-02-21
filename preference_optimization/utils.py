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
        if key in {"map_name", "token", "delay"}:
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


def calculate_ade(trajectory_1: np.ndarray, trajectory_2: np.ndarray) -> float:
    """Calculate Average Displacement Error between two trajectories.

    ADE is the mean Euclidean distance across all timesteps.

    Args:
        trajectory_1: First trajectory [T, 4] (x, y, cos, sin)
        trajectory_2: Second trajectory [T, 4] or [T, 3] (x, y, heading)

    Returns:
        Mean Euclidean distance across all timesteps (in meters)
    """
    positions_1 = trajectory_1[:, :2]
    positions_2 = trajectory_2[:, :2]
    displacements = np.sqrt(np.sum((positions_1 - positions_2) ** 2, axis=1))
    return float(np.mean(displacements))


@torch.no_grad()
def generate_trajectory_pair(
    policy_model,
    model_args,
    data: dict[str, torch.Tensor],
    noise_scale: float = 2.5,
    fde_threshold: float = 2.0,
    ade_threshold: float = 1.0,
    max_retries: int = 50,
    device: torch.device | None = None,
    gt_similarity_mode: bool = True,
    gt_trajectory: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, torch.Tensor]:
    """Generate two diverse trajectories with threshold-based retry logic.

    Generates pairs of trajectories:
    - First trajectory: deterministic (temperature=0)
    - Second trajectory: stochastic (with noise)

    Two modes available:
    - Diversity mode: Retry until FDE between trajectories >= fde_threshold
    - GT-similarity mode (default): Retry until ADE between stochastic and GT <= ade_threshold

    Args:
        policy_model: The diffusion planner model
        model_args: Model configuration arguments
        data: Input observation data
        noise_scale: Noise scale for second trajectory (default: 2.5)
        fde_threshold: FDE threshold - minimum between trajectories (diversity mode)
        ade_threshold: ADE threshold - maximum to GT (GT-similarity mode)
        max_retries: Maximum number of generation attempts (default: 50)
        device: Computation device (default: model's device)
        gt_similarity_mode: If True (default), find stochastic trajectory close to GT using ADE
        gt_trajectory: Ground truth trajectory [T, 3] (x, y, heading) for GT-similarity mode

    Returns:
        Tuple of (trajectory_1, trajectory_2, final_metric, attempts_used, ego_shape)
        - trajectory_1: Deterministic trajectory [T, 4]
        - trajectory_2: Stochastic trajectory [T, 4]
        - final_metric: ADE to GT (GT mode) or FDE between trajectories (diversity mode)
        - attempts_used: Number of attempts used
        - ego_shape: Vehicle shape parameters
    """
    device = device or next(policy_model.parameters()).device
    data = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data = model_args.observation_normalizer(data)
    
    ego_shape = data["ego_shape"]

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Generate deterministic trajectory once (temperature=0)
    data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4).to(device)
    _, outputs = policy_model(data)
    traj_1 = outputs["prediction"][0, 0].cpu().numpy()

    # Initialize best tracking based on mode
    if gt_similarity_mode and gt_trajectory is not None:
        # GT-similarity mode: minimize ADE to GT
        best_metric = float("inf")
    else:
        # Diversity mode: maximize FDE between trajectories
        best_metric = 0.0
    best_traj_2 = None

    for attempt in range(max_retries):
        # Generate stochastic trajectory (with noise)
        data["sampled_trajectories"] = noise_scale * torch.randn(B, P, future_len + 1, 4).to(
            device
        )
        _, outputs = policy_model(data)
        traj_2 = outputs["prediction"][0, 0].cpu().numpy()

        if gt_similarity_mode and gt_trajectory is not None:
            # GT-similarity mode: calculate ADE to ground truth
            # GT trajectory is [T, 3] (x, y, heading), we only need (x, y)
            ade = calculate_ade(traj_2, gt_trajectory)

            # Update best (minimize ADE to GT)
            if ade < best_metric:
                best_metric = ade
                best_traj_2 = traj_2

            # Check if threshold is met (ADE to GT <= threshold)
            if ade <= ade_threshold:
                return traj_1, traj_2, ade, attempt + 1, ego_shape
        else:
            # Diversity mode: calculate FDE between trajectories
            fde = calculate_fde(traj_1, traj_2)

            # Update best (maximize FDE between trajectories)
            if fde > best_metric:
                best_metric = fde
                best_traj_2 = traj_2

            # Check if threshold is met (FDE between trajectories >= threshold)
            if fde >= fde_threshold:
                return traj_1, traj_2, fde, attempt + 1, ego_shape

    # Max retries reached, return best pair found
    return traj_1, best_traj_2, best_metric, max_retries, ego_shape
