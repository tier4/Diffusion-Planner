from pathlib import Path

import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin


def load_npz_data(npz_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load NPZ file into tensors on the specified device."""
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
    """Calculate negative path length (longer paths become smaller values)."""
    xy = trajectory[:, :2]
    diffs = np.diff(xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(-np.sum(dists))


@torch.no_grad()
def generate_deterministic_trajectory(
    policy_model,
    model_args,
    data: dict[str, torch.Tensor],
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate a deterministic trajectory with temperature 0 (no noise)."""
    device = device or next(policy_model.parameters()).device
    data = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data = model_args.observation_normalizer(data)
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4).to(device)
    _, outputs = policy_model(data)
    ego_prediction = outputs["prediction"][0, 0].cpu().numpy()

    return ego_prediction


@torch.no_grad()
def generate_trajectory_pair(
    policy_model,
    model_args,
    data: dict[str, torch.Tensor],
    noise_scale: float = 2.5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two trajectories using identical inputs but different noise."""
    device = device or next(policy_model.parameters()).device
    data = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data = model_args.observation_normalizer(data)
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    trajectories = []
    for _ in range(2):
        data["sampled_trajectories"] = noise_scale * torch.randn(B, P, future_len + 1, 4).to(device)
        _, outputs = policy_model(data)
        ego_prediction = outputs["prediction"][0, 0].cpu().numpy()
        trajectories.append(ego_prediction)

    return trajectories[0], trajectories[1]
