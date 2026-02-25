"""Preference collection methods for DPO training."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from tqdm import tqdm

from preference_optimization.utils import calculate_path_length, generate_trajectory_pair, load_npz_data


def generate_rule_based_preferences(
    policy_model: Diffusion_Planner,
    model_args,
    npz_list: Path,
    device: torch.device,
) -> list[dict]:
    """Generate preference annotations using rule-based scoring.

    Uses path length as the preference criterion: longer paths are preferred.

    Args:
        policy_model: The diffusion planner model
        model_args: Model configuration arguments
        npz_list: Path to JSON file containing list of NPZ paths
        device: Computation device

    Returns:
        List of preference dictionaries with keys:
            - npz_path: Path to observation file
            - trajectory_w: Winning trajectory
            - trajectory_l: Losing trajectory
            - score_w: Winner score (negative path length)
            - score_l: Loser score (negative path length)
    """
    # Set random seed for reproducibility
    seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    print(f"Rule-based annotation seed: {seed}")

    # Load NPZ paths
    with open(npz_list, "r") as f:
        npz_paths = json.load(f)

    preferences: list[dict] = []

    print(f"Total NPZ files to annotate: {len(npz_paths)}")

    # Set model to eval mode
    was_training = policy_model.training
    policy_model.eval()

    print("Starting rule-based annotation...")
    for npz_path in tqdm(npz_paths, desc="Annotating"):
        # Load observation data
        data = load_npz_data(npz_path, device)

        # Generate trajectory pair
        traj_1, traj_2, fde, attempts, *_ = generate_trajectory_pair(
            policy_model, model_args, data, device=device
        )

        # Compute scores (negative path length - longer is better)
        score_1 = calculate_path_length(traj_1)
        score_2 = calculate_path_length(traj_2)

        # Determine winner
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

    # Restore training mode
    if was_training:
        policy_model.train()

    return preferences
