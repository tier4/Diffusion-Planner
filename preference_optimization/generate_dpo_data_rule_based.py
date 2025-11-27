"""
DPO Data Generation Program (Rule-Based)

This program generates paired trajectory samples from Diffusion Planner and automatically
annotates which trajectory is preferred based on path length (shorter is better).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--npz_list", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=Path("dpo_preferences_rule_based.json"))
    parser.add_argument("--excluded_json", type=Path, default=Path("dpo_excluded_rule_based.json"))
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> tuple[Diffusion_Planner, Config]:
    """Load Diffusion Planner model and args."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Load args from checkpoint using Config class
    model_dir = model_path.parent
    args_path = model_dir / "args.json"

    # Use Config class to load configuration (handles normalizers automatically)
    model_args = Config(str(args_path), guidance_fn=None)

    # Initialize model
    model = Diffusion_Planner(model_args)

    # Load checkpoint weights (following diffusion_planner_node.py)
    if "model" in checkpoint:
        # Handle DDP checkpoint
        state_dict = checkpoint["model"]
        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif "ema_state_dict" in checkpoint:
        print("Loading EMA weights")
        model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    return model, model_args


class DPODataGenerator:
    """Generates trajectory pairs and manages the annotation process."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")

        # Set random seed once for the entire session
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        print(f"Random seed: {seed}")

        # Load model
        self.model, self.model_args = load_model(args.model_path, self.device)
        self.model.eval()

        # Load NPZ file list
        with open(args.npz_list, "r") as f:
            self.npz_paths = json.load(f)

        # Load existing annotations if resuming
        self.preferences = []
        self.excluded = []

        if args.output_json.exists():
            print(f"Resuming from {args.output_json}")
            with open(args.output_json, "r") as f:
                self.preferences = json.load(f)

        if args.excluded_json.exists():
            print(f"Loading excluded list from {args.excluded_json}")
            with open(args.excluded_json, "r") as f:
                self.excluded = json.load(f)

        # Filter out already annotated and excluded files
        annotated_paths = {pref["npz_path"] for pref in self.preferences}
        excluded_paths = set(self.excluded)
        self.npz_paths = [
            p for p in self.npz_paths if p not in annotated_paths and p not in excluded_paths
        ]

        print(f"Total NPZ files to annotate: {len(self.npz_paths)}")

    @torch.no_grad()
    def generate_trajectory_pair(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate two different trajectories with different random noise.

        Returns:
            tuple: (trajectory_1, trajectory_2)
                Each is a numpy array of shape [T, 4] representing the predicted trajectory
        """
        # Normalize inputs once
        data = self.model_args.observation_normalizer(data)

        # Generate trajectory parameters
        # B is original batch size (should be 1)
        B = data["ego_current_state"].shape[0]
        P = 1 + self.model_args.predicted_neighbor_num
        future_len = self.model_args.future_len

        # Duplicate data for batch inference (batch size 2)
        batch_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.repeat(2, *([1] * (v.dim() - 1)))
            else:
                batch_data[k] = v  # Should not happen for tensors but just in case

        # Generate random noise for batch size 2
        batch_data["sampled_trajectories"] = 2.5 * torch.randn(2 * B, P, future_len + 1, 4).to(
            self.device
        )

        # Run inference once
        _, outputs = self.model(batch_data)

        # Extract ego predictions
        prediction = outputs["prediction"]  # [2*B, P, T, 4]

        # We expect B=1, so we have 2 predictions
        trajectories = []
        for i in range(2):
            ego_prediction = prediction[i, 0].cpu().numpy()  # [T, 4]
            trajectories.append(ego_prediction)

        return trajectories[0], trajectories[1]

    def save_preferences(self):
        """Save current preferences and excluded list to JSON."""
        with open(self.args.output_json, "w") as f:
            json.dump(self.preferences, f, indent=2)

        with open(self.args.excluded_json, "w") as f:
            json.dump(self.excluded, f, indent=2)


def load_npz_data(npz_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load and preprocess NPZ file."""
    loaded = np.load(str(npz_path))
    data = {}

    for key, value in loaded.items():
        if key == "map_name" or key == "token":
            continue
        # Add batch dimension
        data[key] = torch.tensor(np.expand_dims(value, axis=0)).to(device)

    # Convert heading to cos/sin
    if "goal_pose" in data:
        data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    if "ego_agent_past" in data:
        data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

    # Add ego_shape if not present (following diffusion_planner_node.py:365-367)
    if "ego_shape" not in data:
        # Default values for ego vehicle shape
        wheel_base = 2.79
        ego_length = 4.34
        ego_width = 1.70
        data["ego_shape"] = torch.tensor(
            [[wheel_base, ego_length, ego_width]], dtype=torch.float32, device=device
        )

    return data


def calculate_path_length(trajectory: np.ndarray) -> float:
    """
    Calculate the length of the trajectory.
    trajectory: numpy array of [x, y, yaw, vel] or similar. We only use x, y (indices 0 and 1).
    """
    # Extract x, y
    xy = trajectory[:, :2]
    # Calculate differences between consecutive points
    diffs = np.diff(xy, axis=0)
    # Calculate Euclidean distance for each segment
    dists = np.linalg.norm(diffs, axis=1)
    # Sum distances
    return float(-np.sum(dists))


def main():
    args = parse_args()

    # Create data generator
    generator = DPODataGenerator(args)

    print("Starting rule-based annotation...")

    for i, npz_path in enumerate(tqdm(generator.npz_paths)):
        # Load data
        data = load_npz_data(npz_path, generator.device)

        # Generate trajectory pair
        traj_1, traj_2 = generator.generate_trajectory_pair(data)

        # Calculate path lengths (scores)
        # Use negative path length so that bigger is better
        score_1 = calculate_path_length(traj_1)
        score_2 = calculate_path_length(traj_2)

        # Determine preference (bigger score is better)
        if score_1 > score_2:
            traj_w = traj_1
            traj_l = traj_2
            score_w = score_1
            score_l = score_2
        else:
            # If score_2 > score_1 or equal, we prefer traj_2 or treat it as winner for now
            traj_w = traj_2
            traj_l = traj_1
            score_w = score_2
            score_l = score_1

        # Record preference
        preference_data = {
            "npz_path": npz_path,
            "trajectory_w": traj_w.tolist(),
            "trajectory_l": traj_l.tolist(),
            "score_w": score_w,
            "score_l": score_l,
        }
        generator.preferences.append(preference_data)

        # Auto-save every 100 samples
        if (i + 1) % 100 == 0:
            generator.save_preferences()

    # Final save
    generator.save_preferences()
    print("Annotation complete!")


if __name__ == "__main__":
    main()
