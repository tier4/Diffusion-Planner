"""Dataset classes for DPO training."""

import numpy as np
import torch
from torch.utils.data import Dataset

from preference_optimization.utils import load_npz_data


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization.

    Each sample contains observation data and preferred/dispreferred trajectory pair.
    """

    def __init__(self, preferences: list[dict], device: torch.device):
        """Initialize DPO dataset.

        Args:
            preferences: List of preference dictionaries with keys:
                - npz_path: Path to observation file
                - trajectory_w: Winning trajectory
                - trajectory_l: Losing trajectory
            device: Device to load tensors onto
        """
        self.preferences = preferences
        self.device = device

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, idx: int) -> dict:
        """Get a preference sample.

        Returns:
            Dictionary with:
                - data: Observation tensors
                - trajectory_w: Preferred trajectory [T, 4]
                - trajectory_l: Dispreferred trajectory [T, 4]
        """
        pref = self.preferences[idx]
        return {
            "data": load_npz_data(pref["npz_path"], self.device),
            "trajectory_w": np.asarray(pref["trajectory_w"], dtype=np.float32),
            "trajectory_l": np.asarray(pref["trajectory_l"], dtype=np.float32),
        }


class NPZDataset(Dataset):
    """Dataset for loading NPZ observation files.

    Used for validation visualization.
    """

    def __init__(self, npz_paths: list[str], device: torch.device):
        """Initialize NPZ dataset.

        Args:
            npz_paths: List of paths to NPZ observation files
            device: Device to load tensors onto
        """
        self.npz_paths = npz_paths
        self.device = device

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return observation data."""
        return load_npz_data(self.npz_paths[idx], self.device)
