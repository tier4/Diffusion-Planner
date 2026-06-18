"""Dataset classes for DPO training."""

import numpy as np
import torch
from diffusion_planner.utils.scene_skip import filter_scene_list, is_skipped
from torch.utils.data import Dataset

from preference_optimization.utils import load_npz_data


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization.

    Each sample contains observation data and preferred/dispreferred trajectory pair.
    """

    def __init__(
        self,
        preferences: list[dict],
        device: torch.device,
        skip_filter: bool = True,
        sidecar_root=None,
    ):
        """Initialize DPO dataset.

        Args:
            preferences: List of preference dictionaries with keys:
                - npz_path: Path to observation file
                - trajectory_w: Winning trajectory
                - trajectory_l: Losing trajectory
            device: Device to load tensors onto
            skip_filter: drop preferences whose npz is flagged skip_for_training (default on).
            sidecar_root: where the per-frame JSON sidecars live if not next to the NPZ.
        """
        if skip_filter:
            n0 = len(preferences)
            preferences = [p for p in preferences if not is_skipped(p["npz_path"], sidecar_root)]
            if len(preferences) < n0:
                print(f"[skip-filter] DPODataset: kept {len(preferences)}/{n0} preferences")
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

    def __init__(
        self,
        npz_paths: list[str],
        device: torch.device,
        skip_filter: bool = True,
        sidecar_root=None,
    ):
        """Initialize NPZ dataset.

        Args:
            npz_paths: List of paths to NPZ observation files
            device: Device to load tensors onto
            skip_filter: drop NPZs flagged skip_for_training (default on).
            sidecar_root: where the per-frame JSON sidecars live if not next to the NPZ.
        """
        self.npz_paths = filter_scene_list(
            npz_paths, sidecar_root=sidecar_root, enabled=skip_filter, label="NPZDataset"
        )
        self.device = device

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return observation data."""
        return load_npz_data(self.npz_paths[idx], self.device)
