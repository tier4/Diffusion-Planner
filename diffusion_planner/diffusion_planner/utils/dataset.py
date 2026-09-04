import os

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        if isinstance(data_list, (str, bytes, os.PathLike)):
            self.data_list = openjson(data_list)
        else:
            self.data_list = list(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data.pop("version", None)
        return data


class DiffusionPlannerPairData(Dataset):
    def __init__(self, data_list, expected_gap: int | None = None):
        paths = openjson(data_list)
        expected_gap = expected_gap or None
        self.pairs = list(consecutive_frame_pairs(paths, expected_gap=expected_gap))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, path_a, _, path_b, gap = self.pairs[idx]
        data_a = dict(np.load(path_a, allow_pickle=True))
        data_b = dict(np.load(path_b, allow_pickle=True))
        return {
            "current": data_a,
            "next": data_b,
            "frame_gap": np.array(gap, dtype=np.int64),
        }
