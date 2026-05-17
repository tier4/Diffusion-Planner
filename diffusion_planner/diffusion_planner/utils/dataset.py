import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        data = openjson(data_list)
        # Accept both legacy list format and sampling.py dict format {"seed": ..., "files": [...]}
        self.data_list = data["files"] if isinstance(data, dict) else data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        return data
