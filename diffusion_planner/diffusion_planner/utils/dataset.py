import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        self.data_list = openjson(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict

        # added ego_shape (wheel_base, length, width)
        data["ego_shape"] = np.array([2.75, 4.34, 1.70], dtype=np.float32)

        return data
