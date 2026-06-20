import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.scene_skip import filter_scene_list
from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list, skip_filter: bool = True, sidecar_root=None):
        data = openjson(data_list)
        # Accept both legacy list format and sampling.py dict format {"seed": ..., "files": [...]}
        files = data["files"] if isinstance(data, dict) else data
        # Drop frames the converter flagged skip_for_training (red-light creep, no-future-
        # progress, ...): valid only for the reproducer, never for training/eval. Frames
        # with no resolvable sidecar (older corpora) are kept -> backward compatible.
        self.data_list = filter_scene_list(
            files, sidecar_root=sidecar_root, enabled=skip_filter, label=str(data_list)
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        return data
