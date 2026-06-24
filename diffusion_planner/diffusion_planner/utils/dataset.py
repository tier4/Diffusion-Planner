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


class DiffusionPlannerPairData(Dataset):
    """Consecutive-frame pairs for the cross-frame temporal-consistency loss.

    Builds (frame_t, frame_{t+g}) pairs from the same scenario timeline (path-based,
    via ``planner_metrics.replan_consistency``), keeping only pairs whose frame gap
    equals ``step_g``. ``__getitem__`` returns a flat dict: frame-t's arrays, frame-(t+g)'s
    arrays under ``b__`` keys, plus the GT inter-frame transform (``tc_rel_pos``,
    ``tc_rel_h``) so the consistency loss can align the two predictions.
    """

    def __init__(self, data_list, step_g: int = 3, skip_filter: bool = True, sidecar_root=None):
        from planner_metrics.replan_consistency import (
            consecutive_frame_pairs,
            ego_future_to_4col,
            inter_frame_transform,
        )

        data = openjson(data_list)
        files = data["files"] if isinstance(data, dict) else data
        files = filter_scene_list(
            files, sidecar_root=sidecar_root, enabled=skip_filter, label=str(data_list)
        )
        self.step_g = int(step_g)
        self._to4 = ego_future_to_4col
        self._tf = inter_frame_transform
        self.pairs = [
            (pa, pb)
            for (_ia, pa, _ib, pb, g) in consecutive_frame_pairs(files)
            if g == self.step_g
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pa, pb = self.pairs[idx]
        a = dict(np.load(pa, allow_pickle=True)); a.pop("version", None)
        b = dict(np.load(pb, allow_pickle=True)); b.pop("version", None)
        rel_pos, rel_h = self._tf(self._to4(a["ego_agent_future"]), self.step_g)
        item = dict(a)
        for k, v in b.items():
            item["b__" + k] = v
        item["tc_rel_pos"] = rel_pos
        item["tc_rel_h"] = rel_h
        return item
