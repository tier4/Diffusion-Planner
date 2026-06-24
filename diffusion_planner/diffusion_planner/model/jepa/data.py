"""Annotation-free ego-trajectory datasets for JEPA Stage-I / Stage-II training.

Each NPZ holds an ego-centric trajectory (past + future); concatenated and converted to
4-col [x, y, cos, sin] it is one self-supervised "episode". State = ego pose; action =
per-step pose delta (the same velocity representation the energy derives from the
planner's predicted waypoints). Datasets take a list of (L, 4) trajectories (pre-extracted
for speed) and emit windows. Mirrors refer/sage StateJEPADataset / ACWindowDataset.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ego_traj_from_npz", "EgoWindowDataset", "EgoJEPADataset"]


def ego_traj_from_npz(d: dict) -> torch.Tensor:
    """Concatenate ego past+future into a (L, 4) [x, y, cos, sin] trajectory."""
    past = np.asarray(d["ego_agent_past"], dtype=np.float32)
    fut = np.asarray(d["ego_agent_future"], dtype=np.float32)
    traj = np.concatenate([past, fut], axis=0)  # (L, 3 or 4)
    t = torch.from_numpy(traj)
    if t.shape[-1] == 4:
        return t
    xy, h = t[:, :2], t[:, 2]
    return torch.cat([xy, torch.cos(h)[:, None], torch.sin(h)[:, None]], dim=-1)


def _window_starts(trajs: list[torch.Tensor], span: int) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    for ti, tr in enumerate(trajs):
        for s in range(0, tr.shape[0] - span + 1):
            starts.append((ti, s))
    return starts


class EgoWindowDataset(Dataset):
    """Stage-II windows: (s [W+1, 4], a [W, 4]) where a_k = s_{k+1} − s_k."""

    def __init__(self, trajs: list[torch.Tensor], window: int = 16):
        self.trajs = [torch.as_tensor(t, dtype=torch.float32) for t in trajs]
        self.window = int(window)
        self.starts = _window_starts(self.trajs, self.window + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        ti, s = self.starts[idx]
        seq = self.trajs[ti][s : s + self.window + 1]  # [W+1, 4]
        a = seq[1:] - seq[:-1]  # [W, 4] per-step pose delta (velocity)
        return seq, a


class EgoJEPADataset(Dataset):
    """Stage-I windows: (ctx1, ctx2 [W,4] masked views, targets [M,4], ks [M])."""

    def __init__(self, trajs: list[torch.Tensor], window: int = 16, k_max: int = 5,
                 num_mask: int = 3, feature_mask_ratio: float = 0.3,
                 time_mask_ratio: float = 0.1):
        self.trajs = [torch.as_tensor(t, dtype=torch.float32) for t in trajs]
        self.window = int(window)
        self.k_max = int(k_max)
        self.num_mask = int(num_mask)
        self.feature_mask_ratio = float(feature_mask_ratio)
        self.time_mask_ratio = float(time_mask_ratio)
        self.starts = _window_starts(self.trajs, self.window + self.k_max)

    def __len__(self):
        return len(self.starts)

    def _mask(self, ctx: torch.Tensor) -> torch.Tensor:
        x = ctx.clone()
        W, D = x.shape
        if self.feature_mask_ratio > 0:
            n = max(1, int(D * self.feature_mask_ratio))
            drop = np.random.choice(D, size=min(D, n), replace=False)
            x[:, drop] = 0.0
        if self.time_mask_ratio > 0:
            n = max(1, int(W * self.time_mask_ratio))
            drop = np.random.choice(W, size=min(W, n), replace=False)
            x[drop, :] = 0.0
        return x

    def __getitem__(self, idx):
        ti, s = self.starts[idx]
        tr = self.trajs[ti]
        ctx = tr[s : s + self.window]  # [W, 4]
        ks = np.sort(
            np.random.choice(
                np.arange(1, self.k_max + 1), size=self.num_mask,
                replace=self.num_mask > self.k_max,
            )
        ).astype(np.int64)
        targets = torch.stack([tr[s + self.window - 1 + int(k)] for k in ks], dim=0)  # [M,4]
        return self._mask(ctx), self._mask(ctx), targets, torch.from_numpy(ks)
