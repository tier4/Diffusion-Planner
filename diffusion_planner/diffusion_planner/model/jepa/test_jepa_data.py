"""Tests for the JEPA ego-trajectory data layer (synthetic; no real NPZ needed)."""

from __future__ import annotations

import math

import numpy as np
import torch


def test_ego_traj_from_npz_concats_and_cossins():
    from diffusion_planner.model.jepa.data import ego_traj_from_npz

    past = np.zeros((3, 3), dtype=np.float32)
    past[:, 2] = 0.0  # heading 0
    fut = np.zeros((2, 3), dtype=np.float32)
    fut[:, 2] = math.pi / 2  # heading 90deg
    traj = ego_traj_from_npz({"ego_agent_past": past, "ego_agent_future": fut})
    assert traj.shape == (5, 4)  # 3 past + 2 future, 4-col
    assert torch.allclose(traj[0, 2:], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(traj[-1, 2:], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_ego_window_dataset_action_is_state_delta():
    from diffusion_planner.model.jepa.data import EgoWindowDataset

    # one straight trajectory along +x, heading 0
    L = 20
    traj = torch.zeros(L, 4)
    traj[:, 0] = torch.arange(L) * 0.5
    traj[:, 2] = 1.0
    ds = EgoWindowDataset([traj], window=8)
    s, a = ds[0]
    assert s.shape == (9, 4) and a.shape == (8, 4)  # W+1 states, W actions
    # action = s[1:] - s[:-1]: constant +0.5 in x, 0 elsewhere
    assert torch.allclose(a[:, 0], torch.full((8,), 0.5), atol=1e-5)
    assert torch.allclose(a[:, 1:], torch.zeros(8, 3), atol=1e-5)


def test_ego_jepa_dataset_shapes():
    from diffusion_planner.model.jepa.data import EgoJEPADataset

    traj = torch.randn(40, 4)
    ds = EgoJEPADataset([traj], window=16, k_max=5, num_mask=3)
    ctx1, ctx2, targets, ks = ds[0]
    assert ctx1.shape == (16, 4) and ctx2.shape == (16, 4)
    assert targets.shape == (3, 4)
    assert ks.shape == (3,)
    assert ks.min() >= 1 and ks.max() <= 5


def test_ego_jepa_dataset_masks_differ_from_clean():
    from diffusion_planner.model.jepa.data import EgoJEPADataset

    traj = torch.randn(40, 4) + 5.0  # nonzero so masking (zeroing) is detectable
    ds = EgoJEPADataset([traj], window=16, k_max=5, num_mask=3,
                        feature_mask_ratio=0.5, time_mask_ratio=0.3)
    ctx1, ctx2, _, _ = ds[0]
    # masking zeros entries -> at least one masked view differs from a full window
    assert (ctx1 == 0).any() or (ctx2 == 0).any()
