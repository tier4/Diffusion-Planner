"""Short ego-centric trajectory from world-frame live pose + history."""

from __future__ import annotations

import math

import numpy as np
import torch

from scenario_generation.transforms import _rotation_matrix


def ego_traj_ego_frame(
    live_pose: np.ndarray,
    ego_hist: np.ndarray,
    *,
    n_steps: int,
    device: str,
) -> torch.Tensor:
    """Ego-centric trajectory ``(1, n_steps, 4)`` from real ``ego_hist`` + ``live_pose``.

    Last step is always the origin with heading +x (matches the live-ego observation
    frame). Earlier steps are previous world poses rotated into that frame — same
    transform as rollout ``_live_ego_past`` — so finite-difference speed gates
    (e.g. red-light) see real displacement.
    """
    n = max(1, int(n_steps))
    live = np.asarray(live_pose, dtype=np.float64).reshape(-1)
    hist = np.asarray(ego_hist, dtype=np.float64)
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)
    nh = int(hist.shape[0])
    yaw = float(live[2])
    R = _rotation_matrix(yaw)
    exy = live[:2]

    traj = torch.zeros(1, n, 4, dtype=torch.float32, device=device)
    for i in range(n):
        # Window ending at current: pad by repeating oldest if hist is short.
        src_i = nh - n + i
        if src_i < 0:
            src_i = 0
        if i == n - 1:
            src = live
        else:
            src = hist[src_i]
        d = R @ (np.asarray(src[:2], dtype=np.float64) - exy)
        h = float(src[2] - yaw)
        traj[0, i, 0] = float(d[0])
        traj[0, i, 1] = float(d[1])
        traj[0, i, 2] = math.cos(h)
        traj[0, i, 3] = math.sin(h)
    # Exact current pose (kill float noise on the last step).
    traj[0, n - 1, 0] = 0.0
    traj[0, n - 1, 1] = 0.0
    traj[0, n - 1, 2] = 1.0
    traj[0, n - 1, 3] = 0.0
    return traj
