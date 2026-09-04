"""Parity test: `_moving_diagnostics_batch` vs the scalar `_moving_diagnostics`.

The batched path replaced per-candidate batch-1 clearance passes in
`classify_loaded_scene_candidates_batch`; the scalar function is kept as the
reference implementation and this test pins the two to identical outputs on
randomized scenes (including zero-neighbor and stopped-neighbor cases).
"""

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from rlvr.autoresearch.tools.classify_scene_failures import (
    _moving_diagnostics,
    _moving_diagnostics_batch,
)


def _random_scene(rng: np.random.Generator, n_nb: int, T: int) -> dict[str, torch.Tensor]:
    nf = np.zeros((max(n_nb, 1), T, 4), dtype=np.float32)
    for i in range(n_nb):
        speed = rng.uniform(0.0, 8.0) if rng.random() > 0.3 else 0.0  # some stopped
        yaw = rng.uniform(-np.pi, np.pi)
        start = rng.uniform(-15, 15, size=2)
        steps = start + np.outer(np.arange(T), speed * 0.1 * np.array([np.cos(yaw), np.sin(yaw)]))
        nf[i, :, 0] = steps[:, 0]
        nf[i, :, 1] = steps[:, 1]
        nf[i, :, 2] = np.cos(yaw)
        nf[i, :, 3] = np.sin(yaw)
    return {
        "neighbor_agents_future": torch.from_numpy(nf[None]),
        "neighbor_agents_past": torch.zeros(1, max(n_nb, 1), 3, 11),
        "ego_shape": torch.tensor([[3.0, 4.5, 2.0]], dtype=torch.float32),
    }


def _random_candidates(rng: np.random.Generator, N: int, T: int) -> torch.Tensor:
    trajs = np.zeros((N, T, 4), dtype=np.float32)
    for n in range(N):
        speed = rng.uniform(0.5, 8.0)
        trajs[n, :, 0] = speed * 0.1 * np.arange(1, T + 1)
        trajs[n, :, 1] = rng.uniform(-0.5, 0.5)
        trajs[n, :, 2] = 1.0
    return torch.from_numpy(trajs)


def test_moving_diagnostics_batch_matches_scalar():
    rng = np.random.default_rng(0)
    device = torch.device("cpu")
    config = RewardConfig()
    for trial, n_nb in enumerate([0, 1, 3, 8, 20]):
        T, N = 20, 5
        data = _random_scene(rng, n_nb, T)
        trajs = _random_candidates(rng, N, T)
        batch_rows = _moving_diagnostics_batch(trajs, data, config, 0.2, 0.7, device)
        assert len(batch_rows) == N
        for n in range(N):
            scalar_row = _moving_diagnostics(trajs[n : n + 1], data, config, 0.2, 0.7, device)
            assert batch_rows[n] == scalar_row, (
                f"trial {trial} (n_nb={n_nb}) candidate {n}: {batch_rows[n]} != {scalar_row}"
            )


def test_moving_diagnostics_batch_rear_end_gate():
    """Rear-end suppression path (ignore_rear_end_collisions=True) also matches."""
    rng = np.random.default_rng(1)
    device = torch.device("cpu")
    config = RewardConfig(ignore_rear_end_collisions=True)
    data = _random_scene(rng, 6, 20)
    # Force one neighbor to sit right behind the ego path (rear-end geometry).
    nf = data["neighbor_agents_future"][0]
    nf[0, :, 0] = -1.0
    nf[0, :, 1] = 0.0
    trajs = _random_candidates(rng, 4, 20)
    batch_rows = _moving_diagnostics_batch(trajs, data, config, 0.2, 0.7, device)
    for n in range(4):
        scalar_row = _moving_diagnostics(trajs[n : n + 1], data, config, 0.2, 0.7, device)
        assert batch_rows[n] == scalar_row
