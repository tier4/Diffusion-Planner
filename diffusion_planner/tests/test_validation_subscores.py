# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the validation-loop reward-subscore wiring.

`validate_model` logs the RLVR reward subscores as additive metrics by calling
`planner_metrics.compute_subscores_batch` per scene (it is single-scene /
N-trajectory). This pins the per-scene batched helper against a direct
single-scene call so the batched slicing stays correct.
"""

from __future__ import annotations

import torch
from diffusion_planner.validate_model import _VAL_SUBSCORE_KEYS, _reward_subscores_per_scene

from planner_metrics import RewardConfig, compute_subscores_batch

T = 80


def _ego(speed: float) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32)
    return torch.stack([t * speed, torch.zeros(T), torch.ones(T), torch.zeros(T)], dim=-1)


def _lanes() -> torch.Tensor:
    lanes = torch.zeros(140, 20, 33)
    for seg in range(10):
        for pt in range(20):
            x = (seg * 20 + pt) * 1.0
            lanes[seg, pt, 0] = x
            lanes[seg, pt, 2] = 1.0
            lanes[seg, pt, 4] = x
            lanes[seg, pt, 5] = 1.75
            lanes[seg, pt, 6] = x
            lanes[seg, pt, 7] = -1.75
    return lanes


def _batched_scene(b: int = 2) -> tuple[torch.Tensor, dict]:
    ego_pred = torch.stack([_ego(0.5), _ego(0.3)][:b])  # (b, T, 4)
    data_batched = {
        "ego_shape": torch.tensor([[2.79, 4.34, 1.70]] * b),
        "neighbor_agents_future": torch.zeros(b, 1, T, 4),
        "neighbor_agents_past": torch.zeros(b, 1, 21, 11),
        "lanes": torch.stack([_lanes()] * b),
        "goal_pose": torch.tensor([[100.0, 0.0, 1.0, 0.0]] * b),
    }
    return ego_pred, data_batched


def test_helper_shapes_and_finite() -> None:
    ego_pred, data_batched = _batched_scene(2)
    out = _reward_subscores_per_scene(ego_pred, data_batched, RewardConfig(), _VAL_SUBSCORE_KEYS)
    assert set(out) == set(_VAL_SUBSCORE_KEYS)
    for name in _VAL_SUBSCORE_KEYS:
        assert out[name].shape == (2,), f"{name}: {tuple(out[name].shape)}"
        assert torch.isfinite(out[name]).all(), f"{name} not finite"


def test_per_scene_matches_single_scene() -> None:
    """Per-scene batched slicing == calling compute_subscores_batch on each scene."""
    ego_pred, data_batched = _batched_scene(2)
    out = _reward_subscores_per_scene(ego_pred, data_batched, RewardConfig(), _VAL_SUBSCORE_KEYS)
    for b in range(ego_pred.shape[0]):
        data_b = {k: v[b : b + 1] for k, v in data_batched.items()}
        subs_b = compute_subscores_batch(ego_pred[b : b + 1], data_b, RewardConfig())
        for name in _VAL_SUBSCORE_KEYS:
            assert float(out[name][b]) == float(subs_b[name][0]), f"scene {b} {name} mismatch"
