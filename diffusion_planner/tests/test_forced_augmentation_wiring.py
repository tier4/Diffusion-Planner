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

"""Wiring tests for forced augmentation: the force= contract and cursor alignment."""

import pytest
import torch
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.forced_augmentation import ForcedAugmentationSelector
from torch.utils.data import DataLoader, Dataset


def _perturbation(augment_prob: float) -> StatePerturbation:
    """StatePerturbation with every required argument supplied."""
    return StatePerturbation(
        augment_prob=augment_prob,
        num_refine=1,
        device="cpu",
        ego_past_noise_std=0.0,
        use_smoothing_future_trajectory=False,
    )


def _inputs(batch_size: int, speed: float = 10.0) -> dict:
    """Minimal ego_current_state batch. Column 4 is vx, which gates augmentation."""
    ego_current_state = torch.zeros(batch_size, 10)
    ego_current_state[:, 2] = 1.0  # cos(heading)
    ego_current_state[:, 4] = speed  # vx
    return {
        "ego_current_state": ego_current_state,
        "ego_shape": torch.full((batch_size, 3), 2.5),
    }


class TestForceParameterContract:
    def test_augment_accepts_force_and_defaults_to_none(self):
        aug = _perturbation(0.0)
        flag_default, _ = aug.augment(_inputs(4))
        assert flag_default.shape == (4,)
        assert not flag_default.any(), "prob=0 and no force must flag nothing"

    def test_force_flags_a_row_that_probability_would_not(self):
        """prob=0.0 means the Bernoulli mask is all-False; only force can flag."""
        aug = _perturbation(0.0)
        force = torch.tensor([True, False, True, False])
        flag, _ = aug.augment(_inputs(4), force=force)
        assert flag.tolist() == [True, False, True, False]

    def test_force_cannot_bypass_the_low_speed_validity_gate(self):
        """Forcing raises the chance of augmenting, never of an invalid scene."""
        aug = _perturbation(0.0)
        force = torch.tensor([True, True])
        flag, _ = aug.augment(_inputs(2, speed=0.5), force=force)
        assert not flag.any(), "|vx| < 2.0 must still veto a forced row"

    def test_prob_one_without_force_is_unchanged(self):
        aug = _perturbation(1.0)
        flag, _ = aug.augment(_inputs(4))
        assert flag.all()

    def test_forced_noop_count_records_vetoed_rows(self):
        aug = _perturbation(0.0)
        force = torch.tensor([True, True])
        aug.augment(_inputs(2, speed=0.5), force=force)
        assert aug.pop_forced_noop_count() == 2

    def test_forced_noop_count_zero_when_force_takes_effect(self):
        aug = _perturbation(0.0)
        aug.augment(_inputs(2, speed=10.0), force=torch.tensor([True, True]))
        assert aug.pop_forced_noop_count() == 0

    def test_pop_forced_noop_count_resets(self):
        aug = _perturbation(0.0)
        aug.augment(_inputs(2, speed=0.5), force=torch.tensor([True, True]))
        aug.pop_forced_noop_count()
        assert aug.pop_forced_noop_count() == 0

    def test_pop_forced_noop_count_is_zero_without_forcing(self):
        aug = _perturbation(1.0)
        aug.augment(_inputs(2))
        assert aug.pop_forced_noop_count() == 0


class _IndexDataset(Dataset):
    """Returns its own index so a test can prove flag-to-sample alignment."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"idx": torch.tensor(idx, dtype=torch.long)}


class _RecordingSampler:
    """Deterministic index stream with a known repeat pattern."""

    def __init__(self, indices):
        self.indices = list(indices)
        seen, flags = set(), []
        for idx in self.indices:
            flags.append(idx in seen)
            seen.add(idx)
        self.repeat_flags = flags
        self.repeat_flags_epoch = 0

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TestCursorAlignsWithDataLoaderOrder:
    """The cursor is positional, so alignment is the thing most worth proving."""

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_force_mask_lands_on_the_repeated_samples(self, num_workers):
        indices = [3, 1, 3, 0, 1, 1, 2, 0]
        sampler = _RecordingSampler(indices)
        loader = DataLoader(
            _IndexDataset(4),
            sampler=sampler,
            batch_size=2,
            num_workers=num_workers,
            drop_last=True,
        )
        selector = ForcedAugmentationSelector(["neighbor_noise"], seed=0)
        selector.start_epoch(0, sampler.repeat_flags, sampler.repeat_flags_epoch)

        forced_indices, delivered = [], []
        for batch in loader:
            size = batch["idx"].shape[0]
            masks = selector.masks_for_batch(size, torch.device("cpu"))
            for row in range(size):
                delivered.append(int(batch["idx"][row]))
                if masks["neighbor_noise"][row]:
                    forced_indices.append(int(batch["idx"][row]))

        assert delivered == indices, "DataLoader must preserve sampler order"
        expected = [idx for idx, flag in zip(indices, sampler.repeat_flags) if flag]
        assert forced_indices == expected
