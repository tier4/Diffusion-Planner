"""Shard-backed counterpart of :class:`PlannerDataset`.

Frames are read from a versioned tar-shard dataset (see ``planner_shards``) instead of per-bag
H5 files. Each training rank iterates only its own share of the key-set, so the resulting
``DataLoader`` must **not** be passed through ``accelerator.prepare`` (that would shard it a
second time); ``scripts/train/train.py`` detects this dataset and handles it accordingly.

The same frame transforms as :class:`PlannerDataset` are applied, in the same order, to the
decoded arrays before they become tensors.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from planner_shards.shard_dataset import ShardDataset, ShardDatasetConfig

from .transforms import Transform


class ShardPlannerDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Iterate one rank's share of a packed shard dataset as transformed planner frames."""

    def __init__(
        self,
        root: str | Path,
        version: str,
        keyset_path: str | Path,
        *,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
        world_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        chunk_size: int = 256,
        max_pad_fraction: float = 0.01,
        transforms: Sequence[Transform] = (),
    ) -> None:
        """Plan this rank's iteration; rank and world size default to the torchrun environment."""
        world = (
            int(os.environ.get("WORLD_SIZE", "1"))
            if world_size is None
            else int(world_size)
        )
        this_rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1: {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must not be negative: {num_workers}")
        self.config = ShardDatasetConfig(
            root=Path(root).expanduser().resolve(),
            version=str(version),
            keyset_path=Path(keyset_path).expanduser().resolve(),
            batch_size=int(batch_size),
            world_size=world,
            rank=this_rank,
            num_workers=int(num_workers),
            seed=int(seed),
            shuffle=bool(shuffle),
            chunk_size=int(chunk_size),
            max_pad_fraction=float(max_pad_fraction),
        )
        self._inner = ShardDataset(self.config)
        self._transforms = tuple(transforms)

    def __len__(self) -> int:
        """Samples this rank iterates per epoch (after the plan's padding)."""
        return len(self._inner)

    @property
    def steps_per_epoch(self) -> int:
        return self._inner.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Reseed the per-epoch shuffle; call once per epoch before iterating."""
        self._inner.set_epoch(epoch)

    def run_record(self) -> dict[str, Any]:
        """Dataset version hash, key-set digest and plan parameters for run metadata."""
        return self._inner.run_record()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for arrays in self._inner:
            # Decoded arrays are read-only views over the member payload; transforms expect
            # ordinary writable arrays, like the ones h5py returns.
            frame: dict[str, Any] = {
                key: np.array(value) for key, value in arrays.items()
            }
            for transform in self._transforms:
                frame = transform(frame)
            yield {key: torch.from_numpy(value) for key, value in frame.items()}


def build_shard_dataloader(
    dataset: ShardPlannerDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
    drop_last: bool = True,
) -> DataLoader:
    """Wrap the shard dataset in a DataLoader with the settings its plan was made for."""
    if batch_size != dataset.config.batch_size:
        raise ValueError(
            f"batch_size {batch_size} differs from the dataset plan ({dataset.config.batch_size})"
        )
    if num_workers != dataset.config.num_workers:
        raise ValueError(
            f"num_workers {num_workers} differs from the dataset plan ({dataset.config.num_workers})"
        )
    if not drop_last:
        raise ValueError(
            "shard loading yields whole batches per worker; drop_last must be true"
        )
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **kwargs,
    )
