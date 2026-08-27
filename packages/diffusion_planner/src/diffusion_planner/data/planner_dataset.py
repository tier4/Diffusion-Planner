"""Stream preprocessed diffusion-planner frames from H5 shards."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .data_augmentation import PlannerDataAugmentation
from .normalization import PlannerDataNormalizer
from .traffic_light import fill_unknown_traffic_light_futures

REQUIRED_INDEX_COLUMNS = frozenset({"h5_path", "frame_index", "frame_time_ns"})
H5_FORMAT = "diffusion_planner_frame_dataset"
H5_FORMAT_VERSION = 4

hdf5plugin.register(filters="zstd")


class PlannerDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream model inputs and labels sequentially from shuffled H5 shards.

    Shards are shuffled once per epoch, then the resulting virtual frame stream is
    divided equally among all distributed workers. A split is allowed only at a
    stream boundary, so every worker emits exactly the same batch-aligned number of
    samples even when shard sizes differ. Because every frame is an independent
    HDF5 chunk, frames within an open shard are read in a random order. A bounded
    in-memory buffer additionally mixes samples across shard boundaries.

    ``__getitem__`` is intentionally retained for dataset inspection and export
    tools. PyTorch DataLoader uses ``__iter__`` because this is an IterableDataset.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        shuffle_buffer_size: int = 4096,
        seed: int = 0,
        data_augmentation: PlannerDataAugmentation | None = None,
        data_normalizer: PlannerDataNormalizer | None = None,
    ) -> None:
        """Load the lightweight frame index and build the shard stream table."""
        self._index_path = Path(parquet_path).expanduser().resolve()
        if not self._index_path.is_file():
            raise FileNotFoundError(f"Parquet index not found: {self._index_path}")
        if shuffle_buffer_size < 1:
            raise ValueError(
                f"shuffle_buffer_size must be at least 1: {shuffle_buffer_size}"
            )

        table = pq.read_table(self._index_path)
        missing = sorted(REQUIRED_INDEX_COLUMNS.difference(table.column_names))
        if missing:
            raise ValueError(f"Missing required Parquet columns: {', '.join(missing)}")
        if table.num_rows == 0:
            raise ValueError(f"Frame index is empty: {self._index_path}")

        self._h5_paths = _column(table, "h5_path")
        relative_paths = [
            str(value) for value in self._h5_paths if not Path(str(value)).is_absolute()
        ]
        if relative_paths:
            raise ValueError(f"Parquet h5_path must be absolute: {relative_paths[0]}")
        self._frame_indices = _column(table, "frame_index").astype(np.int64, copy=False)
        self._frame_times_ns = _column(table, "frame_time_ns").astype(
            np.int64, copy=False
        )
        if np.any(self._frame_indices < 0):
            raise ValueError("Parquet index contains a negative frame_index")

        shard_rows: dict[Path, list[tuple[int, int]]] = {}
        for row, (path_value, frame_index) in enumerate(
            zip(self._h5_paths, self._frame_indices, strict=True)
        ):
            path = Path(str(path_value))
            shard_rows.setdefault(path, []).append((int(frame_index), row))
        self._shards = tuple(
            (
                path,
                np.asarray(
                    [row for _, row in sorted(rows)],
                    dtype=np.int64,
                ),
            )
            for path, rows in shard_rows.items()
        )

        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._data_augmentation = data_augmentation
        self._data_normalizer = data_normalizer
        self._epoch = mp.Value("q", 0)
        self._batch_size = 1
        self._configured_num_workers = 1
        self._frame_keys: tuple[str, ...] | None = None

    def __len__(self) -> int:
        """Return the batch-aligned number of samples emitted by this rank."""
        world_size, _ = _distributed_info()
        denominator = world_size * self._configured_num_workers * self._batch_size
        batches_per_worker = len(self._frame_indices) // denominator
        return batches_per_worker * self._configured_num_workers * self._batch_size

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shard permutation used by the next iteration."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative: {epoch}")
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Read and mix the exact global frame interval assigned to this worker."""
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        if num_workers != self._configured_num_workers:
            raise RuntimeError(
                "PlannerDataset worker count changed after DataLoader construction: "
                f"configured={self._configured_num_workers}, actual={num_workers}"
            )

        world_size, rank = _distributed_info()
        stream_id = rank * num_workers + worker_id
        num_streams = world_size * num_workers
        target = (
            len(self._frame_indices)
            // (num_streams * self._batch_size)
            * self._batch_size
        )
        if target == 0:
            return

        epoch = self._epoch.value
        shard_rng = np.random.default_rng(self._seed + epoch)
        shard_order = shard_rng.permutation(len(self._shards))
        stream_start = stream_id * target
        shard_slices = self._slice_stream(
            shard_order, stream_start, stream_start + target
        )
        sample_rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, epoch, stream_id])
        )
        samples = self._shuffle_samples(
            self._read_shards(shard_slices, epoch), sample_rng
        )
        emitted = 0
        for sample in samples:
            yield sample
            emitted += 1
            if emitted == target:
                return
        raise RuntimeError(
            f"stream {stream_id} emitted {emitted} frames instead of {target}"
        )

    def _configure_streaming(self, *, batch_size: int, num_workers: int) -> None:
        """Record DataLoader dimensions needed for length and stream quotas."""
        self._batch_size = batch_size
        self._configured_num_workers = max(num_workers, 1)

    def _slice_stream(
        self,
        shard_order: NDArray[np.integer[Any]],
        start: int,
        stop: int,
    ) -> list[tuple[int, int, int]]:
        """Map a global frame interval to shard-local half-open intervals."""
        slices: list[tuple[int, int, int]] = []
        shard_start = 0
        for shard_value in shard_order:
            shard_index = int(shard_value)
            shard_size = len(self._shards[shard_index][1])
            shard_stop = shard_start + shard_size
            if shard_stop > start and shard_start < stop:
                local_start = max(start - shard_start, 0)
                local_stop = min(stop - shard_start, shard_size)
                slices.append((shard_index, local_start, local_stop))
            if shard_stop >= stop:
                break
            shard_start = shard_stop
        return slices

    def _read_shards(
        self,
        shard_slices: list[tuple[int, int, int]],
        epoch: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Open assigned shards and randomly read the assigned frame chunks."""
        expected_keys: tuple[str, ...] | None = None
        for shard_index, local_start, local_stop in shard_slices:
            path, rows = self._shards[shard_index]
            if not path.is_file():
                raise FileNotFoundError(f"H5 shard not found: {path}")
            with h5py.File(path, "r") as file:
                frames, num_frames, keys = _validate_h5(file, path)
                if expected_keys is None:
                    expected_keys = keys
                elif keys != expected_keys:
                    raise ValueError(
                        f"H5 tensor schema differs from the first opened shard: {path}"
                    )
                frame_rng = np.random.default_rng(
                    np.random.SeedSequence([self._seed, epoch, shard_index])
                )
                shuffled_rows = frame_rng.permutation(rows)
                for row_value in shuffled_rows[local_start:local_stop]:
                    row = int(row_value)
                    frame_index = int(self._frame_indices[row])
                    if frame_index >= num_frames:
                        raise IndexError(
                            f"frame_index {frame_index} is outside {path} with "
                            f"{num_frames} frames"
                        )
                    yield self._load_frame(frames, keys, frame_index, path)

    def _shuffle_samples(
        self,
        samples: Iterator[dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Shuffle a stream with the bounded replacement-buffer algorithm."""
        buffer: list[dict[str, torch.Tensor]] = []
        for sample in samples:
            if len(buffer) < self._shuffle_buffer_size:
                buffer.append(sample)
                continue
            slot = int(rng.integers(len(buffer)))
            yield buffer[slot]
            buffer[slot] = sample
        for index in rng.permutation(len(buffer)):
            yield buffer[int(index)]

    def source(self, index: int) -> tuple[str, int]:
        """Return the H5 path and source frame timestamp for diagnostics."""
        return str(self._resolve_h5_path(index)), int(self._frame_times_ns[index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one frame directly for inspection and export utilities."""
        path = self._resolve_h5_path(index)
        if not path.is_file():
            raise FileNotFoundError(f"H5 shard not found: {path}")
        with h5py.File(path, "r") as file:
            frames, num_frames, keys = _validate_h5(file, path)
            frame_index = int(self._frame_indices[index])
            if frame_index >= num_frames:
                raise IndexError(
                    f"frame_index {frame_index} is outside {path} with "
                    f"{num_frames} frames"
                )
            if self._frame_keys is None:
                self._frame_keys = keys
            elif keys != self._frame_keys:
                raise ValueError(
                    f"H5 tensor schema differs from the first opened shard: {path}"
                )
            return self._load_frame(frames, keys, frame_index, path)

    def _load_frame(
        self,
        frames: h5py.Group,
        keys: tuple[str, ...],
        frame_index: int,
        path: Path,
    ) -> dict[str, torch.Tensor]:
        frame_arrays: dict[str, NDArray[Any]] = {}
        for key in keys:
            dataset = frames[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"H5 'frames/{key}' must be a dataset: {path}")
            frame_arrays[key] = np.asarray(dataset[frame_index])
        if self._data_augmentation is not None:
            frame_arrays = self._data_augmentation(frame_arrays)
        frame_arrays = fill_unknown_traffic_light_futures(frame_arrays)
        if self._data_normalizer is not None:
            frame_arrays = self._data_normalizer(frame_arrays)
        return {key: torch.from_numpy(value) for key, value in frame_arrays.items()}

    def _resolve_h5_path(self, index: int) -> Path:
        return Path(str(self._h5_paths[index]))


def _validate_h5(
    file: h5py.File, path: Path
) -> tuple[h5py.Group, int, tuple[str, ...]]:
    """Validate a shard and return its frame group, size, and tensor keys."""
    if file.attrs.get("format") != H5_FORMAT:
        raise ValueError(f"Unexpected H5 format: {path}")
    if int(file.attrs.get("format_version", -1)) != H5_FORMAT_VERSION:
        raise ValueError(f"Unsupported H5 format version: {path}")
    frames_object = file.get("frames")
    if not isinstance(frames_object, h5py.Group):
        raise ValueError(f"H5 'frames' must be a group: {path}")
    num_frames_value = file.attrs.get("num_frames")
    if not isinstance(num_frames_value, (int, np.integer)):
        raise ValueError(f"H5 'num_frames' must be an integer: {path}")
    keys = tuple(sorted(frames_object.keys()))
    if not keys:
        raise ValueError(f"H5 frames group is empty: {path}")
    return frames_object, int(num_frames_value), keys


def _distributed_info() -> tuple[int, int]:
    """Return world size and rank before or after torch.distributed setup."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise RuntimeError(
            f"Invalid distributed environment: WORLD_SIZE={world_size}, RANK={rank}"
        )
    return world_size, rank


def _column(table: Any, name: str) -> NDArray[Any]:
    """Return one combined Parquet column as a NumPy array."""
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))


def build_dataloader(
    dataset: PlannerDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    **kwargs: Any,
) -> DataLoader:
    """Build a DataLoader for the dataset's already-shuffled sample stream."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1: {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative: {num_workers}")
    if not shuffle:
        raise ValueError("streaming PlannerDataset requires shuffle=True")
    if not bool(kwargs.get("drop_last", False)):
        raise ValueError("streaming PlannerDataset requires drop_last=True")
    dataset._configure_streaming(batch_size=batch_size, num_workers=num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        **kwargs,
    )
