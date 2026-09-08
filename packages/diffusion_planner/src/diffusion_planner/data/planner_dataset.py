"""Torch dataset loading preprocessed diffusion-planner frames from H5 shards."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import hdf5plugin
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from .transforms import Transform

REQUIRED_INDEX_COLUMNS = frozenset({"h5_path", "frame_index", "frame_time_ns"})
H5_FORMAT = "diffusion_planner_frame_dataset"
H5_FORMAT_VERSION = 4
# One chunk holds one frame, so this caches the largest single frame slice per
# dataset. Zero measurably slows the filtered read path; more is never reused
# because each frame is read at most once per epoch.
CHUNK_CACHE_BYTES = 512 * 1024

hdf5plugin.register(filters="zstd")


class _Shard(NamedTuple):
    """One opened H5 shard whose frame datasets are resolved once per open."""

    file: h5py.File
    datasets: tuple[tuple[str, h5py.Dataset], ...]
    num_frames: int


class PlannerDataset(Dataset[dict[str, torch.Tensor]]):
    """Read preprocessed model inputs and labels through a Parquet H5 index."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        file_capacity: int = 8,
        transforms: Sequence[Transform] = (),
    ) -> None:
        """Load the lightweight index and defer H5 opens to DataLoader workers."""
        self._index_path = Path(parquet_path).expanduser().resolve()
        if not self._index_path.is_file():
            raise FileNotFoundError(f"Parquet index not found: {self._index_path}")
        if file_capacity < 1:
            raise ValueError(f"file_capacity must be at least 1: {file_capacity}")

        # The index also carries per-frame statistics this dataset never reads;
        # loading only the addressing columns keeps millions of rows small.
        schema = pq.read_schema(self._index_path)
        missing = sorted(REQUIRED_INDEX_COLUMNS.difference(schema.names))
        if missing:
            raise ValueError(f"Missing required Parquet columns: {', '.join(missing)}")
        table = pq.read_table(self._index_path, columns=sorted(REQUIRED_INDEX_COLUMNS))
        if table.num_rows == 0:
            raise ValueError(f"Frame index is empty: {self._index_path}")

        # Millions of rows address a few thousand shards, so resolve each shard
        # path once and keep only its id per row.
        shard_column = table["h5_path"].dictionary_encode()
        if isinstance(shard_column, pa.ChunkedArray):
            shard_column = shard_column.unify_dictionaries().combine_chunks()
        if not isinstance(shard_column, pa.DictionaryArray):
            raise ValueError(f"Cannot index h5_path by shard: {self._index_path}")
        if shard_column.indices.null_count:
            raise ValueError(
                f"Parquet index contains a null h5_path: {self._index_path}"
            )
        self._shard_paths = tuple(
            str((self._index_path.parent / str(value)).resolve())
            for value in shard_column.dictionary.to_pylist()
        )
        self._shard_ids = np.asarray(
            shard_column.indices.to_numpy(zero_copy_only=False), dtype=np.int32
        )
        self._frame_indices = _column(table, "frame_index").astype(np.int64, copy=False)
        self._frame_times_ns = _column(table, "frame_time_ns").astype(
            np.int64, copy=False
        )
        if np.any(self._frame_indices < 0):
            raise ValueError("Parquet index contains a negative frame_index")

        self._file_capacity = file_capacity
        self._transforms = tuple(transforms)
        self._shards: OrderedDict[int, _Shard] = OrderedDict()
        self._frame_keys: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self._frame_indices)

    def source(self, index: int) -> tuple[str, int]:
        """Return the H5 path and source frame timestamp for diagnostics."""
        return (
            self._shard_paths[self._shard_ids[index]],
            int(self._frame_times_ns[index]),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load, transform, and convert one H5 frame to tensors."""
        shard_id = int(self._shard_ids[index])
        shard = self._shard_for(shard_id)
        frame_index = int(self._frame_indices[index])
        if frame_index >= shard.num_frames:
            raise IndexError(
                f"frame_index {frame_index} is outside {self._shard_paths[shard_id]} "
                f"with {shard.num_frames} frames"
            )
        frame_arrays: dict[str, NDArray[Any]] = {
            key: np.asarray(dataset[frame_index]) for key, dataset in shard.datasets
        }
        for transform in self._transforms:
            frame_arrays = transform(frame_arrays)
        return {key: torch.from_numpy(value) for key, value in frame_arrays.items()}

    def _shard_for(self, shard_id: int) -> _Shard:
        """Return a cached shard, opening and validating it on a cache miss."""
        shard = self._shards.pop(shard_id, None)
        if shard is not None:
            self._shards[shard_id] = shard
            return shard
        shard = self._open_shard(Path(self._shard_paths[shard_id]))
        self._shards[shard_id] = shard
        while len(self._shards) > self._file_capacity:
            _, evicted = self._shards.popitem(last=False)
            evicted.file.close()
        return shard

    def _open_shard(self, path: Path) -> _Shard:
        """Open one shard and resolve every frame dataset it must expose."""
        if not path.is_file():
            raise FileNotFoundError(f"H5 shard not found: {path}")
        file = h5py.File(path, "r", rdcc_nbytes=CHUNK_CACHE_BYTES)
        try:
            if file.attrs.get("format") != H5_FORMAT:
                raise ValueError(f"Unexpected H5 format: {path}")
            if int(file.attrs.get("format_version", -1)) != H5_FORMAT_VERSION:
                raise ValueError(f"Unsupported H5 format version: {path}")
            if "frames" not in file or "num_frames" not in file.attrs:
                raise ValueError(f"Incomplete H5 shard: {path}")
            num_frames = file.attrs["num_frames"]
            if not isinstance(num_frames, (int, np.integer)):
                raise ValueError(f"H5 'num_frames' must be an integer: {path}")
            frames = file["frames"]
            if not isinstance(frames, h5py.Group):
                raise ValueError(f"H5 'frames' must be a group: {path}")
            keys = tuple(sorted(frames.keys()))
            if not keys:
                raise ValueError(f"H5 frames group is empty: {path}")
            if self._frame_keys is None:
                self._frame_keys = keys
            elif keys != self._frame_keys:
                raise ValueError(
                    f"H5 tensor schema differs from the first opened shard: {path}"
                )
            datasets = []
            for key in keys:
                dataset = frames[key]
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"H5 'frames/{key}' must be a dataset: {path}")
                datasets.append((key, dataset))
        except BaseException:
            file.close()
            raise
        return _Shard(file, tuple(datasets), int(num_frames))

    def close(self) -> None:
        """Close every H5 handle opened in this process."""
        for shard in self._shards.values():
            shard.file.close()
        self._shards.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Do not serialize HDF5 handles into DataLoader worker processes."""
        return {**self.__dict__, "_shards": OrderedDict(), "_frame_keys": None}

    def __del__(self) -> None:
        shards = getattr(self, "_shards", None)
        if shards is not None:
            self.close()


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
    """Wrap the H5 dataset in a standard PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        **kwargs,
    )
