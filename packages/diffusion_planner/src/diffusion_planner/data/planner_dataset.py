"""Torch dataset loading preprocessed diffusion-planner frames from H5 shards."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from .transforms import Transform

REQUIRED_INDEX_COLUMNS = frozenset({"h5_path", "frame_index", "frame_time_ns"})
H5_FORMAT = "diffusion_planner_frame_dataset"
H5_FORMAT_VERSION = 4

hdf5plugin.register(filters="zstd")


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

        table = pq.read_table(self._index_path)
        missing = sorted(REQUIRED_INDEX_COLUMNS.difference(table.column_names))
        if missing:
            raise ValueError(f"Missing required Parquet columns: {', '.join(missing)}")
        if table.num_rows == 0:
            raise ValueError(f"Frame index is empty: {self._index_path}")

        relative_h5_paths = _column(table, "h5_path")
        self._h5_paths = np.asarray(
            [
                str((self._index_path.parent / str(value)).resolve())
                for value in relative_h5_paths
            ],
            dtype=np.str_,
        )
        self._frame_indices = _column(table, "frame_index").astype(np.int64, copy=False)
        self._frame_times_ns = _column(table, "frame_time_ns").astype(
            np.int64, copy=False
        )
        if np.any(self._frame_indices < 0):
            raise ValueError("Parquet index contains a negative frame_index")

        self._file_capacity = file_capacity
        self._transforms = tuple(transforms)
        self._files: OrderedDict[Path, h5py.File] = OrderedDict()
        self._frame_keys: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self._frame_indices)

    def source(self, index: int) -> tuple[str, int]:
        """Return the H5 path and source frame timestamp for diagnostics."""
        return str(self._resolve_h5_path(index)), int(self._frame_times_ns[index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load, transform, and convert one H5 frame to tensors."""
        path = self._resolve_h5_path(index)
        file = self._file_for(path)
        frame_index = int(self._frame_indices[index])
        frames_object = file["frames"]
        if not isinstance(frames_object, h5py.Group):
            raise ValueError(f"H5 'frames' must be a group: {path}")
        frames = frames_object

        num_frames_value = file.attrs["num_frames"]
        if not isinstance(num_frames_value, (int, np.integer)):
            raise ValueError(f"H5 'num_frames' must be an integer: {path}")
        num_frames = int(num_frames_value)
        if frame_index >= num_frames:
            raise IndexError(
                f"frame_index {frame_index} is outside {path} with {num_frames} frames"
            )
        keys = self._frame_keys
        if keys is None:
            keys = tuple(sorted(frames.keys()))
            if not keys:
                raise ValueError(f"H5 frames group is empty: {path}")
            self._frame_keys = keys
        current_keys = tuple(sorted(frames.keys()))
        if current_keys != keys:
            raise ValueError(
                f"H5 tensor schema differs from the first opened shard: {path}"
            )
        frame_arrays: dict[str, NDArray[Any]] = {}
        for key in keys:
            dataset = frames[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"H5 'frames/{key}' must be a dataset: {path}")
            frame_arrays[key] = np.asarray(dataset[frame_index])
        for transform in self._transforms:
            frame_arrays = transform(frame_arrays)
        return {key: torch.from_numpy(value) for key, value in frame_arrays.items()}

    def _resolve_h5_path(self, index: int) -> Path:
        return Path(str(self._h5_paths[index]))

    def _file_for(self, path: Path) -> h5py.File:
        file = self._files.pop(path, None)
        if file is not None:
            self._files[path] = file
            return file
        if not path.is_file():
            raise FileNotFoundError(f"H5 shard not found: {path}")
        file = h5py.File(path, "r")
        try:
            if file.attrs.get("format") != H5_FORMAT:
                raise ValueError(f"Unexpected H5 format: {path}")
            if int(file.attrs.get("format_version", -1)) != H5_FORMAT_VERSION:
                raise ValueError(f"Unsupported H5 format version: {path}")
            if "frames" not in file or "num_frames" not in file.attrs:
                raise ValueError(f"Incomplete H5 shard: {path}")
        except BaseException:
            file.close()
            raise
        self._files[path] = file
        while len(self._files) > self._file_capacity:
            _, evicted = self._files.popitem(last=False)
            evicted.close()
        return file

    def close(self) -> None:
        """Close every H5 handle opened in this process."""
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Do not serialize HDF5 handles into DataLoader worker processes."""
        return {**self.__dict__, "_files": OrderedDict(), "_frame_keys": None}

    def __del__(self) -> None:
        files = getattr(self, "_files", None)
        if files is not None:
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
