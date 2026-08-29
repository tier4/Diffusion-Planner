"""H5 and Parquet frame-source loading independent of the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

H5_FORMAT = "diffusion_planner_frame_dataset"
H5_FORMAT_VERSION = 4
PARQUET_REQUIRED_COLUMNS = frozenset({"h5_path", "frame_index", "frame_time_ns"})

hdf5plugin.register(filters="zstd")
STAT_COLUMNS = (
    "ego_speed_mps",
    "ego_yaw_rate_rps",
    "turn_indicator",
    "num_objects",
)


@dataclass(frozen=True)
class FrameIndexRow:
    """One resolved H5 frame selected directly or through Parquet."""

    index: int
    h5_path: str
    frame_index: int
    frame_time_ns: int
    stats: dict[str, object]


@dataclass(frozen=True)
class FrameIndex:
    """In-memory columns needed to browse one H5 file or a Parquet index."""

    path: Path
    h5_paths: NDArray[np.str_]
    frame_indices: NDArray[np.int64]
    frame_times_ns: NDArray[np.int64]
    stats: dict[str, NDArray[np.generic]]

    def __len__(self) -> int:
        return len(self.frame_indices)

    @property
    def sources(self) -> tuple[str, ...]:
        """Return unique H5 paths in first-occurrence order."""
        return tuple(dict.fromkeys(self.h5_paths.tolist()))

    def indices_for_source(self, h5_path: str | None) -> NDArray[np.int64]:
        """Return row indices, optionally filtered by H5 shard."""
        if h5_path is None:
            return np.arange(len(self), dtype=np.int64)
        return np.flatnonzero(self.h5_paths == h5_path).astype(np.int64, copy=False)

    def row(self, index: int) -> FrameIndexRow:
        """Return one row by its absolute index."""
        if not 0 <= index < len(self):
            raise IndexError(
                f"Frame index {index} is out of range for {len(self)} rows"
            )
        return FrameIndexRow(
            index=index,
            h5_path=str(self.h5_paths[index]),
            frame_index=int(self.frame_indices[index]),
            frame_time_ns=int(self.frame_times_ns[index]),
            stats={key: values[index].item() for key, values in self.stats.items()},
        )


def load_frame_index(path: str | Path) -> FrameIndex:
    """Load a direct H5 source or a Parquet H5 frame index."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Data source not found: {source_path}")
    if source_path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_h5_index(source_path)
    if source_path.suffix.lower() == ".parquet":
        return _load_parquet_index(source_path)
    raise ValueError(f"Data source must be an H5 or Parquet file: {source_path}")


def _load_h5_index(path: Path) -> FrameIndex:
    """Construct a frame index directly from one H5 shard."""
    with h5py.File(path, "r") as file:
        _validate_h5(file, path)
        num_frames_value = file.attrs["num_frames"]
        if not isinstance(num_frames_value, (int, np.integer)):
            raise ValueError(f"H5 num_frames must be an integer: {path}")
        num_frames = int(num_frames_value)
        if num_frames == 0:
            raise ValueError(f"H5 frame source is empty: {path}")
        metadata = file["metadata"]
        if not isinstance(metadata, h5py.Group):
            raise ValueError(f"H5 metadata must be a group: {path}")
        if "frame_time_ns" not in metadata:
            raise ValueError(f"H5 metadata/frame_time_ns is missing: {path}")
        frame_time_values = metadata["frame_time_ns"]
        if not isinstance(frame_time_values, h5py.Dataset):
            raise ValueError(f"H5 metadata/frame_time_ns must be a dataset: {path}")
        frame_times = np.asarray(frame_time_values[...], dtype=np.int64)
        if len(frame_times) != num_frames:
            raise ValueError(f"H5 frame time count differs from num_frames: {path}")
        stats: dict[str, NDArray[np.generic]] = {}
        for name in STAT_COLUMNS:
            if name not in metadata:
                continue
            values = metadata[name]
            if not isinstance(values, h5py.Dataset):
                raise ValueError(f"H5 metadata/{name} must be a dataset: {path}")
            stats[name] = np.asarray(values[...])
    return FrameIndex(
        path=path,
        h5_paths=np.asarray([str(path)] * num_frames, dtype=np.str_),
        frame_indices=np.arange(num_frames, dtype=np.int64),
        frame_times_ns=frame_times,
        stats=stats,
    )


def _load_parquet_index(path: Path) -> FrameIndex:
    """Load a generated Parquet index containing relative H5 paths."""
    parquet_file = pq.ParquetFile(path)
    column_names = set(parquet_file.schema_arrow.names)
    missing = sorted(PARQUET_REQUIRED_COLUMNS.difference(column_names))
    if missing:
        raise ValueError(f"Missing required Parquet columns: {', '.join(missing)}")
    selected = [
        *sorted(PARQUET_REQUIRED_COLUMNS),
        *(name for name in STAT_COLUMNS if name in column_names),
    ]
    table = parquet_file.read(columns=selected)
    if table.num_rows == 0:
        raise ValueError(f"Parquet frame index is empty: {path}")

    def column(name: str) -> NDArray[np.generic]:
        return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))

    raw_paths = column("h5_path").astype(np.str_)
    resolved_paths = np.asarray(
        [(path.parent / str(value)).resolve() for value in raw_paths],
        dtype=np.str_,
    )
    frame_indices = column("frame_index").astype(np.int64, copy=False)
    if np.any(frame_indices < 0):
        raise ValueError(f"Parquet index contains a negative frame_index: {path}")
    return FrameIndex(
        path=path,
        h5_paths=resolved_paths,
        frame_indices=frame_indices,
        frame_times_ns=column("frame_time_ns").astype(np.int64, copy=False),
        stats={
            name: column(name) for name in STAT_COLUMNS if name in table.column_names
        },
    )


def _validate_h5(file: h5py.File, path: Path) -> None:
    if file.attrs.get("format") != H5_FORMAT:
        raise ValueError(f"Unexpected H5 format: {path}")
    if int(file.attrs.get("format_version", -1)) != H5_FORMAT_VERSION:
        raise ValueError(f"Unsupported H5 format version: {path}")
    if "frames" not in file or "metadata" not in file or "num_frames" not in file.attrs:
        raise ValueError(f"Incomplete H5 frame source: {path}")
