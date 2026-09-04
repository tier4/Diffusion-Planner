"""Strict indexed reader for new-DP native H5 shards."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the zstd HDF5 filter
import numpy as np
import pyarrow.parquet as pq

from .schema import H5_FORMAT, H5_FORMAT_VERSION, MODEL_INPUT_NAMES


class H5FrameIndex:
    """Read frames and resolve old matrix paths without converting their content."""

    def __init__(self, index_path: str | Path, file_capacity: int = 8) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        table = pq.read_table(self.index_path)
        required = {"h5_path", "frame_index", "frame_time_ns"}
        missing = required.difference(table.column_names)
        if missing:
            raise ValueError(f"H5 index missing columns: {sorted(missing)}")
        self.rows = table.to_pylist()
        self._files: OrderedDict[Path, h5py.File] = OrderedDict()
        self._capacity = file_capacity
        self._by_source: dict[str, int] = {}
        if "source_npz_path" in table.column_names:
            for i, row in enumerate(self.rows):
                key = str(Path(row["source_npz_path"]).resolve())
                if key in self._by_source:
                    raise ValueError(f"Duplicate source_npz_path in H5 index: {key}")
                self._by_source[key] = i

    def __len__(self) -> int:
        return len(self.rows)

    def index_for_source(self, source_path: str | Path) -> int:
        key = str(Path(source_path).expanduser().resolve())
        try:
            return self._by_source[key]
        except KeyError as exc:
            raise KeyError(f"No native H5 frame indexed for source: {key}") from exc

    def frame_for_source(self, source_path: str | Path) -> dict[str, np.ndarray]:
        return self.frame(self.index_for_source(source_path))

    def frame(self, index: int) -> dict[str, np.ndarray]:
        row = self.rows[index]
        path = Path(row["h5_path"])
        if not path.is_absolute():
            path = self.index_path.parent / path
        path = path.resolve()
        file = self._open(path)
        frame_index = int(row["frame_index"])
        if not 0 <= frame_index < int(file.attrs["num_frames"]):
            raise IndexError(f"frame_index {frame_index} outside {path}")
        frames = file["frames"]
        missing = sorted(set(MODEL_INPUT_NAMES).difference(frames.keys()))
        if missing:
            raise ValueError(f"H5 frame is missing native model fields: {missing} ({path})")
        result = {key: np.asarray(value[frame_index]) for key, value in frames.items()}
        neighbors = result["neighbor_agents_past"]
        if result["agent_shape"].shape != (neighbors.shape[0], 2):
            raise ValueError("agent_shape must match neighbor_agents_past slots")
        if result["agent_label"].shape != (neighbors.shape[0], 3):
            raise ValueError("agent_label must match neighbor_agents_past slots")
        if result["ego_agent_past"].shape[-1] != 6 or neighbors.shape[-1] != 4:
            raise ValueError("unexpected native ego/neighbor feature width")
        return result

    def _open(self, path: Path) -> h5py.File:
        cached = self._files.pop(path, None)
        if cached is not None:
            self._files[path] = cached
            return cached
        file = h5py.File(path, "r")
        if file.attrs.get("format") != H5_FORMAT:
            file.close()
            raise ValueError(f"Unexpected H5 format: {path}")
        if int(file.attrs.get("format_version", -1)) != H5_FORMAT_VERSION:
            file.close()
            raise ValueError(f"Unsupported H5 format version: {path}")
        self._files[path] = file
        while len(self._files) > self._capacity:
            self._files.popitem(last=False)[1].close()
        return file

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __enter__(self) -> "H5FrameIndex":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
