"""Strict indexed reader for current new-DP native H5 shards."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the zstd HDF5 filter
import numpy as np
import pyarrow.parquet as pq

from .schema import H5_FORMAT, H5_FORMAT_VERSION, MODEL_INPUT_NAMES


class H5FrameIndex:
    """Read frames addressed by their native H5 path and frame index."""

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
        self._by_frame: dict[tuple[str, int], int] = {}
        for i, row in enumerate(self.rows):
            path = self._resolve_h5_path(row["h5_path"])
            key = (str(path), int(row["frame_index"]))
            if key in self._by_frame:
                raise ValueError(f"Duplicate H5 frame in index: {key}")
            self._by_frame[key] = i

    def __len__(self) -> int:
        return len(self.rows)

    def index_for_frame(
        self,
        h5_path: str | Path,
        frame_index: int,
        frame_time_ns: int | None = None,
        *,
        relative_to: str | Path | None = None,
    ) -> int:
        path = self._resolve_h5_path(h5_path, relative_to=relative_to)
        key = (str(path), int(frame_index))
        try:
            index = self._by_frame[key]
        except KeyError as exc:
            raise KeyError(f"No indexed native H5 frame for {key}") from exc
        if frame_time_ns is not None and int(self.rows[index]["frame_time_ns"]) != int(
            frame_time_ns
        ):
            raise ValueError(
                f"frame_time_ns mismatch for {key}: JSON={frame_time_ns}, "
                f"index={self.rows[index]['frame_time_ns']}"
            )
        return index

    def _resolve_h5_path(self, value: str | Path, *, relative_to: str | Path | None = None) -> Path:
        path = Path(value)
        candidates = [path] if path.is_absolute() else []
        if not path.is_absolute():
            if relative_to is not None:
                candidates.append(Path(relative_to) / path)
            candidates.append(self.index_path.parent / path)

        # A packaged index may have been generated before its H5 collection
        # received its final name.  Collection-local ``group/file`` remains a
        # stable address, so resolve it against the index's collection root.
        # This supports the shipped ``open_loop_basic`` data without knowing
        # anything about legacy NPZ or rosbag layouts.
        if len(path.parts) >= 2:
            candidates.append(self.index_path.parent / path.parts[-2] / path.parts[-1])

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        listed = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Native H5 shard not found for {value}; tried: {listed}")

    def frame(self, index: int) -> dict[str, np.ndarray]:
        row = self.rows[index]
        path = self._resolve_h5_path(row["h5_path"])
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
        if "frames" not in file or "num_frames" not in file.attrs:
            file.close()
            raise ValueError(f"Incomplete native H5 shard: {path}")
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
