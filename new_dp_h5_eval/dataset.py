"""Strict indexed reader for current new-DP native H5 shards."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the zstd HDF5 filter
import numpy as np
import pyarrow.parquet as pq

from .h5_frames import NativeFrames
from .schema import H5_FORMAT, H5_MIN_FORMAT_VERSION, MODEL_INPUT_NAMES, EgoPastLayout


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
        self._files: OrderedDict[Path, tuple[h5py.File, NativeFrames]] = OrderedDict()
        self._capacity = file_capacity
        self._by_frame: dict[tuple[str, int], int] = {}
        self._frames_by_shard: dict[Path, list[int]] = {}
        for i, row in enumerate(self.rows):
            path = self._resolve_h5_path(row["h5_path"])
            self._frames_by_shard.setdefault(path, []).append(int(row["frame_index"]))
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
        file, frames = self._open(path)
        frame_index = int(row["frame_index"])
        if not 0 <= frame_index < int(file.attrs["num_frames"]):
            raise IndexError(f"frame_index {frame_index} outside {path}")
        missing = sorted(set(MODEL_INPUT_NAMES).difference(frames.keys()))
        if missing:
            raise ValueError(f"H5 frame is missing native model fields: {missing} ({path})")
        result = frames.row(frame_index)
        neighbors = result["neighbor_agents_past"]
        if result["agent_shape"].shape != (neighbors.shape[0], 2):
            raise ValueError("agent_shape must match neighbor_agents_past slots")
        if result["agent_label"].shape != (neighbors.shape[0], 3):
            raise ValueError("agent_label must match neighbor_agents_past slots")
        EgoPastLayout.of(result["ego_agent_past"].shape[-1])
        if neighbors.shape[-1] != 4:
            raise ValueError("unexpected native neighbor feature width")
        return result

    def shard_path(self, index: int) -> Path:
        """Resolved path of the shard holding row ``index``."""
        return self._resolve_h5_path(self.rows[index]["h5_path"])

    def project_id(self, path: Path) -> str:
        """``project_id`` attribute of a shard returned by ``shard_path``."""
        project_id = self._open(path)[0].attrs.get("project_id")
        if isinstance(project_id, bytes):
            project_id = project_id.decode()
        if not project_id:
            raise ValueError(f"native H5 shard has no project_id attribute: {path}")
        return str(project_id)

    def ego_shapes(self, path: Path) -> np.ndarray:
        """Native ``ego_shape`` rows of a shard returned by ``shard_path``.

        Every frame of a ``frames/`` shard; of a packed shard, only the indexed frames, since
        reading one packed tensor reads the whole row.
        """
        frames = self._open(path)[1]
        if not frames.packed:
            return frames.column("ego_shape")
        rows = sorted(self._frames_by_shard[path])
        return np.stack([frames.row(index)["ego_shape"] for index in rows])

    def _open(self, path: Path) -> tuple[h5py.File, NativeFrames]:
        cached = self._files.pop(path, None)
        if cached is not None:
            self._files[path] = cached
            return cached
        file = h5py.File(path, "r")
        if file.attrs.get("format") != H5_FORMAT:
            file.close()
            raise ValueError(f"Unexpected H5 format: {path}")
        if int(file.attrs.get("format_version", -1)) < H5_MIN_FORMAT_VERSION:
            file.close()
            raise ValueError(f"Unsupported H5 format version: {path}")
        try:
            if "num_frames" not in file.attrs:
                raise ValueError(f"Incomplete native H5 shard: {path}")
            opened = (file, NativeFrames(file))
        except BaseException:
            file.close()
            raise
        self._files[path] = opened
        while len(self._files) > self._capacity:
            self._files.popitem(last=False)[1][0].close()
        return opened

    def close(self) -> None:
        for file, _ in self._files.values():
            file.close()
        self._files.clear()

    def __enter__(self) -> "H5FrameIndex":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
