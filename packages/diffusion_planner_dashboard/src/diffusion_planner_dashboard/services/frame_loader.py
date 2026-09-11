"""Read preprocessed dashboard frames from H5 shards."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

from diffusion_planner.data.transforms import PlannerUnknownLabelAugmentation

from .frame_index import FrameIndexRow, _validate_h5

hdf5plugin.register(filters="zstd")

_widen_agent_label = PlannerUnknownLabelAugmentation()
"""Pad-only at its default probability, matching what training applies on load.

Shards store a three-column ``agent_label``; the fourth unknown column is
appended when the frame is read. Without this the dashboard hands a three-column
label to a four-class model and every recorded frame fails to run.
"""


class FrameLoader:
    """Keep a small LRU cache of read-only H5 handles across frame selections."""

    def __init__(self, file_capacity: int = 8) -> None:
        if file_capacity < 1:
            raise ValueError(f"file_capacity must be at least 1: {file_capacity}")
        self._file_capacity = file_capacity
        self._files: OrderedDict[Path, tuple[int, h5py.File]] = OrderedDict()

    def load(self, row: FrameIndexRow) -> dict[str, Any]:
        """Load one model-ready frame from a selected H5 index row."""
        path = Path(row.h5_path)
        file = self._file_for(path)
        num_frames_value = file.attrs["num_frames"]
        if not isinstance(num_frames_value, (int, np.integer)):
            raise ValueError(f"H5 num_frames must be an integer: {path}")
        num_frames = int(num_frames_value)
        if not 0 <= row.frame_index < num_frames:
            raise IndexError(
                f"frame_index {row.frame_index} is outside {path} with {num_frames} frames"
            )
        frames = file["frames"]
        if not isinstance(frames, h5py.Group):
            raise ValueError(f"H5 frames must be a group: {path}")
        frame = {
            key: np.asarray(values[row.frame_index]) for key, values in frames.items()
        }
        return dict(_widen_agent_label(frame))

    def _file_for(self, path: Path) -> h5py.File:
        if not path.is_file():
            raise FileNotFoundError(f"H5 shard not found: {path}")
        modification_time_ns = path.stat().st_mtime_ns
        cached = self._files.pop(path, None)
        if cached is not None:
            cached_modification_time_ns, file = cached
            if cached_modification_time_ns == modification_time_ns:
                self._files[path] = cached
                return file
            file.close()
        file = h5py.File(path, "r")
        try:
            _validate_h5(file, path)
        except BaseException:
            file.close()
            raise
        self._files[path] = (modification_time_ns, file)
        while len(self._files) > self._file_capacity:
            _, (_, evicted) = self._files.popitem(last=False)
            evicted.close()
        return file

    def close(self) -> None:
        """Close every open H5 handle."""
        for _, file in self._files.values():
            file.close()
        self._files.clear()

    def __del__(self) -> None:
        files = getattr(self, "_files", None)
        if files is not None:
            self.close()
