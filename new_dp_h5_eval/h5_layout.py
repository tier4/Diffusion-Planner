# Packed layout implementation vendored from tier4/ML-Planner at
# 9ad92edd22f3df0753b77027cb9bdc18b3fe7c7e (PR #88).
"""Packed H5 frame layout: every tensor of a frame in one float32 row (one HDF5 chunk).

Reading a frame from one chunk instead of one chunk per tensor halves the read CPU and
makes cold reads several times faster. Shards written per tensor (a `frames/` group
with one dataset per key) still load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

PACKED_DATASET = "frames_packed"
PACKED_LAYOUT_ATTR = "packed_frame_layout"


@dataclass(frozen=True)
class PackedLayout:
    """Tensor keys and per-frame shapes, in row order."""

    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    @classmethod
    def from_arrays(cls, frames: Mapping[str, Any]) -> PackedLayout:
        """The layout of `(N, ...)` float32 tensors, in sorted key order."""
        keys = tuple(sorted(frames))
        for key in keys:
            if np.asarray(frames[key]).dtype != np.float32:
                raise ValueError(f"frame tensor {key} is not float32")
        return cls(keys, tuple(tuple(np.asarray(frames[k]).shape[1:]) for k in keys))

    @classmethod
    def from_json(cls, text: str | bytes) -> PackedLayout:
        value = json.loads(text)
        return cls(tuple(value["keys"]), tuple(tuple(s) for s in value["shapes"]))

    def to_json(self) -> str:
        return json.dumps({"keys": list(self.keys), "shapes": [list(s) for s in self.shapes]})

    @cached_property
    def offsets(self) -> tuple[int, ...]:
        sizes = [int(np.prod(shape, dtype=np.int64)) for shape in self.shapes]
        return tuple(int(v) for v in np.concatenate(([0], np.cumsum(sizes))))

    @property
    def row_size(self) -> int:
        return self.offsets[-1]

    def pack(self, frames: Mapping[str, Any]) -> NDArray[np.float32]:
        """`(N, row_size)` rows of `(N, ...)` tensors."""
        num_frames = len(np.asarray(frames[self.keys[0]]))
        return np.concatenate(
            [
                np.asarray(frames[key], dtype=np.float32).reshape(num_frames, -1)
                for key in self.keys
            ],
            axis=1,
        )

    def unpack(self, row: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
        """The tensors of one frame row, as views into it."""
        offsets = self.offsets
        return {
            key: row[offsets[i] : offsets[i + 1]].reshape(shape)
            for i, (key, shape) in enumerate(zip(self.keys, self.shapes, strict=True))
        }


def packed_layout(file: h5py.File) -> PackedLayout | None:
    """The packed layout of a shard, or None for a shard written per tensor."""
    if PACKED_DATASET not in file:
        return None
    text = file.attrs[PACKED_LAYOUT_ATTR]
    if not isinstance(text, (str, bytes)):
        raise ValueError(f"H5 {PACKED_LAYOUT_ATTR} must be a JSON string")
    return PackedLayout.from_json(text)


def create_packed_dataset(
    file: h5py.File,
    layout: PackedLayout,
    num_frames: int,
    compression: Mapping[str, Any],
    shuffle: bool,
) -> h5py.Dataset:
    """A resizable `(num_frames, row_size)` dataset chunked one frame at a time."""
    file.attrs[PACKED_LAYOUT_ATTR] = layout.to_json()
    return file.create_dataset(
        PACKED_DATASET,
        shape=(num_frames, layout.row_size),
        maxshape=(None, layout.row_size),
        dtype=np.float32,
        chunks=(1, layout.row_size),
        shuffle=shuffle,
        **compression,
    )


class H5Frames:
    """Read either storage layout without changing the native tensor contract."""

    def __init__(self, file: h5py.File):
        self.file = file
        self.layout = packed_layout(file)
        if self.layout is not None:
            layout = self.layout
            if (
                not layout.keys
                or len(set(layout.keys)) != len(layout.keys)
                or len(layout.keys) != len(layout.shapes)
                or any(any(not isinstance(n, int) or n <= 0 for n in s) for s in layout.shapes)
            ):
                raise ValueError("Invalid packed frame layout")
            dataset = file[PACKED_DATASET]
            if (
                dataset.shape != (int(file.attrs["num_frames"]), layout.row_size)
                or dataset.dtype != np.float32
            ):
                raise ValueError("Packed frame shape or dtype does not match layout")
        elif "frames" not in file:
            raise ValueError("H5 shard contains neither frames nor frames_packed")

    def keys(self):
        return self.layout.keys if self.layout is not None else self.file["frames"].keys()

    def frame(self, index: int):
        if self.layout is not None:
            return self.layout.unpack(self.file[PACKED_DATASET][index])
        return {key: np.asarray(value[index]) for key, value in self.file["frames"].items()}

    def tensor(self, key: str, start: int = 0, stop: int | None = None):
        if self.layout is None:
            return np.asarray(self.file["frames"][key][start:stop])
        i = self.layout.keys.index(key)
        rows = self.file[PACKED_DATASET][
            start:stop, self.layout.offsets[i] : self.layout.offsets[i + 1]
        ]
        return rows.reshape((len(rows), *self.layout.shapes[i]))
