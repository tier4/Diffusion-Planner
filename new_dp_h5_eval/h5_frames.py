"""Per-frame tensor access for both native H5 layouts.

Shards store a frame either as one dataset per tensor below ``frames/``, or (ML-Planner's
packed writer) as one float32 row of ``frames_packed`` that concatenates the flattened
tensors in the order and per-frame shapes of the ``packed_frame_layout`` JSON attribute.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

from .schema import H5_PACKED_DATASET, H5_PACKED_LAYOUT_ATTR


class NativeFrames:
    """Read frame tensors of an open native H5 shard, whatever its layout."""

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        if H5_PACKED_DATASET in file:
            layout = json.loads(file.attrs[H5_PACKED_LAYOUT_ATTR])
            self._packed = file[H5_PACKED_DATASET]
            self._shapes = {
                key: tuple(shape) for key, shape in zip(layout["keys"], layout["shapes"])
            }
            sizes = [int(np.prod(shape, dtype=np.int64)) for shape in self._shapes.values()]
            offsets = np.concatenate(([0], np.cumsum(sizes))).astype(int)
            self._spans = dict(zip(self._shapes, zip(offsets[:-1], offsets[1:])))
            if offsets[-1] != self._packed.shape[1]:
                raise ValueError(
                    f"{H5_PACKED_LAYOUT_ATTR} sums to {offsets[-1]} values, but "
                    f"{H5_PACKED_DATASET} rows hold {self._packed.shape[1]}: {file.filename}"
                )
        elif "frames" in file:
            self._packed = None
            self._frames = file["frames"]
        else:
            raise ValueError(f"native H5 shard has neither frames/ nor {H5_PACKED_DATASET}")

    @property
    def packed(self) -> bool:
        return self._packed is not None

    def keys(self) -> set[str]:
        return set(self._shapes) if self._packed is not None else set(self._frames.keys())

    def row(self, index: int) -> dict[str, np.ndarray]:
        """Every tensor of frame ``index``."""
        if self._packed is None:
            return {key: np.asarray(value[index]) for key, value in self._frames.items()}
        row = np.asarray(self._packed[index])
        return {
            key: row[begin:end].reshape(self._shapes[key])
            for key, (begin, end) in self._spans.items()
        }

    def column(self, key: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Tensor ``key`` of frames ``[start, stop)``.

        A packed row is one HDF5 chunk, so a column costs a full read of every row in the
        range; use it only for short ranges.
        """
        if self._packed is None:
            return np.asarray(self._frames[key][start:stop])
        begin, end = self._spans[key]
        rows = np.asarray(self._packed[start:stop, begin:end])
        return rows.reshape(len(rows), *self._shapes[key])
