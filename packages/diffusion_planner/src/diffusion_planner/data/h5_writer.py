"""Shared, atomic writer for Diffusion Planner frame-dataset H5 shards."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

FORMAT_NAME = "diffusion_planner_frame_dataset"
FORMAT_VERSION = 4


def validate_h5_arrays(
    frames: Mapping[str, Any], metadata: Mapping[str, Any]
) -> int:
    """Validate the shared frame axis and return its length."""
    frame_times = np.asarray(metadata.get("frame_time_ns"))
    if frame_times.ndim != 1:
        raise ValueError("metadata/frame_time_ns must be a vector")
    num_frames = len(frame_times)
    if num_frames == 0:
        return 0
    if not frames:
        raise ValueError("H5 result contains metadata but no frame tensors")
    if num_frames > 1 and not np.all(frame_times[1:] > frame_times[:-1]):
        raise ValueError("frame times must be strictly increasing")
    for group_name, arrays in (("frames", frames), ("metadata", metadata)):
        for key, values in arrays.items():
            array = np.asarray(values)
            if array.ndim == 0 or len(array) != num_frames:
                raise ValueError(
                    f"{group_name}/{key} does not have {num_frames} frame entries"
                )
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"{group_name}/{key} contains NaN or infinity")
    return num_frames


def _compression_options(compression: str | None) -> dict[str, Any]:
    if compression is None:
        return {}
    if compression == "zstd":
        return dict(hdf5plugin.Zstd())
    return {"compression": compression}


def write_h5_shard(
    path: Path | str,
    *,
    attributes: Mapping[str, Any],
    frames: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compression: str | None = "zstd",
) -> int:
    """Atomically write one schema-v4 H5 shard from batched frame arrays."""
    path = Path(path)
    num_frames = validate_h5_arrays(frames, metadata)
    if num_frames == 0:
        raise ValueError(f"refusing to write an empty H5 shard: {path}")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    options = _compression_options(compression)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["format"] = FORMAT_NAME
            output.attrs["format_version"] = FORMAT_VERSION
            output.attrs["num_frames"] = num_frames
            for key, value in attributes.items():
                if key in {"format", "format_version", "num_frames"}:
                    raise ValueError(f"reserved H5 attribute: {key}")
                output.attrs[key] = value
            frame_group = output.create_group("frames")
            metadata_group = output.create_group("metadata")
            for key, values in frames.items():
                array = np.asarray(values)
                frame_group.create_dataset(
                    key,
                    data=array,
                    chunks=(1, *array.shape[1:]),
                    shuffle=compression is not None,
                    **options,
                )
            for key, values in metadata.items():
                metadata_group.create_dataset(key, data=np.asarray(values))
            output.flush()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return num_frames
