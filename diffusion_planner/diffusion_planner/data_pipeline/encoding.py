"""Sample member encoding v1: safetensors container + zstd frame (spec §3)."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import safetensors
import zstandard
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

from diffusion_planner.data_pipeline import FORMAT_VERSION
from diffusion_planner.data_pipeline.defaults import ZSTD_LEVEL
from diffusion_planner.data_pipeline.errors import EncodingError, IntegrityError

MEMBER_EXT = ".safetensors.zst"
# safetensors.numpy dtype support (bfloat16 excluded: no numpy dtype)
_SUPPORTED_KINDS = {"b", "u", "i", "f"}
_SUPPORTED_ITEMSIZES = {"b": {1}, "u": {1, 2, 4, 8}, "i": {1, 2, 4, 8}, "f": {2, 4, 8}}


def load_npz_bytes(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def preflight(arrays: dict[str, np.ndarray]) -> None:
    for name, a in arrays.items():
        if not isinstance(a, np.ndarray):
            raise EncodingError(f"{name}: not an ndarray")
        dt = a.dtype
        if dt.kind not in _SUPPORTED_KINDS or dt.itemsize not in _SUPPORTED_ITEMSIZES[dt.kind]:
            raise EncodingError(f"{name}: unsupported dtype {dt.str}")
        if dt.byteorder == ">" or (dt.byteorder == "=" and not _little_endian()):
            raise EncodingError(f"{name}: big-endian arrays are not supported ({dt.str})")


def _little_endian() -> bool:
    return np.dtype("<i4").byteorder in ("<", "=") and np.little_endian


def encode_sample(arrays: dict[str, np.ndarray]) -> bytes:
    preflight(arrays)
    contiguous = {k: v if v.ndim == 0 else np.ascontiguousarray(v) for k, v in arrays.items()}
    st = st_save(contiguous)
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_checksum=True).compress(st)


def decode_sample(payload: bytes) -> dict[str, np.ndarray]:
    try:
        raw = zstandard.ZstdDecompressor().decompress(payload)
    except zstandard.ZstdError as e:
        raise IntegrityError(f"zstd frame invalid: {e}") from e
    try:
        return st_load(raw)
    except Exception as e:  # safetensors raises its own error types
        raise IntegrityError(f"safetensors payload invalid: {e}") from e


def decode_for_training(payload: bytes) -> dict[str, np.ndarray]:
    arrays = decode_sample(payload)
    arrays.pop("version", None)
    return arrays


def arrays_bitexact(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if x.dtype.str != y.dtype.str or x.shape != y.shape:
            return False
        if x.tobytes(order="C") != y.tobytes(order="C"):
            return False
    return True


def recipe_hash() -> str:
    parts = [
        f"format_version={FORMAT_VERSION}",
        f"zstd_level={ZSTD_LEVEL}",
        f"zstandard={zstandard.__version__}",
        f"safetensors={safetensors.__version__}",
        f"numpy={np.__version__}",
        "container=safetensors",
        "checksum=1",
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
