import io

import numpy as np
import pytest
from diffusion_planner.data_pipeline import encoding
from diffusion_planner.data_pipeline.errors import EncodingError, IntegrityError
from tests.dp_fixtures import make_arrays


def _npz_bytes(arrays, compressed=True):
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    return buf.getvalue()


@pytest.mark.parametrize("compressed", [True, False])
def test_roundtrip_bitexact(compressed):
    arrays = make_arrays(np.random.default_rng(1))
    src = encoding.load_npz_bytes(_npz_bytes(arrays, compressed))
    payload = encoding.encode_sample(src)
    back = encoding.decode_sample(payload)
    assert encoding.arrays_bitexact(src, back)
    assert len(back) == 18
    train = encoding.decode_for_training(payload)
    assert len(train) == 17 and "version" not in train


def test_special_values_are_bitexact():
    nan1 = np.array([np.nan], dtype=np.float32).view(np.uint32)
    nan2 = (nan1 | np.uint32(1)).view(np.float32)  # distinct NaN payload
    arrays = {
        "a": np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32),
        "b": nan2,
        "c": np.zeros((0, 3), dtype=np.float32),
        "d": np.array(7, dtype=np.int64),
        "e": np.arange(6, dtype=np.int16).reshape(2, 3).T,  # non-contiguous (Fortran-like view)
        "f": np.array([True, False]),
        "g": np.arange(4, dtype=np.uint8),
        "h": np.arange(3, dtype=np.float64),
    }
    back = encoding.decode_sample(encoding.encode_sample(arrays))
    assert encoding.arrays_bitexact(arrays, back)
    assert back["a"].tobytes() == arrays["a"].tobytes()  # -0.0 and NaN payload preserved


def test_preflight_rejects_big_endian_and_object():
    with pytest.raises(EncodingError):
        encoding.preflight({"a": np.arange(3, dtype=">f4")})
    with pytest.raises(EncodingError):
        encoding.preflight({"a": np.array([object()], dtype=object)})


def test_corrupted_payload_is_detected():
    payload = bytearray(encoding.encode_sample(make_arrays(np.random.default_rng(2))))
    payload[len(payload) // 2] ^= 0xFF
    with pytest.raises(IntegrityError):
        encoding.decode_sample(bytes(payload))


def test_recipe_hash_is_stable_and_versioned():
    h = encoding.recipe_hash()
    assert h == encoding.recipe_hash() and len(h) == 64
