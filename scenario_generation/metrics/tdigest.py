"""Thin wrappers around ``tdigest.TDigest`` for mergeable float distributions.

Build a digest from samples, serialize with ``to_dict``, merge across segments,
and query an approximate percentile — without retaining the raw series.
"""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
from tdigest import TDigest

# In-memory metric key; stripped from human-readable segments.jsonl and written to
# a ``tdigests*.jsonl`` sidecar for multi-GPU clearance-p5 merge.
TDIGEST_KEY = "_tdigest"

# Arbitrary, but must never change: the percentile a digest reports depends on it.
_RNG_SEED = 20260731


@contextmanager
def _deterministic_rng():
    """Pin the global ``random`` state so that identical samples give an identical digest.

    ``tdigest`` 0.5.x draws from the global ``random`` module while building and compressing,
    so without this the same input lands on different centroids and reports a different
    percentile. The previous state is restored, so other users of ``random`` are unaffected.

    Not reentrant: concurrent digest builds would interleave the save/seed/restore and both
    lose determinism. Both call sites are on the main thread; moving one into a worker needs a
    lock here first.
    """
    state = random.getstate()
    random.seed(_RNG_SEED)
    try:
        yield
    finally:
        random.setstate(state)


def is_tdigest_key(key: str) -> bool:
    return key == TDIGEST_KEY or key.startswith(f"{TDIGEST_KEY}_")


def tdigest_dict_from_values(values: np.ndarray) -> dict | None:
    """Build a serializable t-digest from finite samples; ``None`` if empty."""
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    with _deterministic_rng():
        digest = TDigest()
        digest.batch_update(finite.tolist())
        digest.compress()
        return digest.to_dict()


def merged_percentile(digest_dicts: list[dict], percentile: float) -> float:
    """Merge serialized digests and return an approximate percentile in ``[0, 100]``.

    Order-dependent: t-digest merging is not associative, so a caller pooling shards must feed
    them in a stable order.
    """
    if not digest_dicts:
        return float("inf")
    with _deterministic_rng():
        digest = TDigest()
        for d in digest_dicts:
            digest.update_from_dict(d)
        digest.compress()
        return float(digest.percentile(float(percentile)))
