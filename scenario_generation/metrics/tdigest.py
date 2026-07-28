"""Thin wrappers around ``tdigest.TDigest`` for mergeable float distributions.

Build a digest from samples, serialize with ``to_dict``, merge across segments,
and query an approximate percentile — without retaining the raw series.
"""

from __future__ import annotations

import numpy as np
from tdigest import TDigest

# In-memory metric key; stripped from human-readable segments.jsonl and written to
# a ``tdigests*.jsonl`` sidecar for multi-GPU clearance-p5 merge.
TDIGEST_KEY = "_tdigest"


def is_tdigest_key(key: str) -> bool:
    return key == TDIGEST_KEY or key.startswith(f"{TDIGEST_KEY}_")


def tdigest_dict_from_values(values: np.ndarray) -> dict | None:
    """Build a serializable t-digest from finite samples; ``None`` if empty."""
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    digest = TDigest()
    digest.batch_update(finite.tolist())
    digest.compress()
    return digest.to_dict()


def merged_percentile(digest_dicts: list[dict], percentile: float) -> float:
    """Merge serialized digests and return an approximate percentile in ``[0, 100]``."""
    if not digest_dicts:
        return float("inf")
    digest = TDigest()
    for d in digest_dicts:
        digest.update_from_dict(d)
    digest.compress()
    return float(digest.percentile(float(percentile)))
