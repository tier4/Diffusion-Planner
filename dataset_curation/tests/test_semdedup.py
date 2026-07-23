from __future__ import annotations

import numpy as np
import pytest


def test_semdedup_keeps_unique():
    from dataset_curation.semdedup import semdedup

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 64)).astype(np.float32)
    labels = np.zeros(50, dtype=int)
    keep = semdedup(emb, labels, threshold=0.99)
    assert keep.sum() >= 45, "Most unique samples should be kept"


def test_semdedup_removes_near_duplicates():
    from dataset_curation.semdedup import semdedup

    rng = np.random.default_rng(42)
    base = rng.standard_normal((10, 64)).astype(np.float32)
    dupes = base + rng.normal(0, 0.001, base.shape).astype(np.float32)
    unique = rng.standard_normal((10, 64)).astype(np.float32) * 5
    emb = np.vstack([base, dupes, unique])
    labels = np.array([0] * 20 + [1] * 10)

    keep = semdedup(emb, labels, threshold=0.95)
    assert keep.sum() < len(emb), "Should remove some near-duplicates"
    assert keep[20:].all(), "Unique cluster members should all be kept"


def test_semdedup_threshold_monotonic():
    from dataset_curation.semdedup import semdedup

    rng = np.random.default_rng(42)
    base = rng.standard_normal((20, 64)).astype(np.float32)
    dupes = base + rng.normal(0, 0.01, base.shape).astype(np.float32)
    emb = np.vstack([base, dupes])
    labels = np.zeros(40, dtype=int)

    kept_95 = semdedup(emb, labels, threshold=0.95).sum()
    kept_90 = semdedup(emb, labels, threshold=0.90).sum()
    kept_85 = semdedup(emb, labels, threshold=0.85).sum()
    assert kept_85 <= kept_90 <= kept_95, (
        f"Lower threshold should keep fewer: {kept_85}, {kept_90}, {kept_95}"
    )


def test_semdedup_returns_correct_shape():
    from dataset_curation.semdedup import semdedup

    emb = np.random.default_rng(0).standard_normal((100, 32)).astype(np.float32)
    labels = np.arange(100) % 5
    keep = semdedup(emb, labels, threshold=0.95)
    assert keep.shape == (100,)
    assert keep.dtype == bool
