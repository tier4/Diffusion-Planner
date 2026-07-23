from __future__ import annotations

import numpy as np
import pytest


def test_rarity_scores_shape():
    from dataset_curation.prototypicality import compute_rarity_scores

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((100, 64)).astype(np.float32)
    labels = np.arange(100) % 5
    centroids = np.stack([emb[labels == k].mean(axis=0) for k in range(5)])

    scores = compute_rarity_scores(emb, labels, centroids)
    assert scores.shape == (100,)
    assert (scores >= 0).all()


def test_centroid_sample_has_lowest_score():
    from dataset_curation.prototypicality import compute_rarity_scores

    rng = np.random.default_rng(42)
    center = np.zeros(64, dtype=np.float32)
    near = rng.normal(0, 0.01, (10, 64)).astype(np.float32)
    far = rng.normal(0, 5.0, (10, 64)).astype(np.float32)
    emb = np.vstack([center[None], near, far])
    labels = np.zeros(21, dtype=int)
    centroids = emb.mean(axis=0, keepdims=True)

    scores = compute_rarity_scores(emb, labels, centroids)
    near_mean = scores[1:11].mean()
    far_mean = scores[11:].mean()
    assert far_mean > near_mean, "Far samples should have higher rarity"


def test_rarity_scores_all_finite():
    from dataset_curation.prototypicality import compute_rarity_scores

    emb = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    labels = np.zeros(50, dtype=int)
    centroids = emb.mean(axis=0, keepdims=True)

    scores = compute_rarity_scores(emb, labels, centroids)
    assert np.all(np.isfinite(scores))
