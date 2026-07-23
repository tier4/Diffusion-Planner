from __future__ import annotations

import numpy as np
import pytest


def _make_clustered_embeddings(n_per: int = 30, k: int = 3, dim: int = 256, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim)) * 10
    parts = [rng.normal(c, 0.5, (n_per, dim)) for c in centers]
    return np.vstack(parts).astype(np.float32)


def test_cluster_embeddings_labels_shape():
    from dataset_curation.clustering import cluster_embeddings

    emb = _make_clustered_embeddings()
    labels, model = cluster_embeddings(emb, k=3)
    assert labels.shape == (len(emb),)
    assert set(labels) == {0, 1, 2}


def test_cluster_embeddings_finds_structure():
    from dataset_curation.clustering import cluster_embeddings

    emb = _make_clustered_embeddings(n_per=50, k=3)
    labels, _ = cluster_embeddings(emb, k=3)
    for cluster_id in range(3):
        count = (labels == cluster_id).sum()
        assert count >= 30, f"Cluster {cluster_id} has only {count} members"


def test_compute_umap_shape():
    from dataset_curation.clustering import compute_umap

    emb = _make_clustered_embeddings(n_per=20)
    coords = compute_umap(emb, n_components=2)
    assert coords.shape == (len(emb), 2)


def test_compute_umap_preserves_structure():
    from dataset_curation.clustering import compute_umap

    emb = _make_clustered_embeddings(n_per=30, k=2, dim=50)
    coords = compute_umap(emb, n_components=2)
    group_a = coords[:30]
    group_b = coords[30:]
    intra_a = np.mean(np.linalg.norm(group_a - group_a.mean(axis=0), axis=1))
    intra_b = np.mean(np.linalg.norm(group_b - group_b.mean(axis=0), axis=1))
    inter = np.linalg.norm(group_a.mean(axis=0) - group_b.mean(axis=0))
    assert inter > (intra_a + intra_b) / 2, "UMAP should preserve cluster separation"
