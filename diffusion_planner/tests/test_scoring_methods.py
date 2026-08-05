from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusion_planner.utils.scoring_methods import (
    GMMScorer,
    MahalanobisScorer,
    SphericalKMeansScorer,
)


def _random_l2(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-8)).astype(np.float32)


class TestMahalanobisFit:
    def test_fit_stores_params(self):
        emb = _random_l2(100, 32)
        scorer = MahalanobisScorer.fit(emb)
        assert scorer.mu.shape == (32,)
        assert scorer.precision.shape == (32, 32)
        assert scorer.whitening.shape == (32, 32)

    def test_condition_number_logged(self, capsys):
        emb = _random_l2(100, 32)
        MahalanobisScorer.fit(emb)
        captured = capsys.readouterr()
        assert "condition number" in captured.out.lower()


class TestMahalanobisScore:
    def test_returns_expected_key(self):
        emb = _random_l2(100, 16)
        scorer = MahalanobisScorer.fit(emb)
        result = scorer.score(torch.randn(2, 16))
        assert "mahalanobis" in result
        assert result["mahalanobis"].shape == (2,)

    def test_score_is_squared(self):
        emb = _random_l2(100, 8)
        scorer = MahalanobisScorer.fit(emb)
        result = scorer.score(torch.randn(1, 8))
        assert result["mahalanobis"].item() >= 0

    def test_inlier_lower_than_outlier(self):
        rng = np.random.RandomState(0)
        cluster = rng.randn(200, 8).astype(np.float32) * 0.1
        cluster[:, 0] += 5.0
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        cluster = (cluster / norms).astype(np.float32)
        scorer = MahalanobisScorer.fit(cluster)

        inlier = torch.from_numpy(cluster[:1])
        outlier = torch.from_numpy(-cluster[:1])
        assert (
            scorer.score(inlier)["mahalanobis"].item() < scorer.score(outlier)["mahalanobis"].item()
        )

    def test_1d_input_auto_batched(self):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        result = scorer.score(torch.randn(8))
        assert result["mahalanobis"].shape == (1,)


class TestMahalanobisNearest:
    def test_returns_k_neighbors(self):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        results = scorer.nearest(torch.from_numpy(emb[:1]), k=3)
        assert len(results) == 1
        assert len(results[0]) == 3
        assert "distance" in results[0][0]
        assert "index" in results[0][0]

    def test_self_is_nearest(self):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        results = scorer.nearest(torch.from_numpy(emb[0:1]), k=1)
        assert results[0][0]["index"] == 0
        assert results[0][0]["distance"] < 1e-4


class TestMahalanobisSaveLoad:
    def test_roundtrip(self, tmp_path):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        scorer.save(tmp_path / "maha.npz")

        loaded = MahalanobisScorer.load(tmp_path / "maha.npz")
        z = torch.randn(2, 8)
        torch.testing.assert_close(scorer.score(z)["mahalanobis"], loaded.score(z)["mahalanobis"])


class TestSphericalKMeansFit:
    def test_fit_produces_centroids(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        assert scorer.centroids.shape == (4, 16)

    def test_centroids_are_unit_norm(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        norms = np.linalg.norm(scorer.centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_assignments_cover_all_points(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        assert scorer.assignments.shape == (200,)
        assert set(scorer.assignments.tolist()).issubset(range(4))


class TestSphericalKMeansScore:
    def test_returns_expected_keys(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        result = scorer.score(torch.randn(3, 16))
        assert "kmeans_dist" in result
        assert "cluster_id" in result
        assert "margin" in result
        assert result["kmeans_dist"].shape == (3,)
        assert result["cluster_id"].shape == (3,)

    def test_centroid_scores_near_zero(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        centroid_z = torch.from_numpy(scorer.centroids[:1])
        result = scorer.score(centroid_z)
        assert result["kmeans_dist"].item() < 0.01

    def test_1d_input_auto_batched(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        result = scorer.score(torch.randn(16))
        assert result["kmeans_dist"].shape == (1,)


class TestSphericalKMeansNearest:
    def test_returns_k_neighbors_from_top2_clusters(self):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        results = scorer.nearest(torch.from_numpy(emb[:1]), emb, k=3)
        assert len(results) == 1
        assert len(results[0]) == 3
        assert "distance" in results[0][0]
        assert "index" in results[0][0]


class TestSphericalKMeansSaveLoad:
    def test_roundtrip(self, tmp_path):
        emb = _random_l2(200, 16)
        scorer = SphericalKMeansScorer.fit(emb, k=4)
        scorer.save(tmp_path / "kmeans.npz")

        loaded = SphericalKMeansScorer.load(tmp_path / "kmeans.npz")
        z = torch.randn(2, 16)
        torch.testing.assert_close(scorer.score(z)["kmeans_dist"], loaded.score(z)["kmeans_dist"])


class TestGMMFit:
    def test_fit_stores_params(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 16).astype(np.float32)
        scorer = GMMScorer.fit(emb, k=4)
        assert scorer.n_components == 4
        assert scorer.scaler_mean.shape == (16,)
        assert scorer.scaler_std.shape == (16,)

    def test_assignments_cover_all_points(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 16).astype(np.float32)
        scorer = GMMScorer.fit(emb, k=4)
        assert scorer.assignments.shape == (200,)


class TestGMMScore:
    def test_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 16).astype(np.float32)
        scorer = GMMScorer.fit(emb, k=4)
        result = scorer.score(torch.randn(3, 16))
        assert "gmm_nll" in result
        assert "component_id" in result
        assert "component_prob" in result
        assert result["gmm_nll"].shape == (3,)

    def test_inlier_lower_nll_than_outlier(self):
        rng = np.random.RandomState(42)
        cluster = rng.randn(200, 8).astype(np.float32) * 0.5
        scorer = GMMScorer.fit(cluster, k=2)
        inlier = torch.zeros(1, 8)
        outlier = torch.ones(1, 8) * 100
        assert scorer.score(inlier)["gmm_nll"].item() < scorer.score(outlier)["gmm_nll"].item()

    def test_1d_input_auto_batched(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 8).astype(np.float32)
        scorer = GMMScorer.fit(emb, k=2)
        result = scorer.score(torch.randn(8))
        assert result["gmm_nll"].shape == (1,)


class TestGMMNearest:
    def test_returns_k_neighbors(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 16).astype(np.float32)
        scorer = GMMScorer.fit(emb, k=4)
        results = scorer.nearest(torch.from_numpy(emb[:1]), emb, k=3)
        assert len(results) == 1
        assert len(results[0]) == 3
        assert "distance" in results[0][0]
        assert "index" in results[0][0]


class TestGMMSaveLoad:
    def test_roundtrip(self, tmp_path):
        rng = np.random.RandomState(42)
        emb = rng.randn(200, 8).astype(np.float32)
        scorer = GMMScorer.fit(emb, k=4)
        scorer.save(tmp_path / "gmm.npz", tmp_path / "zscore.npz")

        loaded = GMMScorer.load(tmp_path / "gmm.npz", tmp_path / "zscore.npz")
        z = torch.randn(2, 8)
        torch.testing.assert_close(scorer.score(z)["gmm_nll"], loaded.score(z)["gmm_nll"])
