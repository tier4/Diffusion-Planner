from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusion_planner.utils.latent_ood import LatentOODScorer


def _make_bank(n: int = 100, d: int = 32, seed: int = 42):
    rng = np.random.RandomState(seed)
    embeddings = rng.randn(n, d).astype(np.float32)
    records = [{"row": i, "npz_path": f"/data/train/{i}.npz"} for i in range(n)]
    metadata = {"model_path": "/mock/model.pth", "embedding_dim": d, "bank_format_version": 2}
    return embeddings, records, metadata


class TestBuild:
    def test_normalizes_embeddings(self):
        emb, rec, meta = _make_bank()
        scorer = LatentOODScorer.build(emb, rec, meta)
        norms = np.linalg.norm(scorer._embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            LatentOODScorer.build(np.zeros((10,)), [], {})


class TestScore:
    def test_returns_expected_keys(self):
        emb, rec, meta = _make_bank()
        scorer = LatentOODScorer.build(emb, rec, meta)
        result = scorer.score(torch.randn(2, 32), k=5)
        assert set(result.keys()) >= {"knn_mean", "knn_min", "knn_kth"}
        assert result["knn_mean"].shape == (2,)

    def test_exact_match_is_zero(self):
        emb, rec, meta = _make_bank(n=10, d=4)
        scorer = LatentOODScorer.build(emb, rec, meta)
        z = torch.from_numpy(scorer._embeddings[0:1])
        result = scorer.score(z, k=1)
        assert result["knn_min"].item() < 1e-5

    def test_outlier_scores_higher(self):
        emb = np.ones((50, 4), dtype=np.float32)
        emb += np.random.RandomState(0).randn(50, 4).astype(np.float32) * 0.01
        rec = [{"row": i} for i in range(50)]
        scorer = LatentOODScorer.build(emb, rec, {})
        out = scorer.score(torch.tensor([[-1.0, -1.0, -1.0, -1.0]]), k=5)["knn_mean"]
        inn = scorer.score(torch.tensor([[1.0, 1.0, 1.0, 1.0]]), k=5)["knn_mean"]
        assert out.item() > inn.item() * 2

    def test_1d_input_auto_batched(self):
        emb, rec, meta = _make_bank(n=10, d=4)
        scorer = LatentOODScorer.build(emb, rec, meta)
        result = scorer.score(torch.randn(4), k=3)
        assert result["knn_mean"].shape == (1,)

    def test_k_clamped_to_bank_size(self):
        emb, rec, meta = _make_bank(n=5, d=4)
        scorer = LatentOODScorer.build(emb, rec, meta)
        result = scorer.score(torch.randn(1, 4), k=100)
        assert result["knn_mean"].shape == (1,)


class TestNearest:
    def test_returns_records_with_distances(self):
        emb, rec, meta = _make_bank(n=10, d=4)
        scorer = LatentOODScorer.build(emb, rec, meta)
        z = torch.from_numpy(scorer._embeddings[0:1])
        results = scorer.nearest(z, k=3)
        assert len(results) == 1  # batch size 1
        neighbors = results[0]
        assert len(neighbors) == 3
        assert "distance" in neighbors[0]
        assert "npz_path" in neighbors[0]
        assert neighbors[0]["distance"] < 1e-5


class TestCalibration:
    def test_sets_thresholds(self):
        emb, rec, meta = _make_bank()
        scorer = LatentOODScorer.build(emb, rec, meta)
        scores = np.linspace(0, 1, 1000).astype(np.float32)
        scorer.calibrate(scores, warning_pct=95, high_pct=99)
        assert abs(scorer._calibration["thresholds"]["warning"] - 0.95) < 0.01
        assert abs(scorer._calibration["thresholds"]["high"] - 0.99) < 0.01

    def test_score_includes_percentile_and_level(self):
        emb, rec, meta = _make_bank(n=50, d=8)
        scorer = LatentOODScorer.build(emb, rec, meta)
        scorer.calibrate(np.linspace(0, 2, 1000).astype(np.float32))
        result = scorer.score(torch.randn(2, 8), k=5)
        assert "percentile" in result
        assert "level" in result
        assert result["level"].dtype == torch.long


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        emb, rec, meta = _make_bank(n=20, d=8)
        scorer = LatentOODScorer.build(emb, rec, meta)
        scorer.calibrate(np.random.RandomState(0).rand(100).astype(np.float32))

        scorer.save(tmp_path / "bank")
        loaded = LatentOODScorer.load(tmp_path / "bank")

        assert loaded.num_embeddings == 20
        assert loaded.embedding_dim == 8
        np.testing.assert_allclose(loaded._embeddings, scorer._embeddings)

        z = torch.randn(1, 8)
        orig = scorer.score(z)
        reloaded = loaded.score(z)
        torch.testing.assert_close(orig["knn_mean"], reloaded["knn_mean"])

    def test_load_without_calibration(self, tmp_path):
        emb, rec, meta = _make_bank(n=10, d=4)
        scorer = LatentOODScorer.build(emb, rec, meta)
        scorer.save(tmp_path / "bank")

        loaded = LatentOODScorer.load(tmp_path / "bank")
        result = loaded.score(torch.randn(1, 4))
        assert "percentile" not in result

    def test_rejects_v1_bank(self, tmp_path):
        import json

        bank_dir = tmp_path / "v1_bank"
        bank_dir.mkdir()
        # Write a v1 metadata (no bank_format_version key)
        with open(bank_dir / "metadata.json", "w") as f:
            json.dump({"model_path": "/mock/model.pth", "embedding_dim": 4}, f)
        np.save(bank_dir / "embeddings_l2.npy", np.zeros((5, 4), dtype=np.float32))
        with open(bank_dir / "paths.jsonl", "w") as f:
            for i in range(5):
                f.write(json.dumps({"row": i}) + "\n")

        with pytest.raises(ValueError, match="format version 1"):
            LatentOODScorer.load(bank_dir)
