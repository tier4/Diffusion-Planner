from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class LatentOODScorer:
    """Score scene embeddings by kNN distance to a training embedding bank.

    Usage:
        scorer = LatentOODScorer.build(embeddings, records, metadata)
        scorer.calibrate(validation_scores)
        scorer.save("bank_dir/")

        scorer = LatentOODScorer.load("bank_dir/")
        result = scorer.score(scene_embedding)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        records: list[dict],
        metadata: dict,
        calibration: dict | None = None,
    ):
        self._embeddings = embeddings
        self._records = records
        self._metadata = metadata
        self._calibration = calibration
        self._bank_tensor: torch.Tensor | None = None
        self._device = "cpu"

    @property
    def num_embeddings(self) -> int:
        return self._embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self._embeddings.shape[1]

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        records: list[dict],
        metadata: dict,
        device: str = "cpu",
    ) -> LatentOODScorer:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
        if len(records) != embeddings.shape[0]:
            raise ValueError(
                f"records length ({len(records)}) != embeddings count ({embeddings.shape[0]})"
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = (embeddings / norms).astype(np.float32)
        scorer = cls(embeddings=embeddings, records=records, metadata=metadata)
        scorer._device = device
        return scorer

    def _get_bank(self) -> torch.Tensor:
        if self._bank_tensor is None or self._bank_tensor.device.type != self._device:
            self._bank_tensor = torch.from_numpy(self._embeddings).to(self._device)
        return self._bank_tensor

    def _knn_distances(
        self, z: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        bank = self._get_bank()
        k = min(k, bank.shape[0])
        cos_sim = z @ bank.T
        dists = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0))
        return torch.topk(dists, k, dim=-1, largest=False)

    def score(
        self, scene_embedding: torch.Tensor, k: int = 10
    ) -> dict[str, torch.Tensor]:
        topk_dists, _ = self._knn_distances(scene_embedding, k)
        result: dict[str, torch.Tensor] = {
            "knn_mean": topk_dists.mean(dim=-1),
            "knn_min": topk_dists[:, 0],
            "knn_kth": topk_dists[:, -1],
        }
        if self._calibration:
            quantiles = torch.tensor(
                self._calibration["quantiles"], dtype=torch.float32
            )
            percentiles = torch.searchsorted(quantiles, result["knn_mean"].cpu())
            percentiles = percentiles.clamp(max=100).float().to(result["knn_mean"].device)
            result["percentile"] = percentiles

            warning = self._calibration["thresholds"]["warning"]
            high = self._calibration["thresholds"]["high"]
            levels = torch.zeros_like(result["knn_mean"], dtype=torch.long)
            levels[result["knn_mean"] >= warning] = 1
            levels[result["knn_mean"] >= high] = 2
            result["level"] = levels
        return result

    def nearest(
        self, scene_embedding: torch.Tensor, k: int = 5
    ) -> list[list[dict]]:
        topk_dists, topk_idx = self._knn_distances(scene_embedding, k)
        z = F.normalize(scene_embedding.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        results = []
        for b in range(z.shape[0]):
            neighbors = []
            for i in range(topk_dists.shape[1]):
                idx = topk_idx[b, i].item()
                record = self._records[idx] if idx < len(self._records) else {}
                neighbors.append({"distance": topk_dists[b, i].item(), "index": idx, **record})
            results.append(neighbors)
        return results

    def calibrate(
        self,
        scores: np.ndarray,
        warning_pct: float = 95.0,
        high_pct: float = 99.0,
    ) -> None:
        quantiles = np.percentile(scores, np.arange(101)).tolist()
        self._calibration = {
            "score_type": "knn_mean_l2",
            "quantiles": quantiles,
            "thresholds": {
                "warning": float(np.percentile(scores, warning_pct)),
                "high": float(np.percentile(scores, high_pct)),
            },
        }

    def save(self, bank_dir: str | Path) -> None:
        d = Path(bank_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self._embeddings)
        with open(d / "metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2)
        with open(d / "paths.jsonl", "w") as f:
            for r in self._records:
                f.write(json.dumps(r) + "\n")
        if self._calibration:
            with open(d / "calibration.json", "w") as f:
                json.dump(self._calibration, f, indent=2)

    @classmethod
    def load(cls, bank_dir: str | Path, device: str = "cpu") -> LatentOODScorer:
        d = Path(bank_dir)
        embeddings = np.load(d / "embeddings.npy")
        with open(d / "metadata.json") as f:
            metadata = json.load(f)
        records: list[dict] = []
        with open(d / "paths.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        calibration = None
        cal_path = d / "calibration.json"
        if cal_path.exists():
            with open(cal_path) as f:
                calibration = json.load(f)
        scorer = cls(
            embeddings=embeddings,
            records=records,
            metadata=metadata,
            calibration=calibration,
        )
        scorer._device = device
        return scorer
