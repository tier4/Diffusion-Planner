from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


class MahalanobisScorer:
    def __init__(
        self,
        mu: np.ndarray,
        precision: np.ndarray,
        whitening: np.ndarray,
        whitened_bank: np.ndarray | None = None,
    ):
        self.mu = mu
        self.precision = precision
        self.whitening = whitening
        self.whitened_bank = whitened_bank

    @classmethod
    def fit(cls, embeddings_l2: np.ndarray, eps: float = 1e-5) -> MahalanobisScorer:
        mu = embeddings_l2.mean(axis=0)
        centered = embeddings_l2 - mu
        cov = (centered.T @ centered) / (len(embeddings_l2) - 1)
        cov_reg = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)

        cond = np.linalg.cond(cov_reg)
        print(f"  Mahalanobis covariance condition number: {cond:.2e} (eps={eps})")

        precision = np.linalg.inv(cov_reg).astype(np.float32)

        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        whitening = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)

        whitened_bank = (centered @ whitening.T).astype(np.float32)

        return cls(
            mu=mu.astype(np.float32),
            precision=precision,
            whitening=whitening,
            whitened_bank=whitened_bank,
        )

    def score(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        mu_t = torch.from_numpy(self.mu).to(z.device)
        prec_t = torch.from_numpy(self.precision).to(z.device)
        centered = z - mu_t
        mahal = (centered @ prec_t * centered).sum(dim=-1)
        return {"mahalanobis": mahal}

    def nearest(self, z: torch.Tensor, k: int = 5) -> list[list[dict]]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        mu_t = torch.from_numpy(self.mu).to(z.device)
        w_t = torch.from_numpy(self.whitening).to(z.device)
        bank_t = torch.from_numpy(self.whitened_bank).to(z.device)

        centered = z - mu_t
        whitened_q = centered @ w_t.T

        dists = torch.cdist(whitened_q, bank_t)
        k = min(k, bank_t.shape[0])
        topk_dists, topk_idx = torch.topk(dists, k, dim=-1, largest=False)

        results = []
        for b in range(z.shape[0]):
            neighbors = []
            for i in range(k):
                neighbors.append(
                    {
                        "distance": topk_dists[b, i].item(),
                        "index": topk_idx[b, i].item(),
                    }
                )
            results.append(neighbors)
        return results

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mu=self.mu,
            precision=self.precision,
            whitening=self.whitening,
            whitened_bank=self.whitened_bank,
        )

    @classmethod
    def load(cls, path: Path) -> MahalanobisScorer:
        data = np.load(path)
        return cls(
            mu=data["mu"],
            precision=data["precision"],
            whitening=data["whitening"],
            whitened_bank=data.get("whitened_bank"),
        )


class SphericalKMeansScorer:
    def __init__(
        self,
        centroids: np.ndarray,
        assignments: np.ndarray,
        k: int,
    ):
        self.centroids = centroids
        self.assignments = assignments
        self.k = k

    @classmethod
    def fit(cls, embeddings_l2: np.ndarray, k: int = 32) -> SphericalKMeansScorer:
        d = embeddings_l2.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, spherical=True)
        kmeans.train(embeddings_l2.astype(np.float32))
        centroids = kmeans.centroids.copy()

        _, assignments = kmeans.index.search(embeddings_l2.astype(np.float32), 1)
        assignments = assignments.ravel()

        return cls(centroids=centroids, assignments=assignments, k=k)

    def score(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        c_t = torch.from_numpy(self.centroids).to(z.device)
        cos_sim = z @ c_t.T
        dists = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0))

        sorted_dists, sorted_idx = torch.sort(dists, dim=-1)
        return {
            "kmeans_dist": sorted_dists[:, 0],
            "cluster_id": sorted_idx[:, 0],
            "margin": sorted_dists[:, 1] - sorted_dists[:, 0],
        }

    def nearest(self, z: torch.Tensor, embeddings_l2: np.ndarray, k: int = 5) -> list[list[dict]]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        c_t = torch.from_numpy(self.centroids).to(z.device)
        cos_sim = z @ c_t.T
        dists_to_c = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0))
        _, top2_clusters = torch.topk(dists_to_c, 2, dim=-1, largest=False)

        bank_t = torch.from_numpy(embeddings_l2).to(z.device)
        assignments_t = torch.from_numpy(self.assignments).to(z.device)

        results = []
        for b in range(z.shape[0]):
            c0 = top2_clusters[b, 0].item()
            c1 = top2_clusters[b, 1].item()
            mask = (assignments_t == c0) | (assignments_t == c1)
            candidate_idx = torch.where(mask)[0]

            if len(candidate_idx) == 0:
                results.append([])
                continue

            candidate_emb = bank_t[candidate_idx]
            cos_sim_cand = z[b : b + 1] @ candidate_emb.T
            cand_dists = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim_cand, min=0.0)).squeeze(0)

            actual_k = min(k, len(candidate_idx))
            topk_dists, topk_local = torch.topk(cand_dists, actual_k, largest=False)

            neighbors = []
            for i in range(actual_k):
                neighbors.append(
                    {
                        "distance": topk_dists[i].item(),
                        "index": candidate_idx[topk_local[i]].item(),
                    }
                )
            results.append(neighbors)
        return results

    def save(self, path: Path) -> None:
        np.savez(
            path,
            centroids=self.centroids,
            assignments=self.assignments,
            k=self.k,
        )

    @classmethod
    def load(cls, path: Path) -> SphericalKMeansScorer:
        data = np.load(path)
        return cls(
            centroids=data["centroids"],
            assignments=data["assignments"],
            k=int(data["k"]),
        )


class GMMScorer:
    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        covariance_type: str,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        assignments: np.ndarray,
        n_components: int,
    ):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.covariance_type = covariance_type
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.assignments = assignments
        self.n_components = n_components
        self._gmm = self._rebuild_gmm()

    def _rebuild_gmm(self) -> GaussianMixture:
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
        )
        gmm.weights_ = self.weights
        gmm.means_ = self.means
        gmm.covariances_ = self.covariances
        gmm.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances, self.covariance_type
        )
        return gmm

    @classmethod
    def fit(
        cls,
        embeddings_raw: np.ndarray,
        k: int = 16,
        covariance_type: str = "diag",
    ) -> GMMScorer:
        mean = embeddings_raw.mean(axis=0)
        std = embeddings_raw.std(axis=0)
        std = np.maximum(std, 1e-8)
        standardized = ((embeddings_raw - mean) / std).astype(np.float32)

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=42,
            max_iter=200,
        )
        gmm.fit(standardized)

        assignments = gmm.predict(standardized)

        return cls(
            weights=gmm.weights_.astype(np.float32),
            means=gmm.means_.astype(np.float32),
            covariances=gmm.covariances_.astype(np.float32),
            covariance_type=covariance_type,
            scaler_mean=mean.astype(np.float32),
            scaler_std=std.astype(np.float32),
            assignments=assignments.astype(np.int32),
            n_components=k,
        )

    def _standardize(self, z: np.ndarray) -> np.ndarray:
        return ((z - self.scaler_mean) / self.scaler_std).astype(np.float32)

    def score(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z_np = self._standardize(z.float().cpu().numpy())
        nll = -self._gmm.score_samples(z_np)
        probs = self._gmm.predict_proba(z_np)
        component_id = probs.argmax(axis=1)
        component_prob = probs.max(axis=1)

        return {
            "gmm_nll": torch.from_numpy(nll.astype(np.float32)),
            "component_id": torch.from_numpy(component_id.astype(np.int64)),
            "component_prob": torch.from_numpy(component_prob.astype(np.float32)),
        }

    def nearest(self, z: torch.Tensor, embeddings_raw: np.ndarray, k: int = 5) -> list[list[dict]]:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z_np = self._standardize(z.float().cpu().numpy())
        bank_std = self._standardize(embeddings_raw)

        probs = self._gmm.predict_proba(z_np)

        results = []
        for b in range(z_np.shape[0]):
            top2 = np.argsort(probs[b])[-2:]
            mask = np.isin(self.assignments, top2)
            candidate_idx = np.where(mask)[0]

            if len(candidate_idx) == 0:
                results.append([])
                continue

            dists = np.linalg.norm(bank_std[candidate_idx] - z_np[b], axis=1)
            actual_k = min(k, len(candidate_idx))
            topk_local = np.argpartition(dists, actual_k)[:actual_k]
            topk_local = topk_local[np.argsort(dists[topk_local])]

            neighbors = []
            for i in range(actual_k):
                neighbors.append(
                    {
                        "distance": float(dists[topk_local[i]]),
                        "index": int(candidate_idx[topk_local[i]]),
                    }
                )
            results.append(neighbors)
        return results

    def save(self, path: Path, scaler_path: Path) -> None:
        np.savez(
            path,
            weights=self.weights,
            means=self.means,
            covariances=self.covariances,
            covariance_type=self.covariance_type,
            assignments=self.assignments,
            n_components=self.n_components,
        )
        np.savez(scaler_path, mean=self.scaler_mean, std=self.scaler_std)

    @classmethod
    def load(cls, path: Path, scaler_path: Path) -> GMMScorer:
        data = np.load(path, allow_pickle=True)
        scaler = np.load(scaler_path)
        return cls(
            weights=data["weights"],
            means=data["means"],
            covariances=data["covariances"],
            covariance_type=str(data["covariance_type"]),
            scaler_mean=scaler["mean"],
            scaler_std=scaler["std"],
            assignments=data["assignments"],
            n_components=int(data["n_components"]),
        )


def _compute_precision_cholesky(covariances: np.ndarray, cov_type: str) -> np.ndarray:
    if cov_type == "diag":
        return 1.0 / np.sqrt(covariances)
    elif cov_type == "full":
        n = covariances.shape[0]
        prec_chol = np.empty_like(covariances)
        for k in range(n):
            cov_chol = np.linalg.cholesky(covariances[k])
            prec_chol[k] = np.linalg.inv(cov_chol).T
        return prec_chol
    elif cov_type == "spherical":
        return 1.0 / np.sqrt(covariances)
    elif cov_type == "tied":
        cov_chol = np.linalg.cholesky(covariances)
        return np.linalg.inv(cov_chol).T
    raise ValueError(f"Unknown covariance_type: {cov_type}")
