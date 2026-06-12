# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trajectory feature extraction and clustering pipeline.

Feature pipeline:
    ego_agent_future (80, 3) → flatten (240,) → Z-score → PCA → ClusteringStrategy

Usage example:
    strategy = ElbowKMeansStrategy(k_max=20, random_state=42)
    result = cluster_trajectories(npz_paths, strategy, pca_components=50)
    print(strategy.n_clusters_)
"""

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from elbow import elbow_kmeans

# ──────────────────────────── Strategy interface ─────────────────────────────


class ClusteringStrategy(ABC):
    """Interface for clustering algorithms used in the trajectory pipeline.

    Implementations must set ``self.n_clusters_`` (int) inside ``fit_predict``
    so that callers can inspect the chosen number of clusters after the call.
    """

    n_clusters_: int

    @abstractmethod
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Assign a cluster label to each sample.

        Args:
            features: 2-D array of shape (n_samples, n_features).

        Returns:
            Integer label array of shape (n_samples,).
            Must also set ``self.n_clusters_`` as a side-effect.
        """


# ──────────────────────────── Concrete strategies ────────────────────────────


class ElbowKMeansStrategy(ClusteringStrategy):
    """KMeans whose number of clusters is chosen automatically via the elbow method."""

    def __init__(self, k_max: int = 20, random_state: int = 42) -> None:
        self.k_max = k_max
        self.random_state = random_state

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        print(f"Running elbow KMeans (k_max={self.k_max}, seed={self.random_state})...")
        labels, optimal_k, _ = elbow_kmeans(
            features, k_max=self.k_max, random_state=self.random_state
        )
        self.n_clusters_ = optimal_k
        return labels


# ──────────────────────────── Feature extraction ─────────────────────────────


def extract_features(npz_path: str) -> np.ndarray:
    """Return the flattened ego_agent_future trajectory as a feature vector.

    ego_agent_future has shape (T, 3) with columns [x, y, heading_rad].
    The returned vector has shape (T*3,).
    """
    data = np.load(npz_path, allow_pickle=True)
    ego_future = data["ego_agent_future"].astype(float)
    return ego_future.flatten()


# ──────────────────────────── Pipeline ───────────────────────────────────────


def cluster_trajectories(
    npz_paths: list,
    strategy: ClusteringStrategy,
    pca_components: int = 50,
) -> dict:
    """Run the full trajectory clustering pipeline.

    Preprocessing (feature extraction, Z-score, PCA) is fixed; the clustering
    algorithm is delegated to ``strategy``.  After the call, ``strategy.n_clusters_``
    holds the number of clusters that were used.

    Args:
        npz_paths: list of paths to NPZ files.
        strategy: clustering algorithm to apply after preprocessing.
        pca_components: number of PCA components for dimensionality reduction.

    Returns:
        dict mapping ``"cluster_idN"`` to list of NPZ paths, sorted by id.

    Raises:
        RuntimeError: if no valid NPZ files are found.
    """
    features = []
    valid_paths = []
    for path in tqdm(npz_paths, desc="Extracting features", unit="file"):
        try:
            features.append(extract_features(path))
            valid_paths.append(path)
        except Exception as e:
            tqdm.write(f"  [warn] skipping {path}: {e}")

    if not features:
        raise RuntimeError("No valid NPZ files found.")

    features = np.array(features)

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    n_components = min(pca_components, features_norm.shape[0], features_norm.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    features_pca = pca.fit_transform(features_norm)
    explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: {features_norm.shape[1]}-dim → {n_components}-dim "
        f"({explained * 100:.1f}% variance explained)"
    )

    labels = strategy.fit_predict(features_pca)

    clusters: dict = defaultdict(list)
    for path, label in zip(valid_paths, labels):
        clusters[f"cluster_id{label}"].append(path)

    return {
        k: clusters[k] for k in sorted(clusters, key=lambda x: int(x.replace("cluster_id", "")))
    }
