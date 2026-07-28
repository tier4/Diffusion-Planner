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


def extract_features_enriched(
    npz_path: str,
    top_k: int = 20,
    temporal_hz: int = 2,
) -> tuple:
    """Extract ego block and neighbor block feature vectors for enriched clustering.

    Returns:
        (ego_block, neighbor_block) — two 1-D float arrays.
    """
    if temporal_hz <= 0:
        raise ValueError(f"temporal_hz={temporal_hz} must be positive")
    if 10 % temporal_hz != 0:
        raise ValueError(f"temporal_hz={temporal_hz} must evenly divide 10")

    with np.load(npz_path, allow_pickle=True) as data:
        step = 10 // temporal_hz
        n_past = data["ego_agent_past"].shape[0]
        n_future = data["ego_agent_future"].shape[0]
        past_idx = np.arange(0, n_past, step)
        future_idx = np.unique(np.append(np.arange(0, n_future, step), n_future - 1))

        ego_past = data["ego_agent_past"].astype(float)[past_idx]
        ego_future = data["ego_agent_future"].astype(float)[future_idx]
        ego_state = data["ego_current_state"].astype(float)

        nbr_past = data["neighbor_agents_past"].astype(float)
        nbr_future = data["neighbor_agents_future"].astype(float)

    active_mask = np.any(nbr_past.reshape(nbr_past.shape[0], -1) != 0, axis=1)
    nbr_pos = nbr_past[:, -1, :2]
    distances = np.linalg.norm(nbr_pos, axis=1)
    distances[~active_mask] = np.inf

    # ``past_idx``/``future_idx`` come from the EGO array lengths and are then applied
    # to the neighbor arrays. Shorter neighbor horizons raise IndexError (swallowed by
    # the caller's per-file except, so it looks like a bad file rather than a schema
    # mismatch); longer ones silently drop the tail. Say so explicitly.
    if nbr_past.shape[1] != n_past or nbr_future.shape[1] != n_future:
        raise ValueError(
            f"neighbor/ego horizon mismatch: neighbor_agents_past T="
            f"{nbr_past.shape[1]} vs ego_agent_past T={n_past}, "
            f"neighbor_agents_future T={nbr_future.shape[1]} vs "
            f"ego_agent_future T={n_future}"
        )

    sorted_idx = np.argsort(distances)[:top_k]
    # ``[:top_k]`` silently truncates when the file has fewer neighbor slots than
    # top_k, producing a NARROWER feature vector than its peers. That only surfaces
    # later, when every vector is stacked into one matrix. Pad to exactly top_k with
    # zero trajectories and the -1.0 "absent" distance sentinel instead.
    n_pad = top_k - len(sorted_idx)
    topk_past = nbr_past[sorted_idx][:, past_idx]
    topk_future = nbr_future[sorted_idx][:, future_idx]
    topk_dist = distances[sorted_idx].copy()
    if n_pad > 0:
        topk_past = np.concatenate([topk_past, np.zeros((n_pad, *topk_past.shape[1:]))])
        topk_future = np.concatenate([topk_future, np.zeros((n_pad, *topk_future.shape[1:]))])
        topk_dist = np.concatenate([topk_dist, np.full(n_pad, np.inf)])
    topk_dist[topk_dist == np.inf] = -1.0

    ego_block = np.concatenate(
        [
            ego_past.flatten(),
            ego_future.flatten(),
            ego_state,
            topk_dist,
        ]
    )
    neighbor_block = np.concatenate(
        [
            topk_past.flatten(),
            topk_future.flatten(),
        ]
    )
    return ego_block, neighbor_block


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


def cluster_trajectories_enriched(
    npz_paths: list,
    strategy: ClusteringStrategy,
    pca_components: int = 50,
    neighbor_pca_components: int = 50,
    top_k: int = 20,
    temporal_hz: int = 2,
) -> dict:
    """Run the enriched two-stage PCA clustering pipeline.

    Stage 1: Z-score ego and neighbor blocks independently, PCA the neighbor block.
    Stage 2: Concatenate, Z-score, final PCA, then delegate to strategy.
    """
    if top_k < 1:
        raise ValueError(f"top_k={top_k} must be >= 1")
    if neighbor_pca_components < 1:
        raise ValueError(f"neighbor_pca_components={neighbor_pca_components} must be >= 1")
    if pca_components < 1:
        raise ValueError(f"pca_components={pca_components} must be >= 1")

    ego_list = []
    nbr_list = []
    valid_paths = []
    # Width of the first accepted file. Every later file must match it: a corpus
    # mixing npz v2 (3-col neighbor_agents_future) with npz v3 (4-col [x,y,cos,sin]),
    # or differing horizons, yields ragged vectors that only blow up at
    # ``np.array(nbr_list)`` -- after all 150,000 files have been read off storage.
    # Fail on file 2, not file 150,000.
    widths: tuple[int, int] | None = None
    for path in tqdm(npz_paths, desc="Extracting enriched features", unit="file"):
        try:
            ego, nbr = extract_features_enriched(path, top_k=top_k, temporal_hz=temporal_hz)
            if not (np.all(np.isfinite(ego)) and np.all(np.isfinite(nbr))):
                # NaN/Inf survive extraction and only fail inside PCA, at the end.
                raise ValueError("feature vector contains NaN or Inf")
        except Exception as e:
            tqdm.write(f"  [warn] skipping {path}: {e}")
            continue

        # Deliberately OUTSIDE the try: a width mismatch is a corpus-level problem,
        # not a bad file. Skipping it would silently drop every minority-layout file
        # and emit a cluster JSON covering only part of the dataset.
        if widths is None:
            widths = (ego.size, nbr.size)
        elif (ego.size, nbr.size) != widths:
            raise ValueError(
                f"{path}: feature width {(ego.size, nbr.size)} != {widths} from the "
                f"first accepted file. Mixed NPZ layouts (e.g. npz v2's 3-col "
                f"neighbor_agents_future vs npz v3's 4-col [x, y, cos, sin]) cannot be "
                f"clustered together. Normalize the corpus to one layout first."
            )

        ego_list.append(ego)
        nbr_list.append(nbr)
        valid_paths.append(path)

    if not ego_list:
        raise RuntimeError("No valid NPZ files found.")

    skipped = len(npz_paths) - len(valid_paths)
    print(
        f"Extracted {len(valid_paths)}/{len(npz_paths)} files "
        f"({skipped} skipped, {100.0 * skipped / len(npz_paths):.1f}%)"
    )
    if skipped:
        # Skipped files stay in the training data_list, where the sampler gives them
        # neutral weight -- so a high skip rate quietly defeats the balancing this
        # pipeline exists to provide.
        print(
            "  [warn] skipped files are absent from the cluster JSON; training will "
            "assign them neutral weight. Investigate before using this JSON if the "
            "skip rate is material."
        )

    ego_features = np.array(ego_list)
    nbr_features = np.array(nbr_list)

    ego_mean = ego_features.mean(axis=0)
    ego_std = ego_features.std(axis=0) + 1e-8
    ego_norm = (ego_features - ego_mean) / ego_std

    nbr_mean = nbr_features.mean(axis=0)
    nbr_std = nbr_features.std(axis=0) + 1e-8
    nbr_norm = (nbr_features - nbr_mean) / nbr_std

    n_nbr = min(neighbor_pca_components, nbr_norm.shape[0], nbr_norm.shape[1])
    nbr_pca = PCA(n_components=n_nbr, random_state=0)
    nbr_reduced = nbr_pca.fit_transform(nbr_norm)
    print(
        f"Neighbor PCA: {nbr_norm.shape[1]}-dim → {n_nbr}-dim "
        f"({nbr_pca.explained_variance_ratio_.sum() * 100:.1f}% variance explained)"
    )

    combined = np.hstack([ego_norm, nbr_reduced])
    comb_mean = combined.mean(axis=0)
    comb_std = combined.std(axis=0) + 1e-8
    combined_norm = (combined - comb_mean) / comb_std

    n_final = min(pca_components, combined_norm.shape[0], combined_norm.shape[1])
    final_pca = PCA(n_components=n_final, random_state=0)
    features_pca = final_pca.fit_transform(combined_norm)
    print(
        f"Final PCA: {combined_norm.shape[1]}-dim → {n_final}-dim "
        f"({final_pca.explained_variance_ratio_.sum() * 100:.1f}% variance explained)"
    )

    labels = strategy.fit_predict(features_pca)

    clusters = defaultdict(list)
    for path, label in zip(valid_paths, labels):
        clusters[f"cluster_id{label}"].append(path)

    return {
        k: clusters[k] for k in sorted(clusters, key=lambda x: int(x.replace("cluster_id", "")))
    }
