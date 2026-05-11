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

"""Cluster a dataset of NPZ files using K-Means with the elbow method.

Usage:
    python cluster.py \\
        --data_list /path/to/data_list.json \\
        --output    /path/to/result.json \\
        [--k_max 20] [--pca_components 50] [--seed 42]

Input JSON format (same as train_predictor.py):
    ["path/to/sample_0.npz", "path/to/sample_1.npz", ...]

Output JSON format:
    {
        "cluster_id0": ["path/to/sample_0.npz", ...],
        "cluster_id1": ["path/to/sample_1.npz", ...],
        ...
    }

Feature pipeline:
    ego_agent_future (80, 3) → flatten (240,) → Z-score → PCA → KMeans (elbow)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.elbow import elbow_kmeans


def load_npz_paths(data_list_json: str) -> list:
    with open(data_list_json, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_features(npz_path: str) -> np.ndarray:
    """Return the flattened ego_agent_future trajectory as a feature vector.

    ego_agent_future has shape (T, 3) with columns [x, y, heading_rad].
    The returned vector has shape (T*3,).
    """
    data = np.load(npz_path, allow_pickle=True)
    # ego_agent_future: (80, 3) — [x, y, heading_rad]
    ego_future = data["ego_agent_future"].astype(float)
    return ego_future.flatten()


def get_args():
    parser = argparse.ArgumentParser(description="Cluster NPZ dataset with elbow-method KMeans")
    parser.add_argument(
        "--data_list",
        type=str,
        required=True,
        help="Path to JSON file listing NPZ file paths (same format as train_predictor.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the clustering result JSON",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=20,
        help="Maximum number of clusters to evaluate (default: 20)",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components used to reduce the trajectory feature dimension (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()

    npz_paths = load_npz_paths(args.data_list)
    print(f"Loaded {len(npz_paths)} NPZ paths from {args.data_list}")

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

    features = np.array(features)  # (N, T*3) = (N, 240)

    # Z-score normalise so each dimension contributes equally
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    # PCA: reduce from 240-dim trajectory space to a manageable subspace
    n_components = min(args.pca_components, features_norm.shape[0], features_norm.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    features_pca = pca.fit_transform(features_norm)
    explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: {features_norm.shape[1]}-dim → {n_components}-dim "
        f"({explained * 100:.1f}% variance explained)"
    )

    print(f"Running elbow KMeans (k_max={args.k_max}, seed={args.seed})...")
    labels, optimal_k, wcss = elbow_kmeans(
        features_pca, k_max=args.k_max, random_state=args.seed
    )
    print(f"Optimal k = {optimal_k}")

    clusters: dict = defaultdict(list)
    for path, label in zip(valid_paths, labels):
        clusters[f"cluster_id{label}"].append(path)

    # Sort keys so the output is deterministic
    result = {k: clusters[k] for k in sorted(clusters, key=lambda x: int(x.replace("cluster_id", "")))}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved clustering result to {args.output}")
    for cluster_key, paths in result.items():
        print(f"  {cluster_key}: {len(paths)} samples")


if __name__ == "__main__":
    main()
