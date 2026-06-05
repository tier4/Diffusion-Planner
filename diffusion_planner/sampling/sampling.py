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

"""Sample an equal number of NPZ files from each cluster produced by cluster.py.

The number of samples per cluster equals the size of the smallest cluster.
The output JSON can be passed directly to train_run.sh as TRAIN_SET_LIST.

Two sampling strategies are available:

  random   : uniform random sampling (original behaviour, requires --seed)
  coverage : greedy Farthest Point Sampling (FPS) in PCA-whitened feature space,
             maximising the coverage of each cluster's data distribution.
             Features are re-extracted from NPZ files independently of cluster.py.
             Euclidean distance in PCA-whitened space equals Mahalanobis distance
             in the original feature space, so cluster shape is properly accounted for.
             FPS is deterministic (no seed required).

Usage:
    # Random sampling
    python sampling.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output       /path/to/sampled.json \\
        --method random --seed 42

    # Read seed from a previous random-sampling output
    python sampling.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output       /path/to/sampled.json \\
        --method random --seed_json /path/to/previous_sampled.json

    # Coverage-maximising sampling
    python sampling.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output       /path/to/sampled.json \\
        --method coverage [--pca_components 50] [--num_workers 8]

Input JSON format (cluster.py output):
    {
        "cluster_id0": ["path/to/sample_0.npz", ...],
        "cluster_id1": ["path/to/sample_1.npz", ...],
        ...
    }

Output JSON format:
    # random
    {"method": "random", "seed": 42, "files": ["path/to/sample_0.npz", ...]}

    # coverage
    {"method": "coverage", "pca_components": 50, "files": ["path/to/sample_0.npz", ...]}
"""

import argparse
import gc
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────── Feature extraction (local copy) ─────────────────────
# Duplicated from utils/pipeline.py to keep sampling.py independent of cluster.py.


def _extract_features(npz_path: str) -> np.ndarray:
    """Return a feature vector for one NPZ file.

    ego_agent_future (T, 3) → flattened (T*3,)
    ego_current_state[4:10] → [vx, vy, ax, ay, steering_angle, yaw_rate]
    Concatenated shape: (T*3 + 6,)
    NPZ file is closed immediately after reading.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        ego_future = data["ego_agent_future"].astype(float)
        ego_current = data["ego_current_state"].astype(float)
        result = np.concatenate([ego_future.flatten(), ego_current[4:10]])
    return result


def _load_features_parallel(
    paths: list, max_workers: int, desc: str
) -> tuple[np.ndarray, list]:
    """Load and extract features from NPZ files in parallel using threads.

    Returns (features array of shape (m, d), valid_paths list of length m)
    where m <= len(paths) after skipping unreadable files.
    Explicitly deletes references to avoid memory bloat.
    """
    # Submit all tasks and preserve original order via index
    order: list[tuple[int, np.ndarray | None, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_extract_features, p): (i, p) for i, p in enumerate(paths)
        }
        pbar = tqdm(total=len(paths), desc=desc, leave=False)
        for future in as_completed(future_to_idx):
            i, p = future_to_idx[future]
            try:
                feat = future.result()
                order.append((i, feat, p))
            except Exception as e:
                tqdm.write(f"  [warn] skipping {p}: {e}")
                order.append((i, None, p))
            pbar.update(1)
        pbar.close()
        # Explicitly delete future mapping to free thread results
        del future_to_idx

    order.sort(key=lambda x: x[0])
    features = [feat for _, feat, _ in order if feat is not None]
    valid = [p for _, feat, p in order if feat is not None]

    # Clear order to free memory
    del order

    if not features:
        return np.empty((0, 0)), []

    result = np.array(features, copy=True)
    # Delete intermediate feature list
    del features
    gc.collect()
    return result, valid


# ──────────────────────── Greedy Farthest Point Sampling ─────────────────────


def _fps_indices(X: np.ndarray, k: int, desc: str = "") -> np.ndarray:
    """Return indices of k points selected by greedy Farthest Point Sampling.

    The first point is the one farthest from the centroid (deterministic).
    Each subsequent point maximises the minimum distance to the already-selected set.

    Distance formula uses precomputed squared norms to avoid (n, d) temporaries:
        ||x - y||^2 = ||x||^2 - 2 * x @ y + ||y||^2
    The inner product x @ y is dispatched as a BLAS DGEMV, which is faster
    than the broadcast subtraction (X - X[idx]) used in the naive formulation.
    Complexity: O(n * k).
    """
    n = len(X)

    # Precompute ||x_i||^2 once — reused every iteration
    sq_norms = np.einsum("ij,ij->i", X, X)  # shape (n,)

    centroid = X.mean(axis=0)
    first = int(np.argmax(sq_norms - 2.0 * (X @ centroid)))

    selected = [first]
    min_sq_dists = sq_norms - 2.0 * (X @ X[first]) + sq_norms[first]
    np.maximum(min_sq_dists, 0.0, out=min_sq_dists)  # guard numerical noise

    for _ in tqdm(range(k - 1), desc=desc, leave=False):
        idx = int(np.argmax(min_sq_dists))
        selected.append(idx)
        new_sq_dists = sq_norms - 2.0 * (X @ X[idx]) + sq_norms[idx]
        np.maximum(new_sq_dists, 0.0, out=new_sq_dists)
        np.minimum(min_sq_dists, new_sq_dists, out=min_sq_dists)

    return np.array(selected)


# ──────────────────────── Coverage sampling per cluster ──────────────────────


def _coverage_sample_cluster(
    cluster_key: str, paths: list, k: int, pca_components: int, num_workers: int
) -> list:
    """Select k paths from one cluster using FPS in PCA-whitened feature space.

    Pipeline (per cluster, independent of cluster.py):
        extract_features → Z-score → PCA → whitening → FPS
    Whitening (dividing each PC by its standard deviation) makes Euclidean
    distance in the transformed space equivalent to Mahalanobis distance in
    the original feature space, accounting for the cluster's shape.
    Explicitly deletes intermediate arrays to avoid memory accumulation.
    """
    features, valid = _load_features_parallel(
        paths, max_workers=num_workers, desc=f"{cluster_key} load"
    )
    if len(valid) == 0:
        return []
    if k >= len(valid):
        result = valid
        del features
        gc.collect()
        return result

    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-8
    Xn = (features - mean) / std
    del features, mean, std
    gc.collect()

    n_comp = min(pca_components, len(valid) - 1, Xn.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    Xp = pca.fit_transform(Xn)
    del Xn
    gc.collect()

    # Whitening: Euclidean in whitened space = Mahalanobis in original space
    ev = np.sqrt(pca.explained_variance_) + 1e-8
    Xw = Xp / ev
    del Xp, pca, ev
    gc.collect()

    indices = _fps_indices(Xw, k, desc=f"{cluster_key} FPS ")
    del Xw
    gc.collect()

    result = [valid[i] for i in indices]
    del valid, indices
    gc.collect()
    return result


# ──────────────────────────────── CLI ────────────────────────────────────────


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample equal-sized subsets from each cluster"
    )
    parser.add_argument(
        "--cluster_json",
        type=str,
        required=True,
        help="Path to the clustering result JSON produced by cluster.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the sampled file list JSON",
    )
    parser.add_argument(
        "--method",
        choices=["random", "coverage"],
        default="random",
        help="Sampling strategy: 'random' (uniform) or 'coverage' (FPS, deterministic). Default: random",
    )

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        help="Random seed (required for --method random)",
    )
    seed_group.add_argument(
        "--seed_json",
        type=str,
        help="JSON file containing a 'seed' key, e.g. a previous sampling output (--method random only)",
    )

    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components for coverage sampling (default: 50)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Parallel workers for NPZ loading in coverage mode (default: CPU count). "
        "Reduce this if memory is insufficient (e.g., 1-4 for large clusters)",
    )

    return parser.parse_args()


def _resolve_seed(args) -> int:
    if args.seed is not None:
        return args.seed
    data = load_json(args.seed_json)
    if "seed" not in data:
        print(f"ERROR: 'seed' key not found in {args.seed_json}", file=sys.stderr)
        sys.exit(1)
    return int(data["seed"])


def main():
    args = get_args()

    if args.method == "random" and args.seed is None and args.seed_json is None:
        print("ERROR: --method random requires --seed or --seed_json", file=sys.stderr)
        sys.exit(1)

    clusters: dict = load_json(args.cluster_json)
    if not clusters:
        print("ERROR: cluster JSON is empty", file=sys.stderr)
        sys.exit(1)

    min_count = min(len(paths) for paths in clusters.values())
    print(f"Clusters        : {len(clusters)}")
    print(f"Min cluster size: {min_count}")
    print(f"Method          : {args.method}")

    sampled: list[str] = []

    if args.method == "random":
        seed = _resolve_seed(args)
        print(f"Seed            : {seed}")
        rng = random.Random(seed)
        for cluster_key, paths in clusters.items():
            chosen = rng.sample(paths, min_count)
            sampled.extend(chosen)
            print(f"  {cluster_key}: {len(paths)} → sampled {min_count}")
        result = {"method": "random", "seed": seed, "files": sampled}

    else:  # coverage
        print(f"PCA components  : {args.pca_components}")
        print(f"Num workers     : {args.num_workers}")
        for cluster_key, paths in tqdm(clusters.items(), desc="Clusters", position=0):
            chosen = _coverage_sample_cluster(
                cluster_key, paths, min_count, args.pca_components, args.num_workers
            )
            sampled.extend(chosen)
            tqdm.write(f"  {cluster_key}: {len(paths)} → sampled {len(chosen)}")
            gc.collect()
        result = {"method": "coverage", "pca_components": args.pca_components, "files": sampled}

    print(f"Total sampled   : {len(sampled)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
