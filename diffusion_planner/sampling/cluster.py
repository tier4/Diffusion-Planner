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
        [--k_max 20] [--pca_components 50] [--seed 42] \\
        [--mode trajectory|enriched]

    Enriched mode only (ignored with --mode trajectory):
        [--top_k_neighbors 20] [--neighbor_pca_components 50] [--temporal_hz 2]

Input JSON format (same as train_predictor.py):
    ["path/to/sample_0.npz", "path/to/sample_1.npz", ...]

Output JSON format:
    {
        "cluster_id0": ["path/to/sample_0.npz", ...],
        "cluster_id1": ["path/to/sample_1.npz", ...],
        ...
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.pipeline import ElbowKMeansStrategy, cluster_trajectories, cluster_trajectories_enriched


def load_npz_paths(data_list_json: str) -> list:
    with open(data_list_json, "r", encoding="utf-8") as f:
        return json.load(f)


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
    parser.add_argument(
        "--mode",
        type=str,
        default="trajectory",
        choices=["trajectory", "enriched"],
        help="Clustering mode: 'trajectory' (ego future only) or 'enriched' (ego + neighbors + state)",
    )
    parser.add_argument(
        "--top_k_neighbors",
        type=int,
        default=20,
        help="Number of nearest neighbors to keep (enriched mode only, default: 20)",
    )
    parser.add_argument(
        "--neighbor_pca_components",
        type=int,
        default=50,
        help="PCA components for neighbor block (enriched mode only, default: 50)",
    )
    parser.add_argument(
        "--temporal_hz",
        type=int,
        default=2,
        help="Temporal downsample rate in hz (enriched mode only, default: 2)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Validate before the extraction pass, not after it. --temporal_hz is otherwise
    # only checked per-file inside cluster_trajectories_enriched's swallowed
    # try/except, so a bad value warns 150,000 times and then raises
    # "No valid NPZ files found." -- which blames the data, not the flag.
    if args.mode == "enriched":
        if args.top_k_neighbors < 1:
            raise SystemExit(f"--top_k_neighbors must be >= 1, got {args.top_k_neighbors}")
        if args.neighbor_pca_components < 1:
            raise SystemExit(
                f"--neighbor_pca_components must be >= 1, got {args.neighbor_pca_components}"
            )
        if args.temporal_hz < 1 or 10 % args.temporal_hz != 0:
            raise SystemExit(
                f"--temporal_hz must be a positive divisor of 10 (1, 2, 5, 10), "
                f"got {args.temporal_hz}"
            )
    if args.pca_components < 1:
        raise SystemExit(f"--pca_components must be >= 1, got {args.pca_components}")

    npz_paths = load_npz_paths(args.data_list)
    print(f"Loaded {len(npz_paths)} NPZ paths from {args.data_list}")

    strategy = ElbowKMeansStrategy(k_max=args.k_max, random_state=args.seed)
    if args.mode == "enriched":
        result = cluster_trajectories_enriched(
            npz_paths,
            strategy,
            pca_components=args.pca_components,
            neighbor_pca_components=args.neighbor_pca_components,
            top_k=args.top_k_neighbors,
            temporal_hz=args.temporal_hz,
        )
    else:
        result = cluster_trajectories(npz_paths, strategy, pca_components=args.pca_components)
    print(f"Optimal k = {strategy.n_clusters_}")

    # Write-then-rename. A plain open(..., "w") truncates the destination first, so a
    # crash mid-dump -- or a training job reading the file at that moment -- sees a
    # half-written JSON. os.replace is atomic within a filesystem.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    os.replace(tmp_path, output_path)

    print(f"Saved clustering result to {args.output}")
    for cluster_key, paths in result.items():
        print(f"  {cluster_key}: {len(paths)} samples")


if __name__ == "__main__":
    main()
