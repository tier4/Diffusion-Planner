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
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.pipeline import ElbowKMeansStrategy, cluster_trajectories


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
    return parser.parse_args()


def main():
    args = get_args()

    npz_paths = load_npz_paths(args.data_list)
    print(f"Loaded {len(npz_paths)} NPZ paths from {args.data_list}")

    strategy = ElbowKMeansStrategy(k_max=args.k_max, random_state=args.seed)
    result = cluster_trajectories(npz_paths, strategy, pca_components=args.pca_components)
    print(f"Optimal k = {strategy.n_clusters_}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved clustering result to {args.output}")
    for cluster_key, paths in result.items():
        print(f"  {cluster_key}: {len(paths)} samples")


if __name__ == "__main__":
    main()
