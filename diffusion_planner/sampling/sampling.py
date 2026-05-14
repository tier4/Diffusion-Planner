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

Usage:
    # Specify seed directly
    python sampling.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output       /path/to/sampled.json \\
        --seed 42

    # Read seed from a previous sampling output
    python sampling.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output       /path/to/sampled.json \\
        --seed_json    /path/to/previous_sampled.json

Input JSON format (cluster.py output):
    {
        "cluster_id0": ["path/to/sample_0.npz", ...],
        "cluster_id1": ["path/to/sample_1.npz", ...],
        ...
    }

Output JSON format:
    {
        "seed": 42,
        "files": ["path/to/sample_0.npz", ...]
    }
"""

import argparse
import json
import random
import sys
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample equal-sized subsets from each cluster (balanced sampling)"
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

    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seed",
        type=int,
        help="Random seed value (integer)",
    )
    seed_group.add_argument(
        "--seed_json",
        type=str,
        help="Path to a JSON file that contains a 'seed' key (e.g. a previous sampling output)",
    )

    return parser.parse_args()


def resolve_seed(args) -> int:
    if args.seed is not None:
        return args.seed

    data = load_json(args.seed_json)
    if "seed" not in data:
        print(f"ERROR: 'seed' key not found in {args.seed_json}", file=sys.stderr)
        sys.exit(1)
    return int(data["seed"])


def main():
    args = get_args()
    seed = resolve_seed(args)

    clusters: dict = load_json(args.cluster_json)
    if not clusters:
        print("ERROR: cluster JSON is empty", file=sys.stderr)
        sys.exit(1)

    min_count = min(len(paths) for paths in clusters.values())
    print(f"Clusters      : {len(clusters)}")
    print(f"Min cluster size: {min_count}")
    print(f"Seed          : {seed}")

    rng = random.Random(seed)
    sampled: list[str] = []
    for cluster_key in sorted(clusters.keys()):
        paths = clusters[cluster_key]
        chosen = rng.sample(paths, min_count)
        sampled.extend(chosen)
        print(f"  {cluster_key}: {len(paths)} → sampled {min_count}")

    print(f"Total sampled : {len(sampled)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"seed": seed, "files": sampled}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
