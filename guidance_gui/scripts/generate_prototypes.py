"""Generate prototype trajectories by K-means clustering of training data.

Prototypes are representative ego trajectory shapes in the ego-centric frame,
analogous to the motion mode anchors used in MTR (Motion Transformer).

Output files (saved to --output path):
    prototypes_k<K>.npy        shape (K, 80, 2)  -- cluster centres, metres
    prototypes_k<K>_counts.npy shape (K,)         -- member counts per cluster

Usage
-----
python guidance_gui/scripts/generate_prototypes.py \\
  --npz_list  /path/to/train.json \\
  --k         16 \\
  --output    guidance_gui/prototypes_k16.npy \\
  --max_samples 50000
"""

import argparse
import json
import os
import random

import numpy as np
from scipy.cluster.vq import kmeans2, whiten


def load_future_xy(npz_path: str) -> np.ndarray | None:
    """Load the xy columns from ego_agent_future in one npz file.

    Returns (80, 2) float32 or None if the key is missing.
    """
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    if "ego_agent_future" not in data:
        return None
    future = data["ego_agent_future"]  # (80, 3) = [x, y, yaw_rad]
    return future[:, :2].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="K-means prototype extraction")
    parser.add_argument("--npz_list", required=True, help="Path to path_list.json")
    parser.add_argument("--k", type=int, default=16, help="Number of clusters")
    parser.add_argument("--output", required=True, help="Output path, e.g. prototypes_k16.npy")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Maximum number of samples to load (random subset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.npz_list) as f:
        all_paths = json.load(f)

    print(f"Total paths in list: {len(all_paths)}")

    if len(all_paths) > args.max_samples:
        paths = random.sample(all_paths, args.max_samples)
        print(f"Sampled {args.max_samples} paths")
    else:
        paths = all_paths

    trajectories = []
    skipped = 0
    for p in paths:
        xy = load_future_xy(p)
        if xy is None or xy.shape != (80, 2):
            skipped += 1
            continue
        trajectories.append(xy)

    print(f"Loaded {len(trajectories)} trajectories ({skipped} skipped)")

    # Flatten to (N, 160) for KMeans
    X = np.stack(trajectories, axis=0)  # (N, 80, 2)
    N = X.shape[0]
    X_flat = X.reshape(N, -1)  # (N, 160)

    print(f"Running KMeans(k={args.k}) on {N} samples of shape {X_flat.shape}…")
    # Whitening normalises each feature dimension; we undo it after clustering
    # so the centres are back in metres.
    whitened = whiten(X_flat)
    std = X_flat.std(axis=0).clip(min=1e-6)
    centers_w, labels = kmeans2(whitened, args.k, iter=20, seed=args.seed, minit="points")
    centers = (centers_w * std).reshape(args.k, 80, 2)  # (K, 80, 2)

    # Count members per cluster
    counts = np.bincount(labels, minlength=args.k)  # (K,)

    # Sort by descending frequency so prototype 0 = most common motion mode
    order = np.argsort(-counts)
    centers = centers[order]
    counts = counts[order]

    # Derive output paths
    output_path = args.output
    stem, ext = os.path.splitext(output_path)
    counts_path = stem + "_counts" + ext

    np.save(output_path, centers)
    np.save(counts_path, counts)

    print(f"Saved centres  → {output_path}  shape {centers.shape}")
    print(f"Saved counts   → {counts_path}  shape {counts.shape}")
    print(f"Cluster counts: {counts.tolist()}")


if __name__ == "__main__":
    main()
