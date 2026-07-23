from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def semdedup(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    n = len(embeddings)
    keep = np.ones(n, dtype=bool)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normed = embeddings / norms

    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) <= 1:
            continue

        cluster_emb = normed[indices]
        centroid = cluster_emb.mean(axis=0)
        centroid_dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        sorted_order = np.argsort(centroid_dists)

        sim = cosine_similarity(cluster_emb)
        removed = set()
        for i in reversed(sorted_order):
            if i in removed:
                continue
            for j in sorted_order:
                if j == i or j in removed:
                    continue
                if sim[i, j] >= threshold:
                    removed.add(i)
                    break

        for local_idx in removed:
            keep[indices[local_idx]] = False

    return keep


def main():
    parser = argparse.ArgumentParser(description="SemDeDup: semantic deduplication")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy")
    parser.add_argument("--clustering", required=True, help="clustering_results.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.85,0.90,0.95",
        help="Comma-separated similarity thresholds to evaluate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    cluster_df = pd.read_parquet(args.clustering)
    labels = cluster_df["cluster_id"].values
    paths = cluster_df["npz_path"].tolist()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"Running SemDeDup on {len(embeddings)} samples...")
    for thresh in thresholds:
        keep = semdedup(embeddings, labels, threshold=thresh)
        n_kept = keep.sum()
        n_removed = len(keep) - n_kept
        print(f"\nThreshold {thresh}:")
        print(f"  Kept: {n_kept} ({100 * n_kept / len(keep):.1f}%)")
        print(f"  Removed: {n_removed} ({100 * n_removed / len(keep):.1f}%)")

        kept_paths = [p for p, k in zip(paths, keep) if k]
        removed_paths = [p for p, k in zip(paths, keep) if not k]
        with open(output_dir / f"kept_t{thresh}.json", "w") as f:
            json.dump(kept_paths, f)
        with open(output_dir / f"removed_t{thresh}_sample.json", "w") as f:
            json.dump(removed_paths[:100], f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("Inspect removed pairs with: python -m scenario_generation.visualize <path1> <path2>")


if __name__ == "__main__":
    main()
