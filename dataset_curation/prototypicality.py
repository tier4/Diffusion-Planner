from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_rarity_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(embeddings))
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        scores[i] = np.linalg.norm(emb - centroids[label])
    return scores


def main():
    parser = argparse.ArgumentParser(description="Prototypicality / rarity scoring")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy")
    parser.add_argument("--clustering", required=True, help="clustering_results.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    cluster_df = pd.read_parquet(args.clustering)
    labels = cluster_df["cluster_id"].values
    paths = cluster_df["npz_path"].tolist()

    n_clusters = len(np.unique(labels))
    centroids = np.stack([embeddings[labels == k].mean(axis=0) for k in range(n_clusters)])

    scores = compute_rarity_scores(embeddings, labels, centroids)

    result = pd.DataFrame(
        {
            "npz_path": paths,
            "rarity_score": scores,
            "cluster_id": labels,
        }
    )
    result.to_parquet(output_dir / "rarity_scores.parquet", index=False)

    sorted_idx = np.argsort(scores)
    top_rare = [paths[i] for i in sorted_idx[-args.top_k :]][::-1]
    top_typical = [paths[i] for i in sorted_idx[: args.top_k]]
    with open(output_dir / f"top{args.top_k}_rarest.json", "w") as f:
        json.dump(top_rare, f, indent=2)
    with open(output_dir / f"top{args.top_k}_typical.json", "w") as f:
        json.dump(top_typical, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(scores, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Rarity Score (L2 to centroid)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rarity Score Distribution")

    if "umap_x" in cluster_df.columns:
        sc = axes[1].scatter(
            cluster_df["umap_x"],
            cluster_df["umap_y"],
            c=scores,
            s=3,
            alpha=0.5,
            cmap="plasma",
        )
        axes[1].set_title("UMAP colored by rarity score")
        plt.colorbar(sc, ax=axes[1])
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "rarity_analysis.png", dpi=150)
    plt.close(fig)

    print(f"Rarity score stats:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")
    print(f"\nTop-{args.top_k} rarest / most typical scene paths saved.")
    print(f"Inspect with: python -m scenario_generation.visualize <path>")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
