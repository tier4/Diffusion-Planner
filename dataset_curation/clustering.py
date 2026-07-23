from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_embeddings(
    embeddings: np.ndarray,
    k: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, KMeans]:
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    paths: list[str],
    features_df=None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plots = 2
    color_by = {"cluster_id": labels}
    if features_df is not None:
        for col in ["heading_change_deg", "ego_speed", "n_active_neighbors", "travel_distance"]:
            if col in features_df.columns:
                vals = features_df[col].reindex([p for p in paths if p in features_df.index])
                if len(vals) == len(coords):
                    color_by[col] = vals.values
                    n_plots += 1

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, vals) in enumerate(color_by.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            s=3,
            alpha=0.5,
            cmap="tab20" if name == "cluster_id" else "viridis",
        )
        ax.set_title(f"UMAP colored by {name}")
        plt.colorbar(sc, ax=ax)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "umap_visualization.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    unique, counts = np.unique(labels, return_counts=True)
    ax2.bar(unique, counts)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Count")
    ax2.set_title("Cluster Size Distribution")
    plt.tight_layout()
    fig2.savefig(output_dir / "cluster_sizes.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings + UMAP visualization")
    parser.add_argument("--embeddings", required=True, help="embeddings.npy from experiment 4")
    parser.add_argument("--paths", required=True, help="embedding_paths.json from experiment 4")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--features", default=None, help="features.parquet for coloring UMAP")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    with open(args.paths) as f:
        paths = json.load(f)

    print(f"Clustering {len(embeddings)} embeddings into {args.k} clusters...")
    labels, model = cluster_embeddings(embeddings, k=args.k)

    print("Computing UMAP projection...")
    coords = compute_umap(embeddings)

    import pandas as pd

    features_df = None
    if args.features:
        features_df = pd.read_parquet(args.features)

    _plot_umap(coords, labels, output_dir, paths, features_df)

    result = pd.DataFrame(
        {
            "npz_path": paths,
            "cluster_id": labels,
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        }
    )
    result.to_parquet(output_dir / "clustering_results.parquet", index=False)
    np.save(output_dir / "umap_coords.npy", coords)

    print(f"\nCluster distribution:")
    for cid in sorted(set(labels)):
        count = (labels == cid).sum()
        print(f"  Cluster {cid}: {count} scenes ({100 * count / len(labels):.1f}%)")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
