from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def detect_outliers(
    df: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.05,
) -> pd.Series:
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    if method == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = model.fit_predict(X)
    elif method == "lof":
        model = LocalOutlierFactor(contamination=contamination, n_neighbors=20, n_jobs=-1)
        preds = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'isolation_forest' or 'lof'.")

    return pd.Series(preds == -1, index=df.index, name=f"outlier_{method}")


def _plot_outlier_scatter(
    df: pd.DataFrame, flags: pd.Series, output_dir: Path, method: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_features = df.std().nlargest(6).index.tolist()
    n_pairs = min(len(top_features) * (len(top_features) - 1) // 2, 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    pair_idx = 0
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            if pair_idx >= len(axes):
                break
            ax = axes[pair_idx]
            normal = ~flags
            ax.scatter(
                df.loc[normal, top_features[i]],
                df.loc[normal, top_features[j]],
                s=5,
                alpha=0.3,
                label="normal",
            )
            ax.scatter(
                df.loc[flags, top_features[i]],
                df.loc[flags, top_features[j]],
                s=20,
                c="red",
                alpha=0.8,
                label="outlier",
            )
            ax.set_xlabel(top_features[i])
            ax.set_ylabel(top_features[j])
            ax.legend(fontsize=7)
            pair_idx += 1
    for k in range(pair_idx, len(axes)):
        axes[k].set_visible(False)
    plt.suptitle(f"Outlier Detection: {method}")
    plt.tight_layout()
    fig.savefig(output_dir / f"outlier_scatter_{method}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Detect outliers in feature matrix")
    parser.add_argument("--features", required=True, help="Parquet file from experiment 1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)

    results = pd.DataFrame(index=df.index)
    for method in ["isolation_forest", "lof"]:
        flags = detect_outliers(df, method=method, contamination=args.contamination)
        results[f"outlier_{method}"] = flags
        n_flagged = flags.sum()
        print(f"{method}: {n_flagged}/{len(df)} flagged ({100 * n_flagged / len(df):.1f}%)")
        _plot_outlier_scatter(df, flags, output_dir, method)

    results["outlier_both"] = results["outlier_isolation_forest"] & results["outlier_lof"]
    results.to_parquet(output_dir / "outlier_flags.parquet")

    both_count = results["outlier_both"].sum()
    print(f"\nFlagged by both methods: {both_count}")
    flagged_paths = results.index[results["outlier_both"]].tolist()
    if flagged_paths:
        import json

        with open(output_dir / "outliers_both.json", "w") as f:
            json.dump(flagged_paths[:50], f, indent=2)
        print(f"Top {min(50, len(flagged_paths))} outlier paths saved to outliers_both.json")
        print("Inspect with: python -m scenario_generation.visualize <path>")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
