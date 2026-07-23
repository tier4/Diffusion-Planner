from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import entropy


def _heuristic_labels(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("lane_follow", index=df.index, name="maneuver")
    if "heading_change_deg" in df.columns:
        labels[df["heading_change_deg"] > 20] = "left_turn"
        labels[df["heading_change_deg"] < -20] = "right_turn"
        labels[df["heading_change_deg"].abs() > 20] = np.where(
            df.loc[df["heading_change_deg"].abs() > 20, "heading_change_deg"] > 0,
            "left_turn",
            "right_turn",
        )
    if "closest_neighbor_dist" in df.columns:
        labels[(df["closest_neighbor_dist"] < 5) & (labels == "lane_follow")] = "avoidance"
    return labels


def train_difficulty_classifier(
    df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[lgb.Booster, np.ndarray]:
    label_map = {name: idx for idx, name in enumerate(sorted(labels.unique()))}
    y = labels.map(label_map).values
    n_classes = len(label_map)

    ds = lgb.Dataset(df.values, label=y, feature_name=list(df.columns))
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "verbose": -1,
        "seed": 42,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    model = lgb.train(params, ds, num_boost_round=200)
    probs = model.predict(df.values)
    scores = np.array([entropy(p) for p in probs])
    return model, scores


def compute_difficulty_scores(
    df: pd.DataFrame,
    model: lgb.Booster,
) -> np.ndarray:
    probs = model.predict(df.values)
    return np.array([entropy(p) for p in probs])


def _plot_difficulty(
    df: pd.DataFrame, scores: np.ndarray, labels: pd.Series, model: lgb.Booster, output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(scores, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Prediction Entropy (Difficulty)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Difficulty Score Distribution")

    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.Series(importance, index=df.columns).sort_values(ascending=True)
    feat_imp.tail(15).plot.barh(ax=axes[1])
    axes[1].set_title("Top 15 Feature Importance (Gain)")

    unique_labels = sorted(labels.unique())
    for lbl in unique_labels:
        mask = labels == lbl
        axes[2].hist(scores[mask], bins=30, alpha=0.5, label=lbl)
    axes[2].legend()
    axes[2].set_xlabel("Difficulty Score")
    axes[2].set_title("Difficulty by Maneuver Type")

    plt.tight_layout()
    fig.savefig(output_dir / "difficulty_analysis.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LightGBM difficulty scoring")
    parser.add_argument("--features", required=True, help="Parquet from experiment 1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default=None, help="JSON mapping npz_path -> maneuver label")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)

    if args.labels:
        with open(args.labels) as f:
            label_dict = json.load(f)
        labels = pd.Series(label_dict).reindex(df.index).fillna("unknown")
    else:
        print("No labels provided, deriving heuristic labels from features...")
        labels = _heuristic_labels(df)

    print(f"Label distribution:\n{labels.value_counts().to_string()}\n")
    model, scores = train_difficulty_classifier(df, labels)

    result = pd.DataFrame({"difficulty_score": scores, "maneuver_label": labels}, index=df.index)
    result.to_parquet(output_dir / "difficulty_scores.parquet")
    model.save_model(str(output_dir / "difficulty_model.lgb"))

    _plot_difficulty(df, scores, labels, model, output_dir)

    print(f"\nDifficulty score stats:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")

    top_hard = result.nlargest(50, "difficulty_score")
    top_easy = result.nsmallest(50, "difficulty_score")
    with open(output_dir / "top50_hardest.json", "w") as f:
        json.dump(top_hard.index.tolist(), f, indent=2)
    with open(output_dir / "top50_easiest.json", "w") as f:
        json.dump(top_easy.index.tolist(), f, indent=2)
    print(f"\nTop-50 hardest/easiest scene paths saved.")
    print(f"Inspect with: python -m scenario_generation.visualize <path>")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
