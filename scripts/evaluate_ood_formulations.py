#!/usr/bin/env python3
"""Evaluate OOD feature formulations with LightGBM classifier.

Compares 4 feature sets:
  - Baseline: EPDMS subscores only
  - H-A: EPDMS + raw OOD (knn_mean)
  - H-B: EPDMS + residual OOD (ood_residual)
  - H-AB: EPDMS + knn_mean + ood_residual

5-fold CV stratified by bag. Reports AUPRC, F1, recall@precision.

Usage:
    uv run python scripts/evaluate_ood_formulations.py \
      --input data/feature_matrix.csv \
      --output_dir data/evaluation_results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold

EPDMS_FEATURES = ["nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "ec", "ep"]

FEATURE_SETS = {
    "EPDMS-only": EPDMS_FEATURES,
    "H-A (EPDMS+rawOOD)": EPDMS_FEATURES + ["knn_mean"],
    "H-B (EPDMS+residualOOD)": EPDMS_FEATURES + ["ood_residual"],
    "H-AB (EPDMS+both)": EPDMS_FEATURES + ["knn_mean", "ood_residual"],
}


def load_data(csv_path: Path):
    """Load feature matrix, return X dict, y, groups, categories."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["nc"]:
                continue
            rows.append(row)

    all_features = EPDMS_FEATURES + ["knn_mean", "ood_residual"]
    X = np.full((len(rows), len(all_features)), np.nan, dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    groups = []
    categories = []

    for i, row in enumerate(rows):
        for j, feat in enumerate(all_features):
            val = row.get(feat, "")
            if val:
                X[i, j] = float(val)
        y[i] = int(row["label"])
        groups.append(row["bag"])
        categories.append(row["category"])

    feature_idx = {f: i for i, f in enumerate(all_features)}
    return X, y, groups, categories, feature_idx


def evaluate_fold(X_train, y_train, X_test, y_test, feature_cols):
    """Train LightGBM and return predictions."""
    dtrain = lgb.Dataset(X_train[:, feature_cols], label=y_train)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "is_unbalance": True,
        "seed": 42,
    }

    model = lgb.train(params, dtrain, num_boost_round=200)
    y_pred = model.predict(X_test[:, feature_cols])
    importance = model.feature_importance(importance_type="gain")
    return y_pred, importance


def compute_metrics(y_true, y_pred):
    """Compute AUPRC, best F1, recall@precision thresholds."""
    auprc = average_precision_score(y_true, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    recall_at_p50 = 0.0
    recall_at_p70 = 0.0
    for p, r in zip(precision, recall):
        if p >= 0.50:
            recall_at_p50 = max(recall_at_p50, r)
        if p >= 0.70:
            recall_at_p70 = max(recall_at_p70, r)

    return {
        "auprc": auprc,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "recall_at_p50": recall_at_p50,
        "recall_at_p70": recall_at_p70,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, groups, categories, feature_idx = load_data(args.input)
    print(f"  {X.shape[0]} rows, {X.shape[1]} features")
    prevalence = y.mean()
    trivial_f1 = 2 * prevalence / (1 + prevalence)
    print(f"  Positive: {y.sum()}, Negative: {(1 - y).sum()}")
    print(f"  Prevalence: {prevalence:.4f}")
    print(f"  Chance baselines: AUPRC={prevalence:.4f}, trivial F1={trivial_f1:.4f}")
    print(f"  Unique bags: {len(set(groups))}")
    nan_counts = np.isnan(X).sum(axis=0)
    for j, feat in enumerate(EPDMS_FEATURES + ["knn_mean", "ood_residual"]):
        if nan_counts[j] > 0:
            print(f"  NaN in {feat}: {nan_counts[j]} ({nan_counts[j] / X.shape[0] * 100:.1f}%)")

    # 5-fold CV stratified by bag
    unique_bags = sorted(set(groups))
    bag_to_int = {b: i for i, b in enumerate(unique_bags)}
    group_ids = np.array([bag_to_int[g] for g in groups])

    gkf = GroupKFold(n_splits=args.n_folds)

    results = {}
    all_importances = defaultdict(list)

    for fs_name, fs_features in FEATURE_SETS.items():
        feature_cols = [feature_idx[f] for f in fs_features]
        fold_metrics = []
        fold_predictions = []

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {fs_name}")
        print(f"  Features: {fs_features}")

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_ids)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred, importance = evaluate_fold(X_train, y_train, X_test, y_test, feature_cols)
            metrics = compute_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

            for fi, feat in enumerate(fs_features):
                all_importances[(fs_name, feat)].append(importance[fi])

            fold_predictions.append((test_idx, y_test, y_pred, [categories[i] for i in test_idx]))

            print(f"  Fold {fold}: AUPRC={metrics['auprc']:.4f}, F1={metrics['best_f1']:.4f}")

        # Aggregate
        avg_metrics = {}
        for key in fold_metrics[0]:
            vals = [m[key] for m in fold_metrics]
            avg_metrics[key] = float(np.mean(vals))
            avg_metrics[f"{key}_std"] = float(np.std(vals))

        results[fs_name] = avg_metrics
        print(f"  Mean AUPRC={avg_metrics['auprc']:.4f} +/- {avg_metrics['auprc_std']:.4f}")
        print(f"  Mean F1={avg_metrics['best_f1']:.4f} +/- {avg_metrics['best_f1_std']:.4f}")

        # Per-category breakdown
        all_test_idx = np.concatenate([fp[0] for fp in fold_predictions])
        all_y_test = np.concatenate([fp[1] for fp in fold_predictions])
        all_y_pred = np.concatenate([fp[2] for fp in fold_predictions])
        all_cats = []
        for fp in fold_predictions:
            all_cats.extend(fp[3])

        print(f"\n  Per-category:")
        cat_results = {}
        for cat in sorted(set(all_cats)):
            mask = np.array([c == cat for c in all_cats])
            if mask.sum() < 10 or all_y_test[mask].sum() == 0:
                continue
            cat_metrics = compute_metrics(all_y_test[mask], all_y_pred[mask])
            cat_results[cat] = cat_metrics
            print(
                f"    {cat:<25} AUPRC={cat_metrics['auprc']:.4f}, F1={cat_metrics['best_f1']:.4f}, n={mask.sum()}"
            )

        results[fs_name]["per_category"] = cat_results

    # Save results
    with open(args.output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_dir / 'evaluation_results.json'}")

    # Comparison table
    print(f"\n{'=' * 80}")
    print(f"{'Feature Set':<30} {'AUPRC':>10} {'F1':>10} {'R@P50':>10} {'R@P70':>10}")
    print(f"{'-' * 80}")
    for fs_name in FEATURE_SETS:
        r = results[fs_name]
        print(
            f"{fs_name:<30} {r['auprc']:>10.4f} {r['best_f1']:>10.4f} "
            f"{r['recall_at_p50']:>10.4f} {r['recall_at_p70']:>10.4f}"
        )

    # Feature importance plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (fs_name, fs_features) in zip(axes.flat, FEATURE_SETS.items()):
        importances = [np.mean(all_importances[(fs_name, f)]) for f in fs_features]
        sorted_idx = np.argsort(importances)
        ax.barh([fs_features[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
        ax.set_title(fs_name)
        ax.set_xlabel("Mean Gain")
    plt.suptitle("Feature Importance by Configuration", fontsize=14)
    plt.tight_layout()
    fig.savefig(args.output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir / 'feature_importance.png'}")

    # AUPRC comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(FEATURE_SETS.keys())
    auprcs = [results[n]["auprc"] for n in names]
    stds = [results[n]["auprc_std"] for n in names]
    bars = ax.bar(names, auprcs, yerr=stds, capsize=5)
    ax.set_ylabel("AUPRC")
    ax.set_title("Override Detection: AUPRC by Feature Set (5-fold CV)")
    for bar, val in zip(bars, auprcs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(args.output_dir / "auprc_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir / 'auprc_comparison.png'}")


if __name__ == "__main__":
    main()
