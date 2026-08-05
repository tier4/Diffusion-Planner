#!/usr/bin/env python3
"""Analyze latent OOD scores: distribution plots and top-OOD review list.

Usage:
    python scripts/analyze_latent_ood.py \
        --scores /path/to/latent_ood_scores.jsonl \
        --output_dir /path/to/analysis_output \
        [--top_n 200] \
        [--compare /path/to/other_scores.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores", type=Path, required=True, help="JSONL from score_latent_ood.py")
    p.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory for plots and CSVs"
    )
    p.add_argument(
        "--top_n", type=int, default=200, help="Number of top-OOD frames for review list"
    )
    p.add_argument("--compare", type=Path, default=None, help="Second JSONL for overlay comparison")
    p.add_argument("--label", type=str, default="eval", help="Label for the primary score set")
    p.add_argument(
        "--compare_label", type=str, default="compare", help="Label for the comparison set"
    )
    return p.parse_args()


def load_scores(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot_score_distribution(
    scores: np.ndarray,
    label: str,
    compare: np.ndarray | None,
    compare_label: str,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(scores, bins=100, alpha=0.7, label=label, density=True)
    if compare is not None:
        ax.hist(compare, bins=100, alpha=0.5, label=compare_label, density=True)
        ax.legend()
    ax.set_xlabel("kNN mean distance")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")

    ax = axes[1]
    sorted_scores = np.sort(scores)
    percentiles = np.arange(len(sorted_scores)) / len(sorted_scores) * 100
    ax.plot(percentiles, sorted_scores, label=label)
    if compare is not None:
        sorted_cmp = np.sort(compare)
        pct_cmp = np.arange(len(sorted_cmp)) / len(sorted_cmp) * 100
        ax.plot(pct_cmp, sorted_cmp, label=compare_label)
        ax.legend()
    ax.set_xlabel("Percentile")
    ax.set_ylabel("kNN mean distance")
    ax.set_title("Score CDF")
    ax.axvline(95, color="orange", linestyle="--", alpha=0.5, label="p95")
    ax.axvline(99, color="red", linestyle="--", alpha=0.5, label="p99")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_level_breakdown(entries: list[dict], output_path: Path):
    levels = [e.get("level", "unknown") for e in entries]
    counts = {}
    for lv in levels:
        counts[lv] = counts.get(lv, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"normal": "#4CAF50", "warning": "#FF9800", "high": "#F44336", "unknown": "#9E9E9E"}
    labels = list(counts.keys())
    vals = [counts[l] for l in labels]
    bars = ax.bar(labels, vals, color=[colors.get(l, "#9E9E9E") for l in labels])
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title("OOD Level Breakdown")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def write_top_ood_csv(entries: list[dict], top_n: int, output_path: Path):
    sorted_entries = sorted(entries, key=lambda e: e.get("knn_mean", 0), reverse=True)
    top = sorted_entries[:top_n]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "npz_path",
                "knn_mean",
                "knn_min",
                "percentile",
                "level",
                "nearest_1_path",
                "nearest_1_dist",
            ]
        )
        for i, e in enumerate(top, 1):
            nearest = e.get("nearest", [{}])
            writer.writerow(
                [
                    i,
                    e.get("npz_path", ""),
                    f"{e.get('knn_mean', 0):.6f}",
                    f"{e.get('knn_min', 0):.6f}",
                    f"{e.get('percentile', -1):.1f}",
                    e.get("level", "unknown"),
                    nearest[0].get("npz_path", "") if nearest else "",
                    f"{nearest[0].get('distance', 0):.6f}" if nearest else "",
                ]
            )
    print(f"  Saved: {output_path} ({len(top)} entries)")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {args.scores}")
    entries = load_scores(args.scores)
    scores = np.array([e["knn_mean"] for e in entries], dtype=np.float32)
    print(f"  {len(entries)} frames")
    print(
        f"  knn_mean: min={scores.min():.4f} median={np.median(scores):.4f} "
        f"p95={np.percentile(scores, 95):.4f} p99={np.percentile(scores, 99):.4f} max={scores.max():.4f}"
    )

    compare_scores = None
    if args.compare:
        print(f"Loading comparison scores from {args.compare}")
        cmp_entries = load_scores(args.compare)
        compare_scores = np.array([e["knn_mean"] for e in cmp_entries], dtype=np.float32)
        print(f"  {len(cmp_entries)} frames")

    print("Generating plots...")
    plot_score_distribution(
        scores,
        args.label,
        compare_scores,
        args.compare_label,
        args.output_dir / "ood_score_distribution.png",
    )

    if any("level" in e for e in entries):
        plot_level_breakdown(entries, args.output_dir / "ood_level_breakdown.png")

    print("Writing top-OOD review list...")
    write_top_ood_csv(entries, args.top_n, args.output_dir / "top_ood_review.csv")

    print("Done.")


if __name__ == "__main__":
    main()
