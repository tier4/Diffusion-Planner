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

"""Plot clustering + sampling quality metrics vs compression ratio.

For a given clustering result and sampling result, this script sweeps
compression ratios (by varying n_per_cluster from 1 to min_cluster_size)
and plots four metrics on a single figure:

  ANND      Average Nearest-Neighbour Distance from each full-dataset sample
            to the nearest representative.  Lower = better coverage.

  MaxND     Maximum Nearest-Neighbour Distance (worst-case coverage gap).
            Lower = better.

  Coverage  Fraction of full-dataset samples whose nearest representative
            is within distance ε.  Higher = better.

  Diversity Mean distance from each representative to its nearest other
            representative.  Higher = more spread-out representatives.

Distances are computed in Z-score-normalised, PCA-reduced feature space
(same preprocessing as pipeline.py).

ANND, MaxND, Diversity share the left y-axis; Coverage uses the right y-axis.
The actual sampled_json configuration is marked with a star on each curve.

Performance notes
-----------------
With millions of NPZ files the dominant cost is file I/O, not metric
computation.  Use --load_sample_size (default 50 000) to load only a
stratified random subset of files from the cluster JSON.  All metrics
are then estimated from this subset while the compression-ratio axis
still uses N_total (the true count from the JSON) as its denominator.
Set --load_sample_size 0 to load all files.

Usage:
    python plot_metrics.py \\
        --cluster_json /path/to/cluster_result.json \\
        --sampled_json /path/to/sampled.json \\
        [--load_sample_size 50000] \\
        [--pca_components 50] [--eps 0.5] \\
        [--n_sweep_points 30] [--seed 42] \\
        [--output metrics.png]

JSON formats:
    cluster_result.json : {"cluster_id0": ["a.npz", ...], ...}   (cluster.py output)
    sampled.json        : {"seed": 42, "files": ["a.npz", ...]}  (sampling.py output)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from pipeline import extract_features


# ── metric computation ────────────────────────────────────────────────────────


def compute_metrics(
    features_all: np.ndarray,
    features_reps: np.ndarray,
    eps: float,
) -> dict:
    """Return ANND, MaxND, Coverage, and Diversity for a representative set.

    Args:
        features_all:  (N, D) full-dataset features in PCA space.
        features_reps: (M, D) representative features in PCA space.
        eps:           coverage threshold distance.

    Returns:
        dict with keys "annd", "maxnd", "coverage", "diversity".
    """
    # Distance from every full-dataset sample to its nearest representative
    nn_dists = cdist(features_all, features_reps).min(axis=1)  # (N,)
    annd = float(nn_dists.mean())
    maxnd = float(nn_dists.max())
    coverage = float((nn_dists <= eps).mean())

    # Mean distance from each representative to its nearest other representative
    if len(features_reps) > 1:
        rep_dists = cdist(features_reps, features_reps)
        np.fill_diagonal(rep_dists, np.inf)
        diversity = float(rep_dists.min(axis=1).mean())
    else:
        diversity = 0.0

    return {"annd": annd, "maxnd": maxnd, "coverage": coverage, "diversity": diversity}


# ── feature preprocessing ─────────────────────────────────────────────────────


def load_and_preprocess(
    all_paths: list,
    pca_components: int,
) -> tuple[np.ndarray, list]:
    """Extract features, Z-score normalise, and apply PCA.

    Returns:
        features_pca: (N, n_components) array.
        valid_paths:  list of paths whose features were loaded successfully.
    """
    raw = []
    valid_paths = []
    for p in tqdm(all_paths, desc="Loading features", unit="file"):
        try:
            raw.append(extract_features(p))
            valid_paths.append(p)
        except Exception as e:
            tqdm.write(f"  [warn] skipping {p}: {e}")

    if not raw:
        raise RuntimeError("No valid NPZ files could be loaded.")

    features = np.array(raw)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    n_components = min(pca_components, features_norm.shape[0], features_norm.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    features_pca = pca.fit_transform(features_norm)
    explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: {features_norm.shape[1]}-dim → {n_components}-dim "
        f"({explained * 100:.1f}% variance explained)"
    )
    return features_pca, valid_paths


# ── compression-ratio sweep ───────────────────────────────────────────────────


def stratified_load_sample(
    clusters: dict,
    load_sample_size: int,
    seed: int,
) -> tuple[list[str], dict]:
    """Return a stratified random subsample of cluster files to load.

    Samples proportionally from each cluster so the distribution of cluster
    sizes is preserved.  At least one file is drawn from each cluster.

    This is the primary lever for reducing I/O cost: with millions of NPZ
    files the file-loading step dominates runtime.  Reducing the number of
    loaded files from N to S makes feature extraction O(S) instead of O(N).

    Args:
        clusters:         full cluster assignment dict (cluster_id → path list).
        load_sample_size: target number of files to load in total.
        seed:             random seed.

    Returns:
        load_paths:      flat list of file paths to load.
        loaded_clusters: same key structure as ``clusters`` but containing
                         only the sampled paths (used for rep selection in the
                         sweep).
    """
    N_total = sum(len(v) for v in clusters.values())
    if N_total <= load_sample_size:
        return [p for v in clusters.values() for p in v], dict(clusters)

    rng = np.random.default_rng(seed)
    load_paths: list[str] = []
    loaded_clusters: dict = {}
    for key, paths in clusters.items():
        n = max(1, round(load_sample_size * len(paths) / N_total))
        n = min(n, len(paths))
        chosen_idxs = rng.choice(len(paths), n, replace=False)
        chosen = [paths[i] for i in chosen_idxs]
        load_paths.extend(chosen)
        loaded_clusters[key] = chosen

    print(
        f"Load subsample: {len(load_paths):,} / {N_total:,} files "
        f"({len(load_paths) / N_total * 100:.1f}%, stratified by cluster)."
    )
    return load_paths, loaded_clusters


def subsample_for_eval(
    features_pca: np.ndarray,
    metric_sample_size: int,
    seed: int,
) -> np.ndarray:
    """Return a random subsample of features_pca for metric evaluation.

    Using the full dataset for every cdist call in the sweep is O(N × M × D)
    per step, which becomes prohibitive for large N.  Drawing a fixed subsample
    once and reusing it across all steps reduces cost to O(S × M × D) where
    S = metric_sample_size << N, while still giving a representative estimate
    of ANND, MaxND, and Coverage.

    Args:
        features_pca:       (N, D) full-dataset features.
        metric_sample_size: target subsample size; if N <= this, returns all rows.
        seed:               random seed for reproducible subsampling.

    Returns:
        (min(N, metric_sample_size), D) array.
    """
    N = features_pca.shape[0]
    if N <= metric_sample_size:
        return features_pca
    rng = np.random.default_rng(seed)
    idxs = rng.choice(N, metric_sample_size, replace=False)
    print(f"Subsampling {metric_sample_size} / {N} points for metric evaluation.")
    return features_pca[idxs]


def sweep_metrics(
    clusters: dict,
    features_pca: np.ndarray,
    features_eval: np.ndarray,
    path_to_idx: dict,
    eps: float,
    n_sweep_points: int,
    seed: int,
    N_total: int,
) -> list[dict]:
    """Sweep n_per_cluster from 1 to min_cluster_size and record metrics.

    At each step, the same number of samples is drawn uniformly at random
    from every cluster (balanced equal-size sampling, matching sampling.py).

    Args:
        clusters:       dict mapping cluster_id → list of NPZ paths (may be a
                        load-time subsample; see stratified_load_sample).
        features_pca:   (N_loaded, D) features for the loaded files, used to
                        look up representative vectors.
        features_eval:  (S, D) features for distance computation (may be a
                        further subsample; see subsample_for_eval).
        path_to_idx:    mapping from NPZ path to row index in features_pca.
        eps:            coverage threshold distance.
        n_sweep_points: maximum number of sweep steps.
        seed:           base random seed; each step uses seed + n_per_cluster.
        N_total:        true total file count from the cluster JSON, used as
                        the denominator of compression_ratio so that the
                        x-axis reflects the actual compression achieved.

    Returns:
        List of metric dicts, each containing "compression_ratio" and the
        four metric values, sorted by ascending compression_ratio.
    """
    cluster_lists = list(clusters.values())
    min_size = min(len(c) for c in cluster_lists)

    # Log-spaced sweep: dense at small n (where metrics change rapidly), sparse at large n.
    n_values = np.geomspace(1, min_size, num=min(n_sweep_points, min_size))
    n_values = sorted(set(int(round(v)) for v in n_values))
    if n_values[-1] != min_size:
        n_values.append(min_size)

    records = []
    for n_per_cluster in tqdm(n_values, desc="Sweeping compression ratios", unit="step"):
        rng = np.random.default_rng(seed + n_per_cluster)
        rep_idxs = []
        for paths in cluster_lists:
            chosen = rng.choice(len(paths), size=n_per_cluster, replace=False)
            for i in chosen:
                idx = path_to_idx.get(paths[i])
                if idx is not None:
                    rep_idxs.append(idx)

        if not rep_idxs:
            continue

        features_reps = features_pca[rep_idxs]
        metrics = compute_metrics(features_eval, features_reps, eps)
        # Compression ratio: n_per_cluster × n_clusters / N_total (true denominator)
        metrics["compression_ratio"] = (n_per_cluster * len(cluster_lists)) / N_total
        records.append(metrics)

    return sorted(records, key=lambda r: r["compression_ratio"])


# ── plotting ──────────────────────────────────────────────────────────────────


def plot(
    records: list[dict],
    actual: dict,
    eps: float,
    output: str | None,
) -> None:
    """Overlay all metrics on a single figure with twin y-axes.

    Left y-axis  : ANND, MaxND, Diversity  (distance in PCA feature space)
    Right y-axis : Coverage rate           (dimensionless, 0–1)

    The actual sampled_json configuration is marked with a star on each curve
    and as a vertical dashed line.
    """
    ratios = [r["compression_ratio"] for r in records]

    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()

    # ── distance metrics (left axis) ─────────────────────────────────────────
    dist_series = [
        ("annd",      "#1f77b4", "-",  "ANND ↓"),
        ("maxnd",     "#d62728", "-",  "MaxND ↓"),
        ("diversity", "#2ca02c", "--", "Diversity ↑"),
    ]
    for key, color, ls, label in dist_series:
        vals = [r[key] for r in records]
        ax_left.plot(ratios, vals, color=color, linestyle=ls, linewidth=2, label=label)
        ax_left.scatter(
            [actual["compression_ratio"]], [actual[key]],
            color=color, s=150, zorder=6, marker="*",
        )

    # ── coverage (right axis) ────────────────────────────────────────────────
    cov_vals = [r["coverage"] for r in records]
    ax_right.plot(
        ratios, cov_vals,
        color="#ff7f0e", linestyle="-.", linewidth=2,
        label=f"Coverage (ε={eps:.3f}) ↑",
    )
    ax_right.scatter(
        [actual["compression_ratio"]], [actual["coverage"]],
        color="#ff7f0e", s=150, zorder=6, marker="*",
    )
    ax_right.set_ylim(0.0, 1.05)
    ax_right.set_ylabel("Coverage rate", fontsize=12)

    # ── actual-configuration marker ──────────────────────────────────────────
    ax_left.axvline(
        actual["compression_ratio"],
        color="gray", linestyle=":", linewidth=1.5,
        label=f"actual  r={actual['compression_ratio']:.3f}",
    )

    # ── labels & legend ──────────────────────────────────────────────────────
    ax_left.set_xlabel("Compression ratio  M / N", fontsize=12)
    ax_left.set_ylabel("Distance (PCA feature space)", fontsize=12)
    ax_left.set_title(
        "Clustering + Sampling Quality Metrics vs Compression Ratio\n"
        f"(★ = actual sampled_json configuration,  ε = {eps:.3f})",
        fontsize=12,
    )
    ax_left.grid(True, alpha=0.3)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_l + lines_r, labels_l + labels_r,
        loc="upper left", fontsize=10, framealpha=0.9,
    )

    fig.tight_layout()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clustering+sampling quality metrics vs compression ratio"
    )
    parser.add_argument(
        "--cluster_json",
        required=True,
        help="Cluster assignment JSON produced by cluster.py",
    )
    parser.add_argument(
        "--sampled_json",
        required=True,
        help="Sampling result JSON produced by sampling.py",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components for feature reduction (default: 50)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help=(
            "Coverage threshold distance in PCA feature space. "
            "If omitted, auto-set to the 25th percentile of pairwise distances "
            "among a random sample of 500 data points."
        ),
    )
    parser.add_argument(
        "--n_sweep_points",
        type=int,
        default=30,
        help="Number of compression-ratio steps to evaluate (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument(
        "--load_sample_size",
        type=int,
        default=50000,
        help=(
            "Number of NPZ files to load, sampled proportionally from each cluster "
            "(stratified).  This is the primary lever for reducing I/O cost with "
            "large datasets: e.g. 50 000 instead of 5 000 000 files gives ~100× speedup. "
            "Set to 0 to load all files (default: 50000)."
        ),
    )
    parser.add_argument(
        "--metric_sample_size",
        type=int,
        default=5000,
        help=(
            "Number of loaded data points used for distance computation at each sweep "
            "step.  Reduces cost from O(N_loaded×M×D) to O(S×M×D) per step. "
            "Set to 0 to use all loaded data (default: 5000)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save figure to this path (PNG/PDF/SVG). Omit to display interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    with open(args.cluster_json, encoding="utf-8") as f:
        clusters: dict = json.load(f)
    with open(args.sampled_json, encoding="utf-8") as f:
        sampled_data: dict = json.load(f)
    sampled_files: set = set(sampled_data["files"])

    N_total = sum(len(v) for v in clusters.values())
    print(f"Total files : {N_total:,}")
    print(f"Clusters    : {len(clusters)}")
    print(f"Sampled     : {len(sampled_files):,}")

    # ── load subsample ────────────────────────────────────────────────────────
    load_sample_size = args.load_sample_size if args.load_sample_size > 0 else N_total
    load_paths, loaded_clusters = stratified_load_sample(clusters, load_sample_size, args.seed)

    features_pca, valid_paths = load_and_preprocess(load_paths, args.pca_components)
    path_to_idx = {p: i for i, p in enumerate(valid_paths)}
    N_loaded = len(valid_paths)

    # ── auto eps ─────────────────────────────────────────────────────────────
    eps = args.eps
    if eps is None:
        sample_size = min(500, N_loaded)
        rng = np.random.default_rng(0)
        sample_idxs = rng.choice(N_loaded, sample_size, replace=False)
        sample_feats = features_pca[sample_idxs]
        pw = cdist(sample_feats, sample_feats)
        np.fill_diagonal(pw, np.inf)
        eps = float(np.percentile(pw[np.isfinite(pw)], 25))
        print(f"Auto eps (25th pct of pairwise distances): {eps:.4f}")

    # ── subsample for metric evaluation ──────────────────────────────────────
    metric_sample_size = args.metric_sample_size if args.metric_sample_size > 0 else N_loaded
    features_eval = subsample_for_eval(features_pca, metric_sample_size, seed=args.seed)

    # ── sweep (uses loaded_clusters for rep selection; N_total for compression ratio) ──
    records = sweep_metrics(
        loaded_clusters, features_pca, features_eval, path_to_idx,
        eps, args.n_sweep_points, args.seed, N_total,
    )

    # ── actual configuration ──────────────────────────────────────────────────
    actual_idxs = [path_to_idx[p] for p in sampled_files if p in path_to_idx]
    n_actual_matched = len(actual_idxs)
    if n_actual_matched == 0:
        print("[warn] No sampled files matched loaded paths; actual marker will be skipped.")
    elif n_actual_matched < len(sampled_files):
        print(
            f"[info] {n_actual_matched:,} / {len(sampled_files):,} sampled files found in "
            f"loaded subset; actual metrics estimated from this subset."
        )
    features_actual = features_pca[actual_idxs] if actual_idxs else np.empty((0, features_pca.shape[1]))
    actual = compute_metrics(features_eval, features_actual, eps)
    # Use true count (not loaded subset) so the star lands at the real compression ratio
    actual["compression_ratio"] = len(sampled_files) / N_total

    print("\n--- Actual configuration ---")
    print(f"  Compression ratio : {actual['compression_ratio']:.4f}")
    print(f"  ANND              : {actual['annd']:.4f}")
    print(f"  MaxND             : {actual['maxnd']:.4f}")
    print(f"  Coverage          : {actual['coverage']:.4f}")
    print(f"  Diversity         : {actual['diversity']:.4f}")

    plot(records, actual, eps, args.output)


if __name__ == "__main__":
    main()
