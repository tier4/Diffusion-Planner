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

"""Unit and integration tests for diffusion_planner/sampling/cluster.py
and diffusion_planner/sampling/utils/elbow.py.

Covers:
- load_npz_paths: valid JSON, missing file
- extract_features: output shape, values, missing key
- compute_wcss: list length, monotonically non-increasing inertia
- find_elbow: short list edge cases, synthetic WCSS with clear elbow
- elbow_kmeans: label shape/range, optimal_k bounds, k_max clamping
- main (integration): end-to-end run with synthetic NPZ files

Usage:
    python tests/test_cluster.py          # standalone
    pytest tests/test_cluster.py -v       # with pytest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

SAMPLING_DIR = Path(__file__).resolve().parent.parent / "sampling"
sys.path.insert(0, str(SAMPLING_DIR))

from cluster import load_npz_paths, main
from utils.elbow import compute_wcss, elbow_kmeans, find_elbow
from utils.pipeline import (
    ClusteringStrategy,
    ElbowKMeansStrategy,
    cluster_trajectories,
    extract_features,
)

# ─────────────────────────────── helpers ────────────────────────────────────


def _make_npz(path: str, ego_future: np.ndarray | None = None) -> None:
    """Write a minimal NPZ file with ego_agent_future."""
    if ego_future is None:
        ego_future = np.random.randn(80, 3).astype(np.float32)
    np.savez(path, ego_agent_future=ego_future)


def _synthetic_clusters(n_per_cluster: int = 10, n_clusters: int = 3, seed: int = 42) -> np.ndarray:
    """Return a 2-D feature array with clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])[:n_clusters]
    parts = [rng.normal(c, 0.3, (n_per_cluster, 2)) for c in centers]
    return np.vstack(parts)


# ─────────────────────────── load_npz_paths ─────────────────────────────────


def test_load_npz_paths_valid():
    with tempfile.TemporaryDirectory() as tmp:
        paths = ["/a/b.npz", "/c/d.npz"]
        json_path = Path(tmp) / "list.json"
        json_path.write_text(json.dumps(paths))
        result = load_npz_paths(str(json_path))
    assert result == paths, f"Expected {paths}, got {result}"
    print("  [PASS] load_npz_paths valid JSON")


def test_load_npz_paths_missing_file():
    try:
        load_npz_paths("/nonexistent/path.json")
        assert False, "Should have raised FileNotFoundError"
    except (FileNotFoundError, OSError):
        pass
    print("  [PASS] load_npz_paths missing file raises")


# ─────────────────────────── extract_features ───────────────────────────────


def test_extract_features_shape():
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = str(Path(tmp) / "sample.npz")
        _make_npz(npz_path)
        feat = extract_features(npz_path)
    assert feat.shape == (240,), f"Expected (240,), got {feat.shape}"
    print("  [PASS] extract_features shape (240,)")


def test_extract_features_values():
    """Returned vector equals ego_agent_future.flatten()."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = str(Path(tmp) / "sample.npz")
        ego_future = np.arange(240, dtype=np.float32).reshape(80, 3)
        _make_npz(npz_path, ego_future)
        feat = extract_features(npz_path)
    np.testing.assert_array_almost_equal(feat, ego_future.flatten())
    print("  [PASS] extract_features values match flatten")


def test_extract_features_missing_key():
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = str(Path(tmp) / "sample.npz")
        np.savez(npz_path, wrong_key=np.zeros((80, 3)))
        try:
            extract_features(npz_path)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
    print("  [PASS] extract_features missing key raises KeyError")


def test_extract_features_dtype_is_float():
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = str(Path(tmp) / "sample.npz")
        _make_npz(npz_path, np.ones((80, 3), dtype=np.int16))
        feat = extract_features(npz_path)
    assert np.issubdtype(feat.dtype, np.floating), f"Expected float dtype, got {feat.dtype}"
    print("  [PASS] extract_features dtype is float")


# ──────────────────────────── compute_wcss ──────────────────────────────────


def test_compute_wcss_length():
    X = _synthetic_clusters()
    k_max = 5
    wcss = compute_wcss(X, k_max=k_max)
    assert len(wcss) == k_max, f"Expected {k_max} values, got {len(wcss)}"
    print("  [PASS] compute_wcss length equals k_max")


def test_compute_wcss_monotone():
    """Inertia must be non-increasing as k grows."""
    X = _synthetic_clusters()
    wcss = compute_wcss(X, k_max=6)
    for i in range(len(wcss) - 1):
        assert wcss[i] >= wcss[i + 1] - 1e-6, (
            f"WCSS not non-increasing at k={i + 1}: {wcss[i]:.4f} > {wcss[i + 1]:.4f}"
        )
    print("  [PASS] compute_wcss monotonically non-increasing")


def test_compute_wcss_k1_max_inertia():
    """k=1 must have the largest inertia (whole dataset in one cluster)."""
    X = _synthetic_clusters()
    wcss = compute_wcss(X, k_max=4)
    assert wcss[0] == max(wcss), "k=1 should have the highest inertia"
    print("  [PASS] compute_wcss k=1 has maximum inertia")


# ──────────────────────────── find_elbow ────────────────────────────────────


def test_find_elbow_single_value():
    assert find_elbow([42.0]) == 1
    print("  [PASS] find_elbow single value returns 1")


def test_find_elbow_two_values():
    assert find_elbow([10.0, 5.0]) == 2
    print("  [PASS] find_elbow two values returns 2")


def test_find_elbow_clear_elbow():
    """WCSS drops sharply before k=3 then flattens — elbow expected at k=3."""
    # second_diff = [10, 38, 1, 0] → argmax index 1 → k = 1+2 = 3
    wcss = [100.0, 50.0, 10.0, 8.0, 7.0, 6.0]
    k = find_elbow(wcss)
    assert k == 3, f"Expected elbow at k=3, got k={k}"
    print("  [PASS] find_elbow clear elbow at k=3")


def test_find_elbow_returns_int():
    wcss = [100.0, 50.0, 10.0, 8.0]
    k = find_elbow(wcss)
    assert isinstance(k, int), f"Expected int, got {type(k)}"
    print("  [PASS] find_elbow return type is int")


# ──────────────────────────── elbow_kmeans ──────────────────────────────────


def test_elbow_kmeans_labels_shape():
    X = _synthetic_clusters(n_per_cluster=10, n_clusters=3)
    labels, _, _ = elbow_kmeans(X, k_max=6)
    assert labels.shape == (X.shape[0],), f"Expected ({X.shape[0]},), got {labels.shape}"
    print("  [PASS] elbow_kmeans labels shape")


def test_elbow_kmeans_labels_range():
    X = _synthetic_clusters(n_per_cluster=10, n_clusters=3)
    labels, optimal_k, _ = elbow_kmeans(X, k_max=6)
    assert labels.min() >= 0, f"Negative label: {labels.min()}"
    assert labels.max() < optimal_k, f"Label {labels.max()} >= optimal_k {optimal_k}"
    print("  [PASS] elbow_kmeans label values in [0, optimal_k)")


def test_elbow_kmeans_optimal_k_bounds():
    X = _synthetic_clusters(n_per_cluster=10, n_clusters=3)
    k_max = 8
    _, optimal_k, _ = elbow_kmeans(X, k_max=k_max)
    assert 1 <= optimal_k <= k_max, f"optimal_k={optimal_k} outside [1, {k_max}]"
    print("  [PASS] elbow_kmeans optimal_k within [1, k_max]")


def test_elbow_kmeans_wcss_length():
    X = _synthetic_clusters(n_per_cluster=10, n_clusters=3)
    k_max = 6
    _, _, wcss = elbow_kmeans(X, k_max=k_max)
    assert len(wcss) == k_max, f"Expected wcss length {k_max}, got {len(wcss)}"
    print("  [PASS] elbow_kmeans wcss length equals k_max")


def test_elbow_kmeans_k_max_clamped_to_n_samples():
    """When k_max > n_samples, k_max is clamped to n_samples."""
    n = 5
    X = np.random.default_rng(0).random((n, 4))
    labels, optimal_k, wcss = elbow_kmeans(X, k_max=100)
    assert len(wcss) == n, f"Expected wcss length {n}, got {len(wcss)}"
    assert 1 <= optimal_k <= n, f"optimal_k={optimal_k} outside [1, {n}]"
    assert labels.shape == (n,)
    print("  [PASS] elbow_kmeans k_max clamped to n_samples")


def test_elbow_kmeans_reproducible():
    X = _synthetic_clusters()
    labels1, k1, _ = elbow_kmeans(X, k_max=6, random_state=0)
    labels2, k2, _ = elbow_kmeans(X, k_max=6, random_state=0)
    assert k1 == k2
    np.testing.assert_array_equal(labels1, labels2)
    print("  [PASS] elbow_kmeans same seed gives same result")


# ──────────────────────────── ElbowKMeansStrategy ───────────────────────────


def test_elbow_kmeans_strategy_labels_shape():
    X = _synthetic_clusters(n_per_cluster=10, n_clusters=3)
    strategy = ElbowKMeansStrategy(k_max=6, random_state=42)
    labels = strategy.fit_predict(X)
    assert labels.shape == (X.shape[0],), f"Expected ({X.shape[0]},), got {labels.shape}"
    print("  [PASS] ElbowKMeansStrategy labels shape")


def test_elbow_kmeans_strategy_n_clusters_set():
    X = _synthetic_clusters(n_per_cluster=10, n_clusters=3)
    strategy = ElbowKMeansStrategy(k_max=6, random_state=42)
    strategy.fit_predict(X)
    assert isinstance(strategy.n_clusters_, int), (
        f"n_clusters_ should be int, got {type(strategy.n_clusters_)}"
    )
    assert 1 <= strategy.n_clusters_ <= 6
    print("  [PASS] ElbowKMeansStrategy n_clusters_ set after fit_predict")


def test_elbow_kmeans_strategy_is_subclass():
    assert issubclass(ElbowKMeansStrategy, ClusteringStrategy)
    print("  [PASS] ElbowKMeansStrategy is subclass of ClusteringStrategy")


def test_elbow_kmeans_strategy_reproducible():
    X = _synthetic_clusters()
    s1 = ElbowKMeansStrategy(k_max=6, random_state=0)
    s2 = ElbowKMeansStrategy(k_max=6, random_state=0)
    np.testing.assert_array_equal(s1.fit_predict(X), s2.fit_predict(X))
    assert s1.n_clusters_ == s2.n_clusters_
    print("  [PASS] ElbowKMeansStrategy same seed gives same result")


# ─────────────────────────── cluster_trajectories ───────────────────────────


def test_cluster_trajectories_returns_dict():
    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=15)
        strategy = ElbowKMeansStrategy(k_max=5, random_state=42)
        result = cluster_trajectories(npz_paths, strategy, pca_components=10)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert isinstance(strategy.n_clusters_, int), (
        f"strategy.n_clusters_ should be int after call, got {type(strategy.n_clusters_)}"
    )
    print("  [PASS] cluster_trajectories returns dict, strategy.n_clusters_ is int")


def test_cluster_trajectories_all_paths_present():
    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=20)
        result = cluster_trajectories(
            npz_paths, ElbowKMeansStrategy(k_max=5, random_state=42), pca_components=10
        )
    all_out = [p for paths in result.values() for p in paths]
    assert sorted(all_out) == sorted(npz_paths), "Output paths do not match input paths"
    print("  [PASS] cluster_trajectories all paths present")


def test_cluster_trajectories_key_format():
    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=15)
        strategy = ElbowKMeansStrategy(k_max=5, random_state=42)
        result = cluster_trajectories(npz_paths, strategy, pca_components=10)
    for key in result:
        assert key.startswith("cluster_id"), f"Unexpected key format: {key}"
    assert len(result) == strategy.n_clusters_, (
        f"Number of keys ({len(result)}) does not match strategy.n_clusters_ ({strategy.n_clusters_})"
    )
    print("  [PASS] cluster_trajectories key format and count")


def test_cluster_trajectories_no_valid_files_raises():
    strategy = ElbowKMeansStrategy(k_max=3)
    try:
        cluster_trajectories(["/nonexistent/a.npz", "/nonexistent/b.npz"], strategy)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("  [PASS] cluster_trajectories raises on no valid files")


def test_cluster_trajectories_uses_strategy():
    """cluster_trajectories delegates clustering to the injected strategy."""

    class _FixedKStrategy(ClusteringStrategy):
        def __init__(self, k: int) -> None:
            self._k = k

        def fit_predict(self, features: np.ndarray) -> np.ndarray:
            self.n_clusters_ = self._k
            return np.arange(len(features)) % self._k

    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=9)
        strategy = _FixedKStrategy(k=3)
        result = cluster_trajectories(npz_paths, strategy, pca_components=5)

    assert strategy.n_clusters_ == 3, "Strategy's fit_predict was not called"
    assert len(result) == 3, f"Expected 3 clusters, got {len(result)}"
    all_out = [p for paths in result.values() for p in paths]
    assert sorted(all_out) == sorted(npz_paths)
    print("  [PASS] cluster_trajectories delegates to injected strategy")


# ──────────────────────────── integration: main ─────────────────────────────


def _make_synthetic_dataset(tmp_dir: str, n: int = 30, seed: int = 0) -> list[str]:
    """Create n NPZ files with three distinct trajectory patterns."""
    rng = np.random.default_rng(seed)
    paths = []
    patterns = [
        np.column_stack([np.linspace(0, 20, 80), np.zeros(80), np.zeros(80)]),  # straight
        np.column_stack(
            [np.linspace(0, 10, 80), np.linspace(0, 10, 80), np.linspace(0, np.pi / 2, 80)]
        ),  # left turn
        np.column_stack(
            [np.linspace(0, 10, 80), np.linspace(0, -10, 80), np.linspace(0, -np.pi / 2, 80)]
        ),  # right turn
    ]
    for i in range(n):
        pattern = patterns[i % len(patterns)].copy().astype(np.float32)
        pattern += rng.normal(0, 0.05, pattern.shape).astype(np.float32)
        npz_path = str(Path(tmp_dir) / f"sample_{i:04d}.npz")
        np.savez(npz_path, ego_agent_future=pattern)
        paths.append(npz_path)
    return paths


def test_main_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=30)

        data_list_path = str(Path(tmp) / "data_list.json")
        with open(data_list_path, "w") as f:
            json.dump(npz_paths, f)

        output_path = str(Path(tmp) / "result.json")

        argv = [
            "cluster.py",
            "--data_list",
            data_list_path,
            "--output",
            output_path,
            "--k_max",
            "6",
            "--pca_components",
            "10",
            "--seed",
            "42",
        ]
        with patch("sys.argv", argv):
            main()

        assert Path(output_path).exists(), "Output JSON was not created"

        with open(output_path) as f:
            result = json.load(f)

        # Keys must follow "cluster_idN" pattern
        for key in result:
            assert key.startswith("cluster_id"), f"Unexpected key: {key}"

        # All input paths must appear exactly once in the output
        all_output_paths = [p for paths in result.values() for p in paths]
        assert sorted(all_output_paths) == sorted(npz_paths), (
            "Output paths do not match input paths"
        )

    print("  [PASS] main end-to-end: output JSON has correct structure")


def test_main_output_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=15)

        data_list_path = str(Path(tmp) / "data_list.json")
        with open(data_list_path, "w") as f:
            json.dump(npz_paths, f)

        output_path = str(Path(tmp) / "result.json")

        argv = [
            "cluster.py",
            "--data_list",
            data_list_path,
            "--output",
            output_path,
            "--k_max",
            "5",
            "--pca_components",
            "10",
            "--seed",
            "0",
        ]
        with patch("sys.argv", argv):
            main()

        with open(output_path) as f:
            result = json.load(f)

        all_paths = [p for paths in result.values() for p in paths]
        assert len(all_paths) == len(set(all_paths)), "Duplicate paths in output"

    print("  [PASS] main end-to-end: no duplicate paths in output")


def test_main_output_keys_sorted():
    """Cluster keys in output JSON are sorted by numeric id."""
    with tempfile.TemporaryDirectory() as tmp:
        npz_paths = _make_synthetic_dataset(tmp, n=20)

        data_list_path = str(Path(tmp) / "data_list.json")
        with open(data_list_path, "w") as f:
            json.dump(npz_paths, f)

        output_path = str(Path(tmp) / "result.json")

        argv = [
            "cluster.py",
            "--data_list",
            data_list_path,
            "--output",
            output_path,
            "--k_max",
            "5",
            "--pca_components",
            "10",
            "--seed",
            "7",
        ]
        with patch("sys.argv", argv):
            main()

        with open(output_path) as f:
            result = json.load(f)

        keys = list(result.keys())
        ids = [int(k.replace("cluster_id", "")) for k in keys]
        assert ids == sorted(ids), f"Keys are not sorted: {keys}"

    print("  [PASS] main end-to-end: output keys are sorted")


# ──────────────────────────────── runner ────────────────────────────────────


ALL_TESTS = [
    test_load_npz_paths_valid,
    test_load_npz_paths_missing_file,
    test_extract_features_shape,
    test_extract_features_values,
    test_extract_features_missing_key,
    test_extract_features_dtype_is_float,
    test_compute_wcss_length,
    test_compute_wcss_monotone,
    test_compute_wcss_k1_max_inertia,
    test_find_elbow_single_value,
    test_find_elbow_two_values,
    test_find_elbow_clear_elbow,
    test_find_elbow_returns_int,
    test_elbow_kmeans_labels_shape,
    test_elbow_kmeans_labels_range,
    test_elbow_kmeans_optimal_k_bounds,
    test_elbow_kmeans_wcss_length,
    test_elbow_kmeans_k_max_clamped_to_n_samples,
    test_elbow_kmeans_reproducible,
    test_elbow_kmeans_strategy_labels_shape,
    test_elbow_kmeans_strategy_n_clusters_set,
    test_elbow_kmeans_strategy_is_subclass,
    test_elbow_kmeans_strategy_reproducible,
    test_cluster_trajectories_returns_dict,
    test_cluster_trajectories_all_paths_present,
    test_cluster_trajectories_key_format,
    test_cluster_trajectories_no_valid_files_raises,
    test_cluster_trajectories_uses_strategy,
    test_main_end_to_end,
    test_main_output_no_duplicates,
    test_main_output_keys_sorted,
]


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} tests for cluster.py / elbow.py\n")
    passed, failed, errors = 0, 0, []

    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(ALL_TESTS)} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed!")
