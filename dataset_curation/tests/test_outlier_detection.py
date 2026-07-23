from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_feature_df(n: int = 100, n_outliers: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n - n_outliers, 10))
    outliers = rng.normal(0, 1, (n_outliers, 10)) + 10.0  # far from cluster
    data = np.vstack([normal, outliers])
    paths = [f"/fake/scene_{i:04d}.npz" for i in range(n)]
    cols = [f"feat_{i}" for i in range(10)]
    return pd.DataFrame(data, index=paths, columns=cols)


def test_isolation_forest_returns_series():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df()
    result = detect_outliers(df, method="isolation_forest", contamination=0.05)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert len(result) == len(df)


def test_isolation_forest_detects_outliers():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df(n=200, n_outliers=10)
    result = detect_outliers(df, method="isolation_forest", contamination=0.1)
    n_flagged = result.sum()
    assert n_flagged > 0, "Should flag at least some outliers"
    last_10_flagged = result.iloc[-10:].sum()
    assert last_10_flagged > 5, (
        f"Most of the injected outliers should be flagged, got {last_10_flagged}"
    )


def test_lof_returns_series():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df()
    result = detect_outliers(df, method="lof", contamination=0.05)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool


def test_lof_detects_outliers():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df(n=200, n_outliers=10)
    result = detect_outliers(df, method="lof", contamination=0.1)
    last_10_flagged = result.iloc[-10:].sum()
    assert last_10_flagged > 5


def test_invalid_method_raises():
    from dataset_curation.outlier_detection import detect_outliers

    df = _make_feature_df(n=20)
    with pytest.raises(ValueError, match="Unknown method"):
        detect_outliers(df, method="nonexistent")
