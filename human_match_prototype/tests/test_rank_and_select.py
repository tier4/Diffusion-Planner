import numpy as np
import pandas as pd

from human_match_prototype.rank_and_select import compute_ranks, select_review_set


def _fake_scores(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a plausible scores DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"npz_path": f"/path/frame_{i:05d}.npz"}
        for h in ["2s", "4s", "8s"]:
            row[f"es_obs_{h}"] = rng.exponential(5.0)
            row[f"es_div_{h}"] = rng.exponential(2.0)
            row[f"es_{h}"] = row[f"es_obs_{h}"] - 0.5 * row[f"es_div_{h}"]
        has_frenet = rng.random() > 0.1  # 90% have valid route
        for h in ["2s", "4s", "8s"]:
            row[f"es_lon_{h}"] = rng.exponential(3.0) if has_frenet else float("nan")
            row[f"es_lat_{h}"] = rng.exponential(2.0) if has_frenet else float("nan")
        row["route_valid"] = int(has_frenet)
        rows.append(row)
    return pd.DataFrame(rows)


class TestComputeRanks:
    def test_percentile_range(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        assert "pct_es_2s" in ranked.columns
        assert ranked["pct_es_2s"].min() >= 0
        assert ranked["pct_es_2s"].max() <= 100

    def test_combined_ranks_present(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        assert "R_overall" in ranked.columns
        assert "R_lateral" in ranked.columns
        assert "R_longitudinal" in ranked.columns

    def test_nan_frenet_handled(self):
        """Scenes with NaN Frenet should have NaN R_lateral."""
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        nan_route = ranked[ranked["route_valid"] == 0]
        if len(nan_route) > 0:
            assert nan_route["R_lateral"].isna().all()


class TestSelectReviewSet:
    def test_returns_5_to_10(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        review = select_review_set(ranked, top_k=5)
        assert 5 <= len(review) <= 10

    def test_includes_top_overall(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        review = select_review_set(ranked, top_k=5)
        top5_overall = ranked.nlargest(5, "R_overall")["npz_path"].tolist()
        for p in top5_overall:
            assert p in review["npz_path"].values

    def test_selection_reason_populated(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        review = select_review_set(ranked, top_k=5)
        assert "selection_reason" in review.columns
        assert review["selection_reason"].notna().all()
