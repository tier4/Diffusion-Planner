"""Stage 2: Compute percentile ranks and select review candidates."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from human_match_prototype.energy_score import HORIZONS

ES_SCORE_COLS = [f"es_{h}" for h in HORIZONS]
LAT_SCORE_COLS = [f"es_lat_{h}" for h in HORIZONS]
LON_SCORE_COLS = [f"es_lon_{h}" for h in HORIZONS]


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile rank columns and combined rankings."""
    ranked = df.copy()

    for col in ES_SCORE_COLS + LAT_SCORE_COLS + LON_SCORE_COLS:
        pct_col = f"pct_{col}"
        ranked[pct_col] = ranked[col].rank(pct=True, na_option="keep") * 100

    ranked["R_overall"] = ranked[[f"pct_es_{h}" for h in HORIZONS]].mean(axis=1)

    lat_pcts = ranked[[f"pct_es_lat_{h}" for h in HORIZONS]]
    ranked["R_lateral"] = lat_pcts.mean(axis=1)  # NaN if any horizon is NaN

    lon_pcts = ranked[[f"pct_es_lon_{h}" for h in HORIZONS]]
    ranked["R_longitudinal"] = lon_pcts.mean(axis=1)

    # Label strongest contributing horizon for overall
    horizon_pcts = ranked[[f"pct_es_{h}" for h in HORIZONS]]
    ranked["strongest_overall_horizon"] = horizon_pcts.idxmax(axis=1).str.replace("pct_es_", "")

    return ranked


def select_review_set(ranked: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Select Top-K(R_overall) U Top-K(R_lateral), deduplicated."""
    top_overall = ranked.nlargest(top_k, "R_overall")
    top_overall = top_overall.assign(selection_reason="top_overall")

    lat_valid = ranked.dropna(subset=["R_lateral"])
    if len(lat_valid) >= top_k:
        top_lateral = lat_valid.nlargest(top_k, "R_lateral")
    else:
        top_lateral = lat_valid
    top_lateral = top_lateral.assign(selection_reason="top_lateral")

    combined = pd.concat([top_overall, top_lateral])
    # For duplicates, combine reasons
    deduped = (
        combined.groupby("npz_path", sort=False)
        .agg(
            {
                **{
                    c: "first"
                    for c in combined.columns
                    if c not in ("npz_path", "selection_reason")
                },
                "selection_reason": lambda x: "+".join(sorted(set(x))),
            }
        )
        .reset_index()
    )

    return deduped.sort_values("R_overall", ascending=False).reset_index(drop=True)


def plot_distributions(ranked: pd.DataFrame, output_path: str) -> None:
    """Plot score distributions for overall, longitudinal, and lateral ES."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#fcfcfb")

    for i, h in enumerate(HORIZONS):
        ax = axes[0, i]
        ax.set_facecolor("#fcfcfb")
        col = f"es_{h}"
        vals = ranked[col].dropna()
        ax.hist(vals, bins=50, color="#2a78d6", edgecolor="none", alpha=0.85)
        ax.set_title(f"Overall ES ({h})", fontsize=11)
        ax.set_xlabel("Energy Score")

        ax = axes[1, i]
        ax.set_facecolor("#fcfcfb")
        col = f"es_lat_{h}"
        vals = ranked[col].dropna()
        ax.hist(vals, bins=50, color="#d03b3b", edgecolor="none", alpha=0.85)
        ax.set_title(f"Lateral ES ({h})", fontsize=11)
        ax.set_xlabel("Energy Score")

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Stage 2: Rank scenes and select review set.")
    p.add_argument("--scores", required=True, help="scores.csv from Stage 1")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    ranked = compute_ranks(df)
    ranked.to_csv(out / "ranked.csv", index=False)

    review = select_review_set(ranked, args.top_k)
    review.to_csv(out / "review_set.csv", index=False)

    plot_distributions(ranked, str(out / "distributions.png"))

    print(f"Ranked {len(ranked)} scenes -> {out / 'ranked.csv'}")
    print(f"Selected {len(review)} review candidates -> {out / 'review_set.csv'}")
    print(
        f"  Overall top-{args.top_k}: {ranked.nlargest(args.top_k, 'R_overall')['npz_path'].tolist()}"
    )


if __name__ == "__main__":
    main()
