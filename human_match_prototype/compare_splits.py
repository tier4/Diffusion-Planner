"""Compare Energy Score distributions across train, train_sft and validation.

The train and train_sft samples are scored together as one union list, so this
splits the scored rows back out by membership in each sample list before
comparing. All three samples are the same size (156,204), so differences in the
distributions are attributable to the split, not to sample size.

The model was trained on train/train_sft, so their scores measure fit; the
validation scores measure generalisation. A materially heavier tail on
validation is the signature of overfitting.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from human_match_prototype.energy_score import HORIZONS

METRICS = [f"es_{h}" for h in HORIZONS] + [f"es_lat_{h}" for h in HORIZONS]
COLORS = {"train": "#2a78d6", "train_sft": "#d08a3b", "validation": "#3bb273"}


def summarise(df: pd.DataFrame, name: str) -> dict:
    row = {"split": name, "n": len(df)}
    for m in METRICS:
        s = df[m].dropna()
        row[f"{m}_median"] = s.median()
        row[f"{m}_p90"] = s.quantile(0.90)
        row[f"{m}_p99"] = s.quantile(0.99)
    row["route_valid_pct"] = df["route_valid"].mean() * 100
    return row


def plot_cdfs(splits: dict[str, pd.DataFrame], out_png: Path) -> None:
    """Log-x CDFs — the tail is what distinguishes fit from generalisation."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#fcfcfb")
    for j, h in enumerate(HORIZONS):
        for i, prefix in enumerate(["es_", "es_lat_"]):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            for name, df in splits.items():
                v = df[f"{prefix}{h}"].dropna().to_numpy()
                v = v[v > 0]
                if len(v) == 0:
                    continue
                v.sort()
                ax.plot(
                    v,
                    np.arange(1, len(v) + 1) / len(v),
                    label=f"{name} (n={len(v):,})",
                    color=COLORS.get(name),
                    lw=1.6,
                )
            ax.set_xscale("log")
            ax.set_title(f"{'Overall' if prefix == 'es_' else 'Lateral'} ES {h}", fontsize=11)
            ax.set_xlabel("Energy Score (m, log)")
            ax.set_ylabel("CDF")
            ax.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare ES distributions across splits.")
    p.add_argument("--union_scores", required=True, help="scores_union.csv (train + sft scored)")
    p.add_argument("--train_list", required=True)
    p.add_argument("--sft_list", required=True)
    p.add_argument("--valid_scores", required=True)
    p.add_argument("--mirror", default="/mnt/nvme1/chenglin/train_eval_mirror")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--match_n",
        type=int,
        default=None,
        help="Subsample every split to this many rows so all splits are equal-sized. "
        "Validation is already scored in full, so matching it down costs nothing.",
    )
    p.add_argument("--match_seed", type=int, default=11)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    union = pd.read_csv(args.union_scores)

    # score_scenes records mirror-prefixed paths; strip back to the canonical
    # absolute path so rows can be matched against the sample lists.
    mirror = args.mirror.rstrip("/")
    union["key"] = union["npz_path"].str.replace(f"^{mirror}", "", regex=True)

    train_keys = set(json.load(open(args.train_list)))
    sft_keys = set(json.load(open(args.sft_list)))

    splits = {
        "train": union[union.key.isin(train_keys)],
        "train_sft": union[union.key.isin(sft_keys)],
        "validation": pd.read_csv(args.valid_scores),
    }

    if args.match_n:
        rng = np.random.default_rng(args.match_seed)
        for name, df in splits.items():
            if len(df) > args.match_n:
                keep = rng.choice(len(df), size=args.match_n, replace=False)
                splits[name] = df.iloc[sorted(keep)]
            elif len(df) < args.match_n:
                print(f"  note: {name} has only {len(df):,} rows (< match_n={args.match_n:,})")
        print(f"matched all splits to n={args.match_n:,}")

    summary = pd.DataFrame([summarise(df, n) for n, df in splits.items()])
    summary.to_csv(out / "split_summary.csv", index=False)

    print(f"{'split':<12}{'n':>9}  " + "".join(f"{m + ' med':>16}" for m in METRICS[:3]))
    for _, r in summary.iterrows():
        print(
            f"{r['split']:<12}{int(r['n']):>9,}  "
            + "".join(f"{r[m + '_median']:>16.3f}" for m in METRICS[:3])
        )

    # Two-sample KS against validation quantifies whether the gap is real.
    tests = []
    for name in ("train", "train_sft"):
        for m in METRICS:
            a = splits[name][m].dropna()
            b = splits["validation"][m].dropna()
            ks, pv = stats.ks_2samp(a, b)
            tests.append(
                {
                    "split": name,
                    "metric": m,
                    "ks_stat": ks,
                    "p_value": pv,
                    "median_ratio": a.median() / b.median() if b.median() else np.nan,
                }
            )
    pd.DataFrame(tests).to_csv(out / "ks_vs_validation.csv", index=False)

    plot_cdfs(splits, out / "cdf_comparison.png")
    print(f"\nwrote split_summary.csv, ks_vs_validation.csv, cdf_comparison.png -> {out}")


if __name__ == "__main__":
    main()
