import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_generation.canvas import Figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir1", type=Path)
    parser.add_argument("metrics_dir2", type=Path)
    parser.add_argument("label1", type=Path)
    parser.add_argument("label2", type=Path)
    return parser.parse_args()


def load_mean_metrics(metrics_dir):
    files = sorted(metrics_dir.glob("*.parquet"))
    mean_metrics = {}
    for f in files:
        df = pd.read_parquet(f)
        name = os.path.splitext(os.path.basename(f))[0]
        if df["metric_score"].mean() > 0:
            mean_metrics[name] = df["metric_score"].mean()
    return mean_metrics


if __name__ == "__main__":
    args = parse_args()
    m1 = load_mean_metrics(args.metrics_dir1 / "metrics")
    m2 = load_mean_metrics(args.metrics_dir2 / "metrics")

    # 共通指標に揃える
    all_keys = sorted(set(m1.keys()) | set(m2.keys()))
    v1 = [m1.get(k, 0) for k in all_keys]
    v2 = [m2.get(k, 0) for k in all_keys]

    y = np.arange(len(all_keys))
    h = 0.4

    ax = Figure((8, 6), dpi=100).panel((0.4, 0.1, 0.56, 0.86))
    ax.bars(y + h / 2, v1, width=h, color="#1f77b4", horizontal=True, label=str(args.label1))
    ax.bars(y - h / 2, v2, width=h, color="#ff7f0e", horizontal=True, label=str(args.label2))
    ax.yticks(y, all_keys)
    ax.xlabel("metric_score")
    ax.legend("lower right")

    save_dir = args.metrics_dir1 / "metric_graphs_compare"
    save_dir.mkdir(exist_ok=True)
    ax.save(save_dir / "compare.png")
