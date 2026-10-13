import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_generation.canvas import Figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_dir = args.metrics_dir

    files = sorted(metrics_dir.glob("*.parquet"))
    metrics = {}
    mean_metrics = {}
    for f in files:
        df = pd.read_parquet(f)
        name = os.path.splitext(os.path.basename(f))[0]
        metrics[name] = df
        if df["metric_score"].mean() > 0:
            mean_metrics[name] = df["metric_score"].mean()

    print(mean_metrics)

    # Plot "metric_score"
    save_dir = metrics_dir.parent / "metric_graphs"
    save_dir.mkdir(exist_ok=True)
    ax = Figure((6.4, 4.8), dpi=100).panel((0.4, 0.1, 0.56, 0.86))
    positions = np.arange(len(mean_metrics))
    ax.bars(positions, list(mean_metrics.values()), width=0.8, horizontal=True)
    ax.yticks(positions, list(mean_metrics.keys()))
    ax.save(save_dir / "mean.png")
