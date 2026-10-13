import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_generation.canvas import Figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir

    target_dir_list = sorted(root_dir.glob("*/"))

    epoch_list = []
    score_list = []
    for target_dir in target_dir_list:
        try:
            elements = str(target_dir.name).split("_")
            epoch = int(elements[2])
        except:  # noqa: E722
            continue
        files = sorted((target_dir / "aggregator_metric").glob("*.parquet"))
        assert len(files) == 1
        f = files[0]
        df = pd.read_parquet(f)
        name = os.path.splitext(os.path.basename(f))[0]

        # log_nameがNoneでない行だけ残す
        df = df[df["log_name"].notnull()]

        mean = df["score"].mean()
        print(elements, mean)
        epoch_list.append(epoch)
        score_list.append(mean)

    df = pd.DataFrame({"epoch": epoch_list, "mean_score": score_list})
    df = df.sort_values(by="epoch")
    df.to_csv(root_dir / "mean_score.csv", index=False)

    save_dir = root_dir / "aggregated_score_graphs_dir"
    save_dir.mkdir(exist_ok=True)
    ax = Figure((6.4, 4.8), dpi=100).panel((0.14, 0.12, 0.82, 0.84))
    points = np.stack([epoch_list, score_list], 1)
    ax.lines([points], "#1f77b4")
    ax.markers(points, ".", size=6, color="#1f77b4")
    ax.xlabel("epoch")
    ax.ylabel("mean_score")
    save_path = save_dir / "mean_score.png"
    ax.save(save_path)
    print(f"save to {save_path}")
