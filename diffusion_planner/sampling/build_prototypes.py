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

"""Build trajectory-mode prototypes (anchors) from a dataset of GT ego futures.

Runs fixed-K KMeans directly on the flattened ego_agent_future (x, y) waypoints, so the
KMeans cluster centers ARE the mean trajectories of each mode -- a compact "vocabulary" of
plausible maneuvers (straight at various speeds, turns, stop, lane changes, ...).

The output is a ``(K, T, 2)`` .npy array in the ego-centric frame (origin at the ego t=0
pose), the same format consumed by ``model/guidance/anchor_following.py``.

Usage:
    python build_prototypes.py \
        --data_list /path/to/path_list_train.json \
        --output    /path/to/prototypes.npy \
        --num_clusters 64
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from tqdm import tqdm  # noqa: E402


def torch_kmeans(x, k, iters, seed, device):
    """Plain Lloyd's KMeans with k-means++ init, on ``device``.

    Args:
        x: [N, D] float tensor of features.
        k: number of clusters.
        iters: max Lloyd iterations.
        seed: RNG seed.
        device: torch device.

    Returns:
        centers: [k, D] cluster centers.
        labels: [N] assignment.
    """
    x = x.to(device)
    g = torch.Generator(device=device).manual_seed(seed)

    # k-means++ initialization.
    centers = [x[torch.randint(x.shape[0], (1,), generator=g, device=device)].squeeze(0)]
    min_sq = torch.full((x.shape[0],), float("inf"), device=device)
    for _ in range(1, k):
        min_sq = torch.minimum(min_sq, ((x - centers[-1]) ** 2).sum(-1))
        probs = min_sq / min_sq.sum().clamp_min(1e-12)
        nxt = torch.multinomial(probs, 1, generator=g)
        centers.append(x[nxt].squeeze(0))
    centers = torch.stack(centers)  # [k, D]

    labels = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    for _ in range(iters):
        labels = torch.cdist(x, centers).argmin(dim=1)  # [N]
        new_centers = centers.clone()
        for c in range(k):
            members = labels == c
            if members.any():
                new_centers[c] = x[members].mean(dim=0)
            else:  # reseed an empty cluster to the worst-fit point
                new_centers[c] = x[torch.cdist(x, centers).amin(dim=1).argmax()]
        if torch.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels


def plot_prototypes(prototypes, counts, output_png):
    """Plot the prototype trajectories (ego frame) and their endpoint spread.

    prototypes: [K, T, 2] ; counts: [K] number of members per mode (sets line/marker size).
    """
    K, T, _ = prototypes.shape
    end_x, end_y = prototypes[:, -1, 0], prototypes[:, -1, 1]
    span = max(np.abs(end_y).max(), 1e-3)
    norm = Normalize(vmin=-span, vmax=span)  # blue=right, red=left
    cmap = plt.get_cmap("coolwarm")
    # line width / marker size scale with cluster population (sqrt for a gentle spread).
    frac = np.sqrt(counts / max(counts.max(), 1))

    fig, (ax_xy, ax_end) = plt.subplots(1, 2, figsize=(15, 7.5))
    for k in range(K):
        c = cmap(norm(end_y[k]))
        ax_xy.plot(
            prototypes[k, :, 0],
            prototypes[k, :, 1],
            "-",
            color=c,
            lw=0.6 + 2.4 * frac[k],
            alpha=0.85,
        )
        ax_xy.plot(end_x[k], end_y[k], "o", color=c, ms=2 + 4 * frac[k])
    ax_xy.plot(0, 0, "k*", ms=14, zorder=5, label="ego t=0")
    ax_xy.set_title(f"{K} trajectory-mode prototypes (ego frame; width ~ #members)")
    ax_xy.set_xlabel("x forward [m]")
    ax_xy.set_ylabel("y left [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="upper left")
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_xy,
        fraction=0.046,
        pad=0.04,
        label="endpoint y (left +) [m]",
    )

    ax_end.scatter(end_x, end_y, c=end_y, cmap=cmap, norm=norm, s=20 + 180 * frac)
    for k in range(K):
        ax_end.annotate(str(int(counts[k])), (end_x[k], end_y[k]), fontsize=6, alpha=0.6)
    ax_end.plot(0, 0, "k*", ms=14)
    ax_end.set_title("prototype endpoints (label = #members)")
    ax_end.set_xlabel("endpoint x forward [m]  (~ speed over horizon)")
    ax_end.set_ylabel("endpoint y left [m]")
    ax_end.set_aspect("equal", adjustable="box")
    ax_end.grid(True, alpha=0.3)

    fig.suptitle(f"KMeans prototypes: K={K}, T={T}, N={int(counts.sum())}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def get_args():
    p = argparse.ArgumentParser(description="Build KMeans trajectory prototypes from GT futures")
    p.add_argument(
        "--data_list",
        type=str,
        required=True,
        help="JSON list of NPZ paths (same format as train_predictor.py)",
    )
    p.add_argument("--output", type=str, required=True, help="output path for prototypes .npy")
    p.add_argument("--num_clusters", type=int, default=64, help="number of trajectory modes K")
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="if >0, randomly subsample this many trajectories before clustering",
    )
    p.add_argument("--iters", type=int, default=100, help="max Lloyd iterations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_xy_futures(paths, max_samples, seed):
    """Load ego_agent_future (x, y) for each NPZ -> features [N, T*2], plus T."""
    rng = np.random.default_rng(seed)
    if max_samples and max_samples < len(paths):
        paths = [paths[i] for i in rng.choice(len(paths), size=max_samples, replace=False)]

    feats, T = [], None
    for path in tqdm(paths, desc="Loading ego futures", unit="file"):
        try:
            ego_future = np.load(path, allow_pickle=True)["ego_agent_future"].astype(np.float32)
        except Exception as e:  # noqa: BLE001 - skip unreadable / malformed files
            tqdm.write(f"  [warn] skipping {path}: {e}")
            continue
        if T is None:
            T = ego_future.shape[0]
        if ego_future.shape[0] != T:
            tqdm.write(f"  [warn] skipping {path}: T={ego_future.shape[0]} != {T}")
            continue
        feats.append(ego_future[:, :2].reshape(-1))  # [T*2]
    if not feats:
        raise RuntimeError("No valid NPZ files found.")
    return np.asarray(feats, dtype=np.float32), T


def main():
    args = get_args()
    with open(args.data_list, "r", encoding="utf-8") as f:
        paths = json.load(f)

    features, T = load_xy_futures(paths, args.max_samples, args.seed)
    print(
        f"Clustering {features.shape[0]} trajectories (T={T}) into K={args.num_clusters} modes..."
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    centers, labels = torch_kmeans(
        torch.from_numpy(features), args.num_clusters, args.iters, args.seed, device
    )
    prototypes = centers.cpu().numpy().reshape(args.num_clusters, T, 2).astype(np.float32)
    labels = labels.cpu().numpy()

    np.save(args.output, prototypes)
    counts = np.bincount(labels, minlength=args.num_clusters)
    print(f"Saved prototypes {prototypes.shape} to {args.output}")
    print(f"  cluster sizes: min={counts.min()} max={counts.max()} mean={counts.mean():.0f}")
    print(
        f"  endpoint spread (x,y range over modes): "
        f"x[{prototypes[:, -1, 0].min():.1f},{prototypes[:, -1, 0].max():.1f}] "
        f"y[{prototypes[:, -1, 1].min():.1f},{prototypes[:, -1, 1].max():.1f}]"
    )

    # Visualize the modes right after building them.
    plot_png = os.path.splitext(args.output)[0] + ".png"
    plot_prototypes(prototypes, counts, plot_png)
    print(f"Saved prototype visualization to {plot_png}")


if __name__ == "__main__":
    main()
