"""Build PATH prototypes for anchor_following guidance (path mode).

The original build_prototypes.py clusters raw per-timestep (x, y) futures,
so KMeans distance is dominated by speed (longitudinal progress) and the
resulting library has many duplicate straight-at-different-speed modes and
almost no lateral variety. Path-mode anchor guidance ignores timing
entirely, so what it needs is a vocabulary of PATHS.

This tool resamples each GT ego future to M points equally spaced by
arc length (speed information removed), drops near-stationary futures
(total arc < min_arc metres — they have no path shape), and clusters the
resampled shapes. Output: (K, M, 2) .npy in the ego frame — directly
consumable by anchor_following, which accepts any polyline length.

Usage:
    python -m rlvr.autoresearch.tools.build_path_prototypes \
        --data_list <path_list.json> --output <path_prototypes_k8.npy> \
        --num_clusters 8 --num_points 40 --min_arc 5.0
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402


def _load_torch_kmeans():
    """Reuse torch_kmeans from diffusion_planner/sampling/build_prototypes.py.

    The sampling/ dir is repo code that is not shipped in the diffusion_planner
    wheel (setup.py only packages diffusion_planner.*), so it cannot be imported
    normally; load it from its file location next to the installed package.
    """
    import diffusion_planner

    src = Path(diffusion_planner.__file__).parent.parent / "sampling" / "build_prototypes.py"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found: torch_kmeans lives in the repo's sampling/ dir, which "
            "is not shipped in the diffusion_planner wheel — run from a source "
            "checkout / editable install."
        )
    spec = importlib.util.spec_from_file_location("_dp_build_prototypes", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.torch_kmeans


torch_kmeans = _load_torch_kmeans()


def resample_by_arclength(xy: np.ndarray, num_points: int) -> np.ndarray | None:
    """Resample a polyline [T, 2] to num_points equally spaced by arc length.

    Returns None for degenerate polylines (zero total arc).
    """
    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]
    if total <= 1e-6:
        return None
    s = np.linspace(0.0, total, num_points)
    out = np.empty((num_points, 2), dtype=np.float32)
    out[:, 0] = np.interp(s, arc, xy[:, 0])
    out[:, 1] = np.interp(s, arc, xy[:, 1])
    return out


def load_path_features(paths, num_points, min_arc, max_samples, seed):
    """Load ego futures, arc-length resample -> features [N, M*2] + kept arcs."""
    rng = np.random.default_rng(seed)
    if max_samples and max_samples < len(paths):
        paths = [paths[i] for i in rng.choice(len(paths), size=max_samples, replace=False)]

    feats, arcs = [], []
    for path in tqdm(paths, desc="Loading ego paths", unit="file"):
        try:
            fut = np.load(path, allow_pickle=True)["ego_agent_future"].astype(np.float32)
        except Exception as e:  # noqa: BLE001 - skip unreadable / malformed files
            tqdm.write(f"  [warn] skipping {path}: {e}")
            continue
        xy = fut[:, :2]
        # GT futures pad invalid steps with zeros. Drop zero rows OUTRIGHT
        # (not just the trailing pad): an interior (0,0) row would add a
        # spurious detour-to-origin to the arc.
        valid = ~((xy[:, 0] == 0.0) & (xy[:, 1] == 0.0))
        if valid.sum() < 10:
            continue
        xy = xy[valid]
        rs = resample_by_arclength(xy, num_points)
        if rs is None:
            continue
        arc = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
        if arc < min_arc:
            continue
        feats.append(rs.reshape(-1))
        arcs.append(arc)
    if not feats:
        raise RuntimeError("No valid ego paths found (all degenerate or below --min_arc).")
    return np.asarray(feats, dtype=np.float32), np.asarray(arcs)


def plot_path_prototypes(prototypes, counts, output_png):
    K = prototypes.shape[0]
    cmap = plt.get_cmap("tab20")
    frac = np.sqrt(counts / max(counts.max(), 1))
    fig, ax = plt.subplots(figsize=(9, 9))
    for k in range(K):
        ax.plot(
            prototypes[k, :, 0],
            prototypes[k, :, 1],
            "-",
            color=cmap(k % 20),
            lw=0.8 + 2.4 * frac[k],
            label=f"{k} (n={int(counts[k])})",
        )
    ax.plot(0, 0, "k*", ms=14)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{K} path prototypes (arc-length resampled, speed-free)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_list", required=True, help="JSON list of NPZ paths")
    p.add_argument("--output", required=True, help="output .npy (K, M, 2)")
    p.add_argument("--num_clusters", type=int, default=8)
    p.add_argument("--num_points", type=int, default=40, help="arc-length samples per path")
    p.add_argument(
        "--min_arc",
        type=float,
        default=5.0,
        help="drop futures with total arc below this (no path shape)",
    )
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.data_list) as f:
        paths = json.load(f)

    feats, arcs = load_path_features(
        paths, args.num_points, args.min_arc, args.max_samples, args.seed
    )
    print(
        f"Clustering {feats.shape[0]} paths (M={args.num_points}) into "
        f"K={args.num_clusters}; arc p5/p50/p95 = "
        f"{np.percentile(arcs, 5):.1f}/{np.percentile(arcs, 50):.1f}/"
        f"{np.percentile(arcs, 95):.1f} m"
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    centers, labels = torch_kmeans(
        torch.from_numpy(feats), args.num_clusters, args.iters, args.seed, device
    )
    prototypes = (
        centers.cpu().numpy().reshape(args.num_clusters, args.num_points, 2).astype(np.float32)
    )
    labels = labels.cpu().numpy()
    counts = np.bincount(labels, minlength=args.num_clusters)

    np.save(args.output, prototypes)
    print(f"Saved path prototypes {prototypes.shape} to {args.output}")
    # Sidecar counts file: the guidance GUI reads <output>_counts.npy for the
    # per-cluster frequency shown in the prototype gallery.
    counts_path = str(args.output).replace(".npy", "_counts.npy")
    np.save(counts_path, counts)
    print(f"Saved cluster counts to {counts_path}")
    print(
        f"  endpoint y spread over modes: "
        f"[{prototypes[:, -1, 1].min():.1f}, {prototypes[:, -1, 1].max():.1f}] m; "
        f"cluster sizes min={counts.min()} max={counts.max()}"
    )
    png = args.output.rsplit(".", 1)[0] + ".png"
    plot_path_prototypes(prototypes, counts, png)
    print(f"Saved visualization to {png}")


if __name__ == "__main__":
    main()
