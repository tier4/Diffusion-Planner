"""Score each npz by the mean lateral distance from each ego_agent_future point
to the nearest lane centerline point.

For every step t in ``ego_agent_future`` (80 steps at 10 Hz), find the minimum
Euclidean distance to any valid ``lanes[..., :2]`` centerline point. The score
is the mean of those 80 distances. Larger score → ego is farther from any
lane centerline on average.

Usage:
    python ros_scripts/score_offroad_npz.py path_list.json \\
        --save_path offroad_scores.json
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_path_list(input_json: Path) -> list[str]:
    """Accept both legacy list format and sampling dict format {"seed": ..., "files": [...]}."""
    with open(input_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data["files"])
    return list(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument(
        "--time_stride",
        type=int,
        default=1,
        help="Check every N-th step of ego_agent_future (1 = all 80).",
    )
    parser.add_argument(
        "--use_route_lanes_too",
        action="store_true",
        help="Also include route_lanes centerline points (rarely needed).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
    )
    return parser.parse_args()


def collect_centerline_points(*lane_arrays: np.ndarray) -> np.ndarray:
    """Stack valid centerline (x, y) points from one or more lane arrays."""
    pts = []
    for lanes in lane_arrays:
        valid = np.abs(lanes[..., :2]).sum(axis=-1) > 1e-6
        pts.append(lanes[..., :2][valid])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def score_one(args_tuple) -> tuple[str, dict]:
    path, opts = args_tuple
    data = np.load(path, allow_pickle=True)
    ego_future = data["ego_agent_future"].astype(np.float32)
    ego_xy = ego_future[:: opts["time_stride"], :2]
    T_eval = ego_xy.shape[0]

    lane_arrays = [data["lanes"]]
    if opts["use_route_lanes_too"]:
        lane_arrays.append(data["route_lanes"])
    centerline_pts = collect_centerline_points(*lane_arrays)

    if centerline_pts.shape[0] == 0:
        return (
            str(path),
            {
                "score": float("inf"),
                "mean_distance": float("inf"),
                "max_distance": float("inf"),
                "argmax_step": 0,
                "total_steps": T_eval,
            },
        )

    # (T, M) pairwise squared distance, take min over centerline points.
    diff = ego_xy[:, None, :] - centerline_pts[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=-1).min(axis=-1))  # (T,)
    mean_d = float(d.mean())
    argmax = int(d.argmax())
    return (
        str(path),
        {
            "score": mean_d,
            "mean_distance": mean_d,
            "max_distance": float(d[argmax]),
            "argmax_step": argmax * opts["time_stride"],
            "total_steps": T_eval,
        },
    )


def main() -> None:
    args = parse_args()
    paths = load_path_list(args.input_json)
    print(f"Loaded {len(paths)} paths from {args.input_json}")

    opts = {
        "time_stride": args.time_stride,
        "use_route_lanes_too": args.use_route_lanes_too,
    }
    jobs = [(p, opts) for p in paths]

    results: dict[str, dict] = {}
    if args.num_workers <= 1:
        iterator = (score_one(j) for j in jobs)
    else:
        pool = Pool(processes=args.num_workers)
        iterator = pool.imap_unordered(score_one, jobs, chunksize=16)

    for path, info in tqdm(iterator, total=len(paths)):
        results[path] = info

    if args.num_workers > 1:
        pool.close()
        pool.join()

    results = dict(sorted(results.items(), key=lambda kv: kv[0]))

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=4)

    scores = np.array([v["score"] for v in results.values() if np.isfinite(v["score"])])
    print(f"Saved {args.save_path}")
    print(
        f"  count={len(scores)}  mean={scores.mean():.3f}  std={scores.std():.3f}  "
        f"min={scores.min():.3f}  max={scores.max():.3f}"
    )
    print(
        "  percentiles: "
        + ", ".join(f"p{p}={np.percentile(scores, p):.3f}" for p in [50, 75, 90, 95, 99])
    )


if __name__ == "__main__":
    main()
