"""Filter a path_list.json by dropping npz whose mean distance from
``ego_agent_future`` to the nearest lane centerline point is >= ``--max_score``.

The per-npz score is the same metric computed by ``score_offroad_npz.py``.
This script bundles "compute + filter" so it can be used standalone like
``filter_collision_free_npz.py``.

Usage:
    python ros_scripts/filter_in_lanelet_npz.py path_list.json \\
        --save_path filtered_list.json \\
        --max_score 6.0
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_collision_free_npz import load_path_list  # noqa: E402
from score_offroad_npz import score_one  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path, help="Input path_list.json")
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument(
        "--max_score",
        type=float,
        default=6.0,
        help="Drop samples with mean centerline distance >= this (m).",
    )
    parser.add_argument(
        "--time_stride",
        type=int,
        default=1,
        help="Check every N-th ego_agent_future step (1 = all 80).",
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


def main() -> None:
    args = parse_args()
    paths = load_path_list(args.input_json)
    print(f"Loaded {len(paths)} paths from {args.input_json}")

    opts = {
        "time_stride": args.time_stride,
        "use_route_lanes_too": args.use_route_lanes_too,
    }
    jobs = [(p, opts) for p in paths]

    kept: list[str] = []
    dropped: list[tuple[str, float]] = []

    if args.num_workers <= 1:
        iterator = (score_one(j) for j in jobs)
    else:
        pool = Pool(processes=args.num_workers)
        iterator = pool.imap_unordered(score_one, jobs, chunksize=16)

    for path, info in tqdm(iterator, total=len(paths)):
        if info["score"] < args.max_score:
            kept.append(path)
        else:
            dropped.append((path, float(info["score"])))

    if args.num_workers > 1:
        pool.close()
        pool.join()

    kept.sort()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(kept, f, indent=4)

    log_path = args.save_path.with_suffix(".log")
    dropped.sort(key=lambda kv: -kv[1])
    with open(log_path, "w") as f:
        f.write(f"input_json: {args.input_json}\n")
        f.write(f"save_path: {args.save_path}\n")
        f.write(f"max_score: {args.max_score}\n")
        f.write(f"time_stride: {args.time_stride}\n")
        f.write(f"use_route_lanes_too: {args.use_route_lanes_too}\n")
        f.write(f"total: {len(paths)}\n")
        f.write(f"kept:  {len(kept)}\n")
        f.write(f"dropped: {len(dropped)}\n")
        f.write("\n--- dropped paths (score >= max_score) ---\n")
        for p, s in dropped:
            f.write(f"{s:.4f}\t{p}\n")

    print(f"kept {len(kept)} / {len(paths)} (dropped {len(dropped)})")
    print(f"Saved {args.save_path} (and log {log_path})")


if __name__ == "__main__":
    main()
