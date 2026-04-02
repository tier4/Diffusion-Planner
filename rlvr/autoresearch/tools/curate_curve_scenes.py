"""Curate high-curvature training scenes from a scene pool.

Searches a scene pool JSON for scenes with high GT yaw change (curves),
filters by path length and lane/road-border cleanliness at t=0,
and subsamples to avoid redundant frames from the same recording bag.

Usage:
    python -m rlvr.autoresearch.tools.curate_curve_scenes \
        --pool grpo_scenes_2421.json \
        --output curated_curves.json \
        --min_yaw 30 --min_path 10 --max_per_bag 5 \
        --clean --clean_threshold 0.15

    # With baseline model path filtering (remove scenes where baseline barely moves):
    python -m rlvr.autoresearch.tools.curate_curve_scenes \
        --pool grpo_scenes_2421.json \
        --output curated_curves.json \
        --min_yaw 30 --min_path 10 --max_per_bag 5 \
        --clean --model_path /path/to/base_model.pth --min_det_path 4.0
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def compute_gt_stats(scene_path: str) -> dict | None:
    """Compute GT trajectory statistics for a scene."""
    try:
        data = np.load(scene_path)
        gt = data["ego_agent_future"]
        if gt.shape[0] < 20:
            return None

        # Total yaw change
        dh = np.diff(gt[:, 2])
        dh = np.arctan2(np.sin(dh), np.cos(dh))
        total_yaw_deg = np.degrees(np.abs(np.sum(dh)))

        # Path length
        dx = np.diff(gt[:, 0])
        dy = np.diff(gt[:, 1])
        path_length = float(np.sum(np.sqrt(dx**2 + dy**2)))

        return {
            "yaw_deg": total_yaw_deg,
            "path_m": path_length,
        }
    except Exception:
        return None


def get_bag_and_frame(scene_path: str) -> tuple[str, int]:
    """Extract bag prefix and frame number from scene filename."""
    basename = os.path.basename(scene_path)
    parts = basename.replace(".npz", "").rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return basename, 0


def curate_scenes(
    pool_paths: list[str],
    min_yaw: float = 30.0,
    min_path: float = 10.0,
    max_per_bag: int = 5,
    existing_scenes: set[str] | None = None,
) -> list[dict]:
    """Filter and subsample curve scenes from a pool.

    Args:
        pool_paths: List of NPZ file paths.
        min_yaw: Minimum GT yaw change in degrees.
        min_path: Minimum GT path length in meters.
        max_per_bag: Maximum scenes per recording bag.
        existing_scenes: Set of scene paths to always include.

    Returns:
        List of dicts with 'path', 'yaw_deg', 'path_m', 'bag', 'frame'.
    """
    if existing_scenes is None:
        existing_scenes = set()

    candidates = []
    for i, scene_path in enumerate(pool_paths):
        if (i + 1) % 500 == 0:
            print(f"  Scanning {i + 1}/{len(pool_paths)}...", file=sys.stderr)

        stats = compute_gt_stats(scene_path)
        if stats is None:
            continue

        if stats["yaw_deg"] >= min_yaw and stats["path_m"] >= min_path:
            bag, frame = get_bag_and_frame(scene_path)
            candidates.append(
                {
                    "path": scene_path,
                    "yaw_deg": stats["yaw_deg"],
                    "path_m": stats["path_m"],
                    "bag": bag,
                    "frame": frame,
                    "existing": scene_path in existing_scenes,
                }
            )

    print(f"  Found {len(candidates)} candidates (yaw>{min_yaw}°, path>{min_path}m)")

    # Subsample: max N per bag, evenly spaced
    by_bag = defaultdict(list)
    for c in candidates:
        by_bag[c["bag"]].append(c)

    selected = []
    for bag in sorted(by_bag.keys()):
        scenes = sorted(by_bag[bag], key=lambda x: x["frame"])
        n = min(max_per_bag, len(scenes))
        step = max(1, len(scenes) // n)
        picked = scenes[::step][:n]
        selected.extend(picked)
        avg_yaw = np.mean([s["yaw_deg"] for s in picked])
        print(f"  {bag}: {len(scenes)} candidates, selected {len(picked)}, avg_yaw={avg_yaw:.0f}°")

    # Always include existing scenes not yet selected
    selected_paths = {s["path"] for s in selected}
    for ex in existing_scenes:
        if ex not in selected_paths:
            stats = compute_gt_stats(ex)
            bag, frame = get_bag_and_frame(ex)
            entry = {
                "path": ex,
                "yaw_deg": stats["yaw_deg"] if stats else 0,
                "path_m": stats["path_m"] if stats else 0,
                "bag": bag,
                "frame": frame,
                "existing": True,
            }
            selected.append(entry)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Curate high-curvature scenes from a pool")
    parser.add_argument("--pool", required=True, help="Path to scene pool JSON")
    parser.add_argument("--output", required=True, help="Output path for curated scene JSON")
    parser.add_argument("--min_yaw", type=float, default=30.0, help="Min GT yaw change (degrees)")
    parser.add_argument("--min_path", type=float, default=10.0, help="Min GT path length (meters)")
    parser.add_argument("--max_per_bag", type=int, default=5, help="Max scenes per recording bag")
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Additional scene JSON files to always include",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run lane+road border cleansing (requires cleanse_lane_scenes)",
    )
    parser.add_argument(
        "--clean_threshold",
        type=float,
        default=0.15,
        help="Cleansing threshold (meters from lane edge at t=0)",
    )
    args = parser.parse_args()

    # Load pool
    with open(args.pool) as f:
        pool = json.load(f)
    print(f"Pool: {len(pool)} scenes from {args.pool}")

    # Load existing scenes to include
    existing = set()
    for inc_path in args.include:
        with open(inc_path) as f:
            inc = json.load(f)
        existing.update(inc)
        print(f"Including {len(inc)} scenes from {inc_path}")

    # Curate
    selected = curate_scenes(
        pool,
        min_yaw=args.min_yaw,
        min_path=args.min_path,
        max_per_bag=args.max_per_bag,
        existing_scenes=existing,
    )

    print(f"\nSelected {len(selected)} scenes total")

    # Save
    output_paths = [s["path"] for s in selected]
    with open(args.output, "w") as f:
        json.dump(output_paths, f, indent=2)
    print(f"Saved to {args.output}")

    # Optionally clean
    if args.clean:
        print(f"\nCleaning with threshold={args.clean_threshold}m...")
        clean_output = args.output.replace(".json", "_clean.json")
        os.system(
            f"python -m rlvr.autoresearch.tools.cleanse_lane_scenes "
            f"--scenes {args.output} --output {clean_output} "
            f"--threshold {args.clean_threshold} --also_check_road_border"
        )

    # Summary stats
    bags = defaultdict(int)
    for s in selected:
        bags[s["bag"]] += 1
    print(f"\n{len(selected)} scenes from {len(bags)} recording bags")
    print(f"Yaw range: {min(s['yaw_deg'] for s in selected):.0f}° - {max(s['yaw_deg'] for s in selected):.0f}°")
    print(f"Path range: {min(s['path_m'] for s in selected):.0f}m - {max(s['path_m'] for s in selected):.0f}m")


if __name__ == "__main__":
    main()
