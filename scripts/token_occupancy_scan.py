"""Scan a dataset of npz files and measure token-slot occupancy per input type.

Occupancy definition mirrors the encoder's mask_p:
- neighbors: a slot is valid if any of the last 6 history rows has a nonzero
  value (dims :8)
- lanes/route: valid if any of the 20 points has nonzero geometry (dims :8)
- polygons/line_strings: valid if any point is nonzero (all dims)
- static: nonzero row

Usage:
  uv run python scripts/token_occupancy_scan.py <dataset_root_with_npz>
"""

import argparse
import glob
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("root", help="directory scanned recursively for *.npz")
args = parser.parse_args()

files = sorted(glob.glob(args.root + "/**/*.npz", recursive=True))
print(f"files: {len(files)}", file=sys.stderr)

keys = {
    "neighbor_agents_past": ("neighbors", 320),
    "lanes": ("lanes", 140),
    "route_lanes": ("route", 25),
    "polygons": ("polygons", 10),
    "line_strings": ("line_strings", 60),
    "static_objects": ("static", 5),
}

counts = {name: [] for name, _ in keys.values()}
# distance of farthest valid lane / neighbor (center point) to see distance profile
far = {"lanes": [], "neighbors": []}

for i, f in enumerate(files):
    with np.load(f) as d:
        for k, (name, slots) in keys.items():
            x = d[k]
            if k == "neighbor_agents_past":  # (320, T, 11)
                # NeighborEncoder zeros all earlier history before masking.
                valid = (x[:, -6:, :8] != 0).any(axis=(1, 2))
            elif k in ("lanes", "route_lanes"):  # (P, 20, D)
                valid = (x[..., :8] != 0).any(axis=(1, 2))
            elif k in ("polygons", "line_strings"):  # (P, V, D)
                valid = (x != 0).any(axis=(1, 2))
            else:  # static (P, D)
                valid = (x != 0).any(axis=1)
            counts[name].append(int(valid.sum()))
            if k == "lanes" and valid.any():
                c = x[valid][:, 10, :2]  # mid point of each valid lane
                far["lanes"].append(float(np.linalg.norm(c, axis=1).max()))
            if k == "neighbor_agents_past" and valid.any():
                c = x[valid][:, -1, :2]  # current position
                far["neighbors"].append(float(np.linalg.norm(c, axis=1).max()))
    if (i + 1) % 1000 == 0:
        print(f"{i + 1}/{len(files)}", file=sys.stderr)

print(f"\n=== token slot occupancy over {len(files)} samples ===")
print(f"{'input':<14}{'slots':>6}{'mean':>8}{'p50':>6}{'p95':>6}{'p99':>6}{'max':>6}  mean_util")
total_mean = 0.0
for k, (name, slots) in keys.items():
    a = np.array(counts[name])
    total_mean += a.mean()
    print(
        f"{name:<14}{slots:>6}{a.mean():>8.1f}{int(np.percentile(a, 50)):>6}"
        f"{int(np.percentile(a, 95)):>6}{int(np.percentile(a, 99)):>6}{a.max():>6}"
        f"  {a.mean() / slots * 100:5.1f}%"
    )
fixed = 4  # ego, goal, ego_shape, turn_indicator
print(f"fixed tokens (ego/goal/shape/turn): {fixed}")
print(
    f"total slots: {564}, mean valid tokens: {total_mean + fixed:.1f} "
    f"({(total_mean + fixed) / 564 * 100:.1f}%)"
)

for name in far:
    a = np.array(far[name])
    print(
        f"farthest valid {name}: p50={np.percentile(a, 50):.0f}m "
        f"p95={np.percentile(a, 95):.0f}m max={a.max():.0f}m"
    )
