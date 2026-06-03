"""Select NPZ scenes whose ego arc position falls within specified ranges.

Reuses geometry primitives from _heatmap_common.py.  Optionally injects
ego_shape into selected NPZs (needed when source NPZs lack it).

Usage::

    python -m scenario_generation.tools.select_npz_by_arc_range \
        --npz_dir /path/to/npz/ \
        --route /path/to/route.pkl \
        --arc_ranges 450,510 728,768 1400,1442 \
        --min_spacing_m 5 \
        --inject_ego_shape 4.76,7.24,2.29 \
        --output selected.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    load_route,
    project_to_polyline,
    recover_ego_world_pose_from_goal,
)


def _parse_arc_ranges(raw: list[str]) -> list[tuple[float, float]]:
    ranges = []
    for r in raw:
        lo, hi = r.split(",")
        ranges.append((float(lo), float(hi)))
    return ranges


def _in_any_range(arc: float, ranges: list[tuple[float, float]]) -> bool:
    return any(lo <= arc <= hi for lo, hi in ranges)


def _declutter(entries: list[dict], min_spacing_m: float) -> list[dict]:
    if min_spacing_m <= 0:
        return entries
    entries_sorted = sorted(entries, key=lambda e: e["arc"])
    kept: list[dict] = []
    last_arc = -1e9
    for e in entries_sorted:
        if e["arc"] - last_arc >= min_spacing_m:
            kept.append(e)
            last_arc = e["arc"]
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_dir", required=True, help="Directory with NPZ files")
    p.add_argument("--route", required=True, help="Route pickle path")
    p.add_argument(
        "--arc_ranges",
        nargs="+",
        required=True,
        help="Arc ranges as lo,hi pairs (e.g. 450,510 728,768)",
    )
    p.add_argument("--min_spacing_m", type=float, default=5.0)
    p.add_argument(
        "--inject_ego_shape",
        default=None,
        help="WB,L,W to inject into selected NPZs (e.g. 4.76,7.24,2.29)",
    )
    p.add_argument("--output", required=True, help="Output JSON scene list")
    p.add_argument(
        "--speed_thresh",
        type=float,
        default=1.0,
        help="Min ego speed (m/s) to keep a scene",
    )
    p.add_argument(
        "--val_hold_every",
        type=int,
        default=0,
        help="Hold out every Nth scene for val (0=no split)",
    )
    args = p.parse_args()

    arc_ranges = _parse_arc_ranges(args.arc_ranges)
    print(f"Arc ranges: {arc_ranges}")

    route = load_route(Path(args.route))
    pts, s = build_route_polyline(route)
    print(f"Route polyline: {len(pts)} pts, total arc {s[-1]:.1f}m")

    npz_dir = Path(args.npz_dir)
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"No NPZ files found in {npz_dir}")
    print(f"Scanning {len(npz_files)} NPZ files ...")

    ego_shape_np = None
    if args.inject_ego_shape:
        vals = [float(x) for x in args.inject_ego_shape.split(",")]
        assert len(vals) == 3, "--inject_ego_shape must be WB,L,W"
        ego_shape_np = np.array(vals, dtype=np.float32)

    matched: list[dict] = []
    skipped_speed = 0
    for i, npz_path in enumerate(npz_files):
        d = np.load(npz_path)
        state = d["ego_current_state"]
        speed = float(np.sqrt(state[4] ** 2 + state[5] ** 2))
        if speed < args.speed_thresh:
            skipped_speed += 1
            continue
        ex, ey, eyaw = recover_ego_world_pose_from_goal(d["goal_pose"], route)
        arc, lat_signed, lat_abs = project_to_polyline(
            np.array([ex, ey]), pts, s
        )
        if _in_any_range(arc, arc_ranges):
            matched.append(
                {
                    "path": str(npz_path),
                    "arc": arc,
                    "lat": lat_signed,
                    "speed": speed,
                }
            )
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(npz_files)} scanned, {len(matched)} matched")

    print(
        f"Scan complete: {len(matched)} matched, {skipped_speed} skipped (low speed)"
    )

    kept = _declutter(matched, args.min_spacing_m)
    print(f"After declutter (spacing {args.min_spacing_m}m): {len(kept)} scenes")

    train_scenes: list[dict] = []
    val_scenes: list[dict] = []
    for i, e in enumerate(kept):
        if args.val_hold_every > 0 and (i % args.val_hold_every == 0):
            val_scenes.append(e)
        else:
            train_scenes.append(e)

    if ego_shape_np is not None:
        all_selected = train_scenes + val_scenes
        print(f"Injecting ego_shape {ego_shape_np.tolist()} into {len(all_selected)} NPZs ...")
        for e in all_selected:
            p_path = Path(e["path"])
            d = dict(np.load(p_path))
            if "ego_shape" not in d or not np.allclose(
                d["ego_shape"], ego_shape_np, atol=1e-4
            ):
                d["ego_shape"] = ego_shape_np
                np.savez(p_path, **d)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_list = [e["path"] for e in train_scenes]
    with open(out_path, "w") as f:
        json.dump(scene_list, f, indent=2)
    print(f"Wrote {len(scene_list)} train scenes -> {out_path}")

    if val_scenes:
        val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
        val_list = [e["path"] for e in val_scenes]
        with open(val_path, "w") as f:
            json.dump(val_list, f, indent=2)
        print(f"Wrote {len(val_list)} val scenes -> {val_path}")

    summary = {
        "arc_ranges": arc_ranges,
        "total_scanned": len(npz_files),
        "matched": len(matched),
        "skipped_speed": skipped_speed,
        "after_declutter": len(kept),
        "train": len(train_scenes),
        "val": len(val_scenes),
        "per_range": {},
    }
    for lo, hi in arc_ranges:
        n = sum(1 for e in kept if lo <= e["arc"] <= hi)
        summary["per_range"][f"{lo}-{hi}"] = n
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {summary_path}")
    for k, v in summary["per_range"].items():
        print(f"  arc {k}: {v} scenes")


if __name__ == "__main__":
    main()
