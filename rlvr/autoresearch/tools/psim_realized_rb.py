"""Realized closed-loop ego ROAD-BORDER crossings from a psim bag.

Scores the ACTUAL realized ego poses (from /localization/kinematic_state in the
psim .db3 bag) against the map's road borders — NOT the model's prediction.

Reuses (no hand-rolled geometry):
- bag ego world poses: heatmap_route_deviation._extract_poses_from_bag
- world road borders: LaneletSceneBuilder.road_border_polylines
- ego OBB (rear-axle): lanelet_scene_builder._obb_corners (wheelbase convention)
- border distance: reward._point_to_segments_min_dist, thresh = RewardConfig.rb_cross_thresh
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from scenario_generation.tools._heatmap_common import (
    build_route_polyline, load_route, project_to_polyline,
)
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder, _obb_corners
from scenario_generation.tools.heatmap_route_deviation import _extract_poses_from_bag
from rlvr.reward import _point_to_segments_min_dist, RewardConfig


def _densify(corners, n=8):
    pts = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(a * (1 - t) + b * t)
    return np.array(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--label", default="model")
    ap.add_argument("--stride", type=int, default=10,
                    help="subsample realized poses (localization ~100Hz; stride 10 -> ~10Hz/planning rate)")
    ap.add_argument("--localize", action="store_true",
                    help="bin RB crossings by route arc to show WHERE")
    ap.add_argument("--front_cut", type=float, default=50.0,
                    help="skip first N meters of the route (start not fully in-bounds)")
    ap.add_argument("--tail_cut", type=float, default=50.0,
                    help="skip last N meters of the route (end not fully in-bounds)")
    args = ap.parse_args()

    WB, L, W = [float(x) for x in args.ego_shape.split(",")]
    route = load_route(Path(args.route))
    builder = LaneletSceneBuilder(str(route.map_path))
    borders = builder.road_border_polylines()
    s1, s2 = [], []
    for pl in borders:
        pl = np.asarray(pl)[:, :2]
        if pl.shape[0] >= 2:
            s1.append(pl[:-1]); s2.append(pl[1:])
    seg1 = torch.tensor(np.concatenate(s1), dtype=torch.float32)
    seg2 = torch.tensor(np.concatenate(s2), dtype=torch.float32)

    poses = _extract_poses_from_bag(Path(args.bag))[::args.stride]  # subsample to planning rate
    thresh = RewardConfig().rb_cross_thresh

    # project every pose to route arc, then cut the front/tail (ends not in-bounds)
    pts, arc = build_route_polyline(route)
    arc_max = float(arc.max())
    pose_arc = np.array([float(project_to_polyline(np.array([float(p[0]), float(p[1])]), pts, arc)[0])
                         for p in poses])
    keep = (pose_arc >= args.front_cut) & (pose_arc <= arc_max - args.tail_cut)

    dists = np.empty(len(poses))
    for i, (x, y, yaw, _sp) in enumerate(poses):
        peri = _densify(_obb_corners(float(x), float(y), float(yaw), L, W, wheelbase=WB))
        dists[i] = _point_to_segments_min_dist(torch.tensor(peri), seg1, seg2).min().item()

    d_in = dists[keep]
    if d_in.size == 0:
        raise SystemExit(
            f"no in-bounds poses (front_cut={args.front_cut} tail_cut={args.tail_cut} "
            f"removed all {len(poses)} subsampled poses) — cannot compute RB metrics")
    cross_mask = keep & (dists < thresh)
    n_cross = int(cross_mask.sum())
    print(f"{args.label}: {int(keep.sum())} steps in-bounds (stride {args.stride}, cut {args.front_cut:.0f}/{args.tail_cut:.0f}m) | "
          f"RB crossings (<{thresh:.2f}m): {n_cross}/{int(keep.sum())} "
          f"| border dist: min={d_in.min():.2f}m p5={np.percentile(d_in,5):.2f} "
          f"p50={np.percentile(d_in,50):.2f} mean={d_in.mean():.2f}")
    if args.localize and n_cross:
        ca = pose_arc[cross_mask]
        print("  RB crossings by arc bin:")
        for lo in range(0, int(arc_max) + 250, 250):
            n = int(((ca >= lo) & (ca < lo + 250)).sum())
            if n:
                print(f"    {lo}-{lo+250}m: {n}")


if __name__ == "__main__":
    main()
