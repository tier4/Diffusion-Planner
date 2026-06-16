"""Per-arc CL + road-border metrics for one-or-more psim realized-ego bags.

For each candidate bag: extract realized ego poses (``_extract_poses_from_bag``),
project to the route arc (``project_to_polyline`` -> arc + |lateral| from the route
centerline = CL metric), and compute road-border distance via the reward OBB
(``_obb_corners`` + ``reward._point_to_segments_min_dist`` = RB metric). Bins by arc
and prints a side-by-side per-bin table (clμ mean|lat|, clmx max|lat|, rb min-dist,
X crossings < ``rb_cross_thresh``) for N models, plus an in-bounds total. Reuses
``_heatmap_common`` + ``reward.py`` + ``LaneletSceneBuilder`` only (no hand-rolled
geometry). Run under a ROS env (lanelet2).

Usage:
    python -m rlvr.autoresearch.tools.psim_per_arc_metrics \
        --route <route.pkl> --ego_shape WB,L,W [--bin_m 250] \
        [--front_cut 50] [--tail_cut 50] [--stride 10] \
        --models LABEL1 BAG1 LABEL2 BAG2 ...
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from rlvr.reward import RewardConfig, _point_to_segments_min_dist
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder, _obb_corners
from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    load_route,
    project_to_polyline,
)
from scenario_generation.tools.heatmap_route_deviation import _extract_poses_from_bag


def _peri(corners, n=8):
    """Densify an OBB into perimeter points (n per edge)."""
    out = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(a * (1 - t) + b * t)
    return np.array(out, dtype=np.float32)


def _series(bag, pts, arc, seg1, seg2, wb, length, width, stride):
    """Per realized pose: (arc, |lateral| from route CL, min road-border dist)."""
    poses = _extract_poses_from_bag(Path(bag))[::stride]
    out = []
    for x, y, yaw, _sp in poses:
        a, _sl, al = project_to_polyline(np.array([float(x), float(y)]), pts, arc)
        peri = _peri(_obb_corners(float(x), float(y), float(yaw), length, width, wheelbase=wb))
        rb = _point_to_segments_min_dist(torch.tensor(peri), seg1, seg2).min().item()
        out.append((float(a), float(al), float(rb)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W (no default — fail loudly)")
    ap.add_argument(
        "--bin_m", type=int, default=250, help="arc bin width in m (int — used as a range step)"
    )
    ap.add_argument(
        "--front_cut", type=float, default=50.0, help="skip first N m (ends not in-bounds)"
    )
    ap.add_argument("--tail_cut", type=float, default=50.0, help="skip last N m")
    ap.add_argument(
        "--stride", type=int, default=10, help="subsample ~100Hz localization to planning rate"
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="alternating LABEL BAG LABEL BAG ... (even count)",
    )
    args = ap.parse_args()

    parts = [float(v) for v in args.ego_shape.split(",")]
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise ValueError(
            f"--ego_shape must be 'WB,L,W' with 3 positive values; got {args.ego_shape!r}"
        )
    wb, length, width = parts
    if len(args.models) % 2 != 0:
        raise ValueError("--models must be alternating LABEL BAG pairs (even count)")
    labels = args.models[0::2]
    bags = args.models[1::2]

    thr = RewardConfig().rb_cross_thresh
    fc, tc = args.front_cut, args.tail_cut
    route = load_route(Path(args.route))
    pts, arc = build_route_polyline(route)
    amax = float(arc.max())
    b = LaneletSceneBuilder(str(route.map_path))
    s1, s2 = [], []
    for pl in b.road_border_polylines():
        pl = np.asarray(pl)[:, :2]
        if pl.shape[0] >= 2:
            s1.append(pl[:-1])
            s2.append(pl[1:])
    if not s1:
        raise SystemExit(
            f"map {route.map_path} has no road-border polylines (>=2 points) — cannot score RB"
        )
    seg1 = torch.tensor(np.concatenate(s1), dtype=torch.float32)
    seg2 = torch.tensor(np.concatenate(s2), dtype=torch.float32)

    data = {
        lab: _series(bg, pts, arc, seg1, seg2, wb, length, width, args.stride)
        for lab, bg in zip(labels, bags)
    }

    print("arc-bin     " + "".join(f"| {lab:>22} " for lab in labels))
    print("            " + "".join("|  clμ  clmx  rbmin  X  " for _ in labels))
    for lo in range(0, int(amax) + int(args.bin_m), int(args.bin_m)):
        hi = lo + args.bin_m
        if hi <= fc or lo >= amax - tc:
            continue
        row = f"{lo:>5}-{int(hi):<5} "
        for lab in labels:
            d = data[lab]
            m = (d[:, 0] >= lo) & (d[:, 0] < hi) & (d[:, 0] >= fc) & (d[:, 0] <= amax - tc)
            if m.sum() == 0:
                row += "|   -     -     -    - "
                continue
            cl, rb = d[m, 1], d[m, 2]
            row += f"| {cl.mean():4.2f} {cl.max():5.2f} {rb.min():5.2f} {int((rb < thr).sum()):>2} "
        print(row)
    # in-bounds total (same front/tail cut as the per-bin table)
    totals = {}
    for lab in labels:
        d = data[lab]
        inb = (d[:, 0] >= fc) & (d[:, 0] <= amax - tc)
        totals[lab] = int((d[inb, 2] < thr).sum())
    print(f"TOTAL in-bounds RB crossings (<{thr:.2f}m): {totals}")


if __name__ == "__main__":
    main()
