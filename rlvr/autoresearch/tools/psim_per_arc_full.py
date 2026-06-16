#!/usr/bin/env python3
"""Per-arc FULL-distribution psim certification: centerline + road-border +
realized comfort (lateral accel, jerk), every metric with p5/p25/p50/p95, one
table per route, side-by-side models vs baseline.

This unifies the two prior per-arc tools so a candidate is judged on the
complete dynamic+geometric picture the campaign protocol requires:
  - centerline |lateral| (CL bias): mean / p5 / p25 / p50 / p95
  - road-border dist (RB): min / p5 / p25 / p50  + crossings (<rb_cross_thresh)
  - realized lateral accel |accel.y| (felt comfort): mean / p50 / p95 / max
  - realized lateral jerk |d(accel.y)/dt|: p50 / p95 / max

Geometry (CL/RB) reuses psim_per_arc_metrics' reward-OBB path (no hand-rolled
geometry). Comfort reuses psim_comfort_heatmap's MEASURED-twist path (accel.y
from /localization/acceleration, jerk = SG 1st-derivative with dt from
timestamps — never a position derivative). Both signals are binned along the
same route arc, per bag.

Usage:
    python -m rlvr.autoresearch.tools.psim_per_arc_full \
        --route <route.pkl> --ego_shape WB,L,W [--bin_m 100] \
        [--front_cut 50] [--tail_cut 50] [--stride 10] \
        --models LABEL1 BAG1 LABEL2 BAG2 ... \
        [--output <table.json>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rlvr.autoresearch.tools.psim_comfort_heatmap import (
    _extract_accel_y,
    _extract_kinematics,
    compute_dynamics,
)
from rlvr.reward import RewardConfig, _point_to_segments_min_dist
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder, _obb_corners
from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    load_route,
    project_to_polyline,
)
from scenario_generation.tools.heatmap_route_deviation import _extract_poses_from_bag


def _peri(corners, n=8):
    out = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(a * (1 - t) + b * t)
    return np.array(out, dtype=np.float32)


def _geom_series(bag, pts, arc, seg1, seg2, wb, length, width, stride):
    """Per realized pose: (arc, |lateral| from route CL, min road-border dist)."""
    poses = _extract_poses_from_bag(Path(bag))[::stride]
    out = []
    for x, y, yaw, _sp in poses:
        a, _sl, al = project_to_polyline(np.array([float(x), float(y)]), pts, arc)
        peri = _peri(_obb_corners(float(x), float(y), float(yaw), length, width, wheelbase=wb))
        rb = _point_to_segments_min_dist(torch.tensor(peri), seg1, seg2).min().item()
        out.append((float(a), float(al), float(rb)))
    return np.array(out)


def _comfort_series(bag, pts, arc, stride, max_jump_speed=30.0):
    """Per realized pose: (arc, lat_accel |accel.y|, jerk) from measured twist."""
    t, x, y, speed, yaw_rate = _extract_kinematics(bag)
    sl = slice(None, None, stride)
    t, x, y, speed, yaw_rate = t[sl], x[sl], y[sl], speed[sl], yaw_rate[sl]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    t_acc, accel_y = _extract_accel_y(bag)
    lat_meas = np.interp(t, t_acc, np.abs(accel_y))
    dyn, _ = compute_dynamics(
        speed, yaw_rate, dt, lat_accel_meas=lat_meas, max_jump_speed=max_jump_speed
    )
    a_arc = np.array(
        [project_to_polyline(np.array([float(px), float(py)]), pts, arc)[0] for px, py in zip(x, y)]
    )
    return a_arc, dyn["lat_accel"], dyn["jerk"]


def _pcts(v, ps):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in ps}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--route", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W (no default — fail loudly)")
    ap.add_argument("--bin_m", type=int, default=100)
    ap.add_argument("--front_cut", type=float, default=50.0)
    ap.add_argument("--tail_cut", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max_jump_speed", type=float, default=30.0)
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="alternating LABEL BAG LABEL BAG ... (even count)",
    )
    ap.add_argument("--output", help="optional JSON path for the full per-arc table")
    args = ap.parse_args()

    parts = [float(v) for v in args.ego_shape.split(",")]
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise ValueError(f"--ego_shape must be 'WB,L,W' 3 positive values; got {args.ego_shape!r}")
    wb, length, width = parts
    if len(args.models) % 2 != 0:
        raise ValueError("--models must be alternating LABEL BAG pairs (even count)")
    labels, bags = args.models[0::2], args.models[1::2]

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
        raise SystemExit(f"map {route.map_path} has no road-border polylines — cannot score RB")
    seg1 = torch.tensor(np.concatenate(s1), dtype=torch.float32)
    seg2 = torch.tensor(np.concatenate(s2), dtype=torch.float32)

    geom = {
        lab: _geom_series(bg, pts, arc, seg1, seg2, wb, length, width, args.stride)
        for lab, bg in zip(labels, bags)
    }
    comf = {
        lab: _comfort_series(bg, pts, arc, args.stride, args.max_jump_speed)
        for lab, bg in zip(labels, bags)
    }

    report = {"route": str(args.route), "bin_m": args.bin_m, "bins": [], "totals": {}}
    print(
        f"\n=== PER-ARC FULL DISTRIBUTIONS (bin={args.bin_m}m, route={Path(args.route).stem}) ==="
    )
    for lo in range(0, int(amax) + int(args.bin_m), int(args.bin_m)):
        hi = lo + args.bin_m
        if hi <= fc or lo >= amax - tc:
            continue
        binrow = {"arc": f"{lo}-{int(hi)}", "models": {}}
        print(f"\narc {lo:>5}-{int(hi):<5}")
        for lab in labels:
            g = geom[lab]
            mg = (g[:, 0] >= lo) & (g[:, 0] < hi) & (g[:, 0] >= fc) & (g[:, 0] <= amax - tc)
            ca, la, ja = comf[lab]
            mc = (ca >= lo) & (ca < hi) & (ca >= fc) & (ca <= amax - tc)
            ent = {}
            if mg.sum() > 0:
                cl, rb = g[mg, 1], g[mg, 2]
                ent["cl"] = {"mean": round(float(cl.mean()), 3), **_pcts(cl, [5, 25, 50, 95])}
                ent["rb"] = {
                    "min": round(float(rb.min()), 3),
                    **_pcts(rb, [5, 25, 50]),
                    "cross": int((rb < thr).sum()),
                }
            if mc.sum() > 0:
                lat, jrk = la[mc], ja[mc]
                ent["lat_accel"] = {
                    "mean": round(float(np.nanmean(lat)), 3),
                    **_pcts(lat, [50, 95]),
                    "max": round(float(np.nanmax(lat)), 3),
                }
                ent["jerk"] = {**_pcts(jrk, [50, 95]), "max": round(float(np.nanmax(jrk)), 3)}
            binrow["models"][lab] = ent
            c = ent.get("cl", {})
            r = ent.get("rb", {})
            a = ent.get("lat_accel", {})
            j = ent.get("jerk", {})
            print(
                f"  {lab:>20} | cl μ{c.get('mean', '-')} p25 {c.get('p25', '-')} p95 {c.get('p95', '-')} "
                f"| rb min {r.get('min', '-')} p5 {r.get('p5', '-')} X{r.get('cross', '-')} "
                f"| latA p95 {a.get('p95', '-')} max {a.get('max', '-')} | jerk p95 {j.get('p95', '-')}"
            )
        report["bins"].append(binrow)

    # in-bounds totals: RB crossings + global comfort percentiles
    print("\n=== IN-BOUNDS TOTALS ===")
    for lab in labels:
        g = geom[lab]
        inb = (g[:, 0] >= fc) & (g[:, 0] <= amax - tc)
        ca, la, ja = comf[lab]
        ic = (ca >= fc) & (ca <= amax - tc)
        if not inb.any() or not ic.any():
            raise RuntimeError(
                f"{lab}: no samples within the in-bounds arc window [{fc}, {amax - tc}] "
                "(bag too short, or front_cut+tail_cut exceed the route length)"
            )
        nx = int((g[inb, 2] < thr).sum())
        cl_all = g[inb, 1]
        tot = {
            "rb_cross": nx,
            "cl": {"mean": round(float(cl_all.mean()), 3), **_pcts(cl_all, [50, 95])},
            "rb": {"min": round(float(g[inb, 2].min()), 3), **_pcts(g[inb, 2], [5])},
            "lat_accel": {
                "mean": round(float(np.nanmean(la[ic])), 3),
                **_pcts(la[ic], [95]),
                "max": round(float(np.nanmax(la[ic])), 3),
            },
            "jerk": {**_pcts(ja[ic], [95]), "max": round(float(np.nanmax(ja[ic])), 3)},
        }
        report["totals"][lab] = tot
        print(
            f"  {lab:>20} | RB cross {nx} | cl μ{tot['cl']['mean']} p95 {tot['cl']['p95']} "
            f"| rb min {tot['rb']['min']} p5 {tot['rb']['p5']} "
            f"| latA mean {tot['lat_accel']['mean']} p95 {tot['lat_accel']['p95']} "
            f"| jerk p95 {tot['jerk']['p95']}"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
