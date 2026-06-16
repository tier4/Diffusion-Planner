"""Mine parse_rosbag NPZs in a route arc band and convert reduced→trainable format.

For HEAL/MEND: take psim-bag-parsed NPZs (reduced: ego/nbr future 3-col, 32 neighbor
slots, line_strings 2-col, polygons 2-col), keep only frames whose ego pose projects
onto the route within [arc_lo, arc_hi], and convert each to the full trainable format:
- ego_agent_future / neighbor_agents_future: 3-col (x,y,heading) -> 4-col (x,y,cos,sin)
- neighbor_agents_past (32,31,11) -> (320,31,11); neighbor_agents_future (32,80,3)->(320,80,4)
- line_strings (60,20,2) -> (60,20,4): col2=0, col3=valid (1 where xy nonzero) [campaign approx;
  loss.py treats col3>0.5 as the road-border mask]
- polygons (10,40,2) -> (10,40,3): col2=presence (1 where xy nonzero)
- the rebuilt trajectory/map arrays are cast float32 (other passthrough fields keep their dtype)

Reuses scenario_generation.tools._heatmap_common for route projection (no hand-rolled geometry).
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

from scenario_generation.tools._heatmap_common import build_route_polyline, project_to_polyline


def _f2to4(arr):  # (...,3)[x,y,h] -> (...,4)[x,y,cos,sin]
    x, y, h = arr[..., 0], arr[..., 1], arr[..., 2]
    # padding rows are (0,0,0); the 0.1 m floor (vs the 1e-6 used for map
    # arrays) matches convert_3col_to_4col so near-origin trajectory points
    # get a zero heading instead of a spurious cos/sin.
    v = (np.abs(x) + np.abs(y)) > 0.1
    cos = np.where(v, np.cos(h), 0.0)
    sin = np.where(v, np.sin(h), 0.0)
    return np.stack([x, y, cos, sin], axis=-1).astype(np.float32)


def convert(d):
    out = {k: d[k] for k in d.files}
    out["ego_agent_future"] = _f2to4(d["ego_agent_future"].astype(np.float32))
    out["ego_agent_past"] = d["ego_agent_past"].astype(np.float32)
    # neighbors: pad slots to 320
    npf = d["neighbor_agents_future"].astype(np.float32)  # (32,80,3)
    npp = d["neighbor_agents_past"].astype(np.float32)  # (32,31,11)
    NF = np.zeros((320, npf.shape[1], 4), np.float32)
    NF[: npf.shape[0]] = _f2to4(npf)
    NP = np.zeros((320, npp.shape[1], 11), np.float32)
    NP[: npp.shape[0]] = npp
    out["neighbor_agents_future"] = NF
    out["neighbor_agents_past"] = NP
    # line_strings (60,20,2)->(60,20,4): col2=0, col3=valid
    ls = d["line_strings"].astype(np.float32)
    LS = np.zeros((ls.shape[0], ls.shape[1], 4), np.float32)
    LS[..., :2] = ls[..., :2]
    LS[..., 3] = ((np.abs(ls[..., 0]) + np.abs(ls[..., 1])) > 1e-6).astype(np.float32)
    out["line_strings"] = LS
    # polygons (10,40,2)->(10,40,3): col2=presence
    pg = d["polygons"].astype(np.float32)
    PG = np.zeros((pg.shape[0], pg.shape[1], 3), np.float32)
    PG[..., :2] = pg[..., :2]
    PG[..., 2] = ((np.abs(pg[..., 0]) + np.abs(pg[..., 1])) > 1e-6).astype(np.float32)
    out["polygons"] = PG
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--route_pkl", required=True)
    ap.add_argument("--arc_lo", type=float, required=True)
    ap.add_argument("--arc_hi", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    args = ap.parse_args()
    with open(args.route_pkl, "rb") as f:
        route = pickle.load(f)
    poly, arc = build_route_polyline(route)
    os.makedirs(args.out_dir, exist_ok=True)
    kept = []
    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    for p in files:
        with np.load(p, allow_pickle=True) as d:
            ex, ey = float(d["ego_current_state"][0]), float(d["ego_current_state"][1])
            res = project_to_polyline(np.array([ex, ey], dtype=float), poly, arc)
            s = float(res[0])
            if not (args.arc_lo <= s <= args.arc_hi):
                continue
            out = convert(d)
        op = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(op, **out)
        kept.append(op)
    with open(args.out_list, "w") as f:
        json.dump(kept, f, indent=2)
    print(
        f"mined {len(kept)}/{len(files)} frames in arc [{args.arc_lo},{args.arc_hi}]m -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
