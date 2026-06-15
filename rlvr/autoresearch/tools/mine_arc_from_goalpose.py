"""Filter already-converted psim NPZs to a route arc band via goal_pose world-pose recovery.

The psim-dumped NPZs are ego-centric (ego_current_state = origin), so arc cannot be
read from ego pose directly. Instead recover the ego WORLD pose from each NPZ's
goal_pose + the Route goal (recover_ego_world_pose_from_goal), project it onto the
route polyline (project_to_polyline), and keep frames whose arc s falls in
[arc_lo, arc_hi]. Reuses scenario_generation.tools._heatmap_common — no hand-rolled geometry.

Input NPZs must already be in trainable format (this only selects, does not convert).
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    project_to_polyline,
    recover_ego_world_pose_from_goal,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--route_pkl", required=True)
    ap.add_argument("--arc_lo", type=float, required=True)
    ap.add_argument("--arc_hi", type=float, required=True)
    ap.add_argument("--out_list", required=True)
    args = ap.parse_args()

    with open(args.route_pkl, "rb") as f:
        route = pickle.load(f)
    pts, s = build_route_polyline(route)

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    kept, arcs = [], []
    for p in files:
        d = np.load(p, allow_pickle=True)
        if "goal_pose" not in d.files:
            raise SystemExit(f"{p} has no goal_pose — cannot recover world arc")
        ex, ey, _ = recover_ego_world_pose_from_goal(np.asarray(d["goal_pose"]), route)
        res = project_to_polyline(np.array([ex, ey], dtype=float), pts, s)
        arc = float(res[0])
        if args.arc_lo <= arc <= args.arc_hi:
            kept.append(p)
            arcs.append(arc)

    with open(args.out_list, "w") as f:
        json.dump(kept, f, indent=2)
    if arcs:
        a = np.array(arcs)
        print(
            f"kept {len(kept)}/{len(files)} in arc [{args.arc_lo},{args.arc_hi}]m "
            f"-> {args.out_list}  (arc min={a.min():.0f} p50={np.median(a):.0f} max={a.max():.0f})"
        )
    else:
        print(f"kept 0/{len(files)} in arc [{args.arc_lo},{args.arc_hi}]m — check route/band")


if __name__ == "__main__":
    main()
