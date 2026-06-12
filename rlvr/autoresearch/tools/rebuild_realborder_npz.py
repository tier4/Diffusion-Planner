"""Rewrite a scene NPZ's `line_strings` with the REAL map borders (ego frame).

The psim-converted NPZs carry `line_strings` that conflate lane lines with curbs,
so reward RB scoring and road_border guidance on them are invalid (rb_min≈0 every-
where). This tool recovers each scene's ego WORLD pose (goal_pose + route), rebuilds
the canonical stop_line+road_border tensor from the map at that pose
(`LaneletSceneBuilder.build_line_strings_tensor`, world frame, ch2=stop_line,
ch3=road_border), transforms the point xy into the scene's EGO frame, and writes a
new NPZ with the replaced `line_strings` (+ injected `ego_shape`). All other fields
are copied verbatim. Reuses existing geometry only — no hand-rolled border math.

After this, viz_p4_recovery / compute_reward_batch RB scoring AND road_border
guidance both operate on the true curb.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.tools._heatmap_common import load_route, recover_ego_world_pose_from_goal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", required=True, help="JSON list of source NPZ paths")
    ap.add_argument("--route", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    args = ap.parse_args()

    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")], dtype=np.float32)
    route = load_route(Path(args.route))
    b = LaneletSceneBuilder(str(route.map_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = json.load(open(args.scenes))
    written = []
    for sp in scenes:
        with np.load(sp, allow_pickle=True) as _z:  # close fd promptly on large scene lists
            d = dict(_z)
        ex, ey, eyaw = recover_ego_world_pose_from_goal(np.asarray(d["goal_pose"]), route)
        ls_w = np.asarray(
            b.build_line_strings_tensor(np.array([ex, ey], dtype=float)), dtype=np.float32
        )  # (60,20,4) world: x,y,stop_line,road_border
        c, s = math.cos(eyaw), math.sin(eyaw)
        ls = ls_w.copy()
        valid = (ls_w[..., 2] > 0.5) | (ls_w[..., 3] > 0.5)  # type-flagged points only
        dx = ls_w[..., 0] - ex
        dy = ls_w[..., 1] - ey
        xe = c * dx + s * dy
        ye = -s * dx + c * dy
        ls[..., 0] = np.where(valid, xe, 0.0)
        ls[..., 1] = np.where(valid, ye, 0.0)
        d["line_strings"] = ls.astype(np.float32)
        d["ego_shape"] = ego_shape
        op = out_dir / Path(sp).name
        np.savez(op, **d)
        written.append(str(op))
    json.dump(written, open(args.out_list, "w"), indent=2)
    print(f"rebuilt real-border line_strings for {len(written)} scenes -> {args.out_dir}")
    print(f"  scene list -> {args.out_list}")


if __name__ == "__main__":
    main()
