"""Open-loop per-arc centerline comparison across models (fast psim proxy).

For an all-arc validation set (scenes mined across the WHOLE route, both directions),
run each model's deterministic trajectory, score centerline with reward.py
(compute_centerline_score_batch, the canonical metric), and bin by route arc
(project_to_polyline, the same arc projection psim_per_arc_metrics uses). Prints a
per-arc table per model so you can see WHERE a model is off-center vs baseline/seed —
without waiting for a full psim. Routes are auto-selected per scene by 'm2t'/'t2m' in
the filename.

Reuses ONLY: eval_det_avoidance.{load_model,load_npz_data,det_inference_batched},
reward.compute_centerline_score_batch, _heatmap_common.{build_route_polyline,
project_to_polyline,recover_ego_world_pose_from_goal}. No new metric/geometry code.
"""

import argparse
import json
import os

import numpy as np
import torch

from rlvr.autoresearch.tools.eval_det_avoidance import (
    det_inference_batched,
    load_model,
    load_npz_data,
)
from rlvr.reward import compute_centerline_score_batch
from scenario_generation.route import Route
from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    project_to_polyline,
    recover_ego_world_pose_from_goal,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="all-arc val scene list (m2t+t2m)")
    ap.add_argument("--m2t_route", required=True)
    ap.add_argument("--t2m_route", required=True)
    ap.add_argument("--ego_shape", required=True)
    ap.add_argument("--bin_m", type=float, default=100.0)
    ap.add_argument("--models", nargs="+", required=True, help="LABEL PATH LABEL PATH ...")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = json.load(open(args.scenes))
    routes = {}
    polys = {}
    for k, rp in [("m2t", args.m2t_route), ("t2m", args.t2m_route)]:
        routes[k] = Route.load(rp)
        polys[k] = build_route_polyline(routes[k])

    # precompute per-scene arc + route key (model-independent)
    meta = []
    for p in paths:
        rk = "t2m" if "t2m" in os.path.basename(p) else "m2t"
        d = np.load(p, allow_pickle=True)
        ex, ey, eyaw = recover_ego_world_pose_from_goal(np.asarray(d["goal_pose"]), routes[rk])
        pts, s = polys[rk]
        arc = project_to_polyline(np.array([ex, ey]), pts, s)[0]
        meta.append((p, rk, arc))

    labels = args.models[0::2]
    mpaths = args.models[1::2]
    per_model = {}
    for lab, mp in zip(labels, mpaths):
        model, margs = load_model(mp, dev)
        cls = []
        for p, rk, arc in meta:
            d = load_npz_data(p, dev)
            traj = det_inference_batched(model, margs, [d], dev)  # (1,T,4)
            es = d["ego_shape"]
            es_one = es[0] if es.dim() > 1 else es
            cl = float(compute_centerline_score_batch(traj, es_one, d, usage_mode="baselink")[0])
            cls.append(cl)
        per_model[lab] = np.array(cls)
        del model
        torch.cuda.empty_cache()

    arcs = np.array([m[2] for m in meta])
    maxarc = arcs.max()
    nb = int(maxarc // args.bin_m) + 1
    print(
        f"\nOpen-loop per-arc centerline (reward.py compute_centerline_score, baselink; closer to 0 = centered)"
    )
    print(f"scenes={len(paths)} | {' | '.join(labels)}")
    hdr = "  arc-bin  | " + " | ".join(f"{l:>8}" for l in labels)
    print(hdr)
    print("-" * len(hdr))
    for b in range(nb):
        lo, hi = b * args.bin_m, (b + 1) * args.bin_m
        mask = (arcs >= lo) & (arcs < hi)
        if mask.sum() == 0:
            continue
        cells = " | ".join(f"{per_model[l][mask].mean():>8.3f}" for l in labels)
        print(f" {int(lo):4d}-{int(hi):<4d} | {cells}   (n={int(mask.sum())})")
    print("-" * len(hdr))
    print(" OVERALL  | " + " | ".join(f"{per_model[l].mean():>8.3f}" for l in labels))


if __name__ == "__main__":
    main()
