"""Per-epoch learning curve for a HEAL run on arc-targeted scenes.

For each saved lora_epoch_NNN in a run dir, load it onto the warmstart base,
run deterministic inference on the arc scenes, and report the per-scene L2
between the model det trajectory and:
  - TARGET  = the scene's ego_agent_future (for GRAFT-CL this is the baseline-det
              centerline target; falling L2 => model is learning the centerline)
  - START   = the warmstart (ep8) det trajectory (rising L2 => moving off the wound)

Reports full distribution (mean / p5 / p25 / p50 / p75 / p95 / max) per epoch,
plus per-arc-bin mean L2-to-target so you can see WHERE on the band it converges.
Detects learning-vs-stuck by the epoch-over-epoch trend.

No hand-rolled geometry; reuses eval_det_avoidance + _heatmap_common.
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model
from scenario_generation.tools._heatmap_common import (
    build_route_polyline,
    project_to_polyline,
    recover_ego_world_pose_from_goal,
)


def _l2(a, b):
    return float(np.sqrt(((a[:, :2] - b[:, :2]) ** 2).sum(-1)).mean())


def _dist(vals):
    v = np.array(vals)
    return dict(mean=v.mean(), p5=np.percentile(v, 5), p25=np.percentile(v, 25),
                p50=np.percentile(v, 50), p75=np.percentile(v, 75),
                p95=np.percentile(v, 95), max=v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_model", required=True, help="warmstart base (ep8) .pth")
    ap.add_argument("--scenes", required=True, help="arc scenes whose ego_agent_future is the TARGET")
    ap.add_argument("--route_pkl", default=None, help="if set, also report per-arc-bin L2-to-target")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = json.load(open(args.scenes))
    datas = [load_npz_data(p, dev) for p in scenes]
    targets = [np.load(p, allow_pickle=True)["ego_agent_future"] for p in scenes]

    arcs = None
    if args.route_pkl:
        route = pickle.load(open(args.route_pkl, "rb"))
        pts, s = build_route_polyline(route)
        arcs = []
        for p in scenes:
            gp = np.asarray(np.load(p, allow_pickle=True)["goal_pose"])
            ex, ey, _ = recover_ego_world_pose_from_goal(gp, route)
            arcs.append(float(project_to_polyline(np.array([ex, ey], float), pts, s)[0]))
        arcs = np.array(arcs)

    def det_all(model, model_args):
        out = []
        for i in range(0, len(datas), args.batch_size):
            b = datas[i:i + args.batch_size]
            out.extend(det_inference_batched(model, model_args, b, dev).cpu().numpy())
        return out

    # warmstart (ep8) det = START reference
    base, base_args = load_model(args.base_model, dev)
    start_det = det_all(base, base_args)
    del base
    torch.cuda.empty_cache()

    epochs = sorted(glob.glob(os.path.join(args.run_dir, "lora_epoch_*")))
    print(f"run: {os.path.basename(args.run_dir)}  | {len(scenes)} arc scenes | {len(epochs)} epochs")
    print(f"{'ep':>3} | {'L2->target  mean   p50   p95   max':<38} | {'L2->start mean':>14} | trend")
    prev = None
    for ed in epochs:
        ep = int(ed.split("_")[-1])
        model, model_args = load_model(args.base_model, dev)
        model = load_lora_checkpoint(model, ed)
        model.eval()
        det = det_all(model, model_args)
        del model
        torch.cuda.empty_cache()
        to_tgt = [_l2(det[i], targets[i]) for i in range(len(scenes))]
        to_start = [_l2(det[i], start_det[i]) for i in range(len(scenes))]
        d = _dist(to_tgt)
        trend = "" if prev is None else (f"↓{prev - d['mean']:+.3f}" if d["mean"] < prev else f"↑{d['mean'] - prev:+.3f} (stuck/diverge?)")
        prev = d["mean"]
        print(f"{ep:3d} | {d['mean']:6.3f} {d['p50']:6.3f} {d['p95']:6.3f} {d['max']:6.3f}            | "
              f"{np.mean(to_start):14.3f} | {trend}")
        if arcs is not None:
            line = "      per-arc L2->target: "
            for lo in range(900, 1850, 150):
                m = (arcs >= lo) & (arcs < lo + 150)
                if m.sum():
                    line += f"[{lo}-{lo+150}]{np.mean(np.array(to_tgt)[m]):.2f} "
            print(line)


if __name__ == "__main__":
    main()
