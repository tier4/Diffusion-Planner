#!/usr/bin/env python3
"""Batch closed-loop avoidance eval: does the ego CLEAR the obstacle?

For every scene, runs the 80-step (8 s) closed-loop rollout (per-step
replanning, ego advances to each fresh plan's first point) for the plain
baseline and for baseline+explorer (pure, no gates/rules), then scores the
REALIZED motion:

  cleared   : traveled arc > t0 distance-to-obstacle + obstacle length + 2 m
              AND min realized OBB clearance > 0 (no contact)
  contact   : min realized OBB clearance <= 0 at any step
  stalled   : mean speed over the last 2 s < 0.5 m/s and not cleared
  min_clear : min OBB clearance of realized poses to static neighbors
              (canonical compute_static_collision_penalty)

Usage:
    python -m rlvr.autoresearch.tools.eval_closedloop_avoidance \
        --model_path <base.pth> --policy_dir <dir> --scenes <json> \
        --output <json> [--steps 80] \
        [--lambda_lat 5.0] [--lat_scale 2.0] [--col_scale 9.0]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401
from exploration_policy.utils import run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer
from rlvr.autoresearch.tools.ghost_sim_common import extract_stopped_neighbors
from rlvr.autoresearch.tools.recovery_sim import (
    closed_loop_rollout_with_plans,
    deterministic_predict,
)
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data
from scenario_generation.explorer_runner import plan_static_clearance


def make_guided_predict(policy, heads, args, device):
    def predict(model, model_args, data):
        det = deterministic_predict(model, model_args, data)
        batch = _stack_scene_data([data], device)
        norm = _normalize_batch(batch, model_args)
        x_ref = torch.from_numpy(np.ascontiguousarray(det)).float()
        x_ref = x_ref.unsqueeze(0).to(device)
        norm["reference_trajectory"] = x_ref
        enc = run_frozen_encoder(model, norm)
        out = policy(enc, x_ref, deterministic=True)
        etas = {h: (2.0 * out.dists[h].mean - 1.0).reshape(1) for h in heads}
        return _batched_generate_varied_noise(
            model, model_args, norm, noise_min=0.0, noise_max=0.0,
            first_deterministic=False, composer=make_composer(etas, args),
            device=device,
        )[0].cpu().numpy()
    return predict


def score_rollout(rollout, boxes, ego_shape, device, dt=0.1):
    pos = np.asarray(rollout["positions"])          # [N+1, 3] initial frame
    vel = np.asarray(rollout["velocities"])
    arc = float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum())

    # realized trajectory as (T,4) for the canonical clearance fn
    traj = np.stack([pos[1:, 0], pos[1:, 1],
                     np.cos(pos[1:, 2]), np.sin(pos[1:, 2])], axis=-1)
    min_clear = plan_static_clearance(traj.astype(np.float32), boxes,
                                      ego_shape, device)

    if boxes:
        d0 = min(math.hypot(b[0], b[1]) for b in boxes)
        nearest = min(boxes, key=lambda b: math.hypot(b[0], b[1]))
        need = d0 + nearest[3] + 2.0
    else:
        d0, need = 0.0, 0.0
    cleared = bool(arc > need and min_clear > 0.0)
    tail_speed = float(vel[-20:].mean())
    contact = bool(min_clear <= 0.0)
    stalled = bool(tail_speed < 0.5 and not cleared)
    return {"arc": round(arc, 2), "min_clear": round(min_clear, 3),
            "cleared": cleared, "contact": contact, "stalled": stalled,
            "tail_speed": round(tail_speed, 2)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--trim_backward", action="store_true",
                        help="trim leading behind-ego plan points before "
                             "tracking (see recovery_sim rollout docstring)")
    parser.add_argument("--ego_shape", required=True,
                        help="WB,L,W — no default, must match the platform")
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, margs, device)
    guided_predict = make_guided_predict(policy, heads, args, device)
    ego_shape = tuple(float(x) for x in args.ego_shape.split(","))

    with open(args.scenes) as f:
        paths = json.load(f)

    rows = []
    for sp in paths:
        try:
            data = load_npz_data(sp, device)
            boxes = extract_stopped_neighbors(sp)
            r_base = closed_loop_rollout_with_plans(
                model, margs, data, n_steps=args.steps,
                trim_backward=args.trim_backward)
            r_gui = closed_loop_rollout_with_plans(
                model, margs, data, n_steps=args.steps,
                predict_fn=guided_predict,
                trim_backward=args.trim_backward)
        except Exception as e:  # noqa: BLE001
            print(f"  [err ] {Path(sp).name}: {e}")
            continue
        row = {
            "scene": Path(sp).name,
            "baseline": score_rollout(r_base, boxes, ego_shape, device),
            "explorer": score_rollout(r_gui, boxes, ego_shape, device),
        }
        rows.append(row)
        b, g = row["baseline"], row["explorer"]
        print(f"  {row['scene']:26s} base[clr={b['min_clear']:+.2f} "
              f"{'CLEAR' if b['cleared'] else 'CONTACT' if b['contact'] else 'stall' if b['stalled'] else 'short'}] "
              f"expl[clr={g['min_clear']:+.2f} "
              f"{'CLEAR' if g['cleared'] else 'CONTACT' if g['contact'] else 'stall' if g['stalled'] else 'short'}]")

    def agg(key):
        sub = [r[key] for r in rows]
        return {
            "cleared": sum(s["cleared"] for s in sub),
            "contact": sum(s["contact"] for s in sub),
            "stalled": sum(s["stalled"] for s in sub),
            "min_clear_p5": round(float(np.percentile(
                [s["min_clear"] for s in sub], 5)), 3),
        }
    report = {"n": len(rows), "baseline": agg("baseline"),
              "explorer": agg("explorer"), "scenes": rows}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("n", "baseline", "explorer")},
                     indent=1))


if __name__ == "__main__":
    main()
