#!/usr/bin/env python3
"""Open-loop PLAN comfort — a per-epoch-cheap gentleness metric.

Deterministic inference only (no psim, no ROS) → fast enough to run every epoch
like the avoidance/L2 evals. Measures the comfort of the model's PREDICTED [80,4]
trajectory (x, y, cos_h, sin_h) on val scenes, at the KNOWN plan timestep:

    speed[t]      = ||(x,y)[t+1]-(x,y)[t]|| / dt          (1st diff, known dt)
    yaw_rate[t]   = wrap(heading[t+1]-heading[t]) / dt     (heading = atan2(sin,cos))
    lat_accel     = |yaw_rate * speed|                      (centripetal)
    jerk          = |d(lat_accel)/dt|

dt is the plan step (RewardConfig.dt = 0.1 s) — NOT assumed from a bag rate, and the
plan is a smooth denoised trajectory, so these low-order derivatives are clean (unlike
3rd-derivatives of noisy realized localization — see psim_comfort_heatmap's history).

This is an OPEN-LOOP proxy: the realized closed-loop drive is rougher than the single
plan, so treat it as a relative-improvement signal + optimistic lower bound. The
closed-loop ground truth is psim_comfort_heatmap (post-hoc, ×3 runs). But this metric
runs per-epoch and WOULD catch a comfort regression at training time — the blind spot
the standard geometric metrics miss.

Usage:
    python -m rlvr.autoresearch.tools.eval_plan_comfort \
        --model_path <merged.pth> --scenes <scenes.json> --output_dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model, det_inference_batched


def _dist(vals):
    a = np.asarray([v for v in vals if v == v], dtype=np.float64)
    if a.size == 0:
        return {k: None for k in ("mean", "p5", "p25", "p50", "p75", "p95", "max", "n")}
    q = lambda p: float(np.percentile(a, p))
    return {"mean": float(a.mean()), "p5": q(5), "p25": q(25), "p50": q(50),
            "p75": q(75), "p95": q(95), "max": float(a.max()), "n": int(a.size)}


def plan_comfort(traj_T4: np.ndarray, dt: float) -> tuple[float, float]:
    """Per-trajectory (lat_accel p95, jerk p95) from a [T,4] plan (x,y,cos,sin)."""
    x, y = traj_T4[:, 0], traj_T4[:, 1]
    cos, sin = traj_T4[:, 2], traj_T4[:, 3]
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    speed = np.sqrt(vx ** 2 + vy ** 2)                       # [T-1]
    heading = np.arctan2(sin, cos)
    dyaw = np.diff(heading)
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))            # wrap
    yaw_rate = np.abs(dyaw) / dt                             # [T-1]
    lat_accel = np.abs(yaw_rate * speed)                    # [T-1]
    jerk = np.abs(np.diff(lat_accel) / dt)                  # [T-2]
    return (float(np.percentile(lat_accel, 95)),
            float(np.percentile(jerk, 95)) if jerk.size else float("nan"))


def eval_plan_comfort(model, model_args, scene_paths, device, dt=0.1, batch_size=32):
    la95, jk95 = [], []
    for start in range(0, len(scene_paths), batch_size):
        datas, valid = [], []
        for p in scene_paths[start:start + batch_size]:
            try:
                datas.append(load_npz_data(p, device)); valid.append(p)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {Path(p).name}: {e}")
        if not datas:
            continue
        trajs = det_inference_batched(model, model_args, datas, device).cpu().numpy()
        for bi in range(len(valid)):
            a, j = plan_comfort(trajs[bi], dt)
            la95.append(a); jk95.append(j)
    return la95, jk95


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dt", type=float, default=0.1, help="plan timestep (RewardConfig.dt = 0.1s)")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    scenes = json.load(open(args.scenes))
    la95, jk95 = eval_plan_comfort(model, model_args, scenes, device, args.dt, args.batch_size)

    # headline = the distribution of per-plan p95 lat-accel / jerk across scenes
    out = {"model": args.model_path, "scenes": args.scenes, "n": len(la95), "dt": args.dt,
           "plan_lat_accel_p95": _dist(la95), "plan_jerk_p95": _dist(jk95)}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    jp = Path(args.output_dir) / "plan_comfort.json"
    jp.write_text(json.dumps(out, indent=2))
    la, jk = out["plan_lat_accel_p95"], out["plan_jerk_p95"]
    print(f"PLAN comfort ({len(la95)} scenes, dt={args.dt}): "
          f"lat_accel p95 mean={la['mean']:.2f} p50={la['p50']:.2f} p95={la['p95']:.2f} max={la['max']:.2f} m/s² | "
          f"jerk p95 mean={jk['mean']:.2f} p50={jk['p50']:.2f} p95={jk['p95']:.2f} max={jk['max']:.2f} m/s³")
    print(f"  wrote {jp}")


if __name__ == "__main__":
    main()
