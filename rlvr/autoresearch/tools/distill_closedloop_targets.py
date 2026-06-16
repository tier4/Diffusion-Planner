#!/usr/bin/env python3
"""Distill v23 CLOSED-LOOP reactive trajectories into curated-RSFT targets.

distill_guided_targets bakes v23's ONE-SHOT guided plan (open-loop ~4/50 on
val50) — but v23's real strength is CLOSED-LOOP (0/50): it re-plans every step
for 8 s and the realized motion clears obstacles a single plan cannot. This
tool captures THAT: it runs the 80-step closed-loop rollout under v23 guidance
(the exact rollout eval_closedloop_avoidance scores) and writes the REALIZED
driven trajectory (positions[1:], initial ego frame) into ego_agent_future.

A curated-RSFT LoRA trained on these learns to emit, in ONE plan, the reactive
path v23 took 8 s of re-planning to produce — transferring the closed-loop
avoidance into a deployable single-shot planner.

Screens (loud in summary):
  - only scenes whose REALIZED rollout cleared with min OBB clearance >
    --min_clearance are written (contact/stalled rollouts skipped);
  - inert scenes (rollout barely deviates from baseline, realized arc within
    --inert_arc m of the baseline det arc) are skipped unless --keep_inert.

Usage:
    python -m rlvr.autoresearch.tools.distill_closedloop_targets \
        --model_path <base.pth> --policy_dir <v23_dir> --scenes <list.json> \
        --out_dir <dir> --out_list <json> --ego_shape WB,L,W \
        [--steps 80] [--min_clearance 0.2] \
        [--lambda_lat 5.0 --lat_scale 2.0 --col_scale 9.0 --col_range 8.0]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_closedloop_avoidance import (
    make_guided_predict,
    score_rollout,
)
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy
from rlvr.autoresearch.tools.ghost_sim_common import extract_stopped_neighbors
from rlvr.autoresearch.tools.recovery_sim import (
    closed_loop_rollout_with_plans,
    deterministic_predict,
)
from rlvr.grpo_sft_trainer import _smooth_trajectory


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--policy_dir", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_list", required=True)
    p.add_argument("--ego_shape", required=True, help="WB,L,W — no default")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--min_clearance", type=float, default=0.2)
    p.add_argument(
        "--inert_arc",
        type=float,
        default=1.0,
        help="skip if realized arc within this many m of baseline det arc",
    )
    p.add_argument("--keep_inert", action="store_true")
    # v23 certified guidance envelope (defaults = the strong cert envelope,
    # matching policy_eval.json guidance_args — NOT the weak argparse defaults
    # of eval_policy_avoidance).
    p.add_argument("--lambda_lat", type=float, default=5.0)
    p.add_argument("--lat_scale", type=float, default=2.0)
    p.add_argument("--col_scale", type=float, default=9.0)
    p.add_argument("--col_range", type=float, default=8.0)
    p.add_argument("--lambda_spd", type=float, default=0.2)
    p.add_argument("--stretch_scale", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=0.5)
    p.add_argument("--envelope", choices=["v1", "v2"], default="v1")
    p.add_argument("--lambda_col", type=float, default=3.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, margs, device)
    ego_shape = tuple(float(x) for x in args.ego_shape.split(","))
    guided_predict = make_guided_predict(policy, heads, args, device)

    with open(args.scenes) as f:
        paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, manifest = [], []
    n_inert = n_unsafe = n_err = 0
    for sp in paths:
        try:
            data = load_npz_data(sp, device)
            boxes = extract_stopped_neighbors(sp)
            # baseline det arc (inertness reference)
            det = deterministic_predict(model, margs, data)
            det_arc = float(np.linalg.norm(np.diff(det[:, :2], axis=0), axis=1).sum())
            roll = closed_loop_rollout_with_plans(
                model,
                margs,
                data,
                n_steps=args.steps,
                advance_k=0,
                predict_fn=guided_predict,
                sg_smooth=True,
                trim_backward=True,
            )
            sc = score_rollout(roll, boxes, ego_shape, device)
            if sc["contact"] or sc["stalled"]:
                n_unsafe += 1
                continue
            if boxes and sc["min_clear"] < args.min_clearance:
                n_unsafe += 1
                continue
            if (
                not args.keep_inert
                and abs(sc["arc"] - det_arc) < args.inert_arc
                and (not boxes or sc["min_clear"] > 2.0)
            ):
                # barely deviates from baseline AND already-clear scene → nothing learned
                n_inert += 1
                continue
        except Exception as e:  # noqa: BLE001
            print(f"  [err ] {Path(sp).name}: {e}")
            n_err += 1
            continue

        pos = np.asarray(roll["positions"])  # [steps+1, 3] initial frame
        realized = pos[1 : args.steps + 1]  # drop t0 origin
        traj4 = np.stack(
            [realized[:, 0], realized[:, 1], np.cos(realized[:, 2]), np.sin(realized[:, 2])],
            axis=-1,
        )
        traj4 = _smooth_trajectory(traj4.astype(np.float32), 11, 3)

        raw = dict(np.load(sp, allow_pickle=True))
        fut = raw["ego_agent_future"]
        T = min(fut.shape[0], traj4.shape[0])
        if fut.shape[-1] == 3:
            new = np.stack(
                [traj4[:T, 0], traj4[:T, 1], np.arctan2(traj4[:T, 3], traj4[:T, 2])], axis=-1
            )
        elif fut.shape[-1] == 4:
            new = traj4[:T, :4]
        else:
            raise ValueError(f"{sp}: ego_agent_future width {fut.shape[-1]} not 3/4")
        raw["ego_agent_future"] = new.astype(fut.dtype)

        pool = Path(sp).parent.name
        out_path = out_dir / f"{pool}__{Path(sp).stem}_cldistill.npz"
        np.savez(out_path, **raw)
        written.append(str(out_path))
        manifest.append(
            {
                "source": sp,
                "min_clear": sc["min_clear"],
                "arc": sc["arc"],
                "cleared": sc["cleared"],
                "out": str(out_path),
            }
        )

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"\nClosed-loop distilled {len(written)} targets "
        f"(skipped: {n_inert} inert, {n_unsafe} unsafe/stalled, {n_err} err) -> {args.out_list}"
    )


if __name__ == "__main__":
    main()
