#!/usr/bin/env python3
"""Generate mid-maneuver training scenes from CLOSED-LOOP realized states.

Same idea as rollforward_avoidance_scenes (re-anchor the scene at a later
ego state with the executed prefix as history), but the states come from the
closed-loop rollout of model+explorer (batched_closed_loop) instead of the
open-loop plan — i.e. the states a closed-loop run ACTUALLY visits,
including the slow conflicted ones the open-loop roll never reaches.

Per scene: roll k in --steps, PLUS every step where realized speed drops
below --slow_thresh (the OOD stall states; deduped, capped by --max_slow).

Output filenames are prefixed with the source pool dir to avoid the
same-basename collision across perturbation pools.

Usage:
    python -m rlvr.autoresearch.tools.rollforward_closedloop_scenes \
        --scenes <list.json> --model_path <base.pth> --policy_dir <dir> \
        --steps 10,20,30,40,50,60 --slow_thresh 1.5 --max_slow 4 \
        --out_dir <dir> --out_list <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.batched_closedloop_videos import batched_closed_loop
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy
from rlvr.autoresearch.tools.rollforward_avoidance_scenes import rollforward_scene


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--steps", default="10,20,30,40,50,60")
    parser.add_argument(
        "--slow_thresh",
        type=float,
        default=1.5,
        help="also roll at steps where realized speed < this",
    )
    parser.add_argument(
        "--max_slow", type=int, default=4, help="max extra slow-state rolls per scene"
    )
    parser.add_argument("--n_steps", type=int, default=80)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument(
        "--envelope",
        choices=["v1", "v2"],
        default="v1",
        help="guidance envelope — must match the policy's training labels",
    )
    parser.add_argument("--lambda_col", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, margs, device)
    base_steps = [int(s) for s in args.steps.split(",")]

    with open(args.scenes) as f:
        paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cl-roll] closed-loop rollouts: {len(paths)} scenes")
    datas = [load_npz_data(p, device) for p in paths]
    rollouts, eta_logs = batched_closed_loop(
        model,
        margs,
        datas,
        device,
        policy=policy,
        heads=heads,
        gargs=args,
        n_steps=args.n_steps,
        chunk=args.chunk,
    )
    del datas
    torch.cuda.empty_cache()

    written, manifest = [], []
    for i, sp in enumerate(paths):
        pos = np.asarray(rollouts[i]["positions"])  # [N+1, 3] initial frame
        vel = np.asarray(rollouts[i]["velocities"])
        # realized trajectory as (T, 4) — same shape rollforward_scene expects
        realized = np.stack(
            [pos[1:, 0], pos[1:, 1], np.cos(pos[1:, 2]), np.sin(pos[1:, 2])], axis=-1
        )

        ks = [k for k in base_steps if k < realized.shape[0]]
        slow = [
            int(k) for k in np.nonzero(vel[1:] < args.slow_thresh)[0] if k >= 5 and k not in ks
        ][: args.max_slow]
        ks = sorted(set(ks + slow))

        try:
            raw = dict(np.load(sp, allow_pickle=True))
        except Exception as e:  # noqa: BLE001
            print(f"  [err ] {Path(sp).name}: {e}")
            continue
        pool = Path(sp).parent.name
        stem = Path(sp).stem
        for k in ks:
            try:
                rolled = rollforward_scene(raw, realized, k)
            except Exception as e:  # noqa: BLE001
                print(f"  [err ] {pool}/{stem} k={k}: {e}")
                continue
            out_path = out_dir / f"{pool}__{stem}_clroll{k:02d}.npz"
            np.savez(out_path, **rolled)
            written.append(str(out_path))
            manifest.append(
                {
                    "source": sp,
                    "k": k,
                    "slow": k in slow,
                    "v_at_k": round(float(vel[k + 1]), 2),
                    "etas_at_k": eta_logs[i][k] if eta_logs[i] else None,
                    "out": str(out_path),
                }
            )
        print(f"  [ok  ] {pool}/{stem}: {len(ks)} rolled ({len(slow)} slow-state)")

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nWrote {len(written)} closed-loop rolled scenes -> {args.out_list}")


if __name__ == "__main__":
    main()
