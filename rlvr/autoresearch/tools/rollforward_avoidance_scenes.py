#!/usr/bin/env python3
"""Generate MID-MANEUVER training scenes by rolling the model+explorer forward.

For every (solved) avoidance scene: generate the explorer-guided trajectory
open-loop, then for each rollforward step k create a NEW scene whose ego sits
AT guided[k] with the executed guided prefix as its recorded history — the
states a closed-loop run actually visits. Sweeping labels on these states
teaches the policy to CONTINUE an avoidance in progress and to decay back to
zero after passing (recover/rejoin) — closing the t0-only distribution gap
that causes the closed-loop speed-decay stall.

Re-anchoring reuses the canonical perturbation transform
(disturb_and_replay._apply_inverse_rigid_to_spatial); ego_current_state is
reset to origin with body-frame speed taken from the guided trajectory's
local displacement at k.

Usage:
    python -m rlvr.autoresearch.tools.rollforward_avoidance_scenes \
        --scenes <solved_t0_scenes.json> --model_path <base.pth> \
        --policy_dir <dir> --steps 5,10,15,20,25,30 \
        --out_dir <dir> --out_list <json> \
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
from rlvr.autoresearch.tools.disturb_and_replay import _apply_inverse_rigid_to_spatial
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer
from rlvr.autoresearch.tools.recovery_sim import deterministic_predict
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data


@torch.no_grad()
def guided_trajectory(
    model,
    margs,
    policy,
    heads,
    data,
    args,
    device,
) -> tuple[np.ndarray, dict[str, float]]:
    det = deterministic_predict(model, margs, data)
    batch = _stack_scene_data([data], device)
    norm = _normalize_batch(batch, margs)
    x_ref = torch.from_numpy(np.ascontiguousarray(det)).float().unsqueeze(0).to(device)
    norm["reference_trajectory"] = x_ref
    enc = run_frozen_encoder(model, norm)
    out = policy(enc, x_ref, deterministic=True)
    etas = {h: (2.0 * out.dists[h].mean - 1.0).reshape(1) for h in heads}
    g = (
        _batched_generate_varied_noise(
            model,
            margs,
            norm,
            noise_min=0.0,
            noise_max=0.0,
            first_deterministic=False,
            composer=make_composer(etas, args, envelope=getattr(policy, "guidance_envelope", None)),
            device=device,
        )[0]
        .cpu()
        .numpy()
    )
    return g, {h: float(v.item()) for h, v in etas.items()}


def rollforward_scene(raw: dict, guided: np.ndarray, k: int, dt: float = 0.1) -> dict:
    """Build the scene at guided[k]: re-anchor everything, splice ego history."""
    out = {key: np.array(v, copy=True) for key, v in raw.items()}

    gx, gy = float(guided[k, 0]), float(guided[k, 1])
    gth = float(math.atan2(guided[k, 3], guided[k, 2]))

    # Ego history = original recorded past followed by the EXECUTED guided
    # prefix (still in the ORIGINAL ego frame; the rigid transform below
    # re-anchors it). ego_agent_past is (T, 3) [x, y, yaw].
    past = out["ego_agent_past"]
    g_yaw = np.arctan2(guided[: k + 1, 3], guided[: k + 1, 2])
    g_rows = np.stack([guided[: k + 1, 0], guided[: k + 1, 1], g_yaw], axis=-1)
    combined = np.concatenate([past, g_rows.astype(past.dtype)], axis=0)
    out["ego_agent_past"] = np.ascontiguousarray(combined[-past.shape[0] :])

    # Drop the GT future — these scenes are for label sweeps, which never use
    # it; keep shape with zeros so loaders stay happy.
    if "ego_agent_future" in out:
        out["ego_agent_future"] = np.zeros_like(out["ego_agent_future"])

    _apply_inverse_rigid_to_spatial(out, gx, gy, gth)

    # ego_current_state -> origin pose; body-frame speed from the guided
    # trajectory's local displacement at k.
    ecs = out["ego_current_state"].copy()
    ecs[0] = 0.0
    ecs[1] = 0.0
    ecs[2] = 1.0
    ecs[3] = 0.0
    if k + 1 < guided.shape[0]:
        spd = float(np.linalg.norm(guided[k + 1, :2] - guided[k, :2])) / dt
    else:
        spd = float(np.linalg.norm(guided[k, :2] - guided[k - 1, :2])) / dt
    if ecs.size >= 6:
        ecs[4] = spd
        ecs[5] = 0.0
    if ecs.size >= 8:
        ecs[6] = 0.0
        ecs[7] = 0.0
    out["ego_current_state"] = ecs
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--steps", default="5,10,15,20,25,30")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    parser.add_argument(
        "--lambda_lat",
        type=float,
        default=None,
        help="override the policy's persisted guidance envelope",
    )
    parser.add_argument("--lat_scale", type=float, default=None)
    parser.add_argument("--col_scale", type=float, default=None)
    parser.add_argument("--col_range", type=float, default=None)
    parser.add_argument("--lambda_spd", type=float, default=None)
    parser.add_argument("--stretch_scale", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--envelope",
        choices=["v1", "v2"],
        default=None,
        help="override the policy's persisted guidance envelope family (v1/v2)",
    )
    parser.add_argument("--lambda_col", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, margs, device)
    steps = [int(s) for s in args.steps.split(",")]

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, manifest = [], []
    for sp in scene_paths:
        try:
            data = load_npz_data(sp, device)
            guided, etas = guided_trajectory(model, margs, policy, heads, data, args, device)
            raw = dict(np.load(sp, allow_pickle=True))
        except Exception as e:  # noqa: BLE001
            print(f"  [err ] {Path(sp).name}: {e}")
            continue
        # Pool-prefixed stem: perturbation pools share basenames (e.g.
        # train_parallel/ and train_yaw/ both hold scene_0008_var02.npz),
        # which previously silently overwrote rolls across pools.
        stem = f"{Path(sp).parent.name}__{Path(sp).stem}"
        for k in steps:
            try:
                rolled = rollforward_scene(raw, guided, k)
            except Exception as e:  # noqa: BLE001
                print(f"  [err ] {stem} k={k}: {e}")
                continue
            out_path = out_dir / f"{stem}_roll{k:02d}.npz"
            np.savez(out_path, **rolled)
            written.append(str(out_path))
            manifest.append({"source": sp, "k": k, "t0_etas": etas, "out": str(out_path)})
        print(f"  [ok  ] {stem}: {len(steps)} rolled states (t0 etas {etas})")

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nWrote {len(written)} rolled scenes -> {args.out_list}")


if __name__ == "__main__":
    main()
