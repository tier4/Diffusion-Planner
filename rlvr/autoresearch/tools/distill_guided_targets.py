#!/usr/bin/env python3
"""Distill explorer-guided trajectories into curated-RSFT training NPZs.

For every scene: run the frozen planner + guidance policy (deterministic
etas, batched), Savitzky-Golay-filter the guided trajectory (the standard
pre-SFT smoothing), and write a copy of the NPZ whose ``ego_agent_future``
IS that guided trajectory — the exact format the curated ranked-SFT mode
(``ranked_sft_mode="curated"``) trains on. The explorer thereby becomes a
data-generation engine replacing hand-tuned Branch-Editor sessions.

Safety screens (fail-quiet per scene, loud in the summary):
  - only scenes where the GUIDED plan is collision-free vs stopped
    neighbours (canonical clearance > --min_clearance) are written;
  - scenes where the policy is inert (all |eta| < 0.05) are SKIPPED by
    default (nothing to distill; pass --keep_inert to keep them).

Usage:
    python -m rlvr.autoresearch.tools.distill_guided_targets \
        --model_path <base.pth> --policy_dir <dir> --scenes <list.json> \
        --out_dir <dir> --out_list <json> [--envelope v1] [--min_clearance 0.2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401
from exploration_policy.utils import run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer
from rlvr.autoresearch.tools.ghost_sim_common import extract_stopped_neighbors
from rlvr.autoresearch.tools.recovery_sim import deterministic_predict
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_sft_trainer import _smooth_trajectory
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data
from scenario_generation.explorer_runner import plan_static_clearance


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    parser.add_argument(
        "--ego_shape", required=True, help="WB,L,W — no default, must match the platform"
    )
    parser.add_argument("--min_clearance", type=float, default=0.2)
    parser.add_argument("--keep_inert", action="store_true")
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
    parser.add_argument("--envelope", choices=["v1", "v2"], default=None)
    parser.add_argument("--lambda_col", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, margs, device)
    ego_shape = tuple(float(x) for x in args.ego_shape.split(","))

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
            det = deterministic_predict(model, margs, data)
            batch = _stack_scene_data([data], device)
            norm = _normalize_batch(batch, margs)
            x_ref = torch.from_numpy(np.ascontiguousarray(det)).float()
            x_ref = x_ref.unsqueeze(0).to(device)
            norm["reference_trajectory"] = x_ref
            enc = run_frozen_encoder(model, norm)
            out = policy(enc, x_ref, deterministic=True)
            etas = {h: (2.0 * out.dists[h].mean - 1.0).reshape(1) for h in heads}
            eta_f = {h: float(v.item()) for h, v in etas.items()}
            if not args.keep_inert and all(abs(v) < 0.05 for v in eta_f.values()):
                n_inert += 1
                continue
            guided = (
                _batched_generate_varied_noise(
                    model,
                    margs,
                    norm,
                    noise_min=0.0,
                    noise_max=0.0,
                    first_deterministic=False,
                    composer=make_composer(
                        etas, args, envelope=getattr(policy, "guidance_envelope", None)
                    ),
                    device=device,
                )[0]
                .cpu()
                .numpy()
            )
            guided = _smooth_trajectory(guided, 11, 3)
            if boxes:
                clr = float(
                    plan_static_clearance(guided.astype(np.float32), boxes, ego_shape, device)
                )
                if clr < args.min_clearance:
                    n_unsafe += 1
                    continue
            else:
                clr = 99.0
        except Exception as e:  # noqa: BLE001
            print(f"  [err ] {Path(sp).name}: {e}")
            n_err += 1
            continue

        raw = dict(np.load(sp, allow_pickle=True))
        fut = raw["ego_agent_future"]
        T = min(fut.shape[0], guided.shape[0])
        if fut.shape[-1] == 3:
            # (x, y, yaw) format
            new = np.stack(
                [guided[:T, 0], guided[:T, 1], np.arctan2(guided[:T, 3], guided[:T, 2])], axis=-1
            )
        elif fut.shape[-1] == 4:
            new = guided[:T, :4]
        else:
            raise ValueError(
                f"{sp}: unsupported ego_agent_future width {fut.shape[-1]} "
                "(expected 3 or 4) — refusing to silently truncate"
            )
        raw["ego_agent_future"] = new.astype(fut.dtype)

        pool = Path(sp).parent.name
        out_path = out_dir / f"{pool}__{Path(sp).stem}_distilled.npz"
        np.savez(out_path, **raw)
        written.append(str(out_path))
        manifest.append(
            {"source": sp, "etas": eta_f, "guided_clearance": round(clr, 3), "out": str(out_path)}
        )

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"\nDistilled {len(written)} targets (skipped: {n_inert} inert, "
        f"{n_unsafe} unsafe-guided, {n_err} errors) -> {args.out_list}"
    )


if __name__ == "__main__":
    main()
