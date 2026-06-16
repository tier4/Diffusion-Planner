#!/usr/bin/env python3
"""Repair poison/weak curated targets by synthesizing an AVOIDING target.

For scenes whose stored ego_agent_future collides with (or barely misses) a
stopped obstacle, curated SFT teaches the collision (margin transfer is ~1:1).
Instead of dropping those scenes, this tool regenerates the target: sample K
trajectories from a source model, score each with the FULL reward
(compute_reward_batch — same OBB path as eval), and keep the best-reward sample
whose static-collision margin clears --min_margin. The winner is written into
ego_agent_future (all other NPZ fields copied verbatim).

Scenes with NO sample clearing the margin are NOT written (fail loudly,
reported as UNREPAIRED) — a weak repair is poison with extra steps.

Reuses ONLY existing fns: eval_det_avoidance.{load_model,load_npz_data},
grpo_trainer_batched.{_stack_scene_data,_normalize_batch,generate_all_scenes_batched},
reward.compute_reward_batch.

Usage:
    python -m rlvr.autoresearch.tools.build_avoiding_target \
        --model <source.pth> --scenes <poison.json> --config <reward.json> \
        --ego_shape WB,L,W --min_margin 0.3 \
        --out_dir <dir> --out_list <repaired.json>
"""

import argparse
import json
import os

import numpy as np
import torch

from rlvr.autoresearch.tools.eval_det_avoidance import load_model, load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
)
from rlvr.reward import compute_reward_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="generation source model")
    ap.add_argument("--scenes", required=True, help="JSON list of NPZs to repair")
    ap.add_argument("--config", required=True, help="reward config JSON")
    ap.add_argument("--ego_shape", required=True, help="WB,L,W — validated against each NPZ")
    ap.add_argument(
        "--min_margin",
        type=float,
        required=True,
        help="required sc_min_dist of the new target (e.g. 0.3)",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument(
        "--variant",
        default="rsft_v2",
        help="generation variant (noise-heavy default for diversity)",
    )
    ap.add_argument("--gt_max_speed", type=float, default=9.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = load_reward_config(args.config)
    model, margs = load_model(args.model, dev)
    cli_es = np.array([float(x) for x in args.ego_shape.split(",")])
    paths = json.load(open(args.scenes))
    os.makedirs(args.out_dir, exist_ok=True)

    written, unrepaired = [], []
    for p in paths:
        d = load_npz_data(p, dev)
        npz_es = d["ego_shape"].detach().cpu().numpy().reshape(-1)[:3]
        if not np.allclose(npz_es, cli_es, atol=1e-2):
            raise ValueError(
                f"{p}: --ego_shape {cli_es.tolist()} != NPZ ego_shape "
                f"{npz_es.tolist()} (platform mismatch)"
            )
        nb = _normalize_batch(_stack_scene_data([d], dev), margs)
        trajs = generate_all_scenes_batched(
            model,
            margs,
            nb,
            K=args.K,
            noise_range=(0.5, 2.0),
            device=dev,
            gen_chunk_size=args.K,
            gt_max_speed=args.gt_max_speed,
            generation_variant=args.variant,
            use_route_cl_guidance=True,
        )[0]  # (K, T, 4)
        rewards = compute_reward_batch(trajs, d, rcfg)
        cands = []
        for k, r in enumerate(rewards):
            scm = float(getattr(r, "sc_min_dist", 99.0))
            if scm >= args.min_margin and not r.kinematic_violated and not r.stopped:
                cands.append((float(r.total), scm, k))
        name = os.path.basename(p)
        if not cands:
            best_scm = max(float(getattr(r, "sc_min_dist", 99.0)) for r in rewards)
            unrepaired.append(p)
            print(
                f"  UNREPAIRED {name}: best sample sc_min={best_scm:+.3f} "
                f"< margin {args.min_margin}"
            )
            continue
        cands.sort(reverse=True)
        total, scm, k = cands[0]
        raw = dict(np.load(p, allow_pickle=True))
        raw["ego_agent_future"] = trajs[k].detach().cpu().numpy().astype(np.float32)
        out_p = os.path.join(args.out_dir, name)
        np.savez(out_p, **raw)
        written.append(out_p)
        print(f"  repaired {name}: slot {k}  sc_min={scm:+.3f}  total={total:+.1f}")

    json.dump(written, open(args.out_list, "w"), indent=1)
    print(f"repaired {len(written)}/{len(paths)} -> {args.out_dir}; UNREPAIRED: {len(unrepaired)}")
    if unrepaired:
        ur = os.path.splitext(args.out_list)[0] + "_unrepaired.json"
        json.dump(unrepaired, open(ur, "w"), indent=1)
        print(f"  unrepaired list -> {ur}")


if __name__ == "__main__":
    main()
