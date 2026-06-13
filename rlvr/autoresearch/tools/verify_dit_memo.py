#!/usr/bin/env python3
"""Equivalence + latency check for the guided-sampling DiT forward memo.

For each scene x envelope x eta, generates the guided trajectory (K=1,
zero initial latent -> deterministic) three ways:

  slow : stock GuidanceComposer            (fast=False, no memo)
  fast : FastGuidanceComposer              (memo off)
  memo : FastGuidanceComposer + dit_memo   (the wired default)

and reports the pairwise max |dtraj| over the full [T, 4] ego plan —
the bit-identical claim requires memo-vs-fast == 0 exactly — plus
active-frame latency (mean/p95 ms) per leg and the memo hit/miss
counters from one instrumented generation per scene.

Usage:
    python -m rlvr.autoresearch.tools.verify_dit_memo \
        --model_path <base.pth> --scenes <json> \
        [--n_scenes 3] [--etas -0.75,-0.3,0.3,0.75] \
        [--envelopes v1,v2] [--repeats 3] [--output <json>]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401 -- registers the batched variants
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.recovery_sim import deterministic_predict
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data
from rlvr.guidance_batched import build_head_composer, dit_memo


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gen(model, model_args, norm, composer, device, use_dit_memo):
    return _batched_generate_varied_noise(
        model,
        model_args,
        norm,
        noise_min=0.0,
        noise_max=0.0,
        first_deterministic=False,
        composer=composer,
        device=device,
        use_dit_memo=use_dit_memo,
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--n_scenes", type=int, default=3)
    parser.add_argument("--etas", default="-0.75,-0.3,0.3,0.75")
    parser.add_argument("--envelopes", default="v1,v2")
    parser.add_argument(
        "--repeats", type=int, default=3, help="timing repeats per (scene, envelope, eta)"
    )
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    with open(args.scenes) as f:
        paths = json.load(f)[: args.n_scenes]
    etas_grid = [float(v) for v in args.etas.split(",")]
    envelopes = args.envelopes.split(",")

    legs = ("slow", "fast", "memo")
    diffs = {"fast_vs_slow": [], "memo_vs_fast": [], "memo_vs_slow": []}
    times = {leg: [] for leg in legs}
    memo_stats = []
    cases = []

    for path in paths:
        data = load_npz_data(path, device)
        det = deterministic_predict(model, model_args, data)
        batch = _stack_scene_data([data], device)
        norm = _normalize_batch(batch, model_args)
        x_ref = torch.from_numpy(np.ascontiguousarray(det)).float()
        norm["reference_trajectory"] = x_ref.unsqueeze(0).to(device)

        for env in envelopes:
            for e in etas_grid:
                head_etas = {"lateral": e, "collision": e, "stretch": e}
                composers = {
                    "slow": build_head_composer(
                        head_etas, envelope=env, fast=False, guidance_scale=args.guidance_scale
                    ),
                    "fast": build_head_composer(
                        head_etas, envelope=env, fast=True, guidance_scale=args.guidance_scale
                    ),
                }
                trajs = {}
                for rep in range(args.repeats):
                    for leg in legs:
                        composer = composers["slow" if leg == "slow" else "fast"]
                        # FastGuidanceComposer caches the inverse-normalised
                        # observations per generation; rebuild per timed run
                        # so every repeat pays the same first-call cost.
                        if leg != "slow":
                            composer = build_head_composer(
                                head_etas,
                                envelope=env,
                                fast=True,
                                guidance_scale=args.guidance_scale,
                            )
                        _sync()
                        t0 = time.perf_counter()
                        traj = _gen(
                            model, model_args, norm, composer, device, use_dit_memo=(leg == "memo")
                        )
                        _sync()
                        times[leg].append(time.perf_counter() - t0)
                        if rep == 0:
                            trajs[leg] = traj[0].cpu()
                case = {
                    "scene": path,
                    "envelope": env,
                    "eta": e,
                    "fast_vs_slow": float((trajs["fast"] - trajs["slow"]).abs().max()),
                    "memo_vs_fast": float((trajs["memo"] - trajs["fast"]).abs().max()),
                    "memo_vs_slow": float((trajs["memo"] - trajs["slow"]).abs().max()),
                }
                for k in diffs:
                    diffs[k].append(case[k])
                cases.append(case)

        # one instrumented run per scene for memo hit/miss counters
        composer = build_head_composer(
            {"lateral": etas_grid[0], "collision": etas_grid[0], "stretch": etas_grid[0]},
            envelope=envelopes[0],
            fast=True,
            guidance_scale=args.guidance_scale,
        )
        _orig_fn = model.decoder._guidance_fn
        _orig_scale = model.decoder._guidance_scale
        model.decoder._guidance_fn = composer
        model.decoder._guidance_scale = composer._set_config.global_scale
        P = 1 + model_args.predicted_neighbor_num
        norm["sampled_trajectories"] = torch.zeros(
            1, P, model_args.future_len + 1, 4, device=device
        )
        try:
            with dit_memo(model.decoder) as memo:
                model(norm)
        finally:
            model.decoder._guidance_fn = _orig_fn
            model.decoder._guidance_scale = _orig_scale
        memo_stats.append({"scene": path, "hits": memo.hits, "misses": memo.misses})

    def ms(v):
        a = np.array(v) * 1000
        return {
            "mean_ms": round(float(a.mean()), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
        }

    # Guard against a false "equivalent" verdict from a memo that never fired:
    # if hits == 0 everywhere, memo_vs_fast would be 0.0 simply because the memo
    # did nothing. A real run hits once per guided solver step.
    total_hits = sum(m["hits"] for m in memo_stats)
    if total_hits == 0:
        raise RuntimeError(
            "DiT memo recorded 0 hits across all instrumented scenes — the "
            "optimization never fired, so the equivalence numbers are vacuous. "
            "Check that the composer is active (non-inert) and dit_memo is "
            f"installed. memo_stats={memo_stats}"
        )

    report = {
        "n_scenes": len(paths),
        "etas": etas_grid,
        "envelopes": envelopes,
        "repeats": args.repeats,
        "max_dtraj_m": {k: max(v) for k, v in diffs.items()},
        "active_frame_latency": {leg: ms(v) for leg, v in times.items()},
        "memo_stats": memo_stats,
        "total_memo_hits": total_hits,
        "cases": cases,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=1))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
        print(f"full per-case report -> {args.output}")


if __name__ == "__main__":
    main()
