#!/usr/bin/env python3
"""Latency benchmark: baseline det vs explorer K=1 vs explorer K=8.

Single-scene (B=1) online setting, CUDA-synced, warmed up. Per mode:
  baseline : one plain det generation (what the planner does today)
  k1       : det generation (policy reference) + encoder + policy forward
             + ONE guided generation  (single-pass explorer)
  k8       : same, but 1+8 guided generations in one batched call
             (mean + 8 sampled candidates — the fan / refine mode)

Reports per-stage and total mean/p95 ms over scenes x repeats.

Usage:
    python -m rlvr.autoresearch.tools.bench_explorer_latency \
        --model_path <base.pth> --policy_dir <dir> --scenes <json> \
        [--n_scenes 10] [--repeats 5] \
        [--lambda_lat 5.0] [--lat_scale 2.0] [--col_scale 9.0]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401
from exploration_policy.utils import run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer
from rlvr.autoresearch.tools.recovery_sim import deterministic_predict
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--n_scenes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument(
        "--no_dit_memo",
        action="store_true",
        help="disable the guided-frame DiT forward memo (A/B escape hatch; memo is the default)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)
    with open(args.scenes) as f:
        paths = json.load(f)[: args.n_scenes]
    datas = [load_npz_data(p, device) for p in paths]

    t = {"baseline": [], "k1_det": [], "k1_policy": [], "k1_guided": [], "k8_guided": []}

    # warmup
    for d in datas[:2]:
        deterministic_predict(model, model_args, d)
    _sync()

    for rep in range(args.repeats):
        for d in datas:
            # --- baseline: plain det generation ---
            _sync()
            t0 = time.perf_counter()
            det = deterministic_predict(model, model_args, d)
            _sync()
            t["baseline"].append(time.perf_counter() - t0)

            # --- k1: det (reference) + encoder+policy + 1 guided gen ---
            _sync()
            t0 = time.perf_counter()
            det = deterministic_predict(model, model_args, d)
            _sync()
            t1 = time.perf_counter()
            batch = _stack_scene_data([d], device)
            norm = _normalize_batch(batch, model_args)
            x_ref = torch.from_numpy(np.ascontiguousarray(det)).float()
            x_ref = x_ref.unsqueeze(0).to(device)
            norm["reference_trajectory"] = x_ref
            enc = run_frozen_encoder(model, norm)
            out = policy(enc, x_ref, deterministic=True)
            etas1 = {h: (2.0 * out.dists[h].mean - 1.0).reshape(1) for h in heads}
            _sync()
            t2 = time.perf_counter()
            _ = _batched_generate_varied_noise(
                model,
                model_args,
                norm,
                noise_min=0.0,
                noise_max=0.0,
                first_deterministic=False,
                composer=make_composer(etas1, args),
                device=device,
                use_dit_memo=not args.no_dit_memo,
            )
            _sync()
            t3 = time.perf_counter()
            t["k1_det"].append(t1 - t0)
            t["k1_policy"].append(t2 - t1)
            t["k1_guided"].append(t3 - t2)

            # --- k8: 1 mean + 8 sampled candidates, one batched gen ---
            etas9 = {
                h: torch.cat(
                    [
                        (2.0 * out.dists[h].mean - 1.0).reshape(1),
                        (2.0 * out.dists[h].rsample((8,)).reshape(-1) - 1.0),
                    ]
                )
                for h in heads
            }
            gen = {
                k: (
                    v.expand(9, *v.shape[1:]).contiguous()
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1
                    else v
                )
                for k, v in norm.items()
            }
            _sync()
            t0 = time.perf_counter()
            _ = _batched_generate_varied_noise(
                model,
                model_args,
                gen,
                noise_min=0.0,
                noise_max=0.0,
                first_deterministic=False,
                composer=make_composer(etas9, args),
                device=device,
                use_dit_memo=not args.no_dit_memo,
            )
            _sync()
            t["k8_guided"].append(time.perf_counter() - t0)

    def ms(v):
        a = np.array(v) * 1000
        return {
            "mean_ms": round(float(a.mean()), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
        }

    stages = {k: ms(v) for k, v in t.items()}
    report = {
        "n_scenes": len(datas),
        "repeats": args.repeats,
        "stages": stages,
        "totals": {
            "baseline": stages["baseline"]["mean_ms"],
            "k1_total": round(
                stages["k1_det"]["mean_ms"]
                + stages["k1_policy"]["mean_ms"]
                + stages["k1_guided"]["mean_ms"],
                2,
            ),
            "k8_total": round(
                stages["k1_det"]["mean_ms"]
                + stages["k1_policy"]["mean_ms"]
                + stages["k8_guided"]["mean_ms"],
                2,
            ),
        },
    }
    report["totals"]["k1_overhead_x"] = round(
        report["totals"]["k1_total"] / report["totals"]["baseline"], 2
    )
    report["totals"]["k8_overhead_x"] = round(
        report["totals"]["k8_total"] / report["totals"]["baseline"], 2
    )
    print(json.dumps(report, indent=1))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
