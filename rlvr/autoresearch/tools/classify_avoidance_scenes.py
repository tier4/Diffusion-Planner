#!/usr/bin/env python3
"""Classify scenes as avoidance / non-avoidance using the guidance explorer.

The exploration policy was trained to request lateral / collision-swerve
guidance exactly when the frozen planner's deterministic trajectory needs
an avoidance correction, and to stay inert (eta ~ 0) otherwise. This tool
runs the policy (deterministic = Beta means) over an NPZ scene list and
flags a scene as "avoidance" when the requested guidance exceeds
configurable per-head thresholds — a weak signal below threshold is
treated as not-really-avoidance.

No guided generation and no reward scoring happen here: per scene this is
one deterministic planner pass (for the reference trajectory), one frozen
encoder pass, and the policy head.

Outputs a JSON report (per-scene etas + flag + which head(s) triggered,
plus summary counts and |eta| distributions) and, optionally, plain NPZ
path lists for the two classes, directly usable as dataset lists.

Usage:
    python -m rlvr.autoresearch.tools.classify_avoidance_scenes \
        --model_path <base.pth> --policy_dir <dir with exploration_policy.pth> \
        --scenes <scenes.json> --out <report.json> \
        [--lat_thresh 0.15] [--col_thresh 0.15] [--rule any] \
        [--out_avoidance_list <a.json>] [--out_normal_list <n.json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy


@torch.no_grad()
def scene_etas(model, model_args, policy, heads, npz_path, device) -> dict[str, float]:
    """Deterministic per-head etas in [-1, 1] for one scene."""
    data = load_npz_data(npz_path, device)
    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)
    x_ref_np = generate_reference_trajectory(model, model_args, norm_data, device)
    x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device)
    norm_data["reference_trajectory"] = x_ref
    enc = run_frozen_encoder(model, norm_data)
    out = policy(enc, x_ref, deterministic=True)
    return {h: float(2.0 * out.dists[h].mean - 1.0) for h in heads}


def classify(etas: dict[str, float], lat_thresh: float, col_thresh: float,
             rule: str) -> tuple[bool, list[str]]:
    """Return (is_avoidance, triggered_heads) from the per-head etas."""
    triggers = []
    if "lateral" in etas and abs(etas["lateral"]) >= lat_thresh:
        triggers.append("lateral")
    if "collision" in etas and abs(etas["collision"]) >= col_thresh:
        triggers.append("collision")
    if rule == "any":
        return bool(triggers), triggers
    # rule == "both": only count scenes where BOTH heads fire
    return ("lateral" in triggers and "collision" in triggers), triggers


def _pct(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    a = np.abs(np.asarray(vals))
    qs = np.percentile(a, [5, 25, 50, 75, 95])
    return {
        "mean": round(float(a.mean()), 4),
        "p5": round(float(qs[0]), 4), "p25": round(float(qs[1]), 4),
        "p50": round(float(qs[2]), 4), "p75": round(float(qs[3]), 4),
        "p95": round(float(qs[4]), 4), "max": round(float(a.max()), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True,
                        help="frozen base planner checkpoint (must be the "
                             "model the policy was trained against)")
    parser.add_argument("--policy_dir", required=True,
                        help="dir with exploration_policy.pth + "
                             "exploration_policy_config.json")
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--lat_thresh", type=float, default=0.15,
                        help="min |eta_lateral| to count as an avoidance "
                             "request (policy inertness bar on normal scenes "
                             "is ~0.1; below this = weak signal)")
    parser.add_argument("--col_thresh", type=float, default=0.15,
                        help="min |eta_collision| to count as an avoidance "
                             "request")
    parser.add_argument("--rule", choices=["any", "both"], default="any",
                        help="'any': either head over threshold flags the "
                             "scene; 'both': require both heads")
    parser.add_argument("--out_avoidance_list", default=None,
                        help="optional JSON list of flagged NPZ paths")
    parser.add_argument("--out_normal_list", default=None,
                        help="optional JSON list of non-flagged NPZ paths")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)
    if "lateral" not in heads and "collision" not in heads:
        raise ValueError(
            f"policy heads {heads} contain neither 'lateral' nor 'collision' "
            "— this classifier keys on the avoidance heads")

    with open(args.scenes) as f:
        paths = json.load(f)

    rows, per_head_abs = [], {h: [] for h in heads}
    for i, sp in enumerate(paths):
        etas = scene_etas(model, model_args, policy, heads, sp, device)
        is_avoid, triggers = classify(
            etas, args.lat_thresh, args.col_thresh, args.rule)
        rows.append({
            "scene": sp,
            "etas": {h: round(v, 4) for h, v in etas.items()},
            "avoidance": is_avoid,
            "triggered": triggers,
        })
        for h, v in etas.items():
            per_head_abs[h].append(v)
        if (i + 1) % 50 == 0:
            print(f"  [classify] {i + 1}/{len(paths)}")

    n_avoid = sum(r["avoidance"] for r in rows)
    summary = {
        "n_scenes": len(rows),
        "n_avoidance": n_avoid,
        "n_normal": len(rows) - n_avoid,
        "thresholds": {"lat": args.lat_thresh, "col": args.col_thresh,
                       "rule": args.rule},
        "policy_dir": args.policy_dir,
        "model_path": args.model_path,
        "abs_eta_distribution": {h: _pct(v) for h, v in per_head_abs.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "scenes": rows}, f, indent=1)

    if args.out_avoidance_list:
        with open(args.out_avoidance_list, "w") as f:
            json.dump([r["scene"] for r in rows if r["avoidance"]], f, indent=1)
    if args.out_normal_list:
        with open(args.out_normal_list, "w") as f:
            json.dump([r["scene"] for r in rows if not r["avoidance"]], f, indent=1)

    print(f"\n[classify] {len(rows)} scenes: {n_avoid} avoidance, "
          f"{len(rows) - n_avoid} normal "
          f"(|eta_lat|>={args.lat_thresh} {args.rule} "
          f"|eta_col|>={args.col_thresh})")
    for h, st in summary["abs_eta_distribution"].items():
        if st:
            print(f"  |eta_{h}|: mean {st['mean']} p50 {st['p50']} "
                  f"p95 {st['p95']} max {st['max']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
