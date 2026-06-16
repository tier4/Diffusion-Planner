#!/usr/bin/env python3
"""Score the CURATED TARGETS (ego_agent_future) of a scene list with the
canonical reward path — the no-poison check for curated SFT datasets.

A curated-SFT leg trains the model to imitate ego_agent_future. If that target
itself collides with a stopped neighbor (or grazes inside the static-collision
threshold), the scene can never teach avoidance — it actively teaches the
collision. This tool feeds each scene's stored ego_agent_future through
`compute_reward_batch` (the same reward.py OBB path eval_det_avoidance uses for
model outputs — no reimplemented geometry) and reports per-scene
sc_min_dist / static_crossing / rb / lane flags for the TARGET trajectory.

Usage:
    python -m rlvr.autoresearch.tools.score_curated_targets \
        --scenes <scenes.json> --config <reward_config.json> \
        --ego_shape WB,L,W --output <report.json>

Output: per-scene rows + an aggregate, with the poison list (targets that
cross) printed and saved for direct exclusion from training lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import aggregate_stats
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import compute_reward_batch


def score_targets(
    scene_paths: list[str], rcfg, ego_shape: np.ndarray, device: torch.device
) -> list[dict]:
    results = []
    for p in scene_paths:
        try:
            d = load_npz_data(p, device)
            es = d["ego_shape"].cpu().numpy().reshape(-1)[:3]
            if not np.allclose(es, ego_shape, atol=1e-2):
                print(f"  [skip] {Path(p).name}: ego_shape={es.tolist()}")
                continue
            target = d["ego_agent_future"]
            if target.dim() == 2:
                target = target.unsqueeze(0)
            r = compute_reward_batch(target, d, rcfg)[0]
            results.append(
                {
                    "scene": Path(p).name,
                    "scene_path": str(p),
                    "sc_min_dist": float(getattr(r, "sc_min_dist", 99.0)),
                    "rb_min_dist": float(getattr(r, "rb_min_dist", 99.0)),
                    "cl": float(r.centerline),
                    "total": float(r.total),
                    "static_crossing": bool(r.static_crossing),
                    "rb_cross": bool(r.rb_crossing),
                    "lane_cross": bool(r.lane_crossing),
                    "kin_violated": bool(r.kinematic_violated),
                    "sc_n_stopped": int(getattr(r, "sc_n_stopped", 0)),
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {Path(p).name}: {e}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--config", required=True, help="Reward config JSON")
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--output", required=True, help="Report JSON path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = load_reward_config(args.config)
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    scene_paths = json.load(open(args.scenes))

    results = score_targets(scene_paths, rcfg, ego_shape, device)
    agg = aggregate_stats(results)
    poison = [r for r in results if r["static_crossing"]]

    out = {
        "scenes": args.scenes,
        "aggregate": agg,
        "poison": [r["scene_path"] for r in poison],
        "rows": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    s = agg["sc_min_dist"]
    print(
        f"TARGETS: {agg['n_scenes']} scored | static-crossing targets (POISON): "
        f"{agg['static_crossings']} | rb={agg['rb_crossings']} lane={agg['lane_crossings']} "
        f"kin={agg['kin_violated']}"
    )
    print(
        f"  target sc_min: mean={s['mean']:+.3f} p5={s['p5']:+.3f} p25={s['p25']:+.3f} "
        f"min={s['min']:+.3f}"
    )
    for r in poison:
        print(f"  POISON {r['scene']}  sc={r['sc_min_dist']:+.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
