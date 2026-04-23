#!/usr/bin/env python3
"""Pick pre-trigger replay NPZs from a live metrics log.

``scenario_generation.replay`` writes ``metrics_log.json`` alongside the
per-step NPZs whenever a reward config is supplied. Each record carries
the step index + lane gate / near-frac / border-min-dist / centerline-score
for that step. This tool reads that log, flags trigger steps (scenes where
the ego is drifting) and keeps the ``--lookback`` steps *before* each
trigger as training data — those are the frames where the model could
still steer away.

Trigger (OR — any fires):
    1. ``lane_gate < 0.5``                (lane cross)
    2. ``lane_near_frac > 0``             (within lane_near_thresh of edge;
                                           threshold baked in at sim time)
    3. ``rb_min_dist < --rb_min_dist``    (within N metres of a border)
    4. ``abs(cl_score) > --cl_min_abs``   (off-center by more than N)

Lookback does not cross NPZ-directory boundaries, so combining multiple
replay runs into one log is safe.

Usage:
    python -m rlvr.autoresearch.tools.select_from_metrics_log \\
        --metrics_log /path/to/replay_run/metrics_log.json \\
        --npz_dir     /path/to/replay_run/npz/ \\
        --output      /path/to/kept_scenes.json \\
        --diagnostics /path/to/diag.json \\
        --cl_min_abs 0.35 --rb_min_dist 0.45 --lookback 20

``--npz_dir`` must point at the ``npz/`` subdirectory of a replay run,
not the run directory itself — the tool resolves each step's NPZ as
``<npz_dir>/replay_step_NNNN.npz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _trigger_reasons(
    rec: dict, rb_min_dist: float, cl_min_abs: float,
) -> list[str]:
    reasons: list[str] = []
    if rec.get("lane_gate", 1.0) < 0.5:
        reasons.append("lane_cross")
    if rec.get("lane_near_frac", 0.0) > 0.0:
        reasons.append("lane_near")
    if rec.get("rb_min_dist", 99.0) < rb_min_dist:
        reasons.append("rb_near")
    if abs(rec.get("cl_score", 0.0)) > cl_min_abs:
        reasons.append("cl_far")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_log", type=str, required=True,
                        help="metrics_log.json written by scenario_generation.replay")
    parser.add_argument("--npz_dir", type=str, required=True,
                        help="Directory containing replay_step_NNNN.npz files")
    parser.add_argument("--output", type=str, required=True,
                        help="JSON list of kept (pre-trigger) NPZ paths")
    parser.add_argument("--diagnostics", type=str, default=None,
                        help="Optional per-step trigger/keep dump")
    parser.add_argument("--rb_min_dist", type=float, required=True,
                        help="Trigger if rb_min_dist < this (metres)")
    parser.add_argument("--cl_min_abs", type=float, required=True,
                        help="Trigger if |cl_score| > this")
    parser.add_argument("--lookback", type=int, required=True,
                        help="Number of pre-trigger scenes to keep")
    parser.add_argument("--include_trigger", action="store_true",
                        help="Also keep the trigger scenes (default: drop — "
                             "the lookback is the preventable window)")
    args = parser.parse_args()

    with open(args.metrics_log) as f:
        payload = json.load(f)
    steps = payload["steps"] if isinstance(payload, dict) else payload

    npz_dir = Path(args.npz_dir)

    # Stamp trigger state on each record and collect the ordered step index.
    for rec in steps:
        reasons = _trigger_reasons(rec, args.rb_min_dist, args.cl_min_abs)
        rec["reasons"] = reasons
        rec["is_trigger"] = len(reasons) > 0

    # Resolve NPZ paths once. Missing NPZs are dropped from the kept set but
    # still scored in diagnostics so the user can tell why.
    paths: list[Path | None] = []
    for rec in steps:
        p = npz_dir / f"replay_step_{rec['step']:04d}.npz"
        paths.append(p if p.exists() else None)

    kept_idx: set[int] = set()
    for i, rec in enumerate(steps):
        if not rec["is_trigger"]:
            continue
        lo = max(0, i - args.lookback)
        for j in range(lo, i):
            if paths[j] is not None:
                kept_idx.add(j)
        if args.include_trigger and paths[i] is not None:
            kept_idx.add(i)

    for i, rec in enumerate(steps):
        rec["kept"] = i in kept_idx
        rec["npz_path"] = str(paths[i]) if paths[i] is not None else None

    kept_paths = [str(paths[i]) for i in sorted(kept_idx)]

    reason_counts: dict[str, int] = {}
    for rec in steps:
        if rec["is_trigger"]:
            for r in rec["reasons"]:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    n_trigger = sum(1 for r in steps if r["is_trigger"])
    n_missing = sum(1 for p in paths if p is None)
    print(f"Total steps:      {len(steps)}")
    print(f"Triggers:         {n_trigger}")
    print(f"Missing NPZs:     {n_missing}")
    print(f"Kept (lookback):  {len(kept_paths)}")
    print("Trigger reasons (can overlap):")
    for r, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {r:10s} {n}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(kept_paths, f, indent=2)
    print(f"\nSaved {len(kept_paths)} kept NPZ paths to {out_path}")

    if args.diagnostics:
        diag_path = Path(args.diagnostics)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(steps, f, indent=2)
        print(f"Saved per-step diagnostics to {diag_path}")


if __name__ == "__main__":
    main()
