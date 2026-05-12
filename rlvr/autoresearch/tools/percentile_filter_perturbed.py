"""Filter a PRiSM-perturbed scene set by PERCENTILE-on-top1_cl.

This is the canonical PRiSM filter as of 2026-05-07 (replacing σ-Δ and absolute
Δcl ≥ 0.2 — both of those mixed perturbation difficulty with SFT target quality
and produced unfittable training targets). See memory ``project_prism_method``
+ ``feedback_prism_filter_canonical``.

Selection rule (top P percentile of *eligible* scenes by ``top1_cl``):
  1. Drop scenes whose t=0 is already-fine (``t0_cl > eligible_t0_max``).
  2. Drop scenes whose rank-1 winner is unsafe (rb_cross / lane_cross /
     collision / kinematic gate fired). Reward.py does NOT gate total reward
     on lane departure when ``enable_lane_departure=false``, so a rank-1 by
     total reward CAN cross a lane — those are explicit poison.
  3. Rank remaining by ``top1_cl`` ascending magnitude (closer-to-0 = best
     SFT target).
  4. Keep top ``percentile`` percent.

Input: viz_p4_recovery output dir's ``summary.json``.
Output: filtered scene list JSON + a one-page summary.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True,
                   help="Path to viz_p4_recovery output dir's summary.json.")
    p.add_argument("--output_scenes", type=Path, required=True,
                   help="Where to write filtered scene list (JSON list of NPZ paths).")
    p.add_argument("--output_report", type=Path, default=None,
                   help="Where to write filter decision report. Defaults to "
                        "alongside output_scenes with suffix '_report.json'.")
    p.add_argument("--percentile", type=float, default=50.0,
                   help="Keep top P %% by top1_cl. Default 50. Ignored when "
                        "--det_cl_max / --top1_cl_min are set.")
    p.add_argument("--det_cl_max", type=float, default=None,
                   help="Two-threshold filter: keep scenes where det_cl < "
                        "this value (model's deterministic is meaningfully "
                        "off-CL) AND top1_cl >= --top1_cl_min (rank-1 is "
                        "near-perfect). The actual PRiSM training signal: "
                        "scenes the model handles BADLY where K=N produces "
                        "a clean SFT target. Use e.g. -0.10 / -0.05.")
    p.add_argument("--top1_cl_min", type=float, default=None,
                   help="See --det_cl_max. Floor on top1_cl. e.g. -0.05.")
    p.add_argument("--eligible_t0_max", type=float, default=0.0,
                   help="Drop scenes with t0_cl > this value as already-fine. "
                        "Set to 0 (default) to disable this exclusion.")
    # No kin_gate flag — the check is unconditional. A scene whose rank-1
    # trajectory is kinematically infeasible (top1_kin_gate == False per
    # reward.py:3063 convention: True = gate PASSED = safe) is poison for
    # SFT: training the model to imitate undriveable physics is unsafe.
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    scenes = data["scenes"]

    rejected = {
        "already_fine_t0": 0,
        "top1_rb_cross": 0,
        "top1_lane_cross": 0,
        "top1_collision": 0,
        "top1_kin_gate_failed": 0,
    }
    eligible: list[dict] = []
    for s in scenes:
        if s["t0_cl"] > args.eligible_t0_max:
            rejected["already_fine_t0"] += 1
            continue
        if s["top1_rb_cross"]:
            rejected["top1_rb_cross"] += 1
            continue
        if s["top1_lane_cross"]:
            rejected["top1_lane_cross"] += 1
            continue
        if s["top1_coll_step"] is not None:
            rejected["top1_collision"] += 1
            continue
        if not s["top1_kin_gate"]:
            # top1_kin_gate=True per reward.py means gate PASSED (safe);
            # False means infeasible — drop those as unsafe SFT targets.
            rejected["top1_kin_gate_failed"] += 1
            continue
        eligible.append(s)

    if args.det_cl_max is not None and args.top1_cl_min is not None:
        # Two-threshold mode: real training signal (det bad + rank-1 good).
        kept = [
            s for s in eligible
            if s["det_cl"] < args.det_cl_max and s["top1_cl"] >= args.top1_cl_min
        ]
        kept.sort(key=lambda s: s["top1_cl"], reverse=True)
    else:
        eligible.sort(key=lambda s: s["top1_cl"], reverse=True)  # least-negative first
        n_keep = max(1, int(len(eligible) * args.percentile / 100.0))
        kept = eligible[:n_keep]
    paths = [s["scene"] for s in kept]

    args.output_scenes.parent.mkdir(parents=True, exist_ok=True)
    args.output_scenes.write_text(json.dumps(paths, indent=2))

    report_path = args.output_report or args.output_scenes.with_name(
        args.output_scenes.stem + "_report.json"
    )
    if kept:
        top1_kept = [s["top1_cl"] for s in kept]
        report = {
            "input_summary": str(args.summary),
            "percentile": args.percentile,
            "eligible_t0_max": args.eligible_t0_max,
            "n_total": len(scenes),
            "n_eligible": len(eligible),
            "n_kept": len(kept),
            "rejected_breakdown": rejected,
            "kept_top1_cl_min": min(top1_kept),
            "kept_top1_cl_max": max(top1_kept),
            "kept_top1_cl_median": sorted(top1_kept)[len(top1_kept) // 2],
            "eligibility_cutoff_top1_cl": kept[-1]["top1_cl"] if kept else None,
        }
    else:
        report = {
            "input_summary": str(args.summary),
            "percentile": args.percentile,
            "eligible_t0_max": args.eligible_t0_max,
            "n_total": len(scenes),
            "n_eligible": 0,
            "n_kept": 0,
            "rejected_breakdown": rejected,
        }
    report_path.write_text(json.dumps(report, indent=2))

    print(
        f"Filtered {len(scenes)} → eligible {len(eligible)} → kept {len(kept)} "
        f"(top {args.percentile:.0f}%). Rejected: {rejected}. "
        f"Wrote {args.output_scenes}."
    )


if __name__ == "__main__":
    main()
