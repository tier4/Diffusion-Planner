"""Filter a PRiSM-perturbed scene set by PERCENTILE-on-top1_cl.

This is the canonical PRiSM filter as of 2026-05-07 (replacing σ-Δ and absolute
Δcl ≥ 0.2 — both of those mixed perturbation difficulty with SFT target quality
and produced unfittable training targets). See memory ``project_prism_method``
+ ``feedback_prism_filter_canonical``.

The gate-check logic (``is_scene_eligible``) is the single source of truth
for whether a rank-1 trajectory is safe for training. Both this filter and
``viz_p4_recovery`` import it — never duplicate the checks.

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


def is_scene_eligible(top1: dict, t0_cl: float = -999.0,
                      eligible_t0_max: float = 0.0) -> bool:
    """Check if a rank-1 result passes all safety gates for training.

    This is the single source of truth — used by both this filter script
    and viz_p4_recovery to classify scenes as improve/no_improve.

    Returns True if the scene is safe for training (no gate violations).
    """
    if t0_cl > eligible_t0_max:
        return False
    if top1.get("rb_cross", False):
        return False
    if top1.get("lane_cross", False):
        return False
    if top1.get("coll_step") is not None:
        return False
    if top1.get("static_crossing", False):
        return False
    if top1.get("kin_violated", False):
        return False
    return True


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
    p.add_argument("--min_top1_vs_det", type=float, default=0.0,
                   help="Reject scenes where top1_cl - det_cl <= this value. "
                        "Both top1_cl and det_cl are full-80-frame trajectory "
                        "cl scores (apples-to-apples). Default 0.0 = strict "
                        "improvement over deterministic. Set negative to "
                        "relax (e.g. -0.05 allows ties / mild regressions).")
    p.add_argument("--eligible_t0_max", type=float, default=0.0,
                   help="Drop scenes with t0_cl > this value as already-fine. "
                        "Set to 0 (default) to disable this exclusion.")
    # No kin flag — the check is unconditional. A scene whose rank-1
    # trajectory has top1_kin_violated == True (infeasible) is poison for
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
        "top1_static_crossing": 0,
        "top1_kin_violated": 0,
        "no_improvement": 0,
    }
    eligible: list[dict] = []
    for s in scenes:
        top1_dict = {
            "rb_cross": s["top1_rb_cross"],
            "lane_cross": s["top1_lane_cross"],
            "coll_step": s["top1_coll_step"],
            "static_crossing": s.get("top1_static_crossing", False),
            "kin_violated": s["top1_kin_violated"],
        }
        if not is_scene_eligible(top1_dict, t0_cl=s["t0_cl"],
                                 eligible_t0_max=args.eligible_t0_max):
            if s["t0_cl"] > args.eligible_t0_max:
                rejected["already_fine_t0"] += 1
            elif s["top1_rb_cross"]:
                rejected["top1_rb_cross"] += 1
            elif s["top1_lane_cross"]:
                rejected["top1_lane_cross"] += 1
            elif s["top1_coll_step"] is not None:
                rejected["top1_collision"] += 1
            elif s.get("top1_static_crossing", False):
                rejected["top1_static_crossing"] += 1
            elif s["top1_kin_violated"]:
                rejected["top1_kin_violated"] += 1
            continue
        # Require the rank-1 winner to actually IMPROVE on the model's
        # DETERMINISTIC output. Both top1_cl and det_cl come from
        # compute_reward_batch on the FULL 80-frame trajectory — same
        # quantity, apples-to-apples comparison. Scenes where rank-1 is
        # no better than det produce no SFT signal (training the model on
        # its own output is a no-op). The min_top1_vs_det floor is
        # configurable; default 0.0 = strict improvement.
        #
        # WARNING: do NOT compare top1_cl to t0_cl. t0_cl is a 1-frame
        # cl-score at ego_current_state (single point); top1_cl is the
        # 80-frame mean — incommensurate. Filter version 2026-05-12 had
        # this bug and over-rejected scenes whose trajectory naturally
        # accumulates cl across a curve.
        delta = s["top1_cl"] - s["det_cl"]
        if delta <= args.min_top1_vs_det:
            rejected["no_improvement"] += 1
            continue
        eligible.append(s)

    if (args.det_cl_max is None) != (args.top1_cl_min is None):
        raise SystemExit("--det_cl_max and --top1_cl_min must both be set or both omitted.")
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
            "n_eligible": len(eligible),
            "n_kept": 0,
            "rejected_breakdown": rejected,
        }
    report_path.write_text(json.dumps(report, indent=2))

    if args.det_cl_max is not None:
        mode_desc = f"det_cl<{args.det_cl_max} & top1_cl>={args.top1_cl_min}"
    else:
        mode_desc = f"top {args.percentile:.0f}%"
    print(
        f"Filtered {len(scenes)} → eligible {len(eligible)} → kept {len(kept)} "
        f"({mode_desc}). Rejected: {rejected}. "
        f"Wrote {args.output_scenes}."
    )


if __name__ == "__main__":
    main()
