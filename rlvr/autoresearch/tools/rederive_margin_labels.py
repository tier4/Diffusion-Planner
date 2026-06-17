#!/usr/bin/env python3
"""Re-derive sweep labels with a PROACTIVE clearance margin.

The original labels teach "act only when the det plan collides"
(sc crossing at <0.2 m). This transform re-labels the SAME per-scene combo
tables so that any scene whose det clearance is below --margin counts as
needing action, and the target is the best-reward combo that achieves at
least the margin (clean: no sc/rb crossing, not stopped):

  det sc_min_dist <  margin -> "solved" with best clean combo
                               having sc_min_dist >= margin
                               (fallback: best clean combo that improves
                                det clearance by >= --min_gain; else
                                "unsolved" = excluded)
  det sc_min_dist >= margin -> "already_clean" (eta = 0 target)

Output is sweep_labels-format JSON, directly consumable by
rlvr.train_explorer_regression.

Usage:
    python -m rlvr.autoresearch.tools.rederive_margin_labels \
        --labels <sweep_labels.json ...> --margin 0.7 --min_gain 0.3 \
        --out <margin_labels.json>
"""

from __future__ import annotations

import argparse
import json


def _neighborhood_min_clearance(c: dict, combos: list[dict], radius: float) -> float:
    """Worst sc clearance over the combo's eta-neighbourhood (incl. itself).

    A combo whose neighbours collide sits on a response CLIFF (prior mode
    boundary): regression eta error — and the Beta mean's compression of
    +-1 targets — lands the policy on the bad side at inference (measured:
    label eta +1.0 -> clearance +1.6, policy eta +0.82 -> -0.97 on the
    same scene). Robust labels live mid-plateau. Crossing/stopped
    neighbours count as clearance 0.
    """
    worst = c["sc_min_dist"]
    for o in combos:
        if (
            abs(o["eta_lat"] - c["eta_lat"]) <= radius + 1e-6
            and abs(o["eta_col"] - c["eta_col"]) <= radius + 1e-6
            and o.get("stretch", 1.0) == c.get("stretch", 1.0)
        ):
            v = 0.0 if (o["static_crossing"] or o["rb_cross"] or o["stopped"]) else o["sc_min_dist"]
            worst = min(worst, v)
    return worst


def relabel_scene(
    r: dict,
    margin: float,
    min_gain: float,
    robust_radius: float = 0.0,
    robust_min_required: float = 0.0,
    eta_cap: float = 0.0,
) -> dict:
    out = dict(r)
    det = r["det"]
    combos = r["combos"]
    if det["sc_min_dist"] >= margin or r["sc_n_stopped"] == 0:
        out["status"] = "already_clean"
        out["best"] = None
        return out

    def pick(cands):
        if robust_radius > 0:
            best = max(
                cands,
                key=lambda c: (_neighborhood_min_clearance(c, combos, robust_radius), c["total"]),
            )
            out["robust_min_clr"] = round(
                _neighborhood_min_clearance(best, combos, robust_radius), 3
            )
            return best
        return max(cands, key=lambda c: c["total"])

    clean = [
        c for c in combos if not c["static_crossing"] and not c["rb_cross"] and not c["stopped"]
    ]
    if eta_cap > 0:
        # Grid-edge labels (|eta| = 1) are UNREACHABLE by Beta-mean heads
        # (the original cliff lesson) and their one-sided neighborhoods
        # look spuriously robust — exclude edge combos from CANDIDATES.
        # Neighborhood scoring still sees all combos (plateau check intact).
        clean = [
            c
            for c in clean
            if abs(c["eta_lat"]) <= eta_cap + 1e-6 and abs(c["eta_col"]) <= eta_cap + 1e-6
        ]
    at_margin = [c for c in clean if c["sc_min_dist"] >= margin]
    if at_margin:
        out["status"] = "solved"
        out["best"] = pick(at_margin)
    else:
        improved = [c for c in clean if c["sc_min_dist"] >= det["sc_min_dist"] + min_gain]
        if improved:
            out["status"] = "solved"
            # No combo reaches the margin: take the one with max clearance
            # (reward as tiebreak) — partial-but-real improvement.
            best_clr = max(c["sc_min_dist"] for c in improved)
            out["best"] = pick([c for c in improved if c["sc_min_dist"] >= best_clr - 0.05])
        else:
            out["status"] = "unsolved"
            out["best"] = None
    # Cliffy-scene exclusion (BOTH solved branches): a "solution" whose
    # eta-neighbourhood contains a crossing is unlearnable by regression
    # (the policy's eta error lands on the cliff) — exclude from training
    # like unsolved scenes instead of teaching a target it cannot hit.
    if (
        out["status"] == "solved"
        and robust_radius > 0
        and robust_min_required > 0
        and out.get("robust_min_clr", 0.0) < robust_min_required
    ):
        out["status"] = "unsolved"
        out["best"] = None
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", required=True, nargs="+")
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--min_gain", type=float, default=0.3)
    parser.add_argument(
        "--robust_radius",
        type=float,
        default=0.0,
        help="eta-neighbourhood radius for plateau-centred "
        "(cliff-avoiding) label selection; 0 = argmax "
        "reward (original behaviour). Grid step is "
        "0.25, so 0.25 = adjacent combos, 0.5 = 2 steps.",
    )
    parser.add_argument(
        "--robust_min_required",
        type=float,
        default=0.0,
        help="exclude (mark unsolved) solved scenes whose "
        "best label's neighbourhood-min clearance is "
        "below this — cliff-edge scenes are unlearnable "
        "by regression",
    )
    parser.add_argument(
        "--eta_cap",
        type=float,
        default=0.0,
        help="exclude grid-edge combos (|eta| > cap) from "
        "label CANDIDATES — edge labels are unreachable "
        "by Beta-mean heads and spuriously robust. "
        "0 = off (original behaviour). Suggested 0.75.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scenes, seen = [], set()
    for lp in args.labels:
        with open(lp) as f:
            data = json.load(f)
        for r in data["scenes"]:
            if r["scene_path"] in seen:
                continue
            seen.add(r["scene_path"])
            scenes.append(
                relabel_scene(
                    r,
                    args.margin,
                    args.min_gain,
                    args.robust_radius,
                    args.robust_min_required,
                    eta_cap=args.eta_cap,
                )
            )

    n = {"solved": 0, "already_clean": 0, "unsolved": 0}
    for r in scenes:
        n[r["status"]] += 1
    summary = {
        "margin": args.margin,
        "min_gain": args.min_gain,
        "robust_radius": args.robust_radius,
        "n_scenes": len(scenes),
        **{f"n_{k}": v for k, v in n.items()},
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "scenes": scenes}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
