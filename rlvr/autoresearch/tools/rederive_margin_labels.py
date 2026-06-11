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


def relabel_scene(r: dict, margin: float, min_gain: float) -> dict:
    out = dict(r)
    det = r["det"]
    combos = r["combos"]
    if det["sc_min_dist"] >= margin or r["sc_n_stopped"] == 0:
        out["status"] = "already_clean"
        out["best"] = None
        return out

    clean = [c for c in combos
             if not c["static_crossing"] and not c["rb_cross"] and not c["stopped"]]
    at_margin = [c for c in clean if c["sc_min_dist"] >= margin]
    if at_margin:
        out["status"] = "solved"
        out["best"] = max(at_margin, key=lambda c: c["total"])
    else:
        improved = [c for c in clean
                    if c["sc_min_dist"] >= det["sc_min_dist"] + min_gain]
        if improved:
            out["status"] = "solved"
            # No combo reaches the margin: take the one with max clearance
            # (reward as tiebreak) — partial-but-real improvement.
            best_clr = max(c["sc_min_dist"] for c in improved)
            out["best"] = max(
                [c for c in improved if c["sc_min_dist"] >= best_clr - 0.05],
                key=lambda c: c["total"],
            )
        else:
            out["status"] = "unsolved"
            out["best"] = None
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", required=True, nargs="+")
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--min_gain", type=float, default=0.3)
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
            scenes.append(relabel_scene(r, args.margin, args.min_gain))

    n = {"solved": 0, "already_clean": 0, "unsolved": 0}
    for r in scenes:
        n[r["status"]] += 1
    summary = {
        "margin": args.margin, "min_gain": args.min_gain,
        "n_scenes": len(scenes), **{f"n_{k}": v for k, v in n.items()},
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "scenes": scenes}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
