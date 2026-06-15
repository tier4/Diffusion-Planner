#!/usr/bin/env python3
"""Group-aware train/holdout split for sweep-label files.

Splits by BASE SCENE identity (the ``scene_NNNN`` stem shared by every
perturbation variant and rolled state), so no base scene contributes labels
to both sides — the strict anti-leakage split for overfit detection. The
trainer's internal random split can place perturbation siblings of a train
scene into its early-stop val, which is fine for stopping but too soft to
measure generalization.

Writes, per input label file, ``<stem>_train.json`` next to it, plus ONE
combined holdout file with every held-out scene row (sweep_labels format,
directly consumable by the same eval code paths). Prints the composition
(groups, solved/clean counts per side) so the split can be sanity-checked.

Usage:
    python -m rlvr.autoresearch.tools.split_labels_holdout \
        --labels <labels1.json> <labels2.json> ... \
        --holdout_frac 0.15 --out_holdout <holdout.json> [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_BASE_RE = re.compile(r"(scene_\d+)")


def base_id(scene_path: str) -> str:
    m = _BASE_RE.search(Path(scene_path).stem)
    if not m:
        raise ValueError(f"no scene_NNNN stem in {scene_path}")
    return m.group(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", required=True, nargs="+")
    parser.add_argument("--holdout_frac", type=float, default=0.15)
    parser.add_argument("--out_holdout", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Collect every base id across ALL files, then deterministically assign
    # groups to the holdout by salted-hash order (stable across runs/files).
    groups: set[str] = set()
    loaded = []
    for lp in args.labels:
        with open(lp) as f:
            d = json.load(f)
        loaded.append((lp, d))
        for r in d["scenes"]:
            groups.add(base_id(r["scene_path"]))

    def rank(g: str) -> int:
        return int(hashlib.sha256(f"{args.seed}:{g}".encode()).hexdigest()[:8], 16)

    ordered = sorted(groups, key=rank)
    n_hold = max(1, int(len(ordered) * args.holdout_frac))
    hold = set(ordered[:n_hold])
    print(f"[split] {len(groups)} base-scene groups -> {len(hold)} held out ({sorted(hold)})")

    # Compute the full split FIRST; only write files after validation so a
    # failed run doesn't leave truncated *_train.json next to the originals.
    held_rows, pending = [], []
    for lp, d in loaded:
        tr = [r for r in d["scenes"] if base_id(r["scene_path"]) not in hold]
        hd = [r for r in d["scenes"] if base_id(r["scene_path"]) in hold]
        held_rows.extend(hd)
        out = dict(d)
        out["scenes"] = tr
        if "summary" in out:
            out["summary"] = dict(out["summary"], n_scenes=len(tr), holdout_removed=len(hd))
        out_path = str(Path(lp).with_name(Path(lp).stem + "_train.json"))
        pending.append((lp, out_path, out, tr, hd))

    n_solved = sum(1 for r in held_rows if r["status"] == "solved")
    if n_solved == 0:
        raise SystemExit(
            "holdout contains NO solved avoidance rows — "
            "useless for overfit detection; re-seed or raise "
            "frac (no files written)"
        )

    for lp, out_path, out, tr, hd in pending:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)

        def comp(rows):
            s = sum(1 for r in rows if r["status"] == "solved")
            c = sum(1 for r in rows if r["status"] == "already_clean")
            return f"{len(rows)} rows ({s} solved / {c} clean)"

        print(f"  {Path(lp).name}: train {comp(tr)} | holdout {comp(hd)}")

    with open(args.out_holdout, "w") as f:
        json.dump(
            {
                "summary": {"n_scenes": len(held_rows), "holdout_groups": sorted(hold)},
                "scenes": held_rows,
            },
            f,
            indent=1,
        )
    print(f"[split] holdout: {len(held_rows)} rows, {n_solved} solved -> {args.out_holdout}")


if __name__ == "__main__":
    main()
