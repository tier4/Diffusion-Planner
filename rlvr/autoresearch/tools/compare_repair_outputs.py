#!/usr/bin/env python3
"""Correctness oracle for repair-stage perf work: compare two repair output dirs.

Compares repaired_targets.jsonl rows (keyed by output NPZ basename; path-like
fields ignored), the unrepaired lists, and every repaired NPZ array (bitwise by
default, --atol for float tolerance). Exit 0 = equivalent.

Usage: compare_repair_outputs.py <dir_A> <dir_B> [--atol 0]
"""

import argparse
import json
import sys
from pathlib import Path

from rlvr.autoresearch.tools.compare_mining_outputs import cmp_value, compare_npzs, load_jsonl

_PATH_FIELDS = {"scene_path", "source_scene_path", "window_dir"}


def compare_rows(name: str, a_rows, b_rows, rtol: float) -> list[str]:
    diffs = []
    a_by = {Path(r["scene_path"]).name: r for r in a_rows}
    b_by = {Path(r["scene_path"]).name: r for r in b_rows}
    only_a, only_b = set(a_by) - set(b_by), set(b_by) - set(a_by)
    if only_a:
        diffs.append(f"{name}: {len(only_a)} rows only in A, e.g. {sorted(only_a)[:3]}")
    if only_b:
        diffs.append(f"{name}: {len(only_b)} rows only in B, e.g. {sorted(only_b)[:3]}")
    for k in sorted(set(a_by) & set(b_by)):
        ra, rb = a_by[k], b_by[k]
        bad = [
            f
            for f in sorted(set(ra) | set(rb))
            if f not in _PATH_FIELDS and not cmp_value(ra.get(f), rb.get(f), rtol)
        ]
        if bad:
            diffs.append(f"{name} {k}: fields differ: {bad[:10]}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--float_fields_rtol", type=float, default=1e-6)
    args = ap.parse_args()

    diffs: list[str] = []
    diffs += compare_rows(
        "repaired",
        load_jsonl(args.dir_a / "repaired_targets.jsonl"),
        load_jsonl(args.dir_b / "repaired_targets.jsonl"),
        args.float_fields_rtol,
    )
    for unrep in ("repaired_targets_unrepaired.json",):
        pa, pb = args.dir_a / unrep, args.dir_b / unrep
        if pa.exists() or pb.exists():
            ra = json.loads(pa.read_text()) if pa.exists() else []
            rb = json.loads(pb.read_text()) if pb.exists() else []
            diffs += compare_rows(unrep, ra, rb, args.float_fields_rtol)
    diffs += compare_npzs(
        args.dir_a / "repaired_targets", args.dir_b / "repaired_targets", args.atol
    )

    if diffs:
        print(f"DIFFERENT ({len(diffs)} findings):")
        for d in diffs[:60]:
            print(" ", d)
        return 1
    print("IDENTICAL (within tolerance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
