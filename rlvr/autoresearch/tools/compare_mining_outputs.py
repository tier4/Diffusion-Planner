#!/usr/bin/env python3
"""Correctness oracle for R2LPL mining-perf work: compare two mining output dirs.

Compares (a) credit_windows.jsonl rows (keyed by event-dir name + scene file),
(b) segments.jsonl result rows, (c) every saved NPZ array in windows/ (exact
match first, then allclose with reported max-abs-diff). Exit 0 = identical
within tolerance, 1 = differences found.

Usage: compare_mining_outputs.py <dir_A> <dir_B> [--atol 0] [--float_fields_rtol 1e-6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(row: dict) -> tuple:
    p = row.get("scene_path", "")
    return (Path(p).parent.name, Path(p).name)


def cmp_scalar(a, b, rtol: float) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if np.isnan(fa) and np.isnan(fb):
            return True
        return bool(np.isclose(fa, fb, rtol=rtol, atol=1e-12))
    return a == b


def cmp_value(a, b, rtol: float) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(cmp_value(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(cmp_value(x, y, rtol) for x, y in zip(a, b))
    return cmp_scalar(a, b, rtol)


def compare_jsonl(name: str, a_rows: list[dict], b_rows: list[dict], rtol: float) -> list[str]:
    diffs = []
    a_by, b_by = {row_key(r): r for r in a_rows}, {row_key(r): r for r in b_rows}
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
            # scene_path/window_dir embed the output dir name — not semantics.
            if f not in ("scene_path", "window_dir") and not cmp_value(ra.get(f), rb.get(f), rtol)
        ]
        if bad:
            diffs.append(f"{name} {k}: fields differ: {bad[:8]}")
    return diffs


def compare_npzs(a_root: Path, b_root: Path, atol: float) -> list[str]:
    diffs = []
    a_files = sorted(p.relative_to(a_root) for p in a_root.rglob("*.npz"))
    b_files = sorted(p.relative_to(b_root) for p in b_root.rglob("*.npz"))
    if a_files != b_files:
        only_a = set(a_files) - set(b_files)
        only_b = set(b_files) - set(a_files)
        diffs.append(
            f"npz sets differ: {len(only_a)} only-A {sorted(only_a)[:3]}, "
            f"{len(only_b)} only-B {sorted(only_b)[:3]}"
        )
    for rel in sorted(set(a_files) & set(b_files)):
        with (
            np.load(a_root / rel, allow_pickle=True) as za,
            np.load(b_root / rel, allow_pickle=True) as zb,
        ):
            if set(za.files) != set(zb.files):
                diffs.append(f"{rel}: key sets differ {set(za.files) ^ set(zb.files)}")
                continue
            for k in za.files:
                va, vb = za[k], zb[k]
                if va.shape != vb.shape or va.dtype != vb.dtype:
                    diffs.append(
                        f"{rel}[{k}]: shape/dtype {va.shape}/{va.dtype} vs {vb.shape}/{vb.dtype}"
                    )
                    continue
                if va.dtype.kind in "USO":
                    if not np.array_equal(va, vb):
                        diffs.append(f"{rel}[{k}]: string/object mismatch")
                    continue
                if np.array_equal(va, vb, equal_nan=True):
                    continue
                d = np.nanmax(np.abs(va.astype(np.float64) - vb.astype(np.float64)))
                if d > atol:
                    diffs.append(f"{rel}[{k}]: max|diff|={d:.3e} > atol={atol}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--atol", type=float, default=0.0, help="NPZ array tolerance (0 = bitwise)")
    ap.add_argument("--float_fields_rtol", type=float, default=1e-6)
    args = ap.parse_args()

    diffs: list[str] = []
    diffs += compare_jsonl(
        "credit_windows",
        load_jsonl(args.dir_a / "credit_windows.jsonl"),
        load_jsonl(args.dir_b / "credit_windows.jsonl"),
        args.float_fields_rtol,
    )
    a_seg = load_jsonl(args.dir_a / "segments.jsonl")
    b_seg = load_jsonl(args.dir_b / "segments.jsonl")
    if len(a_seg) != len(b_seg):
        diffs.append(f"segments.jsonl: {len(a_seg)} vs {len(b_seg)} rows")
    else:
        for i, (ra, rb) in enumerate(zip(a_seg, b_seg)):
            bad = [
                f
                for f in sorted(set(ra) | set(rb))
                if not cmp_value(ra.get(f), rb.get(f), args.float_fields_rtol)
            ]
            if bad:
                diffs.append(f"segments row {i}: fields differ: {bad[:8]}")
    if (args.dir_a / "windows").exists() or (args.dir_b / "windows").exists():
        diffs += compare_npzs(args.dir_a / "windows", args.dir_b / "windows", args.atol)

    if diffs:
        print(f"DIFFERENT ({len(diffs)} findings):")
        for d in diffs[:60]:
            print(" ", d)
        return 1
    print("IDENTICAL (within tolerance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
