#!/usr/bin/env python3
"""Statistical A/B for tracker changes (mpc vs mpc_batched mining runs).

The batched solver is not bit-identical, so this reports the M2 validation
protocol metrics instead of a bitwise diff:
- mined event-window counts per label (accept: within ±2% or ±1 event),
- credit-row overlap Jaccard on (event_key, scene_file) (accept: ≥ 0.97),
- simulated/skipped chunk counts (must match),
- segment termination-reason histogram.

Usage: compare_tracker_ab.py <dir_A> <dir_B>
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_key(row: dict) -> tuple:
    # window_dir basename encodes route/start/frame/label without the run dir.
    return (Path(row.get("window_dir", "")).name, Path(row.get("scene_path", "")).name)


def label_counts(rows: list[dict]) -> Counter:
    windows = {}
    for r in rows:
        windows[Path(r.get("window_dir", "")).name] = r.get("label") or r.get("credit_label")
    return Counter(windows.values())


def main() -> int:
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a_rows = load_jsonl(a_dir / "credit_windows.jsonl")
    b_rows = load_jsonl(b_dir / "credit_windows.jsonl")
    a_sum = json.loads((a_dir / "summary.json").read_text())
    b_sum = json.loads((b_dir / "summary.json").read_text())

    ok = True
    for f in ("simulated_chunks", "skipped_chunks"):
        match = a_sum[f] == b_sum[f]
        ok &= match
        print(f"{f}: A={a_sum[f]} B={b_sum[f]} {'OK' if match else 'MISMATCH'}")

    ca, cb = label_counts(a_rows), label_counts(b_rows)
    for label in sorted(set(ca) | set(cb)):
        na, nb = ca.get(label, 0), cb.get(label, 0)
        tol = max(1, round(0.02 * max(na, nb)))
        match = abs(na - nb) <= tol
        ok &= match
        print(f"events[{label}]: A={na} B={nb} (tol ±{tol}) {'OK' if match else 'FAIL'}")

    ka, kb = {event_key(r) for r in a_rows}, {event_key(r) for r in b_rows}
    union = ka | kb
    jac = len(ka & kb) / len(union) if union else 1.0
    match = jac >= 0.97
    ok &= match
    print(f"credit-row Jaccard: {jac:.4f} (need >= 0.97) {'OK' if match else 'FAIL'}")

    for tag, path in (("A", a_dir), ("B", b_dir)):
        segs = load_jsonl(path / "segments.jsonl")
        term = Counter(str(r.get("terminated")) for r in segs)
        print(f"terminations[{tag}]: {dict(sorted(term.items()))}")

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
