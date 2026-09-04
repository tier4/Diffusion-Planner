#!/usr/bin/env python3
"""Statistical A/B for tracker changes (mpc vs mpc_batched mining runs).

The batched solver is not bit-identical, and a closed-loop rollout amplifies
per-tick 1e-3-level solver differences over hundreds of steps — so exact
frame-keyed comparison is the wrong ruler. The gates:
- simulated/skipped chunk counts must match,
- mined event-window counts per label within ±2% or ±1 event,
- EVENT-level match: every window paired by (route, chunk_start, label) —
  accept when ≥ 95% of windows pair up and the paired offense-frame shift is
  small (median ≤ 3 frames = 0.3 s); the exact-key Jaccard is reported as
  information only,
- segment termination-reason histogram (reported).

Usage: compare_tracker_ab.py <dir_A> <dir_B>
"""

import argparse
import json
import re
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_a", type=Path, help="mining output dir of run A (reference)")
    ap.add_argument("dir_b", type=Path, help="mining output dir of run B (candidate)")
    args = ap.parse_args()
    a_dir, b_dir = args.dir_a, args.dir_b
    for d in (a_dir, b_dir):
        if not (d / "summary.json").exists():
            ap.error(f"{d} is not a mining output dir (missing summary.json)")
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
    print(f"exact credit-row Jaccard (informational): {jac:.4f}")

    # Event-level pairing: (route, chunk_start, label) + nearest offense frame.
    def parse_windows(rows: list[dict]) -> list[tuple]:
        out = []
        seen = set()
        for r in rows:
            wd = Path(r.get("window_dir", "")).name
            if wd in seen:
                continue
            seen.add(wd)
            m = re.match(r"(.+?)_(\d+)_(\d+)_(?:danger|event|credit)_(.+)$", wd)
            if not m:
                raise ValueError(f"unparseable window dir name: {wd}")
            out.append((m.group(1), int(m.group(2)), m.group(4), int(m.group(3))))
        return out

    wa, wb = parse_windows(a_rows), parse_windows(b_rows)
    used: set[int] = set()
    deltas: list[int] = []
    for route, start, label, frame in wa:
        best = None
        for i, (rb, sb, lb, fb) in enumerate(wb):
            if i in used or (rb, sb, lb) != (route, start, label):
                continue
            d = abs(fb - frame)
            if best is None or d < best[1]:
                best = (i, d)
        if best is not None:
            used.add(best[0])
            deltas.append(best[1])
    denom = max(len(wa), len(wb))
    frac = len(deltas) / denom if denom else 1.0
    med = sorted(deltas)[len(deltas) // 2] if deltas else 0
    match = frac >= 0.95 and med <= 3
    ok &= match
    print(
        f"event-level match: {len(deltas)}/{denom} paired ({frac:.2%}, need >= 95%), "
        f"offense-frame |delta| median={med} (need <= 3) max={max(deltas) if deltas else 0} "
        f"{'OK' if match else 'FAIL'}"
    )

    # Termination reasons and rollout lengths are GATED, not informational: a
    # tracker can match chunk/event counts while materially changing how or
    # how long segments run, and that must not read as PASS.
    a_segs = load_jsonl(a_dir / "segments.jsonl")
    b_segs = load_jsonl(b_dir / "segments.jsonl")
    term_a = Counter(str(r.get("terminated")) for r in a_segs)
    term_b = Counter(str(r.get("terminated")) for r in b_segs)
    for reason in sorted(set(term_a) | set(term_b)):
        na, nb = term_a.get(reason, 0), term_b.get(reason, 0)
        tol = max(1, round(0.02 * max(na, nb)))
        match = abs(na - nb) <= tol
        ok &= match
        print(f"terminations[{reason}]: A={na} B={nb} (tol ±{tol}) {'OK' if match else 'FAIL'}")
    steps_a = sum(int(r.get("n_steps_run", 0)) for r in a_segs)
    steps_b = sum(int(r.get("n_steps_run", 0)) for r in b_segs)
    steps_tol = max(1, round(0.02 * max(steps_a, steps_b)))
    match = abs(steps_a - steps_b) <= steps_tol
    ok &= match
    print(
        f"total simulated steps: A={steps_a} B={steps_b} (tol ±{steps_tol}) "
        f"{'OK' if match else 'FAIL'}"
    )

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
