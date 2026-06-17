#!/usr/bin/env python3
"""Check L2 eval results from existing log files.

Usage:
    python -m rlvr.autoresearch.tools.check_l2_results <dir>

Scans <dir> for l2_*.log files and prints a summary table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BASELINE_EGO = 5.388
BASELINE_NEIGH = 4.393


def parse_log(logfile: Path) -> dict | None:
    text = logfile.read_text()
    ego_m = re.search(r"avg_loss_ego=([0-9.]+)", text)
    neigh_m = re.search(r"avg_loss_neighbor=([0-9.]+)", text)
    if not ego_m or not neigh_m:
        return None
    return {"ego": float(ego_m.group(1)), "neigh": float(neigh_m.group(1))}


def fmt_pct(val: float, baseline: float) -> str:
    return f"{(val / baseline - 1) * 100:+.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Check L2 eval results from log files")
    parser.add_argument("dir", type=Path, help="Directory to scan for l2_*.log files")
    parser.add_argument("--baseline_ego", type=float, default=BASELINE_EGO)
    parser.add_argument("--baseline_neigh", type=float, default=BASELINE_NEIGH)
    args = parser.parse_args()

    logs = sorted(args.dir.glob("l2_*.log"))
    if not logs:
        print(f"No l2_*.log files found in {args.dir}")
        return

    print(f"L2 Results (baseline: ego={args.baseline_ego}, neigh={args.baseline_neigh})")
    print("=" * 72)
    print(f"{'Model':<40s} {'Ego':>8s} {'Ego%':>8s} {'Neigh':>8s} {'Neigh%':>8s}")
    print("-" * 72)

    for log in logs:
        stem = log.stem.removeprefix("l2_")
        result = parse_log(log)
        if result:
            ego_pct = fmt_pct(result["ego"], args.baseline_ego)
            neigh_pct = fmt_pct(result["neigh"], args.baseline_neigh)
            print(
                f"{stem:<40s} {result['ego']:>8.4f} {ego_pct:>8s} {result['neigh']:>8.4f} {neigh_pct:>8s}"
            )

    print("=" * 72)


if __name__ == "__main__":
    main()
