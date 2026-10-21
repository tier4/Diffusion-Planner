"""The suite-level ``summary.json`` of one scenario_sim run, counted over what was submitted.

``aggregate`` only sees the cases that wrote a ``row.json``. A case the driver killed at its
time limit writes none, so rolling up rows alone drops it from the denominator and the pass
count then depends on how loaded the machine was. Here the denominator is the submitted list,
and a case without a row is a non-pass that the summary names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import aggregate, segment_row_for_json
from scenario_generation.scenario_sim_viewer_export import (
    is_passed,
    load_submitted,
    read_verdict,
)

# The reference they need does not exist on this path: no recorded drive, no resolved route.
_UNMEASURED_KEYS = ("mean_route_completion", "mean_gt_deviation_m", "mean_centerline_dist_m")


def _timed_out_keys(run_dir: Path) -> set[str]:
    path = run_dir / "timeouts.txt"
    if not path.is_file():
        return set()
    return {
        line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def summarize_suite(
    run_dir: Path, near_miss_thresh: float = 1.0
) -> tuple[dict[str, Any], list[dict]]:
    """``(summary, rows)`` for ``run_dir``; ``rows`` are the cases that produced one."""
    rows: list[dict] = []
    passed = 0
    keys_with_row: set[str] = set()
    for row_path in sorted(run_dir.glob("*/row.json")):
        try:
            row = segment_row_for_json(json.loads(row_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        rows.append(row)
        keys_with_row.add(row_path.parent.name)
        passed += is_passed(row, read_verdict(row_path.parent))

    submitted = [key for key, _ in load_submitted(run_dir)]
    # Without a manifest the rows are all there is to count.
    n_submitted = len(submitted) if submitted else len(rows)
    timed_out = _timed_out_keys(run_dir) - keys_with_row

    summary = aggregate(rows, near_miss_thresh) if rows else {}
    for key in _UNMEASURED_KEYS:
        if key in summary:
            summary[key] = None
    summary.update(
        {
            "n_submitted": n_submitted,
            "n_rows": len(rows),
            "n_timeout": len(timed_out),
            "n_pass": passed,
            "pass_rate": passed / n_submitted if n_submitted else None,
        }
    )
    return summary, rows
