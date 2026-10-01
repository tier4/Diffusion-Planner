"""Score a finished closed-loop run with the per-label scenario metrics.

Reads the run directory ``run_all_groups_closed_loop.py`` wrote for one manifest
(``<out_root>/<timestamp>/<manifest name>/``) plus the manifest itself, and writes

- ``<group>/scenario_metrics.jsonl``: one row per (window, anchor);
- ``scenario_summary.json``: per group, the scored / passed / not-applicable counts,
  ``pass_rate`` and ``success_rate_percent`` (``pass_rate * 100``; both None when nothing
  was scored), and ``values``: for each metric value key, its mean over the *scored* rows
  (``passed`` not None) that carry it, as ``{"mean", "n", "n_nonfinite"}``. Not-applicable
  rows are left out because they can carry partial values (e.g. a lateral error measured
  before the metric gave up); non-finite values (inf/nan) are skipped and counted in
  ``n_nonfinite``, and ``mean`` is None when no finite value is left.

``summary_log_dict`` flattens that summary into ``scenario_based_closed_loop/<label>/<key>``
scalars, mirroring the open-loop ``scenario_based_open_loop/<label>/<key>`` namespace.

The manifest is the grouped closed-loop input (``label -> [window dir]``). Each window
directory holds the window's recorded frames plus a ``scenario.json``:

    {"eval_label": "vehicle_yield",
     "anchors": [{"timestamp": <ns>, "frame_offset": <anchor's index in the window>,
                  "span_frame_start": <index>, "span_frame_stop": <exclusive index>}]}

``span_frame_*`` (the label's event span, optional) sets the interval the geometry metrics
score over. A rollout trace is matched to its window directory through the same route keys
the evaluator used (``enumerate_multi_root_routes`` over the group's entries), so no naming
convention is reconstructed here.

    python -m scenario_generation.score_scenarios \\
        --run_dir <out_root>/<timestamp>/close_loop_scenario \\
        --manifest /path/to/close_loop_scenario.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scenario_generation.closed_loop_eval import enumerate_multi_root_routes
from scenario_generation.scenario_metrics import score
from scenario_generation.scenario_metrics.loader import load_input_from_frames

# Summary fields copied into the log dict as-is (``pass_rate`` is left out: it is
# ``success_rate_percent / 100``).
_LOGGED_SUMMARY_KEYS = ("success_rate_percent", "n_anchors", "n_scored", "n_not_applicable")
# Value keys that echo a metric parameter (the same number on every row), so their mean is
# not a measurement; kept in ``scenario_summary.json`` but not logged.
_PARAMETER_VALUE_KEYS = frozenset(
    {"threshold_m", "tolerance_m", "horizon_s", "reach_m", "position_tolerance_m"}
)


def _windows(entries: list[str]) -> dict[str, tuple[list[Path], Path, str, list[dict]]]:
    """``route key -> (npz paths, window dir, label, anchors)`` for a group's window dirs."""
    routes, route_root = enumerate_multi_root_routes(entries)
    by_key = {}
    for key, paths in routes.items():
        window_dir = Path(route_root[key])
        meta = json.loads((window_dir / "scenario.json").read_text())
        by_key[key] = (paths, window_dir, meta["eval_label"], meta["anchors"])
    return by_key


def score_group(group_dir: Path, entries: list[str], config=None) -> list[dict]:
    """Score every (window, anchor) of one group. ``config`` carries the thresholds shared
    with open loop (``ScenarioOpenLoopConfig`` or an args object with its fields); None
    uses its defaults."""
    windows = _windows(entries)
    rows = []
    for seg_file in sorted(group_dir.glob("segments*.jsonl")):
        for line in seg_file.read_text().splitlines():
            seg = json.loads(line)
            key, (start, end) = seg["route"], seg["segment"]
            paths, window_dir, label, anchors = windows[key]
            trace = group_dir / f"{key}_{start}_{end}.rollout.jsonl"
            for anchor in anchors:
                base = {
                    "route": key,
                    "segment": [start, end],
                    "window": str(window_dir),
                    "anchor": anchor,
                }
                if not trace.is_file():
                    rows.append(
                        {
                            **base,
                            "metric": "none",
                            "passed": None,
                            "reason": f"missing trace {trace.name}",
                        }
                    )
                    continue
                span = (
                    (int(anchor["span_frame_start"]), int(anchor["span_frame_stop"]) - 1)
                    if "span_frame_start" in anchor
                    else None
                )
                inp = load_input_from_frames(
                    trace,
                    paths,
                    window_dir,
                    label=label,
                    anchor_frame=int(anchor["frame_offset"]),
                    span_frames=span,
                )
                rows.append({**base, **score(inp, config).to_json()})
    return rows


def summarize(rows: list[dict]) -> dict:
    verdicts = Counter(
        "na" if r["passed"] is None else ("pass" if r["passed"] else "fail") for r in rows
    )
    scored = verdicts["pass"] + verdicts["fail"]
    finite, n_nonfinite = defaultdict(list), Counter()
    for r in rows:
        if r["passed"] is None:
            continue
        for key, v in r.get("values", {}).items():
            if math.isfinite(v):
                finite[key].append(v)
            else:
                n_nonfinite[key] += 1
    values = {
        key: {
            "mean": sum(finite[key]) / len(finite[key]) if finite[key] else None,
            "n": len(finite[key]),
            "n_nonfinite": n_nonfinite[key],
        }
        for key in sorted(finite.keys() | n_nonfinite.keys())
    }
    return {
        "n_anchors": len(rows),
        "n_scored": scored,
        "n_pass": verdicts["pass"],
        "n_fail": verdicts["fail"],
        "n_not_applicable": verdicts["na"],
        "pass_rate": verdicts["pass"] / scored if scored else None,
        "metrics": sorted({r["metric"] for r in rows}),
        "not_applicable_reasons": dict(Counter(r["reason"] for r in rows if r["passed"] is None)),
        "success_rate_percent": 100.0 * verdicts["pass"] / scored if scored else None,
        "values": values,
    }


def summary_log_dict(summary: dict[str, dict]) -> dict[str, float]:
    """Flatten ``{label: summarize(...)}`` into W&B-ready scalars.

    Keys are ``scenario_based_closed_loop/<label>/<key>``: the ``_LOGGED_SUMMARY_KEYS``
    fields plus each value's ``mean`` under the value key itself (as open loop logs
    ``average_lateral_error_m``). None entries and ``_PARAMETER_VALUE_KEYS`` are dropped.
    """
    log = {}
    for label, s in summary.items():
        prefix = f"scenario_based_closed_loop/{label}"
        for key in _LOGGED_SUMMARY_KEYS:
            if s.get(key) is not None:
                log[f"{prefix}/{key}"] = float(s[key])
        for key, agg in s.get("values", {}).items():
            if key not in _PARAMETER_VALUE_KEYS and agg["mean"] is not None:
                log[f"{prefix}/{key}"] = float(agg["mean"])
    return log


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory of one manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="The label -> window-dir manifest that was evaluated",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    summary = {}
    for group, entries in sorted(manifest.items()):
        group_dir = args.run_dir / group
        if not group_dir.is_dir():
            continue
        rows = score_group(group_dir, entries)
        with open(group_dir / "scenario_metrics.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[group] = summarize(rows)
        s = summary[group]
        print(
            f"{group:20s} scored={s['n_scored']:4d} pass={s['n_pass']:4d} na={s['n_not_applicable']:4d} pass_rate={s['pass_rate']}"
        )
    (args.run_dir / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
