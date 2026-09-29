"""Score a finished closed-loop run with the per-label scenario metrics.

Reads the run directory ``run_all_groups_closed_loop.py`` wrote for one manifest
(``<out_root>/<timestamp>/<manifest name>/``) plus the manifest itself, and writes

- ``<group>/scenario_metrics.jsonl``: one row per (window, anchor);
- ``scenario_summary.json``: per-group scored / passed / not-applicable counts.

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
from collections import Counter
from pathlib import Path

from scenario_generation.closed_loop_eval import enumerate_multi_root_routes
from scenario_generation.scenario_metrics import score
from scenario_generation.scenario_metrics.loader import load_input_from_frames


def _windows(entries: list[str]) -> dict[str, tuple[list[Path], Path, str, list[dict]]]:
    """``route key -> (npz paths, window dir, label, anchors)`` for a group's window dirs."""
    routes, route_root = enumerate_multi_root_routes(entries)
    by_key = {}
    for key, paths in routes.items():
        window_dir = Path(route_root[key])
        meta = json.loads((window_dir / "scenario.json").read_text())
        by_key[key] = (paths, window_dir, meta["eval_label"], meta["anchors"])
    return by_key


def score_group(group_dir: Path, entries: list[str]) -> list[dict]:
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
                rows.append({**base, **score(inp).to_json()})
    return rows


def summarize(rows: list[dict]) -> dict:
    verdicts = Counter(
        "na" if r["passed"] is None else ("pass" if r["passed"] else "fail") for r in rows
    )
    scored = verdicts["pass"] + verdicts["fail"]
    return {
        "n_anchors": len(rows),
        "n_scored": scored,
        "n_pass": verdicts["pass"],
        "n_fail": verdicts["fail"],
        "n_not_applicable": verdicts["na"],
        "pass_rate": verdicts["pass"] / scored if scored else None,
        "metrics": sorted({r["metric"] for r in rows}),
        "not_applicable_reasons": dict(Counter(r["reason"] for r in rows if r["passed"] is None)),
    }


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
