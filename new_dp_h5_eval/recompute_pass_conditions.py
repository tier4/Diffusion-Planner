"""Recompute closed-loop pass/fail fields from saved segment metrics.

This does not load a model or rerun a rollout. It applies a pass-condition YAML
to an existing native-H5 closed-loop result tree, updates every group summary and
rebuilds its collection/root ``groups.json`` aggregates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from diffusion_planner.config.closed_loop_config import ClosedLoopPassConditionGroups


def _passed(row: dict, condition) -> bool:
    if condition.collision and int(row.get("object", {}).get("collision_count", 0)) > 0:
        return False
    if condition.road_border and int(row.get("road_border", {}).get("collision_count", 0)) > 0:
        return False
    if condition.red_light_violation and int(row.get("red_light_violation", {}).get("count", 0)) > 0:
        return False
    if condition.strong_brake and int(row.get("strong_brake", {}).get("count", 0)) > 0:
        return False
    if condition.snap and int(row.get("reproducer", {}).get("snap_count", 0)) > 0:
        return False
    return not condition.goal_reach or row.get("terminated") == "goal"


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=float) + "\n")


def _write_groups_manifest(out_dir: Path, summaries: dict[str, dict]) -> None:
    n_segments = sum(int(summary.get("n_segments", 0) or 0) for summary in summaries.values())
    total_steps = sum(int(summary.get("total_steps", 0) or 0) for summary in summaries.values())
    route_completion = sum(
        float(summary.get("mean_route_completion", 0.0) or 0.0)
        * int(summary.get("n_segments", 0) or 0)
        for summary in summaries.values()
    )
    dev_num = dev_steps = 0
    for summary in summaries.values():
        value = summary.get("mean_gt_deviation_m")
        steps = int(summary.get("total_steps", 0) or 0)
        if value is not None and math.isfinite(float(value)) and steps:
            dev_num += float(value) * steps
            dev_steps += steps
    payload = {
        "n_groups": len(summaries),
        "n_segments": n_segments,
        "total_steps": total_steps,
        "mean_route_completion": route_completion / n_segments if n_segments else 0.0,
        "mean_gt_deviation_m": dev_num / dev_steps if dev_steps else float("inf"),
        "total_curb_hits": sum(
            int(s.get("road_border", {}).get("collision_count", 0) or 0) for s in summaries.values()
        ),
        "total_snaps": sum(
            int(s.get("reproducer", {}).get("snap_count", 0) or 0) for s in summaries.values()
        ),
        "total_red_light_violations": sum(
            int(s.get("red_light_violation", {}).get("count", 0) or 0) for s in summaries.values()
        ),
        "total_strong_brakes": sum(
            int(s.get("strong_brake", {}).get("count", 0) or 0) for s in summaries.values()
        ),
        "n_segments_diverged": sum(
            int(s.get("n_segments_diverged", 0) or 0) for s in summaries.values()
        ),
        "n_pass_segments": sum(int(s.get("pass_count", 0) or 0) for s in summaries.values()),
        "n_fail_segments": sum(int(s.get("fail_count", 0) or 0) for s in summaries.values()),
    }
    payload["pass_rate"] = payload["n_pass_segments"] / n_segments if n_segments else 0.0
    _write_json(out_dir / "groups.json", payload)


def _group_summary_paths(result_root: Path) -> list[Path]:
    return sorted(
        path
        for path in result_root.rglob("summary.json")
        if (path.parent / "segments.jsonl").is_file()
    )


def recompute(result_root: Path, pass_conditions: Path, *, dry_run: bool = False) -> list[tuple[str, int, int]]:
    """Update pass fields below one closed-loop output root and return group counts."""
    result_root = result_root.resolve()
    conditions = ClosedLoopPassConditionGroups.from_yaml(pass_conditions)
    summaries: dict[Path, dict] = {}
    changes: list[tuple[str, int, int]] = []

    for summary_path in _group_summary_paths(result_root):
        group_dir = summary_path.parent
        condition = conditions.get_condition(group_dir.name)
        rows = _load_rows(group_dir / "segments.jsonl")
        for row in rows:
            row["passed"] = _passed(row, condition)
        refreshed = json.loads(summary_path.read_text(encoding="utf-8"))
        refreshed["pass_count"] = sum(row["passed"] for row in rows)
        refreshed["fail_count"] = len(rows) - refreshed["pass_count"]
        refreshed["pass_rate"] = refreshed["pass_count"] / len(rows) if rows else 0.0
        refreshed["pass_condition"] = condition.to_dict()
        summaries[group_dir] = refreshed
        changes.append((str(group_dir.relative_to(result_root)), refreshed["pass_count"], refreshed["n_segments"]))

        if not dry_run:
            _write_rows(group_dir / "segments.jsonl", rows)
            for shard_path in sorted(group_dir.glob("segments_[0-9]*.jsonl")):
                shard_rows = _load_rows(shard_path)
                for row in shard_rows:
                    row["passed"] = _passed(row, condition)
                _write_rows(shard_path, shard_rows)
            _write_json(summary_path, refreshed)

    if not summaries:
        raise FileNotFoundError(f"No group summaries with segments.jsonl below {result_root}")
    if not dry_run:
        collections: dict[Path, dict[str, dict]] = {}
        for group_dir, summary in summaries.items():
            collections.setdefault(group_dir.parent, {})[group_dir.name] = summary
        for collection_dir, collection_summaries in collections.items():
            _write_groups_manifest(collection_dir, collection_summaries)
        _write_groups_manifest(
            result_root,
            {
                str(group_dir.relative_to(result_root)): summary
                for group_dir, summary in summaries.items()
            },
        )
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path, help="One closed_loop/YYYYMMDD_HHMM output directory")
    parser.add_argument("--pass-conditions", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name, passed, total in recompute(
        args.result_root, args.pass_conditions, dry_run=args.dry_run
    ):
        print(f"{name}: {passed}/{total} ({passed / total:.1%})")
    print("Dry run: no files changed." if args.dry_run else "Pass summaries updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
