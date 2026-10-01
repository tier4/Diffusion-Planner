"""Build a closed-loop route manifest from an ML-Planner route-shard Parquet index."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scenario_generation.reproducer_rollout import DT

from .run_all_groups_closed_loop import _load_groups


def build_manifest(
    index: Path,
    output: Path,
    *,
    group_column: str = "area_map_id",
    selection: Path | None = None,
) -> dict[str, list[dict]]:
    """Use indexed H5 frames; an optional grouped selection supplies labels and time spans."""
    index = index.resolve()
    output = output.resolve()
    table = pq.read_table(index)
    required = {"h5_path", "frame_index", "frame_time_ns"}
    if selection is None:
        required.update({group_column, "route_group_id"})
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"H5 index missing columns: {sorted(missing)}")

    by_path: dict[Path, list[dict]] = defaultdict(list)
    for row in table.to_pylist():
        path = Path(row["h5_path"])
        path = (path if path.is_absolute() else index.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        by_path[path].append(row)

    def indexed_times(path: Path) -> np.ndarray:
        rows = sorted(by_path[path], key=lambda row: int(row["frame_index"]))
        indices = [int(row["frame_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"{path} index must cover every frame in order")
        times = np.array([int(row["frame_time_ns"]) for row in rows], dtype=np.int64)
        if len(times) > 1 and not np.all(np.abs(np.diff(times) - round(DT * 1e9)) <= 1_000_000):
            raise ValueError(f"non-contiguous 0.1 s closed-loop frames in {path}")
        return times

    def indexed_path(value: str, relative_to: Path) -> Path:
        path = Path(value)
        path = (path if path.is_absolute() else relative_to / path).resolve()
        if path not in by_path:
            raise ValueError(f"H5 shard is not in the index: {value}")
        return path

    groups: dict[str, list[dict]] = defaultdict(list)
    if selection is None:
        for path, rows in by_path.items():
            indexed_times(path)
            names = {str(row[group_column]) for row in rows}
            if len(names) != 1:
                raise ValueError(f"{path} has inconsistent {group_column} values")
            groups[names.pop()].append({"h5_path": os.path.relpath(path, output.parent)})
    else:
        selection = selection.resolve()
        choices = json.loads(selection.read_text(encoding="utf-8"))
        if not isinstance(choices, dict):
            raise ValueError("selection must map group names to H5 route entries")
        for name, entries in choices.items():
            if not isinstance(entries, list):
                raise ValueError(f"selection group {name!r} must be a list")
            for value in entries:
                route = {"h5_path": value} if isinstance(value, str) else dict(value)
                path = indexed_path(route["h5_path"], selection.parent)
                times = indexed_times(path)
                if "segment_start_ns" in route:
                    route["frame_start"] = int(
                        np.searchsorted(times, int(route["segment_start_ns"]))
                    )
                if "segment_end_ns" in route:
                    route["frame_stop"] = int(
                        np.searchsorted(times, int(route["segment_end_ns"]), side="right")
                    )
                start, stop = (
                    int(route.get("frame_start", 0)),
                    int(route.get("frame_stop", len(times))),
                )
                if not 0 <= start < stop <= len(times):
                    raise ValueError(
                        f"empty or invalid selected window in {path}: [{start}, {stop})"
                    )
                route["h5_path"] = os.path.relpath(path, output.parent)
                groups[str(name)].append(route)

    result = {name: routes for name, routes in sorted(groups.items())}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _load_groups(temporary)  # Reject duplicate route identities before publication.
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="ML-Planner route-shard Parquet index")
    parser.add_argument("output", type=Path, help="evaluator manifest to write")
    parser.add_argument("--group-column", default="area_map_id")
    parser.add_argument(
        "--selection", type=Path, help="optional grouped H5 paths, spans, and anchors"
    )
    args = parser.parse_args()
    build_manifest(
        args.index, args.output, group_column=args.group_column, selection=args.selection
    )


if __name__ == "__main__":
    main()
