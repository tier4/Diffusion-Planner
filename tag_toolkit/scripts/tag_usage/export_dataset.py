#!/usr/bin/env python3
"""Export dataset segments based on tag queries.

Frames are selected via a tag query against a :class:`tag_toolkit.TagStore`,
then organized and symlinked into an output directory.

Usage::

    # Export close-loop segments from a directory, grouped by devops_override_label value
    python export_dataset.py --base /path/to/dataset --output /data/out \\
        --mode close_loop --dimension devops_override_label

    # Export only frames where devops_override_label equals "a"
    python export_dataset.py --base /path/to/dataset --output /data/out \\
        --query "devops_override_label:a" --mode close_loop --dimension devops_override_label

    # Export only frames where devops_override_label equals "b"
    python export_dataset.py --base /path/to/dataset --output /data/out \\
        --query "devops_override_label:b" --mode close_loop --dimension devops_override_label

    # Use a pre-built SQLite index for faster startup on large datasets
    python export_dataset.py --db /data/tags.db --output /data/out \\
        --mode close_loop --dimension devops_override_label

The script accepts either a dataset source (directory, path-list JSON,
.npz) or a pre-built SQLite index. The database mode is the fast path
for large datasets.

Modes
-----
* ``open_loop`` — export individual frames matching the query.
* ``close_loop`` — group frames into segments with margin context around
  each tagged frame; symlink each segment into its own directory.

``--dimension`` adds a sub-directory layer (``<dim>/<value>/<route>/...``)
so multiple dimension values can be exported side-by-side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TAG_TOOLKIT_PARENT = _REPO_ROOT / "Diffusion-Planner" / "tag_toolkit"
if _TAG_TOOLKIT_PARENT.exists():
    sys.path.insert(0, str(_TAG_TOOLKIT_PARENT.parent))

from tag_toolkit import TagStore
from tag_toolkit.routes import extract_frame_number, route_of


def _merge_segments(
    frame_numbers: list[int],
    margin_before: int,
    margin_after: int,
) -> list[tuple[int, int]]:
    """Merge nearby frames into segments. Adjacent frames within
    ``margin_before + margin_after`` gap are folded into one segment."""
    if not frame_numbers:
        return []

    margin_total = margin_before + margin_after
    segments: list[tuple[int, int]] = []
    seg_start = seg_end = frame_numbers[0]

    for frame in frame_numbers[1:]:
        if frame - seg_end <= margin_total:
            seg_end = frame
        else:
            segments.append((seg_start, seg_end))
            seg_start = seg_end = frame
    segments.append((seg_start, seg_end))
    return segments


def _resolve_source_npz(src: Path) -> Path:
    """Map a frame path to its on-disk NPZ. The dataset has two layouts:
    ``<route>/routes/<file>.npz`` (closed-loop) and ``<route>/<file>.npz``
    (flat). Prefer the closed-loop ``routes/`` form when both exist."""
    routes_form = src.parent / "routes" / src.name
    return routes_form if routes_form.exists() else src
    """Map a frame path to its on-disk NPZ. The dataset has two layouts:
    ``<route>/routes/<file>.npz`` (closed-loop) and ``<route>/<file>.npz``
    (flat). Prefer the closed-loop ``routes/`` form when both exist."""
    routes_form = src.parent / "routes" / src.name
    return routes_form if routes_form.exists() else src


def _find_neighbor_npz(
    route: Path,
    frame_num: int,
    cache: dict[Path, dict[int, Path]],
) -> Path | None:
    """Locate a file in *route* with the given frame number. Filenames
    match ``<route>_<prefix>_<8 digits>.npz``; the prefix is opaque so we
    pre-scan the route once per call."""
    if route not in cache:
        cache[route] = {
            n: p
            for p in route.iterdir()
            if p.suffix == ".npz" and (n := extract_frame_number(p)) is not None
        }
    p = cache[route].get(frame_num)
    return _resolve_source_npz(p) if p is not None else None


def export_open_loop(
    store: TagStore,
    query: str,
    output_dir: Path,
    dimension: str | None,
) -> None:
    """Export frames for open-loop evaluation."""
    matching_frames = store.query(query, granularity="frame")
    if not matching_frames:
        print(f"No frames match query: {query}")
        return

    print(f"Found {len(matching_frames)} frames matching query")

    if dimension:
        dim_values = store.dim_values_for(matching_frames, dimension)
        groups: dict[str, list[str]] = defaultdict(list)
        for npz in matching_frames:
            val = dim_values.get(npz)
            groups[val if val is not None else "-"].append(str(npz))
        output_file = output_dir / f"open_loop_{dimension}.json"
        with open(output_file, "w") as f:
            json.dump(dict(groups), f, indent=2)
        print(f"Wrote {len(groups)} groups to {output_file}")
    else:
        output_file = output_dir / "open_loop.json"
        frame_list = [str(npz) for npz in sorted(matching_frames)]
        with open(output_file, "w") as f:
            json.dump(frame_list, f, indent=2)
        print(f"Wrote {len(frame_list)} frames to {output_file}")


def export_close_loop(
    store: TagStore,
    query: str,
    output_dir: Path,
    dimension: str | None,
    margin_before: int,
    margin_after: int,
) -> None:
    """Export segments for closed-loop evaluation."""
    matching_frames = store.query(query, granularity="frame")
    if not matching_frames:
        print(f"No frames match query: {query}")
        return

    print(f"Found {len(matching_frames)} frames matching query")

    # Group (frame_num, npz) by route. We keep the npz path (not just the
    # frame number) because the filename prefix isn't derivable from the
    # route alone — e.g. "r1_frame_00000002.npz" vs "r1_camera_00000002.npz".
    route_frames: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    for npz in matching_frames:
        frame_num = extract_frame_number(npz)
        if frame_num is None:
            continue
        route_frames[route_of(npz)].append((frame_num, npz))

    npz_dim_value = store.dim_values_for(matching_frames, dimension) if dimension else None

    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[str]] = defaultdict(list)
    neighbor_cache: dict[Path, dict[int, Path]] = {}

    for route, frames in sorted(route_frames.items(), key=lambda kv: str(kv[0])):
        frames.sort(key=lambda x: x[0])
        num_to_npz: dict[int, Path] = {f: p for f, p in frames}
        segments = _merge_segments([f for f, _ in frames], margin_before, margin_after)

        for seg_start, seg_end in segments:
            if dimension:
                # First frame in the segment that has a dimension value wins.
                dim_value = "-"
                for frame_num in range(seg_start, seg_end + 1):
                    npz = num_to_npz.get(frame_num)
                    if npz is None:
                        continue
                    v = npz_dim_value.get(npz)
                    if v is not None:
                        dim_value = v
                        break
                seg_dir = output_dir / dim_value / route.name / str(seg_start)
            else:
                seg_dir = output_dir / route.name / str(seg_start)

            seg_dir.mkdir(parents=True, exist_ok=True)

            # Symlink every frame in [seg_start - margin_before, seg_end + margin_after].
            # Try the fast-path lookup first (frame is in matching_frames); fall
            # back to probing the directory for the right filename prefix.
            linked_count = 0
            for frame_num in range(seg_start - margin_before, seg_end + margin_after + 1):
                if frame_num < 0:
                    continue
                src = num_to_npz.get(frame_num)
                if src is not None:
                    src = _resolve_source_npz(src)
                else:
                    src = _find_neighbor_npz(route, frame_num, neighbor_cache)
                if src is not None and src.exists():
                    dst = seg_dir / src.name
                    if not dst.exists():
                        os.symlink(src, dst)
                    linked_count += 1

                    sidecar_json = src.with_suffix(".json")
                    dst_json = seg_dir / sidecar_json.name
                    if sidecar_json.exists() and not dst_json.exists():
                        os.symlink(sidecar_json, dst_json)

            entry_dimension = dim_value if dimension else "unbucketed"
            buckets[entry_dimension].append(str(seg_dir.resolve()))
            print(f"  {seg_dir.relative_to(output_dir)}: {linked_count} frames ({entry_dimension})")

    buckets_file = output_dir / "close_loop.json"
    with open(buckets_file, "w") as f:
        json.dump(dict(sorted(buckets.items())), f, indent=2)
    total = sum(len(v) for v in buckets.values())
    print(f"Wrote {len(buckets)} scenarios / {total} segments to {buckets_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Export dataset segments based on tag queries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--base",
        type=Path,
        help=(
            "Dataset source: directory, path-list JSON, or single .npz. "
            "An in-memory TagStore is built at startup."
        ),
    )
    source_group.add_argument(
        "--db",
        type=Path,
        help=(
            "Path to a pre-built TagStore SQLite database "
            "(.db / .sqlite / .tags.db). Recommended for large datasets."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--query",
        help="Tag query clause. Defaults to '<dimension>:*' if --dimension is set but --query is not.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["open_loop", "close_loop"],
        help="Export mode: open_loop or close_loop.",
    )
    parser.add_argument(
        "--dimension",
        help="Dimension for sub-directory grouping and default query when --query is omitted.",
    )
    parser.add_argument(
        "--margin-before",
        type=int,
        default=50,
        help="Frames before each frame (close_loop only, default: 50).",
    )
    parser.add_argument(
        "--margin-after",
        type=int,
        default=50,
        help="Frames after each frame (close_loop only, default: 50).",
    )

    args = parser.parse_args()

    if args.query is None and args.dimension is None:
        print(
            "Error: at least one of --query or --dimension must be provided",
            file=sys.stderr,
        )
        return 1

    if args.query is None:
        args.query = f"{args.dimension}:*"

    if args.db is not None:
        if not args.db.exists():
            print(f"Error: TagStore database not found: {args.db}", file=sys.stderr)
            return 1
        print(f"Loading TagStore from {args.db}...")
        store = TagStore(args.db)
    else:
        print(f"Building in-memory index from {args.base}...")
        store = TagStore(args.base)
    print(f"  {len(store.route_paths())} routes, {len(store.npz_paths())} frames")

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "open_loop":
        export_open_loop(store, args.query, args.output, args.dimension)
    else:
        export_close_loop(
            store,
            args.query,
            args.output,
            args.dimension,
            args.margin_before,
            args.margin_after,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
