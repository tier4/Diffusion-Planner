#!/usr/bin/env python3
"""
Export dataset segments based on tag queries.

This script exports frames from a tagged dataset for open-loop or closed-loop evaluation.
Frames are selected via tag queries (passed to TagStore), then organized and symlinked
into an output directory.

Usage Examples:
---------------

    # close_loop: Export override segments (centerline + departure) for closed-loop evaluation.
    # Creates symlinks for each segment with margin frames, outputs close_loop.json.
    # Dataset: /mnt/storage_rdma/diffusion_planner/dataset/20260729_basic_dataset
    python export_dataset.py \
        --base /mnt/storage_rdma/diffusion_planner/dataset/20260729_basic_dataset \
        --output /data/override_closed_loop \
        --query "override:*" \
        --mode close_loop

    # open_loop: Export first frames of override events for open-loop evaluation.
    # Groups by override_first_frame value, outputs open_loop.json.
    python export_dataset.py \
        --base /mnt/storage_rdma/diffusion_planner/dataset/20260729_basic_dataset \
        --output /data/override_open_loop \
        --query "override_first_frame:*" \
        --mode open_loop

    # dimension-aware export: export all override frames grouped by their value.
    # Automatically queries all values within the specified dimension.
    python export_dataset.py \
        --base /mnt/storage_rdma/diffusion_planner/dataset/20260729_basic_dataset \
        --output /data/override_closed_loop \
        --query "override:*" \
        --dimension override \
        --mode close_loop

Arguments:
---------
    --base: Base dataset path containing routes with .npz frames and tags.tag index.
    --output: Output directory for exported segments and JSON index.
    --query: Tag query clause (passed directly to TagStore.query).
             Examples: "override:centerline", "override_first_frame:departure", "split:auto"
    --mode: Export mode. Either "open_loop" or "close_loop".
             - open_loop: Export individual frames matching the query.
             - close_loop: Export continuous segments with margin context around each frame.
    --index: Path to tags.tag index file. Defaults to <base>/tags.tag.
    --dimension: (optional) When used with --query, creates subdirectory <dimension>/<value>/ before route.
                 When used alone (no --query), queries all values in this dimension and exports each group.
                 - open_loop: Output becomes open_loop_<dimension>.json mapping values to frame lists.
                 - close_loop: Creates subdirectory <dimension>/ before route path.
    --margin-before: (close_loop only) Frames to include before each tagged frame. Default: 50.
                     Closed-loop evaluation needs context before the event.
    --margin-after: (close_loop only) Frames to include after each tagged frame. Default: 50.
                    Closed-loop evaluation needs context after the event.

Output Structure:
----------------
    open_loop (no dimension):
        output/open_loop.json           # List of npz paths

    open_loop with dimension:
        output/open_loop_<dimension>.json  # Dict mapping dimension value -> list of npz paths

    close_loop (no dimension):
        output/<route>/<segment_start>/     # Symlinked npz files
        output/close_loop.json               # Index JSON

    close_loop with dimension:
        output/<dimension>/<route>/<segment_start>/  # Symlinked npz files
        output/close_loop.json                         # Index JSON (includes dimension field)
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from tag_toolkit.routes import extract_frame_number, route_of
from tag_toolkit.store import TagStore


def _merge_segments(
    frame_numbers: list[int],
    margin_before: int,
    margin_after: int,
) -> list[tuple[int, int]]:
    """Merge nearby frames into continuous segments.

    Args:
        frame_numbers: Sorted list of frame numbers.
        margin_before: Frames to include before each frame.
        margin_after: Frames to include after each frame.

    Returns:
        List of (segment_start, segment_end) tuples (inclusive).
        Adjacent frames within (margin_before + margin_after) gap are merged.
    """
    if not frame_numbers:
        return []

    margin_total = margin_before + margin_after
    segments = []
    seg_start = frame_numbers[0]
    seg_end = frame_numbers[0]

    for frame in frame_numbers[1:]:
        gap = frame - seg_end
        if gap <= margin_total:
            # Merge: extend current segment
            seg_end = frame
        else:
            # New segment
            segments.append((seg_start, seg_end))
            seg_start = frame
            seg_end = frame

    segments.append((seg_start, seg_end))
    return segments


def export_open_loop(
    store: TagStore,
    query: str,
    output_dir: Path,
    dimension: str | None,
) -> None:
    """Export frames for open-loop evaluation.

    Args:
        store: TagStore with loaded index.
        query: Tag query clause.
        output_dir: Output directory.
        dimension: If specified, group frames by this dimension.
    """
    # Query matching frames
    matching_frames = store.query(query, granularity="frame")

    if not matching_frames:
        print(f"No frames match query: {query}")
        return

    print(f"Found {len(matching_frames)} frames matching query")

    if dimension:
        # Group by dimension value
        groups: dict[str, list[str]] = defaultdict(list)
        for npz in matching_frames:
            tags = store._index.frame_tags.get(npz, frozenset())
            dim_values = [t.split(":")[1] for t in tags if t.startswith(f"{dimension}:")]
            if dim_values:
                for val in dim_values:
                    groups[val].append(str(npz))
            else:
                groups["-"].append(str(npz))

        # Output one file with all dimension groups
        output_file = output_dir / f"open_loop_{dimension}.json"
        with open(output_file, "w") as f:
            json.dump(dict(groups), f, indent=2)
        print(f"Wrote {len(groups)} groups to {output_file}")
    else:
        # Flat list output
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
    """Export segments for closed-loop evaluation.

    Args:
        store: TagStore with loaded index.
        query: Tag query clause.
        output_dir: Output directory.
        dimension: If specified, create subdirectory for dimension value.
        margin_before: Frames to include before each tagged frame.
        margin_after: Frames to include after each tagged frame.
    """
    # Query matching frames
    matching_frames = store.query(query, granularity="frame")

    if not matching_frames:
        print(f"No frames match query: {query}")
        return

    print(f"Found {len(matching_frames)} frames matching query")

    # Group frames by route, tracking dimension per frame
    route_frames: dict[Path, list[int]] = defaultdict(list)
    # Cache dimension values by npz name for O(1) lookup
    npz_name_to_dim_value: dict[str, str] = {}

    for npz in matching_frames:
        r = route_of(npz)
        frame_num = extract_frame_number(npz)
        if frame_num is not None:
            route_frames[r].append(frame_num)
            # Cache dimension value for this npz
            if dimension:
                tags = store._index.frame_tags.get(npz, frozenset())
                for t in tags:
                    if t.startswith(f"{dimension}:"):
                        npz_name_to_dim_value[npz.name] = t.split(":")[1]
                        break

    # Merge segments and create symlinks
    index_entries = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for route, frames in sorted(route_frames.items()):
        frames.sort()
        segments = _merge_segments(frames, margin_before, margin_after)

        for seg_start, seg_end in segments:
            # Determine output path
            if dimension:
                # Find dimension value from frames within this segment (O(1) lookup)
                dim_value = "-"
                for frame_num in range(seg_start, seg_end + 1):
                    npz_name = f"{route.name}_{frame_num:08d}.npz"
                    if npz_name in npz_name_to_dim_value:
                        dim_value = npz_name_to_dim_value[npz_name]
                        break
                seg_dir = output_dir / dim_value / route.name / str(seg_start)
            else:
                seg_dir = output_dir / route.name / str(seg_start)

            seg_dir.mkdir(parents=True, exist_ok=True)

            # Create symlinks for all frames in [seg_start-margin, seg_end+margin]
            linked_count = 0
            for frame_num in range(seg_start - margin_before, seg_end + margin_after + 1):
                if frame_num < 0:
                    continue  # Skip negative frame numbers
                npz_name = f"{route.name}_{frame_num:08d}.npz"
                src = route / "routes" / npz_name
                if not src.exists():
                    src = route / npz_name  # Try flat layout
                if src.exists():
                    dst = seg_dir / npz_name
                    if not dst.exists():
                        os.symlink(src, dst)
                    linked_count += 1

            # Index entry
            entry = {
                "route": str(route),
                "segment_start": seg_start,
                "start_frame": seg_start - margin_before,
                "end_frame": seg_end + margin_after,
                "path": str(seg_dir.relative_to(output_dir)),
            }
            if dimension:
                entry["dimension"] = dim_value

            index_entries.append(entry)
            print(f"  Created segment {seg_dir.relative_to(output_dir)}: {linked_count} frames")

    # Write index
    index_file = output_dir / "close_loop.json"
    with open(index_file, "w") as f:
        json.dump(index_entries, f, indent=2)
    print(f"Wrote {len(index_entries)} segments to {index_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Export dataset segments based on tag queries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base", type=Path, required=True, help="Base dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--query", required=True, help='Tag query clause (e.g. "override:*" for any override value)')
    parser.add_argument(
        "--mode",
        required=True,
        choices=["open_loop", "close_loop"],
        help="Export mode: open_loop or close_loop",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Path to tags.tag index (default: <base>/tags.tag)",
    )
    parser.add_argument(
        "--dimension",
        help="Dimension for subdirectory grouping (e.g. --dimension override creates override/centerline/...)",
    )
    parser.add_argument(
        "--margin-before",
        type=int,
        default=50,
        help="Frames before each frame (close_loop only, default: 50)",
    )
    parser.add_argument(
        "--margin-after",
        type=int,
        default=50,
        help="Frames after each frame (close_loop only, default: 50)",
    )

    args = parser.parse_args()

    # Resolve index path
    if args.index:
        index_path = args.index
    else:
        index_path = args.base / "tags.tag"

    if not index_path.exists():
        print(f"Error: Index file not found: {index_path}")
        return 1

    # Load TagStore
    print(f"Loading index from {index_path}...")
    store = TagStore(str(index_path))

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Export based on mode
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
