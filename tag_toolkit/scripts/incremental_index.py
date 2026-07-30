#!/usr/bin/env python3
"""Append new frames to an existing ``.tag`` index, without rescanning old frames.

This script is a helper for the common "only new data is added" workflow:

    day 1: dataset has 1k routes  -> TagStore.build_index(data, "tags.tag")
    day 2: dataset has 1k + N     -> incremental_index("tags.tag", data)
    day 3: dataset has 1k + N + M -> incremental_index("tags.tag", data)
    ...

Every call uses the existing on-disk sidecars for the **new** frames only.
Old frames are never read, never re-tagged, never re-validated. The resulting
``.tag`` index contains the union of (old index) and (newly-scanned frames).

This script is *not* part of the public ``tag_toolkit`` API; it lives next to
the other dataset helper. It uses ``tag_toolkit``'s public ``expand_source``
plus the package-private helpers that build / persist the ``_Index`` pickle.
If those internals change, this script must follow.

Assumptions
-----------

1. **Old frames are immutable.** Tags on frames already in the old index are
   not changed by adding new data. If you change a tag on an existing frame
   *and* add new frames, this script will quietly ignore the tag change —
   you must rebuild the index from scratch for that.

2. **Old index has already been built.** Pass the path to a previously-written
   ``.tag`` file. If the file is missing or corrupt, the script fails fast.

3. **The "new" source contains the new frames.** The source is fed through
   ``expand_source`` (any form ``TagStore`` accepts: directory, NPZ, path-list
   JSON, list of paths, or any combination). Every NPZ it returns is treated
   as a candidate new frame.

4. **NPZ already in the old index is silently skipped.** The script warns and
   drops the frame; nothing on disk is rewritten. This makes the script
   idempotent — re-running it on the same new data yields the same index.

5. **New frames must have a sidecar.** The script uses the same
   "missing sidecar" check as ``TagStore.add_tags`` and refuses to start
   if any candidate frame lacks a ``<stem>.json``. This mirrors the
   behaviour of the regular ``TagStore.build_index`` flow: a half-built
   index is worse than no new run.

Output
------

By default the new index overwrites the old index file in place. Pass
``--output`` to write to a different path instead (the old file is left
untouched in that case).

Usage
-----

::

    # In-place (the usual case)
    python incremental_index.py existing.tags.tag /path/to/dataset

    # To a new file
    python incremental_index.py existing.tags.tag /path/to/dataset \\
        --output new.tags.tag

    # Multiple sources / path-list JSON (any shape expand_source takes)
    python incremental_index.py existing.tags.tag /path/to/added/routes \\
        --output new.tags.tag
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Make ``import tag_toolkit`` work when this script is run directly (the
# editable install covers the package path for pytest, but not for ad-hoc
# runs of the script).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "Diffusion-Planner"))

from tag_toolkit.sidecar import sidecar_path
from tag_toolkit.source import Source, expand_source
from tag_toolkit.store import (
    _build_index_from_frames,
    _load_index_from_pickle,
    _save_index_to_pickle,
)


def _partition_new_and_skipped(
    source: Source,
    *,
    known_frame_keys: set[Path],
) -> tuple[list[Path], list[Path]]:
    """Expand *source* and split into (new_frames, skipped_frames).

    Comparison is on the resolved absolute ``Path``, the canonical form
    every ``_Index`` field uses internally (and the natural key type for
    the index's ``frame_tags`` dict). A frame already in the old index
    goes into *skipped_frames* and is warned about.
    """
    new_frames: list[Path] = []
    skipped: list[Path] = []
    for npz in expand_source(source, sort=False):
        resolved = npz.resolve()
        if resolved in known_frame_keys:
            skipped.append(npz)
            continue
        new_frames.append(resolved)
    for dup in skipped:
        warnings.warn(
            f"skipping frame already in old index: {dup}",
            stacklevel=2,
        )
    return new_frames, skipped


def _check_sidecars_present(frames: list[Path]) -> None:
    """Raise ``FileNotFoundError`` if any candidate frame lacks its sidecar.

    Same semantics as ``TagStore._mutate_frames``: the whole job is rejected
    rather than partially started, so a failed run never leaves a half-built
    index.
    """
    missing = [sidecar_path(npz) for npz in frames if not sidecar_path(npz).is_file()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(f"missing sidecar(s) for new frames: {preview}{more}")


def _merge_indices(old, new) -> None:
    """Merge *new* into *old* in-place.

    *old* and *new* are both ``tag_toolkit.store._Index``. After this call
    *old* contains the union of the two. The merge assumes every NPZ in
    *new* is **not** in *old* (caller is responsible for de-duplication).
    """
    # Frame-level dicts: keys from new are guaranteed disjoint.
    old.frames.extend(new.frames)
    old.frame_tags.update(new.frame_tags)
    old.route_of_frame.update(new.route_of_frame)

    # Per-route: append new frames, then re-sort so frame order stays stable.
    for route, new_frames in new.frames_of_route.items():
        combined = old.frames_of_route.get(route, []) + new_frames
        old.frames_of_route[route] = sorted(combined)

    # Route-level tag union.
    for route, new_route_tags in new.route_tags.items():
        existing = old.route_tags.get(route, frozenset())
        old.route_tags[route] = existing | new_route_tags

    # Routes list: dedupe by identity, keep a stable sorted order.
    seen_routes: set = set()
    merged_routes: list[Path] = []
    for r in list(old.routes) + list(new.routes):
        if r not in seen_routes:
            seen_routes.add(r)
            merged_routes.append(r)
    old.routes = sorted(merged_routes)

    # tag_to_frames / tag_to_routes: union, preserving defaultdict-on-write.
    for tag, frame_set in new.tag_to_frames.items():
        merged = old.tag_to_frames.get(tag)
        if merged is None:
            old.tag_to_frames[tag] = set(frame_set)
        else:
            merged.update(frame_set)
    for tag, route_set in new.tag_to_routes.items():
        merged = old.tag_to_routes.get(tag)
        if merged is None:
            old.tag_to_routes[tag] = set(route_set)
        else:
            merged.update(route_set)

    # route_tag_counts[tag][route]: sum counts.
    for tag, route_counts in new.route_tag_counts.items():
        target = old.route_tag_counts[tag]
        for route, count in route_counts.items():
            target[route] = target.get(route, 0) + count


def build_incremental_index(
    old_index_path: str | Path,
    new_source: Source,
    *,
    output_path: str | Path | None = None,
) -> dict:
    """Merge frames from *new_source* into the index at *old_index_path*.

    Args:
        old_index_path: Path to a previously-written ``.tag`` index.
        new_source: Anything ``expand_source`` accepts — directory, NPZ,
                    path-list JSON, list of paths, or any combination.
                    Frames already in the old index are skipped (warned).
                    Frames without a sidecar cause the whole call to fail.
        output_path: Where to write the merged index. ``None`` (default)
                     means overwrite *old_index_path* in place.

    Returns:
        Summary dict with counts of (new frames added), (skipped frames)
        and the path that was written.

    Raises:
        FileNotFoundError: if any candidate new frame lacks a sidecar.
        ValueError: if *old_index_path* is not a valid pickled index.
    """
    old_index_path = Path(old_index_path)
    output_path = Path(output_path) if output_path is not None else old_index_path

    old = _load_index_from_pickle(old_index_path)
    # Index dict keys are Path objects — match by Path, not str, so the
    # comparison is the same identity test the rest of the toolkit uses.
    known_keys = set(old.frame_tags.keys())

    new_frames, skipped = _partition_new_and_skipped(new_source, known_frame_keys=known_keys)
    _check_sidecars_present(new_frames)

    if not new_frames:
        # No new work — still rewrite the file so callers can rely on it
        # existing after the call. The content is identical to the old
        # index, so the atomic rename is a safe no-op semantically.
        _save_index_to_pickle(old, output_path)
        return {
            "new_frames": 0,
            "skipped": len(skipped),
            "output": output_path,
        }

    new_idx = _build_index_from_frames(new_frames)
    _merge_indices(old, new_idx)
    _save_index_to_pickle(old, output_path)

    return {
        "new_frames": len(new_frames),
        "skipped": len(skipped),
        "output": output_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Append new frames to an existing tag_toolkit .tag index without "
            "rescanning old frames. Old index is overwritten in place unless "
            "--output is given."
        )
    )
    parser.add_argument(
        "old_index",
        type=Path,
        help="path to the existing .tag index file",
    )
    parser.add_argument(
        "new_source",
        type=Path,
        nargs="+",
        help=(
            "path(s) covering the new frames — directory, NPZ, path-list JSON, "
            "or any combination; passed through expand_source"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write merged index here instead of overwriting the old file",
    )
    args = parser.parse_args()

    if not args.old_index.is_file():
        raise FileNotFoundError(f"old index not found: {args.old_index}")

    # If the user passed multiple sources, hand them to expand_source as a
    # list. If just one, pass it directly so a single NPZ / single path
    # works without ceremony.
    new_source = args.new_source if len(args.new_source) > 1 else args.new_source[0]

    summary = build_incremental_index(
        args.old_index,
        new_source,
        output_path=args.output,
    )
    print(
        f"new frames added: {summary['new_frames']}, already-in-index skipped: {summary['skipped']}"
    )
    print(f"wrote {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
