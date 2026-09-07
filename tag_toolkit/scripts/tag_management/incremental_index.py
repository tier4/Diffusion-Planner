#!/usr/bin/env python3
"""Append new frames to an existing tag index, without rescanning old frames.

The script walks *new_source* through ``expand_source``, drops anything already
in the old index, and merges the rest into a SQLite database. Old frames are
never read, never re-tagged, never re-validated.

Usage::

    python incremental_index.py existing.tags.db /path/to/dataset
    python incremental_index.py existing.tags.db /path/to/dataset --output new.tags.db
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

# Make `tag_toolkit` importable when the script is run directly (an editable
# install isn't always present).
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "Diffusion-Planner"))

from tag_toolkit import TagStore
from tag_toolkit.sidecar import sidecar_path
from tag_toolkit.source import Source, expand_source


def _partition_new_and_skipped(
    source: Source,
    known_paths: set[str],
) -> tuple[list[Path], list[Path]]:
    """Split *source* into (new, skipped) frames. Already-indexed paths warn."""
    new_frames: list[Path] = []
    skipped: list[Path] = []
    for npz in expand_source(source, sort=False):
        if str(npz) in known_paths:
            skipped.append(npz)
        else:
            new_frames.append(npz)
    for dup in skipped:
        warnings.warn(f"skipping frame already in old index: {dup}", stacklevel=2)
    return new_frames, skipped


def _check_sidecars_present(frames: list[Path]) -> None:
    """Fail fast if any new frame lacks a sidecar."""
    missing = [p for p in frames if not sidecar_path(p).is_file()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(f"missing sidecar(s) for new frames: {preview}{more}")


def _merge(
    old_index_path: Path,
    new_source: Source,
    output_path: Path,
) -> dict:
    """Build the merged index in a temp file then move it to *output_path*.

    Working in a temp copy keeps the original index byte-identical (TagStore
    triggers WAL mode, which checkpoints back to the main file on close).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "merged.tags.db"
        shutil.copy2(old_index_path, tmp_path)
        store = TagStore(str(tmp_path))
        try:
            known_paths = {str(p) for p in store.npz_paths()}
            new_frames, skipped = _partition_new_and_skipped(new_source, known_paths)
            _check_sidecars_present(new_frames)
            if not new_frames:
                return {"new_frames": 0, "skipped": len(skipped), "output": output_path}
            store.append_frames(new_frames)
        finally:
            # Drop the in-memory reference so the connection closes and the
            # WAL is checkpointed to tmp_path before the tempdir is removed.
            store = None
        shutil.move(str(tmp_path), output_path)
    return {"new_frames": len(new_frames), "skipped": len(skipped), "output": output_path}


def build_incremental_index(
    old_index_path: str | Path,
    new_source: Source,
    *,
    output_path: str | Path | None = None,
) -> dict:
    """Merge frames from *new_source* into the index at *old_index_path*.

    Args:
        old_index_path: Path to a previously-written SQLite index.
        new_source: Anything ``expand_source`` accepts.
        output_path: Where to write the merged index. ``None`` means overwrite
            *old_index_path* in place.

    Returns:
        Summary dict with counts of new frames added / skipped frames
        already in the index, plus the output path written.
    """
    old_index_path = Path(old_index_path)
    output_path = Path(output_path) if output_path is not None else old_index_path

    if not old_index_path.is_file():
        raise FileNotFoundError(f"old index not found: {old_index_path}")

    # Fail fast on non-SQLite files so the caller gets a clear ValueError
    # rather than a stray sqlite3.DatabaseError from inside _init_db.
    import sqlite3

    try:
        probe = sqlite3.connect(f"file:{old_index_path}?mode=ro", uri=True, timeout=1)
        try:
            probe.execute("SELECT 1 FROM frames LIMIT 1")
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"'{old_index_path}' is not a valid SQLite index: {exc}") from exc
        finally:
            probe.close()
    except sqlite3.Error as exc:
        raise ValueError(f"'{old_index_path}' is not a valid SQLite index: {exc}") from exc

    return _merge(old_index_path, new_source, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Append new frames to an existing tag_toolkit SQLite index without "
            "rescanning old frames. Old index is overwritten in place unless "
            "--output is given."
        )
    )
    parser.add_argument(
        "old_index",
        type=Path,
        help="path to the existing SQLite index file (.tags.db)",
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

    new_source = args.new_source if len(args.new_source) > 1 else args.new_source[0]
    summary = build_incremental_index(args.old_index, new_source, output_path=args.output)
    print(
        f"new frames added: {summary['new_frames']}, already-in-index skipped: {summary['skipped']}"
    )
    print(f"wrote {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
