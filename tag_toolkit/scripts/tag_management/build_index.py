#!/usr/bin/env python3
"""Build a TagStore SQLite index (.tags.db) from a dataset source on the CLI.

This is a thin wrapper around ``TagStore.build_index(source, output)``;
it exists so the index can be materialised without writing Python code.
A pre-built index is the fast path for downstream tools such as
``export_dataset.py`` on large datasets.

Usage::

    python build_index.py /path/to/dataset -o /data/tags.tags.db
    python build_index.py /path/to/dataset -o /data/tags.tags.db --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TAG_TOOLKIT_PARENT = _REPO_ROOT / "Diffusion-Planner" / "tag_toolkit"
if _TAG_TOOLKIT_PARENT.exists():
    sys.path.insert(0, str(_TAG_TOOLKIT_PARENT.parent))

from tag_toolkit import TagStore

_DB_SUFFIXES = (".tags.db",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TagStore SQLite index from a dataset source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "TagToolkit source (see tag_toolkit.source): a directory, a "
            "path-list .json / .json.zst, a single .npz, or a sequence of "
            "those. An in-memory index is built from this and written to "
            "--output."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help=("Output .tags.db path. Must end in one of: " + ", ".join(_DB_SUFFIXES) + "."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists. Default: fail-fast.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not any(args.output.name.endswith(suf) for suf in _DB_SUFFIXES):
        print(
            f"Error: --output must end in one of {_DB_SUFFIXES}: {args.output}",
            file=sys.stderr,
        )
        return 1

    if not args.source.exists():
        print(f"Error: source not found: {args.source}", file=sys.stderr)
        return 1

    if args.output.exists() and not args.force:
        print(
            f"Error: output already exists: {args.output} (pass --force to overwrite)",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building index from {args.source}...")
    store = TagStore.build_index(args.source, args.output)

    print(f"  {len(store.route_paths())} routes, {len(store.npz_paths())} frames")
    print(f"Wrote index to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
