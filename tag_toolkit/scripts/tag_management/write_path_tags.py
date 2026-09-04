#!/usr/bin/env python3
"""Write path-derived tags into NPZ sidecars, then build a TagStore SQLite index.

For every NPZ under the input source it writes the following tag dimensions:

==============  ===================================================================  ===================================================
Tag              Source                                                                Example
==============  ===================================================================  ===================================================
``site``         path component immediately before ``manual`` / ``auto`` / ``valid``   ``1234_odaiba``
``split``        the ``manual`` / ``auto`` / ``valid`` component itself                 ``manual``
``project``      the path component one level above ``site``                           ``project_a``
==============  ===================================================================  ===================================================

And, *only* when ``--project-vehicle-map`` is supplied and points at a readable
JSON file:

==============  ===================================================================  ===================================================
``vehicle``      ``project_vehicle_map[project]``                                      ``j6_xxx2``
==============  ===================================================================  ===================================================

``split```` can be refined per-route by a ``split_labels.json`` file (via
``--split-labels`` or auto-loaded from the input source's directory); this
overrides the path-derived ``split`` value, allowing ``manual`` routes to be
labelled ``train`` / ``valid`` etc.

Path parsing is delegated to :func:`site_discovery.parse_npz_path`, which is
also used by the closed-loop site runner, so site / split / project tags are
consistent across the entire project.

After sidecar tags are written, a TagStore SQLite index is materialised next
to the input source so downstream tools (closed-loop report, W&B rollup, etc.)
can query without re-reading every sidecar.

Usage::

    python write_path_tags.py path_list.json
    python write_path_tags.py data_root/                                  # auto-loads split_labels.json from data_root/ if it exists
    python write_path_tags.py path_list.json --project-vehicle-map project_vehicle_map.json
    python write_path_tags.py path_list.json --project-vehicle-map ... --index-output my.tags.db
    python write_path_tags.py data_root/ --split-labels overrides.json

If ``--index-output`` is omitted, the DB is written as
``<path_list_stem>.tags.db`` next to the input source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

from tag_toolkit import TagStore, format_tag
from tag_toolkit.sidecar import drop_dimension, normalize_tags, read_tags, write_tags
from tag_toolkit.source import Source, expand_source


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


sys.path.insert(0, str(_repo_root() / "Diffusion-Planner" / "scenario_generation"))

from site_vehicle_discovery import NPZPathInfo, parse_npz_path  # noqa: E402

# Dimensions this script owns. The tag-merge loop drops these from existing
# sidecars before appending fresh values, so reruns overwrite cleanly.
_OWNED_DIMS = ("site", "split", "project", "vehicle")


def load_split_labels(path: str | Path) -> dict[str, str]:
    """Load ``split_labels.json`` -> ``{route_key_without_mode: split}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("split_labels.json must be an object")
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "split" in value:
            out[str(key)] = str(value["split"])
        elif isinstance(value, str):
            out[str(key)] = value
    return out


def load_project_vehicle_map(path: str | Path | None) -> dict[str, str] | None:
    """Load ``{project: vehicle_type}`` from *path*, or return ``None`` if absent.

    A missing / unreadable map is not an error: callers fall back to writing
    only the path-derived tags (site / split / project). The user can still
    run the script without ever owning a map.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"project_vehicle_map not found, skipping vehicle tags: {p}", file=sys.stderr)
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"project_vehicle_map must be a JSON object: {p}")
    return {str(k): str(v) for k, v in data.items()}


def parse_site_split(npz: str | Path) -> tuple[str, str]:
    """Infer ``(site, split)`` from an NPZ path.

    Falls back to ``"unknown"`` for either component if the path has no
    recognised split directory.
    """
    info = parse_npz_path(npz)
    if info is None:
        return ("unknown", "unknown")
    return (info.site, info.split)


def apply_path_tags(
    source: Source,
    *,
    project_vehicle_map: dict[str, str] | None = None,
    split_labels: str | Path | Mapping[str, str] | None = None,
) -> int:
    """Write ``site:`` / ``split:`` / ``project:`` / ``vehicle:`` for every NPZ under *source*.

    If *split_labels* is supplied (a path to ``split_labels.json`` or a dict), it
    takes precedence over the ``split`` value derived from the directory layout,
    allowing ``manual`` routes to be refined into ``train`` / ``valid`` etc.

    Returns the number of sidecars whose tag set changed.
    """
    labels: Mapping[str, str] | None
    if split_labels is None:
        labels = None
    elif isinstance(split_labels, Mapping):
        labels = split_labels
    else:
        labels = load_split_labels(split_labels)

    npz_paths = expand_source(source, sort=False)
    if not npz_paths:
        return 0

    n = 0
    for npz in npz_paths:
        info = parse_npz_path(npz)
        if info is None:
            continue

        split = info.split

        # split_labels takes precedence over the path-derived split.
        if labels is not None:
            # Historically split_labels.json used keys both with and without the
            # split token; generate both forms to match whatever the file has.
            key_with_split = info.route_key(include_split=True)
            key_without_split = info.route_key(include_split=False)
            for key in (key_with_split, key_without_split):
                if key in labels:
                    split = labels[key]
                    break

        # project_vehicle_map refines project -> vehicle_type.
        vehicle_type = ""
        if project_vehicle_map is not None:
            vehicle_type = project_vehicle_map.get(info.project, "")

        desired = [
            format_tag("site", info.site),
            format_tag("split", split),
            format_tag("project", info.project),
        ]
        if vehicle_type:
            desired.append(format_tag("vehicle", vehicle_type))

        current = read_tags(npz)
        for dim in _OWNED_DIMS:
            current = drop_dimension(current, dim)
        merged = normalize_tags(list(current) + desired)
        if merged != normalize_tags(read_tags(npz)):
            write_tags(npz, merged)
            n += 1
    return n


def _default_index_output(source: Source) -> Path:
    """Default ``--index-output`` location: ``<path_list_stem>.tags.db`` next to the JSON."""
    if isinstance(source, (list, tuple)) or not source:
        return Path("path_list.tags.db")
    p = Path(source)
    return p.with_name(f"{p.stem}.tags.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write path-derived tags (site / split / project / vehicle) into NPZ "
            "sidecars, then build a TagStore SQLite index."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "TagToolkit source: a directory, a path-list .json / .json.zst, a "
            "single .npz, or any sequence of those."
        ),
    )
    parser.add_argument(
        "--project-vehicle-map",
        type=Path,
        default=None,
        help=(
            "optional JSON file mapping project -> vehicle_type. If supplied "
            "and the file exists, vehicle:<type> tags are also written. "
            "Missing or unreadable -> only site / split / project are written."
        ),
    )
    parser.add_argument(
        "--split-labels",
        type=Path,
        default=None,
        help=(
            "optional split_labels.json that refines ``manual`` into train / valid "
            "etc. If omitted, the script attempts to load split_labels.json from the "
            "input source's directory automatically."
        ),
    )
    parser.add_argument(
        "--index-output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Where to write the TagStore SQLite index. Defaults to "
            "<source_stem>.tags.db next to the source."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the index DB if it already exists. Default: fail-fast.",
    )
    return parser.parse_args()


def _default_split_labels(source: Source) -> Path | None:
    """Try to locate split_labels.json next to the input source."""
    if isinstance(source, (list, tuple)):
        return None
    p = Path(source) if source else Path(".")
    if p.is_file():
        p = p.parent
    candidate = p / "split_labels.json"
    return candidate if candidate.is_file() else None


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"source not found: {args.source}")

    project_vehicle_map = load_project_vehicle_map(args.project_vehicle_map)
    split_labels = args.split_labels or _default_split_labels(args.source)
    updated = apply_path_tags(
        args.source,
        project_vehicle_map=project_vehicle_map,
        split_labels=split_labels,
    )

    if split_labels is None:
        print(f"updated {updated} sidecars with path-derived tags")
    else:
        print(f"updated {updated} sidecars using split labels: {split_labels}")

    index_output = args.index_output or _default_index_output(args.source)
    if index_output.exists() and not args.force:
        raise FileNotFoundError(
            f"index output already exists: {index_output} (pass --force to overwrite)"
        )
    index_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"building TagStore index from {args.source} ...")
    store = TagStore.build_index(args.source, index_output)
    print(f"  {len(store.route_paths())} routes, {len(store.npz_paths())} frames")
    print(f"wrote index to {index_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
