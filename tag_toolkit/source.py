"""Expand a ``source`` into NPZ paths — fast path for large path lists.

Training/eval never ``rglob`` tens of millions of files at load time: they read a
pre-built ``path_list*.json`` (optionally ``.zst``) and keep **string** paths.
See ``diffusion_planner.utils.train_utils.openjson`` and
``DiffusionPlannerData``. This module follows the same rules:

* Path-list entries that already end in ``.npz``: load JSON, optional string
  dedupe, wrap in ``Path`` — **no** ``exists`` / ``resolve`` / ``rglob``.
* Directory expansion is for small trees / samples only; prefer a path list.
* ``Path.resolve()`` is avoided on the hot path (symlink/stat cost × N).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

Source = str | Path | list[str | Path]


def load_json(path: str | Path):
    """Load JSON; transparently handles zstd-compressed ``*.zst`` (same as openjson)."""
    path_s = str(path)
    if path_s.endswith(".zst"):
        import io

        import zstandard

        with open(path_s, "rb") as handle:
            reader = zstandard.ZstdDecompressor().stream_reader(handle)
            return json.load(io.TextIOWrapper(reader, encoding="utf-8"))
    with open(path_s, encoding="utf-8") as handle:
        return json.load(handle)


def _is_npz_path(entry: str | Path) -> bool:
    """Check if entry looks like an NPZ path (ends with .npz)."""
    return str(entry).endswith(".npz")


def _is_path_list_file(path: Path) -> bool:
    """Check if path is a path-list file (.json, .json.zst, .zst)."""
    name = path.name
    return name.endswith(".json") or name.endswith(".json.zst") or name.endswith(".zst")


def expand_source(source: Source, *, sort: bool = False) -> list[Path]:
    """Resolve *source* to ``.npz`` paths.

    Accepted forms:

    - one ``.npz`` path (no filesystem check)
    - a directory (recursive ``*.npz`` — **slow** on large trees; prefer path lists)
    - a path-list ``.json`` / ``.json.zst`` (JSON array of npz and/or directory strings)
    - an explicit sequence of any of the above

    Parameters
    ----------
    sort:
        If true, sort by path string at the end. Default false — large path lists
        keep file order (same as training loaders).
    """
    if isinstance(source, (str, Path)):
        paths = list(_expand_spec(os.path.expanduser(str(source))))
    else:
        paths = list(_expand_sequence(source))
    if sort:
        paths.sort(key=str)
    return paths


def _expand_sequence(items: Sequence[str | Path]) -> list[Path]:
    """Expand a sequence of sources, deduplicating paths."""
    if not items:
        return []

    # Fast path: already a list of npz path strings (the training path_list shape).
    if all(_is_npz_path(x) for x in items):
        return _dedupe_npz_strings(os.path.expanduser(str(x)) for x in items)

    # Mixed: recurse into each item
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        for path in _expand_spec(os.path.expanduser(str(item))):
            key = str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _dedupe_npz_strings(strings: list[str] | tuple[str, ...]) -> list[Path]:
    """Dedupe NPZ path strings, preserving order."""
    seen: set[str] = set()
    out: list[Path] = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            out.append(Path(s))
    return out


def _expand_spec(path_s: str) -> list[Path]:
    """Expand a single source specification."""
    # NPZ file: return as-is (light exists check — 1 stat() call, no rglob)
    if path_s.endswith(".npz"):
        p = Path(path_s)
        if not p.exists():
            raise FileNotFoundError(f"source not found: {path_s}")
        return [p]

    path = Path(path_s)

    # Path-list JSON file
    if _is_path_list_file(path) and path.is_file():
        return _expand_path_list_file(path)

    # Directory
    if path.is_dir():
        return _expand_directory(path)

    # File exists but unrecognized type
    if path.exists():
        raise ValueError(f"source is neither .npz, directory, nor path-list JSON: {path_s}")
    raise FileNotFoundError(f"source not found: {path_s}")


def _expand_path_list_file(path: Path) -> list[Path]:
    """Expand a path-list JSON file."""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"path list must be a JSON array: {path}")
    if not data:
        return []
    for entry in data:
        if not isinstance(entry, str):
            raise ValueError(f"path list entries must be strings: {path}")

    # Training lists: every entry is an npz path — no stat, no rglob.
    if all(_is_npz_path(entry) for entry in data):
        return _dedupe_npz_strings([os.path.expanduser(entry) for entry in data])

    # Mixed / closed-loop lists may contain route directories — expand those only.
    out: list[Path] = []
    seen: set[str] = set()
    for item in data:
        for npz in _expand_spec(os.path.expanduser(item)):
            key = str(npz)
            if key not in seen:
                seen.add(key)
                out.append(npz)
    return out


def _expand_directory(root: Path) -> list[Path]:
    """Collect ``*.npz`` under *root*. Prefer a path list for large datasets."""
    out: list[Path] = []
    # os.walk is lighter than Path.rglob on deep trees; still O(files).
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".npz"):
                out.append(Path(dirpath, name))
    return out
