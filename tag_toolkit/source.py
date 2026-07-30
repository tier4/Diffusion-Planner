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
from typing import Iterable, Iterator, Sequence

Source = str | Path | Sequence[str | Path]


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


def _looks_like_npz(entry: str | Path) -> bool:
    return str(entry).endswith(".npz")


def _looks_like_path_list_file(path: Path) -> bool:
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


def iter_source(source: Source) -> Iterator[Path]:
    """Same as :func:`expand_source` but yields paths (still materializes path lists)."""
    yield from expand_source(source, sort=False)


def _expand_sequence(items: Sequence[str | Path]) -> list[Path]:
    seq = list(items)
    if not seq:
        return []
    # Fast path: already a list of npz path strings (the training path_list shape).
    if all(_looks_like_npz(x) for x in seq):
        return _dedupe_npz_strings(str(os.path.expanduser(str(x))) for x in seq)

    out: list[Path] = []
    seen: set[str] = set()
    for item in seq:
        for path in _expand_spec(os.path.expanduser(str(item))):
            key = str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _dedupe_npz_strings(strings: Iterable[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            out.append(Path(s))
    return out


def _expand_spec(path_s: str) -> list[Path]:
    if path_s.endswith(".npz"):
        return [Path(path_s)]

    path = Path(path_s)
    if _looks_like_path_list_file(path) and path.is_file():
        return _expand_path_list_file(path)

    if path.is_dir():
        return _expand_directory(path)

    if path.exists():
        raise ValueError(
            f"source is neither .npz, directory, nor path-list JSON: {path_s}"
        )
    raise FileNotFoundError(f"source not found: {path_s}")


def _expand_path_list_file(path: Path) -> list[Path]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"path list must be a JSON array: {path}")
    if not data:
        return []
    for entry in data:
        if not isinstance(entry, str):
            raise ValueError(f"path list entries must be strings: {path}")

    # Training lists: every entry is an npz path — no stat, no rglob.
    if all(_looks_like_npz(entry) for entry in data):
        return _dedupe_npz_strings(os.path.expanduser(entry) for entry in data)

    # Mixed / closed-loop lists may contain route directories — expand those only.
    out: list[Path] = []
    seen: set[str] = set()
    for entry in data:
        for npz in _expand_spec(os.path.expanduser(entry)):
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
