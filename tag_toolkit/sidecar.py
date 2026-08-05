"""Parse and rewrite the ``tags`` field on NPZ sidecar JSON files.

Sidecar reads tolerate malformed entries: a tag that does not match
``^[a-z0-9_]+:[a-z0-9_]+$`` is dropped with a :mod:`warnings` message so a
single bad sidecar cannot fail an otherwise healthy build. Structural
errors — bad JSON, top-level not a JSON object, ``tags`` not a list —
still raise :class:`ValueError` because those mean the sidecar itself is
broken in a way the user must fix.

Writes are strict (they go through :func:`normalize_tags`) and atomic:
a sibling ``.json.tmp`` is created and renamed over the target on
success. On any error the temp file is removed and the original sidecar
is left untouched.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import IO, Callable, Iterable

TAG_RE = re.compile(r"^([a-z0-9_]+):([a-z0-9_]+)$")
DIM_RE = re.compile(r"^[a-z0-9_]+$")


def parse_tag(tag: str) -> tuple[str, str]:
    """``dimension:value`` -> ``(dimension, value)``.

    Raises ValueError on malformed input. Both sides must match
    ``^[a-z0-9_]+$`` (lowercase ASCII letters, digits, underscore).
    """
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(
            f"bad tag {tag!r}: expected 'dimension:value' with "
            f"[a-z0-9_]+ on each side (lowercase letters, digits, underscore)"
        )
    return match.group(1), match.group(2)


def format_tag(dimension: str, value: str) -> str:
    """``dimension:value`` string, validated."""
    tag = f"{dimension}:{value}"
    parse_tag(tag)  # validates
    return tag


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Validate, dedupe, and sort tag strings.

    The input is typically a list, so this is O(n) where n is the number of tags.
    Returns a new sorted list without duplicates. Raises ``ValueError`` on any
    entry that does not match ``^[a-z0-9_]+:[a-z0-9_]+$``.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags:
        parse_tag(raw)  # raises ValueError on invalid
        if raw not in seen:
            seen.add(raw)
            out.append(raw)
    out.sort()
    return out


def is_valid_dimension(name: str) -> bool:
    """True iff *name* matches ``^[a-z0-9_]+$``."""
    return bool(DIM_RE.fullmatch(name))


def sidecar_path(npz: str | Path) -> Path:
    """Return the path to the JSON sidecar for an NPZ.

    Args:
        npz: A path whose final component ends with ``.npz``. The sibling
             sidecar lives next to it as ``<stem>.json``. Common examples:
             ``/data/route_15/frame.npz`` → ``/data/route_15/frame.json``,
             ``frame.npz`` → ``frame.json``.

    Raises:
        ValueError: if *npz* doesn't end in ``.npz`` (e.g. someone passed a
             path ending in ``.json`` thinking it was the sidecar itself).
             Use :func:`read_sidecar` if you already have a sidecar path.
    """
    path = Path(npz)
    if not path.name.endswith(".npz"):
        raise ValueError(
            f"sidecar_path: expected a path ending with '.npz', got {path}. "
            f"Did you mean to pass a sidecar (.json) directly to read_tags / "
            f"read_sidecar instead of going through sidecar_path?"
        )
    return path.with_suffix(".json")


def read_sidecar(path_like: str | Path) -> dict:
    """Load sidecar JSON from a path. Accepts either an NPZ or a sidecar path.

    Missing file -> empty dict. Structural errors (bad JSON, non-object)
    raise :class:`ValueError`.
    """
    side = path_like if str(path_like).endswith(".json") else sidecar_path(path_like)
    if not side.is_file():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"sidecar is not valid JSON: {side}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"sidecar must be a JSON object: {side}")
    return data


class StaleIndexError(RuntimeError):
    """The on-disk sidecar has drifted from the in-memory index.

    Raised by :func:`write_tags` when the caller supplies
    ``expected_tags`` and the actual ``tags`` field on disk does not match.
    The sidecar is left untouched on this error — re-read the index
    (:meth:`TagStore.reindex_tags`) and retry.
    """


def read_tags(npz: str | Path) -> list[str]:
    """Return the ``tags`` list for *npz* (missing ≡ ``[]``).

    Per-entry malformed tags (wrong shape, non-string, doesn't match
    ``^[a-z0-9_]+:[a-z0-9_]+$``) are dropped with a warning. Structural
    errors — bad JSON, top-level not a JSON object, ``tags`` not a list
    — raise :class:`ValueError`, since those mean the sidecar itself is
    broken. Result is sorted and free of duplicates — callers can rely on
    the contract: ``read_tags`` is idempotent against repeat reads of the
    same sidecar.
    """
    side = sidecar_path(npz)
    data = read_sidecar(side)
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError(f"sidecar 'tags' must be a list: {side}")
    good: list[str] = []
    for t in tags:
        if not isinstance(t, str):
            warnings.warn(f"skipping non-string tag entry {t!r} in {side}", stacklevel=2)
            continue
        try:
            parse_tag(t)
        except ValueError:
            warnings.warn(f"skipping invalid tag {t!r} in {side}", stacklevel=2)
            continue
        good.append(t)
    return sorted(set(good))


def _atomic_write_bytes(
    path: Path,
    write_fn: Callable[[IO[bytes]], None],
    *,
    sync: bool = True,
) -> None:
    """Atomically write *path* by calling *write_fn* with an open binary file.

    Writes go through a sibling ``.tmp`` (so the rename is atomic on the same
    filesystem), get an ``os.fsync`` on the file before ``os.replace``, and
    finally ``os.fsync`` the parent directory so the new name is durable on
    power-loss. *write_fn* may write as many times and as much as it likes;
    the file is flushed and fsync'd after it returns and before the rename.

    On any error inside the ``with``, the tmp file is removed and *path* is
    left untouched.

    Args:
        path: Target file path.
        write_fn: Callable receiving the open binary file handle. Must do
            its own encoding if it wants text on disk.
        sync: If True (default), fsync file and directory for durability.
            If False, skip both fsyncs; caller is responsible for durability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
            f.flush()
            if sync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
        if sync:
            _fsync_dir(path.parent)
    except Exception:
        # Best-effort cleanup; tmp may not exist if open() failed early.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str, *, sync: bool = True) -> None:
    """Atomic text write. Thin wrapper over :func:`_atomic_write_bytes`."""
    payload = text.encode("utf-8")
    _atomic_write_bytes(
        path,
        lambda f: f.write(payload),
        sync=sync,
    )


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory. Some filesystems reject it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def cleanup_tmp_files(directory: Path) -> int:
    """Remove any ``*.json.tmp`` files left behind by an aborted write.

    Returns the number of files removed. Safe to call repeatedly.
    """
    if not directory.is_dir():
        return 0
    removed = 0
    for tmp in directory.glob("*.json.tmp"):
        try:
            tmp.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def write_tags(
    npz: str | Path,
    tags: Iterable[str],
    *,
    sync: bool = True,
    expected_tags: Iterable[str] | None = None,
) -> list[str]:
    """Write normalized ``tags`` into the sidecar, preserving other fields.

    Atomic: a sibling ``.json.tmp`` is written and renamed over the target
    on success. If the sidecar is missing, raises :class:`FileNotFoundError`
    — :func:`write_tags` will never create a sidecar that didn't already
    exist.

    Args:
        npz: Path to the NPZ file.
        tags: Tags to write.
        sync: If True (default), fsync for durability. If False, skip fsync
              for batch operations where caller flushes separately.
        expected_tags: If given, the current ``tags`` list on disk must
              equal ``frozenset(expected_tags)`` exactly — otherwise
              :class:`StaleIndexError` is raised and the sidecar is left
              untouched. Pass ``None`` (the default) to skip the check;
              useful when callers don't have an index to compare against.
    """
    side = sidecar_path(npz)
    if not side.is_file():
        raise FileNotFoundError(f"sidecar not found: {side}")
    data = read_sidecar(side)

    # Structural check: 'tags' must be a list (possibly empty). Non-list
    # values are a corruption signal we cannot recover from silently —
    # coerce them to a set and the bug is hidden.
    raw_tags = data.get("tags")
    if raw_tags is None:
        raw_tags = []
    if not isinstance(raw_tags, list):
        raise ValueError(
            f"sidecar {side} has non-list 'tags' field: "
            f"type={type(raw_tags).__name__}"
        )
    on_disk = frozenset(raw_tags)

    if expected_tags is not None:
        expected = frozenset(expected_tags)
        if on_disk != expected:
            raise StaleIndexError(
                f"sidecar {side} has drifted from index: "
                f"disk={sorted(on_disk)} expected={sorted(expected)}"
            )

    normalized = normalize_tags(tags)
    data["tags"] = normalized
    _atomic_write_text(side, json.dumps(data, indent=1, ensure_ascii=False) + "\n", sync=sync)
    return normalized


def drop_dimension(tags: Iterable[str], dimension: str) -> list[str]:
    """Remove all tags for *dimension* from the input."""
    return [t for t in tags if parse_tag(t)[0] != dimension]
