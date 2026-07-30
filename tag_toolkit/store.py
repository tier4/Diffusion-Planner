"""TagStore — query and mutate ``tags`` from sidecar JSON files.

Tags travel with each NPZ's sidecar JSON. The store scans a source (one
or more directories / path-list JSON files / NPZ files), builds an
in-memory index, and exposes query / mutation on top of that index.

The index can be persisted with :meth:`TagStore.build_index` (pickle) and
loaded back via ``TagStore("path/to/index.tag")`` for fast repeated queries.

Read methods are lock-free; concurrent writers are serialised via an
internal ``RLock``. Mutations update the in-memory index immediately;
``.tag`` files persist on disk only when ``build_index`` runs again.
"""

from __future__ import annotations

import fnmatch
import os
import pickle
import re
import threading
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union

from .routes import route_of
from .sidecar import (
    format_tag,
    is_valid_dimension,
    normalize_tags,
    parse_tag,
    read_tags,
    sidecar_path,
    write_tags,
)
from .source import expand_source

Clause = str | dict[str, Any]  # "dim:value" or {"all"|"any"|"not": ...}
Granularity = Literal["route", "frame"]
FrameFilter = tuple[int, int] | str | None
_GRANULARITIES = ("route", "frame")
_CLAUSE_OPS = frozenset({"all", "any", "not"})

# Scope: a single route/npz, list of routes/npz, TagStore, or None
# Accepts the same flexibility as expand_source - operates on the npz set
ScopeItem = Union[Path, "TagStore"]
Scope = Union[None, ScopeItem, Sequence[ScopeItem]]

# Pickle index on-disk format (C1): 4-byte magic + 1-byte version, then a
# standard pickle. Legacy files without the header still load — the loader
# detects "no magic" and falls back to unpickling the whole file.
_PICKLE_MAGIC = b"TGID"
_PICKLE_VERSION = b"\x01"
_NO_INDEX_HINT = "no index loaded (hint: use TagStore(source) or TagStore.build_index)"


@dataclass
class Bucket:
    """One cell of :meth:`TagStore.group_by`.

    Attributes:
        values: One entry per requested dimension (the same order as the
            ``dimensions`` arg of ``group_by``). A value of ``None`` means
            no member of this cell carries that dimension.
        count: Number of unique paths in this cell.
        members: Concrete paths in this cell, sorted by ``str(path)``.
            Always populated (an empty list when ``group_by`` finds no
            items matching the clause / scope).
    """

    values: dict[str, str | None]
    count: int
    members: list[Path] = field(default_factory=list)

    def label(self, sep: str = " | ") -> str:
        return sep.join("-" if v is None else v for v in self.values.values())


@dataclass
class _Index:
    """In-memory index."""

    frames: list[Path] = field(default_factory=list)
    frame_tags: dict[Path, frozenset[str]] = field(default_factory=dict)
    route_of_frame: dict[Path, Path] = field(default_factory=dict)
    frames_of_route: dict[Path, list[Path]] = field(default_factory=dict)
    route_tags: dict[Path, frozenset[str]] = field(default_factory=dict)
    tag_to_frames: dict[str, set[Path]] = field(default_factory=lambda: defaultdict(set))
    tag_to_routes: dict[str, set[Path]] = field(default_factory=lambda: defaultdict(set))
    routes: list[Path] = field(default_factory=list)
    # Per-route tag counts: tag -> route -> count
    # Used for O(1) index updates instead of O(route_frames) scans
    route_tag_counts: dict[str, dict[Path, int]] = field(default_factory=lambda: defaultdict(dict))


def format_buckets(buckets: list[Bucket], dimensions: Sequence[str]) -> str:
    """Plain-text table for CLI / notebooks.

    For multi-value dimensions, one route/frame may appear in multiple
    cells. The TOTAL row counts each unique member once via ``Bucket.members``,
    so deduping is correct regardless of how many cells a member lands in.
    """
    headers = list(dimensions) + ["count"]
    rows: list[list[str]] = []
    for b in buckets:
        row = [("-" if b.values.get(d) is None else str(b.values.get(d))) for d in dimensions]
        row.append(str(b.count))
        rows.append(row)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    unique_members: set[Path] = set()
    for b in buckets:
        unique_members.update(b.members)
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    total_row: list[str] = []
    for i in range(len(headers)):
        if i == 0:
            total_row.append("TOTAL".ljust(widths[i]))
        elif headers[i] == "count":
            total_row.append(str(len(unique_members)).ljust(widths[i]))
        else:
            total_row.append("".ljust(widths[i]))
    lines.append("  ".join(total_row))
    return "\n".join(lines)


#: Regex to extract frame number from filename like "XXX_00000000_XXXXXXXX.npz"
_FRAME_NUMBER_RE = re.compile(r"_(\d{8})\.npz$")


def _extract_frame_number(path: Path) -> int | None:
    """Extract frame number from NPZ filename.

    Filenames follow the pattern: <bag_time>_<prefix>_<frame_number>.npz
    where frame_number is an 8-digit zero-padded integer.

    Returns the frame number as int, or None if extraction fails.
    """
    match = _FRAME_NUMBER_RE.search(path.name)
    if match:
        return int(match.group(1))
    return None


def _match_frame_filter(path: Path, frame_filter: FrameFilter) -> bool:
    """Check if a path matches the given frame filter.

    Args:
        path: Path to the NPZ file
        frame_filter: Either a (min, max) tuple for range matching,
                      a glob pattern string, or None (always matches)

    Note:
        A range tuple ``(min, max)`` matches NPZs whose filename contains an
        8-digit zero-padded frame number segment right before ``.npz``
        (the convention ``<bag_time>_<prefix>_<8 digits>.npz``). NPZs that
        don't match the convention are silently treated as "not matching";
        use a glob string for non-conventional filenames.
    """
    if frame_filter is None:
        return True

    if isinstance(frame_filter, str):
        return fnmatch.fnmatch(path.name, frame_filter)

    # Range filter: (min, max) inclusive
    frame_num = _extract_frame_number(path)
    if frame_num is None:
        return False
    return frame_filter[0] <= frame_num <= frame_filter[1]


def _build_index_from_frames(frames: list[Path]) -> _Index:
    """Build an _Index from a list of NPZ paths.

    All paths are stored as absolute paths for portability. Sidecars are
    read via :func:`sidecar.read_tags` which warns and skips on per-tag
    shape errors, so one bad tag cannot fail the whole scan.
    """
    idx = _Index()
    route_tag_sets: dict[Path, set[str]] = defaultdict(set)
    route_frames: dict[Path, list[Path]] = defaultdict(list)

    for npz in frames:
        # Ensure absolute path
        npz = npz.resolve()
        tags = read_tags(npz)
        tags_set = frozenset(tags)
        route = route_of(npz).resolve()

        idx.frames.append(npz)
        idx.frame_tags[npz] = tags_set
        idx.route_of_frame[npz] = route
        route_frames[route].append(npz)

        route_tag_sets[route].update(tags_set)

    for route in sorted(route_frames.keys()):
        frames_list = sorted(route_frames[route])
        idx.routes.append(route)
        idx.frames_of_route[route] = frames_list
        idx.route_tags[route] = frozenset(route_tag_sets[route])

        for tag in route_tag_sets[route]:
            idx.tag_to_routes[tag].add(route)

        for frame in frames_list:
            for tag in idx.frame_tags[frame]:
                idx.tag_to_frames[tag].add(frame)

    # Build tag counts per route for O(1) index updates
    for route, frames_list in idx.frames_of_route.items():
        for frame in frames_list:
            for tag in idx.frame_tags[frame]:
                idx.route_tag_counts[tag][route] = idx.route_tag_counts[tag].get(route, 0) + 1

    return idx


def _save_index_to_pickle(idx: _Index, output: str | Path) -> None:
    """Atomically write the index to a pickle file.

    On-disk format: ``_PICKLE_MAGIC + _PICKLE_VERSION + pickle_bytes``.
    Writes go through a sibling ``.tmp`` and ``os.replace``; the parent
    directory is fsync'd so a power-loss cannot leave a renamed file
    whose directory entry hasn't been durably committed.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(_PICKLE_MAGIC + _PICKLE_VERSION)
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(out.parent)


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory entry. Some filesystems don't support it."""
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


def _load_index_from_pickle(path: str | Path) -> _Index:
    """Load an index from a pickle file.

    Accepts files written by current ``_save_index_to_pickle`` (magic +
    version prefix) as well as legacy pickles that have no header. On
    a recognised future version we raise so the caller can rebuild.
    """
    with open(path, "rb") as f:
        head = f.read(len(_PICKLE_MAGIC))
        if head == _PICKLE_MAGIC:
            version = f.read(1)
            if version != _PICKLE_VERSION:
                raise ValueError(
                    f"incompatible index version in {path} "
                    f"(got {version!r}, expected {_PICKLE_VERSION!r}); "
                    f"rebuild with TagStore.build_index"
                )
            idx = pickle.load(f)
        else:
            # Legacy: rewind and unpickle the whole file.
            f.seek(0)
            idx = pickle.load(f)
    if not isinstance(idx, _Index):
        raise ValueError(f"unrecognized pickle payload in {path}: {type(idx).__name__}")
    # Routes in scan mode are kept sorted by path string (see
    # _build_index_from_frames). Preserve that contract here so callers see
    # the same order whether the index came from a fresh scan or a reload.
    idx.routes.sort(key=str)
    return idx


class TagStore:
    """Read/write ``tags`` from sidecar files with optional index caching.

    Construct from one of:
      - an existing ``*.tag`` index file (fast repeated queries)
      - a directory, path-list JSON, list of paths, or single ``.npz`` file
        (scans and builds an in-memory index)
      - ``None`` (empty store; useful before adding sources programmatically)

    Mutations are atomic at the file level and update the in-memory index
    so subsequent queries see the changes.
    """

    def __init__(self, source: str | Path | Sequence[str | Path] | None = None) -> None:
        """Create a TagStore.

        Args:
            source: Path to a ``.tag`` index file, a directory, a path-list
                    JSON, a list of paths, a single ``.npz``, or ``None``.

        Raises:
            FileNotFoundError: if ``source`` looks like an index file (its
                name ends with ``.tag``) but the file is missing. Pass a
                directory / path-list / NPZ to scan, or build the index
                first with :meth:`TagStore.build_index`.
        """
        self._source = source
        self._index: _Index | None = None
        self._lock = threading.RLock()

        if source is None:
            return

        # Pickled index takes priority over anything else. Detection is by
        # suffix ``.tag`` (extension only — anything else is an NPZ-shaped
        # source). A missing .tag file is a clear caller mistake, so we
        # raise before falling through to scan.
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.name.endswith(".tag"):
                if not p.is_file():
                    raise FileNotFoundError(
                        f"index file not found: {p}. Pass a directory, "
                        f"path-list JSON, list of paths, or single .npz "
                        f"to scan, or build the index first with "
                        f"TagStore.build_index(source, {p.name!r})."
                    )
                self._index = _load_index_from_pickle(p)
                return

        # Everything else is treated as a scan source (path / dir / path-list).
        if isinstance(source, (list, tuple)):
            self._scan_source(list(source))
        else:
            self._scan_source(source)

    def _require_index(self) -> _Index:
        """Return the in-memory index or raise with a uniform hint."""
        if self._index is None:
            raise ValueError(_NO_INDEX_HINT)
        return self._index

    @staticmethod
    def _check_granularity(g: str) -> str:
        if g not in _GRANULARITIES:
            raise ValueError(f"bad granularity {g!r}; expected one of {_GRANULARITIES}")
        return g

    @property
    def source(self) -> str | Path | None:
        """The source this store was initialized with."""
        return self._source

    # --- index building ---------------------------------------------------

    def _scan_source(self, source: str | Path | Sequence[str | Path]) -> None:
        """Scan source and build in-memory index."""
        frames: list[Path]
        if isinstance(source, (list, tuple)):
            frames = []
            for s in source:
                frames.extend(expand_source(s, sort=False))
        else:
            frames = expand_source(source, sort=False)
        self._index = _build_index_from_frames(frames)

    @staticmethod
    def build_index(
        source: str | Path | Sequence[str | Path],
        output: str | Path,
    ) -> "TagStore":
        """Scan *source*, build an in-memory index, and pickle it to *output*.

        Returns a TagStore holding the same index in memory. *output* is
        written atomically; the caller is free to load it again later with
        ``TagStore("path/to/index.tag")``.

        Args:
            source: Directory, path-list JSON, list of paths, or single NPZ.
            output: Where to write the pickled index. The file name MUST
                    end with ``.tag``: ``TagStore`` only reloads files whose
                    suffix is ``.tag``, so a ``.pkl`` or unsuffixed file
                    would be re-scanned as a dataset and silently fail.
        """
        out = Path(output)
        if not out.name.endswith(".tag"):
            raise ValueError(
                f"output must end with '.tag' so TagStore can reload it; got {out.name!r}"
            )
        store = TagStore(source)
        if store._index is None:
            raise ValueError("source produced no frames; cannot build index")
        _save_index_to_pickle(store._index, out)
        return store

    # --- accessors --------------------------------------------------------

    def npz_paths(self) -> list[Path]:
        """Return all NPZ paths in the index, in scan order (not sorted)."""
        return list(self._require_index().frames)

    def route_paths(self) -> list[Path]:
        """Return all route directories in the index, sorted by path string."""
        return list(self._require_index().routes)

    def has_index(self) -> bool:
        """True if this store has a loaded index."""
        return self._index is not None

    # --- mutate -----------------------------------------------------------

    def _mutate_frames(
        self,
        frame_filter: FrameFilter = None,
        scope: Scope = None,
    ) -> list[Path]:
        """All frames in index matching filter and scope, with sidecar preflight.

        Args:
            frame_filter: Optional filter - either (min, max) tuple for frame number
                         range, a glob pattern string, or None for all frames.
            scope: Limit to specific npz/route paths (None = entire index).
        """
        if self._index is None:
            return []

        scope_npzs = self._resolve_scope(scope, granularity="frame")
        frames = [f for f in self._index.frames if f in scope_npzs]

        if frame_filter is not None:
            frames = [f for f in frames if _match_frame_filter(f, frame_filter)]

        missing = [sidecar_path(npz) for npz in frames if not sidecar_path(npz).is_file()]
        if missing:
            preview = ", ".join(str(p) for p in missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(f"missing sidecar(s) before mutate: {preview}{more}")
        return frames

    def _update_index_for(self, npz: Path) -> None:
        """Update in-memory index for a mutated NPZ.

        Uses precomputed route_tag_counts for O(1) tag presence checks.
        Raises RuntimeError if *npz* is not in the route index — that means
        a mutation slipped past `_mutate_frames`'s scope check, or someone
        called this helper with a path outside the authoritative set.
        """
        idx = self._index
        if idx is None:
            return
        new_tags = read_tags(npz)
        new_tags_set = frozenset(new_tags)
        old_tags_set = idx.frame_tags.get(npz, frozenset())

        idx.frame_tags[npz] = new_tags_set

        route = idx.route_of_frame.get(npz)
        if route is None:
            raise RuntimeError(
                f"_update_index_for: {npz} not in route index; "
                f"refusing to mutate an inconsistent in-memory state"
            )

        # For each removed tag, check count to decide whether to remove from route index
        for tag in old_tags_set - new_tags_set:
            route_counts = idx.route_tag_counts.get(tag, {})
            if route in route_counts:
                route_counts[route] -= 1
                count = route_counts[route]
                if count <= 0:
                    # No frames in this route have this tag anymore
                    idx.tag_to_routes.get(tag, set()).discard(route)
                    idx.tag_to_frames.get(tag, set()).discard(npz)
                    # Clean up zero count
                    del route_counts[route]

        # Add new tags
        for tag in new_tags_set - old_tags_set:
            idx.tag_to_routes.setdefault(tag, set()).add(route)
            idx.tag_to_frames.setdefault(tag, set()).add(npz)
            idx.route_tag_counts[tag][route] = idx.route_tag_counts[tag].get(route, 0) + 1

        # Update route_tags
        idx.route_tags[route] = frozenset(
            (idx.route_tags.get(route, frozenset()) - old_tags_set) | new_tags_set
        )

    # --- scope resolution -------------------------------------------------

    def _resolve_scope(
        self,
        scope: Scope,
        granularity: Granularity = "route",
    ) -> set[Path]:
        """Turn a Scope into a set of paths at the requested granularity.

        A Scope is the "which routes?" half of every query / mutation. It is
        intentionally loose: a single path, a sequence (list/tuple/...) of
        paths or other TagStores, another TagStore, or ``None`` (meaning
        "the whole index"). Anything outside the index is silently dropped —
        scope never adds new frames, it only narrows.

        Output:
            - granularity="route" → set of route directories
            - granularity="frame" → set of .npz files

        Resolution is delegated to _expand_item (per path) or _all_at
        (whole index / other store). See those docstrings for the rules.
        """
        idx = self._require_index()

        if scope is None:
            return self._all_at(granularity)
        if isinstance(scope, TagStore):
            # A route that exists in scope._index but not in self._index
            # has no effect on self — silently drop it.
            their_paths = self._all_at(granularity, source=scope)
            return self._intersect_with_self(their_paths, granularity)

        # Sequence of items: each item is a path, a TagStore, or a nested
        # sequence. We recurse one level for nested lists rather than
        # support arbitrarily deep nesting.
        items: list
        if isinstance(scope, (list, tuple)):
            items = list(scope)
        else:
            items = [scope]
        flat: list = []
        for raw in items:
            if isinstance(raw, (list, tuple)):
                for nested in raw:
                    flat.append(nested)
            elif isinstance(raw, TagStore):
                flat.extend(self._all_at(granularity, source=raw))
            else:
                flat.append(raw)
        out: set[Path] = set()
        for raw in flat:
            out.update(self._expand_item(raw, granularity))
        return out

    def _intersect_with_self(
        self,
        paths: set[Path],
        granularity: Granularity,
    ) -> set[Path]:
        """Keep only paths that exist in self._index, at the given granularity."""
        if granularity == "route":
            return {r for r in paths if r in self._index.frames_of_route}
        return {f for f in paths if f in self._index.frame_tags}

    def _all_at(
        self,
        granularity: Granularity,
        *,
        source: "TagStore | None" = None,
    ) -> set[Path]:
        """Return every path at the requested granularity from one index.

        Used when scope is None ("whole index") or another TagStore
        ("take routes from that other store's index"). For granularity="route"
        the result is the set of routes that have at least one indexed frame;
        for granularity="frame" it is the set of all indexed NPZ paths.
        """
        idx = source._index if source is not None else self._index
        if idx is None:
            return set()
        if granularity == "route":
            return {idx.route_of_frame[npz] for npz in idx.frames}
        return set(idx.frames)

    def _expand_item(self, item, granularity: Granularity) -> set[Path]:
        """Resolve one scope path to paths at the requested granularity.

        Order of attempts — first match wins:

        1. **Route path in the index** (O(1) dict lookup in
           ``frames_of_route``). The user gave us a route directory that we
           already know about, so we trust the index without touching disk.

        2. **NPZ path in the index** (O(1) dict lookup in ``route_of_frame``).
           The user gave us an NPZ; we resolve it to its containing route
           when granularity is "route", or return the NPZ itself when
           granularity is "frame".

        3. **Fallback** (directory, path-list JSON, or any other path):
           call ``expand_source`` to list NPZ files on disk, then keep only
           the ones that are also in the index. This is the slow path and is
           only reached for sources the index does not already know about.

        Paths that resolve to nothing in the index are silently dropped, in
        line with "scope never adds frames outside source".

        Output:
            - granularity="route" → set of route directories
            - granularity="frame" → set of .npz files
        """
        p = Path(item)
        known_routes = self._index.frames_of_route
        route_of_frame = self._index.route_of_frame

        # Fast path 1: route path already in index → O(1) dict lookup
        if p in known_routes:
            if granularity == "route":
                return {p}
            return set(known_routes[p])

        # Fast path 2: npz path already in index → O(1) dict lookup
        if p.suffix == ".npz" and p in route_of_frame:
            if granularity == "route":
                return {route_of_frame[p]}
            return {p}

        # Fallback: directory / path list / unknown path
        npzs = {npz for npz in expand_source(p) if npz in route_of_frame}
        if granularity == "route":
            return {route_of_frame[npz] for npz in npzs}
        return npzs

    # --- mutation ---------------------------------------------------------

    def add_tags(
        self,
        tags: Sequence[str],
        *,
        frame_filter: FrameFilter = None,
        scope: Scope = None,
    ) -> int:
        """Union *tags* onto matching frames. Returns files changed.

        Idempotent in side effects: a frame that already has every requested
        tag is not rewritten and is not counted in the return value.

        Args:
            tags: Tags to add
            frame_filter: Optional filter - (min, max) tuple for frame number range,
                         or glob pattern string (e.g. "*_0000003[1-9]*.npz")
            scope: Limit to specific routes (None = entire index)

        Raises:
            ValueError: if no index is loaded (empty store). Use
                ``TagStore(source)`` or ``TagStore.build_index(source, ...)``
                before mutating.
            FileNotFoundError: if any matching frame lacks a sidecar.
        """
        self._require_index()
        to_add = normalize_tags(tags)
        to_add_set = frozenset(to_add)
        n = 0
        with self._lock:
            for npz in self._mutate_frames(frame_filter=frame_filter, scope=scope):
                current = read_tags(npz)
                if to_add_set - frozenset(current):
                    merged = normalize_tags(list(current) + to_add)
                    write_tags(npz, merged)
                    self._update_index_for(npz)
                    n += 1
        return n

    def remove_tags(
        self,
        tags: Sequence[str],
        *,
        frame_filter: FrameFilter = None,
        scope: Scope = None,
    ) -> int:
        """Delete exact tag strings from matching frames. Returns files changed.

        Tags must be full ``dim:value`` strings; partial matches (e.g.
        removing ``site`` from ``site:1423``) are not supported. Use
        :meth:`remove_dimension` for that.

        Args:
            tags: Tags to remove
            frame_filter: Optional filter - (min, max) tuple for frame number range,
                         or glob pattern string
            scope: Limit to specific routes (None = entire index)

        Raises:
            ValueError: if no index is loaded.
            FileNotFoundError: if any matching frame lacks a sidecar.
        """
        self._require_index()
        # Validate upfront — fail fast on malformed input.
        to_remove = set(normalize_tags(tags))
        n = 0
        with self._lock:
            for npz in self._mutate_frames(frame_filter=frame_filter, scope=scope):
                current = read_tags(npz)
                remaining = [t for t in current if t not in to_remove]
                if len(remaining) != len(current):
                    write_tags(npz, remaining)
                    self._update_index_for(npz)
                    n += 1
        return n

    def remove_dimension(
        self,
        dimension: str,
        *,
        scope: Scope = None,
        frame_filter: FrameFilter = None,
    ) -> int:
        """Delete every tag whose dimension is *dimension*. Returns files changed.

        Args:
            dimension: Dimension name (must match ``^[a-z0-9_]+$``).
            scope: Limit to specific routes (None = entire index).
            frame_filter: Optional filter - (min, max) tuple for frame number range,
                         or glob pattern string.

        Raises:
            ValueError: if no index is loaded, or *dimension* fails the
                ``[a-z0-9_]+`` regex.
            FileNotFoundError: if any matching frame lacks a sidecar.
        """
        self._require_index()
        if not is_valid_dimension(dimension):
            raise ValueError(f"bad dimension {dimension!r}: expected [a-z0-9_]+")
        n = 0
        with self._lock:
            for npz in self._mutate_frames(frame_filter=frame_filter, scope=scope):
                current = read_tags(npz)
                # read_tags already validated every tag is "dim:value", so
                # splitting is safe and avoids a parse_tag call per tag.
                kept = [t for t in current if t.split(":", 1)[0] != dimension]
                if len(kept) != len(current):
                    write_tags(npz, kept)
                    self._update_index_for(npz)
                    n += 1
        return n

    def replace_tags(
        self,
        *,
        tag_pairs: Mapping[str, str],
        scope: Scope = None,
        frame_filter: FrameFilter = None,
    ) -> int:
        """Replace specific tags with specific other tags, atomically.

        *tag_pairs* maps ``old_tag → new_tag``. Both keys and values must be
        valid ``dim:value`` strings. For each matching frame: if the frame
        has ``old_tag``, swap it for ``new_tag``. Tags not present on the
        frame are silently skipped. Entries where ``old_tag == new_tag`` are
        no-ops. The whole per-frame replacement is done in memory and
        committed with a single atomic write to the sidecar.

        Args:
            tag_pairs: Mapping of ``dim:value`` → ``dim:value`` to rewrite.
            scope: Limit to specific routes (None = entire index).
            frame_filter: Optional filter - (min, max) tuple for frame number range,
                         or glob pattern string.

        Raises:
            ValueError: if no index is loaded, or any key/value fails
                ``parse_tag``.
            FileNotFoundError: if any matching frame lacks a sidecar.
        """
        self._require_index()
        # Validate all keys/values up front so a bad call fails fast and
        # without partial work.
        validated: dict[str, str] = {}
        for old, new in tag_pairs.items():
            parse_tag(old)  # raises ValueError on invalid
            parse_tag(new)
            validated[old] = new
        n = 0
        with self._lock:
            for npz in self._mutate_frames(frame_filter=frame_filter, scope=scope):
                current = read_tags(npz)
                current_set = set(current)
                # Skip frames where the rewrite would be a no-op: old tag is
                # absent, or the new tag is already present (a prior replace
                # already did the work). Calling replace twice is then
                # idempotent in both side effects and return value.
                if not any(old in current_set for old in validated):
                    continue
                new_list: list[str] = []
                changed = False
                seen: set[str] = set()
                for t in current:
                    if t in validated and t != validated[t]:
                        repl = validated[t]
                        if repl not in seen:
                            new_list.append(repl)
                            seen.add(repl)
                        # The replacement counts as a change only when the
                        # tag set actually moves: old was present and new was
                        # not, OR the new tag was already on the frame so the
                        # old one is simply removed without gain — that case
                        # leaves the set unchanged and we shouldn't write.
                        if repl not in current_set:
                            changed = True
                    else:
                        if t not in seen:
                            new_list.append(t)
                            seen.add(t)
                if changed:
                    write_tags(npz, new_list)
                    self._update_index_for(npz)
                    n += 1
        return n

    def add_tags_to_route(
        self,
        tags: Sequence[str],
        route: str | Path,
        *,
        frame_filter: FrameFilter = None,
    ) -> int:
        """Merge *tags* into every frame of a known route. Returns files changed.

        Equivalent to ``add_tags(tags, scope=route, frame_filter=frame_filter)``.
        The difference is the failure mode when *route* is not in the index:
        this method raises ``ValueError`` immediately, while ``add_tags`` with
        ``scope=unknown_route`` would silently drop the unknown route via
        scope fallback. Use this when "tagging frames the index doesn't know
        about" would silently make the new tags invisible to subsequent
        queries.

        Args:
            tags: Tags to add (merged into each frame's existing tags).
            route: Route directory. Must be present in the index.
            frame_filter: Optional filter (range or glob) applied to that
                         route's frames.

        Raises:
            ValueError: if no index is loaded, or *route* is not in the index.
        """
        idx = self._require_index()
        route_path = Path(route)
        if route_path not in idx.frames_of_route:
            raise ValueError(f"route not in index: {route_path}")
        # Reuse the standard scope path so preflight + index-update logic
        # stays in one place. The scope is a single known route.
        return self.add_tags(tags, frame_filter=frame_filter, scope=route_path)

    # --- query ------------------------------------------------------------

    def tags_of(
        self,
        *,
        scope: Scope = None,
        dimensions: Sequence[str] | None = None,
        granularity: Granularity = "route",
    ) -> list[str]:
        """Return the union of tags visible to the call.

        Unified "give me tags" accessor for routes or frames.

        Args:
            scope: Limit to specific routes (None = entire index).
            dimensions: Optional list of dimension prefixes to keep. Tags
                       whose dimension is not in *dimensions* are dropped.
                       None means keep every tag.
            granularity: ``"route"`` returns route-level unions; ``"frame"``
                         returns per-NPZ tags.

        Returns:
            Sorted unique list of ``dim:value`` strings.
        """
        idx = self._require_index()
        self._check_granularity(granularity)
        for d in dimensions or []:
            if not is_valid_dimension(d):
                raise ValueError(f"bad dimension {d!r}: expected [a-z0-9_]+")

        scope_set = self._resolve_scope(scope, granularity=granularity)
        if granularity == "frame":
            items = [f for f in idx.frames if f in scope_set]
            tags_iter = (idx.frame_tags[f] for f in items)
        else:
            items = [r for r in idx.routes if r in scope_set]
            tags_iter = (idx.route_tags[r] for r in items)

        out: set[str] = set()
        if dimensions is None:
            for s in tags_iter:
                out.update(s)
        else:
            dim_set = set(dimensions)
            for s in tags_iter:
                for t in s:
                    if parse_tag(t)[0] in dim_set:
                        out.add(t)
        return sorted(out)

    def query(
        self,
        clause: Clause | None = None,
        *,
        granularity: Granularity = "route",
        scope: Scope = None,
    ) -> list[Path]:
        """Return matching routes (default) or NPZ frames.

        Args:
            clause: Optional. A ``"dim:value"`` string, or one of
                ``{"all": [...]}`` / ``{"any": [...]}`` / ``{"not": ...}``.
                ``None`` (default) means "everything in scope".
            granularity: "route" (default) uses union semantics; "frame"
                requires co-occurrence (every clause must match the same
                frame).
            scope: Limit to specific routes (None = entire index)
        """
        idx = self._require_index()
        self._check_granularity(granularity)
        scope_set = self._resolve_scope(scope, granularity=granularity)

        # Convenience "show me everything in scope" — saves callers a
        # unique-dimension-and-pick-all workaround.
        if clause is None:
            if granularity == "frame":
                return [p for p in idx.frames if p in scope_set]
            return [r for r in idx.routes if r in scope_set]

        if granularity == "frame":
            if isinstance(clause, str):
                parse_tag(clause)
                return [
                    p
                    for p in idx.frames
                    if clause in idx.frame_tags.get(p, frozenset()) and p in scope_set
                ]
            return [
                p
                for p in idx.frames
                if self._match(clause, set(idx.frame_tags.get(p, frozenset()))) and p in scope_set
            ]

        if isinstance(clause, str):
            parse_tag(clause)
            hit = idx.tag_to_routes.get(clause, set())
            return [r for r in idx.routes if r in hit and r in scope_set]
        return [
            r for r in idx.routes if self._match(clause, set(idx.route_tags[r])) and r in scope_set
        ]

    def group_by(
        self,
        dimensions: str | Sequence[str],
        clause: Clause | None = None,
        *,
        granularity: Granularity = "route",
        drop_missing: bool = False,
        scope: Scope = None,
    ) -> list[Bucket]:
        """Group routes (default) or frames by tag dimensions.

        Args:
            dimensions: One or more tag dimensions to group by.
            clause: Optional filter clause applied before grouping.
            granularity: "route" (default) or "frame".
            drop_missing: If True, items missing any of the requested
                          dimensions are skipped from all buckets. Default
                          False — items missing a dimension still appear in
                          buckets for the other dimensions (with a ``None``
                          slot, displayed as ``-``). Under multi-value
                          dimensions this cartesian-product behaviour makes
                          ``bucket.count`` an "appearances" count rather than
                          "unique items in this cell"; use ``Bucket.members``
                          when you need exact counts.
            scope: Limit to specific npz paths/route(s) (None = entire index)

        Sort order:
            Buckets are sorted primarily by the first dimension in
            ``dimensions``, secondarily by the second, and so on. Within
            each slot the values use ``normalize_tags`` ordering (alphabetical
            over ``dim:value`` strings); ``None`` (item missing that
            dimension) sorts last in its slot. The sort respects the order
            of ``dimensions`` as given — it is not alphabetical across
            dimensions.
        """
        idx = self._require_index()
        self._check_granularity(granularity)

        dims = [dimensions] if isinstance(dimensions, str) else list(dimensions)
        for d in dims:
            if not is_valid_dimension(d):
                raise ValueError(f"bad dimension {d!r}: expected [a-z0-9_]+")
        scope_set = self._resolve_scope(scope, granularity=granularity)

        if granularity == "frame":
            items = [p for p in idx.frames if p in scope_set]
            if clause is not None:
                items = [p for p in items if self._match(clause, set(idx.frame_tags[p]))]
            tags_of = idx.frame_tags
        else:
            items = [r for r in idx.routes if r in scope_set]
            if clause is not None:
                items = [r for r in items if self._match(clause, set(idx.route_tags[r]))]
            tags_of = idx.route_tags

        buckets: dict[tuple[str | None, ...], list[Path]] = defaultdict(list)
        for item in items:
            tags = tags_of[item]
            per_dim: list[list[str | None]] = []
            item_skipped = False
            for dim in dims:
                vals = sorted({parse_tag(t)[1] for t in tags if parse_tag(t)[0] == dim})
                if not vals:
                    if drop_missing:
                        item_skipped = True
                        break
                    vals = [None]
                per_dim.append(vals)
            if item_skipped:
                continue
            for combo in product(*per_dim):
                buckets[combo].append(item)

        # Sort primarily by dims[0] in combo[0], then dims[1] in combo[1], …
        # None sorts last in its slot.
        def sort_key(kv: tuple[tuple[str | None, ...], list[Path]]) -> tuple:
            combo = kv[0]
            # Each slot: None sorts after strings. We give None the highest
            # sortable value by checking each slot independently.
            key: list = []
            for slot in combo:
                if slot is None:
                    key.append((1, ""))
                else:
                    key.append((0, slot))
            return tuple(key)

        out: list[Bucket] = []
        for combo, members_list in sorted(buckets.items(), key=sort_key):
            values = {dims[i]: combo[i] for i in range(len(dims))}
            uniq: list[Path] = []
            seen: set[Path] = set()
            for m in members_list:
                if m not in seen:
                    seen.add(m)
                    uniq.append(m)
            uniq.sort(key=str)
            out.append(
                Bucket(
                    values=values,
                    count=len(uniq),
                    members=uniq,
                )
            )
        return out

    def _match(self, clause: Clause, tags: set[str]) -> bool:
        """Match a clause against a set of tags."""
        if isinstance(clause, str):
            parse_tag(clause)
            return clause in tags
        if not isinstance(clause, dict) or len(clause) != 1:
            raise ValueError(f"bad clause: {clause!r} (operators: {sorted(_CLAUSE_OPS)})")
        ((op, body),) = clause.items()
        if op == "all":
            if not body:
                raise ValueError("empty 'all' clause")
            return all(self._match(c, tags) for c in body)
        if op == "any":
            if not body:
                raise ValueError("empty 'any' clause")
            return any(self._match(c, tags) for c in body)
        if op == "not":
            return not self._match(body, tags)
        # Fall through to explicit error so a typo like {"And": [...]} raises.
        raise ValueError(f"bad clause op {op!r}; expected one of {sorted(_CLAUSE_OPS)}")
