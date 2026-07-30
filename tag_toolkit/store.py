"""TagStore — query and mutate sidecar ``tags`` for a ``source``.

Load builds a **route-level** cache first: a route carries a tag if **any** frame
under it does (within the expanded source). Query / group_by default to
``granularity="route"`` because most workflows care about routes, not individual
NPZ files.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Literal, Sequence

from .routes import route_of
from .sidecar import (
    drop_dimension,
    format_tag,
    normalize_tags,
    parse_tag,
    read_tags,
    sidecar_path,
    tag_values_for,
    write_tags,
)
from .source import Source, expand_source

Clause = Any  # "dim:value" | {"all": [...]} | {"any": [...]} | {"not": ...}
WriteMode = Literal["merge", "replace"]
Granularity = Literal["route", "frame"]


@dataclass
class Bucket:
    """One cell of :meth:`TagStore.group_by`."""

    values: dict[str, str | None]
    count: int
    members: list[Path] | None = None
    #: When ``granularity="route"``, number of NPZ frames in the bucket.
    frame_count: int | None = None

    def label(self, sep: str = " | ") -> str:
        return sep.join("-" if v is None else v for v in self.values.values())

    @property
    def routes(self) -> int:
        """Route count for route-granularity buckets."""
        return self.count


@dataclass
class _Index:
    """In-memory index over one expanded source."""

    frames: list[Path]
    frame_tags: dict[Path, frozenset[str]] = field(default_factory=dict)
    route_of_frame: dict[Path, Path] = field(default_factory=dict)
    frames_of_route: dict[Path, list[Path]] = field(default_factory=dict)
    route_tags: dict[Path, frozenset[str]] = field(default_factory=dict)
    tag_to_frames: dict[str, set[Path]] = field(default_factory=lambda: defaultdict(set))
    tag_to_routes: dict[str, set[Path]] = field(default_factory=lambda: defaultdict(set))
    routes: list[Path] = field(default_factory=list)


def format_buckets(buckets: list[Bucket], dimensions: Sequence[str]) -> str:
    """Plain-text table for CLI / notebooks.

    Per-row ``count`` is the size of that cell. ``TOTAL`` prefers the number of
    unique members across cells when ``members`` lists are present (so multi-value
    dimensions that place one route in several cells are not double-counted).
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
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    unique_members: set[Path] = set()
    have_members = True
    for b in buckets:
        if b.members is None:
            have_members = False
            break
        unique_members.update(b.members)
    total = len(unique_members) if have_members else sum(b.count for b in buckets)

    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    total_row: list[str] = []
    for i in range(len(headers)):
        if i == 0:
            total_row.append("TOTAL".ljust(widths[i]))
        elif headers[i] == "count":
            total_row.append(str(total).ljust(widths[i]))
        else:
            total_row.append("".ljust(widths[i]))
    lines.append("  ".join(total_row))
    return "\n".join(lines)


class TagStore:
    """Read/write ``tags`` on every NPZ under *source*, with a route-level cache."""

    def __init__(self, source: Source | None = None) -> None:
        self._source = source
        self._index: _Index | None = None

    def set_source(self, source: Source | None) -> None:
        self._source = source
        self.invalidate()

    def invalidate(self) -> None:
        self._index = None

    def _ensure_index(self, source: Source | None = None) -> _Index:
        if source is not None:
            return self._build_index(expand_source(source, sort=False))
        if self._source is None:
            raise ValueError("no source: pass source=... or construct TagStore(source=...)")
        if self._index is None:
            self._index = self._build_index(expand_source(self._source, sort=False))
        return self._index

    def _build_index(self, frames: list[Path]) -> _Index:
        """Read every sidecar once; build route unions and inverted indexes.

        Route unions / counts are relative to the expanded *frames* list (the
        current source), not necessarily every NPZ on disk under each bag.
        """
        idx = _Index(frames=frames)
        route_tag_sets: dict[Path, set[str]] = defaultdict(set)

        for npz in frames:
            tags = frozenset(self._safe_tags(npz))
            idx.frame_tags[npz] = tags
            route = route_of(npz)
            idx.route_of_frame[npz] = route
            idx.frames_of_route.setdefault(route, []).append(npz)
            route_tag_sets[route].update(tags)
            for tag in tags:
                idx.tag_to_frames[tag].add(npz)

        # Preserve first-seen route order (follows path-list / walk order).
        seen_routes: list[Path] = []
        seen: set[Path] = set()
        for npz in frames:
            route = idx.route_of_frame[npz]
            if route not in seen:
                seen.add(route)
                seen_routes.append(route)
        idx.routes = seen_routes

        for route, tag_set in route_tag_sets.items():
            frozen = frozenset(tag_set)
            idx.route_tags[route] = frozen
            for tag in frozen:
                idx.tag_to_routes[tag].add(route)
        return idx

    @staticmethod
    def _safe_tags(npz: Path) -> list[str]:
        """Read tags, dropping invalid strings (warn once per bad tag)."""
        raw = read_tags(npz)
        good: list[str] = []
        for tag in raw:
            try:
                parse_tag(tag)
            except ValueError:
                warnings.warn(
                    f"skipping invalid tag {tag!r} in {sidecar_path(npz)}",
                    stacklevel=2,
                )
                continue
            good.append(tag)
        return normalize_tags(good)

    def _mutate_frames(self, source: Source | None) -> list[Path]:
        """Frames under *source*, with a sidecar preflight before any write."""
        frames = self.npz_paths(source)
        missing = [sidecar_path(npz) for npz in frames if not sidecar_path(npz).is_file()]
        if missing:
            preview = ", ".join(str(p) for p in missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(f"missing sidecar(s) before mutate: {preview}{more}")
        return frames

    def npz_paths(self, source: Source | None = None) -> list[Path]:
        return list(self._ensure_index(source).frames)

    def route_paths(self, source: Source | None = None) -> list[Path]:
        return list(self._ensure_index(source).routes)

    # --- mutate -----------------------------------------------------------

    def add_tags(self, source: Source | None, tags: Sequence[str]) -> int:
        """Union *tags* onto every sidecar under *source*. Returns files changed."""
        to_add = set(normalize_tags(tags))
        n = 0
        for npz in self._mutate_frames(source):
            current = set(self._safe_tags(npz))
            merged = normalize_tags(current | to_add)
            if merged != normalize_tags(current):
                write_tags(npz, merged)
                n += 1
        self.invalidate()
        return n

    def remove_tags(self, source: Source | None, tags: Sequence[str]) -> int:
        to_remove = set(normalize_tags(tags))
        n = 0
        for npz in self._mutate_frames(source):
            current = self._safe_tags(npz)
            merged = [t for t in current if t not in to_remove]
            if merged != current:
                write_tags(npz, normalize_tags(merged))
                n += 1
        self.invalidate()
        return n

    def replace_tags(
        self,
        source: Source | None,
        *,
        dimension: str,
        values: Sequence[str],
    ) -> int:
        """Clear *dimension* on every frame, then add ``dimension:v`` for each v."""
        new_tags = [format_tag(dimension, v) for v in values]
        n = 0
        for npz in self._mutate_frames(source):
            current = self._safe_tags(npz)
            merged = normalize_tags(list(drop_dimension(current, dimension)) + new_tags)
            if merged != normalize_tags(current):
                write_tags(npz, merged)
                n += 1
        self.invalidate()
        return n

    def set_tags(
        self,
        source: Source | None,
        tags: Sequence[str],
        *,
        mode: WriteMode = "merge",
    ) -> int:
        normalized = normalize_tags(tags)
        n = 0
        for npz in self._mutate_frames(source):
            current = self._safe_tags(npz)
            if mode == "replace":
                if normalize_tags(current) != normalized:
                    write_tags(npz, normalized)
                    n += 1
            else:
                merged = normalize_tags(set(current) | set(normalized))
                if merged != normalize_tags(current):
                    write_tags(npz, merged)
                    n += 1
        self.invalidate()
        return n

    # --- query ------------------------------------------------------------

    def tags_for(self, path: str | Path, source: Source | None = None) -> list[str]:
        """Tags for one NPZ, or the **union** of tags for a route directory.

        For a route directory: use the cached (source-relative) union when that
        route is indexed; otherwise scan the directory on disk.
        """
        p = Path(path)
        if p.suffix == ".npz":
            idx = self._index
            if source is None and idx is not None and p in idx.frame_tags:
                return normalize_tags(idx.frame_tags[p])
            return self._safe_tags(p)

        route = route_of(p)
        idx: _Index | None = None
        if source is not None:
            idx = self._build_index(expand_source(source, sort=False))
        elif self._source is not None:
            idx = self._ensure_index()
        elif self._index is not None:
            idx = self._index

        if idx is not None and route in idx.route_tags:
            return normalize_tags(idx.route_tags[route])

        tags: set[str] = set()
        for npz in expand_source(p, sort=False):
            tags.update(self._safe_tags(npz))
        return normalize_tags(tags)

    def tag_values(
        self, path: str | Path, dimension: str, source: Source | None = None
    ) -> list[str]:
        return tag_values_for(self.tags_for(path, source=source), dimension)

    def query(
        self,
        clause: Clause,
        source: Source | None = None,
        *,
        granularity: Granularity = "route",
    ) -> list[Path]:
        """Return matching routes (default) or NPZ frames."""
        idx = self._ensure_index(source)
        if granularity == "frame":
            if isinstance(clause, str):
                parse_tag(clause)
                return [p for p in idx.frames if p in idx.tag_to_frames.get(clause, ())]
            return [p for p in idx.frames if self._match(clause, set(idx.frame_tags[p]))]

        # route granularity — prefer inverted index for a bare tag
        if isinstance(clause, str):
            parse_tag(clause)
            hit = idx.tag_to_routes.get(clause, set())
            return [r for r in idx.routes if r in hit]
        return [r for r in idx.routes if self._match(clause, set(idx.route_tags[r]))]

    def group_by(
        self,
        dimensions: str | Sequence[str],
        source: Source | None = None,
        *,
        members: bool | None = None,
        clause: Clause | None = None,
        granularity: Granularity = "route",
    ) -> list[Bucket]:
        dims = [dimensions] if isinstance(dimensions, str) else list(dimensions)
        idx = self._ensure_index(source)

        if granularity == "frame":
            items = list(idx.frames)
            if clause is not None:
                items = [p for p in items if self._match(clause, set(idx.frame_tags[p]))]
            tags_of = idx.frame_tags
            frame_count_of = None
            include_members = bool(members) if members is not None else False
        else:
            items = list(idx.routes)
            if clause is not None:
                items = [r for r in items if self._match(clause, set(idx.route_tags[r]))]
            tags_of = idx.route_tags
            frame_count_of = idx.frames_of_route
            include_members = True if members is None else members

        buckets: dict[tuple[str | None, ...], list[Path]] = defaultdict(list)
        for item in items:
            tags = tags_of[item]
            per_dim: list[list[str | None]] = []
            for dim in dims:
                vals = tag_values_for(tags, dim)
                per_dim.append(vals if vals else [None])
            for combo in product(*per_dim):
                buckets[combo].append(item)

        out: list[Bucket] = []
        for combo, members_list in sorted(
            buckets.items(), key=lambda kv: tuple("" if x is None else x for x in kv[0])
        ):
            values = {dims[i]: combo[i] for i in range(len(dims))}
            # preserve first-seen order within the bucket
            uniq: list[Path] = []
            seen: set[Path] = set()
            for m in members_list:
                if m not in seen:
                    seen.add(m)
                    uniq.append(m)
            frame_count = None
            if frame_count_of is not None:
                frame_count = sum(len(frame_count_of.get(r, ())) for r in uniq)
            out.append(
                Bucket(
                    values=values,
                    count=len(uniq),
                    members=uniq if include_members else None,
                    frame_count=frame_count,
                )
            )
        return out

    def _match(self, clause: Clause, tags: set[str]) -> bool:
        if isinstance(clause, str):
            parse_tag(clause)
            return clause in tags
        if not isinstance(clause, dict) or len(clause) != 1:
            raise ValueError(f"bad clause: {clause!r}")
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
        raise ValueError(f"unknown clause op {op!r}")
