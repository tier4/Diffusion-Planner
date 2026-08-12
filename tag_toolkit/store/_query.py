"""Query operations for TagStore."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..sidecar import is_valid_dimension
from ._types import match_frame_filter

if TYPE_CHECKING:
    from ._types import Bucket, Clause, FrameFilter, Granularity


_GRANULARITIES = ("route", "frame")


def format_buckets(buckets: list["Bucket"], dimensions: Sequence[str]) -> str:
    """Plain-text table for CLI / notebooks."""
    headers = list(dimensions) + ["count"]
    n_cols = len(headers)
    rows = [
        [("-" if b.values.get(d) is None else str(b.values.get(d))) for d in dimensions]
        + [str(b.count)]
        for b in buckets
    ]
    cell_widths = [max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(n_cols)]
    lines = [
        "  ".join(headers[i].ljust(cell_widths[i]) for i in range(n_cols)),
        "  ".join("-" * cell_widths[i] for i in range(n_cols)),
    ]
    for row in rows:
        lines.append("  ".join(row[i].ljust(cell_widths[i]) for i in range(n_cols)))

    unique_members = {m for b in buckets for m in b.members}
    lines.append("  ".join("-" * cell_widths[i] for i in range(n_cols)))
    total_count = str(len(unique_members))
    # Without dimensions the "TOTAL" row is just a single merged count cell.
    # Match cell_widths[0] for 'TOTAL' when 0 dims; use cell_widths[1] (count
    # column) for the number. Both shapes are covered by the joined rendering.
    cells = ["TOTAL".ljust(cell_widths[0])]
    if n_cols >= 2:
        cells.append(total_count.ljust(cell_widths[1]))
        cells.extend("".ljust(cell_widths[i]) for i in range(2, n_cols))
    else:
        cells.append(total_count.ljust(cell_widths[0]))
    lines.append("  ".join(cells))
    return "\n".join(lines)


class _QueryMixin:
    """Mixin providing query methods for TagStore."""

    def _validate_granularity(self, g: "Granularity") -> None:
        if g not in _GRANULARITIES:
            raise ValueError(f"bad granularity {g!r}; expected one of {_GRANULARITIES}")

    def tags_of(
        self,
        *,
        scope=None,
        dimensions: Sequence[str] | None = None,
        granularity: "Granularity" = "route",
    ) -> list[str]:
        """Return the union of tags visible to the call."""
        self._require_conn()
        self._validate_granularity(granularity)
        for d in dimensions or []:
            if not is_valid_dimension(d):
                raise ValueError(f"bad dimension {d!r}: expected [a-z0-9_]+")

        scope_set = self._resolve_scope(scope, granularity=granularity)
        conn = self._require_conn()

        if granularity == "frame":
            if not scope_set:
                return []
            placeholders = ",".join("?" * len(scope_set))
            if dimensions is None:
                rows = conn.execute(
                    f"SELECT tag FROM tags WHERE path IN ({placeholders})",
                    sorted(scope_set),
                ).fetchall()
                return sorted(set(r[0] for r in rows))
            dim_placeholder = ",".join("?" * len(dimensions))
            rows = conn.execute(
                f"SELECT DISTINCT tag FROM tags "
                f"WHERE path IN ({placeholders}) AND dim IN ({dim_placeholder})",
                sorted(scope_set) + list(dimensions),
            ).fetchall()
            return sorted(set(r[0] for r in rows))

        if dimensions is None:
            out: set[str] = set()
            for route in scope_set:
                out.update(self._route_tags_cache.get(route, {}).keys())
            return sorted(out)

        # Route granularity + dimensions filter: single batched query, not
        # one SELECT per route (N+1).
        if not scope_set:
            return []
        routes_placeholder = ",".join("?" * len(scope_set))
        dim_placeholder = ",".join("?" * len(dimensions))
        rows = conn.execute(
            f"SELECT DISTINCT t.tag FROM tags t "
            f"JOIN frames f ON t.path = f.path "
            f"WHERE f.route IN ({routes_placeholder}) AND t.dim IN ({dim_placeholder})",
            sorted(scope_set) + list(dimensions),
        ).fetchall()
        return sorted(set(r[0] for r in rows))

    def query(
        self,
        clause: "Clause | None" = None,
        *,
        granularity: "Granularity" = "route",
        scope=None,
    ) -> list[Path]:
        """Return matching routes (default) or NPZ frames."""
        self._require_conn()
        self._validate_granularity(granularity)
        scope_set = self._resolve_scope(scope, granularity=granularity)

        if clause is None:
            return [Path(p) for p in sorted(scope_set)]

        if granularity == "frame":
            return [Path(p) for p in self._query_frame(clause, scope_set)]
        return [Path(r) for r in self._query_route(clause, scope_set)]

    def dim_values_for(
        self,
        npz_paths: "Sequence[Path]",
        dimension: str,
    ) -> dict[Path, str]:
        """Return ``{npz: val}`` for each frame that has a *dimension* tag.

        Used by callers that need to slice frames by dimension value
        without firing one query per frame. Frames without the dimension
        are simply absent from the returned dict.
        """
        self._require_conn()
        if not is_valid_dimension(dimension):
            raise ValueError(f"bad dimension {dimension!r}: expected [a-z0-9_]+")
        if not npz_paths:
            return {}
        conn = self._require_conn()
        placeholders = ",".join("?" * len(npz_paths))
        rows = conn.execute(
            f"SELECT path, val FROM tags WHERE path IN ({placeholders}) AND dim = ?",
            [str(p) for p in npz_paths] + [dimension],
        ).fetchall()
        return {Path(r[0]): r[1] for r in rows}

    def _query_route(self, clause: "Clause", scope_set: set[str]) -> list[str]:
        """Query routes matching clause."""
        return self._query_by_path(clause, scope_set, route_granularity=True)

    def _query_frame(self, clause: "Clause", scope_set: set[str]) -> list[str]:
        """Query frames matching clause."""
        return self._query_by_path(clause, scope_set, route_granularity=False)

    def _query_by_path(
        self,
        clause: "Clause",
        scope_set: set[str],
        *,
        route_granularity: bool,
    ) -> list[str]:
        """Shared clause-matching logic for route and frame granularities.

        Both granularities differ only in column names (``frames.route`` vs
        ``tags.path``) and how they project a wildcard (``dim:*``); pull the
        common shape out so the two callers stay identical.
        """
        if not scope_set:
            if isinstance(clause, dict) and clause:
                ((op, body),) = clause.items()
                if op == "not":
                    return []
            return []
        conn = self._require_conn()

        # String clause: equality, wildcard, or a single-key dict.
        if isinstance(clause, str):
            placeholders = ",".join("?" * len(scope_set))
            if clause.endswith(":*"):
                dim = clause[:-2]
                if not dim or ":" in dim:
                    raise ValueError(f"invalid wildcard dimension: {dim!r}")
                prefix = f"{dim}:"
                if route_granularity:
                    rows = conn.execute(
                        f"SELECT DISTINCT f.route FROM frames f "
                        f"JOIN tags t ON f.path = t.path "
                        f"WHERE f.route IN ({placeholders}) AND t.tag LIKE ?",
                        sorted(scope_set) + [f"{prefix}%"],
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"SELECT DISTINCT t.path FROM tags t "
                        f"WHERE t.path IN ({placeholders}) AND t.tag LIKE ?",
                        sorted(scope_set) + [f"{prefix}%"],
                    ).fetchall()
                return sorted(r[0] for r in rows)
            if route_granularity:
                rows = conn.execute(
                    f"SELECT DISTINCT f.route FROM frames f "
                    f"JOIN tags t ON f.path = t.path "
                    f"WHERE f.route IN ({placeholders}) AND t.tag=?",
                    sorted(scope_set) + [clause],
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT DISTINCT t.path FROM tags t "
                    f"WHERE t.path IN ({placeholders}) AND t.tag=?",
                    sorted(scope_set) + [clause],
                ).fetchall()
            return sorted(r[0] for r in rows)

        if not isinstance(clause, dict) or len(clause) != 1:
            raise ValueError(f"bad clause: {clause!r}")
        ((op, body),) = clause.items()
        if op == "all":
            if not body:
                raise ValueError("empty 'all' clause")
            working = set(scope_set)
            for c in body:
                working &= set(self._query_by_path(c, working, route_granularity=route_granularity))
            return sorted(working)
        if op == "any":
            if not body:
                raise ValueError("empty 'any' clause")
            out: set[str] = set()
            for c in body:
                out |= set(self._query_by_path(c, scope_set, route_granularity=route_granularity))
            return sorted(out)
        if op == "not":
            excluded = set(
                self._query_by_path(body, scope_set, route_granularity=route_granularity)
            )
            return sorted(scope_set - excluded)
        raise ValueError(f"bad clause op {op!r}")

    def group_by(
        self,
        dimensions: str | Sequence[str],
        clause: "Clause | None" = None,
        *,
        granularity: "Granularity" = "route",
        drop_missing: bool = False,
        scope=None,
    ) -> list["Bucket"]:
        """Group routes or frames by tag dimensions."""
        from ._types import Bucket

        self._require_conn()
        self._validate_granularity(granularity)
        dims = [dimensions] if isinstance(dimensions, str) else list(dimensions)
        for d in dims:
            if not is_valid_dimension(d):
                raise ValueError(f"bad dimension {d!r}: expected [a-z0-9_]+")
        scope_set = self._resolve_scope(scope, granularity=granularity)

        all_tags_in_scope: dict[str, tuple[str, str]] = {}
        conn = self._require_conn()
        if granularity == "frame":
            if scope_set:
                placeholders = ",".join("?" * len(scope_set))
                rows = conn.execute(
                    f"SELECT path, tag, dim, val FROM tags WHERE path IN ({placeholders})",
                    sorted(scope_set),
                ).fetchall()
            else:
                rows = []
        else:
            if scope_set:
                placeholders = ",".join("?" * len(scope_set))
                rows = conn.execute(
                    f"SELECT t.tag, t.dim, t.val FROM tags t "
                    f"JOIN frames f ON t.path = f.path "
                    f"WHERE f.route IN ({placeholders})",
                    sorted(scope_set),
                ).fetchall()
            else:
                rows = []

        if granularity == "frame":
            all_tags_in_scope = {r[1]: (r[2], r[3]) for r in rows}
            tags_by_path: dict[str, set[str]] = defaultdict(set)
            for r in rows:
                tags_by_path[r[0]].add(r[1])
        else:
            all_tags_in_scope = {r[0]: (r[1], r[2]) for r in rows}
            tags_by_path = self._route_tags_cache

        if clause is not None:
            if granularity == "frame":
                matched = set(self._query_frame(clause, scope_set))
                scope_set &= matched
            else:
                matched = set(self._query_route(clause, scope_set))
                scope_set &= matched

        items = sorted(scope_set)

        buckets: dict[tuple[str | None, ...], list[str]] = defaultdict(list)
        for item in items:
            tag_map = tags_by_path.get(item, {})
            item_tag_set = tag_map.keys() if isinstance(tag_map, dict) else tag_map
            per_dim: list[list[str | None]] = []
            item_skipped = False
            for dim in dims:
                vals: set[str] = set()
                for tag in item_tag_set:
                    info = all_tags_in_scope.get(tag)
                    if info and info[0] == dim:
                        vals.add(info[1])
                vals_sorted = sorted(vals)
                if not vals_sorted:
                    if drop_missing:
                        item_skipped = True
                        break
                    vals_sorted = [None]
                per_dim.append(vals_sorted)
            if item_skipped:
                continue
            for combo in product(*per_dim):
                buckets[combo].append(item)

        def sort_key(kv: tuple[tuple[str | None, ...], list[str]]) -> tuple:
            combo = kv[0]
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
            uniq = sorted(set(members_list), key=str)
            out.append(Bucket(values=values, members=[Path(p) for p in uniq]))
        return out
