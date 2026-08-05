"""Scope resolution for TagStore mutations and queries."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ..source import expand_source

if TYPE_CHECKING:
    from ._types import Granularity


class _ScopeResolver:
    """Helper class for resolving scope expressions to concrete route/path sets."""

    def __init__(self, store: "TagStore") -> None:
        self._store = store

    def resolve(
        self,
        scope,
        granularity: "Granularity" = "route",
    ) -> set[str]:
        """Turn a Scope into a set of route strings (or path strings).

        Items that resolve to nothing in the index are dropped silently; the
        aggregate ``UserWarning`` at the end summarises how many items were
        skipped rather than spamming one warning per missing path.
        """
        if scope is None:
            return self._all_at(granularity)
        if isinstance(scope, self._store.__class__):
            their_paths = self._all_at(granularity, source=scope)
            return self._intersect_with_self(their_paths, granularity)

        items: list
        if isinstance(scope, (list, tuple)):
            items = list(scope)
        else:
            items = [scope]
        flat: list = []
        for raw in items:
            if isinstance(raw, (list, tuple)):
                flat.extend(raw)
            elif isinstance(raw, self._store.__class__):
                flat.extend(self._all_at(granularity, source=raw))
            else:
                flat.append(raw)

        out: set[str] = set()
        missed: list = []
        for raw in flat:
            expanded = self._expand_item(raw, granularity)
            if not expanded:
                missed.append(raw)
            out.update(expanded)

        if missed:
            preview = ", ".join(repr(m) for m in missed[:3])
            extra = f" (+{len(missed) - 3} more)" if len(missed) > 3 else ""
            warnings.warn(
                f"{len(missed)} scope item(s) resolved to zero {granularity}s "
                f"(not in this store's index): {preview}{extra}",
                UserWarning,
                stacklevel=4,
            )
        return out

    def _intersect_with_self(
        self,
        paths: set[str],
        granularity: "Granularity",
    ) -> set[str]:
        """Keep only paths that exist in the DB."""
        if not paths:
            return set()
        conn = self._store._require_conn()
        placeholders = ",".join("?" * len(paths))
        if granularity == "route":
            rows = conn.execute(
                f"SELECT DISTINCT route FROM frames WHERE route IN ({placeholders})",
                sorted(paths),
            ).fetchall()
            return {r[0] for r in rows}
        rows = conn.execute(
            f"SELECT path FROM frames WHERE path IN ({placeholders})",
            sorted(paths),
        ).fetchall()
        return {r[0] for r in rows}

    def _all_at(self, granularity: "Granularity", *, source=None) -> set[str]:
        """Return every route/path from one index."""
        store = source if source is not None else self._store
        conn = store._require_conn()
        if granularity == "route":
            rows = conn.execute("SELECT DISTINCT route FROM frames").fetchall()
            return {r[0] for r in rows}
        rows = conn.execute("SELECT path FROM frames").fetchall()
        return {r[0] for r in rows}

    def _expand_item(self, item, granularity: "Granularity") -> set[str]:
        """Resolve one scope item to route strings (or path strings)."""
        p = Path(item)
        conn = self._store._require_conn()

        if granularity == "route":
            # Direct route lookup.
            row = conn.execute(
                "SELECT 1 FROM frames WHERE route=? LIMIT 1", (str(p),)
            ).fetchone()
            if row:
                return {str(p)}
            # Path -> route lookup.
            row = conn.execute(
                "SELECT route FROM frames WHERE path=? LIMIT 1", (str(p),)
            ).fetchone()
            if row:
                return {row[0]}
        else:
            row = conn.execute(
                "SELECT 1 FROM frames WHERE path=? LIMIT 1", (str(p),)
            ).fetchone()
            if row:
                return {str(p)}

        # Fallback: walk the filesystem and intersect with the DB.
        npz_set = self._expand_npzs(p)
        if not npz_set:
            return set()
        placeholders = ",".join("?" * len(npz_set))
        if granularity == "route":
            rows = conn.execute(
                f"SELECT DISTINCT route FROM frames WHERE path IN ({placeholders})",
                sorted(npz_set),
            ).fetchall()
            return {r[0] for r in rows}
        rows = conn.execute(
            f"SELECT path FROM frames WHERE path IN ({placeholders})",
            sorted(npz_set),
        ).fetchall()
        return {r[0] for r in rows}

    def _expand_npzs(self, p: Path) -> set[str]:
        """Return absolute string paths for NPZs under p that are in the DB."""
        conn = self._store._require_conn()
        try:
            candidates = {str(npz) for npz in expand_source(p)}
        except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError):
            # The scope path is broken (missing, a non-directory, unreadable,
            # or otherwise unparseable). Scope resolution drops it; the
            # caller's aggregate warning at the end of ``resolve`` will
            # surface the count. Other exceptions (e.g. KeyboardInterrupt)
            # still propagate.
            return set()
        if not candidates:
            return set()
        placeholders = ",".join("?" * len(candidates))
        rows = conn.execute(
            f"SELECT path FROM frames WHERE path IN ({placeholders})",
            sorted(candidates),
        ).fetchall()
        return {r[0] for r in rows}
