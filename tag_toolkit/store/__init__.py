"""TagStore — query and mutate ``tags`` from sidecar JSON files via SQLite.

Tags travel with each NPZ's sidecar JSON. The store scans a source,
builds a SQLite index, and exposes query / mutation on top of that index.

If ``source`` is a ``.db`` / ``.sqlite`` / ``.tags.db`` file the SQLite
index is opened there and all operations are persisted. Otherwise the
index lives in memory; mutations stay in memory and can be exported to a
file with ``store.export_index(path)``.

Read methods are lock-free; concurrent writers are serialised via an
internal ``RLock``. The SQLite WAL ensures crash-safe writes without
requiring per-file ``fsync`` on the data directory.

Index ownership contract:
    The SQLite database is the **authoritative** view of tag state.
    Every mutation verifies that the sidecar on disk matches what the
    index thinks is there — if it doesn't, the mutation aborts with
    :class:`StaleIndexError` and tells the caller to reconcile via
    :meth:`TagStore.reindex_tags`. The store never silently overwrites
    a sidecar it doesn't recognise and never creates a sidecar that
    didn't exist before.

    The atomic-write / fsync / verify-then-write protocol lives in
    :mod:`sidecar`; ``TagStore`` is responsible only for resolving scope,
    updating the SQLite index, and routing to :func:`sidecar.write_tags`.
    If you suspect the sidecar JSONs on disk have drifted from the
    SQLite index, call :meth:`TagStore.diff_index_against_disk` for a
    structured report and :meth:`TagStore.reindex_tags` to bring the
    index back in line.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ..routes import route_of
from ..sidecar import parse_tag, read_tags, sidecar_path
from ..source import expand_source

from ._db import _init_db, _is_db_path
from ._index import _IndexMixin
from ._mutate import _MutateMixin
from ._query import _QueryMixin, format_buckets
from ._scope import _ScopeResolver
from ._types import (
    Bucket,
    FrameTagDiff,
    IndexDiff,
    MutationResult,
)

__all__ = [
    "TagStore",
    "Bucket",
    "expand_source",
    "FrameTagDiff",
    "IndexDiff",
    "MutationResult",
    "format_buckets",
    "read_tags",
    "route_of",
]


class TagStore(_QueryMixin, _MutateMixin, _IndexMixin):
    """SQLite-backed store for tags on NPZ sidecar files.

    Construct from one of:
      - a ``.db`` / ``.sqlite`` / ``.tags.db`` file: opens/creates that
        SQLite database; all operations are persisted
      - a directory, path-list JSON, list of paths, or single ``.npz``:
        creates an in-memory SQLite index; mutations stay in memory;
        call ``export_index(path)`` to persist
      - ``None``: empty in-memory store

    Mutations are atomic at the file level and update the SQLite index
    immediately so subsequent queries see the changes.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path] | None = None,
    ) -> None:
        self._source = source
        self._route_tags_cache: dict[str, frozenset[str]] = {}
        self._lock = threading.RLock()
        self._conn_per_thread: dict[int, sqlite3.Connection] = {}

        if source is None:
            self._init_in_memory()
            return

        if _is_db_path(source):
            db_path = Path(source)
            if not db_path.exists():
                raise FileNotFoundError(f"database file not found: {db_path}")
            self._db_path = str(db_path.resolve())
            self._db_uri = None
            self._conn = self._open_thread_conn()
            self._conn_per_thread[threading.current_thread().ident] = self._conn
            self._warm_route_tags_cache()
            return

        self._init_in_memory()
        self.rebuild_index(source)

    def _init_in_memory(self) -> None:
        """Create a private per-process in-memory DB and remember its handle."""
        self._db_path = None
        self._db_uri = f"file:{uuid.uuid4().hex}?mode=memory&cache=private"
        self._conn = self._open_thread_conn()
        self._conn_per_thread[threading.current_thread().ident] = self._conn

    # --- internal helpers ---------------------------------------------------

    def _open_thread_conn(self) -> sqlite3.Connection:
        if self._db_uri:
            conn = sqlite3.connect(self._db_uri, uri=True)
        else:
            conn = sqlite3.connect(database=self._db_path, check_same_thread=False)
        _init_db(conn)
        return conn

    def _require_conn(self) -> sqlite3.Connection:
        thread_id = threading.current_thread().ident
        with self._lock:
            if thread_id not in self._conn_per_thread:
                self._conn_per_thread[thread_id] = self._open_thread_conn()
            return self._conn_per_thread[thread_id]

    def _warm_route_tags_cache(self) -> None:
        """Populate route_tags cache from DB on startup."""
        conn = self._require_conn()
        rows = conn.execute(
            """
            SELECT f.route, json_group_array(t.tag) AS tags
            FROM frames f
            JOIN tags t ON f.path = t.path
            GROUP BY f.route
            """
        ).fetchall()
        with self._lock:
            self._route_tags_cache = {
                row[0]: frozenset(json.loads(row[1])) if row[1] else frozenset()
                for row in rows
            }

    def _sync_route_tags_cache(self, route: str, added: set[str], removed: set[str]) -> None:
        """Update route_tags cache after a mutation. route must be a string key."""
        with self._lock:
            current = self._route_tags_cache.get(route, frozenset())
            self._route_tags_cache[route] = frozenset((current | added) - removed)

    def _resolve_scope(
        self,
        scope,
        granularity: str = "route",
    ) -> set[str]:
        """Turn a Scope into a set of route strings (or path strings)."""
        resolver = _ScopeResolver(self)
        return resolver.resolve(scope, granularity=granularity)

    @property
    def source(self) -> str | Path | None:
        """The source this store was initialized with."""
        return self._source

    # --- accessors ---------------------------------------------------------

    def npz_paths(self) -> list[Path]:
        """Return all NPZ paths in the index."""
        conn = self._require_conn()
        rows = conn.execute("SELECT path FROM frames").fetchall()
        return [Path(r[0]) for r in rows]

    def route_paths(self) -> list[Path]:
        """Return all route directories in the index, sorted by path string."""
        conn = self._require_conn()
        rows = conn.execute("SELECT DISTINCT route FROM frames ORDER BY route").fetchall()
        return [Path(r[0]) for r in rows]

    def has_index(self) -> bool:
        """True if this store has a DB (in-memory or persistent)."""
        return bool(self._conn_per_thread)

    def append_frames(self, npz_paths: list[Path]) -> int:
        """Add *npz_paths* to the index. **All paths must be new — existing frames raise ``ValueError``.**

        Uses ``INSERT OR IGNORE`` for both ``frames`` and ``tags`` rows after a
        pre-check that all paths are absent from the index. The caller must
        pre-filter duplicates (e.g. ``SELECT path FROM frames WHERE path IN (...)``)
        if needed; passing duplicates raises ``ValueError``.

        The ``_route_tags_cache`` is updated for all routes touched by the new
        frames after the insert.

        Returns the number of frames added.

        Raises:
            ValueError: if any path in *npz_paths* already exists in the index.

        Contrast with ``rebuild_index`` (full wipe + rescan) and ``reindex_tags``
        (re-reads sidecars for frames already in the index, refreshes tag
        content without changing the frame set).
        """
        if not npz_paths:
            return 0

        npz_strs = [str(npz) for npz in npz_paths]

        with self._write_transaction() as conn:
            # Fast pre-check via PK lookup — fail fast before any sidecar reads.
            placeholders = ",".join("?" * len(npz_strs))
            existing = conn.execute(
                f"SELECT path FROM frames WHERE path IN ({placeholders})",
                sorted(npz_strs),
            ).fetchall()
            if existing:
                dup_paths = [r[0] for r in existing]
                raise ValueError(
                    f"append_frames: {len(dup_paths)} path(s) already in index "
                    f"(pre-filter duplicates before calling): {dup_paths[:3]}"
                    f"{' ...' if len(dup_paths) > 3 else ''}"
                )

            frames_rows: list[tuple[str, str, int]] = []
            tags_rows: list[tuple[str, str, str, str]] = []
            for npz in npz_paths:
                side = sidecar_path(npz)
                tags_list = read_tags(npz)
                mtime = int(side.stat().st_mtime) if side.is_file() else 0
                npz_s = str(npz)
                frames_rows.append((npz_s, str(route_of(npz)), mtime))
                for tag in tags_list:
                    try:
                        dim, val = parse_tag(tag)
                    except ValueError:
                        dim, val = ("", tag)
                    tags_rows.append((npz_s, tag, dim, val))

            if frames_rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO frames (path, route, sidecar_mtime) VALUES (?, ?, ?)",
                    frames_rows,
                )
            if tags_rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO tags (path, tag, dim, val) VALUES (?, ?, ?, ?)",
                    tags_rows,
                )

            # Update _route_tags_cache: single query for all new frames, group by route.
            route_tag_sets: dict[str, set[str]] = defaultdict(set)
            placeholders = ",".join("?" * len(npz_strs))
            rows = conn.execute(
                f"SELECT f.route, t.tag FROM tags t "
                f"JOIN frames f ON t.path = f.path "
                f"WHERE t.path IN ({placeholders})",
                sorted(npz_strs),
            ).fetchall()
            for route, tag in rows:
                route_tag_sets[route].add(tag)
            for route, tags in route_tag_sets.items():
                current = self._route_tags_cache.get(route, frozenset())
                self._route_tags_cache[route] = frozenset(current | tags)

        return len(npz_paths)
