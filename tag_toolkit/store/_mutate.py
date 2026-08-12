"""Mutation operations for TagStore."""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..routes import extract_frame_number, route_of
from ..sidecar import (
    StaleIndexError,
    is_valid_dimension,
    normalize_tags,
    parse_tag,
    read_tags,
    sidecar_path,
    write_tags,
)
from ._types import MutationResult, match_frame_filter

if TYPE_CHECKING:
    from ._types import Scope


def _resolved_sync(sync: bool | None) -> bool:
    """Single source of truth for the ``sync`` default (``True``)."""
    return True if sync is None else sync


def _check_stale(conn, npz: Path, npz_str: str) -> None:
    """Raise :class:`StaleIndexError` if the sidecar has drifted since indexing.

    This is the *index-vs-disk* half of the verify-then-write protocol. The
    *read-vs-write* half (catching drift between our own read and our own
    atomic write) lives in :func:`sidecar.write_tags` via ``expected_tags``.
    """
    side = sidecar_path(npz)
    if not side.is_file():
        return
    expected_mtime = conn.execute(
        "SELECT sidecar_mtime FROM frames WHERE path=?", (npz_str,)
    ).fetchone()
    if expected_mtime is None:
        return  # frame not in index (shouldn't happen during a mutation)
    if int(side.stat().st_mtime) == expected_mtime[0]:
        return  # mtime unchanged → no reason to re-read tags
    indexed_tags = frozenset(
        r[0] for r in conn.execute("SELECT tag FROM tags WHERE path=?", (npz_str,)).fetchall()
    )
    disk_tags = frozenset(read_tags(npz))
    if disk_tags != indexed_tags:
        raise StaleIndexError(
            f"sidecar {side} has been modified outside the store "
            f"(expected: {sorted(indexed_tags)}, found on disk: {sorted(disk_tags)})"
        )


class _MutateMixin:
    """Mixin providing mutation methods for TagStore."""

    @contextmanager
    def _write_transaction(self):
        """Acquire self._lock, BEGIN...COMMIT/ROLLBACK on exit."""
        with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _read_tags(self, npz: Path) -> frozenset[str]:
        """Read tags from sidecar via module-level lookup (monkeypatch hook)."""
        from .. import sidecar as sidecar_module

        return frozenset(sidecar_module.read_tags(npz))

    def _db_mutate(
        self,
        npz: Path,
        old_tags: frozenset[str],
        new_tags: frozenset[str],
    ) -> None:
        """Sync SQLite + cache after a sidecar write. Caller manages transaction."""
        conn = self._require_conn()
        npz_s = str(npz)

        if old_tags:
            placeholders = ",".join("?" * len(old_tags))
            conn.execute(
                f"DELETE FROM tags WHERE path=? AND tag IN ({placeholders})",
                [npz_s] + sorted(old_tags),
            )
        if new_tags:
            rows = []
            for tag in new_tags:
                try:
                    dim, val = parse_tag(tag)
                except ValueError:
                    dim, val = ("", tag)
                rows.append((npz_s, tag, dim, val))
            conn.executemany(
                "INSERT OR REPLACE INTO tags (path, tag, dim, val) VALUES (?, ?, ?, ?)",
                rows,
            )

        added = new_tags - old_tags
        removed = old_tags - new_tags
        if added or removed:
            self._sync_route_tags_cache(str(route_of(npz)), added, removed)

    def add_tags(
        self,
        tags: Sequence[str],
        *,
        frame_filter=None,
        scope: "Scope" = None,
        sync: bool | None = None,
    ) -> "MutationResult":
        """Union *tags* onto matching frames."""
        self._require_conn()
        to_add = normalize_tags(tags)
        if not to_add:
            return MutationResult(changed=0, skipped=0)
        to_add_set = frozenset(to_add)

        scope_set = self._resolve_scope(scope, granularity="frame")
        sync = _resolved_sync(sync)
        changed = skipped = 0
        failed: list[str] = []
        first_error: BaseException | None = None
        with self._write_transaction() as conn:
            for npz_str in scope_set:
                npz = Path(npz_str)
                if not match_frame_filter(npz, frame_filter):
                    skipped += 1
                    continue
                try:
                    disk_tags = self._read_tags(npz)
                    _check_stale(conn, npz, npz_str)
                    missing = to_add_set - disk_tags
                    if not missing:
                        skipped += 1
                        continue
                    merged = normalize_tags(list(disk_tags) + to_add)
                    write_tags(npz, merged, sync=sync, expected_tags=disk_tags)
                    self._db_mutate(npz, disk_tags, frozenset(merged))
                    changed += 1
                except FileNotFoundError:
                    # Frame was indexed but its sidecar is now missing — skip
                    # it. Reindex is the right tool to clear the index row.
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        return MutationResult(
            changed=changed, skipped=skipped, failed=failed, first_error=first_error
        )

    def remove_tags(
        self,
        tags: Sequence[str],
        *,
        frame_filter=None,
        scope: "Scope" = None,
        sync: bool | None = None,
    ) -> "MutationResult":
        """Delete exact tag strings from matching frames."""
        self._require_conn()
        to_remove = frozenset(normalize_tags(tags))
        if not to_remove:
            return MutationResult(changed=0, skipped=0)

        scope_set = self._resolve_scope(scope, granularity="frame")
        skipped = 0
        if frame_filter is None and scope_set:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(scope_set))
            route_rows = conn.execute(
                f"SELECT path, route FROM frames WHERE path IN ({placeholders})",
                sorted(scope_set),
            ).fetchall()
            routes_to_skip: dict[str, set[str]] = defaultdict(set)
            for path_str, route_str in route_rows:
                routes_to_skip[route_str].add(path_str)
            for route_str, frame_set in routes_to_skip.items():
                cached_tags = self._route_tags_cache.get(route_str, frozenset())
                if not (cached_tags & to_remove):
                    skipped += len(frame_set)
                    scope_set = scope_set - frame_set
            if not scope_set:
                return MutationResult(changed=0, skipped=skipped, failed=[], first_error=None)

        sync = _resolved_sync(sync)
        changed = 0
        failed: list[str] = []
        first_error: BaseException | None = None
        with self._write_transaction() as conn:
            for npz_str in scope_set:
                npz = Path(npz_str)
                if not match_frame_filter(npz, frame_filter):
                    skipped += 1
                    continue
                try:
                    disk_tags = self._read_tags(npz)
                    _check_stale(conn, npz, npz_str)
                    remaining_set = disk_tags - to_remove
                    if remaining_set == disk_tags:
                        skipped += 1
                        continue
                    remaining = sorted(remaining_set)
                    write_tags(npz, remaining, sync=sync, expected_tags=disk_tags)
                    self._db_mutate(npz, disk_tags, remaining_set)
                    changed += 1
                except FileNotFoundError:
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        return MutationResult(
            changed=changed, skipped=skipped, failed=failed, first_error=first_error
        )

    def remove_dimension(
        self,
        dimension: str,
        *,
        scope: "Scope" = None,
        frame_filter=None,
        sync: bool | None = None,
    ) -> "MutationResult":
        """Delete every tag whose dimension is *dimension*."""
        self._require_conn()
        if not is_valid_dimension(dimension):
            raise ValueError(f"bad dimension {dimension!r}: expected [a-z0-9_]+")

        scope_set = self._resolve_scope(scope, granularity="frame")
        sync = _resolved_sync(sync)
        changed = skipped = 0
        failed: list[str] = []
        first_error: BaseException | None = None
        with self._write_transaction() as conn:
            for npz_str in scope_set:
                npz = Path(npz_str)
                if not match_frame_filter(npz, frame_filter):
                    skipped += 1
                    continue
                try:
                    disk_tags = self._read_tags(npz)
                    _check_stale(conn, npz, npz_str)
                    remaining = frozenset(t for t in disk_tags if t.split(":", 1)[0] != dimension)
                    if remaining == disk_tags:
                        skipped += 1
                        continue
                    remaining_list = sorted(remaining)
                    write_tags(npz, remaining_list, sync=sync, expected_tags=disk_tags)
                    self._db_mutate(npz, disk_tags, remaining)
                    changed += 1
                except FileNotFoundError:
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        return MutationResult(
            changed=changed, skipped=skipped, failed=failed, first_error=first_error
        )

    def replace_tags(
        self,
        *,
        tag_pairs: dict[str, str],
        scope: "Scope" = None,
        frame_filter=None,
        sync: bool | None = None,
    ) -> "MutationResult":
        """Replace ``old_tag → new_tag`` on matching frames."""
        self._require_conn()
        validated: dict[str, str] = {}
        for old, new in tag_pairs.items():
            parse_tag(old)
            parse_tag(new)
            validated[old] = new
        if not validated:
            return MutationResult(changed=0, skipped=0)

        scope_set = self._resolve_scope(scope, granularity="frame")
        skipped = 0
        if frame_filter is None and scope_set:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(scope_set))
            route_rows = conn.execute(
                f"SELECT path, route FROM frames WHERE path IN ({placeholders})",
                sorted(scope_set),
            ).fetchall()
            routes_to_skip: dict[str, set[str]] = defaultdict(set)
            for path_str, route_str in route_rows:
                routes_to_skip[route_str].add(path_str)
            affected_old_tags = frozenset(validated.keys())
            for route_str, frame_set in routes_to_skip.items():
                cached_tags = self._route_tags_cache.get(route_str, frozenset())
                if not (cached_tags & affected_old_tags):
                    skipped += len(frame_set)
                    scope_set = scope_set - frame_set
            if not scope_set:
                return MutationResult(changed=0, skipped=skipped, failed=[], first_error=None)

        sync = _resolved_sync(sync)
        changed = 0
        failed: list[str] = []
        first_error: BaseException | None = None
        with self._write_transaction() as conn:
            for npz_str in scope_set:
                npz = Path(npz_str)
                if not match_frame_filter(npz, frame_filter):
                    skipped += 1
                    continue
                try:
                    disk_tags = self._read_tags(npz)
                    _check_stale(conn, npz, npz_str)
                    if not any(t in disk_tags for t in validated):
                        skipped += 1
                        continue
                    new_list: list[str] = []
                    seen: set[str] = set()
                    made_change = False
                    for t in sorted(disk_tags):
                        if t in validated and validated[t] != t:
                            repl = validated[t]
                            if repl not in seen:
                                new_list.append(repl)
                                seen.add(repl)
                            # Flag as changed if the tag value is actually different
                            made_change = True
                        elif t not in seen:
                            new_list.append(t)
                            seen.add(t)
                    # Also add new tags from validated that weren't already present
                    for t in validated.values():
                        if t not in seen:
                            new_list.append(t)
                            seen.add(t)
                            made_change = True
                    if made_change:
                        write_tags(npz, new_list, sync=sync, expected_tags=disk_tags)
                        self._db_mutate(npz, disk_tags, frozenset(new_list))
                        changed += 1
                    else:
                        skipped += 1
                except FileNotFoundError:
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        return MutationResult(
            changed=changed, skipped=skipped, failed=failed, first_error=first_error
        )

    def add_tags_to_route(
        self,
        tags: Sequence[str],
        route: str | Path,
        *,
        frame_filter=None,
        sync: bool | None = None,
    ) -> "MutationResult":
        """Merge *tags* into every frame of a known route.

        Raises ``ValueError`` if *route* is not in the index. ``route`` is
        stored as given (no symlink resolution) to match the on-disk form
        that the index was built from.
        """
        self._require_conn()
        route_str = str(Path(route))
        conn = self._require_conn()
        row = conn.execute("SELECT 1 FROM frames WHERE route=? LIMIT 1", (route_str,)).fetchone()
        if not row:
            raise ValueError(f"route not in index: {route_str}")
        return self.add_tags(tags, frame_filter=frame_filter, scope=route_str, sync=sync)
