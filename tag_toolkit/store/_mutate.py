"""Mutation operations for TagStore."""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..routes import route_of
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
    if int(side.stat().st_mtime * 1000) == expected_mtime[0]:
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
        """Sync SQLite index after a sidecar write. Caller manages transaction."""
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

        # Update sidecar_mtime so _check_stale's fast path (mtime comparison)
        # remains valid after this mutation.  Without this, the mtime in the
        # DB would be stale and _check_stale would always fall through to the
        # expensive SELECT + read_tags path on the next call.
        side = sidecar_path(npz)
        new_mtime = int(side.stat().st_mtime * 1000) if side.is_file() else 0
        conn.execute(
            "UPDATE frames SET sidecar_mtime=? WHERE path=?",
            (new_mtime, npz_s),
        )

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
        affected_routes: set[str] = set()

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
                    affected_routes.add(str(route_of(npz)))
                except FileNotFoundError:
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        if affected_routes:
            self._recompute_route_tags_cache_for_routes(list(affected_routes))

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
        sync = _resolved_sync(sync)

        changed = skipped = 0
        failed: list[str] = []
        first_error: BaseException | None = None
        affected_routes: set[str] = set()

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
                    affected_routes.add(str(route_of(npz)))
                except FileNotFoundError:
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        if affected_routes:
            self._recompute_route_tags_cache_for_routes(list(affected_routes))

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
        affected_routes: set[str] = set()

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
                    affected_routes.add(str(route_of(npz)))
                except FileNotFoundError:
                    skipped += 1
                except Exception as exc:
                    if isinstance(exc, StaleIndexError):
                        raise
                    failed.append(npz_str)
                    if first_error is None:
                        first_error = exc

        if affected_routes:
            self._recompute_route_tags_cache_for_routes(list(affected_routes))

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
        """Replace ``old_tag → new_tag`` on matching frames.

        Only tags that the frame actually carries are considered. For each frame:
        - If it carries a source tag (key in tag_pairs), the source tag is replaced
          with the corresponding replacement tag (value in tag_pairs).
        - If the replacement tag is the same as the source (``a:1 → a:1``), the tag
          is kept as-is.
        - Tags not in tag_pairs are preserved.
        - Replacement tags whose source tag the frame never carried are NOT added.
        """
        self._require_conn()
        validated: dict[str, str] = {}
        for old, new in tag_pairs.items():
            parse_tag(old)
            parse_tag(new)
            validated[old] = new
        if not validated:
            return MutationResult(changed=0, skipped=0)

        scope_set = self._resolve_scope(scope, granularity="frame")
        sync = _resolved_sync(sync)

        changed = skipped = 0
        failed: list[str] = []
        first_error: BaseException | None = None
        affected_routes: set[str] = set()

        with self._write_transaction() as conn:
            for npz_str in scope_set:
                npz = Path(npz_str)
                if not match_frame_filter(npz, frame_filter):
                    skipped += 1
                    continue
                try:
                    disk_tags = self._read_tags(npz)
                    _check_stale(conn, npz, npz_str)

                    # Find which source tags this frame actually carries.
                    source_tags_present = [t for t in disk_tags if t in validated]
                    if not source_tags_present:
                        skipped += 1
                        continue

                    # Build the new tag list: replace source→replacement, keep others.
                    new_list: list[str] = []
                    seen: set[str] = set()
                    for t in sorted(disk_tags):
                        if t in validated:
                            repl = validated[t]
                            if repl != t:
                                # Actual replacement: add the new tag.
                                if repl not in seen:
                                    new_list.append(repl)
                                    seen.add(repl)
                            elif t not in seen:
                                # Same-value pair (a:1→a:1): tag is unchanged on disk,
                                # so include it so the final comparison is correct.
                                new_list.append(t)
                                seen.add(t)
                        elif t not in seen:
                            new_list.append(t)
                            seen.add(t)

                    # Only write if the result actually differs from what's on disk.
                    if frozenset(new_list) != disk_tags:
                        write_tags(npz, new_list, sync=sync, expected_tags=disk_tags)
                        self._db_mutate(npz, disk_tags, frozenset(new_list))
                        changed += 1
                        affected_routes.add(str(route_of(npz)))
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

        if affected_routes:
            self._recompute_route_tags_cache_for_routes(list(affected_routes))

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
