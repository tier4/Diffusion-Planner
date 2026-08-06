"""Index operations for TagStore."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from tqdm import tqdm

from ..routes import route_of
from ..sidecar import parse_tag, read_tags, sidecar_path
from ..source import expand_source

if TYPE_CHECKING:
    from ._types import FrameTagDiff, IndexDiff


_BATCH_SIZE = 100_000


class _IndexMixin:
    """Mixin providing index management methods for TagStore."""

    def rebuild_index(
        self,
        source: str | Path | Sequence[str | Path],
    ) -> None:
        """Scan *source*, wipe the current index, and rebuild it from sidecars."""
        frames: list[Path]
        if isinstance(source, (list, tuple)):
            frames = []
            for s in source:
                frames.extend(expand_source(s, sort=False))
        else:
            frames = expand_source(source, sort=False)

        with self._write_transaction() as conn:
            conn.execute("DELETE FROM tags")
            conn.execute("DELETE FROM frames")
            frames_batch: list[tuple[str, str, int]] = []
            tags_batch: list[tuple[str, str, str, str]] = []
            for npz in tqdm(frames, desc="Scanning sidecars"):
                tags_list = read_tags(npz)
                route = str(route_of(npz))
                side = sidecar_path(npz)
                mtime = int(side.stat().st_mtime) if side.is_file() else 0
                npz_s = str(npz)
                frames_batch.append((npz_s, route, mtime))
                for tag in tags_list:
                    # read_tags already filters malformed entries with a
                    # warning, so parse_tag here will not raise.
                    dim, val = parse_tag(tag)
                    tags_batch.append((npz_s, tag, dim, val))
            if frames_batch:
                conn.executemany(
                    "INSERT INTO frames (path, route, sidecar_mtime) VALUES (?, ?, ?)",
                    frames_batch,
                )
            if tags_batch:
                conn.executemany(
                    "INSERT INTO tags (path, tag, dim, val) VALUES (?, ?, ?, ?)",
                    tags_batch,
                )

        self._warm_route_tags_cache()

    def export_index(self, path: str | Path) -> None:
        """Export the in-memory index to a SQLite file at *path* using VACUUM INTO."""
        conn = self._require_conn()
        out = Path(path)
        # VACUUM INTO refuses to overwrite; clear the way for callers that
        # legitimately want to replace an existing .db (e.g. CLI scripts that
        # rebuild the index in place).
        if out.exists():
            out.unlink()
        conn.execute("VACUUM INTO ?", (str(out),))

    @staticmethod
    def build_index(
        source: str | Path | Sequence[str | Path],
        output: str | Path,
    ) -> "TagStore":
        """Scan *source*, build a SQLite index, and write it to *output*.

        Returns a TagStore backed by the on-disk file at *output*.
        """
        from . import TagStore

        if isinstance(source, TagStore):
            # Persist the existing in-memory index to the output file.
            source.export_index(output)
            return TagStore(output)

        store = TagStore()
        store.rebuild_index(source)
        store.export_index(output)
        return store

    def diff_index_against_disk(
        self,
        *,
        max_per_frame: int = 100,
    ) -> "IndexDiff":
        """Compare the index to every sidecar on disk."""
        from ._types import FrameTagDiff, IndexDiff

        conn = self._require_conn()
        orphan_frames: list[Path] = []
        tags_added: Counter[str] = Counter()
        tags_removed: Counter[str] = Counter()
        per_frame: list[FrameTagDiff] = []
        frames_with_tag_diff = 0

        rows = conn.execute("SELECT path, sidecar_mtime FROM frames").fetchall()
        for row in rows:
            npz = Path(row[0])
            side = sidecar_path(npz)
            if not side.is_file():
                orphan_frames.append(npz)
                continue
            disk_tags = frozenset(read_tags(npz))
            index_rows = conn.execute("SELECT tag FROM tags WHERE path=?", (row[0],)).fetchall()
            index_tags = frozenset(r[0] for r in index_rows)
            if disk_tags == index_tags:
                continue
            frames_with_tag_diff += 1
            tags_added.update(disk_tags - index_tags)
            tags_removed.update(index_tags - disk_tags)
            if len(per_frame) < max_per_frame:
                per_frame.append(FrameTagDiff(npz=npz, index_tags=index_tags, disk_tags=disk_tags))

        return IndexDiff(
            frames_checked=len(rows),
            frames_with_tag_diff=frames_with_tag_diff,
            orphan_frames=orphan_frames,
            tags_added=tags_added,
            tags_removed=tags_removed,
            per_frame=per_frame,
        )

    def reindex_tags(self) -> tuple[int, list[Path]]:
        """Re-read every sidecar and update the tags table.

        Frames whose sidecar is missing or unreadable (other than a structural
        error like a non-list ``tags`` field, which we let bubble up) become
        orphans and are reported back. The in-memory ``_route_tags_cache`` is
        rebuilt from scratch and then merged with any tag sets collected here.
        """
        conn = self._require_conn()
        orphan_frames: list[Path] = []
        reindexed_count = 0

        rows = conn.execute("SELECT path, route FROM frames").fetchall()

        # Hold the store lock through the rebuild so concurrent mutations
        # don't smuggle in stale rows between our DELETE and our INSERTs.
        with self._lock:
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM tags")
                self._route_tags_cache.clear()
                route_tag_sets: dict[str, set[str]] = defaultdict(set)
                batch_tags: list[tuple[str, str, str, str]] = []

                for row in rows:
                    npz = Path(row[0])
                    route = row[1]
                    side = sidecar_path(npz)
                    if not side.is_file():
                        orphan_frames.append(npz)
                        continue

                    try:
                        tags_list = read_tags(npz)
                    except (OSError, ValueError):
                        # I/O error or structural sidecar corruption (bad
                        # JSON, non-list ``tags``). Either way, treat the
                        # frame as unindexable for this pass; other
                        # exceptions (e.g. KeyboardInterrupt) keep
                        # propagating.
                        orphan_frames.append(npz)
                        continue

                    mtime = int(side.stat().st_mtime)
                    conn.execute(
                        "UPDATE frames SET sidecar_mtime=? WHERE path=?",
                        (mtime, row[0]),
                    )
                    for tag in tags_list:
                        dim, val = parse_tag(tag)
                        batch_tags.append((row[0], tag, dim, val))
                        route_tag_sets[route].add(tag)
                    reindexed_count += 1
                    if len(batch_tags) >= _BATCH_SIZE:
                        conn.executemany(
                            "INSERT INTO tags (path, tag, dim, val) VALUES (?, ?, ?, ?)",
                            batch_tags,
                        )
                        batch_tags.clear()

                if batch_tags:
                    conn.executemany(
                        "INSERT INTO tags (path, tag, dim, val) VALUES (?, ?, ?, ?)",
                        batch_tags,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            self._route_tags_cache.update({r: frozenset(t) for r, t in route_tag_sets.items()})

        return reindexed_count, orphan_frames
