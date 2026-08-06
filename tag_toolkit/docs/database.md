# `tag_toolkit` persistence layer

`tag_toolkit` builds a small SQLite database that mirrors the `tags` field on
every NPZ sidecar. The DB is the authoritative view of tag state; mutations
verify that the sidecar on disk still matches the index before writing, so a
half-applied external edit can't quietly slip through. The conceptual model
("why a separate index?") is in [`design.md`](design.md); this document is
about the actual schema and SQLite config.

The schema and PRAGMA setup live in [`store/_db.py`](../store/_db.py) and run
unconditionally on every fresh connection (`_init_db`).

## Schema

```sql
CREATE TABLE IF NOT EXISTS frames (
    path          TEXT PRIMARY KEY,
    route         TEXT NOT NULL,
    sidecar_mtime INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_route ON frames(route);
CREATE INDEX IF NOT EXISTS frames_route_path ON frames(route, path);

CREATE TABLE IF NOT EXISTS tags (
    path TEXT NOT NULL,
    tag  TEXT NOT NULL,
    dim  TEXT NOT NULL,
    val  TEXT NOT NULL,
    PRIMARY KEY (path, tag)
);
CREATE INDEX IF NOT EXISTS tags_tag_path ON tags(tag, path);
CREATE INDEX IF NOT EXISTS tags_path ON tags(path);
CREATE INDEX IF NOT EXISTS tags_dim_val_path ON tags(dim, val, path);
```

### `frames`

One row per indexed NPZ. The primary key is the absolute path string; `route`
is the bag directory (`route_of(npz)`); `sidecar_mtime` is the `int`
`stat().st_mtime` of the sidecar at the time the row was last touched
(initial build, `append_frames`, `reindex_tags`).

The stale-detection protocol compares the on-disk mtime against
`frames.sidecar_mtime` for the path. If the file's mtime has moved, the
mutation reads the tags back from the DB and compares against the just-read
disk state — see `_check_stale` in `store/_mutate.py`. So `sidecar_mtime`
is both a fast skip-check and an early-warning signal, not the source of truth.

Indexes:

- `frames_route(route)` — backs `tags_of(scope=route, …)` and the
  `frames WHERE route IN (...)` lookup in `_ScopeResolver`.
- `frames_route_path(route, path)` — same queries, but the composite index
  serves `(route, path)` lookups (e.g. `WHERE route=? AND path IN (...)`).

### `tags`

Long-tail: one row per `(path, tag)` pair, denormalised into
`(dim, val)` so wildcard scans (`dim:val`, `dim:*`, `dim = ? AND val = ?`)
avoid parsing the tag string every time. The `(path, tag)` primary key
backs `INSERT OR IGNORE` and `INSERT OR REPLACE` for batched tag upserts
(see `add_tags`/`append_frames`).

Indexes:

- `tags_tag_path(tag, path)` — backs `WHERE tag = ?` / `WHERE tag LIKE 'prefix%'`
  (clause equality and `dim:*` wildcard).
- `tags_path(path)` — backs `WHERE path = ?` (per-frame reads inside
  `diff_index_against_disk`, `_check_stale`, `_db_mutate`).
- `tags_dim_val_path(dim, val, path)` — backs composite scans used by
  `replace_tags` and similar filtered lookups.

## SQLite configuration

Every connection runs `_init_db` once on first use:

```sql
PRAGMA busy_timeout = 5000;
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
```

| Setting | Value | Why |
|---|---|---|
| `busy_timeout` | `5000` (5s) | Lets concurrent writers wait briefly instead of failing with `SQLITE_BUSY`. The single `RLock` on `TagStore` already serialises writers, so this only matters under multi-process or multi-thread contention. |
| `journal_mode` | `WAL` | Allows readers to proceed without blocking the writer. Critical for the `read_tags` calls inside mutation loops: we want concurrent queries to still see a consistent snapshot mid-transaction. WAL also avoids per-write `fsync` of the data directory. |
| `synchronous` | `NORMAL` | Skips the rollback-journal `fsync` that `FULL` would require; pairs with `journal_mode=WAL`. Crash safety is still good enough for tag-state: a power loss at worst loses the last in-flight transaction's sidecar write (which `sidecar.write_tags` already fsync's on its own). |

`check_same_thread=False` is passed to `sqlite3.connect` on the persistent
DB path. Each thread gets its own connection (see
`_conn_per_thread` in `store/__init__.py`), but a single `RLock` on the
store serialises writers so transactions don't interleave across threads.
Readers are lock-free.

In-memory stores use `sqlite3.connect("file:<uuid>?mode=memory&cache=private", uri=True)`
— separate per-thread connections point at the same backing page cache.
This lets the `RLock` semantics work identically across both backends.

## Stale-index protocol

A mutation (`add_tags`, `remove_tags`, `remove_dimension`, `replace_tags`)
follows this loop per frame:

1. `sidecar_path(npz).is_file()` — missing sidecar triggers
   `FileNotFoundError` (raised by `write_tags`); the mutation counts it as
   `skipped`.
2. `read_tags(npz)` from disk (raises `ValueError` on structural corruption;
   `skipped` for per-tag malformation with a warning).
3. `_check_stale(conn, npz, …)` — compare `disk_tags` against the index's
   stored tags for this path. Skipped as a fast path when the sidecar
   mtime is unchanged; otherwise re-reads the index and raises
   `StaleIndexError` if the sets differ.
4. Compute the new tag set.
5. `write_tags(npz, new, expected_tags=disk_tags)` — atomic
   `.json.tmp + rename + fsync`. The `expected_tags` argument runs the
   read-vs-write half of the drift check (catches concurrent drift between
   our own read and our own write).
6. `_db_mutate(npz, old, new)` — `DELETE` the old tags, `INSERT OR REPLACE`
   the new ones, then `_sync_route_tags_cache` to update the in-memory cache.

All frames in the call share one outer transaction (`_write_transaction`)
which commits on success and rolls back on `StaleIndexError`. Other
exceptions are caught per-frame, recorded in `MutationResult.failed`, and
the loop continues with the next frame.

## Index lifecycle

| Method | What it touches | Use case |
|---|---|---|
| `rebuild_index(source)` | `DELETE FROM frames`, `DELETE FROM tags`, then full re-scan of *source*. `_warm_route_tags_cache()` at the end. | First-time build, or after a wholesale source move. |
| `append_frames(paths)` | `INSERT OR IGNORE` into `frames` and `tags` after a PK pre-check (`SELECT path FROM frames WHERE path IN (...)`) that raises `ValueError` on duplicates. Single `SELECT f.route, t.tag ... WHERE t.path IN (...)` for cache update. | Adding new NPZs to an existing index. Caller pre-filters duplicates for performance. |
| `reindex_tags()` | `DELETE FROM tags`, then re-read each sidecar already in `frames`, `INSERT` tags and `UPDATE frames.sidecar_mtime` in batches of 100 000. Cache rebuilt from scratch. | External tools edited sidecars; refresh the index to match. |
| `diff_index_against_disk()` | Read-only. `SELECT path, sidecar_mtime FROM frames`, compare each sidecar. | Inspection; returns an `IndexDiff`. |
| `export_index(path)` | `VACUUM INTO '<path>'` on the in-memory DB. | Persist an in-memory store to a file. |
| `build_index(source, output)` (static) | `rebuild_index` + `export_index`. | One-shot build-and-save. |

## Tag taxonomy

`docs/tag_taxonomy.yaml` documents the recognised dimensions and values. It
is loaded only by `taxonomy.load_taxonomy` / `list_known_tags`. The query
and mutate APIs **never** read it — the index is the source of truth, and
mutations don't validate tags against the taxonomy (any `dim:val` matching
`^[a-z0-9_]+:[a-z0-9_]+$` is accepted). The YAML is a human-facing reference,
not a contract.

## Performance notes

- `query("dim:val", granularity="route")` is a single
  `SELECT DISTINCT f.route FROM frames f JOIN tags t ON f.path=t.path
   WHERE f.route IN (...) AND t.tag=?` query — PK lookup on `tags_tag_path`
  plus the `frames_route` index. O(matching_frames).
- `tags_of(granularity="route")` (no scope filter) is in-memory and served
  entirely from `_route_tags_cache` — no SQLite roundtrip.
- `group_by(granularity="frame", dimensions=...)` does one
  `SELECT path, tag, dim, val FROM tags WHERE path IN (...)` then groups
  in Python. For route granularity it uses the cache.
- `append_frames` is fastest when the caller pre-filters via
  `SELECT path FROM frames WHERE path IN (...)`: that's a single O(1)-per-row
  PK lookup on `frames`, faster than letting SQLite run INSERT-conflict
  detection on every row.

## Compatibility

The schema is stable enough to live across `TagStore` versions. New columns
or tables should be added with `CREATE TABLE IF NOT EXISTS` /
`ALTER TABLE` migrations gated by `PRAGMA user_version`, not by hand.
Today there is no migration framework; if you change the schema, ship a
script that recreates the index via `rebuild_index`.
