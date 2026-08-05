# TagStore: design notes

What `TagStore` does under the hood, and the contracts that aren't obvious
from the API. For method signatures and examples see
[`api.md`](api.md) and [`usage.md`](usage.md). For schema, indexes and SQLite
PRAGMAs see [`database.md`](database.md).

---

## Source (the authoritative frame set)

A TagStore is built from a `source` at construction time and that set is
**the union of frames known to the index at any moment**. The constructor
scans, the index is built, and subsequent calls only ever operate on frames
that are present in that scan — but new frames can be added incrementally via
`append_frames(paths)` (raises `ValueError` on duplicates; caller
pre-filters). For wholesale changes, use `rebuild_index`.

`clause`, `scope`, and `frame_filter` are **only filters** — they narrow
the set, they never extend it. A path that isn't in the index never
becomes "visible" through `scope`.

For large or repeated use, prefer a pre-built `.db` index file. The
on-init directory scan reads every sidecar on disk; the loaded index
skips that.

---

## Index structure (SQLite)

The SQLite schema, indexes, and PRAGMA configuration live in
[`database.md`](database.md). The summary: two tables (`frames`,
`tags`) with denormalised `(dim, val)` columns on the `tags` rows, and
a small in-memory `_route_tags_cache` holding the union of tags per
route for fast route-granularity queries.

The cache is rebuilt from the DB on startup and on `reindex_tags()`,
and synchronised incrementally on every mutation. The cache is purely a
performance optimization — every query can be recomputed from the SQL
tables.

---

## Atomic writes

A mutation writes to one sidecar at a time via a `.json.tmp` sibling
that gets `os.replace`d over the target on success. If anything fails, the
original `.json` is left untouched and the temp file is removed. A partial
mutation can leave the dataset in a half-tagged state across frames — but
never corrupts a single sidecar.

### Verify-then-write

Every mutation runs a stale-check just before the atomic write. The check
piggy-backs on the `read_sidecar` call that `write_tags` already makes to
preserve non-tag fields: after reading the sidecar, it compares the
on-disk `tags` field against an `expected_tags` value the store passes in
(the index's view of what the frame had).

- **On-disk matches what we just read** — the write proceeds.
- **On-disk has drifted since we read** — `StaleIndexError` is raised,
  the sidecar is left untouched, and the mutation is aborted. No further
  frames are touched in this call.
- **No `expected_tags` passed** — the check is skipped. This is for
  out-of-band tools without an index to compare against. The store's
  mutation methods always pass it.

A second guard, `_check_stale(conn, npz, npz_str)`, runs at the top of
each mutation to catch the trickier drift case: the sidecar on disk
differs from what the SQLite index holds (e.g. an out-of-band edit
whose mtime was preserved). It uses an `mtime` fast path — if the
sidecar's mtime matches the `sidecar_mtime` row in the `frames` table,
the index and disk are guaranteed to agree and the tag-set comparison
is skipped. When the mtime *has* moved, the function re-reads the tag
set from disk and compares it against the indexed tag set, raising
`StaleIndexError` on mismatch. Both checks raise the same exception so
the recovery path in the caller is uniform.

`StaleIndexError` is the recovery signal. After fixing the drift
(manually, or via `reindex_tags()` to refresh the index from disk), retry
the mutation:

```python
try:
    store.add_tags(["site:new"], scope=route)
except StaleIndexError:
    store.reindex_tags()
    store.add_tags(["site:new"], scope=route)
```

`write_tags` will not create a sidecar that didn't already exist on disk.
A frame without a sidecar raises `FileNotFoundError` from `write_tags`;
the mutation methods catch this and report the frame in `result.skipped`
(rather than `result.failed`), so missing sidecars are a recoverable
condition rather than a hard error.

### fsync

Each write by default calls `fsync` on the file and on the parent
directory. The directory fsync ensures the filename → inode mapping is
persisted; without it, a power loss can leave the file content on disk
but the filename missing from the directory.

fsync is expensive (most SSDs handle only 500–2000 fsyncs/s). For batch
workloads, pass `sync=False` to the mutating methods and rely on the
`WAL` mode for crash safety; each call still flushes its own file
without the parent-directory fsync.

---

## Scope resolution

`_resolve_scope(scope, granularity)` returns a `set[str]` of either
route directory strings (`granularity="route"`) or NPZ path strings
(`granularity="frame"`). Three cases, in order:

1. **`scope is None`** — return every path in the index at the requested
   granularity. No filesystem work.
2. **`scope` is a `TagStore`** — return every path from that store's
   index, then **intersect** with this store's index. Routes from the
   scope store that don't exist here drop out silently. The intersection
   is what makes nested queries behave identically to passing a list of
   paths: a route only contributes if it also has frames in this store.
3. **`scope` is a Path or list of Paths** — for each item, try in order:
   - **route path already in the index** — single-row lookup in
     `frames`. Return `{route}` (route granularity) or the route's NPZ
     list (frame granularity).
   - **NPZ path already in the index** — single-row lookup in `frames`.
     Return `{route}` (route granularity) or `{npz}` (frame).
   - **fallback** — for anything else (directory, path-list JSON,
     unknown path), use `expand_source` to enumerate NPZ files on disk,
     then keep only the ones that are also in the index. For
     `granularity="route"`, the matched paths are projected to their
     distinct route columns.

Anything that ends up with no match in the index is silently dropped. The
fallback is the only place that touches disk during scope resolution, so
passing already-known routes or NPZs (cases 3a/3b) avoids it entirely.

`add_tags_to_route` is a deliberate exception: it rejects with
`ValueError` when the route isn't in the index, because silently tagging
frames that aren't indexed would make those tags invisible to every
subsequent query.

---

## Source vs Scope

`source` and `scope` look similar but are deliberately distinct:

| | `source` (constructor) | `scope` (per-call) |
|---|---|---|
| When set | Once, at `TagStore(source)` | Every query / mutate call |
| Effect | Defines the **authoritative frame set** | Narrows the **operation** to a subset |
| Can extend the frame set? | Yes — scanning discovers frames on disk | **No** — unknown paths are silently dropped |
| Disk cost | One-time scan or `.db` load | None for indexed paths; one `expand_source` fallback per non-indexed path |
| Mutability | Frozen for the lifetime of the store | Re-evaluated per call |

`source` answers "what does this store see?". `scope` answers "of what
this store sees, which subset does this call care about?". Confusing
them is the most common source of bugs:

- Passing a `Path` that doesn't exist to `TagStore(source)` raises
  `FileNotFoundError` — the store refuses to fabricate an authoritative
  set.
- Passing the same `Path` to `query(..., scope=path)` silently drops it
  — the store already has its authoritative set; the path just didn't
  match any of it.

Treat them as different categories. `source` is a build-time contract;
`scope` is a runtime filter.
