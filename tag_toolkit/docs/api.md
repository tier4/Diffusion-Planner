# `tag_toolkit` API reference

Tags live as `tags` arrays on the JSON sidecar next to every NPZ. `tag_toolkit`
reads those sidecars into a SQLite index, exposes query and mutation on top of
it, and writes mutations back to the same sidecars atomically. See
[`design.md`](design.md) for the conceptual model and
[`database.md`](database.md) for the persistence layer.

For concrete code recipes see [`usage.md`](usage.md). For a one-line description
of every CLI script see [`scripts/README.md`](../scripts/README.md).

## Module-level entry points

```python
from tag_toolkit import (
    TagStore,
    Bucket, FrameTagDiff, IndexDiff, MutationResult,
    StaleIndexError,
    expand_source, route_of, extract_frame_number,
    parse_tag, format_tag, normalize_tags, read_tags, write_tags,
    load_taxonomy, list_known_tags, format_buckets,
)
```

| Symbol | Source | Summary |
|---|---|---|
| `TagStore` | `store/__init__.py` | The main class; see below. |
| `Bucket`, `FrameTagDiff`, `IndexDiff`, `MutationResult` | `store/_types.py` | Result dataclasses for `group_by`, `diff_index_against_disk`, and mutations. |
| `StaleIndexError` | `sidecar.py` | Raised when a sidecar has drifted from the index; re-call `reindex_tags` and retry. |
| `expand_source` | `source.py` | Resolve a source spec (path, directory, path-list JSON, list) into NPZ paths. |
| `route_of` | `routes.py` | Map an NPZ path (or directory) to its route directory (parent of the NPZ). |
| `extract_frame_number` | `routes.py` | Parse `<bag>_<prefix>_<8 digits>.npz` → `int`, or `None`. |
| `parse_tag`, `format_tag`, `normalize_tags` | `sidecar.py` | Validate / format / sort-dedupe `dim:val` strings. |
| `read_tags`, `write_tags` | `sidecar.py` | Direct read/write of sidecar JSON. Bypasses the index; use `TagStore` when you have a DB. |
| `load_taxonomy`, `list_known_tags` | `taxonomy.py` | Load `docs/tag_taxonomy.yaml`. Documentation only — the index ignores it. |
| `format_buckets` | `store/_query.py` | Pretty-print a list of `Bucket` objects. |

## Tag format

A tag is `"<dimension>:<value>"`. Both sides must match
`^[a-z0-9_]+$` — lowercase ASCII letters, digits, underscore.
Anything else raises `ValueError` from `parse_tag` / `normalize_tags`.

```python
parse_tag("split:auto")            # ("split", "auto")
parse_tag("Split:Auto")            # ValueError (uppercase)
normalize_tags(["b:2", "a:1"])      # ["a:1", "b:2"]
normalize_tags(["a:1", "a:1"])     # ["a:1"]   (dedupe)
```

## `TagStore` constructor

```python
TagStore(source=None) -> TagStore
```

Construct from one of:

- `None` (default) — empty in-memory store. Open it with `rebuild_index(source)` later.
- A `.tags.db` file — opens or creates that SQLite DB. All operations are persisted.
- Anything else (a directory, a path-list JSON, a list of paths, or a single `.npz`) — in-memory index seeded from that source. Mutations stay in memory; call `export_index(path)` to persist.

Concurrent writers are serialized by an internal `RLock`; readers are lock-free.

```python
store = TagStore("/data/dataset.tags.db")                  # persisted
store = TagStore("/data/routes/")                      # in-memory, seeded
store = TagStore("/data/close_loop_path_list.json")    # in-memory, seeded
store = TagStore()                                     # empty; call rebuild_index() later
```

## Index management

### `rebuild_index(source)`

Wipe the current index and rescan `source` from sidecars. `source` may be a
directory, a path-list JSON, a list of mixed items, or a single `.npz` file.
Shows a tqdm progress bar. **This is a full rebuild** — any rows currently in
the index are gone afterward.

```python
store.rebuild_index("/data/routes/")
```

### `append_frames(npz_paths) -> int`

Add new frames to an existing index. **Raises `ValueError` if any path
already exists in the index** — the caller must pre-filter duplicates
(`SELECT path FROM frames WHERE path IN (...)`). Updates
`_route_tags_cache` for every touched route. Returns the number of frames added.

Pre-filtering duplicates lets `append_frames` skip per-row SQLite conflict
checks and is faster than an "auto-ignore" implementation. The typical caller
is `scripts/tag_management/incremental_index.py`.

```python
existing = {p for (p,) in store._require_conn().execute(
    "SELECT path FROM frames WHERE path IN (SELECT path FROM frames)"
).fetchall()}
new_paths = [p for p in discovered if str(p) not in existing]
store.append_frames(new_paths)
```

### `reindex_tags() -> tuple[int, list[Path]]`

Re-read every sidecar already in the index, refresh the `tags` table, and
update `sidecar_mtime`. Frames without a sidecar become "orphans" and are
returned in the second list. Use this after external tools have edited
sidecars out-of-band.

```python
count, orphans = store.reindex_tags()
```

### `diff_index_against_disk(max_per_frame=100) -> IndexDiff`

Compare the index to every sidecar on disk. Returns an `IndexDiff` with
counters of added/removed tags per frame plus orphan-frame list. `IndexDiff.is_consistent`
is `True` when everything matches.

```python
diff = store.diff_index_against_disk()
if not diff.is_consistent:
    store.reindex_tags()    # reconcile
```

### `export_index(path)`

Write the current index to a SQLite file at `path` via `VACUUM INTO`. The
in-memory store is unchanged. Persisted files are also valid inputs to the
constructor.

### `build_index(source, output) -> TagStore` (static)

Convenience: build an index from `source`, persist it to `output`, and return
a `TagStore` backed by that file. Equivalent to
`store = TagStore(); store.rebuild_index(source); store.export_index(output)`.

```python
store = TagStore.build_index("/data/routes/", "/data/index.tags.db")
```

The CLI wrapper at `scripts/tag_management/build_index.py` exposes the same
flow without writing Python. It is the fast-path entry point for large
datasets whose downstream tools (`scripts/tag_usage/export_dataset.py`,
etc.) accept a pre-built `.tags.db`:

```bash
python scripts/tag_management/build_index.py /data/routes/ -o /data/index.tags.db
python scripts/tag_management/build_index.py /data/routes/ -o /data/index.tags.db --force
```

| Flag | Meaning |
|---|---|
| `source` (positional) | Same shape as `TagStore`: directory, path-list `.json` / `.json.zst`, single `.npz`, or a sequence of those. |
| `--output` / `-o` (required) | Destination `.tags.db` path. Parent dirs are auto-created. |
| `--force` | Overwrite `--output` if it already exists. Default is fail-fast. |

Exit codes: `0` on success; `1` if the source is missing, the output suffix is
wrong, or the output already exists without `--force`. On success the script
prints the route / frame counts and the resolved output path.

### `npz_paths() -> list[Path]`

All NPZ paths in the index (unsorted; route order from the `frames` table).

### `route_paths() -> list[Path]`

Distinct route directories in the index, sorted by path string.

### `has_index() -> bool`

`True` after construction (whether persisted or in-memory). Useful in tests.

### `source`

The constructor argument, as given. Read-only property.

## Query API

### `query(clause=None, *, granularity="route", scope=None) -> list[Path]`

Match routes (default) or frames against a tag clause.

- `clause=None` returns everything in scope.
- `clause="dim:val"` (or `"dim:*"` for a wildcard) is the simple equality form.
- `clause={"all": [c1, c2, ...]}` / `{"any": [...]}` / `{"not": c}` composes.

```python
store.query()                                    # all routes
store.query("split:auto")                        # routes tagged split:auto
store.query("split:*")                           # any split:* tag
store.query({"all": ["split:auto", "weather:clear"]})
store.query({"any": ["override:turn_left", "override:turn_right"]})
store.query({"not": "weather:clear"}, scope="/data/routes/")
store.query(granularity="frame")                 # frames, not routes
```

### `tags_of(*, scope=None, dimensions=None, granularity="route") -> list[str]`

Union of all tags visible to the call.

```python
store.tags_of()                                  # every distinct tag
store.tags_of(granularity="frame")               # tags on any frame
store.tags_of(dimensions=["split"])              # tags under split:*
store.tags_of(scope="/data/route_15/")
```

### `group_by(dimensions, clause=None, *, granularity="route", drop_missing=False, scope=None) -> list[Bucket]`

Group routes or frames by one or more tag dimensions. A frame/route is placed
into a bucket for each combination of tag values it carries across the
requested dimensions. Missing dimensions render as `None` unless `drop_missing=True`.

```python
buckets = store.group_by(["split", "weather"])
buckets = store.group_by(["split"], drop_missing=True)
print(format_buckets(buckets, ["split", "weather"]))
```

## Mutation API

All mutations return a `MutationResult(changed, skipped, failed, first_error)`.
`bool(result)` is `True` iff any frame was changed. Each mutation:

1. acquires `self._lock`,
2. for each frame in scope reads the sidecar from disk,
3. verifies `sidecar_mtime` matches the index (`StaleIndexError` if not),
4. atomically writes a new sidecar (`.tmp` + `rename` + `fsync`),
5. updates the SQLite `tags` rows for the changed frame,
6. updates `_route_tags_cache` for the touched route.

If `sync=True` (default) the file is `fsync`'d. Pass `sync=False` for batch
operations; the caller is then responsible for durability.

### `add_tags(tags, *, frame_filter=None, scope=None, sync=True) -> MutationResult`

Union `tags` onto matching frames. `tags` is normalized and validated.

```python
store.add_tags(["override_metric:centerline"], scope="/data/route_15/")
store.add_tags(["weather:clear"], frame_filter=(0, 100))   # only frames 0..100
```

### `remove_tags(tags, *, frame_filter=None, scope=None, sync=True) -> MutationResult`

Delete exact tag strings from matching frames.

```python
store.remove_tags(["weather:rain"], scope=my_other_store)
```

### `remove_dimension(dimension, *, scope=None, frame_filter=None, sync=True) -> MutationResult`

Delete every tag whose dimension equals `dimension` (the value is irrelevant).

```python
store.remove_dimension("override_metric")
```

### `replace_tags(*, tag_pairs, scope=None, frame_filter=None, sync=True) -> MutationResult`

Apply `{old_tag: new_tag}` substitutions on matching frames. Frames where the
substitution would be a no-op (or where neither tag is present) are skipped.

> **Warning:** `tag_pairs` must not contain chains or cycles. Chaining (e.g.
> `{"A": "B", "B": "C"}`) and cycling (e.g. `{"A": "B", "B": "A"}`) both produce
> undefined results and may silently resurrect tags that were already replaced.
> Supply only flat, non-overlapping mappings.

```python
store.replace_tags(tag_pairs={"split:auto": "split:manual"}, scope="/data/old/")
```

### `add_tags_to_route(tags, route, *, frame_filter=None, sync=True) -> MutationResult`

Convenience: union `tags` onto every frame of a route already in the index.
Raises `ValueError` if `route` is not in the index.

```python
store.add_tags_to_route(["weather:clear"], "/data/routes/route_15/")
```

### Failure modes

- `StaleIndexError` — the on-disk sidecar doesn't match what the index
  recorded (someone edited it out-of-band). Sidecar is left untouched. Call
  `reindex_tags()` and retry.
- `FileNotFoundError` — a frame in scope has no sidecar on disk. Mutations
  count it as `skipped`, not `failed`. The store never silently creates
  a sidecar that didn't exist; tools that want fresh sidecars must
  create them outside the tag pipeline (or fix the index with
  `reindex_tags()`).
- `ValueError` (from `parse_tag` / `normalize_tags`) — bad tag strings in
  the input. Fix the input.
- Other exceptions (e.g. permission errors) are recorded as `failed[i]` /
  `first_error`, the loop continues, the transaction rolls back, and the
  remaining frames are left as they were.

## Source expansion

```python
from tag_toolkit import expand_source
```

```python
expand_source("/data/routes/")                  # recursive *.npz
expand_source("/data/routes/frame.npz")         # [Path] (exists check)
expand_source("/data/close_loop_path_list.json")    # loads JSON array
expand_source(["/data/a.npz", "/data/b.json"])  # mixed
```

`source.py` follows the training-loader's path-list rules: `.json` /
`.json.zst` / `.zst` lists with all-`.npz` entries are loaded as strings
without `stat()` per file (cheap for huge lists). Mixed lists that contain
route directories expand those only.

## Sidecar I/O (no DB)

When you have no `TagStore` handy:

```python
from tag_toolkit import parse_tag, normalize_tags, read_tags, write_tags, sidecar_path

side = sidecar_path("/data/.../frame.npz")      # /data/.../frame.json
read_tags("/data/.../frame.npz")                # sorted, deduped list
write_tags(npz, ["split:auto", "weather:clear"])
write_tags(npz, ["split:auto"], expected_tags=["weather:clear"])  # StaleIndexError
```

`write_tags` validates and normalizes the input, then performs the atomic
`.json.tmp` + `rename` + `fsync` protocol. With `expected_tags`, the on-disk
tags must match exactly — useful for "I just read the index, write back the
diff" workflows. **`write_tags` will never create a sidecar that didn't
already exist**; missing sidecar → `FileNotFoundError`.

`read_tags` is tolerant at the per-tag level (malformed entries are dropped
with a warning) but strict at the structural level (bad JSON / non-object /
non-list `tags` → `ValueError`).

## Result dataclasses

```python
@dataclass
class MutationResult:
    changed: int
    skipped: int
    failed: list[str] = field(default_factory=list)
    first_error: BaseException | None = field(default=None, repr=False)
    # bool(result) -> result.changed > 0

@dataclass
class Bucket:
    values: dict[str, str | None]   # dim -> selected value (or None)
    members: list[Path]             # routes or frames in this bucket
    count: int                      # len(members)
    label(sep=" | ") -> str         # formatted label

@dataclass
class FrameTagDiff:
    npz: Path
    index_tags: frozenset[str]
    disk_tags: frozenset[str]

@dataclass
class IndexDiff:
    frames_checked: int
    frames_with_tag_diff: int
    orphan_frames: list[Path]
    tags_added: Counter[str]
    tags_removed: Counter[str]
    per_frame: list[FrameTagDiff]
    is_consistent: bool             # frames_with_tag_diff == 0 and not orphans
```

## Errors

| Exception | When | Caller's move |
|---|---|---|
| `ValueError` (from `parse_tag`) | Bad tag string passed in. | Fix the input. |
| `ValueError` (from `append_frames`) | Duplicate path in input. | Pre-filter via `SELECT path FROM frames WHERE path IN (...)`. |
| `ValueError` (from `add_tags_to_route`) | Route not in the index. | Use a route from `store.route_paths()`. |
| `ValueError` (from `tags_of`/`group_by`/`query`) | Bad granularity, bad dimension name, malformed clause. | Fix the argument. |
| `FileNotFoundError` (from `write_tags` / a mutation) | Frame has no sidecar on disk. | Create the sidecar externally, then re-run; mutations count this as `skipped`. |
| `StaleIndexError` | Sidecar mtime or content drifted from index since last build. | `store.reindex_tags()`, then retry. |
