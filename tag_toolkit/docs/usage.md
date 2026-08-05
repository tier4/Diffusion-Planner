# tag_toolkit usage

Tags for Diffusion Planner data live on each NPZ's sidecar JSON (`<stem>.json`
next to `<stem>.npz`). `tag_toolkit` reads and writes the `tags` field, and
queries at **route** or **frame** granularity. For large datasets, build the
index once with `TagStore.build_index` and load the resulting `.db` file
thereafter.

Scope resolution rules live in [design.md](design.md).

---

## Tag format

```json
{
  "timestamp": 1738632874843986836,
  "tags": [
    "site:xxxx_site_example",
    "split:manual",
    "lateral:turn"
  ]
}
```

- Type: JSON array of strings. Missing `tags` ≡ `[]`.
- Entry: `dimension:value` (exactly one colon).
- Names: `[a-z0-9_]+` on both sides.
- A frame may carry several tags. Writers sort for stable diffs; readers
  must not depend on order.

Writes are atomic at the file level (a sibling `.json.tmp` is created and
renamed over the target on success; the temp is cleaned up on either branch).

---

## Route

The converted layout (`dataset/generate_from_labeled.sh`):

```text
…/<project>/<map_id>/{manual|auto}/<date>/<bag_time>/routes/*.npz
```

A **route** is the bag directory `…/<bag_time>/` (parent of `routes/` when
present). `route_of(path)` maps an NPZ or a route dir to this root.

---

## Route vs frame

Tags are stored per frame. Most workflows care about whole routes, so
`query` and `group_by` default to `granularity="route"`. At route
granularity a route has a tag if any frame under it has that tag —
`{"all": ["a", "b"]}` can succeed even when no single frame has both.
At `granularity="frame"`, tags must co-occur on the same NPZ.

| Granularity       | Matching            | Result paths      |
| ----------------- | ------------------- | ----------------- |
| `route` (default) | Union of frame tags | route directories |
| `frame`           | Tags on one NPZ     | `.npz` files      |

---

## `site` and `split`

Once written, `site` and `split` are ordinary tags — query and mutate only
look at sidecar `tags`. To fill them for a whole dataset from paths (with
optional split labels), use
`scripts/write_site_split_tags.py` (a dataset helper, not a library entry
point).

---

## Taxonomy

`tag_taxonomy.yaml` is a human reference for known dimensions and values. It
is not an allow-list — any `dimension:value` on a sidecar is valid. Query
and mutate ignore this file. `list_known_tags()` loads it and warns that it
can differ from tags actually present on disk; malformed entries are
skipped silently.

---

## API

### `TagStore(source=None)`

| `source`               | Behavior                                           |
| ---------------------- | -------------------------------------------------- |
| `None`                 | Empty in-memory store                              |
| `"/path/to/index.db"`  | Open pre-built SQLite index                        |
| `"/path/to/dataset"`   | Scan directory recursively for NPZ + sidecar pairs |
| `"/path/to/list.json"` | Scan paths listed in the JSON file                 |
| `[path1, path2, ...]`  | Scan multiple sources                              |
| single `.npz`          | Scan a single frame                                |

```python
store = TagStore()                       # empty
store = TagStore("/path/to/dataset")     # scan on init
store = TagStore("/path/to/index.db")    # open pre-built index
```

`TagStore` recognises `.db`, `.sqlite`, and `.tags.db` as SQLite index
extensions on init.

### `TagStore.build_index(source, output)`

Scan `source`, build the in-memory index, and persist it to `output`
(typically a `.db` file) via `VACUUM INTO`. Returns a `TagStore` backed
by the file at `output`.

```python
store = TagStore.build_index("/path/to/dataset", "/path/to/tags.db")
```

### Scope

`query`, `group_by`, `tags_of`, `add_tags`, `remove_tags`,
`remove_dimension`, and `replace_tags` all share a `scope` parameter that
narrows without ever adding frames outside the store's index.

| Value                    | Meaning                                          |
| ------------------------ | ------------------------------------------------ |
| `None`                   | Entire index (default)                           |
| `Path` (route directory) | Single route                                     |
| `Path` (`.npz` file)     | The route containing that NPZ                    |
| `list[Path]`             | Mix of routes / NPZ / `TagStore`; flattened union |
| `TagStore`               | Routes from another store's index (nested query) |

Resolution order, intersection with the index, the nested-store case, and
performance are in [design.md](design.md).

### `store.query(clause=None, *, granularity="route", scope=None)`

Return paths matching the clause.

- `clause`: `"dim:value"` exact match, `"dim:*"` wildcard, or
  `{"all": [...]} / {"any": [...]} / {"not": ...}`. `None` returns
  everything in scope.
- `granularity`: `"route"` (default, union semantics) or `"frame"`
  (co-occurrence).

```python
store.query("split:auto")
store.query("override:*")
store.query({"all": ["lateral:turn", "longitudinal:yield"]})
store.query("lateral:turn", granularity="frame")

route = store.route_paths()[0]
store.query("split:auto", scope=route)
store.query("split:auto", scope=store.query("site:xxxx_site_a"))
```

### `store.tags_of(scope=None, dimensions=None, granularity="route")`

Union of tags across the active scope. `dimensions` filters to specific
prefixes.

```python
store.tags_of()                                # all tags in the index
store.tags_of(granularity="frame")             # per-NPZ union
store.tags_of(dimensions=["site"])             # only site:* tags
store.tags_of(scope="/path/to/route")          # tags on one route
```

### `store.group_by(dimensions, clause=None, *, granularity="route", drop_missing=False, scope=None)`

Group matching paths by tag dimensions. `Bucket` attributes:

- `values` (`dict[str, str | None]`) — one entry per requested dimension; `None` means the item lacks it.
- `count` (`int`) — unique members in this cell (== `len(members)`).
- `members` (`list[Path]`) — concrete paths, sorted.

`drop_missing=True` skips items missing any requested dimension. `None`
sorts last in its slot.

```python
buckets = store.group_by(["site", "lateral"], clause="split:auto")
```

### `store.add_tags(tags, *, frame_filter=None, scope=None, sync=None) -> MutationResult`

Union `tags` onto matching frames. Idempotent — a frame that already has
every requested tag is not rewritten and is not counted. Returns a
`MutationResult` with `.changed`, `.skipped`, `.failed`, `.first_error`.

`frame_filter` is `(min, max)` for a frame-number range (inclusive), or a
glob pattern string, or `None`. `sync=False` skips the fsync for this
write (the data is still flushed).

```python
result = store.add_tags(["override:centerline"])
result = store.add_tags(["override:centerline"], frame_filter=(31, 100))
result = store.add_tags(["override:centerline"], scope=route, sync=False)
```

### `store.remove_tags(tags, *, frame_filter=None, scope=None, sync=None) -> MutationResult`

Remove exact tag strings. `remove_dimension(dimension, ...)` removes
every tag whose dimension matches. Returns a `MutationResult`.

### `store.replace_tags(*, tag_pairs, scope=None, frame_filter=None, sync=None) -> MutationResult`

`tag_pairs` maps `old_tag → new_tag`. Frames without the old tag are
skipped. Entries where key == value are no-ops. Returns a `MutationResult`.

```python
result = store.replace_tags(tag_pairs={"split:eval": "split:train"})
```

### `store.add_tags_to_route(tags, route, *, frame_filter=None, sync=None) -> MutationResult`

Same as `add_tags(scope=route)`, but `route` is a required path. Raises
`ValueError` if the route is not in the index (rather than silently
dropping the unknown route).

### `format_buckets(buckets, dimensions) -> str`

Render a `group_by(...)` result as a plain-text table. The `TOTAL` row
counts each unique member once across cells.

### Concurrency

Read methods are lock-free. Concurrent writers are serialised via an
internal `RLock`; the per-thread SQLite connection and the route-tags
cache are the two invariants a multi-threaded caller can rely on.

### Out-of-band edits

Mutations use a verify-then-write protocol: `write_tags` reads the
sidecar, compares its tag set to the index's view, and raises
`StaleIndexError` on drift — the sidecar is left untouched. To recover,
call `store.reindex_tags()` and retry.

`store.diff_index_against_disk()` produces a structured report
(`IndexDiff`) listing frames with tag drift and orphan frames.
