# tag_toolkit usage

How tags are stored, what a route is, and how to use the library day to day.
Scope resolution behavior is in `[design.md](design.md)`.

Tags for Diffusion Planner data live on each NPZ's sidecar JSON (`<stem>.json`
next to `<stem>.npz`). `tag_toolkit` reads and writes that `tags` field, and
queries it at **route** or **frame** granularity.

For large datasets, build the index once with `TagStore.build_index` and load
the pickled `.tag` file thereafter; the index avoids per-frame I/O.

---

## Why tags live on the sidecar

Dataset directory layouts change across builds. A separate tag DB keyed by path
quickly goes stale when data is re-converted or partially copied.

Putting labels on the same sidecar as the frame keeps them together: move or
subset the data, and the tags move with it.

---



## Tag format

```json
{
  "timestamp": 1738632874843986836,
  "project_id": "proj_c",
  "vehicle_id": "532d0885-…",
  "tags": [
    "site:xxxx_site_example",
    "split:manual",
    "lateral:turn",
    "longitudinal:yield"
  ]
}
```


| Rule  | Detail                                                             |
| ----- | ------------------------------------------------------------------ |
| Type  | JSON array of strings. Missing `tags` ≡ `[]`.                      |
| Entry | `dimension:value` (exactly one colon).                             |
| Names | Prefer `[a-z0-9_]+` on both sides.                                 |
| Multi | A frame may carry several tags.                                    |
| Order | Writers sort for stable diffs; readers should not depend on order. |


Writes are atomic at the file level: a sibling `.json.tmp` is created and
renamed over the target on success. If a write fails, the original sidecar is
untouched and the temp file is removed.

---



## Route

`[dataset/generate_from_labeled.sh](../../../dataset/generate_from_labeled.sh)`
defines the converted layout:

```text
…/<project>/<map_id>/{manual|auto}/<date>/<bag_time>/routes/*.npz
```

A **route** is that bag directory (`…/<bag_time>/` — parent of `routes/` when
present). Closed-loop path lists use the same directories.
`route_of(path)` maps an NPZ (or a route dir) to this route root.

---



## Route vs frame

Tags are stored per frame. Most workflows care about whole routes, so
`query` / `group_by` default to `granularity="route"`.

At route granularity, a route has a tag if **any** frame under it (within the
active frame set for that call) has that tag. Matching uses that union, so
`{"all": ["lateral:turn", "longitudinal:yield"]}` can succeed even when no
single frame has both. With `granularity="frame"`, tags must co-occur on the
same NPZ.


| Granularity       | Matching            | Result paths      |
| ----------------- | ------------------- | ----------------- |
| `route` (default) | Union of frame tags | route directories |
| `frame`           | Tags on one NPZ     | `.npz` files      |


---



## `site` and `split`

Once written, `site` and `split` are ordinary tags. Query and mutate only look
at sidecar `tags`.


| Dimension | Examples                                                                    |
| --------- | --------------------------------------------------------------------------- |
| `site`    | `site:xxxx_site_example`, `site:unknown`                                |
| `split`   | `split:auto`, `split:train`, `split:valid`, `split:manual`, `split:unknown` |


In the layout above, the path usually shows only `manual` / `auto`. Train vs
valid is often collapsed under `manual/`; optional
`[split_labels.json](../../../dataset/create_split_labels.py)` can refine that
when writing tags.

To fill `site` / `split` for a whole dataset from paths (and optional split
labels), use:

`[scripts/write_site_split_tags.py](../scripts/write_site_split_tags.py)`

That script is a dataset helper that calls into `tag_toolkit`; it is not a
library entry point.

---



## Taxonomy

`[tag_taxonomy.yaml](tag_taxonomy.yaml)` is a human reference for known
dimensions and values. It is not an allow-list: any `dimension:value` on a
sidecar is valid. Query and mutate ignore this file. Helpers such as
`list_known_tags()` may load it and warn that it can differ from tags actually
present on disk; malformed entries are skipped with a warning rather than
raising.

---



## API Reference



### TagStore(source=None)

Create a TagStore. The `source` parameter accepts any shape that
`[expand_source](../source.py)` (or a list of them) understands:


| `source`               | Behavior                                           |
| ---------------------- | -------------------------------------------------- |
| `None`                 | Empty in-memory store                              |
| `"/path/to/index.tag"` | Load pre-built pickled index                       |
| `"/path/to/dataset"`   | Scan directory recursively for NPZ + sidecar pairs |
| `"/path/to/list.json"` | Scan paths listed in the JSON file                 |
| `[path1, path2, ...]`  | Scan multiple sources                              |
| single `.npz`          | Scan a single frame                                |


```python
store = TagStore()                       # empty
store = TagStore("/path/to/dataset")     # scan on init
store = TagStore("/path/to/index.tag")   # load index
```



### TagStore.build_index(source, output)

Scan `source`, build the in-memory index, and write it to `output` (pickle).
Returns a new TagStore with the built index loaded.

```python
TagStore.build_index("/path/to/dataset", "/path/to/tags.tag")
```



### store.npz_paths() / store.route_paths() / store.has_index()

Read-only accessors for the indexed frame set. `has_index()` is True for any
store that has loaded or scanned anything.

`npz_paths()` and `route_paths()` raise `ValueError("no index loaded")` when
called on a store with no source (`TagStore()` / `TagStore(None)`). This is
the same fail-fast behaviour as `query` and `group_by`. The only way to read
paths from an empty store is `has_index()` first.

### Scope (common parameter)

`query`, `group_by`, `tags_of`, `add_tags`, `remove_tags`, `remove_dimension`,
and `replace_tags` all share a `scope` parameter that narrows the operation
without ever adding frames outside the store's index.


| Value                    | Meaning                                          |
| ------------------------ | ------------------------------------------------ |
| `None`                   | Entire index (default)                           |
| `Path` (route directory) | Single route                                     |
| `Path` (`.npz` file)     | The route containing that NPZ                    |
| `list[Path]`             | Mix of routes, NPZ paths, and `TagStore` entries; flattened union |
| `TagStore`               | Routes from another store's index (nested query) |


Resolution order, intersection with the index, the nested-store case, and
performance characteristics are in `[design.md](design.md#scope-resolution)`.

### store.query(clause=None, *, granularity="route", scope=None)

Return paths matching the clause.

| Parameter     | Values                               | Default   |
| ------------- | ------------------------------------ | --------- |
| `clause`      | tag string, `dict`, or `None` (all)  | `None`    |
| `granularity` | `"route"` or `"frame"`               | `"route"` |
| `scope`       | see [Scope](#scope-common-parameter) | `None`    |

`clause` is **keyword-only** — call it positionally only via the legacy
positional slot at the start of the signature (`query("split:auto")`). To
get "every route in scope" without a filter, pass `None` (or omit entirely):

```python
store.query()                         # every route in the index
store.query(granularity="frame")      # every NPZ in the index
store.query("split:auto")             # positional legacy form
store.query(clause="split:auto")      # keyword form
```


Clause forms:

- `"dim:value"` — exact match
- `"dim:*"` — match any value in that dimension (wildcard)
- `{"all": ["a", "b"]}` — must have all tags
- `{"any": ["a", "b"]}` — must have at least one
- `{"not": "dim:value"}` — must not have

```python
store.query("split:auto")
store.query("override:*")              # any override: value (centerline, departure, ...)
store.query({"all": ["lateral:turn", "longitudinal:yield"]})
store.query("lateral:turn", granularity="frame")  # returns NPZ paths

# Wildcards can be combined with other operators
store.query({"not": "override:*"})         # exclude all override frames
store.query({"all": ["split:auto", "lateral:*"]})  # auto frames with any lateral action
store.query({"any": ["override:centerline", "override:departure"]})  # equivalent to "override:*"

# Scope usage — passing a real route from this store's index, the scope
# narrows results to that route. (Unknown path-like scopes raise
# FileNotFoundError; path-like scopes that hit only NPZ files are silently
# intersected with the index.)
route = store.route_paths()[0]
store.query("split:auto", scope=route)                  # single route
store.query("split:auto", scope=[route])                # list of routes
store.query("split:auto", scope=store.query("site:1423"))  # nested query
```



### store.tags_of(scope=None, dimensions=None, granularity="route") -> list[str]

Unified "give me tags" call. Returns the union of tags across whatever scope
and dimensions select.

```python
store.tags_of()                                # all tags in the index (route granularity)
store.tags_of(granularity="frame")             # per-NPZ union
store.tags_of(dimensions=["site"])             # only site:* tags
store.tags_of(scope="/path/to/route")          # tags on one route
store.tags_of(scope=["route1", "route2"])      # union over a list
```



### store.group_by(dimensions, clause=None, *, granularity="route", drop_missing=False, scope=None)

Group matching paths by tag dimensions. Buckets sort primarily by
`dimensions[0]`, then `dimensions[1]`, etc. — the order of `dimensions`
controls the sort. `None` (item missing that dimension) sorts last in its
slot.

```python
buckets = store.group_by(["site", "lateral"], clause="split:auto")
```

Each returned `Bucket` always has a `members` list (never `None`), so
`format_buckets` can always emit a `TOTAL` row that dedupes across cells.

- `values` (`dict[str, str | None]`) — one entry per requested dimension; `None` means the item lacks that dimension.
- `count` (`int`) — number of unique members in that cell.
- `members` (`list[Path]`) — concrete paths in this cell, sorted by `str(path)`.

**`drop_missing=False` (default):** an item missing a requested dimension
still appears — its `None` slot is grouped together with items that
also miss that dimension. Under multi-value dimensions this cartesian-
product behaviour means the same item can land in multiple buckets,
once per (slot × other-slot) combination; `bucket.count` is therefore
"appearances in this cell" rather than "unique items in this cell".
Use `bucket.members` if you need exact counts.

**`drop_missing=True`:** items missing any requested dimension are
silently skipped from all buckets. Use this when missingness is
not interesting (e.g. "show me (site, split) breakdown of complete routes").

| `group_by` bucket example on the sample dataset:

```text
2 buckets
---
bucket[0]
  values  = {'site': 'xxxx_site_a', 'lateral': 'turn'}
  count   = 2
  members = [PosixPath('.../sample_dataset/proj_a/xxxx_site_a/auto/2026-06-23/10-55-13'),
             PosixPath('.../sample_dataset/proj_a/xxxx_site_a/auto/2026-07-07/15-16-36')]

bucket[1]
  values  = {'site': 'xxxx_site_b', 'lateral': 'turn'}
  count   = 1
  members = [PosixPath('.../sample_dataset/proj_b/.../manual/2026-04-15/psim_training_bag_0_0')]
```

Render `buckets` as a plain-text table with `[format_buckets](#format_bucketsbuckets-dimensions---str)`:

```text
site                                   lateral  count
-------------------------------------  -------  -----
xxxx_site_a  turn     2
xxxx_site_b                          turn     1
-------------------------------------  -------  -----
TOTAL                                           3
```

The `TOTAL` row counts each unique member once across cells — because
`group_by` always populates `Bucket.members`, dedupe is correct
regardless of multi-value dimensions.

### store.add_tags(tags, frame_filter=None, scope=None)

Append tags to matching frames. Idempotent: a frame that already has every
requested tag is not rewritten and is not counted in the return value.


| Parameter      | Type                                 | Description                                    |
| -------------- | ------------------------------------ | ---------------------------------------------- |
| `tags`         | `Sequence[str]`                      | Tags to add                                    |
| `frame_filter` | `(int, int)` or `str` or `None`      | Filter by frame number range or glob pattern   |
| `scope`        | see [Scope](#scope-common-parameter) | Limit to specific routes (None = entire index) |


```python
n = store.add_tags(["override:centerline"])
n = store.add_tags(["override:centerline"], frame_filter=(31, 100))
n = store.add_tags(["override:centerline"], scope="/path/to/route")
n = store.add_tags(["override:centerline"], scope=store.query("split:auto"))
```

**Frame filter formats:**

- `(31, 100)` — frame number range (inclusive), handles non-contiguous frames automatically. The filter matches NPZs whose filename contains an 8-digit zero-padded frame number segment right before `.npz` (the convention `<bag_time>_<prefix>_<8 digits>.npz`). NPZs that don't match the convention are silently treated as "not matching"; pass a glob string for non-conventional filenames.
- `"*_0000003[0-9]*.npz"` — glob pattern matching filename.



### store.remove_tags(tags, frame_filter=None, scope=None)

Remove tags from matching frames. Same `frame_filter` and `scope` shapes as
`add_tags`. Idempotent: frames without the tag are not rewritten.

```python
n = store.remove_tags(["override:centerline"])
n = store.remove_dimension("lateral")
```



### store.remove_dimension(dimension, scope=None, frame_filter=None)

Delete every tag whose dimension is `dimension` from every frame in scope.

### store.replace_tags(*, tag_pairs, scope=None, frame_filter=None)

Replace specific tags with specific other tags, atomically per sidecar.

`tag_pairs` is a mapping of `dim:value → dim:value` strings. For each
matching frame: if the frame has the old tag, swap it for the new one.
Entries where the key equals the value are no-ops. Frames without the
old tag are silently skipped.

```python
n = store.replace_tags(tag_pairs={"split:eval": "split:train"})
```



### store.add_tags_to_route(tags, route, *, frame_filter=None)

Fast path for tagging all frames of a route already in the index. Raises
`ValueError` if the route is not in the index — tagging frames the index
doesn't know about would make them invisible to subsequent queries, so we
reject the call rather than silently scanning disk. Equivalent to
`add_tags(tags, scope=route, frame_filter=frame_filter)` with an extra
preflight check.

```python
n = store.add_tags_to_route(["site:1423", "split:train"], "/path/to/route")
n = store.add_tags_to_route(["site:1423", "split:train"], "/path/to/route", frame_filter=(31, 100))
```



### format_buckets(buckets, dimensions) -> str

Render a `group_by(...)` result as a plain-text table. `dimensions` are the
column headers; each row's values come from `bucket.values`. The `TOTAL`
row counts each unique member once across cells. Because `group_by` always
populates `Bucket.members`, the `TOTAL` row is always present and
correctly handles multi-value dimensions.
