# tag_toolkit

`tag_toolkit` is a small library for reading and writing **tags** stored on NPZ
sidecar JSON files, and for querying them at **route** or **frame** granularity.

Instead of keeping a separate tag database, each frame's `<stem>.json` holds a
`tags` field:

```json
{
  "timestamp": 1738632874843986836,
  "project_id": "proj_c",
  "vehicle_id": "532d0885-...",
  "tags": [
    "site:xxxx_site_example",
    "split:manual",
    "lateral:turn"
  ]
}
```

Tags travel with the data. If a dataset is moved or partially copied, the labels
stay attached to the same frames.

> **Try it against `tag_toolkit/sample_dataset/`** — every example in this
> README and in `docs/usage.md` runs as-is against the checked-in fixture.

## Three usage modes

### 1. Scan mode (no index file needed)

For small datasets or one-off queries, pass a directory directly:

```python
from tag_toolkit import TagStore

store = TagStore("/path/to/dataset")
routes = store.query("split:auto")  # instant
```

### 2. Index mode (fast for large datasets)

For large datasets, build a pickled index once and load it for repeated
queries. The output path must end in `.tag` — `TagStore` only reloads
files whose suffix is `.tag`, so a `.pkl` or unsuffixed file would be
re-scanned as a dataset and silently fail to load.

```python
from tag_toolkit import TagStore

TagStore.build_index("/path/to/dataset", "/path/to/tags.tag")
store = TagStore("/path/to/tags.tag")
routes = store.query("split:auto")
```

Note: `add_tags` and friends update the in-memory index immediately, but
they do **not** rewrite the saved `.tag` file. To refresh the on-disk
pickle after on-the-fly mutations, run `build_index` again (with the same
source).

## Route vs frame

Tags are stored per frame, but most workflows care about whole routes. So
`query(...)` and `group_by(...)` default to `granularity="route"`.

At route granularity, a route has a tag if **any** frame under it has
that tag. Use `granularity="frame"` when you need per-frame results.

## Tag format

Tags are stored in the NPZ sidecar JSON file, alongside native fields:

```json
{
  "timestamp": 1738632874843986836,
  "project_id": "proj_c",
  "vehicle_id": "532d0885-...",
  "tags": [
    "site:xxxx_site_example",
    "split:manual",
    "lateral:turn"
  ]
}
```

Each tag is `dimension:value` (lowercase `[a-z0-9_]+` on both sides). A frame may have
zero or more tags. Do not duplicate native sidecar fields (`timestamp`, `project_id`, …)
as tags.

**Examples:** `site:xxxx_site_example`, `split:train`, `lateral:lane_change`,
`override_metric:centerline`.

## Typical workflows

### 1. Quick query on small dataset

```python
from tag_toolkit import TagStore

store = TagStore("/path/to/small_dataset")
routes = store.query("split:auto")
print(f"Found {len(routes)} routes")
```

### 2. Build and cache index for large dataset

```python
from tag_toolkit import TagStore

# Build once (slow)
TagStore.build_index("/path/to/large_dataset", "/path/to/tags.tag")

# Load from index (fast, repeated use)
store = TagStore("/path/to/tags.tag")
routes = store.query("split:auto")
```

### 3. Add tags to frames

```python
from tag_toolkit import TagStore

store = TagStore("/path/to/tags.tag")

# Tag every frame in the index — the typical "label the whole dataset" case.
n = store.add_tags(["override_metric:centerline"])
print(f"updated {n} sidecars")

# Or tag just one NPZ by passing it as scope. Use npz_paths() (or a known
# route) so the scope hits something already in the index.
target_npz = store.npz_paths()[0]
n = store.add_tags(["override_metric:centerline"], scope=target_npz)
print(f"updated {n} sidecar for {target_npz}")
```

Mutations verify that the on-disk sidecar matches the in-memory index
before writing. If anything has drifted (for example an out-of-band
script edited a sidecar), the mutation raises `StaleIndexError` and
leaves the sidecar untouched. Reconcile with `store.reindex_tags()` and
retry:

```python
from tag_toolkit import StaleIndexError

try:
    store.add_tags(["override_metric:centerline"], scope=target_npz)
except StaleIndexError:
    print("drift detected — reindexing")
    store.reindex_tags()
    store.add_tags(["override_metric:centerline"], scope=target_npz)
```

### 4. Batch writes (fast for large datasets)

For bulk operations, use `mutation_scope` — all per-file fsyncs are
automatically batched and a single directory-fsync happens at scope exit:

```python
from tag_toolkit import TagStore

store = TagStore("/path/to/tags.tag")

# Every mutating call inside the scope is automatically batched —
# no per-file fsync, just one directory-fsync on exit.
with store.mutation_scope():
    store.add_tags(["site:foo"], scope=route1)
    store.add_tags(["site:bar"], scope=route2)
    store.add_tags_to_route(["env:prod"], route3)
# All fsync happens here, once.
```

### 5. Group by multiple labels

```python
from tag_toolkit import TagStore, format_buckets

store = TagStore("/path/to/tags.tag")
buckets = store.group_by(["site", "split"])
print(format_buckets(buckets, ["site", "split"]))
print(buckets[0].members[:2])  # route list in this bucket
```

```text
site                                   split   count
-------------------------------------  ------  -----
xxxx_site_a                            auto    1
xxxx_site_a                            train   1
xxxx_site_c                            manual  1
xxxx_site_c                            valid   1
-------------------------------------  ------  -----
TOTAL                                          3
```

### 6. Replace and remove

```python
from tag_toolkit import TagStore

store = TagStore("/path/to/tags.tag")

# Add
store.add_tags(["lateral:turn"])

# Remove exact tag strings
store.remove_tags(["lateral:turn"])

# Remove every tag under a dimension (e.g. all `lateral:*` tags)
store.remove_dimension("lateral")

# Replace one specific tag with another, atomically per sidecar.
store.replace_tags(tag_pairs={"split:eval": "split:train"})
```

## More detail

- **Usage (format, route, API examples)**: [`docs/usage.md`](docs/usage.md)
- **Design (source contract, scope resolution, atomic writes)**: [`docs/design.md`](docs/design.md)
- **Reference taxonomy** (draft only; not used by query/mutate):
  [`docs/tag_taxonomy.yaml`](docs/tag_taxonomy.yaml)
