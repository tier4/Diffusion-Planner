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

# Scan and build in-memory index
store = TagStore("/path/to/dataset")
routes = store.query("split:auto")  # instant
```

### 2. Index mode (fast for large datasets)

For large datasets, build a pickled index once and load it for repeated queries:

```python
from tag_toolkit import TagStore

# Build index (do once after updating tags). The output path must end in
# `.tag` — TagStore only reloads files whose suffix is `.tag`, so a `.pkl`
# or unsuffixed file would be re-scanned as a dataset and silently fail
# to load.
TagStore.build_index("/path/to/dataset", "/path/to/tags.tag")

# Load from index (fast)
store = TagStore("/path/to/tags.tag")
routes = store.query("split:auto")
```

## Route vs frame

Tags are stored per frame, but most workflows care about whole routes. So
`query(...)` and `group_by(...)` default to `granularity="route"`.

At route granularity, a route has a tag if **any** frame under it has that tag.
Use `granularity="frame"` when you need per-frame results instead.

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
from pathlib import Path
from tag_toolkit import TagStore

store = TagStore("/path/to/tags.tag")

# Tag every frame in the index — the typical "label the whole dataset" case.
n = store.add_tags(["override_metric:centerline"])
print(f"updated {n} sidecars")

# Or tag just one NPZ by passing it as scope. Use ``npz_paths()`` (or a
# known route) so the scope hits something already in the index — paths
# outside the index are silently dropped, but unknown paths that look like
# directories raise FileNotFoundError during scope expansion.
target_npz = store.npz_paths()[0]
n = store.add_tags(["override_metric:centerline"], scope=target_npz)
print(f"updated {n} sidecar for {target_npz}")
```

### 4. Group by multiple labels

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
xxxx_site_a  auto    1
xxxx_site_a  train   1
xxxx_site_b                          manual  1
xxxx_site_b                          valid   1
-------------------------------------  ------  -----
TOTAL                                          3
[
  PosixPath('/path/to/dataset/proj_a/xxxx_site_a/auto/2026-06-23/10-55-13'),
]
```

### 5. Replace and remove

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
# Like add_tags, scopes resolve against the in-memory index; paths not in
# the index are silently skipped.
store.replace_tags(tag_pairs={"split:eval": "split:train"})
```

## More detail

- **Usage (format, route, API examples)**: [`docs/usage.md`](docs/usage.md)
- **Design (source contract, scope resolution, atomic writes)**: [`docs/design.md`](docs/design.md)
- **Reference taxonomy** (draft only; not used by query/mutate):
  [`docs/tag_taxonomy.yaml`](docs/tag_taxonomy.yaml)
