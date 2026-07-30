# tag_toolkit

`tag_toolkit` is a small library for reading and writing **tags** stored on NPZ
sidecar JSON files, and for querying them at **route** or **frame** granularity.

Instead of keeping a separate tag database, each frame's `<stem>.json` holds a
`tags` field:

```json
{
  "timestamp": 1738632874843986836,
  "project_id": "prd_jt",
  "vehicle_id": "532d0885-...",
  "tags": [
    "site:1423_shinagawa_odaiba",
    "split:manual",
    "lateral:turn"
  ]
}
```

Tags travel with the data. If a dataset is moved or partially copied, the labels
stay attached to the same frames.

## Route vs frame

Tags are stored per frame, but most workflows care about whole routes. So
`query(...)` and `group_by(...)` default to `granularity="route"`.

At route granularity, a route has a tag if **any** frame under it has that tag.
Use `granularity="frame"` when you need per-frame results instead.


## What a `source` means

Most APIs operate on a **source**:

- one `.npz` file
- one directory (recursive `*.npz`)
- one `path_list.json` / `.json.zst`
- a Python list mixing the above

For large datasets, prefer a pre-built `path_list.json` (same shape as training).

## Tag format

- Stored in `tags: list[str]`
- Each entry is `dimension:value` (prefer `[a-z0-9_]+` on both sides)
- A frame may carry several tags; missing `tags` ≡ `[]`
- Do not duplicate native sidecar fields (`timestamp`, `project_id`, …) as tags

Examples: `site:1423_shinagawa_odaiba`, `split:train`, `lateral:lane_change`,
`override_metric:centerline`.

## Typical workflows

### 1. Tag every frame under a route directory

```python
from tag_toolkit import TagStore

store = TagStore()  # you can also pass a dataset root / path list to cache all npz data

route_dir = (
    "your_dataset/"
    "rAwaNfK1/2479_Nishishinjuku_Ward_DP/manual/2026-06-12/09-36-32"
)
frame_path = route_dir + "/routes/000123.npz"

updated_route = store.add_tags(route_dir, ["override_metric:centerline"])
updated_frame = store.add_tags(frame_path, ["scene:merge"])
print(updated_route, updated_frame)
```

```text
187 1
```

This tags either a whole route or one specific frame, depending on the `source`
you pass.

### 2. Query routes from a path list

```python
from tag_toolkit import TagStore

store = TagStore("your_path_list.json")
routes = store.query("split:auto")          # default: route level
frames = store.query("split:auto", granularity="frame")
print(len(routes), len(frames))
print(routes[0])
```

```text
15 31
your_dataset/x2_dev/2231_odaiba_shinagawa/auto/2026-06-09/11-31-31
```

Fifteen routes in that eval list carry `split:auto` (31 frames total). Members
are bag directories, ready to feed closed-loop tooling.

### 3. Group by multiple labels

```python
from tag_toolkit import TagStore, format_buckets

store = TagStore("your_path_list.json")
buckets = store.group_by(
    ["site", "lateral"],
    clause={"any": ["lateral:turn", "lateral:lane_change"]},
)
print(format_buckets(buckets, ["site", "lateral"]))
print(buckets[0].members[:2])  # route list in this bucket
```

```text
site                    lateral      count
----------------------  -----------  -----
1423_shinagawa_odaiba   turn         12
1423_shinagawa_odaiba   lane_change  4
2416_odaiba             turn         7
----------------------  -----------  -----
TOTAL                                23
[
  PosixPath('your_dataset/prd_jt/1423_shinagawa_odaiba/manual/2025-02-04/10-34-24'),
  PosixPath('your_dataset/prd_jt/1423_shinagawa_odaiba/manual/2025-02-04/10-41-08'),
]
```
At default route granularity they are route directories; with `granularity="frame"`
they are NPZ paths. Route unions / counts are relative to the current `source`.

## CLI

```bash
# Routes matching a tag (default)
python -m tag_toolkit \
  --source try_train_assets/open_loop_matrix_flat.json \
  query 'split:auto'

# Frames instead
python -m tag_toolkit \
  --source try_train_assets/open_loop_matrix_flat.json \
  query 'split:auto' --granularity frame

# Tag one route directory
python -m tag_toolkit \
  --source /path/to/one/route_dir \
  add override_metric:centerline

# Group routes
python -m tag_toolkit \
  --source try_train_assets/open_loop_matrix_flat.json \
  group-by site split

```

## Notes on scale

- Prefer `path_list.json` over walking a multi-TB dataset root.
- Pure NPZ path lists load without per-entry `resolve()` / `exists()` / `rglob()`.
- First query builds an in-memory index: per-frame tags, per-route unions, and
  inverted maps `tag → routes` / `tag → frames`. Mutate invalidates the index.
- The dataset-wide `site` / `split` auto-tagging flow lives in an external
  script (`tag_toolkit/scripts/write_site_split_tags.py`), not in the public
  `tag_toolkit` API.

## More detail

- **Design and API contract**: [`docs/design.md`](docs/design.md)
- **Reference taxonomy** (draft only; not used by query/mutate):
  [`docs/tag_taxonomy.yaml`](docs/tag_taxonomy.yaml)
