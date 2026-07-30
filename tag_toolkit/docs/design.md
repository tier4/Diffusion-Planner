# tag_toolkit design

Tags for Diffusion Planner data live on each NPZ’s sidecar JSON (`<stem>.json`
next to `<stem>.npz`). `tag_toolkit` reads and writes that `tags` field, and
queries it at **route** or **frame** granularity.

The unit of work is a **`source`**: one `.npz`, a directory of `*.npz` (small
trees only), a path-list JSON / `.json.zst`, or a list of those. Prefer a
pre-built path list for large datasets.

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
  "project_id": "prd_jt",
  "vehicle_id": "532d0885-…",
  "tags": [
    "site:1423_shinagawa_odaiba",
    "split:manual",
    "lateral:turn",
    "longitudinal:yield"
  ]
}
```

| Rule | Detail |
|---|---|
| Type | JSON array of strings. Missing `tags` ≡ `[]`. |
| Entry | `dimension:value` (exactly one colon). |
| Names | Prefer `[a-z0-9_]+` on both sides. |
| Multi | A frame may carry several tags. |
| Order | Writers sort for stable diffs; readers should not depend on order. |

Do **not** duplicate native sidecar fields as tags (`timestamp`, `project_id`,
`vehicle_id`, pose, `date`, `bag_time`, …). Use `tags` for scene / eval labels
such as `site`, `split`, `lateral`, `longitudinal`, `override_metric`.

Unless you ask for a dimension replace or a full replace, writes **merge** into
the existing list and leave other dimensions alone.

---

## Route

[`dataset/generate_from_labeled.sh`](../../../dataset/generate_from_labeled.sh)
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

At route granularity, a route has a tag if **any indexed frame under it** has
that tag. “Indexed” means frames present in the current `source` (e.g. a path
list may cover only a subset of a bag). Matching uses that union, so
`{"all": ["lateral:turn", "longitudinal:yield"]}` can succeed even when no
single frame has both. With `granularity="frame"`, tags must co-occur on the
same NPZ.

`tags_for(route_dir)` uses the cached source-relative union when that route is
indexed; otherwise it scans the directory on disk.

| Granularity | Matching | Result paths |
|---|---|---|
| `route` (default) | Union of frame tags | route directories |
| `frame` | Tags on one NPZ | `.npz` files |

---

## `site` and `split`

Once written, `site` and `split` are ordinary tags. Query and mutate only look
at sidecar `tags`.

| Dimension | Examples |
|---|---|
| `site` | `site:1423_shinagawa_odaiba`, `site:unknown` |
| `split` | `split:auto`, `split:train`, `split:valid`, `split:manual`, `split:unknown` |

In the layout above, the path usually shows only `manual` / `auto`. Train vs
valid is often collapsed under `manual/`; optional
[`split_labels.json`](../../../dataset/create_split_labels.py) can refine that
when writing tags.

To fill `site` / `split` for a whole dataset from paths (and optional split
labels), use:

[`scripts/write_site_split_tags.py`](../scripts/write_site_split_tags.py)

That script is a dataset helper that calls into `tag_toolkit`; it is not a
library entry point.

---

## Taxonomy

[`tag_taxonomy.yaml`](tag_taxonomy.yaml) is a human reference for known
dimensions and values. It is not an allow-list: any `dimension:value` on a
sidecar is valid. Query and mutate ignore this file. Helpers such as
`list_known_tags()` may load it and warn that it can differ from tags on a
given `source`.

---

## API

### Mutate

Always writes frame sidecars. Pass the `source` you want to update (one frame,
one route directory, a path list, …).

```python
from tag_toolkit import TagStore

store = TagStore()  # optional: TagStore(dataset_or_path_list) to set a default source

store.add_tags(route_dir, ["lateral:turn", "split:manual"])
store.add_tags(frame_path, ["scene:merge"])
store.remove_tags(route_dir, ["lateral:turn"])
store.replace_tags(route_dir, dimension="lateral", values=["turn"])
store.set_tags(route_dir, tags=[...], mode="merge")   # or mode="replace"
```

A route directory as `source` updates every NPZ under it. Unknown dimensions
and values are allowed.

### Query / group

```python
store = TagStore(source=...)

store.query("lateral:turn")                       # route dirs
store.query("lateral:turn", granularity="frame")  # npz paths
store.query({"all": ["split:manual", "lateral:turn"]})

buckets = store.group_by(["site", "lateral"])
print(format_buckets(buckets, ["site", "lateral"]))  # dimensions + count
print(buckets[0].members)  # matching routes (or frames if granularity="frame")

store.tags_for(route_dir)   # union of frame tags
store.tags_for(npz_path)  # one frame
```

`format_buckets(...)` prints `dimensions` + per-cell `count`. Detailed paths live
in `bucket.members`. Multi-value dimensions can put one member in several cells;
`TOTAL` then counts unique members when those lists are present.

Clause forms: `"dim:value"`, `{"all": [...]}`, `{"any": [...]}`, `{"not": ...}`.

---

## Sidecar examples

```json
{ "timestamp": 1, "tags": [] }
```

```json
{
  "timestamp": 1,
  "project_id": "xx1_psim",
  "tags": ["site:unknown", "split:auto"]
}
```

```json
{
  "timestamp": 1,
  "tags": [
    "site:879_hiratsuka",
    "split:train",
    "lateral:lane_change",
    "longitudinal:gap_search"
  ]
}
```
