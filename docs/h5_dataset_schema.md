# H5 Dataset Schema

This document describes H5 format version 4 produced by
`scripts/dataset/create_h5_dataset.py`.

## Dataset layout

One source rosbag produces one `frames.h5` file. The rosbag directory hierarchy is
preserved below the configured output root. A split-level Parquet file indexes every
frame stored in those H5 shards.

```text
<output_root>/
├── <source rosbag hierarchy>/
│   └── frames.h5
└── indexes/
    ├── train.parquet
    ├── valid.parquet
    └── auto.parquet
```

Every H5 frame tensor uses `float32`. The first dimension, `N`, is the number of
valid frames generated from the source rosbag. Tensor coordinates are expressed in
the ego frame at `frame_time_ns`.

## H5 root attributes

| Attribute | Description |
|---|---|
| `format` | `diffusion_planner_frame_dataset` |
| `format_version` | `4` |
| `source_bag_path` | Source rosbag path relative to the scanned input root |
| `source_map_path` | Lanelet2 map path used for preprocessing |
| `project_id` | Vehicle/project identifier |
| `area_map_id` | Map area identifier |
| `area_map_version_id` | Map version identifier |
| `split` | `train`, `valid`, or `auto` |
| `num_frames` | Number of stored frames, `N` |
| `frame_interval_s` | Interval used when selecting frames |
| `traffic_light_timeout_s` | Maximum accepted traffic-light observation age |
| `neighbor_observation_timeout_s` | Maximum accepted tracked-object observation age |

Frame datasets use one frame per chunk, the HDF5 shuffle filter, and Zstandard
compression at level 3. Reading them requires the `hdf5plugin` package.

## H5 groups

```text
frames.h5
├── frames/
│   ├── model inputs
│   └── training labels
└── metadata/
    └── frame-level scalar metadata
```

Datasets below `frames/` are chunked one frame at a time:

```text
chunks = (1, *per_frame_shape)
```

### Model inputs

| Key | Shape | Layout |
|---|---:|---|
| `ego_agent_past` | `(N, 31, 6)` | `[x, y, cos_yaw, sin_yaw, velocity, yaw_rate]` |
| `neighbor_agents_past` | `(N, 320, 31, 4)` | `[x, y, cos_yaw, sin_yaw]` |
| `agent_shape` | `(N, 320, 2)` | `[width, length]` |
| `agent_label` | `(N, 320, 3)` | One-hot `[vehicle, pedestrian, bicycle]` |
| `lanes` | `(N, 140, 20, 6)` | Lane geometry; see below |
| `lane_types` | `(N, 140, 20)` | Left and right boundary line types; see below |
| `lanes_speed_limit` | `(N, 140, 1)` | Speed limit in m/s; zero means unavailable |
| `lane_traffic_light_past` | `(N, 140, 31, 6)` | Lane-associated traffic-light history |
| `route_lanes` | `(N, 25, 20, 6)` | Route-lane geometry; same layout as `lanes` |
| `route_lane_types` | `(N, 25, 20)` | Route boundary line types; same layout as `lane_types` |
| `route_lanes_speed_limit` | `(N, 25, 1)` | Route speed limit in m/s; zero means unavailable |
| `route_traffic_light_past` | `(N, 25, 31, 6)` | Route-associated traffic-light history |
| `intersection_area` | `(N, 10, 40, 2)` | `[x, y]` |
| `stop_lines` | `(N, 30, 2, 2)` | Two `[x, y]` points per stop line |
| `road_borders` | `(N, 30, 20, 2)` | `[x, y]` polyline points |
| `goal_pose` | `(N, 4)` | `[x, y, cos_yaw, sin_yaw]` |
| `ego_shape` | `(N, 3)` | `[base_link_to_front, vehicle_length, vehicle_width]` |
| `turn_indicators` | `(N, 31)` | `TurnIndicatorsReport.report` history |

### Lane geometry and type separation

`lanes` and `route_lanes` contain only point geometry. Their final dimension is:

| Index | Field |
|---:|---|
| 0 | Centerline `x` |
| 1 | Centerline `y` |
| 2 | Left-boundary `x - centerline_x` |
| 3 | Left-boundary `y - centerline_y` |
| 4 | Right-boundary `x - centerline_x` |
| 5 | Right-boundary `y - centerline_y` |

Boundary types are stored once per lane segment rather than repeated for every point:

```text
lane_types[..., 0:10]   = left-boundary type one-hot
lane_types[..., 10:20]  = right-boundary type one-hot
```

The ten type indices are:

| Index | Type |
|---:|---|
| 0 | `crosswalk` |
| 1 | `curbstone` |
| 2 | `guard_rail` |
| 3 | `line_thick` |
| 4 | `line_thin` |
| 5 | `pedestrian_marking` |
| 6 | `road_border` |
| 7 | `road_shoulder` |
| 8 | `virtual` |
| 9 | `zebra_marking` |

### Traffic-light encoding

The final dimension of every traffic-light tensor is:

| Index | Field |
|---:|---|
| 0 | Green |
| 1 | Amber |
| 2 | Red |
| 3 | Unknown or unavailable |
| 4 | White or no light |
| 5 | Arrow flag |

The first five entries form the color/state one-hot encoding. The arrow flag is an
additional binary attribute.

### Training labels

| Key | Shape | Layout |
|---|---:|---|
| `ego_agent_future` | `(N, 80, 6)` | `[x, y, cos_yaw, sin_yaw, velocity, yaw_rate]` |
| `neighbor_agents_future` | `(N, 320, 80, 4)` | `[x, y, cos_yaw, sin_yaw]` |
| `turn_indicators_future` | `(N, 80)` | `TurnIndicatorsReport.report` future sequence |
| `lane_traffic_light_future` | `(N, 140, 80, 6)` | Future lane-associated traffic-light states |
| `route_traffic_light_future` | `(N, 25, 80, 6)` | Future route-associated traffic-light states |

Unused or unavailable padded elements are all-zero. For pose sequences, a valid pose
has a normalized yaw pair, so an all-zero pose is unambiguously invalid.

### Frame metadata

| Key | Shape | Dtype | Description |
|---|---:|---|---|
| `frame_time_ns` | `(N,)` | `int64` | Source frame timestamp in nanoseconds |
| `ego_speed_mps` | `(N,)` | `float32` | Ego longitudinal speed |
| `ego_yaw_rate_rps` | `(N,)` | `float32` | Ego yaw rate |
| `turn_indicator` | `(N,)` | `uint8` | Current `TurnIndicatorsReport.report` value |
| `num_objects` | `(N,)` | `int32` | Number of tracked objects at the frame |

## Segment tag sidecar

Curated tags are not stored inside `frames.h5`. They are mutable dataset metadata,
whereas H5 shards are versioned preprocessing outputs. A shard may instead have an
adjacent, authoritative JSON sidecar named `frames.tags.json`:

```text
<source rosbag hierarchy>/
├── frames.h5
└── frames.tags.json
```

Keeping the sidecar adjacent means tags travel when a bag directory is copied, while
adding or revising tags never requires rewriting the H5 payload or changing its format
version. A missing sidecar means that the shard has no curated tags.

### Sidecar format

The initial sidecar format is `diffusion_planner_h5_tags`, version `1`:

```json
{
  "format": "diffusion_planner_h5_tags",
  "format_version": 1,
  "h5": {
    "format": "diffusion_planner_frame_dataset",
    "format_version": 4,
    "source_bag_path": "x2_dev/example/train/2026-01-15/13-24-21",
    "num_frames": 873,
    "first_frame_time_ns": 1736598240000000000,
    "last_frame_time_ns": 1736598676000000000
  },
  "spans": [
    {
      "frame_start": 0,
      "frame_stop": 873,
      "first_frame_time_ns": 1736598240000000000,
      "last_frame_time_ns": 1736598676000000000,
      "tags": ["split:validation", "site:odaiba", "split:train", "project:x2_dev"]
    },
    {
      "frame_start": 120,
      "frame_stop": 171,
      "first_frame_time_ns": 1736598300000000000,
      "last_frame_time_ns": 1736598300000000000,
      "tags": ["override:lane_change"]
    },
    {
      "frame_start": 125,
      "frame_stop": 126,
      "first_frame_time_ns": 1736598300000000000,
      "last_frame_time_ns": 1736598300000000000,
      "tags": ["turn_indicator:left"]
    }
  ]
}
```

`h5` is an identity and compatibility check for the sibling shard, not a path to it.
The sidecar must not contain an absolute path. `source_bag_path`, H5 format/version,
frame count, and the first/last source timestamps must match the H5 file before a
consumer uses its tags.

Each item in `spans` attaches one or more tags to a half-open H5 frame interval
`[frame_start, frame_stop)`. The required constraints are:

- `0 <= frame_start < frame_stop <= num_frames`.
- `first_frame_time_ns` equals `metadata/frame_time_ns[frame_start]` and
  `last_frame_time_ns` equals `metadata/frame_time_ns[frame_stop - 1]`.
- A one-frame tag is represented by `frame_stop = frame_start + 1`.
- The complete shard is `[0, num_frames)`; its final real timestamp is
  `metadata/frame_time_ns[num_frames - 1]`. There is deliberately no inferred
  timestamp after the final frame, because source frame spacing need not be exact.
- Spans may overlap. The effective tags at a frame are the union of tags from every
  span containing that frame.
- A span has a non-empty, sorted, duplicate-free `tags` array. Identical spans are
  invalid; overlapping spans with different tags are valid.

Tags use the same open vocabulary as `tag_toolkit`: `dimension:value`, with both
components matching lowercase `[a-z0-9_]+`. The format does not impose a taxonomy
allow-list, so projects can introduce new dimensions and values without a schema
release.

`project_id`, `split`, and other values already represented by H5 root attributes
should not be duplicated in the sidecar. Tag query tools may expose such values as
derived tags (for example, `project:x2_dev` or `split:train`) together with the
curated sidecar tags.

### Derived tag indexes and evaluation output

`frames.tags.json` is the source of truth. A dataset-level SQLite or Parquet index is
optional and must be rebuildable from sidecars. A normalized index stores one row per
`span` and tag, with at least:

```text
h5_path, source_bag_path, frame_start, frame_stop,
first_frame_time_ns, last_frame_time_ns, tag
```

`h5_path` in such an index is relative to the index file. Tags must not become
authoritative columns in the normal training frame index; a tag query may instead
materialize a separate filtered frame index for a particular training or evaluation
selection.

Open-loop result rows should include the effective tags at their referenced frame.
Closed-loop result rows should include the H5 path, global `[frame_start, frame_stop)`
range, the union of overlapping tags, and per-tag tagged-frame coverage. Coverage
makes a tag that touches only part of an evaluated segment distinguishable from one
that covers the full segment.

## Parquet frame index

Each Parquet row addresses exactly one H5 frame.

| Column | Type | Description |
|---|---|---|
| `h5_path` | string | Relative POSIX path from the Parquet index directory to `frames.h5` |
| `frame_index` | int64 | Index into the first dimension of every H5 dataset |
| `frame_time_ns` | int64 | Source frame timestamp |
| `ego_speed_mps` | float32 | Copied frame metadata |
| `ego_yaw_rate_rps` | float32 | Copied frame metadata |
| `turn_indicator` | uint8 | Copied frame metadata |
| `num_objects` | int32 | Copied frame metadata |
| `project_id` | string | Project identifier |
| `area_map_id` | string | Map area identifier |
| `area_map_version_id` | string | Map version identifier |
| `split` | string | Dataset split |

## Compatibility

Format version 4 is incompatible with version 3 because it adds closed-loop pose
metadata (`ego_x`, `ego_y`, and `ego_yaw`). Existing H5 shards must be regenerated
with `overwrite=true`; `resume=true` rejects shards with an older format version.
