# Diffusion Planner Tools

Offline tools for benchmarking and testing the diffusion planner outside of Autoware runtime.
These tools link against the **actual autoware_universe inference code**, so results always reflect
the real host-side behavior (bind-once, pinned memory, etc.) of whatever branch you build against.

## Prerequisites

- Autoware workspace built (`~/autoware/install/` exists)
- ONNX model + param files in `~/autoware_data/diffusion_planner/`

## Quick Start

```bash
# 1. Source Autoware
source /opt/ros/humble/setup.bash
source ~/autoware/install/setup.bash

# 2. Build tools
cd cpp_tools
./build.sh

# 3. Source tools
source install/setup.bash

# 4. Run benchmark
ros2 run autoware_diffusion_planner_tools benchmark_tool
```

## Benchmark Tool

Measures end-to-end TRT inference latency (H2D + inference + D2H) using the real
`TensorrtInference` class from `autoware_diffusion_planner`.

```bash
# Use default config from installed autoware_diffusion_planner package
ros2 run autoware_diffusion_planner_tools benchmark_tool

# Custom runs/warmup
ros2 run autoware_diffusion_planner_tools benchmark_tool --warmup 50 --runs 300

# Use a custom config
ros2 run autoware_diffusion_planner_tools benchmark_tool --config /path/to/diffusion_planner.param.yaml
```

### Options

| Option           | Description                        | Default                          |
| ---------------- | ---------------------------------- | -------------------------------- |
| `--config PATH`  | Path to planner param yaml         | from installed package           |
| `--warmup N`     | Warmup iterations                  | `50`                             |
| `--runs N`       | Benchmark iterations               | `300`                            |

## Inference Tool

Runs the full diffusion planner pipeline (preprocessing + inference + postprocessing)
on a rosbag and writes results to an output rosbag.

```bash
ros2 run autoware_diffusion_planner_tools inference_tool \
  <rosbag_path> <vector_map_path> <output_rosbag_path>
```

## Data Converter

Converts rosbag data into Diffusion Planner training/evaluation files. The converter
writes `.npz` frame tensors and JSON sidecars under the output directory.

### Convert one rosbag

Use `data_converter` when the rosbag and vector map are known explicitly.

```bash
ros2 run autoware_diffusion_planner_tools data_converter \
  <rosbag_path> \
  <vector_map_path> \
  <save_dir>
```

`vector_map_path` is usually the Lanelet2 map file, for example
`/path/to/map/lanelet2_map.osm`.

### Convert rosbag directories

Use `parse_rosbag_for_directory_with_map_version` to recursively search one or more directories for
rosbag `metadata.yaml` files and convert all discovered bags. The tool resolves
`lanelet2_map.osm` from nearby `map/` directories and writes each bag's output below
`--save_root`, preserving the relative bag path where possible.

```bash
ros2 run autoware_diffusion_planner_tools parse_rosbag_for_directory_with_map_version \
  <target_dir> [<target_dir> ...] \
  --save_root <save_root> \
  --map_version_source log_file_info \
  --num_workers 32
```

Set `--num_workers` to `0` or a negative value to use hardware concurrency.
Set `--map_version_source` to `log_file_info` to read `area_map_version_id` from
`log_file_info.json`, or `metadata` to read it from `metadata.yaml`. The default is
`log_file_info`, matching `ros_scripts/parse_rosbag_for_directory_with_map_version.py`.

### Converter options

Both converter commands accept these options:

| Option | Description | Default |
| --- | --- | --- |
| `--step N` | Frame sampling interval in 10 Hz ticks | `3` |
| `--limit N` | Maximum rosbag messages to read; `-1` reads all messages | `-1` |
| `--min_frames N` | Minimum assembled frames required to accept a sequence | `1700` |
| `--min_distance M` | Minimum traveled ego distance in meters | `50.0` |
| `--search_nearest_route 0/1` | Use the latest route at or before each frame timestamp | `1` |
| `--convert_yellow 0/1` | Keep yellow-light frames instead of skipping them | `0` |
| `--convert_red 0/1` | Keep red-light frames instead of skipping them | `0` |
| `--interpolation 0/1` | Use timestamp-based interpolation for ego trajectories | `1` |
| `--ego_wheel_base M` | Ego vehicle wheel base in meters | `-1.0` |
| `--ego_length M` | Ego vehicle length in meters | `-1.0` |
| `--ego_width M` | Ego vehicle width in meters | `-1.0` |
| `--static_object_margin M` | Static-object collision filter margin | `0.0` |
| `--neighbor_margin M` | Neighbor-agent collision filter margin | `0.0` |
| `--road_border_margin M` | Road-border collision filter margin | `0.0` |
| `--collision_time_stride N` | Time stride for trajectory collision filters | `5` |
| `--offlane_max_score M` | Off-lane filter maximum average distance from lane centerlines | `6.0` |
| `--offlane_time_stride N` | Time stride for the off-lane filter | `1` |
| `--write_skipped_npz 0/1` | Also write `.npz` files for skipped frames | `0` |

## Per-frame JSON sidecar (data converter)

The data converter writes one JSON sidecar next to each frame's `.npz`, carrying the
absolute map ego pose plus two fields used by downstream tooling:

| field | meaning |
| --- | --- |
| `is_skipped` (bool) | `true` if the production filter would have dropped this frame (stopped at a red/yellow light, no future progress, GT collision, off-lane, stale data). See also `skipping_info.label`. |
| `neighbor_ids` (list[str]) | perception track UUIDs of the kept neighbors, aligned 1:1 with the `neighbor_past` slots (sorted by ego distance, trimmed). Lets a consumer associate the same agent across frames. |

By default the converter **drops** flagged frames (writes only accepted ones). Pass
`--write_skipped_npz=1` to instead write **every** 10 Hz frame, flagged in the sidecar —
producing one unified corpus that the closed-loop perception reproducer can replay
gap-free while training/eval skip the flagged frames (see "Skip-for-training filtering"
in `scenario_generation/README.md`).

Both fields are additive and backward-compatible: the `.npz` tensors are unchanged, and
a sidecar lacking them is treated as "not skipped" / "no track ids".
