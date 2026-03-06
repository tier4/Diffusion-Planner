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
