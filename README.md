# Diffusion Planner

Diffusion Planner is an Autoware trajectory planner built with diffusion and flow
matching models. This repository contains the training pipeline, dataset tools,
visualization dashboard, ONNX export tools, and the ROS 2 inference node.

## Components

| Path | Role |
| --- | --- |
| `packages/diffusion_planner/` | Model, dataset, and visualization library |
| `packages/diffusion_planner_dashboard/` | Streamlit dataset and inference viewer |
| `scripts/dataset/` | Rosbag-to-H5 dataset generation and checks |
| `scripts/train/` | Hydra-based training |
| `scripts/export/` | ONNX export and validation |
| `configs/` | Dataset and training configuration |
| `ros2_ws/src/ml_planner_data/` | Rosbag preprocessing and label generation |
| `ros2_ws/src/deps/autoware_universe/planning/autoware_ml_planner/` | Autoware ROS 2 inference node |

The dataset tools and ROS 2 node share their C++ preprocessing implementation so
that training and inference use the same model inputs.

## Setup

Python packages use a shared [uv](https://docs.astral.sh/uv/) workspace. From the
repository root, install them with:

```bash
uv sync
```

Dataset generation also requires a built ROS 2 workspace. Import the repositories
listed in `ros2_ws/diffusion_planner.repos`, build the workspace with `colcon`, and
source it before running ROS-dependent commands:

```bash
source ros2_ws/install/setup.bash
```

## Dataset

Generate one `frames.h5` shard per rosbag and a Parquet frame index:

```bash
source ros2_ws/install/setup.bash
uv run python scripts/dataset/create_h5_dataset.py \
  root=/data/rosbags \
  output_root=/data/diffusion_planner_h5 \
  split=train
```

Configuration is defined in `configs/dataset/create_h5_dataset.yaml`. Completed H5
shards are reused when an interrupted index build is resumed.

Inspect loading performance with:

```bash
uv run --package diffusion-planner python scripts/dataset/check_dataset.py \
  /data/diffusion_planner_h5/indexes/train.parquet \
  --jobs 32
```

For tensor inspection and frame visualization, start the dashboard and select an
H5 shard or Parquet index from the sidebar:

```bash
uv run diffusion-planner-dashboard
```

The dataset format is documented in `docs/h5_dataset_schema.md`.

## Training

Training is configured with Hydra under `configs/train/` and logs runs to Weights &
Biases.

```bash
source ros2_ws/install/setup.bash
uv run --package diffusion-planner python scripts/train/train.py \
  experiment_name=my_experiment \
  dataloader.dataset.parquet_path=/data/diffusion_planner_h5/indexes/train.parquet
```

Checkpoints are written to `checkpoints/` by default. Dataset tensors already contain
the preprocessed map, agent, route, and vehicle-shape features; training does not read
rosbags directly.

## ONNX and Autoware

Export a checkpoint for runtime inference with:

```bash
uv run --package diffusion-planner python scripts/export/export_onnx.py \
  checkpoints/<checkpoint> \
  --parquet-path /data/diffusion_planner_h5/indexes/train.parquet
```

The exporter validates the generated ONNX models with ONNX Runtime. For ROS 2 node
parameters, topics, model compatibility, and launch instructions, see
`ros2_ws/src/deps/autoware_universe/planning/autoware_ml_planner/README.md`.
