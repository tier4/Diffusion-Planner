# New DP native-H5 evaluation

This directory is the compatibility boundary for evaluating the new DP ONNX model in
the old repository. It does **not** convert old NPZ features into new model inputs.
Only H5 files produced by the new DP preprocessing pipeline (format version 4) are read.

Open-loop uses the old repository's unchanged `planner_metrics` scorers and writes the
same `summary.json` and `details/<metric>/details.jsonl` hierarchy:

```bash
PYTHONPATH=.:diffusion_planner ../new-DP/.venv/bin/python -m new_dp_h5_eval.open_loop \
  ../data/new_DP_dataset/open_loop_native_h5/matrix.json \
  ../data/new_DP_dataset/open_loop_native_h5/index.parquet \
  ../data/hisaki_new_DP_model/diffusion_planner_sampler.onnx \
  ../data/new_DP_dataset/open_loop_native_h5/old_metric_result
```

The matrix paths are joined strictly through the Parquet `source_npz_path` column. A
missing or duplicate mapping aborts before inference, preventing silent frame mismatch.

## Preparing native H5 from the existing rosbags

The old NPZ is used only as a timestamp/matrix selector. The tensors are regenerated
from rosbag + Lanelet2 map by the new-DP converter. The ROS 2 workspace and the
`ml_planner_data` extension must be built first (`ros2_ws/install` is required).

For the 20260814 basic matrix, use the original labelled-bag root and the original
dataset-root marker:

```bash
cd /path/to/new-DP
source ros2_ws/install/setup.bash
uv run --package diffusion-planner python \
  ../Diffusion-Planner/packages/diffusion_planner/dataset/convert_matrix_rosbag_to_h5.py \
  /path/to/basic_matrix.json \
  /mnt/storage_rdma/diffusion_planner/rosbags_from_label \
  /mnt/storage_rdma/workspaces/kem/Diffusion-Planner-Meta-Repository/data/new_DP_dataset/basic_native_h5 \
  --dataset-root-name 20260814_basic_dataset
```

For the evaluator override matrix, the source bags are under the dedicated
`source_rosbags/x2_dev` root and the NPZ hierarchy marker is `dataset_all`:

```bash
uv run --package diffusion-planner python \
  ../Diffusion-Planner/packages/diffusion_planner/dataset/convert_matrix_rosbag_to_h5.py \
  /path/to/override_matrix.json \
  /mnt/storage_rdma/diffusion_planner/dataset/evaluator_override_dataset/source_rosbags/x2_dev \
  /mnt/storage_rdma/workspaces/kem/Diffusion-Planner-Meta-Repository/data/new_DP_dataset/override_native_h5 \
  --dataset-root-name dataset_all
```

The converter checks the sibling `.json`, rosbag metadata, and map before launching
workers. It is therefore safe to run on a filtered matrix first. Do not point it at
`dataset_all`'s generated NPZ directory as a rosbag root: that would merely preserve
the old representation and is not a new-DP H5 conversion.

For a full site route set (where bags are organized as `train`/`valid`/`auto`), use
the new-DP `scripts/dataset/create_h5_dataset.py` with root
`/mnt/storage_rdma/diffusion_planner/rosbags_from_label`, the project vehicle config,
and `split=valid` (or the split containing the route). This produces one native H5
shard per bag and a Parquet index; then pass that index to the evaluator above.

Closed-loop keeps the existing reproducer, per-step scorers, `segments.jsonl`, and
`summary.json` aggregation. The old route list is used only to establish timeline order
and load recorded world-pose/UUID sidecars; scene/model data always comes from native H5:

```bash
PYTHONPATH=.:diffusion_planner ../new-DP/.venv/bin/python -m new_dp_h5_eval.closed_loop \
  /path/to/native/index.parquet \
  ../data/hisaki_new_DP_model/diffusion_planner_sampler.onnx \
  /path/to/path_list_closed_loop_by_site.json \
  /path/to/output
```

Coordinate recentering handles pose tensors, lane boundary offsets, intersection areas,
stop lines, and road borders according to the new schema. `--gpu_transform` is deliberately
not exposed: the old GPU transform understands the old packed schema and is rejected for
native H5 instead of silently producing wrong coordinates.
