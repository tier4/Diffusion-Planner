# Training from the new-architecture H5 dataset

`tier4-main` trains from one NPZ per frame addressed by a JSON path list. The
`new-architecture/main` branch replaced that with one `frames.h5` shard per rosbag plus a
split-level Parquet frame index. This page describes the option that lets `tier4-main` train
from those H5 shards directly.

## Usage

```bash
python diffusion_planner/train_run.py \
  --exp_name h5_run \
  --h5_train_index /mnt/nvme/dataset/hdf5_dataset/indexes/train.parquet \
  --h5_valid_index /mnt/nvme/dataset/hdf5_dataset/indexes/valid.parquet \
  --h5_converter_param_path dataset/data_converter_param.json
```

| Option | Meaning |
|---|---|
| `--h5_train_index` | Parquet frame index of the training split. Replaces `--train_set_list`. |
| `--h5_valid_index` | Parquet frame index of the validation split. Replaces `--valid_set_list`. |
| `--h5_file_capacity` | Open H5 shards cached per DataLoader worker (default 8). |
| `--h5_converter_param_path` | Data-converter parameter JSON supplying the true `ego_wheel_base` per project id. Optional; see *Ego shape* below. |

The two splits are chosen independently, because both sources hand over the same canonical
model inputs. When neither index is set, training reads NPZ path lists exactly as before, so
existing commands are unaffected.

Training on H5 while validating on the existing NPZ split is the useful mixed case: the
validation split is what the closed-loop, open-loop and replan-consistency evaluations are
built around, and keeping it on NPZ keeps all three available and comparable with earlier runs.

```bash
python diffusion_planner/train_run.py \
  --exp_name h5_train_npz_valid \
  --h5_train_index /mnt/nvme/dataset/hdf5_dataset/indexes/train.parquet \
  --h5_converter_param_path dataset/data_converter_param.json \
  --valid_set_list .../path_list_valid_sft_balanced_every_100.json
```

`--train_subsample_step` works in both modes. The W&B dataset artifact records the Parquet
index in place of the NPZ path list, keeping lineage intact.

## Why the network is unchanged

The H5 schema was built against the same scene dimensions this branch's model already uses:

| | tier4-main | H5 (format version 4) |
|---|---|---|
| Past steps | `INPUT_T + 1` = 31 | 31 |
| Future steps | `OUTPUT_T` = 80 | 80 |
| Neighbours | `MAX_NUM_NEIGHBORS` = 320 | 320 |
| Lanes × points | 140 × 20 | 140 × 20 |
| Route lanes × points | 25 × 20 | 25 × 20 |
| Polygons × points | 10 × 40 | 10 × 40 (`intersection_area`) |
| Line strings × points | 60 × 20 | 30 stop lines + 30 road borders × 20 |

What differs is how the tensors are *factored*, not how big the scene is. So the H5 frame is
re-assembled into the canonical NPZ layout in
`diffusion_planner/utils/h5_dataset.py`, and nothing downstream of the dataset changes: the
encoder, the decoder, the augmenter, the ONNX export, the ROS 2 node and existing checkpoints
all stay as they are. Teaching the encoder a second input contract would have forked all of
them for no gain.

## Field mapping

| Canonical input | Source in H5 |
|---|---|
| `ego_agent_past` (31, 4) | `ego_agent_past[:, :4]` — already cos/sin |
| `ego_current_state` (10) | last `ego_agent_past` step: pose, `velocity` → vx, `yaw_rate` |
| `neighbor_agents_past` (320, 31, 11) | `neighbor_agents_past` + `agent_shape` (width, length) + `agent_label` (one-hot) |
| `static_objects` (5, 10) | no H5 source; all-zero, fully masked |
| `lanes` / `route_lanes` (·, 20, 33) | `lanes` geometry + centreline forward difference + `lane_types` + `lane_traffic_light_past[:, -1]` |
| `lanes_speed_limit` / `..._has_speed_limit` | `lanes_speed_limit`; the flag is `> 0` |
| `polygons` (10, 40, 3) | `intersection_area` + type flag |
| `line_strings` (60, 20, 4) | `stop_lines` (flag col 2) then `road_borders` (flag col 3) |
| `goal_pose` (4) | `goal_pose` |
| `ego_shape` (3) | `ego_shape`; see below |
| `turn_indicators` (31) | `turn_indicators` |
| `ego_agent_future` (80, 3) | `ego_agent_future[:, :4]` → (x, y, yaw) |
| `neighbor_agents_future` (320, 80, 4) | `neighbor_agents_future` |

### Layouts that are not a straight copy

**Ego future is 3-wide, neighbour future is 4-wide.** `StatePerturbation`'s quintic refinement
reads column 2 of the ego future as a raw heading angle, so cos/sin is converted back to a yaw
there. The neighbour future is fixed at cos/sin by the canonical contract and passes through.

**Lane attributes are per point here, per segment in H5.** `LaneEncoder` reads them from point
0 of each segment, so the segment's boundary types and light are broadcast over its points --
but only over *valid* points. Writing an attribute onto a padded point would break the
all-zero padding convention, and `ObservationNormalizer` keys padding on the whole row: a row
that is not exactly zero gets mean-shifted, after which the encoder's own validity test (on
columns 0..7) reads the padding as a real lane.

**Traffic lights lose two distinctions.** H5 encodes
`[green, amber, red, unknown_or_unavailable, white_or_no_light, arrow_flag]` per step; this
branch has `[green, yellow, red, white, no_traffic_light]` and one slot for the current state.
So the last history entry is used, both H5 "unknown" and "white or no light" map to
`no_traffic_light`, the `white` slot stays zero, and the arrow flag is dropped.

**Neighbour velocities are absent.** H5 has no neighbour vx/vy. This costs nothing, because
`NeighborEncoder.forward` zeroes those two channels before encoding regardless of what is fed
in.

**Ego lateral state is absent.** H5 has no ego vy, ax, ay or steering angle. The augmenter
re-derives the steering angle from the yaw rate and the wheelbase, and `compute_training_loss`
only reads the pose and the longitudinal velocity, so the zeros are only visible to
`ego_current_state`'s own encoder input.

### Ego shape

H5 stores `[base_link_to_front, length, width]`; this branch stores `[wheelbase, length,
width]`. Slot 0 has two genuine consumers:

- `StatePerturbation.augment` turns a yaw rate into a steering angle with it (a bicycle model,
  so it wants the real wheelbase);
- `compute_ego_bbox_corners` shifts the box centre forward by half of it.

Without `--h5_converter_param_path` the value is approximated as
`2 × (base_link_to_front − length / 2)`, which makes the bounding-box shift exact and leaves
only the augmenter's bicycle model on an estimate. Passing `dataset/data_converter_param.json`
from the meta repository supplies the real per-project `ego_wheel_base`, resolved through each
shard's `project_id` attribute; that removes the estimate and is the recommended setting.

## Limits

- `--enable_replan_consistency_eval` is rejected when `--h5_valid_index` is set. The pair
  dataset finds replan pairs by walking consecutive NPZ frame *paths*; H5 frames are addressed
  by (shard, frame index) and carry no equivalent ordering. Only the validation split matters,
  so an H5 training split with an NPZ validation split keeps this evaluation.
- `static_objects` is always empty, so any behaviour that depended on static obstacles is not
  learnable from an H5 run.
- The H5 frame stride is a generation-time setting (`frame_interval`, 0.5 s by default) and is
  coarser than the NPZ pipeline's. It changes how many frames a rosbag yields, not their
  contents.

## Tests

`diffusion_planner/tests/test_h5_dataset.py` builds synthetic H5 shards and checks the
conversion against the canonical shape contract, the padding invariant, the traffic-light
collapse, the wheelbase handling, and index/subsampling behaviour. Two of them are end-to-end:
one runs a converted batch through the augmenter, the normalizer and `compute_training_loss`
and asserts gradients arrive, and one drives the ego onto a converted road border and
neighbour box to confirm the auxiliary losses really read the columns this converter fills.
