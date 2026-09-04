# README

## 1. Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management with a workspace structure.

```bash
# Sync workspace and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Install git hooks
uv run pre-commit install

# check torch
python3 -c "import torch; print(torch.cuda.is_available())"
```

Run all configured hooks manually with:

```bash
uv run pre-commit run --all-files
```

## Autoresearch Control Panel

The autoresearch workflows can be launched from a unified Gradio control panel:

```bash
source .venv/bin/activate
python -m control_panel
```

See `control_panel/README.md` for the workspace layout, asset registry, training/eval tabs,
PRiSM flow, Perception Reproducer route mining, rendering, and Scene Editor integration.

## 2. Create dataset

### 2.1. Prepare rosbags

We assume the following directory structure:

```bash
driving_dataset$ tree . -L 2
.
├── bag
│   ├── 2024-07-18
│   │ ├── 10-05-28
│   │ ├── 10-05-51
│   │ ├── ...
│   │ ├── 16-10-07
│   │ └── 16-27-15
│   ├── 2024-12-11
│   ├── 2025-01-24
│   ├── 2025-02-04
│   ├── 2025-03-25
│   └── 2025-04-16
└── map
     ├── 2024-07-18
     │   ├── lanelet2_map.osm
     │   ├── pointcloud_map_metadata.yaml
     │   ├── pointcloud_map.pcd
     │   └── stop_points.csv
     ├── 2024-12-11
     ├── 2025-01-24
     ├── 2025-02-04
     ├── 2025-03-25
     └── 2025-04-16
```

### 2.2. Convert to diffusion_planner's format (npz)

use `parse_rosbag_for_directory.py` directly.

```bash
python3 ./ros_scripts/parse_rosbag_for_directory.py <target_dir_list> --save_root <save_root> [--step <step>] [--limit <limit>]
```

### 2.3. Generate path_list.json

This script search `*.npz` files and create `path_list.json`.

```bash
python3 ./diffusion_planner/util_scripts/create_train_set_path.py <root_dir_list>
```

## 3. Train

Run (launches train_predictor.py across all visible GPUs):

```bash
cd ./diffusion_planner
python3 train_run.py \
    --exp_name <exp_name> \
    --train_set_list <train.json> \
    --valid_set_list <valid.json>
    # optional: --resume_model_path <.pth> --wandb_run_id <id> --wandb_project_name <name>
```

### 3.1. Data augmentation

`--augment_type` selects how the ego start state is perturbed during training. All
three augmenters keep the recorded future as the target, so the model is trained to
*recover* toward the drive that was recorded rather than only to copy it.

| value | what it does |
|---|---|
| `quintic` | default. Fixed-offset quintic bridge from a perturbed t=0 state. |
| `bridge` | extended bridge perturbation. |
| `frenet` | corridor-constrained lateral offset with feasibility filtering, an exact footprint check, and a kinematic rewrite of the ego history. |

Turn augmentation off entirely with `--use_data_augment False`.

#### Ego history perturbation (all augmenters)

```bash
--ego_past_noise_std 0.1        # default; 0 disables
```

One factor per augmented scene, drawn from `N(1, std)` and clamped to `±2·std`,
scaling the ego history about the t=0 sample. The recorded *shape* is preserved and
the spacing changes, so it perturbs the implied speed history rather than adding
per-point noise. `quintic` scales the recorded history; `frenet` scales the history
it rewrote. t=0 itself never moves.

#### Frenet-only options

Every flag below is **off by default**; with all of them at their defaults the
augmenter reproduces the previous behaviour exactly.

```bash
--augment_type frenet \
  --frenet_recovery_rounds 1 \      # retry after a footprint veto
  --frenet_min_clearance 0.2 \      # metres of clearance to keep from recorded vehicles
  --frenet_toward_parked_prob 0.3 \ # fraction of eligible scenes nudged toward a parked vehicle
  --frenet_hist_jitter_lat 0.3 \    # smooth lateral jitter of the history, metres at the oldest sample
  --frenet_hist_jitter_lon 0.3      # same, along the direction of travel
```

- **`--frenet_recovery_rounds N`** (default 0). A candidate whose footprint overlaps a
  recorded vehicle is vetoed, and the scene falls back to plain ground truth. Each
  round lets such a scene re-select. A retry burns the losing lateral *offset*, not
  the path shape, because every shape of one offset overlaps the same vehicle.
  `1` is the operating point; further rounds recover almost nothing, because what
  survives is blocked geometrically and re-rolling does not move geometry.

- **`--frenet_min_clearance C`** (default 0.0). Metres of exact footprint clearance
  every accepted candidate must keep from every recorded vehicle, on top of the
  corridor margin. Applies to the vehicle cut only — the road-edge margin is
  unaffected. At `0.0` the check is overlap-only, which is the historical behaviour.

- **`--frenet_toward_parked_prob P`** (default 0.0). The corridor is symmetric, so a
  scene that passes a parked vehicle is as likely to be nudged away from it as toward
  it. This directs that fraction of eligible scenes toward the vehicle and takes the
  largest feasible offset, making the t=0 state a harder avoidance than the recorded
  one, while restricting the merge to horizons that rejoin the recording *before* the
  vehicle. A scene is eligible only when a parked vehicle bounds the corridor and is
  still ahead at t=0; if it is already alongside there is no avoidance left to harden.

- **`--frenet_hist_jitter_lat L` / `--frenet_hist_jitter_lon G`** (default 0.0).
  Smooth per-sample jitter of the ego history, in metres of standard deviation at the
  oldest sample, perpendicular to and along the direction of travel. The two axes are
  drawn independently. Unlike `--ego_past_noise_std`, which can only make a
  correctly-shaped history be traversed at the wrong speed, this makes the history
  itself imperfect. Applied after the footprint check and with t=0 pinned, so the
  target and the state the model plans from are unchanged.

Frenet also exposes the sampling grid itself — `--frenet_n_draws`, `--frenet_dy_max`,
`--frenet_dth_max`, `--frenet_merge_times`, `--frenet_anchors`, `--frenet_acc0_fracs`,
`--frenet_ranked_temp_s`, `--frenet_seed`. Their defaults are the measured
configuration; see `diffusion_planner/utils/data_augmentation_frenet.py`.

Use `--seed` to vary model init, data order and augmentation draws. Training is
otherwise deterministic: two runs at the same seed produce identical weights, so a
same-seed rerun cannot serve as a control when measuring a recipe change.
