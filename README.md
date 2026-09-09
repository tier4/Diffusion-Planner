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

#### Ego history perturbation (`quintic` and `frenet`)

```bash
--ego_past_noise_std 0.2        # unset = each augmenter's own default; 0 disables
```

One factor per augmented scene, drawn from `N(1, std)` and clamped to `±2·std`,
scaling the ego history about the t=0 sample. The recorded *shape* is preserved and
the spacing changes, so it perturbs the implied speed history rather than adding
per-point noise. t=0 itself never moves.

Left unset it resolves per augmenter, because they do not perturb the same thing:

| augmenter | default | what the factor scales |
|---|---|---|
| `quintic` | 0.1 | the RECORDED history, and the current velocity and acceleration with it |
| `frenet` | **0.0** (off) | the history it rewrote from the perturbed polyline; `ego_current_state` is left bit-identical |
| `bridge` | n/a | no history perturbation; passing the flag is an error, not a silent no-op |

**Why frenet defaults to off while quintic does not.** Quintic scales a history the
recorder actually observed; frenet has already rewritten the past kinematically from the
perturbed polyline, so the same factor would perturb a history it synthesised. That is
the reason `tier4-main` hard-passes `0.0` there, and this branch keeps it — a stock
`--augment_type frenet` run trains exactly what it trains on main.

Measured either way, so the `0.0` is a decision and not an omission: on a 2-seed, 8-arm
A/B (~1,113 perturbed closed-loop rollouts per arm, both trackers, replan intervals 1
and 3) enabling it moved recovery 28.3% → 30.2% at replan 1 and 52.7% → 52.2% at replan
3, with `lost%` 20.3 → 19.8 and 7.7 → 8.0 — every difference inside the seed spread
(17.2 points on recovered% at replan 1, 1.0 on `lost%` at replan 3). Nothing is given up
by leaving it off, and `--ego_past_noise_std 0.1` turns it on for anyone revisiting it at
full dataset scale.

`--ego_past_noise_mode` selects **which** mechanism that std drives, for
`augment_type=frenet`. Exactly one runs — they are mutually exclusive:

| mode | what it does | units of the std | bounded? |
|---|---|---|---|
| `scale` (default) | one factor multiplies the whole rewritten history: the track keeps its shape and is traversed at the wrong speed | dimensionless (`N(1, std)`) | yes, ±2σ |
| `jitter` | a smooth lateral bend of the track, so its *shape* is wrong; three low-frequency modes, exactly zero at t=0 | **metres** at the oldest sample | **no** |

A mode rather than two magnitude flags, because two flags could both be set and the
combination is reachable by accident — quintic's scale default is non-zero, so asking
for jitter alone used to silently apply both, and that combination has never been
evaluated (every jitter arm of the A/B pinned the scale to 0). `jitter` is
implemented for frenet only; asking for it with another `augment_type` is an error.

Note the std means different things per mode. `0.1` is ±10% of a traversal speed in
`scale`, and 0.1 m of lateral displacement in `jitter`. The A/B tested **0.3 m** for
jitter, so that is the value to pass; and because the jitter is unclamped, 31.7% of
draws exceed the std and 4.5% exceed twice it (measured over 200k draws).

**In `jitter` mode the bent history is not re-checked against the footprint
constraint.** The corridor bounds certify the clean polyline, and the jitter is applied
afterwards on purpose, so that enabling it cannot change which scenes are accepted. The
consequence: on a drive that passed a parked vehicle closely, the resulting history can
place the ego footprint inside that vehicle **in the past**. The future, the training
target and t=0 are unaffected; what the encoder reads is. Keep the magnitude small
relative to the clearances in the data.

Passing a number applies it to whichever augmenter is selected, so the flag stays
sweepable. Note that a `quintic`-vs-`frenet` A/B on it is not measuring the same
perturbation — quintic also rescales the current velocity and acceleration.

#### Frenet-only options

Every flag below is off at its default, and so is `--ego_past_noise_std` for frenet, so a
stock run reproduces the previous behaviour exactly — measured through the path a training
takes (`build_parser` → `build_config` → `augmenter_from_args`): 432 arrays over
`ego_agent_past`, `ego_current_state` and `goal_pose`, three `augment_type` values, two
seeds, worst `|base − head|` of **0.000e+00** for all three.

```bash
--augment_type frenet \
  --frenet_recovery_rounds 1 \      # retry after a footprint veto
  --frenet_min_clearance 0.2 \      # metres of clearance to keep from recorded vehicles
  --frenet_toward_parked_prob 0.3 \ # fraction of eligible scenes nudged toward a parked vehicle
  --ego_past_noise_mode jitter \   # bend the history instead of stretching it (frenet only)
  --ego_past_noise_std 0.3          # metres at the oldest sample, in jitter mode
```

- **`--frenet_recovery_rounds N`** (default 0). A candidate whose footprint overlaps a
  recorded vehicle is vetoed, and the scene falls back to plain ground truth. Each
  round lets such a scene re-select. A retry burns the losing lateral *offset*, not
  the path shape, because every shape of one offset overlaps the same vehicle.
  `1` is the operating point; further rounds recover almost nothing, because what
  survives is blocked geometrically and re-rolling does not move geometry.

- **`--frenet_min_clearance C`** (default 0.0). Metres of exact footprint clearance
  required from every recorded vehicle. It acts in **two** places, and they are not
  windowed the same way — worth knowing before choosing a value:

  | where | scope | effect |
  |---|---|---|
  | corridor half-width | **whole merge window** | the neighbour cut uses `max(0.10, C)`, so a `C` above the 0.10 corridor margin narrows the band a candidate may occupy at every masked timestep |
  | exact footprint veto | **only the timesteps the perturbation moved the ego** | a candidate can be accepted while passing closer than `C` at a timestep where it is bit-identical to ground truth |

  The veto's windowing exists because a candidate coincides with the recorded drive
  outside its merge window, so vetoing there would reject scenes for the *recording's*
  own clearance. The corridor cut is **not** windowed, so a large `C` can still drop a
  tight-squeeze scene from augmentation via the corridor rather than the veto. Keep `C`
  modest for that reason, and note `C ≤ 0.10` leaves the corridor untouched entirely and
  only tightens the veto. True overlap is rejected everywhere regardless. Applies to the
  vehicle cut only; the road-edge margin is unaffected. At `0.0` the check is
  overlap-only, which is the historical behaviour.

- **`--frenet_toward_parked_prob P`** (default 0.0). The corridor is symmetric, so a
  scene that passes a parked vehicle is as likely to be nudged away from it as toward
  it. `P` directs a fraction of scenes toward the vehicle instead.

  Precisely, `P` is an independent per-scene coin ANDed with two other conditions —
  `toward = eligible & (r < P) & do_aug` — so it is the fraction of scenes that are
  **both** eligible **and** already selected for augmentation by `--augment_prob`. At
  `P = 1.0` every such scene is nudged, with no further filtering. At the default
  `augment_prob 0.5` the realised rate is therefore about half of `P × eligible`.

  A scene is eligible only when a *parked* vehicle (not a kerb, not a moving car) bounds
  the corridor within `--frenet_dy_max` reach **and** is reached at or after the shortest
  merge horizon; if it is already alongside at t=0 there is no avoidance left to harden.
  Eligibility is uncommon: **10,958 of 105,093 scenes (10.4%)** on a full-size corpus, and
  **zero** on the 2,297-scene pipeline-test set, which contains no parked vehicle bounding
  the corridor ahead — so on that set the flag does nothing at any `P`.

  On a row where it can, the augmenter then takes the *largest* feasible offset and
  restricts the merge to horizons that rejoin the recording **before** the vehicle,
  which is what makes the t=0 state a harder avoidance than the recorded one. Neither
  is unconditional: on a tight pass that merge restriction can strike out every
  horizon, and rather than drop the scene from augmentation the row keeps its ungated
  candidates and takes the ordinary first-feasible offset. Such a row is still nudged
  toward the vehicle but is **not** guaranteed to rejoin in front of it, so the
  proportion of genuinely hardened scenes is lower than `P` suggests.

  Enabling the flag costs augmented rows, and the aggregate hides how uneven that cost
  is. On a 105k-scene set at `P = 1.0` the count goes 10,330 → 9,742 (−6%). But on the
  tight passes the flag exists to serve it is far more expensive: pointing every offset
  at the vehicle means most candidates are then rejected by the exact footprint check.
  Measured on one stationary vehicle 25 m ahead at 1.8 m lateral, 128 draws — 91 rows
  accepted with the flag off, 8 with it on. The fallback above is what keeps that from
  being 0; it does not recover the rest, and no selection rule can, because a 2.0 m ego
  does not fit a 1.8 m gap. Use `P` as a small fraction, not a global setting.

Frenet also exposes the sampling grid itself — `--frenet_n_draws`, `--frenet_dy_max`,
`--frenet_dth_max`, `--frenet_merge_times`, `--frenet_anchors`, `--frenet_acc0_fracs`,
`--frenet_ranked_temp_s`, `--frenet_seed`. Their defaults are the measured
configuration; see `diffusion_planner/utils/data_augmentation_frenet.py`.

Use `--seed` to vary model init, data order and augmentation draws. Training is
otherwise deterministic: two runs at the same seed produce identical weights, so a
same-seed rerun cannot serve as a control when measuring a recipe change.
