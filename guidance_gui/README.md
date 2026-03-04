# Guidance GUI

Interactive Gradio web app for exploring guidance functions on the Diffusion-Planner model.
Unlike the DPO annotation GUI (which generates preference pairs for training), this tool is
purely for exploration: generate N independent trajectory samples under configurable noise
levels and guidance functions, and visualize them side-by-side.

## Prerequisites

- A trained model checkpoint (`.pth` file)
- A dataset NPZ list JSON (path list produced by `create_train_set_path.py`)
- A prototypes file (see **Prototypes** below)

## Launch

```bash
source .venv/bin/activate
python guidance_gui/app.py \
  --model_path /path/to/model.pth \
  --npz_list   /path/to/train_or_valid.json \
  --prototypes guidance_gui/prototypes_k16.npy
```

The app launches on port 7860 and opens a browser tab automatically.

## Prototypes

`prototypes_k16.npy` is a KMeans clustering of ego future trajectories from the training set.
It captures the dataset's distribution of motion patterns (straight, left turn, right turn, etc.)
and is used by the **Anchor Following** guidance function to steer the model toward a chosen
prototype shape.

The committed `prototypes_k16.npy` was generated from the Shinagawa-Odaiba training dataset.
If you use a different dataset, regenerate it:

```bash
python guidance_gui/scripts/generate_prototypes.py \
  --npz_list /path/to/train.json \
  --k 16 \
  --output guidance_gui/prototypes_k16.npy
```

Both this app and the DPO annotation GUI load the same prototypes file for the Anchor Following
guidance function.

## Guidance Functions

| Function | Description |
|---|---|
| Collision | Penalizes predicted trajectories that overlap with neighbor agents |
| Route Following | Rewards staying close to the route lanes |
| Lane Keeping | Rewards remaining within a drivable lane |
| Centerline Following | Rewards tracking the lane centerline |
| Anchor Following | Steers the trajectory toward a selected prototype shape |

All functions are active during DPM-Solver sampling and can be individually enabled, disabled,
and scaled via the UI sliders.
