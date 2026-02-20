# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diffusion-Planner is a diffusion-model-based trajectory planner for autonomous driving, integrated with Autoware/ROS 2. The core model uses a Diffusion Transformer (DiT) encoder-decoder with SDE/ODE-based sampling and optional guidance. There is also a Direct Preference Optimization (DPO) fine-tuning pipeline for preference-based learning on top of the base model.

## Repository Structure

```
diffusion_planner/       # Core ML package (training, model, utilities)
diffusion_planner_ros/   # ROS 2 node wrapping the model for Autoware
preference_optimization/ # DPO fine-tuning pipeline
lichtblick_extensions/   # Lichtblick visualization UI extensions
ros_scripts/             # Data pipeline scripts (rosbag parsing, ONNX export)
test_scripts/            # Misc test/validation scripts
```

## Common Commands

### Linting
```bash
ruff check .          # Check linting (import sorting only, configured in pyproject.toml)
ruff check . --fix    # Auto-fix import ordering
```

### Running Tests
```bash
python3 preference_optimization/test_fde_standalone.py
```

### Training

**Base model (multi-GPU, uses torchrun):**
```bash
cd diffusion_planner
./train_run.sh <exp_name>          # 8-GPU distributed training
./valid_run.sh <model_dir> <valid_set_list>
```

**DPO fine-tuning:**
```bash
python preference_optimization/train_dpo.py \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode [gui|rule] \
  --exp_name <name>
```

### Data Pipeline
```bash
# 1. Parse rosbags → NPZ
python ros_scripts/parse_rosbag_for_directory.py <dirs> --save_root <output>

# 2. Create train/valid list JSON files
python diffusion_planner/util_scripts/create_train_set_path.py <root_dirs>

# 3. Export PyTorch → ONNX (for ROS deployment)
python ros_scripts/torch2onnx.py <model_dir>
```

### ROS 2 Node
```bash
ros2 launch diffusion_planner_ros diffusion_planner.launch.xml
```

## Architecture

### Core Model (`diffusion_planner/diffusion_planner/`)

- **`model/diffusion_planner.py`** – Top-level `Diffusion_Planner` class; composes encoder and decoder.
- **`model/module/encoder.py`** – Encodes ego history, agent states, and map context into a latent.
- **`model/module/decoder.py`** – Diffusion Transformer decoder that generates future trajectories.
- **`model/module/dit.py`** – DiT (Diffusion Transformer) backbone.
- **`model/diffusion_utils/sde.py`** – SDE process for diffusion (forward/reverse).
- **`model/diffusion_utils/dpm_solver_pytorch.py`** – DPM-Solver for fast sampling.
- **`model/flow_matching_utils/ode_solver.py`** – ODE solver for flow matching variant.
- **`model/guidance/`** – Classifier-free guidance: `collision.py`, `route_following.py`, `guidance_wrapper.py`.
- **`dimensions.py`** – Centralized data shape constants; read this before touching tensor shapes.
- **`loss.py`** – Training loss functions.
- **`utils/normalizer.py`** – State normalization using `normalization.json`.
- **`utils/config.py`** – `Config` object loaded from a JSON file; used everywhere.
- **`utils/dataset.py`** – PyTorch `Dataset` reading `.npz` files.

Training entry points: `train_predictor.py` (calls `train_epoch.py`) and `valid_predictor.py`.

### DPO Pipeline (`preference_optimization/`)

- **`train_dpo.py`** – Entry point; sets up data, model, and calls `DPOTrainer`.
- **`trainer.py`** – `DPOTrainer`: training loop, evaluation, checkpoint saving.
- **`dpo_loss.py`** – DPO loss (Bradley-Terry preference model).
- **`datasets.py`** – `DPODataset` (preference pairs) and `NPZDataset` (raw observations).
- **`utils.py`** – `calculate_fde()`, `generate_trajectory_pair_with_retry()`, `generate_rule_based_preferences()`.
- **`preference_collection.py`** – `PreferenceAnnotator` state machine for GUI/rule-based collection.
- **`annotation_ws_server.py`** – WebSocket server for Lichtblick-based annotation UI.

Preference modes: `gui` (Gradio web UI for human annotation) or `rule` (automatic FDE/path-length criterion).

### ROS 2 Node (`diffusion_planner_ros/`)

- **`diffusion_planner_node.py`** – Subscribes to perception, localization, routing, and traffic light topics; runs ONNX inference; publishes `CandidateTrajectories`.
- **`utils.py`** – ROS message ↔ numpy conversions.
- **`conversion/`** – Fine-grained message type converters.
- **`lanelet2_utils/lanelet_converter.py`** – Converts Lanelet2 map primitives to model input tensors.

The ROS node uses the ONNX export of the model (not PyTorch directly) for inference.

## Key Data Format

Training data is stored as `.npz` files (one per scene) containing ego trajectory, agent states, and map features. Lists of `.npz` paths are stored as JSON files passed to `--train_npz_list` / `--valid_npz_list`. Normalization statistics are in `normalization.json` at the repo root and also inside `diffusion_planner/`.

## Package Installation

```bash
# Core ML package
pip install -e diffusion_planner/

# ROS package (requires ROS 2 Humble + Autoware)
cd diffusion_planner_ros && colcon build
```

Dependencies live in `diffusion_planner/requirements.txt`. The NuPlan devkit has its own pinned list in `diffusion_planner/requirements_nuplan-devkit_fixed.txt`.
