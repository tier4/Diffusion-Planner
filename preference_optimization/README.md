# Preference Optimization for Diffusion Planner

Direct Preference Optimization (DPO) pipeline for fine-tuning the Diffusion Planner model. Supports full fine-tuning and parameter-efficient LoRA training.

## Quick Start

```bash
# Install dependencies
pip install gradio>=4.0.0 peft

# Train with LoRA + GUI annotation (recommended)
python3 -m preference_optimization.train_dpo \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode gui \
  --train_epochs 15 \
  --use_lora \
  --learning_rate 5e-4

# Merge LoRA adapter into a single .pth for deployment
python3 -m preference_optimization.merge_lora \
  --model_path <experiment_dir>/latest.pth \
  --output <experiment_dir>/merged.pth

# Export to ONNX
python3 ros_scripts/torch2onnx.py <dir_containing_merged.pth>
```

### Full fine-tuning (no LoRA)

```bash
python3 -m preference_optimization.train_dpo \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode rule \
  --train_epochs 10
```

## Lichtblick Launch

1. Install the lichtblick extension with `cd lichtblick_extensions/annotation_ui/; npm run install-local`
2. Open three terminals, run `source ~/pilot-auto/install/setup.bash` to collect pilot-auto (or projects) message paths.
3. On each terminals


```bash
## Terminal 1: Launch Lichtblick
lichtblick  ## Select layout from `lichtblick_extensions/annotation_ui/layouts/dpo.json`


## Terminal 2: Launch foxglove_bridge
ros2 launch foxglove_bridge foxglove_bridge_launch.xml

## Terminal 3: Launch DPO train
# Train with GUI annotation
python train_dpo.py \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode lichtblick \
  --train_epochs 10 \
  --exp_name my_experiment

```


## LoRA Training

### Overview

LoRA (Low-Rank Adaptation) freezes the base model and trains small adapter matrices
on the DiT decoder's attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`).
Only ~1.3% of parameters are trainable, which prevents catastrophic forgetting, requires
less GPU memory, and produces small checkpoint files (adapter weights only).

### How it works

At LoRA init time, `apply_lora()` in `lora_utils.py` replaces each `nn.MultiheadAttention`
in the DiT decoder blocks with `UnfusedMHA` -- a numerically identical module that exposes
separate `nn.Linear` sub-layers. PEFT then applies its standard `LinearLoRA` to each
projection. The encoder and all other model components remain frozen.

### LoRA CLI flags

| Flag | Default | Description |
|---|---|---|
| `--use_lora` | `False` | Enable LoRA (otherwise full fine-tuning) |
| `--lora_rank` | `16` | LoRA rank `r`. Lower = less capacity, less forgetting |
| `--lora_alpha` | `16` | Scaling factor. Effective delta = `alpha/r * B @ A` |
| `--lora_dropout` | `0.05` | Dropout on LoRA activations |
| `--learning_rate` | `1e-5` | Recommended: `5e-4` for LoRA, `1e-5` for full FT |

### LoRA checkpoint layout

```
<experiment_dir>/
  latest.pth              # Base model (frozen, copied at start)
  args.json               # Model config
  dpo_args.json           # Training args
  lora_epoch_001/         # Adapter weights + optimizer state
    adapter_config.json
    adapter_model.safetensors
    optimizer.pth
  lora_epoch_002/
    ...
  lora_latest -> lora_epoch_002   # Symlink to most recent
```

### Resuming LoRA training

Point `--model_path` at a previous experiment's `latest.pth`. The script auto-detects
`lora_latest/` next to the `.pth` and resumes from that adapter, including optimizer
state (AdamW moments):

```bash
python3 -m preference_optimization.train_dpo \
    --model_path <prev_experiment>/latest.pth \
    --train_npz_list <train.json> \
    --valid_npz_list <valid.json> \
    --preference_mode gui \
    --train_epochs 10 \
    --use_lora
```

### Merging LoRA for deployment

Use `merge_lora.py` to bake the LoRA deltas into the base weights and produce a single
`.pth` file. The merged checkpoint has no PEFT dependency and can be used directly with
`torch2onnx.py`:

```bash
# Auto-detect lora_latest/ next to the .pth
python3 -m preference_optimization.merge_lora \
    --model_path <experiment_dir>/latest.pth \
    --output <experiment_dir>/merged.pth

# Or specify a specific adapter
python3 -m preference_optimization.merge_lora \
    --model_path <experiment_dir>/latest.pth \
    --lora_dir <experiment_dir>/lora_epoch_005 \
    --output <experiment_dir>/merged_epoch5.pth

# Then export to ONNX
python3 ros_scripts/torch2onnx.py <experiment_dir>
```

## Architecture

### Module Structure

```
preference_optimization/
├── train_dpo.py              # Main entry point, CLI args, training loop
├── trainer.py                # DPOTrainer class, checkpoints, metrics
├── dpo_loss.py               # DPO loss (shared-model or separate reference)
├── lora_utils.py             # UnfusedMHA, apply/save/load LoRA, merge
├── merge_lora.py             # Standalone: merge LoRA adapter into single .pth
├── model_utils.py            # load_model() from checkpoint + args.json
├── utils.py                  # Trajectory generation, FDE, pair sampling
├── preference_collection.py  # Rule-based preference generation
├── annotation_gui.py         # Gradio web UI for human annotation
├── visualization.py          # Validation epoch visualizations
├── datasets.py               # DPO and NPZ dataset classes
└── test_fde_standalone.py    # Unit tests
```

### Design Principles

1. **Single Responsibility** - Each module has one clear purpose
2. **Dependency Injection** - Easy testing and flexibility
3. **Type Safety** - Full type hints for better IDE support
4. **Documentation** - Comprehensive docstrings
5. **Testability** - Clean interfaces and minimal coupling

## Module Overview

### `train_dpo.py`
Main entry point. Orchestrates:
- Argument parsing
- Experiment setup
- Training loop coordination

### `trainer.py`
**DPOTrainer Class** - Manages training:
- Epoch execution
- Checkpoint saving
- Metric logging
- Validation visualization

### `dpo_loss.py`
DPO loss computation:
- `compute_trajectory_loss()` - MSE loss for a trajectory
- `compute_dpo_loss()` - DPO loss for preference pairs

### `model_utils.py`
Model management:
- `load_model()` - Load checkpoint and config
- Handles different checkpoint formats

### `preference_collection.py`
Preference generation:
- `generate_rule_based_preferences()` - Automatic annotation
- Uses path length as criterion

### `annotation_gui.py`
Web-based annotation interface:
- **PreferenceAnnotator** class - State management
- Gradio UI with parameter controls
- Real-time FDE display

### `visualization.py`
Validation visualization:
- `visualize_validation()` - Generate prediction plots
- Saves to `validation_vis/` directory

### `datasets.py`
PyTorch datasets:
- **DPODataset** - Preference pairs for training
- **NPZDataset** - Observation data for validation

### `utils.py`
Core utilities:
- `load_npz_data()` - Load observation files
- `calculate_fde()` - Final Displacement Error
- `generate_trajectory_pair()` - Generate diverse pairs

## Usage Examples

### Basic Training

```bash
python train_dpo.py \
  --model_path checkpoints/model.pth \
  --train_npz_list data/train.json \
  --valid_npz_list data/valid.json \
  --preference_mode gui
```

### Advanced Configuration

```bash
python train_dpo.py \
  --model_path checkpoints/best_model.pth \
  --train_npz_list data/train_1000.json \
  --valid_npz_list data/valid_200.json \
  --preference_mode rule \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 5e-6 \
  --beta 0.2 \
  --exp_name high_beta_experiment
```

### Programmatic Usage

```python
from model_utils import load_model
from trainer import DPOTrainer
from preference_collection import generate_rule_based_preferences
import torch

# Load model
device = torch.device("cuda")
model, model_args = load_model("path/to/model.pth", device)

# Create trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
trainer = DPOTrainer(
    policy_model=model,
    model_args=model_args,
    optimizer=optimizer,
    device=device,
    run_dir=Path("experiments/my_run"),
    batch_size=32,
    beta=0.1
)

# Collect preferences
preferences = generate_rule_based_preferences(
    model, model_args, Path("data/train.json"), device
)

# Train
metrics = trainer.train_epoch(preferences, epoch=1)
print(f"Loss: {metrics['loss']:.4f}")
```

## Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | Path | required | Initial model checkpoint |
| `--train_npz_list` | Path | required | Training data list (JSON) |
| `--valid_npz_list` | Path | required | Validation data list (JSON) |
| `--exp_name` | str | "dpo_experiment" | Experiment name |
| `--preference_mode` | str | "rule" | "rule", "gui", or "lichtblick" |
| `--train_epochs` | int | 3 | Number of epochs |
| `--batch_size` | int | 32 | Training batch size |
| `--learning_rate` | float | 1e-5 | Optimizer learning rate (use 5e-4 for LoRA) |
| `--beta` | float | 0.1 | DPO regularization |
| `--use_lora` | flag | False | Enable LoRA adapter training |
| `--lora_rank` | int | 16 | LoRA rank |
| `--lora_alpha` | int | 16 | LoRA alpha scaling |
| `--lora_dropout` | float | 0.05 | LoRA dropout |

### Trajectory Generation Parameters

Adjustable in GUI or programmatically:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| noise_scale | 2.5 | 0.5-5.0 | Stochastic trajectory diversity |
| fde_threshold | 2.0 | 0.5-10.0 | Minimum endpoint distance (m) |
| max_retries | 50 | 10-200 | Generation attempts |


## License

Part of the Diffusion Planner project.
