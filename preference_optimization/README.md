# Preference Optimization for Diffusion Planner

Clean, modular implementation of Direct Preference Optimization (DPO) for trajectory planning.

## Features

- 🎯 **Modular Architecture** - Clean separation of concerns
- 🚀 **FDE-based Generation** - Ensures diverse trajectory pairs
- 🌐 **Web UI** - Modern Gradio interface for annotation
- 📊 **Comprehensive Logging** - Track training metrics
- 🔧 **Type-Safe** - Full type hints throughout

## Quick Start

```bash
# Install dependencies
pip install gradio>=4.0.0

# Train with GUI annotation
python train_dpo.py \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode gui \
  --train_epochs 10 \
  --exp_name my_experiment

# Train with automatic (rule-based) annotation
python train_dpo.py \
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


## Architecture

### Module Structure

```
preference_optimization/
├── train_dpo.py              # Main entry point (170 lines)
├── trainer.py                # DPOTrainer class (220 lines)
├── dpo_loss.py               # DPO loss computation (210 lines)
├── model_utils.py            # Model loading (65 lines)
├── preference_collection.py  # Preference generation (90 lines)
├── annotation_gui.py         # Gradio web UI (450 lines)
├── visualization.py          # Validation visualization (100 lines)
├── datasets.py               # Dataset classes (70 lines)
├── utils.py                  # Core utilities (150 lines)
└── test_fde_standalone.py    # Unit tests (155 lines)
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
| `--preference_mode` | str | "rule" | "rule" or "gui" |
| `--train_epochs` | int | 10 | Number of epochs |
| `--batch_size` | int | 32 | Training batch size |
| `--learning_rate` | float | 1e-5 | Optimizer learning rate |
| `--beta` | float | 0.1 | DPO regularization |

### Trajectory Generation Parameters

Adjustable in GUI or programmatically:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| noise_scale | 2.5 | 0.5-5.0 | Stochastic trajectory diversity |
| fde_threshold | 2.0 | 0.5-10.0 | Minimum endpoint distance (m) |
| max_retries | 50 | 10-200 | Generation attempts |


## License

Part of the Diffusion Planner project.
