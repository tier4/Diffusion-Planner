# Quick Start Guide - Preference Optimization

Clean, modular preference optimization system for trajectory planning with DPO.

## Installation

```bash
# Install Gradio for web UI
pip install gradio>=4.0.0

# Or install all requirements
cd /home/yukky/workspace/Diffusion-Planner
pip install -r diffusion_planner/requirements.txt
```

## Quick Start

### Option 1: GUI Mode (Recommended)

```bash
cd /home/yukky/workspace/Diffusion-Planner/preference_optimization

python train_dpo.py \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode gui \
  --train_epochs 10 \
  --exp_name my_experiment
```

**What happens:**
1. Browser opens to http://localhost:7860
2. Annotate trajectory preferences via web interface
3. Training starts automatically after annotation
4. Results saved to timestamped directory

### Option 2: Automatic Mode

```bash
python train_dpo.py \
  --model_path <path/to/model.pth> \
  --train_npz_list <path/to/train.json> \
  --valid_npz_list <path/to/valid.json> \
  --preference_mode rule \
  --train_epochs 10
```

**What happens:**
- Preferences generated automatically using path length
- No manual annotation needed
- Faster but less flexible

## File Structure

```
preference_optimization/
├── train_dpo.py              # Main entry point
├── trainer.py                # DPOTrainer class
├── dpo_loss.py               # Loss computation
├── model_utils.py            # Model loading
├── preference_collection.py  # Preference generation
├── annotation_gui.py         # Web UI
├── visualization.py          # Validation plots
├── datasets.py               # Dataset classes
├── utils.py                  # Core utilities
├── test_fde_standalone.py    # Tests
└── README.md                 # Detailed documentation
```

## Using the Web UI

### Interface Overview

1. **Progress Bar** - Shows completion status
2. **FDE Display** - Endpoint distance between trajectories
3. **Parameters**
   - Noise Scale (0.5-5.0): Controls trajectory diversity
   - FDE Threshold (0.5-10.0m): Minimum endpoint distance
   - Max Retries (10-200): Generation attempts
4. **Visualizations**
   - Left: Trajectory comparison with map
   - Right: Velocity profiles
5. **Action Buttons**
   - ✓ Trajectory 1 (Green) is Better
   - ✓ Trajectory 2 (Orange) is Better
   - 🔄 Regenerate Pair
6. **Navigation** - Jump ±1, ±10, ±30 samples

### Annotation Tips

**Good Preferences Consider:**
- ✅ Safety (collision avoidance)
- ✅ Comfort (smooth motion)
- ✅ Efficiency (shorter paths when safe)
- ✅ Naturalness (human-like behavior)

**Best Practices:**
- Be consistent in criteria
- Use regenerate if both are poor
- Take breaks to avoid fatigue
- Focus on "why" not just "what"

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model_path` | Path | required | Initial model checkpoint |
| `--train_npz_list` | Path | required | Training data (JSON list) |
| `--valid_npz_list` | Path | required | Validation data (JSON list) |
| `--exp_name` | str | "dpo_experiment" | Experiment name |
| `--preference_mode` | str | "rule" | "rule" or "gui" |
| `--train_epochs` | int | 10 | Number of epochs |
| `--batch_size` | int | 32 | Batch size |
| `--learning_rate` | float | 1e-5 | Learning rate |
| `--beta` | float | 0.1 | DPO regularization |

## Examples

### Minimal Example

```bash
python train_dpo.py \
  --model_path checkpoints/model.pth \
  --train_npz_list data/train.json \
  --valid_npz_list data/valid.json \
  --preference_mode gui
```

### Full Configuration

```bash
python train_dpo.py \
  --model_path checkpoints/best_model.pth \
  --train_npz_list data/train_1000.json \
  --valid_npz_list data/valid_200.json \
  --preference_mode gui \
  --train_epochs 20 \
  --batch_size 64 \
  --learning_rate 5e-6 \
  --beta 0.2 \
  --exp_name high_beta_run
```

### Quick Test Run

```bash
# Test with small dataset
python train_dpo.py \
  --model_path checkpoints/model.pth \
  --train_npz_list data/test_10.json \
  --valid_npz_list data/test_10.json \
  --preference_mode rule \
  --train_epochs 1 \
  --exp_name quick_test
```

## Programmatic Usage

### Basic Training

```python
from model_utils import load_model
from trainer import DPOTrainer
from preference_collection import generate_rule_based_preferences
from pathlib import Path
import torch

# Setup
device = torch.device("cuda")
model, model_args = load_model(Path("model.pth"), device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Create trainer
trainer = DPOTrainer(
    policy_model=model,
    model_args=model_args,
    optimizer=optimizer,
    device=device,
    run_dir=Path("experiments/run_001"),
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

# Save
trainer.save_checkpoint(epoch=1, args_dict={})
```

### Custom Preference Collection

```python
from annotation_gui import collect_preferences

# GUI-based collection
preferences = collect_preferences(
    policy_model=model,
    model_args=model_args,
    npz_list=Path("data/train.json"),
    target_count=100
)

# Each preference is a dict:
# {
#     "npz_path": "sample_001.npz",
#     "trajectory_w": [[x, y, h, v], ...],  # Winner
#     "trajectory_l": [[x, y, h, v], ...]   # Loser
# }
```

### Using Individual Modules

```python
# Load model
from model_utils import load_model
model, args = load_model(Path("model.pth"), device)

# Create dataset
from datasets import DPODataset
dataset = DPODataset(preferences, device)

# Compute loss
from dpo_loss import compute_dpo_loss
loss, metrics = compute_dpo_loss(
    policy_model, reference_model, batch, beta=0.1,
    model_args=args, device=device
)

# Visualize
from visualization import visualize_validation
visualize_validation(
    model, valid_loader, args,
    save_dir=Path("output"), epoch=1, device=device
)
```

## Output Structure

After training:

```
experiments/
└── 20260126-120000_my_experiment/
    ├── args.json                    # Model config
    ├── dpo_args.json                # Training config
    ├── latest.pth                   # Latest checkpoint
    ├── epoch_010.pth                # Periodic checkpoints
    ├── dpo_train_log.tsv            # Training metrics
    └── validation_vis/              # Visualizations
        ├── sample_000_epoch_0000.png
        ├── sample_001_epoch_0000.png
        └── ...
```

## Testing

```bash
# Run unit tests
python3 test_fde_standalone.py

# Expected output:
# ============================================================
# ALL TESTS PASSED! ✓
# ============================================================
```

## Troubleshooting

### ImportError: No module named 'gradio'

```bash
pip install gradio>=4.0.0
```

### CUDA out of memory

```bash
# Reduce batch size
python train_dpo.py --batch_size 16 ...
```

### ModuleNotFoundError: diffusion_planner

```bash
# Make sure you're in the right directory
cd /path/to/Diffusion-Planner/preference_optimization
```

### Gradio interface won't open

- Check port 7860 is available
- Check browser isn't blocking localhost
- Look for URL in console output
- Try accessing http://127.0.0.1:7860 manually

### Trajectories too similar

Increase diversity parameters:
```bash
# In GUI: adjust sliders
# Noise Scale: 3.5-4.5
# FDE Threshold: 4.0-6.0
# Max Retries: 100-150
```

### Training is slow

```bash
# Reduce batch size or use fewer samples
python train_dpo.py --batch_size 16 ...
```

## Performance

### Expected Times
- Model loading: ~5 seconds
- Preference annotation: ~5-10 seconds per sample
- Training epoch (1000 samples): ~5-10 minutes
- Visualization: ~30 seconds per epoch

### Resource Usage
- GPU memory: ~2-4GB (model-dependent)
- RAM: ~1GB for data loading
- Disk: ~100MB per experiment

## Tips & Best Practices

### Preference Annotation
1. **Consistency** - Apply same criteria throughout
2. **Safety First** - Prioritize collision avoidance
3. **Natural Behavior** - Prefer human-like trajectories
4. **Use Regenerate** - Get better pairs if needed
5. **Take Breaks** - Avoid annotation fatigue

### Training
1. **Start Small** - Test with small dataset first
2. **Monitor Metrics** - Check loss and accuracy
3. **Tune Beta** - Start at 0.1, adjust if needed
4. **Save Often** - Checkpoints every 10 epochs
5. **Visualize** - Review validation predictions

### Experimentation
1. **Name Experiments** - Use descriptive `exp_name`
2. **Version Control** - Track training configs
3. **Compare Results** - Use consistent validation set
4. **Document Changes** - Note what you tried
5. **Iterate** - Start simple, add complexity

## Next Steps

1. **Test Installation**
   ```bash
   python3 test_fde_standalone.py
   ```

2. **Small Run**
   ```bash
   python train_dpo.py --preference_mode rule \
     --train_epochs 1 --exp_name test_run ...
   ```

3. **GUI Annotation**
   ```bash
   python train_dpo.py --preference_mode gui ...
   ```

4. **Full Training**
   ```bash
   python train_dpo.py --preference_mode gui \
     --train_epochs 20 --exp_name production_run ...
   ```

5. **Evaluate Results**
   - Check `dpo_train_log.tsv`
   - Review visualizations
   - Compare with baseline

## Documentation

- `README.md` - Detailed module documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `COMPREHENSIVE_REFACTORING.md` - Architecture overview

## Support

For issues:
1. Check this guide
2. Review module README
3. Run tests to verify installation
4. Check error messages carefully

## Contributing

When extending the code:
1. Follow existing module structure
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Add tests for new features
5. Update documentation

## License

Part of the Diffusion Planner project.
