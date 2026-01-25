# Quick Start Guide - Preference Optimization UI

## What's New?

The preference optimization system now supports:
1. **FDE-based trajectory generation** - Ensures trajectory pairs have distinct endpoints
2. **Runtime parameter controls** - Adjust noise, FDE threshold, and retries on-the-fly
3. **Gradio web UI** - Modern browser-based interface (alternative to Tkinter)

## Installation

Install the new Gradio dependency:

```bash
pip install gradio>=4.0.0
```

Or install all requirements:

```bash
cd /home/yukky/workspace/Diffusion-Planner
pip install -r diffusion_planner/requirements.txt
```

## Quick Start

### Option 1: Tkinter UI (Desktop)

```bash
cd /home/yukky/workspace/Diffusion-Planner

python preference_optimization/train_dpo.py \
  --model_path <path/to/your/model.pth> \
  --train_npz_list <path/to/train_list.json> \
  --valid_npz_list <path/to/valid_list.json> \
  --preference_mode gui \
  --ui_framework tkinter \
  --train_epochs 10 \
  --exp_name my_dpo_experiment
```

**What you'll see:**
- Desktop window with trajectory visualization
- Parameter sliders for noise scale, FDE threshold, and max retries
- FDE display showing current distance and attempts
- Three action buttons: Select Trajectory 1, Select Trajectory 2, Regenerate

### Option 2: Gradio UI (Web Browser)

```bash
cd /home/yukky/workspace/Diffusion-Planner

python preference_optimization/train_dpo.py \
  --model_path <path/to/your/model.pth> \
  --train_npz_list <path/to/train_list.json> \
  --valid_npz_list <path/to/valid_list.json> \
  --preference_mode gui \
  --ui_framework gradio \
  --train_epochs 10 \
  --exp_name my_dpo_experiment
```

**What you'll see:**
- Browser automatically opens to http://localhost:7860
- Clean web interface with two visualization panels
- Parameter controls at the top
- Navigation buttons for jumping between samples

### Option 3: Rule-Based (No GUI)

```bash
cd /home/yukky/workspace/Diffusion-Planner

python preference_optimization/train_dpo.py \
  --model_path <path/to/your/model.pth> \
  --train_npz_list <path/to/train_list.json> \
  --valid_npz_list <path/to/valid_list.json> \
  --preference_mode rule \
  --train_epochs 10 \
  --exp_name my_dpo_experiment
```

**No GUI - automatic annotation based on path length**

## Parameter Guide

### Noise Scale
Controls how different the second trajectory is from the deterministic one.

- **Default: 2.5**
- **Range: 0.5 - 5.0**
- **Lower (1.0-2.0)**: Subtle variations, good for fine-grained preferences
- **Higher (3.0-5.0)**: Dramatic differences, easier to distinguish

### FDE Threshold
Minimum distance (in meters) between trajectory endpoints.

- **Default: 2.0m**
- **Range: 0.5 - 10.0m**
- **Lower (1.0-2.0m)**: Faster generation, trajectories may be similar
- **Higher (4.0-8.0m)**: Slower generation, very distinct trajectories

### Max Retries
Maximum attempts to generate a pair meeting the FDE threshold.

- **Default: 50**
- **Range: 10 - 200**
- **Lower (10-30)**: Faster but may not meet threshold
- **Higher (100-200)**: More likely to meet threshold

## Workflow

### Typical Annotation Session

1. **Start the training script** with your preferred UI
2. **Review the first trajectory pair** that appears
3. **Adjust parameters if needed**:
   - If trajectories are too similar → increase noise scale or FDE threshold
   - If generation is slow → decrease max retries or FDE threshold
   - If trajectories are unrealistic → decrease noise scale
4. **Select the better trajectory** or regenerate for a new pair
5. **Navigate** through samples using the buttons
6. **Complete annotation** for all samples
7. **Training begins** automatically using your preferences

### Keyboard Tips (Tkinter)

- Use tab to navigate between controls
- Use arrow keys to adjust sliders
- Use space/enter to click buttons

### Browser Tips (Gradio)

- Works on any modern browser (Chrome, Firefox, Safari)
- Can keep browser tab open and return later
- Refreshing the page will restart annotation (preferences saved)

## Verification Tests

### Test FDE Calculation

```bash
cd /home/yukky/workspace/Diffusion-Planner/preference_optimization
python3 test_fde_standalone.py
```

**Expected output:**
```
============================================================
FDE Calculation Test Suite (Standalone)
============================================================

Test 1 - Identical trajectories: FDE = 0.0000 (expected: 0.0000)
✓ PASSED

Test 2 - Known distance: FDE = 4.0000 (expected: 4.0000)
✓ PASSED

...

ALL TESTS PASSED! ✓
============================================================
```

## Troubleshooting

### "gradio not found"
```bash
pip install gradio>=4.0.0
```

### "Module 'diffusion_planner' not found"
Make sure you're in the correct directory and have the diffusion_planner package installed.

### Tkinter window doesn't appear
- Check if you're on a system with display (not headless server)
- Use Gradio UI instead for remote/headless systems

### Gradio interface doesn't open
- Check if port 7860 is available
- Check browser isn't blocking localhost connections
- Look for the URL in the console output

### Trajectories are always too similar
- Increase noise scale (try 3.5-4.5)
- Increase FDE threshold (try 4.0-6.0m)
- Increase max retries (try 100-150)

### Generation is too slow
- Decrease max retries (try 20-30)
- Decrease FDE threshold (try 1.5-2.5m)
- Use faster GPU if available

## Example Commands

### Minimal Example (Tkinter)
```bash
python preference_optimization/train_dpo.py \
  --model_path checkpoints/model.pth \
  --train_npz_list data/train.json \
  --valid_npz_list data/valid.json \
  --preference_mode gui \
  --train_epochs 5
```

### Full Example (Gradio)
```bash
python preference_optimization/train_dpo.py \
  --model_path checkpoints/best_model.pth \
  --train_npz_list data/train_samples.json \
  --valid_npz_list data/valid_samples.json \
  --preference_mode gui \
  --ui_framework gradio \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --beta 0.1 \
  --exp_name high_quality_preferences
```

### Rule-Based Baseline
```bash
python preference_optimization/train_dpo.py \
  --model_path checkpoints/model.pth \
  --train_npz_list data/train.json \
  --valid_npz_list data/valid.json \
  --preference_mode rule \
  --train_epochs 20
```

## File Locations

- **Main training script**: `preference_optimization/train_dpo.py`
- **Tkinter UI**: `preference_optimization/annotation_gui.py`
- **Gradio UI**: `preference_optimization/annotation_gui_gradio.py`
- **Utils (FDE, retry logic)**: `preference_optimization/utils.py`
- **Tests**: `preference_optimization/test_fde_standalone.py`
- **Documentation**: `IMPLEMENTATION_SUMMARY.md`, `QUICK_START_GUIDE.md`

## Next Steps

1. **Test with small dataset** - Verify UI works with 10-20 samples
2. **Tune parameters** - Find optimal settings for your data
3. **Run full training** - Collect preferences for all training samples
4. **Evaluate results** - Compare DPO model vs baseline
5. **Iterate** - Adjust parameters based on results

## Support

For detailed implementation information, see:
- `IMPLEMENTATION_SUMMARY.md` - Complete technical details
- Plan transcript: `/home/yukky/.claude/projects/-home-yukky-workspace-Diffusion-Planner/46dad98c-d393-40fb-ae35-0b09ff567bf2.jsonl`

## Tips for Better Preferences

1. **Be consistent** - Use the same criteria for all comparisons
2. **Focus on safety** - Prefer trajectories that avoid collisions
3. **Consider comfort** - Prefer smooth, natural driving behavior
4. **Think about efficiency** - Prefer shorter, more direct paths when safe
5. **Use regenerate** - If both trajectories are poor, regenerate
6. **Take breaks** - Annotating 100+ samples can be tiring

## Advanced Usage

### Custom Parameters in Code

You can also call the GUI functions directly:

```python
from annotation_gui import collect_preferences_gui
from annotation_gui_gradio import collect_preferences_gui_gradio

# Tkinter with custom parameters
preferences = collect_preferences_gui(
    policy_model=model,
    model_args=args,
    npz_list=Path("data/train.json"),
    target_count=100,
    noise_scale=3.0,
    fde_threshold=4.0,
    max_retries=75
)

# Gradio with custom parameters
preferences = collect_preferences_gui_gradio(
    policy_model=model,
    model_args=args,
    npz_list=Path("data/train.json"),
    target_count=100,
    noise_scale=2.5,
    fde_threshold=3.0,
    max_retries=100
)
```

### Using FDE Calculation Directly

```python
from preference_optimization.utils import calculate_fde
import numpy as np

traj_1 = np.array([[0, 0, 0, 5], [1, 1, 0.1, 5], [2, 2, 0.2, 5]])
traj_2 = np.array([[0, 0, 0, 5], [1, 0, 0, 5], [3, 0, 0, 5]])

fde = calculate_fde(traj_1, traj_2)
print(f"FDE: {fde:.2f}m")
```
