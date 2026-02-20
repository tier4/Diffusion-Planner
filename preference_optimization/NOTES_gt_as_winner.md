# GT-as-Winner DPO Strategy

## Concept

Use the ground truth trajectory (`ego_agent_future` from the NPZ file) as `traj_w`, paired
against one of the model's generated trajectories as `traj_l`. This gives a strong,
unambiguous preference signal without requiring the annotator to compare two model outputs.

The preferred UI approach is a **manual button** in the annotation interface: a "GT is Best"
button alongside the existing orange/green selection buttons. The annotator presses it when
they can see the model's outputs are clearly worse than the recorded behaviour.

---

## GT smoothing during normal training

The raw `ego_agent_future` stored in NPZ files is `[T, 3]` (x, y, heading in radians),
extracted without any smoothing from the rosbag in `ros_scripts/parse_rosbag.py`.

During training, smoothing is applied **only when data augmentation is active** via
`smoothing_future_trajectory()` in
`diffusion_planner/diffusion_planner/utils/unicycle_accel_curvature.py`.

### What the smoothing does

It converts the raw trajectory into unicycle action space (acceleration + curvature),
applies **Tikhonov regularisation** (no scipy — pure torch, solved via Cholesky), then
reconstructs the trajectory from the smoothed actions using a kinematic model:

| Quantity | Smoothing order | Lambda default |
|---|---|---|
| Heading θ | 3rd order | `theta_lambda=1e-1` |
| Velocity v | 3rd order | `v_lambda=1e-6` |
| Acceleration a | 2nd order | `a_lambda=1e-4` |
| Curvature κ | 2nd order | `kappa_lambda=1e-4` |

Entry point:
```python
# data_augmentation.py:230-232
from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory

smoothed_ego_future4d = smoothing_future_trajectory(
    ego_past4d,                   # [B, T_past, 4]
    inputs["ego_current_state"],  # [B, 10]
    ego_future4d,                 # [B, T_future, 4]  (already in cos/sin format here)
)
```

`UnicycleAccelCurvatureActionSpace` is instantiated with default lambdas inside
`smoothing_future_trajectory` — no external config needed.

### When to apply it for DPO

The same smoothing **should be applied** to the GT trajectory before using it as `traj_w` in
DPO, for consistency with the distribution the model was trained on. Without it, the GT
carries localization noise (GPS/LIDAR drift) that was smoothed away during base training,
which would introduce a distribution mismatch.

The smoothing expects the trajectory already in `[B, T, 4]` (cos/sin) format, so the
conversion from `[T, 3]` (heading) must happen first (see Format conversion below).

---

## Format conversion

GT is stored as `[T, 3]` (x, y, heading in radians).
`compute_trajectory_loss` and the smoothing function expect `[T, 4]` (x, y, cos(yaw), sin(yaw)).

```python
import numpy as np
import torch

def gt_to_model_format(gt: np.ndarray) -> np.ndarray:
    """Convert GT trajectory [T, 3] (x, y, heading) → [T, 4] (x, y, cos, sin)."""
    cos_yaw = np.cos(gt[:, 2:3])
    sin_yaw = np.sin(gt[:, 2:3])
    return np.concatenate([gt[:, :2], cos_yaw, sin_yaw], axis=1)
```

---

## GT validity check

`ego_agent_future` is all-zeros when the recording segment is invalid. Must filter before use:

```python
def gt_is_valid(gt: np.ndarray, min_valid_ratio: float = 0.8) -> bool:
    """Return True if at least min_valid_ratio of GT points are non-zero."""
    valid = ~((gt[:, 0] == 0) & (gt[:, 1] == 0))
    return valid.mean() >= min_valid_ratio
```

---

## UI implementation — "GT is Best" button

Add a third selection button to the Gradio annotation interface alongside
"Orange is Better" and "Regenerate":

```
[ ✓ Orange (Stochastic) is Better ]   [ 🎯 GT is Best ]   [ 🔄 Regenerate ]
```

### What it does when clicked

1. Load `ego_agent_future` from `self.current_data`
2. Validate with `gt_is_valid()`; show error message if invalid
3. Convert `[T, 3]` → `[T, 4]` with `gt_to_model_format()`
4. Apply `smoothing_future_trajectory()` for consistency with training distribution
5. Record preference:
   - `traj_w` = smoothed GT (4D)
   - `traj_l` = `self.trajectory_1` (deterministic, the model's best guess)
   - `npz_path` = current sample path
6. Advance to next sample (same flow as `select_winner`)

### Loser choice rationale

Pairing GT against the **deterministic** output (green) is preferred over the stochastic
(orange) because:
- The deterministic trajectory is the model's MAP estimate — the strongest loser signal
- The stochastic trajectory is already noisy and variable; using it as loser would make
  the training signal noisier
- It directly tells the model: "your best guess is still worse than recorded behaviour"

### Changes required

**`annotation_gui.py`**
- Add `select_gt_btn` Gradio button
- Add `select_gt_as_winner()` method to `PreferenceAnnotator`:
  - Needs access to `self.current_data["ego_agent_future"]`
  - Applies validity check, format conversion, smoothing
  - Records `(gt_smoothed, traj_1)` as `(traj_w, traj_l)`
  - Calls `load_sample(...)` for the next sample
- `select_gt_btn` disabled when GT is invalid for current sample (check on `load_sample`)

**`annotation_ws_server.py`**
- Add `"select_gt_as_winner"` action in `_handle_action`
- Include `gt_available` flag in state payload so the Lichtblick UI can disable the button

**`annotation_gui.py` — `create_interface`**
- Add button to UI layout
- Wire click → `select_gt_as_winner(...)` with same inputs as other buttons
- Add to `_full_outputs` if button interactivity needs to be controlled

---

## Smoothing call in DPO context — full sketch

```python
from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory
from preference_optimization.utils import load_npz_data

def get_smoothed_gt(current_data: dict, device: torch.device) -> np.ndarray | None:
    """Return smoothed GT as [T, 4] numpy array, or None if GT is invalid."""
    if "ego_agent_future" not in current_data:
        return None

    gt_raw = current_data["ego_agent_future"][0].cpu().numpy()  # [T, 3]

    # Validity check
    valid = ~((gt_raw[:, 0] == 0) & (gt_raw[:, 1] == 0))
    if valid.mean() < 0.8:
        return None

    # Convert to 4D (cos/sin) and add batch dim
    gt_4d = gt_to_model_format(gt_raw)                          # [T, 4]
    gt_tensor = torch.tensor(gt_4d, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 4]

    # Need ego_agent_past in 4D for smoothing (already converted by load_npz_data)
    ego_past = current_data["ego_agent_past"]                   # [1, T_past, 4]
    ego_current = current_data["ego_current_state"]             # [1, 10]

    # Apply same smoothing used during training
    smoothed = smoothing_future_trajectory(ego_past, ego_current, gt_tensor)  # [1, T, 4]

    return smoothed[0].cpu().numpy()   # [T, 4]
```

---

## Open questions

- Should the "GT is Best" button also be available as a rule-based automatic mode
  (e.g., run before launching the GUI, auto-label all samples where ADE > threshold)?
- Should the loser be the deterministic only, or let the user choose
  (green = loser, orange = loser, or worse-of-two = loser)?
- Does smoothing need the unnormalised `ego_agent_past`? Check whether
  `load_npz_data` already converts it with `heading_to_cos_sin` (it does, line 31)
  — so `ego_agent_past` in `current_data` is already in 4D cos/sin format, which is
  what `smoothing_future_trajectory` expects.
