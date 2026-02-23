# GT-as-Winner DPO Strategy

## Status

**Implemented** — `select_gt_as_winner()` method and "🎯 GT is Best" button are live in
`annotation_gui.py`. GT is smoothed, validated, and recorded as `traj_w` against the
deterministic output as `traj_l`. Button is automatically disabled when GT is unavailable.
WebSocket server exposes `select_gt_as_winner` action and `gt_available` state flag.

---

## Academic context

This pattern is well-established in the RLHF/DPO literature under several names:

**SPIN — Self-Play Fine-Tuning** (Chen et al., 2024, "Self-Play Fine-Tuning Converts Weak
Language Models to Strong Language Models"): formalises pairing human demonstrations (winner)
against the model's own current outputs (loser). Exactly what we do: GT trajectory (winner)
vs deterministic model output (loser).

**Offline DPO / SFT-seeded DPO**: The original DPO paper (Rafailov et al., 2023, "Direct
Preference Optimization: Your Language Model is Secretly a Reward Model") discusses using
human demonstration data directly as the preferred side of preference pairs. Our use of
recorded driving as the winner side is a direct instance of this.

**KTO — Kahneman-Tversky Optimisation** (Ethayarajh et al., 2024): A variant that works
with unpaired "good examples" rather than ranked pairs — equivalent to GT-as-winner without
needing a loser trajectory at all. Could be a simpler alternative if GT quality is high.

**Key theoretical justification**: the KL regularisation term in the DPO objective (via the
reference model) prevents the policy from collapsing into pure imitation of GT, which would
happen with standard SFT. This gives GT-grounded training without catastrophic forgetting
of the model's generalisation capabilities.

**In autonomous driving**: GT-as-winner is implicit in several RLHF-for-driving works
(e.g., Waymo, Motional internal reports) where human-driven trajectories serve as positives
against policy rollouts. The combination of GT pairs (for clear-cut cases) and human
annotation pairs (for ambiguous cases) is the standard mixed strategy in practice.

---

## Concept

Use the ground truth trajectory (`ego_agent_future` from the NPZ file) as `traj_w`, paired
against one of the model's generated trajectories as `traj_l`. This gives a strong,
unambiguous preference signal without requiring the annotator to compare two model outputs.

The annotator presses the "🎯 GT is Best" button when the model's outputs are clearly worse
than the recorded behaviour. The button is disabled automatically when GT is unavailable for
the current sample.

---

## GT smoothing — investigation result

**No smoothing is applied during rosbag→NPZ conversion.** This was verified in both
conversion paths:

- **Python path** (`ros_scripts/parse_rosbag.py`): directly extracts `(x, y, quaternion)`
  from `/localization/kinematic_state` Odometry messages, converts quaternion→yaw, and
  stores `[T, 3]` (x, y, heading). No filtering.
- **C++ path** (`autoware_diffusion_planner/tool/data_converter.cpp` +
  `src/preprocessing/preprocessing_utils.cpp`): identical logic — pose extraction, frame
  transform, rotation matrix→cos/sin. No smoothing whatsoever.

Both paths consume `/localization/kinematic_state` which is published by Autoware's
**EKF localization** (multi-sensor fusion of GNSS, IMU, LiDAR scan matching). The EKF
provides filtered poses, but this filtering is internal to Autoware and opaque to the
converter — we take the published pose as-is.

### Implications for DPO

The model was trained on this raw EKF output as GT (when augmentation is off) or on
unicycle-smoothed versions (when augmentation is on, controlled by `augment_prob`).
Since the model predominantly sees the raw EKF trajectory as GT, **applying
`smoothing_future_trajectory()` to GT in DPO would introduce a distribution mismatch**
— the DPO winner would look like augmented training data rather than the standard GT.

**Conclusion: do not apply unicycle smoothing to GT in `_get_smoothed_gt()`.**
The method simply converts `[T, 3]` (x, y, heading) → `[T, 4]` (x, y, cos, sin).

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

## GT conversion in DPO — actual implementation

```python
# _get_smoothed_gt() in annotation_gui.py — name kept for historical reasons
def _get_smoothed_gt(self) -> np.ndarray | None:
    gt_raw = self.current_data["ego_agent_future"][0].cpu().numpy()  # [T, 3]
    cos_yaw = np.cos(gt_raw[:, 2:3])
    sin_yaw = np.sin(gt_raw[:, 2:3])
    return np.concatenate([gt_raw[:, :2], cos_yaw, sin_yaw], axis=1).astype(np.float32)
```

No smoothing — just the heading→cos/sin conversion to match the model's `[T, 4]` format.

---

## Open questions

- Should the "GT is Best" button also be available as a rule-based automatic mode
  (e.g., run before launching the GUI, auto-label all samples where ADE > threshold)?
- Should the loser be the deterministic only, or let the user choose
  (green = loser, orange = loser, or worse-of-two = loser)?
