# GT-as-Winner DPO Strategy

## Concept

Use the ground truth trajectory (`ego_agent_future` from the NPZ file) as `traj_w` paired against the
deterministic model output (`traj_1`) as `traj_l`, instead of relying on human annotation.
This gives automatic, zero-cost preference pairs wherever GT is available and the model is clearly worse.

## Format conversion needed

GT is stored as `[T, 3]` (x, y, heading in radians).
`compute_trajectory_loss` expects `[T, 4]` (x, y, cos(yaw), sin(yaw)).

```python
import numpy as np

def gt_to_model_format(gt: np.ndarray) -> np.ndarray:
    """Convert GT trajectory [T, 3] (x, y, heading) to model format [T, 4] (x, y, cos, sin)."""
    xy = gt[:, :2]
    cos_yaw = np.cos(gt[:, 2:3])
    sin_yaw = np.sin(gt[:, 2:3])
    return np.concatenate([xy, cos_yaw, sin_yaw], axis=1)
```

## GT validity check

`ego_agent_future` is all-zeros when the recording is invalid. Must filter before using:

```python
def gt_is_valid(gt: np.ndarray, min_valid_ratio: float = 0.8) -> bool:
    """Return True if enough GT points are non-zero."""
    valid = ~((gt[:, 0] == 0) & (gt[:, 1] == 0))
    return valid.mean() >= min_valid_ratio
```

## Where to plug it in

### Option 1 — New rule-based mode in `preference_collection.py`

Add a `generate_gt_vs_deterministic_preferences()` function alongside the existing
`generate_rule_based_preferences()`. For each NPZ:

1. Check GT validity
2. If ADE(deterministic, GT) > `min_ade_threshold` (model is clearly wrong): add pair
   with `traj_w = gt_formatted`, `traj_l = traj_1` (deterministic)
3. Otherwise skip (model already close to GT — weak signal)

Suggested `min_ade_threshold`: ~1.0 m (same as the existing `ade_threshold` default).

```python
def generate_gt_vs_deterministic_preferences(
    policy_model,
    model_args,
    npz_list: Path,
    device: torch.device,
    min_ade_threshold: float = 1.0,
) -> list[dict]:
    ...
    for npz_path in npz_paths:
        data = load_npz_data(npz_path, device)
        if "ego_agent_future" not in data:
            continue
        gt_raw = data["ego_agent_future"][0].cpu().numpy()   # [T, 3]
        if not gt_is_valid(gt_raw):
            continue

        # Generate deterministic trajectory
        data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4).to(device)
        _, outputs = policy_model(data)
        traj_det = outputs["prediction"][0, 0].cpu().numpy()   # [T, 4]

        gt_4d = gt_to_model_format(gt_raw)                     # [T, 4]
        ade = calculate_ade(traj_det, gt_raw)                  # existing util
        if ade < min_ade_threshold:
            continue  # model already close — skip

        preferences.append({
            "npz_path": npz_path,
            "trajectory_w": gt_4d.tolist(),
            "trajectory_l": traj_det.tolist(),
        })
    ...
```

### Option 2 — Mixed strategy (most viable)

Run `generate_gt_vs_deterministic_preferences()` first to get automatic pairs.
Then run the GUI annotation only on the remaining samples where GT was invalid
or ADE was below the threshold. Combine both lists before calling `DPOTrainer.train_epoch()`.

### Option 3 — GUI toggle

Add a "Use GT as winner" button/checkbox in the Gradio UI. When active, clicking
"select" automatically records GT as `traj_w` without requiring the user to visually
compare trajectories. Useful for quickly labeling clear-cut cases during annotation.

## Risks / things to watch

- **GT localization noise**: GPS/LIDAR drift makes GT positions slightly jittery.
  Not a blocker but the "perfect GT" assumption is approximate.
- **Weak gradient when model is close to GT**: the DPO loss margin shrinks. Use
  `min_ade_threshold` to filter these out.
- **Imitation learning effect**: GT-as-winner pushes the model toward recorded
  human driving style. This is usually desirable but may conflict with any
  comfort/safety preferences expressed through human annotation.
- **Reference model updated per epoch** (`copy.deepcopy` in `trainer.py:94`):
  the KL penalty resets each epoch, so the anti-drift effect is per-epoch only.
