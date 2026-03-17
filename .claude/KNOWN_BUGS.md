# Known Bugs

Bugs discovered during GRPO reward implementation (2026-03-17).
Training loss is unaffected -- these impact guidance sampling and validation metrics only.

## 1. `loss.py` `neighbor_clearance_penalty` -- P/T reshape bug

**File:** `diffusion_planner/diffusion_planner/loss.py`, line 239

**Bug:** `neighbor_rect` has shape `(B, P, T, 6)` but the reshape assumes `(B, T, P, 6)`:
```python
neighbor_flat = neighbor_rect.reshape(B * T, P, 6)  # mixes P and T dimensions
valid_mask = neighbors_future_valid.reshape(B * T, P)  # same bug
```

Element `[b, p=5, t=10, d=0]` at flat index `5*T*6 + 10*6 + 0` gets placed at the wrong
position in the reshaped tensor. The SAT collision distances are computed against scrambled
neighbor data.

**Fix:**
```python
neighbor_rect = neighbor_rect.permute(0, 2, 1, 3).contiguous()  # (B, T, P, 6)
neighbor_flat = neighbor_rect.reshape(B * T, P, 6)
valid_mask = neighbors_future_valid.permute(0, 2, 1).contiguous().reshape(B * T, P)
```

**Impact:**
- `valid_predictor.py` line 124: `ego_safety_margin_loss` and `ego_neighbor_margin_loss`
  validation metrics are incorrect. The logged values during training are unreliable.
- `decoder.py` line 102: `compute_safety_penalty` is commented out, so training loss is
  NOT affected.
- The `lane_boundary_penalty` function in the same file does NOT have this bug (it uses
  `_lane_corner_clearance` which handles dimensions correctly).

**Severity:** Medium -- validation metrics only, no impact on trained weights.

---

## 2. `lane_keeping.py` -- boundary offset subtraction bug (FIXED in feat/rlvr-grpo)

**File:** `diffusion_planner/diffusion_planner/model/guidance/lane_keeping.py`, lines 89-90

**Bug:** Lane boundary data at indices 4-7 are offset vectors from centerline (not absolute
positions), but the code subtracts the center position again:
```python
# Was (wrong):
left_hw  = ((lb - c) * lat).sum(dim=-1)
right_hw = ((rb - c) * lat).sum(dim=-1)

# Fixed to:
left_hw  = (lb * lat).sum(dim=-1)
right_hw = (rb * lat).sum(dim=-1)
```

**Evidence:** Lane boundary values have magnitudes of 1.5-2.5m (half-lane-widths), while
center positions are at 5-15m from origin. `(lb - c)` produces vectors of 10-15m magnitude
instead of the correct 1.5-2.5m.

**Impact:**
- Lane keeping guidance energy (DPM-Solver sampling corrections) computed wrong boundary
  distances. Disabled by default in all configs due to known instability.
- LaneKeepingGuidance.reward() for DPO/GRPO scoring was incorrect.
- NOT used in training loss.

**Severity:** Low -- guidance was disabled by default. Fixed in feat/rlvr-grpo branch.

---

## Verification

Both bugs were discovered by comparing reward scores against visual trajectory plots in
the trajectory ranker GUI. The lane_keeping bug was confirmed by checking raw lane data:
```python
# Lane segment: center=(5.34, -1.79), left_boundary=(-0.74, 1.93), right_boundary=(0.74, -1.93)
# left_boundary is an offset vector (magnitude ~2.0m), not an absolute position
# (lb - c) = (-0.74 - 5.34, 1.93 + 1.79) = (-6.08, 3.72) -- clearly wrong
```

The neighbor_clearance_penalty bug was confirmed by testing individual NPCs vs combined:
each NPC individually showed no collision at t=0, but the combined call with 13 NPCs
reported collisions at t=0 due to the scrambled P/T indexing.
