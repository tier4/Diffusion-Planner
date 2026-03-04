# Guidance Playground Design Document

## 1. Goal and Scope

A Gradio-based interactive tool for exploring the Diffusion Planner's guidance system. The key behavioral difference from the DPO annotation GUI is:

**DPO GUI**: noise_scale and guidance are separated. `traj_1` = deterministic (near-zero noise), `traj_2` = stochastic (noise + optional guidance). They are compared as a pair.

**This playground**: noise_scale and guidance are applied together to every sample. There is no pairing. The user sets `noise_scale = 3.0` and `collision guidance scale = 1.5` simultaneously and sees N independent samples, all generated under that combined configuration. The goal is to understand how noise diversity and guidance energy interact.

A second new feature is the **MTR-style anchor guidance**: pre-clustered prototype trajectories extracted from training data, selectable as guidance targets via a visual gallery in the UI.

---

## 2. Existing Guidance System: Quick Reference

All guidance functions live in `diffusion_planner/diffusion_planner/model/guidance/`.

| Function | File | Internal scale | What it does |
|---|---|---|---|
| `collision_guidance_fn` | `collision.py` | `300.0` | Penalises ego-NPC bounding box overlap |
| `route_following_fn` | `route_following.py` | `0.05` | Minimises distance from ego to any route lane point |
| `lane_keeping_fn` | `lane_keeping.py` | `0.05` | Penalises ego footprint protruding past lane boundaries |
| `centerline_following_fn` | `centerline_following.py` | `0.1` | Quadratic penalty for lateral deviation from centerline |

All functions receive `x: [B, P, T+1, 4]` already in **physical ego-centric meters** (guidance_wrapper applies `state_normalizer.inverse` before calling them). All return `[B]` energy.

`GuidanceWrapper.__call__` sums the energies. The summed energy is passed to `dpm.model_wrapper` which multiplies its gradient by `guidance_scale` (a single global float from `Config`, default 0.5). So the final influence on sampling is:

```
score_corrected = score + guidance_scale * gradient(sum of all guidance energies)
```

Guidance is active only during the DPM-Solver sampling phase, and each guidance function gates its gradient to `t ∈ (0.005, 0.1)` via a mask to avoid instability.

Noise scale controls the magnitude of the initial noise tensor `xT` fed to the DPM-Solver. It is not currently a parameter of the model itself but is applied in the trajectory generation call before passing to the decoder.

---

## 3. New Feature: MTR-Style Anchor Guidance

### 3.1 Concept

MTR (Motion Transformer) maintains a file of K prototype trajectories obtained by K-means clustering of the training set. Each prototype is a representative shape for a common motion mode (straight, gentle left, sharp right, etc.) in the vehicle's own frame.

This project will adopt the same idea: generate a `prototypes.npy` file once from the training data, then use any selected prototype as a soft guidance target during diffusion sampling.

The guidance function computes:

```
reward = -sum_t ||ego_pred[t, :2] - anchor[t, :2]||^2
```

This is a distance-minimizing energy: the diffusion is guided toward trajectories whose shape matches the chosen prototype, while all other scene constraints (lanes, route, etc.) continue to apply if their guidance is also enabled.

### 3.2 What "Prototype" Means Here

Prototypes are **shape templates** in the ego-centric frame - they encode curvature and speed profile, not absolute position. A "turn right" prototype has its final point to the right of the origin regardless of where in the map the ego is. This is exactly the right abstraction because the Diffusion Planner always operates in base_link frame.

Prototypes are extracted from `ego_agent_future` (80, 3) → first 2 columns (x, y). Clustering is done on the flattened (80*2,) = 160-dimensional representation. K = 16 is the recommended starting value (matches MTR's vehicle motion mode count), but 32 can be tried.

### 3.3 Prototype Generation Script

File: `guidance_playground/scripts/generate_prototypes.py`

```
python guidance_playground/scripts/generate_prototypes.py \
  --npz_list  /path/to/train.json \
  --k         16 \
  --output    guidance_playground/prototypes_k16.npy \
  --max_samples 50000
```

Output: `prototypes_k16.npy` of shape `(K, 80, 2)`, all in ego-centric frame (meters). Also saves `prototypes_k16_counts.npy` of shape `(K,)` with how many training samples mapped to each cluster, so the UI can show cluster frequency.

Implementation notes:
- Load at most `--max_samples` npz files at random
- Extract `data["ego_agent_future"][:, :2]` (the xy columns from the (80, 3) array)
- Flatten to (N, 160), run `sklearn.cluster.KMeans(n_clusters=k, n_init=10)`
- Sort clusters by frequency (descending) so prototype 0 = most common motion
- Save cluster centers reshaped to `(K, 80, 2)`

---

## 4. Code Changes Required

**The playground uses the guidance framework described in `GUIDANCE_FRAMEWORK_DESIGN.md`.** Read that document first. This section only describes what is specific to the playground.

The `AnchorFollowingGuidance` class is defined in `diffusion_planner/diffusion_planner/model/guidance/anchor_following.py` following the `BaseGuidance` interface from `GUIDANCE_FRAMEWORK_DESIGN.md` Section 4. Its `_compute` implementation:

```python
def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
    # x: [B, P, T+1, 4], index 0 along T+1 is the pinned current state
    ego_pred = x[:, 0, 1:, :2]                                  # [B, T, 2]
    T = ego_pred.shape[1]
    anchor = self._anchor.to(x.device)[:T]                       # [T, 2]
    sq_dist = ((ego_pred - anchor.unsqueeze(0)) ** 2).sum(-1)    # [B, T]
    return -sq_dist.sum(-1)                                      # [B]
```

No changes to `guidance_wrapper.py` or `decoder.py` are needed for the playground. The playground uses `GuidanceComposer` directly, which is set on the model before each inference call (see Section 6).

---

## 5. Playground App

### 5.1 File Location

`guidance_playground/app.py`

### 5.2 Launch

```bash
source .venv/bin/activate
cd guidance_playground
python app.py \
  --model_path  /path/to/model.pth \
  --npz_list    /path/to/train_or_valid.json \
  --prototypes  guidance_playground/prototypes_k16.npy
```

### 5.3 UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Guidance Playground          Sample 42 / 8317   [Shuffle]   [Resample]     │
├─────────────────────────┬───────────────────────────────────────────────────┤
│  LEFT: Controls         │  RIGHT: Outputs                                   │
│                         │                                                   │
│  ── Navigation ──       │  ┌───────────────────────────────────────────┐   │
│  [|←-30][←-10][←-1]    │  │  Trajectory Plot                          │   │
│  [+1→][+10→][+30→|]     │  │  (N colored samples + GT dashed +         │   │
│  Jump to: [____] Enter  │  │   anchor prototype dotted)                │   │
│                         │  └───────────────────────────────────────────┘   │
│  ── Generation ──       │                                                   │
│  Noise Scale [0.0-5.0]  │  ┌─────────────────┐  ┌───────────────────────┐  │
│  [ ] Deterministic      │  │  Velocity        │  │  Lateral Curvature    │  │
│  N Samples   [1-8]      │  │  (N + GT)        │  │  (N + GT)             │  │
│  Zoom Level  [1-10]     │  └─────────────────┘  └───────────────────────┘  │
│  Time Step   [0-80]     │                                                   │
│                         │  ── Prototype Gallery ────────────────────────   │
│  ── Guidance ──         │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐      │
│  Global Scale [0.1-3.0] │  │0 │ │1 │ │2*│ │3 │ │4 │ │5 │ │6 │ │7 │      │
│  [x] Collision          │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘      │
│      scale [0.1-5.0]    │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐      │
│  [ ] Route Following    │  │8 │ │9 │ │10│ │11│ │12│ │13│ │14│ │15│      │
│      scale [0.1-5.0]    │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘      │
│  [ ] Lane Keeping       │           (* = selected anchor)                  │
│      scale [0.1-5.0]    │                                                   │
│  [ ] Centerline         │  ── Stats ──────────────────────────────────────  │
│      scale [0.1-5.0]    │  min ADE/GT: X.XX m   max ADE/GT: X.XX m        │
│  [ ] Anchor             │  ADE/anchor: X.XX m   spread FDE std: X.XX m    │
│      scale [0.1-5.0]    │                                                   │
│                         │  Status: [Generating... / Ready]                 │
└─────────────────────────┴───────────────────────────────────────────────────┘
```

### 5.4 Component Details

**NPZ file browser and navigation**:
- The app takes `--npz_list` pointing to a `path_list.json` (same format as the DPO GUI and training pipeline)
- Current position shown as "Sample 42 / 8317"
- Navigation buttons: `|← -30`, `← -10`, `← -1`, `+1 →`, `+10 →`, `+30 →`
- Jump-to-index: number input that loads an arbitrary sample on `Enter`
- "Shuffle" button randomizes the loaded list order so repeated sessions explore different samples first

**Automatic regeneration - no Generate button**:
- Every parameter control triggers regeneration automatically when its value changes
- Sliders use Gradio's `.release()` event (fires once when the user releases, not on every intermediate drag position) to avoid flooding the GPU during slow drags
- Checkboxes, dropdowns, and anchor gallery clicks use `.change()` which fires immediately
- Navigation buttons also trigger generation - loading a new sample implicitly regenerates
- A status indicator shows "Generating..." while inference is running so the user knows their change was registered
- No manual "Regenerate" button exists by design. If the user wants a new stochastic draw with identical parameters, a small "Resample" button (distinct from navigation) can serve that purpose without implying the user needs to press it normally

**Noise Scale + Guidance (unified)**:
- Every one of the N samples is generated with the same `noise_scale` and the same guidance configuration
- `noise_scale` controls the magnitude of `xT = current_states + noise_scale * randn(...)` fed to DPM-Solver
- Guidance is applied inside the same DPM-Solver call
- A **"Deterministic (noise=0)"** checkbox zeroes the noise entirely: all N samples share the same `xT = current_states` (no randomness), so any trajectory variation comes purely from guidance. This isolates the guidance effect from the stochastic baseline. With noise=0 and no guidance, all N samples are identical (the model's MAP output).

**Per-guidance scales**:
- Each guidance toggle enables/disables that function
- The scale slider for each guidance multiplies that function's output via `per_fn_scales` (Section 4.2)
- Guidance scale values are interpreted as multipliers on top of the function's internal `_SCALE` constant (e.g., collision's internal `300.0`). A slider value of `1.0` = default behavior, `2.0` = double strength
- The global `Config.guidance_scale` is exposed as a separate "Global Scale" slider (multiplies the total gradient correction from all guidance functions combined)

**Prototype Gallery**:
- Each cell shows a small matplotlib figure of the prototype trajectory, with the cluster index and member count (e.g., "Mode 2 (n=4812)")
- Clicking a cell sets the selected anchor index
- The selected cell is highlighted (different border color)
- The anchor is shown as a dotted line in the main trajectory plot when anchor guidance is enabled
- Implemented as a `gr.Gallery` component populated with pre-rendered thumbnail images at app startup

**N Samples**:
- All N trajectories are generated sequentially (or in a batched call if batch_size > 1) using the current noise_scale + guidance config
- Each sample gets a distinct color
- GT trajectory shown as dashed gray
- Selected anchor prototype shown as dotted orange (only when anchor guidance is enabled)

**Stats row**:
- `min ADE to GT`: min over N samples of mean L2 distance to GT (how close is the closest sample to GT?)
- `max ADE to GT`: max over N samples (how far is the furthest?)
- `ADE to anchor`: mean over N samples of mean L2 to selected anchor (how well did guidance pull toward the anchor?)
- `spread (std FDE)`: std of final positions across N samples (how diverse are the samples?)

---

## 6. How Noise Scale Interacts with Guidance (Implementation Detail)

Do **not** reuse `generate_trajectory_pair` from `preference_optimization/utils.py` - it has DPO-specific pairing, threshold, and pruning logic. The playground needs a focused `generate_samples` function in `guidance_playground/generate_samples.py`.

### `generate_samples` signature

```python
@torch.no_grad()
def generate_samples(
    model,
    model_args,
    data: dict[str, torch.Tensor],
    noise_scale: float,           # 0.0 = deterministic, >0 = stochastic
    n_samples: int,
    composer: "GuidanceComposer | None",  # None = no guidance
    device: torch.device,
) -> np.ndarray:
    """
    Returns: (n_samples, OUTPUT_T, 4) float32 array.
    Each row: [x, y, cos_yaw, sin_yaw] in ego-centric meters.
    """
```

### Implementation notes for `generate_samples`

**Step 1 - Set guidance on the decoder before each call:**
```python
model.decoder._guidance_fn = composer        # GuidanceComposer is callable, same interface as GuidanceWrapper
model.decoder._guidance_scale = composer._set_config.global_scale if composer else 0.5
```

**Step 2 - Build `sampled_trajectories` (the noisy initial state):**

`SAMPLED_TRAJECTORIES_SHAPE = (1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM)` from `dimensions.py`.
`MAX_NUM_AGENTS = 33`, `OUTPUT_T = 80`, `POSE_DIM = 4`.

The existing pattern from `generate_trajectory_pair` in `preference_optimization/utils.py` (look at how it builds `sampled_trajectories` before calling the model):
```python
# Extract current states for ego + predicted neighbors
P = 1 + model_args.predicted_neighbor_num
ego_current = data["ego_current_state"][:, :4]                          # [B, 4]
neighbors_current = data["neighbor_agents_past"][:, :P-1, -1, :4]      # [B, P-1, 4]
current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

# Sample noise and build xT
xT = current_states[:, :, None, :].expand(-1, -1, OUTPUT_T + 1, -1).clone()
xT[:, :, 1:, :] = noise_scale * torch.randn(B, P, OUTPUT_T, 4, device=device)

data["sampled_trajectories"] = xT.reshape(B, P, -1)
```

**Step 3 - Call the model:**
```python
_, decoder_output = model(data)
# For x_start model: decoder_output["prediction"] shape [B, P, OUTPUT_T, 4]
ego_trajectory = decoder_output["prediction"][:, 0]  # [B, OUTPUT_T, 4]
```

**Step 4 - Loop n_samples times** (each call with a freshly sampled `xT`), collect results, stack to `(n_samples, OUTPUT_T, 4)` and convert to numpy.

### How `GuidanceComposer` integrates with `Decoder._inference_x_start`

`GuidanceComposer.__call__` has the same signature as `GuidanceWrapper.__call__` (both take `x_in, t_input, cond, *args, **kwargs`). The Decoder reads `self._guidance_fn` and passes it to `dpm.model_wrapper` as `classifier_fn`. Setting `model.decoder._guidance_fn = composer` before calling `model(data)` is all that is needed - no other Decoder changes required.

---

## 7. File Layout

```
guidance_playground/
├── DESIGN.md                        # this document
├── app.py                           # Gradio application entry point
├── generate_samples.py              # focused inference wrapper (Section 6)
├── visualization.py                 # trajectory/velocity/lateral plots
├── prototypes_k16.npy               # generated by scripts/generate_prototypes.py
├── prototypes_k16_counts.npy        # cluster member counts
└── scripts/
    └── generate_prototypes.py       # K-means clustering of training trajectories

New files in existing packages (see GUIDANCE_FRAMEWORK_DESIGN.md for full spec):
  diffusion_planner/diffusion_planner/model/guidance/
    base.py        [NEW]
    config.py      [NEW]
    registry.py    [NEW]
    composer.py    [NEW]
    anchor_following.py  [NEW]
    collision.py         [MODIFIED - add CollisionGuidance class, keep function alias]
    route_following.py   [MODIFIED - add RouteFollowingGuidance class, keep alias]
    lane_keeping.py      [MODIFIED - add LaneKeepingGuidance class, keep alias]
    centerline_following.py  [MODIFIED - add CenterlineFollowingGuidance class, keep alias]
    __init__.py          [MODIFIED - export framework, import all modules to trigger registration]
    guidance_wrapper.py  [UNCHANGED]

  preference_optimization/utils.py   [MODIFIED - migrate generate_trajectory_pair, see FRAMEWORK §7 Step 5]
  preference_optimization/annotation_gui.py  [MODIFIED - build GuidanceSetConfig from UI controls]
```

New pip dependency (add to `diffusion_planner/requirements.txt`):
```
scikit-learn   # for KMeans in generate_prototypes.py
```
`gradio>=4.0.0` is already present.

---

## 8. Implementation Order

Implement the framework first (Steps 1-3), then the playground on top of it (Steps 4-7). The framework steps are specified in detail in `GUIDANCE_FRAMEWORK_DESIGN.md` Section 7.

**Step 1** (Framework): Add `base.py`, `config.py`, `registry.py`, `composer.py` with zero changes to existing files. `python -c "from diffusion_planner.model.guidance import GuidanceComposer"` must succeed.

**Step 2** (Framework): Convert all four existing guidance functions to classes, keeping module-level aliases. `python -c "from diffusion_planner.model.guidance.registry import list_available; print(list_available())"` must print all four names.

**Step 3** (Framework): Add `anchor_following.py`. Validate `AnchorFollowingGuidance._compute` by running it with a prototype tensor and checking gradient flows toward the anchor.

**Step 4** (Playground prep): Run `generate_prototypes.py` to produce `prototypes_k16.npy`. Plot all 16 prototypes and confirm visual diversity.

**Step 5** (Playground core): Implement `generate_samples.py`. Test: `noise_scale=0` → all N outputs identical; `noise_scale=2.5` → outputs visibly diverse.

**Step 6** (Playground core): Implement `visualization.py`. Reuse patterns from `preference_optimization/annotation_gui.py`'s `_create_trajectory_plot`, `_create_velocity_plot`, `_create_lateral_curvature_plot` methods.

**Step 7** (Playground UI): Implement `app.py` incrementally:
- First: navigation + trajectory plot only (auto-regenerates on nav button click)
- Then: noise scale slider + deterministic checkbox (`.release()` / `.change()` events)
- Then: guidance toggles + per-scale sliders
- Then: prototype gallery (`gr.Gallery` with pre-rendered thumbnails; clicking triggers anchor index state update which triggers regeneration)
- Then: stats row

---

## 9. Acceptance Criteria

| Test | Expected Result |
|---|---|
| `noise_scale=0.01`, all guidance off, N=4 | All 4 samples nearly identical (near-deterministic) |
| Deterministic checkbox on, guidance off, N=4 | All 4 samples exactly identical (MAP output) |
| Deterministic checkbox on, collision guidance (scale=1.0), N=4 | All 4 samples identical to each other but shifted from the no-guidance MAP output, proving guidance alone changes the trajectory |
| `noise_scale=3.0`, all guidance off, N=4 | Samples visibly diverse, spread across plausible trajectories |
| `noise_scale=2.0`, collision guidance only (scale=1.0), N=4 | Samples avoid NPCs more than unguided equivalents |
| `noise_scale=2.0`, anchor guidance (scale=2.0, prototype=sharp_right), N=4 | Mean ADE to selected anchor < mean ADE to all other anchors |
| `noise_scale=2.0`, anchor + lane_keeping, N=4 | Trajectories biased toward anchor shape but stay within lane boundaries |
| Prototype gallery | 16 thumbnails rendered, clicking highlights selection, anchor shown in main plot |

---

## 10. Design Rationale: Why the DPO GUI Uses noise_scale = 0 for the Guided Trajectory

The DPO annotation GUI intentionally fixes noise to near-zero for the trajectory that receives
guidance. High noise scales introduce instability in the first predicted waypoints — the model's
output at step t=0 becomes erratic when the initial noise tensor is large. When trajectories
generated under high noise are then used as "preferred" examples in DPO training, the model
learns to replicate that erratic behavior, causing successive training iterations to produce
increasingly noisy predictions.

To avoid conflating guidance signal with noise-induced artifacts, the DPO GUI's guided
trajectory is generated deterministically (noise_scale ≈ 0). Guidance alone then shifts the
trajectory — cleanly attributable to the guidance function and not to sampling noise. The
unguided baseline (`traj_1`) remains stochastic to retain diversity in the preference pair.

The Guidance GUI lifts this restriction intentionally: it is an exploration tool, not a
training data generator. Both noise_scale and guidance can be varied simultaneously to study
their interaction.
