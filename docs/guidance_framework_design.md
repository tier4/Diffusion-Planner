# Guidance Framework Design

## 1. The Problem with the Current Approach

The guidance system currently works but does not scale. Adding a new guidance function today requires touching at least five places:

1. Create `model/guidance/<new_fn>.py`
2. Add a boolean flag to `GuidanceWrapper.__init__`
3. Add that boolean to `generate_trajectory_pair()` in `preference_optimization/utils.py`
4. Add a checkbox to the DPO annotation GUI in `preference_optimization/annotation_gui.py`
5. Wire it through every caller all the way up to `train_dpo.py`

The signature of `generate_trajectory_pair` already has `use_collision`, `use_route_following`, `use_lane_keeping`, `use_centerline_following`. It will have more. This pattern breaks the moment guidance becomes something shared across three separate training pipelines (DPO, GRPO, playground).

A second problem: guidance currently only has an energy function (for steering diffusion sampling). For GRPO, you need a **reward function** - the same physical intuitions evaluated on a *completed* trajectory, not during the denoising process. Today these are the same code path but they shouldn't be, because:
- During sampling: gradients must flow, diffusion timestep `t` matters, the mask `t ∈ (0.005, 0.1)` applies
- For reward evaluation: no gradients needed, `t` is irrelevant, you just want a scalar quality score

---

## 2. Design Goals

1. **One place to add a guidance function.** Adding `my_new_guidance` means creating one file and one class. Zero changes to any consumer.
2. **Serializable config.** A guidance configuration is a JSON-able object that can be saved with every training run and reconstructed from it.
3. **Dual interface.** Every guidance class exposes both `energy()` (for diffusion sampling) and `reward()` (for DPO/GRPO evaluation) as separate, well-defined methods.
4. **No boolean explosion.** Consumers take a single `GuidanceSetConfig` object instead of N boolean flags.
5. **Backward compatibility.** Existing `GuidanceWrapper` keeps working unchanged. The new framework is additive; the old API is not removed until every consumer has been migrated.

---

## 3. Framework Components

### 3.1 `GuidanceConfig` - Per-Function Configuration

```python
# diffusion_planner/diffusion_planner/model/guidance/config.py

from dataclasses import dataclass, field
import json

@dataclass
class GuidanceConfig:
    """Serializable configuration for a single guidance function."""
    name: str              # Registry key, e.g. "collision", "anchor_following"
    enabled: bool = True
    scale: float = 1.0     # Multiplier applied to this function's energy/reward output
    params: dict = field(default_factory=dict)  # Function-specific parameters
    # Examples of params:
    #   anchor_following: {"prototypes_path": "...", "anchor_index": 3}
    #   (future) speed_limit:  {"max_speed_mps": 13.9}


@dataclass
class GuidanceSetConfig:
    """Full set of guidance functions for one experiment or inference call."""
    functions: list[GuidanceConfig] = field(default_factory=list)
    global_scale: float = 0.5   # Multiplies the total gradient correction in DPM-Solver

    def to_json(self) -> str:
        import dataclasses
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "GuidanceSetConfig":
        data = json.loads(s)
        data["functions"] = [GuidanceConfig(**f) for f in data["functions"]]
        return cls(**data)

    @classmethod
    def from_file(cls, path: str) -> "GuidanceSetConfig":
        return cls.from_json(open(path).read())

    def save(self, path: str) -> None:
        open(path, "w").write(self.to_json())

    def active_functions(self) -> list[GuidanceConfig]:
        return [f for f in self.functions if f.enabled]
```

A `GuidanceSetConfig` is saved alongside every DPO experiment's `dpo_args.json` and every GRPO run's config. When you reproduce a run, you reconstruct the exact guidance setup by loading that JSON.

### 3.2 `BaseGuidance` - Abstract Class for All Guidance Functions

```python
# diffusion_planner/diffusion_planner/model/guidance/base.py

from abc import ABC, abstractmethod
import torch


class BaseGuidance(ABC):
    """
    Base class for all guidance functions.

    Subclasses must define:
        name: ClassVar[str]   -- the registry key
        _energy_scale: float  -- internal normalization constant (see note below)

    And implement:
        _compute(x, inputs) -> torch.Tensor [B]

    The distinction between energy() and reward():
        energy()  -- called during DPM-Solver sampling. Gradients flow.
                     The diffusion timestep t is used to gate the gradient window.
        reward()  -- called on completed trajectories for DPO/GRPO scoring.
                     No gradients needed. t is irrelevant. Uses torch.no_grad().

    _energy_scale note:
        Each guidance function has a hardcoded internal scale (e.g., collision uses 300.0,
        lane_keeping uses 0.05). This internal scale sets the natural unit of the energy
        and should NOT be changed when adding new functions - it is calibrated so that
        the default guidance_scale=0.5 produces reasonable corrections.
        The user-facing `GuidanceConfig.scale` multiplier is applied on top of this.
    """

    name: str          # Subclass must set this as a class attribute
    _energy_scale: float = 1.0

    def __init__(self, config: "GuidanceConfig"):
        self.config = config

    @abstractmethod
    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        Core computation shared by energy() and reward().

        Args:
            x: [B, P, T+1, 4] - full trajectory in physical ego-centric meters.
               Index 0 along T+1 is the pinned current state (t=0).
               P includes ego (index 0) and neighbors (indices 1..Pn).
               In reward() mode, P=1 (ego only, no neighbors) so x[:, 1:] is empty.
               Guidance functions that depend on neighbors (e.g. collision) must
               guard against empty neighbor slices in reward() mode, or override
               reward() directly if neighbor data is required for meaningful scoring.
            inputs: observation dict already in physical units. In reward() mode,
               only keys that were explicitly passed by the caller are present.

        Returns:
            [B] raw (unscaled) reward tensor. Higher = better.
        """
        ...

    def energy(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        """
        For use during DPM-Solver sampling.

        x: [B, P, T+1, 4] - trajectory including current state at index 0,
                             already in physical ego-centric meters (state_normalizer.inverse applied).
        t: [B] diffusion timestep in [0, 1].

        Returns [B] energy. Gradient flows through x only when t ∈ (0.005, 0.1).
        """
        mask = (t < 0.1) * (t > 0.005)
        mask = mask.view(x.shape[0], 1, 1, 1)
        x_gated = torch.where(mask, x, x.detach())
        raw = self._compute(x_gated, inputs)
        return self._energy_scale * self.config.scale * raw

    @torch.no_grad()
    def reward(
        self,
        trajectory: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        """
        For use in DPO preference scoring and GRPO reward evaluation.

        trajectory: [B, T, 4] - completed ego trajectory in physical ego-centric meters.
                    [x, y, cos_yaw, sin_yaw] per timestep. No current state prepended.
                    No neighbors - P=1 in the wrapped tensor passed to _compute.

        inputs: observation dict in physical units. Must include any keys your
                _compute implementation reads. For ego-only guidance (route_following,
                lane_keeping, centerline_following, anchor_following) this just needs
                "lanes" and "route_lanes". For collision guidance, you need
                "neighbor_agents_past" to reconstruct neighbor positions; if that key
                is absent, CollisionGuidance.reward() returns zeros and logs a warning.

        Returns [B] scalar reward. Higher = better.
        """
        B, T, D = trajectory.shape
        # Prepend a zero current-state slot so _compute indices are consistent:
        # x[:, 0, 0, :] = current state (zeroed), x[:, 0, 1:, :] = future trajectory
        current_slot = torch.zeros(B, 1, 1, D, device=trajectory.device)
        x_padded = torch.cat([current_slot, trajectory.unsqueeze(1)], dim=2)  # [B, 1, T+1, 4]
        raw = self._compute(x_padded, inputs)
        return self._energy_scale * self.config.scale * raw
```

The separation of `_compute` from `energy`/`reward` means:
- Subclasses only implement physics logic once
- Gradient gating and no_grad context are handled by the base class
- The scale formula is consistent everywhere: `_energy_scale * config.scale * raw_output`

### 3.3 `GuidanceRegistry` - Discovery Without Hardcoding

```python
# diffusion_planner/diffusion_planner/model/guidance/registry.py

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base import BaseGuidance
    from .config import GuidanceConfig

_REGISTRY: dict[str, type["BaseGuidance"]] = {}


def register(cls):
    """Class decorator that registers a BaseGuidance subclass by its name."""
    assert hasattr(cls, "name"), f"{cls} must define a `name` class attribute"
    _REGISTRY[cls.name] = cls
    return cls


def build(config: "GuidanceConfig", **kwargs) -> "BaseGuidance":
    """Instantiate a guidance function from its config."""
    if config.name not in _REGISTRY:
        raise KeyError(
            f"Guidance '{config.name}' not found. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[config.name](config, **kwargs)


def list_available() -> list[str]:
    return sorted(_REGISTRY.keys())
```

### 3.4 `GuidanceComposer` - Replaces `GuidanceWrapper`

```python
# diffusion_planner/diffusion_planner/model/guidance/composer.py

import torch
from .config import GuidanceSetConfig
from .registry import build


class GuidanceComposer:
    """
    Drop-in replacement for GuidanceWrapper.

    Built from a GuidanceSetConfig. Compatible with the DPM-Solver's
    `classifier_fn` interface (same __call__ signature as GuidanceWrapper).
    """

    def __init__(self, set_config: GuidanceSetConfig, **build_kwargs):
        self._set_config = set_config
        self._functions = [
            build(fn_cfg, **build_kwargs)
            for fn_cfg in set_config.active_functions()
        ]

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """Same interface as GuidanceWrapper.__call__. Called by DPM-Solver."""
        state_normalizer = kwargs["state_normalizer"]
        observation_normalizer = kwargs["observation_normalizer"]

        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]

        # Apply x_start correction (identical to existing GuidanceWrapper logic)
        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x_in = x_in + x_fix.reshape(B, P, -1)

        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
        kwargs["inputs"] = observation_normalizer.inverse(kwargs["inputs"])

        energy = torch.zeros(B, device=x_in.device)
        for fn in self._functions:
            e = fn.energy(x_in, t_input, kwargs["inputs"])
            if torch.isnan(e).any():
                print(f"Warning: NaN energy from {fn.name}, skipping")
                continue
            energy = energy + e

        return energy

    def compute_rewards(
        self,
        trajectory: torch.Tensor,
        inputs: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Evaluate all active guidance functions as reward signals.

        Used by DPO (to score trajectory pairs) and GRPO (to score rollouts).

        trajectory: [B, T, 4] completed ego trajectory in physical ego-centric meters.
        inputs: observation dict in physical units.

        Returns:
            {
                "collision":          [B] tensor,
                "route_following":    [B] tensor,
                ...
                "total":              [B] tensor  (sum of all, respecting scales)
            }
        """
        rewards = {}
        total = torch.zeros(trajectory.shape[0], device=trajectory.device)
        for fn in self._functions:
            r = fn.reward(trajectory, inputs)
            rewards[fn.name] = r
            total = total + r
        rewards["total"] = total
        return rewards
```

`compute_rewards` is the entry point for DPO and GRPO. It returns a dict so callers can log per-function reward breakdowns, which is critical for debugging and understanding which guidance signals are driving the policy.

---

## 4. Refactoring Existing Guidance Functions

Each existing module-level function becomes a class. The old function name is kept as an alias for backward compatibility with any direct callers, but `GuidanceWrapper` should be migrated to `GuidanceComposer`.

**Pattern** (same for all four):

```python
# diffusion_planner/diffusion_planner/model/guidance/collision.py

from .base import BaseGuidance
from .registry import register

@register
class CollisionGuidance(BaseGuidance):
    name = "collision"
    _energy_scale = 300.0

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        # ... move existing collision_guidance_fn body here ...
        # x is [B, P, T+1, 4], inputs already in physical units
        # return [B] reward (existing function already returns this)
        ...

# Backward-compatible alias
def collision_guidance_fn(x, t, cond, inputs, *args, **kwargs):
    """Deprecated: use CollisionGuidance via GuidanceComposer."""
    from .config import GuidanceConfig
    fn = CollisionGuidance(GuidanceConfig(name="collision"))
    return fn.energy(x, t, inputs)
```

Same pattern for `RouteFollowingGuidance` (`_energy_scale = 0.05`), `LaneKeepingGuidance` (`_energy_scale = 0.05`), `CenterlineFollowingGuidance` (`_energy_scale = 0.1`).

**Collision guidance special case for `reward()`**: The existing `collision_guidance_fn` body reads `inputs["neighbor_current_mask"]` which is set by the Decoder at sampling time but will NOT be present when `reward()` is called externally. `CollisionGuidance` must override `reward()` to either reconstruct `neighbor_current_mask` from `inputs["neighbor_agents_past"]`, or return a zero tensor with a warning when neighbor data is absent:

```python
class CollisionGuidance(BaseGuidance):
    ...
    @torch.no_grad()
    def reward(self, trajectory, inputs):
        if "neighbor_agents_past" not in inputs:
            return torch.zeros(trajectory.shape[0], device=trajectory.device)
        # Reconstruct neighbor_current_mask from neighbor_agents_past
        neighbors_current = inputs["neighbor_agents_past"][:, :, -1, :4]
        neighbor_current_mask = (torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0)
        inputs_with_mask = {**inputs, "neighbor_current_mask": neighbor_current_mask}
        # Now build the full [B, P, T+1, 4] tensor including neighbors at t=0
        # ... (implementer should look at Decoder._prepare_current_states for reference)
        ...
```

**New anchor guidance** (see `guidance_playground/DESIGN.md` Section 4.1):

```python
@register
class AnchorFollowingGuidance(BaseGuidance):
    name = "anchor_following"
    _energy_scale = 0.05

    def __init__(self, config: GuidanceConfig):
        super().__init__(config)
        import numpy as np
        protos = np.load(config.params["prototypes_path"])    # (K, 80, 2)
        idx = config.params["anchor_index"]
        self._anchor = torch.tensor(protos[idx], dtype=torch.float32)  # (80, 2)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        B = x.shape[0]
        T = x.shape[2] - 1
        ego_pred = x[:, 0, 1:, :2]                              # [B, T, 2]
        anchor = self._anchor.to(x.device)[:T]                  # [T, 2]
        sq_dist = ((ego_pred - anchor.unsqueeze(0)) ** 2).sum(dim=-1)  # [B, T]
        return -sq_dist.sum(dim=-1)                              # [B]
```

---

## 5. How Each Consumer Uses the Framework

### 5.1 Guidance Playground

```python
# guidance_playground/app.py

from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.registry import list_available

# At app startup:
print(list_available())
# → ["anchor_following", "centerline_following", "collision", "lane_keeping", "route_following"]

# When the user clicks Generate:
set_config = GuidanceSetConfig(
    global_scale=float(global_scale_slider.value),
    functions=[
        GuidanceConfig("collision",         enabled=collision_toggle.value,     scale=float(collision_scale.value)),
        GuidanceConfig("lane_keeping",      enabled=lane_keeping_toggle.value,  scale=float(lk_scale.value)),
        GuidanceConfig("anchor_following",  enabled=anchor_toggle.value,        scale=float(anchor_scale.value),
                       params={"prototypes_path": PROTOTYPES_PATH, "anchor_index": selected_anchor_idx}),
    ],
)
composer = GuidanceComposer(set_config)
# Pass composer to generate_samples() which sets it on the decoder
samples = generate_samples(model, model_args, data, noise_scale, n_samples, composer, ...)
```

No boolean explosion. Adding a new guidance function to the UI is one new row in the list above.

### 5.2 DPO Pipeline

`GuidanceSetConfig` replaces all the individual `use_*` boolean parameters in `generate_trajectory_pair`.

**Before (current state)**:
```python
generate_trajectory_pair(
    model, args, data,
    enable_guidance=True,
    use_collision=True,
    use_route_following=False,
    use_lane_keeping=True,
    use_centerline_following=False,
    guidance_scale=0.5,
)
```

**After**:
```python
generate_trajectory_pair(
    model, args, data,
    guidance=GuidanceSetConfig(
        global_scale=0.5,
        functions=[
            GuidanceConfig("collision",    enabled=True,  scale=1.0),
            GuidanceConfig("lane_keeping", enabled=True,  scale=1.0),
        ]
    )
)
```

The `GuidanceSetConfig` is also saved to `dpo_args.json` via `set_config.to_json()`, so every DPO run records exactly which guidance produced its training pairs.

Additionally, `compute_rewards` can optionally be used in the DPO pipeline to score trajectories for reward-weighted sampling or logging, beyond the current FDE/ADE-based selection.

### 5.3 GRPO (Future)

The GRPO training loop will have two uses of the framework:

**During policy rollout (sampling)**:
```python
# Build a GuidanceComposer for diffusion guidance during sampling
sampling_guidance = GuidanceComposer(GuidanceSetConfig.from_file("grpo_sampling_guidance.json"))
model.decoder._guidance_fn = sampling_guidance
model.decoder._guidance_scale = sampling_guidance._set_config.global_scale
trajectories = run_policy(model, observations)   # produces N rollouts per scene
```

**For reward computation**:
```python
# Build a GuidanceComposer for reward evaluation (may differ from sampling guidance)
reward_guidance = GuidanceComposer(GuidanceSetConfig.from_file("grpo_reward_guidance.json"))

# Score each rollout
reward_dict = reward_guidance.compute_rewards(trajectories, observations)
# reward_dict["total"]: [B*N] combined reward signal
# reward_dict["collision"]: [B*N] collision component (for logging)

# Optionally add TeraSim simulation reward (from rlvr/terasim_bridge.py)
# sim_reward = terasim_bridge.get_reward(trajectories)
# total_reward = reward_dict["total"] + sim_reward
```

Two separate JSON configs (`grpo_sampling_guidance.json` and `grpo_reward_guidance.json`) allow different guidance during sampling vs evaluation - a common setup in RL where you may want diverse sampling (low guidance) but strict reward evaluation (high guidance).

---

## 6. File Layout

```
diffusion_planner/diffusion_planner/model/guidance/
├── __init__.py               # exports GuidanceComposer, GuidanceSetConfig, GuidanceConfig, register
├── base.py                   # BaseGuidance abstract class  [NEW]
├── config.py                 # GuidanceConfig, GuidanceSetConfig dataclasses  [NEW]
├── registry.py               # register decorator, build(), list_available()  [NEW]
├── composer.py               # GuidanceComposer class  [NEW]
├── collision.py              # CollisionGuidance class + backward-compat alias  [MODIFIED]
├── route_following.py        # RouteFollowingGuidance class + alias  [MODIFIED]
├── lane_keeping.py           # LaneKeepingGuidance class + alias  [MODIFIED]
├── centerline_following.py   # CenterlineFollowingGuidance class + alias  [MODIFIED]
├── anchor_following.py       # AnchorFollowingGuidance class  [NEW]
├── guidance_wrapper.py       # Unchanged, kept for backward compat  [UNCHANGED]
└── documentation_guidance.md # Updated to reflect new framework  [MODIFIED]
```

### What to update in the guidance module `__init__.py`

```python
from .composer import GuidanceComposer
from .config import GuidanceConfig, GuidanceSetConfig
from .registry import register, build, list_available

# Force all guidance classes to register themselves by importing the modules
from . import collision, route_following, lane_keeping, centerline_following, anchor_following
```

This import-time registration pattern means every module in the `guidance/` directory is self-registering. No central switch-case list to maintain.

---

## 7. Migration Plan

The migration is designed to be non-breaking. The old `GuidanceWrapper` and the old boolean-parameter API in `generate_trajectory_pair` continue to work throughout. Migration happens file by file.

**Step 1**: Add `base.py`, `config.py`, `registry.py`, `composer.py`. Zero changes to existing files. Tests should all pass.

**Step 2**: Convert guidance functions to classes (add class, keep function alias). The existing `GuidanceWrapper` still uses the function aliases, so nothing breaks.

**Step 3**: Add `anchor_following.py` (pure addition, no existing code touched).

**Step 4**: Update `guidance_playground/app.py` and `generate_samples.py` to use `GuidanceComposer` and `GuidanceSetConfig`. The playground is new code so there's no migration cost here.

**Step 5**: Update `preference_optimization/utils.py` - migrate `generate_trajectory_pair` to accept `GuidanceSetConfig`. The new signature must remain backward compatible:

```python
def generate_trajectory_pair(
    policy_model,
    model_args,
    data,
    noise_scale=2.5,
    fde_threshold=2.0,
    ade_threshold=1.0,
    max_retries=50,
    device=None,
    gt_similarity_mode=True,
    gt_trajectory=None,
    enable_initial_pruning=True,
    initial_pos_threshold=0.055,
    initial_yaw_threshold_deg=0.55,
    # New unified parameter:
    guidance: GuidanceSetConfig | None = None,
    # Deprecated - kept for backward compat, ignored if guidance is not None:
    enable_guidance: bool = False,
    use_collision: bool = True,
    use_route_following: bool = False,
    use_lane_keeping: bool = False,
    use_centerline_following: bool = False,
    guidance_scale: float | None = None,
):
    # If new-style guidance config provided, use it directly.
    # Otherwise, build one from the legacy booleans for backward compat.
    if guidance is None and enable_guidance:
        import warnings
        warnings.warn("Boolean guidance flags are deprecated. Use GuidanceSetConfig.", DeprecationWarning)
        from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
        guidance = GuidanceSetConfig(
            global_scale=guidance_scale or 0.5,
            functions=[
                GuidanceConfig("collision",            enabled=use_collision),
                GuidanceConfig("route_following",      enabled=use_route_following),
                GuidanceConfig("lane_keeping",         enabled=use_lane_keeping),
                GuidanceConfig("centerline_following", enabled=use_centerline_following),
            ]
        )
    ...
```

Inside the function body, wherever the current code builds a `GuidanceWrapper` instance, replace it with `GuidanceComposer(guidance)` if `guidance is not None`, else `None`.

**Step 6**: Update `preference_optimization/annotation_gui.py` to build a `GuidanceSetConfig` from its UI controls and pass it to `generate_trajectory_pair`.

**Step 7** (future, when GRPO is implemented): Use `compute_rewards` in the GRPO reward loop directly.

**Step 8** (cleanup, after all consumers migrated): Remove deprecated boolean parameters from `generate_trajectory_pair`, remove `GuidanceWrapper`.

---

## 8. How to Add a New Guidance Function (One Place, Zero Consumer Changes)

After the framework is in place, adding a new guidance function is:

1. Create `diffusion_planner/diffusion_planner/model/guidance/my_guidance.py`:

```python
import torch
from .base import BaseGuidance
from .registry import register

@register
class MyGuidance(BaseGuidance):
    name = "my_guidance"
    _energy_scale = 0.1   # calibrate so default guidance_scale=0.5 is reasonable

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        # x: [B, P, T+1, 4] physical ego-centric meters, index 0 = current state
        # inputs: observation dict in physical units
        # return [B] reward (higher = better trajectory)
        ego_xy = x[:, 0, 1:, :2]  # [B, T, 2]
        ...
        return reward  # [B]
```

2. Add the import to `__init__.py`:

```python
from . import my_guidance   # triggers @register
```

That's it. The playground UI, DPO pipeline, and GRPO reward loop can immediately use `"my_guidance"` as a string key in their `GuidanceSetConfig`. No other files need to change.

## Batched guidance v2 set (rlvr/guidance_batched.py)

> **Envelope is persisted with the policy.** The guidance envelope (the
> `lambda_lat` / `lat_scale` / `col_scale` / ... knobs + the `v1`/`v2` family)
> is the calibration the explorer's eta labels were swept against, so it is
> stored in `ExplorationPolicyConfig.guidance_envelope` and the guided
> eval/deploy tools load it from the policy (CLI flags are override-only and
> hard-fail on a disagreeing value unless `--force_envelope_override`). See
> "Guidance envelope" in `rlvr/autoresearch/tools/README.md`. The `--envelope`
> selection below is one such knob; a v2 policy persists `envelope: "v2"`.

Opt-in registry names (the v1 functions are unchanged; select via
`--envelope v2` in `sweep_guidance_params` / `eval_policy_avoidance` /
`valid_predictor_guided`, or `envelope="v2"` in
`guidance_batched.build_head_composer`):

- `lateral_ramp_batched` — Eq.2 lateral energy with a linear feasibility
  ramp on the target (0 -> lambda*eta over `ramp_steps`, default 20). The
  un-ramped target demands the full offset at the first future step, which
  concentrates gradient on the plan head and distorts it.
- `collision_swerve_v2_batched` — bounded-target swerve. Proximity gate is
  computed from the (detached) reference trajectory (the current noisy x is
  garbage early in denoising); the target profile is the cummax of the gate
  (ramps up on approach, HOLDS after passing); energy is a plain mean-over-T
  quadratic. Design notes from probe iterations: the v1 linear energy has a
  scene-dependent endpoint gain (it pushes forever, scaled by in-range step
  count); support-normalized quadratics concentrate curvature on few steps
  and diverge under DPM guidance steps; bump-shaped targets (return-to-zero
  mid-pass) fight themselves and sign-invert at useful scales. Full-horizon
  / held targets with mean-over-T energies are the stable family.

Calibration guidance: response-curve probes (eta sweep -> realized lateral
displacement and clearance) are the acceptance test for any new guidance —
require monotonic, side-symmetric, scene-consistent curves before use.
Larger lambda is not "stronger avoidance": past the corridor width it
trades obstacle clearance for road-border violations.
