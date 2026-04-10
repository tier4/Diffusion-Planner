"""JSON-based GRPO configuration.

Supports two modes:
- On-policy (M=1): Single gradient step per rollout, no importance sampling.
- Multi-epoch (M>1): Reuse rollouts for M inner epochs with PPO-clipped
  importance sampling for higher sample efficiency.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class GRPOConfig:
    """Complete GRPO training configuration.

    Attributes:
        num_generations: N trajectories to sample per scene.
        train_epochs: Total number of outer training epochs.
        inner_epochs: M gradient steps on the same batch before regenerating.
            M=1 is pure on-policy. M>1 reuses rollouts with importance sampling.
        ppo_clip_epsilon: PPO clipping range for importance sampling ratio.
            Only active when inner_epochs > 1.
        kl_coef: Coefficient for KL divergence against fixed SFT reference.
        learning_rate: AdamW learning rate.
        grad_accum_groups: Number of groups (scenes) to accumulate before
            stepping the optimizer.
        sampling_randomization: Whether to randomize noise_scale and guidance
            per trajectory (True) or use fixed configs (False).
        noise_scale_range: (min, max) noise scale for stochastic trajectories.
        guidance_scale_range: (min, max) global guidance scale.
        enable_guidance: Master guidance toggle.
        enable_centerline: Include centerline guidance in random pool.
        enable_anchor: Include anchor guidance in random pool.
        enable_collision: Include collision guidance in random pool.
        enable_route_following: Include route-following guidance in random pool.
        enable_lane_keeping: Include lane-keeping guidance in random pool.
        guidance_prob: Per-type coin-flip probability.
        prototypes_path: Path to anchor prototypes .npy (null to disable anchor).
        w_safety: Reward weight for collision/proximity.
        w_progress: Reward weight for goal progress.
        w_smooth: Reward weight for jerk penalty.
        w_feasibility: Reward weight for lane/acceleration feasibility.
        w_centerline: Reward weight for centerline following.
        use_lora: Whether to use LoRA adapters.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
    """

    # Core GRPO
    num_generations: int = 32
    train_epochs: int = 10
    inner_epochs: int = 1
    ppo_clip_epsilon: float = 0.2
    kl_coef: float = 0.1

    # Optimizer
    learning_rate: float = 1e-5
    grad_accum_groups: int = 4

    # Sampling
    sampling_randomization: bool = True
    noise_scale_range: list[float] = field(default_factory=lambda: [0.5, 2.0])
    guidance_scale_range: list[float] = field(default_factory=lambda: [0.1, 2.0])
    enable_guidance: bool = True
    enable_centerline: bool = True
    enable_anchor: bool = True
    enable_collision: bool = False
    enable_route_following: bool = False
    enable_lane_keeping: bool = False
    enable_road_border: bool = True
    enable_speed: bool = True
    enable_lateral: bool = False
    enable_longitudinal: bool = False
    lambda_lat: float = 2.5    # max lateral offset in metres (PlannerRFT Eq. 2)
    lambda_lon: float = 0.25   # max speed deviation fraction (PlannerRFT Eq. 3)
    guidance_prob: float = 0.5
    prototypes_path: str | None = None

    # Reward weights
    w_safety: float = 5.0
    w_progress: float = 2.0
    w_smooth: float = 0.5
    w_feasibility: float = 5.0
    w_centerline: float = 5.0

    # Reward tuning (passed to RewardConfig for training)
    # Road border penalty scales and thresholds
    rb_near_scale: float = 3.0
    rb_wide_scale: float = 0.2
    rb_cont_scale: float = 0.0
    rb_gate_enabled: bool = True   # if True, rb crossing is a hard safety gate
    rb_penalty_mode: str = "frac"  # "frac" = fraction of timesteps (original), "survival" = first-violation time-decay
    rb_cross_thresh: float = 0.20  # metres — ego perimeter within this = crossing
    rb_near_thresh: float = 0.45   # metres — near zone boundary (+20cm vs lane)
    rb_wide_thresh: float = 0.60   # metres — wide zone boundary (+20cm vs lane)
    rb_cont_thresh: float = 1.00   # metres — continuous penalty max distance (+20cm vs lane)
    # Lane departure penalty scales and thresholds
    enable_lane_departure: bool = False
    lane_gate_enabled: bool = False
    lane_near_scale: float = 3.0
    lane_wide_scale: float = 0.2
    lane_cont_scale: float = 0.0
    lane_near_thresh: float = 0.25  # metres — near zone boundary
    lane_wide_thresh: float = 0.40  # metres — wide zone boundary
    lane_cont_thresh: float = 0.80  # metres — continuous penalty max distance
    max_lat_accel: float = 2.0
    lat_accel_scale: float = 3.0
    enable_overprogress: bool = True
    overprogress_margin: float = 1.1
    overprogress_penalty: float = 0.3
    stopped_penalty: float = 5.0
    underprogress_penalty: float = 0.0  # penalize model driving << GT (0=disabled)
    underprogress_threshold: float = 0.5  # penalize if model_path/gt_path < threshold
    progress_norm_scale: float = 20.0  # max progress points at 100% GT match

    # Reward aggregation mode (passed to RewardConfig):
    # "gate": binary safety gates (default). Any terminal event → floor.
    # "survival": PlannerRFT-style proportional credit. Crash at t=60/80 gets
    #   75% quality score. Prevents gradient death on hard scenes.
    reward_mode: str = "gate"

    # Loss mode: controls how gradients flow to affect the deterministic output.
    # "diffusion" (default): standard advantage-weighted diffusion loss at random t.
    # "direct_best": regress the model's deterministic output toward the best-in-group
    #     trajectory via MSE, bypassing diffusion timestep sampling entirely.
    # "diffusion_low_t": sample t from a narrow range near 0 where denoising is
    #     closest to the final clean output.
    # "diffusion_multistep": average loss over K timesteps spread across the schedule
    #     for better coverage of the diffusion probability.
    loss_mode: str = "diffusion"
    direct_loss_weight: float = 1.0
    diffusion_t_range: list[float] = field(default_factory=lambda: [0.001, 0.1])
    diffusion_k_steps: int = 8  # K (noise, t) samples averaged per GRPO loss (matches DPO K=8)

    # DiT GRPO loss type:
    #   "advantage_mse" (default): advantage-weighted MSE diffusion loss.
    #   "advantage_logprob": advantage-weighted Gaussian log-probabilities from
    #       a truncated denoising rollout (DDV2-style). Enables proper policy
    #       gradient for trajectory shape.
    grpo_loss_type: str = "advantage_mse"
    logprob_num_steps: int = 10  # denoising steps for rollout
    logprob_t_start: float = 0.01  # starting noise level (truncated)
    logprob_discount: float = 0.8  # per-step advantage discount (DDV2 uses 0.8)
    logprob_min_std: float = 0.1  # minimum std for log-prob stability (DDV2 supplementary)
    il_loss_weight: float = 0.1  # IL (imitation learning) regularization weight
    il_adaptive: bool = True  # if True: IL weight=1.0 when no positive advantages, else il_loss_weight

    # Advantage computation mode:
    # "normalized" (default): standard GRPO per-group normalization (mean=0, std=1).
    # "vd_grpo": Variance-Decoupled GRPO (Plan-R1). Center only (subtract mean),
    #   divide by a fixed scale instead of per-group std. Preserves absolute
    #   magnitude of negative rewards (e.g. crashes) across groups.
    advantage_mode: str = "normalized"
    advantage_fixed_scale: float = 10.0

    # KL coefficient scheduling over training epochs.
    # "constant" (default): kl_coef stays fixed.
    # "linear": linearly interpolate from kl_coef to kl_coef_final.
    # "cosine": cosine annealing from kl_coef to kl_coef_final.
    # "step": hold kl_coef for kl_warmup_fraction of training, then drop to kl_coef_final.
    kl_schedule: str = "constant"
    kl_coef_initial: float | None = None  # set automatically from kl_coef at first call
    kl_coef_final: float = 0.01
    kl_warmup_fraction: float = 0.5  # fraction of epochs to hold initial kl_coef (for "step" schedule)

    # Rejection sampling: generate num_generations trajectories but keep only
    # the top rejection_keep by reward. Set to 0 or None to disable (keep all).
    rejection_keep: int = 0
    # Reward trimming: drop top and bottom X% of scenes by their mean group
    # reward before training. Prevents learning from outlier scenes (e.g.,
    # high-progress lane-departing scenes at top, heavily crashed scenes at bottom).
    reward_trim_pct: float = 0.0  # 0.05 = trim 5% of scenes from each end
    lane_dep_trim_n: int = 0  # drop N scenes with highest lane departure fraction (0=disabled)
    neighbor_loss_weight: float = 0.0  # weight for neighbor prediction regularization (0=disabled)

    # Ranked SFT mode: generate N trajectories, pick best by reward, SFT on it.
    # "none": standard GRPO training (default).
    # "gt_neighbor": use real GT neighbors from NPZ as neighbor target.
    # "baseline_neighbor": use baseline (no-LoRA) model prediction as neighbor target.
    ranked_sft_mode: str = "none"
    sg_filter_window: int = 11  # Savitzky-Golay filter window length (must be odd)
    sg_filter_order: int = 3    # Savitzky-Golay filter polynomial order
    # Neighbor regularization: penalize LoRA neighbor outputs diverging from base model.
    # Computes MSE(lora_neighbor_pred, base_neighbor_pred) at the same (noise, timestep)
    # by running a second forward pass with LoRA disabled. Adds ~2x training cost.
    # loss += neighbor_reg_weight * MSE(neighbor_pred_lora, neighbor_pred_base)
    # Active in both ranked SFT and GRPO paths. 0 = disabled.
    neighbor_reg_weight: float = 0.0
    # Controls whether to include the neighbor SFT loss (MSE vs GT neighbors).
    # When True (recommended): loss = ego_sft + neighbor_reg (no GT neighbor loss).
    #   The base model already learned good neighbor predictions; the reg term anchors
    #   them while ego improves freely. Neighbor L2 degradation: +1.8% (vs +91% without).
    # When False: loss = ego_sft + neighbor_sft + neighbor_reg (all 3 terms).
    #   Adding GT neighbor loss on top of reg causes overfitting at high LR (collapses by ep12).
    neighbor_reg_only: bool = True

    # Ego IL (imitation learning) regularization: anchors ego output to a reference.
    # loss += ego_il_weight * MSE(model_ego_pred, reference_ego)
    # Active only in ranked SFT when ego_il_weight > 0. The ranked SFT ego loss trains
    # toward the best-of-K trajectory (lane keeping), while this term pulls back toward
    # the reference (L2 preservation). Intended for 500-scene training where L2 drifts.
    ego_il_weight: float = 0.0
    # ego_il_mode: "gt" uses real GT ego trajectory as reference.
    # "baseline" uses base model (no-LoRA) ego prediction at the same (noise, timestep).
    # "baseline" is conceptually analogous to neighbor_reg (anchor to base, not GT) and
    # reuses the base model forward pass from neighbor_reg (free when neighbor_reg > 0).
    ego_il_mode: str = "gt"

    # Selective training: skip SFT update for scenes where best-of-K reward barely
    # improves over the deterministic trajectory. Focuses learning on problem scenes
    # (where guidance-aided trajectories are much better) while preserving L2 on normal
    # scenes (where baseline is already good). 0 = train all scenes (default).
    selective_threshold: float = 0.0
    # selective_mode: "threshold" (binary select/skip at threshold), "advantage" (scale
    # ego loss by normalized improvement — smooth version of selective). In advantage mode,
    # selective_threshold is the minimum improvement to include (scenes below get weight 0).
    selective_mode: str = "threshold"

    # Ranked SFT batching: how many scenes per forward pass (default 1 = sequential).
    # With sft_batch_size=B, each forward pass processes B scenes. Grad accumulation
    # steps = grad_accum_groups // sft_batch_size, so the effective batch per optimizer
    # step stays the same only when B evenly divides grad_accum_groups (enforced at
    # runtime). Set to grad_accum_groups for maximum throughput.
    sft_batch_size: int = 1

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "first"  # "first" = block 0 only (recommended), "all" = all decoder blocks, "blocks01" = blocks 0+1, "last" = block 2 only

    # Exploration Policy: learned guidance for adaptive GRPO sampling.
    # When enabled, a learned policy outputs (eta_lat, eta_lon) from Beta
    # distributions instead of uniform random sampling.
    # NOTE: requires using GRPOExplorationTrainer (rlvr/grpo_exploration_trainer.py)
    # instead of GRPOTrainer. The standard GRPOTrainer ignores this flag.
    use_exploration_policy: bool = False
    exploration_hidden_dim: int = 128
    exploration_n_mixer_layers: int = 2
    exploration_n_attn_heads: int = 4
    exploration_dropout: float = 0.1
    exploration_lr: float = 1e-4
    exploration_checkpoint_path: str | None = None
    # REINFORCE loss coefficients for exploration policy training
    exploration_entropy_coef: float = 0.05
    exploration_kl_coef: float = 0.01
    # Inverse KL scheduling: policy KL ramps UP over training (opposite of DiT KL).
    # Early: low policy KL (free exploration) + high DiT KL (stable planner).
    # Late: high policy KL (anchor learned policy) + low DiT KL (planner adapts).
    exploration_kl_schedule: str = "constant"  # "constant", "linear", "cosine"
    exploration_kl_coef_final: float = 0.05
    # Lateral/longitudinal guidance parameters for exploration policy
    exploration_lambda_lat: float = 2.5   # max lateral offset in metres
    exploration_lambda_lon: float = 0.25  # max speed deviation fraction
    exploration_guidance_scale: float = 0.5  # global guidance scale for policy-guided trajectories
    # GuidanceHead init mode: "zeros" (recommended) or "normal"
    exploration_head_init: str = "zeros"
    exploration_head_init_std: float = 0.01
    # Scale factor for raw output before softplus. Amplifies gradient flow.
    # 1.0 = no scaling (default, preserves backward compat).
    # 10.0 = recommended for policy learning (12x stronger eta signal).
    exploration_head_raw_scale: float = 1.0
    # Inner PPO epochs for exploration policy. Each scene's rollout is reused
    # for this many gradient steps with PPO clipping. Default 1 = REINFORCE.
    exploration_inner_epochs: int = 1
    exploration_clip_epsilon: float = 0.2
    # Freeze exploration policy after this many epochs (0 = never freeze).
    # The policy still runs inference (produces η for trajectory generation)
    # but its weights stop updating. Useful when the policy helps early but
    # develops harmful global bias in later epochs.
    exploration_freeze_after_epoch: int = 0
    # Step the policy optimizer per-group instead of accumulating across all
    # groups and stepping once at epoch end. Per-group stepping gives immediate
    # per-scene gradient signal, forcing the network to learn scene-dependent
    # guidance instead of a global mean shift. Default False = accumulate.
    exploration_step_per_group: bool = False
    # Number of groups to accumulate before stepping the policy optimizer.
    # Only used when exploration_step_per_group=True.
    # 1 = step every scene (original per-group), 4 = match DiT rhythm.
    exploration_grad_accum_groups: int = 1

    # Explorer loss type:
    #   "advantage_logprob" (default): advantage-weighted negative log_prob of sampled etas.
    #       Pushes policy toward etas that got above-average reward.
    #   "best_sample_mse": MSE regression of policy mean toward the best-reward eta.
    #       Directly supervises policy to output the best eta per scene.
    #       Same principle as ranked SFT for the DiT (best trajectory MSE).
    exploration_loss_type: str = "advantage_logprob"

    # Use a pre-trained exploration policy during ranked SFT generation.
    # The explorer provides per-scene (eta_lat, eta_lon) guidance for trajectory generation.
    # Set exploration_checkpoint_path to the .pth file.
    ranked_sft_use_explorer: bool = False
    # If True, explorer stays frozen during RSFT. If False, explorer trains jointly
    # with DiT (explorer via REINFORCE/MSE on rewards, DiT via SFT loss).
    ranked_sft_freeze_explorer: bool = True

    # Random guidance mode: replaces exploration policy with direct η sampling.
    # "explorer" (default): use learned exploration policy (Beta distributions).
    # "uniform": random η ~ U[-1, 1] (matches zero-init explorer output).
    # "narrow": random η_lat ~ U[-0.5, 0.5], η_lon ~ U[-0.25, 0.25].
    # "gaussian": random η_lat ~ N(0, 0.3), η_lon ~ N(0, 0.15).
    # "none": η=0 always (no guidance, pure noise diversity).
    random_guidance_mode: str = "explorer"

    # --- Closed-loop training ---
    # When True, uses ClosedLoopExplorationTrainer instead of GRPOExplorationTrainer.
    # The explorer operates per-step (0.1s) with GAE temporal credit assignment.
    use_closed_loop: bool = False
    closed_loop_rollout_steps: int = 40     # 4s at 10Hz
    closed_loop_gamma: float = 0.99         # GAE discount factor
    closed_loop_gae_lambda: float = 0.95    # GAE lambda
    closed_loop_value_coef: float = 0.5     # value loss coefficient
    closed_loop_alive_bonus: float = 0.5    # per-step alive reward
    closed_loop_freeze_dit: bool = True     # freeze DiT during explorer training
    closed_loop_batch_size: int = 8        # scenes per batch in rollout (8 fits ~24GB VRAM)
    closed_loop_drop_last: bool = True     # drop incomplete last batch
    closed_loop_online_interval: int = 0   # online explorer update every N steps (0=off, 10=PlannerRFT-style)
    closed_loop_explorer_mini_batch: int = 0  # step explorer optimizer every N scenes (0=all scenes at once)

    # --- Per-epoch scheduling for arbitrary parameters ---
    # Generic scheduling system for reward weights, guidance scales, etc.
    # Each entry maps a parameter name to a schedule spec:
    #   {"type": "constant"|"linear"|"cosine"|"step"|"peak", "start": float, "end": float,
    #    "warmup_fraction": float (for "step" only, default 0.5),
    #    "peak": float, "peak_fraction": float (for "peak" only, default 0.5)}
    #
    # Schedulable parameters:
    #   Reward weights: w_progress, w_safety, w_smooth, w_feasibility, w_centerline,
    #                   stopped_penalty, underprogress_penalty, progress_norm_scale
    #   Guidance:       longitudinal_eta, longitudinal_lambda, longitudinal_scale
    #
    # Example config JSON:
    #   "schedules": {
    #     "w_progress": {"type": "linear", "start": 3.0, "end": 10.0},
    #     "longitudinal_eta": {"type": "linear", "start": 0.0, "end": 1.0}
    #   }
    schedules: dict = field(default_factory=dict)

    # Backward compat: old field names → new field names
    _FIELD_RENAMES: ClassVar[dict[str, str]] = {
        "near_edge_scale": "rb_near_scale",
        "wide_edge_scale": "rb_wide_scale",
        "cont_edge_scale": "rb_cont_scale",
    }

    @classmethod
    def from_json(cls, path: str | Path) -> GRPOConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Rename legacy fields so old config JSONs keep working
        for old, new in cls._FIELD_RENAMES.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
            elif old in data:
                data.pop(old)  # new name takes precedence
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def get_kl_coef(self, epoch: int, total_epochs: int) -> float:
        """Compute KL coefficient for the given epoch based on schedule.

        Uses kl_coef_initial (captured from kl_coef on first call) as the
        stable starting point, so the schedule is stateless and independent
        of any mutations to kl_coef during training.

        Args:
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of training epochs.

        Returns:
            KL coefficient for this epoch.
        """
        # Capture the initial kl_coef on first call
        if self.kl_coef_initial is None:
            self.kl_coef_initial = self.kl_coef

        if self.kl_schedule == "constant" or total_epochs <= 1:
            return self.kl_coef_initial

        # progress: 0.0 at epoch 1, 1.0 at final epoch
        progress = (epoch - 1) / (total_epochs - 1)
        start, end = self.kl_coef_initial, self.kl_coef_final

        if self.kl_schedule == "linear":
            return start + (end - start) * progress

        if self.kl_schedule == "cosine":
            return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.kl_schedule == "step":
            if progress < self.kl_warmup_fraction:
                return start
            return end

        raise ValueError(
            f"Unknown kl_schedule: {self.kl_schedule!r}. "
            f"Expected 'constant', 'linear', 'cosine', or 'step'."
        )

    def get_exploration_kl_coef(self, epoch: int, total_epochs: int) -> float:
        """Compute exploration policy KL coefficient (ramps UP, inverse of DiT KL).

        Early training: low KL → policy explores freely.
        Late training: high KL → anchor learned policy to prevent drift.

        Uses a captured initial value so the schedule is stateless w.r.t.
        later mutations to exploration_kl_coef.

        Args:
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of training epochs.

        Returns:
            Exploration policy KL coefficient for this epoch.
        """
        # Capture the initial value on first call (same pattern as get_kl_coef)
        if not hasattr(self, "_exploration_kl_coef_initial"):
            self._exploration_kl_coef_initial = self.exploration_kl_coef

        if self.exploration_kl_schedule == "constant" or total_epochs <= 1:
            return self._exploration_kl_coef_initial

        progress = (epoch - 1) / (total_epochs - 1)
        start = self._exploration_kl_coef_initial
        end = self.exploration_kl_coef_final

        if self.exploration_kl_schedule == "linear":
            return start + (end - start) * progress

        if self.exploration_kl_schedule == "cosine":
            return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

        raise ValueError(
            f"Unknown exploration_kl_schedule: {self.exploration_kl_schedule!r}. "
            f"Expected 'constant', 'linear', or 'cosine'."
        )

    def get_scheduled_value(
        self, name: str, epoch: int, total_epochs: int,
    ) -> float | None:
        """Get the scheduled value for a parameter at the given epoch.

        Returns None if no schedule is defined for this parameter.
        Looks up the schedule spec in self.schedules[name].

        Args:
            name: Parameter name (e.g. "w_progress", "longitudinal_eta").
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of training epochs.
        """
        spec = self.schedules.get(name)
        if spec is None:
            return None

        stype = spec.get("type", "linear")
        start = float(spec["start"])
        end = float(spec.get("end", start))  # default to start for constant schedules

        if stype == "constant" or total_epochs <= 1:
            return start

        progress = (epoch - 1) / (total_epochs - 1)

        if stype == "linear":
            # Optional end_epoch: ramp completes at this epoch, then holds end value.
            # E.g. {"type": "linear", "start": 1.2, "end": 1.0, "end_epoch": 8}
            end_ep = spec.get("end_epoch")
            if end_ep is not None:
                end_ep = int(end_ep)
                if not 1 < end_ep <= total_epochs:
                    raise ValueError(
                        f"end_epoch for '{name}' must be in (1, {total_epochs}], "
                        f"got {end_ep}"
                    )
                if epoch >= end_ep:
                    return end
                local_progress = (epoch - 1) / (end_ep - 1)
                return start + (end - start) * local_progress
            return start + (end - start) * progress

        if stype == "cosine":
            return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if stype == "step":
            warmup = float(spec.get("warmup_fraction", 0.5))
            if not 0.0 <= warmup <= 1.0:
                raise ValueError(
                    f"warmup_fraction for '{name}' must be in [0, 1], got {warmup}"
                )
            return start if progress < warmup else end

        if stype == "peak":
            # Ramp start → peak → end. Linear interpolation on each half.
            #   {"type": "peak", "start": 0.0, "end": 0.0, "peak": 0.3, "peak_fraction": 0.5}
            peak_val = float(spec["peak"])
            peak_frac = float(spec.get("peak_fraction", 0.5))
            if not 0.0 < peak_frac < 1.0:
                raise ValueError(
                    f"peak_fraction for '{name}' must be in (0, 1), got {peak_frac}"
                )
            if progress <= peak_frac:
                t = progress / peak_frac
                return start + (peak_val - start) * t
            else:
                t = (progress - peak_frac) / (1.0 - peak_frac)
                return peak_val + (end - peak_val) * t

        raise ValueError(
            f"Unknown schedule type for '{name}': {stype!r}. "
            f"Expected 'constant', 'linear', 'cosine', 'step', or 'peak'."
        )

    def get_all_scheduled_values(
        self, epoch: int, total_epochs: int,
    ) -> dict[str, float]:
        """Get all scheduled values for the given epoch.

        Returns dict of {name: value} for all parameters with schedules defined.
        """
        result: dict[str, float] = {}
        for name in self.schedules:
            value = self.get_scheduled_value(name, epoch, total_epochs)
            if value is not None:
                result[name] = value
        return result

    def __post_init__(self):
        """Normalize legacy loss type names to current names."""
        _loss_renames = {
            "mse": "advantage_mse",
            "logprob": "advantage_logprob",
            "reinforce": "advantage_logprob",
            "rsft": "best_sample_mse",
            "best_eta_mse": "best_sample_mse",
            "grpo": "advantage_logprob",
        }
        if self.grpo_loss_type in _loss_renames:
            self.grpo_loss_type = _loss_renames[self.grpo_loss_type]
        if self.exploration_loss_type in _loss_renames:
            self.exploration_loss_type = _loss_renames[self.exploration_loss_type]
        # Validate: best_sample_mse is not compatible with PPO (inner_epochs > 1)
        if (self.exploration_loss_type == "best_sample_mse"
                and self.exploration_inner_epochs > 1):
            raise ValueError(
                "exploration_loss_type='best_sample_mse' is not compatible with "
                f"exploration_inner_epochs={self.exploration_inner_epochs} (PPO). "
                "Use exploration_inner_epochs=1 or exploration_loss_type='advantage_logprob'."
            )

    @property
    def uses_importance_sampling(self) -> bool:
        """Whether this config uses PPO-clipped importance sampling (M > 1)."""
        return self.inner_epochs > 1
