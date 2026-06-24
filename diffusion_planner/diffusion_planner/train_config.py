from dataclasses import dataclass, field
from typing import Literal, Optional

from diffusion_planner.dimensions import (
    INPUT_T,
    MAX_NUM_NEIGHBORS,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    OUTPUT_T,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
)
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


@dataclass
class TrainConfig:
    # ---------------------------------------------------------
    # Required Arguments (Fields without default values must be declared first)
    # ---------------------------------------------------------
    exp_name: str
    save_dir: str
    train_set_list: str
    valid_set_list: str

    # ---------------------------------------------------------
    # Data Dimensions
    # ---------------------------------------------------------
    future_len: int = OUTPUT_T
    time_len: int = INPUT_T + 1
    ego_prediction_horizon: int = OUTPUT_T

    agent_state_dim: int = 11
    agent_num: int = MAX_NUM_NEIGHBORS

    static_objects_state_dim: int = 10
    static_objects_num: int = 5

    lane_num: int = NUM_SEGMENTS_IN_LANE
    lane_len: int = POINTS_PER_LANELET

    route_num: int = NUM_SEGMENTS_IN_ROUTE
    route_len: int = POINTS_PER_LANELET

    polygon_num: int = NUM_POLYGONS
    polygon_len: int = POINTS_PER_POLYGON

    line_string_num: int = NUM_LINE_STRINGS
    line_string_len: int = POINTS_PER_LINE_STRING

    # ---------------------------------------------------------
    # DataLoader Parameters
    # ---------------------------------------------------------
    use_data_augment: bool = True
    augment_prob: float = 0.5
    augment_type: Literal["quintic", "bridge"] = "quintic"
    num_refine: int = 20
    ego_past_noise_std: float = 0.1
    use_smoothing_future_trajectory: bool = True
    normalization_file_path: str = "normalization.json"
    num_workers: int = 8
    pin_mem: bool = True

    # ---------------------------------------------------------
    # Training Parameters
    # ---------------------------------------------------------
    seed: int = 3407
    train_epochs: int = 100
    batch_size: int = 512
    save_utd: int = 10
    learning_rate: float = 1e-4
    warm_up_epoch: int = 5
    encoder_drop_path_rate: float = 0.1
    decoder_drop_path_rate: float = 0.1
    use_ego_history: bool = True
    ego_history_dropout_rate: float = 0.4
    use_turn_indicators: bool = True

    # Loss Coefficients
    coeff_position_lat_loss: float = 1.0
    coeff_position_lon_loss: float = 1.0
    coeff_heading_l2_loss: float = 1.0
    coeff_velocity: float = 1.0
    # Use default_factory for mutable default values like lists
    coeff_timestep: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    coeff_road_border_loss: float = 1.0
    road_border_margin: float = 0.25
    road_border_n_interp: int = 2

    coeff_neighbor_collision_loss: float = 0.0
    neighbor_collision_margin: float = 0.25

    # JEPA latent-consistency loss (SAGE-JEPA Use A). Default-off: coeff 0 => bit-identical.
    # Ego-only frozen energy (won the AUROC comparison; scene context degraded it).
    coeff_jepa_consistency_loss: float = 0.0
    jepa_prefix_K: int = 10  # 1 s prefix at 10 Hz (paper sweet spot; K>=20 degraded)
    jepa_encoder_ckpt: Optional[str] = None
    jepa_predictor_ckpt: Optional[str] = None

    # Cross-frame temporal-consistency loss (reduces frame-to-frame flicker). Default-off:
    # coeff 0 => single-frame training (bit-identical). When > 0, training uses paired
    # consecutive frames: the plan at frame t propagated forward must agree with the plan
    # at frame t+g on the overlap. Validated to cut the replan-jump tail at flat accuracy.
    coeff_temporal_consistency: float = 0.0
    tc_step_g: int = 3  # frame cadence == trajectory-step offset between paired frames
    tc_fixed_t: float = 0.5  # near-clean diffusion-t for the consistency forward
    tc_cons_scale: float = 10.0  # normalise cons (metres) into planning-loss units
    tc_w_heading: float = 1.0
    # SDEdit-aware training: when True (and the paired dataset is on via
    # coeff_temporal_consistency>0), frame_{t+g}'s ego diffusion input is noised from the
    # previous frame's propagated plan and supervised toward GT — teaches the model to use
    # an SDEdit prior-init at inference (keep prior where scene unchanged, correct it where
    # changed). Default-off => the soft consistency-loss path is used instead.
    sdedit_train: bool = False
    # Explicit prior-conditioning training: feed the previous frame's propagated plan as
    # cross-attention tokens (PriorEncoder) with 50% dropout, GT-supervised — the model learns
    # to use the prior selectively (cut flicker where the scene is unchanged, override it
    # where it changed). Default-off.
    prior_cond_train: bool = False
    # History-context conditioning (pivot): feed the previous frame's POOLED SCENE ENCODING (not
    # the plan) as cross-attention tokens to frame_{t+g}, with 50% dropout + optional consistency
    # loss (coeff_temporal_consistency>0). Temporal context with no trajectory to copy. Default-off.
    history_cond_train: bool = False
    # Scene-aware consistency gating (synthesis follow-up). When True, the consistency loss
    # on each paired sample is weighted by w = exp(-gt_dev / tc_gate_tau), where gt_dev is the
    # GT-vs-GT scene-change deviation (how much GT_{t+g} departs from the propagated GT_t).
    # Normal frames (gt_dev ~ 0 => w ~ 1) keep full consistency (stability); scene-change frames
    # (large gt_dev => w ~ 0) drop it so GT wins (responsiveness). Decouples flicker-gain from
    # copying — the global coeff could only trade them off uniformly. Default-off (no gating).
    tc_scene_gate: bool = False
    tc_gate_q: float = 0.5    # adaptive floor = this quantile of gt_dev (frames <= it keep w=1)
    tc_gate_tau: float = 6.0  # exp decay above the floor, in consistency-loss units (m + rad)

    # Matmul precision. True => TF32 tensor cores on H100 (~2x, training-equivalent to fp32).
    # False => strict fp32 ("highest"). Only set False for the TF32-vs-fp32 speed benchmark.
    tf32: bool = True

    alpha_planning_loss: float = 1.0
    alpha_neighbor_loss: float = 0.1

    # Velocity Representation & Hybrid Loss
    use_velocity_representation: bool = False
    hybrid_loss_omega: float = 0.1
    hybrid_loss_window: int = 10

    guidance_scale: float = 0.5
    device: str = "cuda"
    use_ema: bool = True

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    encoder_mixer_depth: int = 6
    encoder_fusion_depth: int = 6
    decoder_depth: int = 3
    num_heads: int = 8
    hidden_dim: int = 256
    diffusion_model_type: Literal["x_start", "flow_matching"] = "x_start"
    predicted_neighbor_num: int = MAX_NUM_NEIGHBORS
    resume_model_path: Optional[str] = None
    # Weights-only warm start (loads model weights, fresh optimizer/scheduler/epoch) — for
    # fine-tuning from a checkpoint with a new objective + fresh short schedule.
    init_weights_path: Optional[str] = None

    # ---------------------------------------------------------
    # Logging & Distributed Setup
    # ---------------------------------------------------------
    use_wandb: bool = False
    wandb_run_id: Optional[str] = None
    wandb_project_name: str = "Diffusion-Planner"
    # Log training metrics every N steps (0 = per-epoch only). >0 commits continuously so
    # curves are visible within seconds, instead of buffering until the next epoch.
    wandb_step_log_interval: int = 0
    notes: str = ""
    ddp: bool = True
    port: str = "22323"

    # ---------------------------------------------------------
    # Normalizers (Placeholders to be initialized and set during training execution)
    # ---------------------------------------------------------
    state_normalizer: Optional[StateNormalizer] = None
    observation_normalizer: Optional[ObservationNormalizer] = None
    # Frozen JEPA energy (set during execution when coeff_jepa_consistency_loss > 0).
    jepa_energy: Optional[object] = None
