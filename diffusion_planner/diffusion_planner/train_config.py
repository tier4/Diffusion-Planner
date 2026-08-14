from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

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

# This dataclass is the single source of truth for every training setting: name, type,
# default and documentation. To change a setting, edit its default here.
#
# The handful of settings that genuinely have to vary per invocation are marked with
# ``cli(...)`` below. ``train_cli.py`` turns exactly those into argparse flags for both
# train_predictor.py and train_run.py, so neither entrypoint restates a name, a default,
# a type or a help string. Everything not marked is deliberately not on the command line.


def cli(help_text: str, *, path: bool = False, **kwargs: Any) -> Any:
    """Mark a field as settable from the command line.

    Args:
        help_text: shown in ``--help``.
        path: resolve to an absolute path before handing it to a subprocess. Needed
            because train_run.py runs the trainer from the entrypoint directory, so a
            path typed relative to the user's cwd would otherwise be misread.
        kwargs: ``default`` / ``default_factory``. Omit both to make the flag required.
    """
    return field(metadata={"cli": True, "help": help_text, "path": path}, **kwargs)


@dataclass
class TrainConfig:
    # ---------------------------------------------------------
    # Required Arguments (Fields without default values must be declared first)
    # ---------------------------------------------------------
    exp_name: str = cli("name of this run; appears in the save directory and in wandb")
    train_set_list: str = cli("JSON list of training NPZ paths", path=True)
    valid_set_list: str = cli("JSON list of validation NPZ paths", path=True)

    # ---------------------------------------------------------
    # Run output
    #
    # One rule, shared by both entrypoints: `save_dir` is derived from `output_root`
    # unless it is given explicitly. train_run.py derives it once and passes the result
    # down, so the launcher and the trainer always agree on the directory (previously
    # train_run.py built the timestamped path itself while train_predictor.py demanded a
    # ready-made `save_dir`, so the two disagreed about what a run directory even was).
    # ---------------------------------------------------------
    output_root: str = cli(
        "parent directory for run directories", default="/mnt/nvme/training_result", path=True
    )
    save_dir: str = cli(
        "this run's directory; derived as <output_root>/<timestamp>_<exp_name> when empty",
        default="",
        path=True,
    )

    train_subsample_step: int = 1

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
    train_epochs: int = 80
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
    neighbor_collision_margin_vehicle: float = 0.25
    neighbor_collision_margin_pedestrian: float = 1.0
    neighbor_collision_margin_bicycle: float = 0.5

    # Validation-only Autoware-aligned EPDMS metrics. train_predictor.py reads
    # these defaults when constructing argparse, so this remains the single
    # default source while keeping existing behavior unchanged unless explicitly enabled.
    enable_epdms_eval: bool = False
    # Backward-compatible alias for local scripts that used PDMS naming.
    enable_pdms_eval: bool = False
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True

    alpha_planning_loss: float = 1.0
    alpha_neighbor_loss: float = 0.1

    # Velocity Representation & Hybrid Loss
    use_velocity_representation: bool = False
    hybrid_loss_omega: float = 0.1
    hybrid_loss_window: int = 10

    guidance_scale: float = 0.5
    device: str = "cuda"
    use_ema: bool = True
    compile_model: bool = cli("compile the model with torch.compile before training", default=False)
    use_amp: bool = cli("train with Automatic Mixed Precision (bf16 autocast)", default=False)
    # ModelEma decay; 0.999 needs ~3000 steps to absorb a behavior change —
    # lower for short fine-tune rounds (e.g. 0.996 for ~800-step rounds).
    ema_decay: float = 0.999

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
    resume_model_path: Optional[str] = cli(
        "resume training from this .pth", default=None, path=True
    )

    # ---------------------------------------------------------
    # Logging & Distributed Setup
    # ---------------------------------------------------------
    use_wandb: bool = cli("log the run to Weights & Biases", default=True)
    wandb_run_id: Optional[str] = cli(
        "attach to this existing wandb run instead of creating one", default=None
    )
    wandb_project_name: str = cli("Weights & Biases project name", default="Diffusion-Planner")
    notes: str = ""
    ddp: bool = True
    port: str = "22323"

    # Validation-only temporal stability metrics. Replan consistency requires full-sequence
    # Step-1 NPZ frames in valid_set_list; the default gap=1 avoids treating skip-N lists
    # as true frame-to-frame replanning data.
    enable_temporal_stability_eval: bool = cli(
        "validation-only ego jerk / curvature-rate metrics. Computed from the trajectory "
        "the normal validation pass already predicts, so turning this off saves little.",
        default=True,
    )
    enable_replan_consistency_eval: bool = cli(
        "validation-only inter-frame replan consistency. Needs a Step-1 valid_set_list and "
        "runs TWO extra forwards per adjacent frame pair every epoch, so on a full Step-1 "
        "list this roughly doubles validation cost.",
        default=True,
    )
    replan_consistency_expected_gap: int = 1

    # ---------------------------------------------------------
    # Closed-loop validation (rendered rollout + wandb video), run on the checkpoint-save cadence
    # (``save_utd``). Disabled unless ``closed_loop_npz_root`` or ``closed_loop_sites_npz_root`` is set.
    #
    # ``closed_loop_sites_npz_root`` is an alternative/addition to ``closed_loop_npz_root`` for
    # multi-site validation: a curated .json path-list manifest, grouped into per-site route pools
    # by scenario_generation.site_discovery.discover_sites_with_vehicles_from_json and evaluated
    # as independent npz_roots, wandb-logged under "closed_loop_scores/<metric>/<site_name>".
    # Both may be set at once — each fires independently and contributes its own rows to the
    # combined episode table / cross-site aggregate.
    # ---------------------------------------------------------
    closed_loop_npz_root: str = cli(
        "dir tree of route NPZ frames for closed-loop validation, OR a .json path list of "
        "such dirs. Empty = disabled.",
        default="",
        path=True,
    )
    closed_loop_sites_npz_root: str = cli(
        "curated .json path-list manifest grouped into per-site route pools and evaluated "
        "as independent sites. May be set together with closed_loop_npz_root.",
        default="",
        path=True,
    )
    # Object-mode ablation per source: "objects"=normal, "noobj"=empty-world (no dynamic/static
    # objects, map kept — isolates "reacts badly to traffic" from "can't follow the
    # route/map"). npz_root defaults to objects-only (usually a single curated scene);
    # sites_root defaults to both (the objects-vs-noobj comparison).
    closed_loop_npz_object_modes: list[str] = field(default_factory=lambda: ["objects"])
    closed_loop_sites_object_modes: list[str] = field(default_factory=lambda: ["objects", "noobj"])
    closed_loop_seg_len: int = 100000  # large -> one route = one segment = one trial
    # Re-plan every N steps: replan=1 is a model forward EVERY step (~minutes/epoch over a full
    # route); 40 keeps per-epoch cost to ~tens of seconds. Lower it for higher-fidelity validation.
    closed_loop_replan_interval: int = 4
    closed_loop_draw_every: int = 4  # render 1 of every N steps (matplotlib is the dominant cost)
    # draw on this many worker processes (minimum 1)
    closed_loop_draw_workers: int = 4
    closed_loop_fps: int = 10
    closed_loop_near_miss_thresh: float = 0.5
    closed_loop_search_radius: float = 1.5
    closed_loop_warmup_steps: int = 0
    closed_loop_unstick_after: int = 300
    closed_loop_unstick_advance_m: float = 5.0
    closed_loop_unstick_radius_mult: float = 10.0
    closed_loop_unstick_teleport_after: int = 300
    # Early-abort a badly-diverged segment instead of burning the full step budget (see
    # RolloutParams / reproducer_rollout.render_segment for the exact trigger condition).
    # 0 = disabled for either knob.
    closed_loop_abort_deviation_m: float = 50.0
    closed_loop_abort_after: int = 30
    closed_loop_abort_max_snaps: int = 0
    # wandb payload shaping (see scenario_generation.wandb_closed_loop): only ONE representative
    # episode's video + trajectory-colormap image is uploaded per site per checkpoint (not all
    # routes — those stay in the local out_dir), picked by "worst" (default, most collisions) /
    # "first" / "longest".
    closed_loop_wandb_video_pick: str = "worst"
    closed_loop_colormap_metrics: list[str] = field(
        default_factory=lambda: [
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ]
    )
    closed_loop_report_base_url: str = ""

    # Scenario-based Open-loop validation. The list JSON maps metric names to NPZ paths.
    scenario_based_open_loop_list: str = cli(
        "JSON mapping scenario-based open-loop metric names to NPZ path lists. Empty = disabled.",
        default="",
        path=True,
    )
    scenario_centerline_horizon_seconds: float = 8.0
    scenario_departure_horizon_seconds: float = 3.0
    scenario_departure_minimum_displacement_m: float = 2.0

    # ---------------------------------------------------------
    # Normalizers (Placeholders to be initialized and set during training execution)
    # ---------------------------------------------------------
    state_normalizer: Optional[StateNormalizer] = None
    observation_normalizer: Optional[ObservationNormalizer] = None

    # ---------------------------------------------------------
    # Model fixes ported from tier4/dev.
    #
    # Each of these changes what the network computes, but none of them changes a weight
    # shape, so a checkpoint trained with the flag off still loads and still exports a
    # byte-identical ONNX graph. They default off to keep tier4-main / deployed-ONNX
    # compatibility; switch them on for new training runs. The value is recorded in
    # args.json, so ONNX export rebuilds the architecture the checkpoint was trained with.
    #
    # Checkpoints predating a flag simply lack the key; utils.config.model_flag reads it
    # as off, which is exactly how they were trained.
    # ---------------------------------------------------------
    # Stop padded tokens from contributing downstream: zero the encoder's output at
    # padded positions, mask those positions out of the DiT's cross-attention, and skip
    # them when pooling the encoding for the turn indicator. 313 of 564 tokens are
    # padding for a typical scene, and today they carry whatever the fusion attention
    # produced for them.
    #
    # dev ships these as three commits, but only the first has any effect on its own:
    # the other two detect padding as an all-zero token, so without the zeroing they are
    # provably no-ops (measured: bit-identical outputs). One flag, since no other
    # combination is meaningful.
    use_encoder_padding_mask: bool = cli(
        "zero padded encoder tokens, and skip them in cross-attention and pooling",
        default=False,
    )
    # Feed the pre-norm activation to the self-attention key/value as well as the query.
    # The current code norms the query only, so key/value see a differently scaled input.
    use_prenorm_kv_self_attention: bool = cli(
        "feed the pre-norm activation to the self-attention key/value, not just the query",
        default=False,
    )
    # Give the turn-indicator token its own class type in the positional embedding. It
    # currently reuses CLASS_TYPE_EGO_SHAPE, so the two tokens are indistinguishable there.
    use_turn_indicator_class_type: bool = cli(
        "give the turn-indicator token its own positional class type", default=False
    )
    # Drop real-time chunking: condition the denoiser on one scalar timestep per sample
    # instead of a timestep per agent per horizon step, and stop training on randomly
    # delayed prefixes. Unlike the flags above this swaps the timestep embedder, so the
    # weights differ and a checkpoint does not carry across -- it needs its own run.
    # The exported ONNX keeps its input signature either way (see onnx_export), so the
    # deployed ROS node loads both.
    disable_real_time_chunking: bool = cli(
        "condition on a scalar diffusion timestep instead of real-time chunking",
        default=False,
    )

    # ---------------------------------------------------------
    # Deterministic
    # ---------------------------------------------------------
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.save_dir:
            self.save_dir = self.build_save_dir(self.output_root, self.exp_name)

    @staticmethod
    def build_save_dir(output_root: str, exp_name: str) -> str:
        """The one place a run directory name is defined."""
        return str(Path(output_root) / f"{datetime.now():%Y%m%d-%H%M%S}_{exp_name}")
