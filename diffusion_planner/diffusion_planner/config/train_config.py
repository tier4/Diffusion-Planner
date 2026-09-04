from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from .closed_loop_config import ClosedLoopConfig
from .config_cli import cli
from .model_config import ModelConfig
from .scenario_open_loop_config import ScenarioOpenLoopConfig


@dataclass
class TrainConfig(ClosedLoopConfig, ScenarioOpenLoopConfig, ModelConfig):
    # ---------------------------------------------------------
    # Required Arguments
    # ---------------------------------------------------------
    exp_name: str = cli("name of this run; appears in the save directory and in wandb", default="")
    train_set_list: str = cli("JSON list of training NPZ paths", path=True, default="")
    valid_set_list: str = cli("JSON list of validation NPZ paths", path=True, default="")

    # ---------------------------------------------------------
    # Run output
    # ---------------------------------------------------------
    output_root: str = cli(
        "parent directory for run directories",
        default="/mnt/nvme/training_result",
        path=True,
    )
    save_dir: str = cli(
        "this run's directory; derived as <output_root>/<timestamp>_<exp_name> when empty",
        default="",
        path=True,
    )

    # ---------------------------------------------------------
    # DataLoader Parameters
    # ---------------------------------------------------------
    batch_size: int = cli("batch size across all GPUs", default=512)
    num_workers: int = 8
    pin_mem: bool = True

    use_data_augment: bool = True
    augment_prob: float = 0.5
    augment_type: Literal["quintic", "bridge", "frenet"] = cli(
        "data augmentation method: quintic (default) = fixed-offset quintic bridge; "
        "bridge = extended bridge perturbation; frenet = corridor-constrained lateral "
        "perturbation with feasibility filtering and history rewrite.",
        default="quintic",
    )
    num_refine: int = 20
    ego_past_noise_std: float = 0.1
    use_smoothing_future_trajectory: bool = True

    # --- frenet augmentation knobs (ignored unless augment_type=frenet) ---
    # The defaults are the measured configuration: narrowing the offsets to the
    # quintic range or dropping the ranked merge selection each cost most of the
    # closed-loop benefit, so treat these as a sweep handle, not a tuning dial.
    frenet_n_draws: int = cli("frenet: joint (offset, heading) draws per scene.", default=16)
    frenet_dy_max: float = cli("frenet: maximum lateral offset in metres.", default=2.0)
    frenet_dth_max: float = cli("frenet: maximum heading offset in radians.", default=0.17)
    frenet_merge_times: list[float] = cli(
        "frenet: horizons (s) at which a candidate may rejoin the recorded path; "
        "one is sampled per scene among the feasible ones.",
        default_factory=lambda: [2.0, 3.0, 4.0, 5.0],
    )
    frenet_anchors: list[float] = cli(
        "frenet: how far back (s) the rewritten history departs from the recording.",
        default_factory=lambda: [2.0, 3.0],
    )
    frenet_acc0_fracs: list[float] = cli(
        "frenet: initial lateral curvature, as a fraction of the value a quintic "
        "naturally takes for the drawn offset and horizon.",
        default_factory=lambda: [0.0, -0.5, 0.5, -1.0, 1.0],
    )
    frenet_ranked_temp_s: float = cli(
        "frenet: time constant (s) of the merge-horizon sampling; smaller favours "
        "faster convergence more strongly.",
        default=1.0,
    )
    frenet_seed: int = cli(
        "frenet: base RNG seed for the augmentation draws (offset per DDP rank).",
        default=0,
    )
    normalization_file_path: str = "normalization.json"
    # Override channel for launch wrappers: JSON object of non-CLI TrainConfig
    # fields; unknown keys fail loudly (see train_predictor.apply_overrides_json).
    train_overrides_json: str = cli(
        "JSON object of TrainConfig field overrides applied after CLI parsing",
        path=True,
        default="",
    )

    train_subsample_step: int = 1

    # ---------------------------------------------------------
    # Training Parameters
    # ---------------------------------------------------------
    seed: int = 3407
    train_epochs: int = cli("total training epochs", default=80)
    save_utd: int = cli("checkpoint save cadence in epochs", default=10)
    learning_rate: float = 1e-4
    warm_up_epoch: int = 5
    lr_schedule: Literal["constant", "cosine"] = cli(
        "post-warm-up LR schedule: 'constant' holds the configured LR (default; matches "
        "every existing checkpoint), 'cosine' anneals it to 0 by the final epoch.",
        default="constant",
    )
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
    coeff_timestep: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    coeff_road_border_loss: float = 1.0
    road_border_margin: float = 0.25
    road_border_n_interp: int = 2

    coeff_neighbor_collision_loss: float = 0.0
    neighbor_collision_margin_vehicle: float = 0.25
    neighbor_collision_margin_pedestrian: float = 1.0
    neighbor_collision_margin_bicycle: float = 0.5

    alpha_planning_loss: float = 1.0
    alpha_neighbor_loss: float = 0.1

    # Velocity Representation & Hybrid Loss
    use_velocity_representation: bool = False
    hybrid_loss_omega: float = 0.1
    hybrid_loss_window: int = 10

    guidance_scale: float = 0.5
    device: str = "cuda"
    use_ema: bool = True
    ema_decay: float = 0.999
    resume_model_path: Optional[str] = cli(
        "resume training from this .pth", default=None, path=True
    )

    # ---------------------------------------------------------
    # Logging & Distributed
    # ---------------------------------------------------------
    use_wandb: bool = cli("log the run to Weights & Biases", default=True)
    wandb_run_id: Optional[str] = cli(
        "attach to this existing wandb run instead of creating one", default=None
    )
    wandb_project_name: str = cli("Weights & Biases project name", default="Diffusion-Planner")
    notes: str = ""
    ddp: bool = True
    port: str = "22323"

    enable_temporal_stability_eval: bool = cli(
        "validation-only ego jerk / curvature-rate metrics",
        default=True,
    )
    enable_replan_consistency_eval: bool = cli(
        "validation-only inter-frame replan consistency (doubles validation cost)",
        default=True,
    )
    replan_consistency_expected_gap: int = 1

    enable_epdms_eval: bool = False
    enable_pdms_eval: bool = False
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True

    # ---------------------------------------------------------
    # Normalizers (set at runtime)
    # ---------------------------------------------------------
    state_normalizer: Optional[Any] = field(default=None, repr=False)
    observation_normalizer: Optional[Any] = field(default=None, repr=False)

    # ---------------------------------------------------------
    # Deterministic
    # ---------------------------------------------------------
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.save_dir:
            self.save_dir = self.build_save_dir(self.output_root, self.exp_name)

    @staticmethod
    def build_save_dir(output_root: str, exp_name: str) -> str:
        return str(Path(output_root) / f"{datetime.now():%Y%m%d-%H%M%S}_{exp_name}")
