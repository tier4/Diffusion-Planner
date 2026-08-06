import argparse

from diffusion_planner.dimensions import *
from diffusion_planner.train import model_training
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _train_config_default(name):
    return TrainConfig.__dataclass_fields__[name].default


def get_args(args_list=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="save path for model ckpt", required=True)

    # Data
    parser.add_argument("--train_set_list", type=str, required=True)
    parser.add_argument("--valid_set_list", type=str, required=True)
    parser.add_argument("--train_subsample_step", type=int, default=1)

    parser.add_argument("--future_len", type=int, default=OUTPUT_T)
    parser.add_argument("--time_len", type=int, default=INPUT_T + 1)
    parser.add_argument("--ego_prediction_horizon", type=int, default=OUTPUT_T)

    parser.add_argument("--agent_state_dim", type=int, help="past state dim for agents", default=11)
    parser.add_argument("--agent_num", type=int, default=MAX_NUM_NEIGHBORS)

    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_num", type=int, default=NUM_SEGMENTS_IN_LANE)
    parser.add_argument("--lane_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--route_num", type=int, default=NUM_SEGMENTS_IN_ROUTE)
    parser.add_argument("--route_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--polygon_num", type=int, default=NUM_POLYGONS)
    parser.add_argument("--polygon_len", type=int, default=POINTS_PER_POLYGON)

    parser.add_argument("--line_string_num", type=int, default=NUM_LINE_STRINGS)
    parser.add_argument("--line_string_len", type=int, default=POINTS_PER_LINE_STRING)

    # DataLoader parameters
    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--augment_prob", type=float, help="augmentation probability", default=0.5)
    parser.add_argument(
        "--augment_type", type=str, choices=["quintic", "bridge"], default="quintic"
    )
    parser.add_argument(
        "--num_refine", type=int, default=20, help="number of refinement steps for augmentation"
    )
    parser.add_argument(
        "--ego_past_noise_std",
        type=float,
        default=0.1,
        help="std of noise applied to ego past trajectory during augmentation",
    )
    parser.add_argument(
        "--use_smoothing_future_trajectory",
        default=True,
        type=boolean,
        help="whether to apply smoothing to future trajectory",
    )
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true", help="Pin CPU memory in DataLoader")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--save_utd", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_ego_history", type=boolean, default=True)
    parser.add_argument("--ego_history_dropout_rate", type=float, default=0.4)
    parser.add_argument("--use_turn_indicators", type=boolean, default=True)

    parser.add_argument("--coeff_position_lat_loss", type=float, default=1.0)
    parser.add_argument("--coeff_position_lon_loss", type=float, default=1.0)
    parser.add_argument("--coeff_heading_l2_loss", type=float, default=1.0)
    parser.add_argument("--coeff_velocity", type=float, default=1.0)
    parser.add_argument(
        "--coeff_timestep",
        type=list,
        default=[1.0, 1.0, 1.0, 1.0],
        help="Set for 4 sections [0,20), [20, 40), [40, 60), [60, 80)",
    )

    parser.add_argument("--coeff_road_border_loss", type=float, default=1.0)
    parser.add_argument("--road_border_margin", type=float, default=0.25)
    parser.add_argument("--road_border_n_interp", type=int, default=2)

    parser.add_argument("--coeff_neighbor_collision_loss", type=float, default=0.0)
    parser.add_argument(
        "--neighbor_collision_margin_vehicle",
        type=float,
        default=0.25,
        help="per-side neighbor box inflation [m] for vehicles",
    )
    parser.add_argument(
        "--neighbor_collision_margin_pedestrian",
        type=float,
        default=1.0,
        help="per-side neighbor box inflation [m] for pedestrians",
    )
    parser.add_argument(
        "--neighbor_collision_margin_bicycle",
        type=float,
        default=0.5,
        help="per-side neighbor box inflation [m] for bicycles",
    )

    parser.add_argument(
        "--enable_epdms_eval",
        default=_train_config_default("enable_epdms_eval"),
        type=boolean,
    )
    parser.add_argument(
        "--enable_pdms_eval",
        default=_train_config_default("enable_pdms_eval"),
        type=boolean,
    )
    parser.add_argument(
        "--epdms_eval_use_agent_boxes",
        default=_train_config_default("epdms_eval_use_agent_boxes"),
        type=boolean,
    )
    parser.add_argument(
        "--epdms_eval_use_road_border",
        default=_train_config_default("epdms_eval_use_road_border"),
        type=boolean,
    )

    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--alpha_neighbor_loss", type=float, default=0.1)

    # Velocity representation & hybrid loss (HDP paper, Section IV-B)
    parser.add_argument(
        "--use_velocity_representation",
        type=boolean,
        default=False,
        help="Output trajectory as per-frame displacement instead of absolute waypoints",
    )
    parser.add_argument(
        "--hybrid_loss_omega",
        type=float,
        default=0.1,
        help="Weight for waypoint loss term in hybrid loss (omega in the paper)",
    )
    parser.add_argument(
        "--hybrid_loss_window",
        type=int,
        default=10,
        help="Gradient detach window size W for the waypoint loss term",
    )

    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, help="run on which device", default="cuda")

    parser.add_argument("--use_ema", default=True, type=boolean)
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="ModelEma decay; 0.999 (default) needs ~3000 steps to absorb a "
        "behavior change — lower for short fine-tune rounds (e.g. 0.996 for "
        "~800-step rounds).",
    )

    # Model
    parser.add_argument("--encoder_mixer_depth", type=int, default=6)
    parser.add_argument("--encoder_fusion_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, help="number of decoding layers", default=3)
    parser.add_argument("--num_heads", type=int, help="number of multi-head", default=8)
    parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=256)
    parser.add_argument(
        "--diffusion_model_type",
        type=str,
        choices=["x_start", "flow_matching"],
        default="x_start",
    )
    parser.add_argument("--predicted_neighbor_num", type=int, default=MAX_NUM_NEIGHBORS)

    parser.add_argument("--resume_model_path", type=str, help="path to resume model", default=None)

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument(
        "--wandb_run_id", type=str, default=None, help="Existing wandb run ID to attach to"
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="Diffusion-Planner",
        help="Weights & Biases project name",
    )
    parser.add_argument("--notes", default="", type=str)

    # distributed training parameters
    parser.add_argument("--ddp", default=True, type=boolean, help="use ddp or not")
    parser.add_argument("--port", default="22323", type=str, help="port")
    parser.add_argument(
        "--enable_temporal_stability_eval",
        default=_train_config_default("enable_temporal_stability_eval"),
        type=boolean,
    )
    parser.add_argument(
        "--enable_replan_consistency_eval",
        default=_train_config_default("enable_replan_consistency_eval"),
        type=boolean,
    )
    parser.add_argument(
        "--replan_consistency_expected_gap",
        type=int,
        default=_train_config_default("replan_consistency_expected_gap"),
        help="Expected consecutive-frame gap for replan consistency. 0 = auto per timeline.",
    )

    # per-epoch closed-loop validation (rendered rollout + wandb video).
    # Disabled unless --closed_loop_npz_root or --closed_loop_sites_npz_root is given.
    parser.add_argument(
        "--closed_loop_npz_root",
        type=str,
        default="",
        help="dir tree of route NPZ frames for closed-loop validation, run on the checkpoint-save "
        "cadence (save_utd). Empty = disabled. One route per trial.",
    )
    parser.add_argument(
        "--closed_loop_sites_npz_root",
        type=str,
        default="",
        help="alternative/addition to --closed_loop_npz_root: a curated .json path-list manifest, "
        "grouped into per-site route pools by site_discovery.discover_sites_from_json and evaluated "
        "as independent sites (own npz_root each). Both may be set at once (each fires "
        "independently).",
    )
    parser.add_argument(
        "--closed_loop_npz_object_modes",
        nargs="+",
        choices=("objects", "noobj"),
        default=["objects"],
        help="object-mode(s) to run --closed_loop_npz_root under: 'objects'=normal, "
        "'noobj'=empty-world ablation (no dynamic/static objects, map kept; label gets a "
        "'__noobj' suffix). Default is objects-only (single arbitrary path, usually a curated "
        "scene set).",
    )
    parser.add_argument(
        "--closed_loop_sites_object_modes",
        nargs="+",
        choices=("objects", "noobj"),
        default=["objects", "noobj"],
        help="object-mode(s) to run each --closed_loop_sites_npz_root site under (see "
        "--closed_loop_npz_object_modes). Default runs both, for the objects-vs-noobj "
        "comparison in wandb.",
    )
    parser.add_argument(
        "--closed_loop_seg_len",
        type=int,
        default=100000,
        help="frames per segment; large => one route = one segment = one trial",
    )
    parser.add_argument(
        "--closed_loop_replan_interval",
        type=int,
        default=4,
        help="re-plan every N steps; 1 = forward every step (slow, ~minutes/epoch). 40 default",
    )
    parser.add_argument(
        "--closed_loop_draw_every",
        type=int,
        default=4,
        help="render 1 of every N steps (matplotlib render is the dominant cost)",
    )
    parser.add_argument(
        "--closed_loop_draw_workers",
        type=int,
        default=_train_config_default("closed_loop_draw_workers"),
        help="render the PNGs on this many worker processes (minimum 1). Output is "
        "byte-identical whatever the count; costs ~780 MB of RSS per worker",
    )
    parser.add_argument("--closed_loop_fps", type=int, default=10)
    parser.add_argument("--closed_loop_near_miss_thresh", type=float, default=0.5)
    parser.add_argument("--closed_loop_search_radius", type=float, default=1.5)
    parser.add_argument("--closed_loop_warmup_steps", type=int, default=0)
    parser.add_argument("--closed_loop_unstick_after", type=int, default=300)
    parser.add_argument("--closed_loop_unstick_advance_m", type=float, default=5.0)
    parser.add_argument("--closed_loop_unstick_radius_mult", type=float, default=10.0)
    parser.add_argument("--closed_loop_unstick_teleport_after", type=int, default=300)
    parser.add_argument(
        "--closed_loop_abort_deviation_m",
        type=float,
        default=50.0,
        help="early-abort a segment (terminated='diverged') once GT deviation exceeds this "
        "(m) for --closed_loop_abort_after steps (0=disabled)",
    )
    parser.add_argument("--closed_loop_abort_after", type=int, default=30)
    parser.add_argument(
        "--closed_loop_abort_max_snaps",
        type=int,
        default=0,
        help="early-abort a segment after this many unstick teleports (0=disabled)",
    )
    parser.add_argument(
        "--closed_loop_wandb_video_pick",
        choices=("worst", "first", "longest"),
        default="worst",
        help="which single episode per site gets its video + trajectory colormap uploaded "
        "to wandb (all episodes still get rendered to out_dir either way)",
    )
    parser.add_argument(
        "--closed_loop_colormap_metrics",
        nargs="*",
        choices=(
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ),
        default=[
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ],
        help="per-step metrics rendered as trajectory-colormap images for the wandb "
        "representative episode (one image each — cheap, unlike video/episode picking, so "
        "all of them render by default)",
    )
    parser.add_argument(
        "--closed_loop_report_base_url",
        type=str,
        default="",
        help="if the out_dir tree is served over HTTP from this base, wandb records a "
        "clickable report URL instead of just the local path",
    )

    # Scenario-based Open-loop validation. The list selects samples per metric; metric parameters
    # are regular TrainConfig fields.
    parser.add_argument(
        "--scenario_based_open_loop_list",
        type=str,
        default="",
        help="JSON mapping Scenario-based Open-loop metric names to NPZ path lists. Empty = disabled.",
    )
    parser.add_argument(
        "--scenario_centerline_horizon_seconds",
        type=float,
        default=_train_config_default("scenario_centerline_horizon_seconds"),
    )
    parser.add_argument(
        "--scenario_departure_horizon_seconds",
        type=float,
        default=_train_config_default("scenario_departure_horizon_seconds"),
    )
    parser.add_argument(
        "--scenario_departure_minimum_displacement_m",
        type=float,
        default=_train_config_default("scenario_departure_minimum_displacement_m"),
    )

    # Deterministic
    parser.add_argument(
        "--deterministic",
        type=boolean,
        default=True,
        help="Set True to run PyTorch GPU kernels in deterministic mode (may be slightly slower).",
    )
    args = parser.parse_args(args_list)
    return args


def main():
    args = get_args()
    args_dict = vars(args)
    train_config = TrainConfig(**args_dict)

    train_config.state_normalizer = StateNormalizer.from_json(train_config)
    train_config.observation_normalizer = ObservationNormalizer.from_json(train_config)

    model_training(train_config)


if __name__ == "__main__":
    main()
