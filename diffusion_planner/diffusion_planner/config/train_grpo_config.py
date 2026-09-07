from dataclasses import dataclass, field

from .config_cli import cli
from .train_config import TrainConfig


@dataclass
class GRPOConfig(TrainConfig):
    batch_size: int = 64
    learning_rate: float = 5e-6
    train_epochs: int = 30
    save_utd: int = 1
    closed_loop_unstick_advance_m: float = 2.5
    train_subsample_step: int = 10

    # GRPO-specific sampling
    num_generations: int = cli("N: trajectories sampled per scene (the GRPO group size)", default=8)
    grpo_noise_scale: float = cli(
        "MAX initial-noise std during sampling; each row draws from U[0, this]",
        default=3.0,
    )
    advantage_eps: float = 1e-6

    # Reward weights
    w_collision: float = cli("weight on the neighbor-collision penalty in the reward", default=1.0)
    w_road_border: float = cli("weight on the road-border penalty in the reward", default=1.0)
    w_gt_l2: float = cli(
        "ADE weight between generated and GT ego trajectory",
        default=0.1,
    )
    w_kinematic: float = cli(
        "kinematic-feasibility penalty weight",
        default=1.0,
    )
    sft_prob: float = cli(
        "probability of supervised step instead of GRPO",
        default=0.5,
    )

    # Synthetic collider augmentation
    neighbor_inject_max: int = cli(
        "max synthetic colliders injected per scene (count ~ U[1, max])",
        default=1,
    )
    neighbor_inject_prob: float = cli(
        "per-scene probability of injecting any synthetic colliders", default=0.5
    )
    pedestrian_prob: float = cli("fraction of injected colliders that are pedestrians", default=0.3)
    bicycle_prob: float = cli("fraction of injected colliders that are bicycles", default=0.2)
    collider_keep_clear_radius: float = cli(
        "min distance collider path keeps from ego t=0 pose",
        default=3.0,
    )
    collider_straight_line: bool = True

    # Real-neighbor DB augmentation
    neighbor_db_path: str = cli(
        "path to neighbor-pattern DB; empty = use synthetic colliders",
        default="/mnt/storage_rdma/diffusion_planner/dataset/basic_dataset/neighbor_db.npz",
        path=True,
    )
    neighbor_db_collision_margin: float = cli(
        "(DB) max distance [m] from ego GT waypoint to count as colliding",
        default=10.0,
    )
    neighbor_min_collision_time: float = cli(
        "(DB) earliest future time [s] a collision may occur", default=0.8
    )
    neighbor_search_subsample: int = cli(
        "(DB) cap per-scene search to this many patterns (0=all)", default=0
    )
