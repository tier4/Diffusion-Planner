from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config_cli import cli


@dataclass
class ClosedLoopPassCondition:
    """Pass conditions for closed-loop evaluation. Each boolean field controls whether
    that metric's count must be zero for a segment to pass.

    A condition being True means "enabled" - the segment must have 0 events of that type.
    A condition being False means "disabled" - that metric is ignored when deciding pass/fail.
    """

    collision: bool = True  # object.collision_count == 0
    road_border: bool = True  # road_border.collision_count == 0
    red_light_violation: bool = True  # red_light_violation.count == 0
    strong_brake: bool = True  # strong_brake.count == 0
    goal_reach: bool = True
    snap: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClosedLoopPassCondition":
        valid_fields = {
            "collision",
            "road_border",
            "red_light_violation",
            "strong_brake",
            "goal_reach",
            "snap",
        }
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ClosedLoopPassConditionGroups:
    """Container for per-group pass conditions. Supports loading from YAML.

    Usage:
        # Direct construction
        groups = ClosedLoopPassConditionGroups()

        # From YAML file
        groups = ClosedLoopPassConditionGroups.from_yaml("/path/to/config.yaml")

        # Get condition for a specific group (falls back to default)
        condition = groups.get_condition("pedestrian_stop")
    """

    default: ClosedLoopPassCondition = field(default_factory=ClosedLoopPassCondition)
    groups: dict[str, ClosedLoopPassCondition] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ClosedLoopPassConditionGroups":
        """Load pass conditions from a YAML file.

        YAML format:
            default:
              collision: true
              road_border: true
              red_light_violation: true
              strong_brake: true
              goal_reach: true
              snap: true

            groups:
              pedestrian_stop:
                collision: true
                strong_brake: false  # ignore strong_brake for this group
              parking_lot:
                road_border: false
                snap: false
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Pass condition config not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        default = ClosedLoopPassCondition()
        if "default" in data:
            default = ClosedLoopPassCondition.from_dict(data["default"])

        groups: dict[str, ClosedLoopPassCondition] = {}
        if "groups" in data:
            for group_name, conditions in data["groups"].items():
                # Merge with default: specified fields override, rest come from default
                merged = default.to_dict()
                merged.update(conditions)
                groups[group_name] = ClosedLoopPassCondition.from_dict(merged)

        return cls(default=default, groups=groups)

    def get_condition(self, group_name: str) -> ClosedLoopPassCondition:
        """Get the pass condition for a group. Falls back to default if not found."""
        return self.groups.get(group_name, self.default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "default": self.default.to_dict(),
            "groups": {k: v.to_dict() for k, v in self.groups.items()},
        }


@dataclass
class ClosedLoopConfig:
    closed_loop_npz_root: list[str] = cli(
        "JSON file(s) or folder(s) for closed-loop validation. "
        "Empty = disabled. Supports: folder, flat JSON (list), grouped JSON (dict). "
        "Multiple inputs each become their own top-level namespace.",
        default_factory=list,
        path=True,
    )
    closed_loop_object_modes: list[str] = cli(
        "object-mode(s): 'objects'=normal, 'noobj'=empty-world ablation",
        default_factory=lambda: ["objects"],
    )
    device: str = cli("device for model and evaluation", default="cuda")
    # Mirror BaseConfig.ddp so ddp_setup_universal(...) can be called on this config
    # directly; ``True`` here means "respect RANK/WORLD_SIZE if set" (single-process
    # CLI runs with no torchrun env vars stay non-distributed).
    ddp: bool = cli("enable DDP when RANK/WORLD_SIZE are present", default=True)
    render_media: bool = cli(
        "render video/colormap artifacts during wandb logging",
        default=True,
    )
    wandb_project_name: str = cli("Weights & Biases project name (empty=disabled)", default="")
    exp_name: str = cli("name of this run; appears in the save directory and in wandb", default="")
    port: str = "22323"

    # FullRouteClosedLoopEvaluation
    closed_loop_seg_len: int = 100000
    # ClosedLoopEvalConfig
    closed_loop_fps: int = 10
    # RolloutParams
    closed_loop_near_miss_thresh: float = 0.5
    closed_loop_search_radius: float = 1.5
    closed_loop_warmup_steps: int = 0
    closed_loop_unstick_after: int = 50
    closed_loop_unstick_advance_m: float = 1.5
    closed_loop_unstick_radius_mult: float = 3.0
    closed_loop_unstick_teleport_after: int = 50
    closed_loop_draw_every: int = 2
    closed_loop_draw_workers: int = cli("render on this many worker processes", default=4)
    closed_loop_replan_interval: int = 1
    closed_loop_tracker_mode: str = "mpc"
    closed_loop_neighbor_history_mode: str = "recorded"
    closed_loop_yaw_gate: bool = True
    closed_loop_strong_brake_mps2: float = -2.5
    closed_loop_abort_deviation_m: float = 50.0
    closed_loop_abort_after: int = 30
    closed_loop_abort_max_snaps: int = 0
    closed_loop_goal_mode: str = "segment"
    closed_loop_title_prefix: str | None = None
    closed_loop_distance_label_offset_m: float = 1.2
    closed_loop_view_half_m: float = 50.0
    closed_loop_max_stuck_steps: int = 0
    closed_loop_goal_reach_m: float = 5.0
    closed_loop_interpolate: bool = True
    closed_loop_color_by_uuid: bool = True
    closed_loop_window: tuple[int, int] | None = None
    closed_loop_max_steps: int | None = None
    closed_loop_timeline_progress_mode: str = "pose"

    # validation in training part
    closed_loop_wandb_video_pick: str = cli(
        "which episode gets video+colormap: 'worst'/'first'/'longest'",
        default="worst",
    )
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

    # for OpenSCENARIO evaluation
    scenario_sim_driver: str = cli(
        "shell driver that evaluates a saved checkpoint against the OpenSCENARIO suite. "
        "Empty = disabled. It receives the checkpoint and an output directory in CKPT / OUT; "
        "every other knob is its own environment's.",
        default="",
        path=True,
    )

    # Per-group pass conditions
    closed_loop_pass_conditions: str = cli(
        "Optional YAML file with per-group pass conditions. See "
        "closed_loop_pass_conditions.yaml for field definitions and defaults. "
        "Pass Conditions: collision=no collision, road_border=no curb hit, "
        "red_light_violation=no red-light running, strong_brake=no hard brake, "
        "goal_reach=reached goal, snap=no emergency fallback. Set to False to "
        "ignore that metric. Omit (default '') for all True (strict).",
        default="",
        path=True,
    )
    _closed_loop_pass_conditions_loaded: ClosedLoopPassConditionGroups | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        """Auto-load ``closed_loop_pass_conditions`` (a YAML path) into the object form.

        Always sets ``_closed_loop_pass_conditions_loaded`` to a non-None
        ``ClosedLoopPassConditionGroups``:

        - YAML path provided + file exists  → load from YAML
        - YAML path provided + file missing  → fall back to all-True default
          (typo'd path shouldn't disable pass evaluation; strict is the safer
          default — it surfaces every violation rather than silently passing
          everything)
        - no path / empty path               → all-True default

        Callers can therefore always do ``cfg.pass_conditions.get_condition(...)``
        without a None-check.
        """
        if self._closed_loop_pass_conditions_loaded is not None:
            return
        if self.closed_loop_pass_conditions:
            try:
                self._closed_loop_pass_conditions_loaded = ClosedLoopPassConditionGroups.from_yaml(
                    self.closed_loop_pass_conditions
                )
                return
            except FileNotFoundError:
                pass
        self._closed_loop_pass_conditions_loaded = ClosedLoopPassConditionGroups()

    @property
    def pass_conditions(self) -> ClosedLoopPassConditionGroups:
        return self._closed_loop_pass_conditions_loaded
