"""Build RewardConfig from a GRPO config JSON file.

Maps reward-related fields from the config JSON to RewardConfig.
Fields not present in the JSON fall back to RewardConfig dataclass defaults.
Note: only maps commonly-used reward fields, not every RewardConfig attribute.
"""

import json
from pathlib import Path
from rlvr.reward import RewardConfig


def load_reward_config(config_path: str | Path) -> RewardConfig:
    """Load RewardConfig from a GRPO experiment config JSON.

    Reads reward-related fields from the config and builds a RewardConfig
    that matches what the trainer uses during GRPO.

    Args:
        config_path: Path to grpo_config.json or experiment config JSON.

    Returns:
        RewardConfig with all fields set from the config.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    # Use RewardConfig dataclass defaults for any missing fields
    defaults = RewardConfig()
    return RewardConfig(
        w_safety=cfg.get("w_safety", defaults.w_safety),
        w_progress=cfg.get("w_progress", defaults.w_progress),
        w_smooth=cfg.get("w_smooth", defaults.w_smooth),
        w_feasibility=cfg.get("w_feasibility", defaults.w_feasibility),
        w_centerline=cfg.get("w_centerline", defaults.w_centerline),
        near_edge_scale=cfg.get("near_edge_scale", defaults.near_edge_scale),
        wide_edge_scale=cfg.get("wide_edge_scale", defaults.wide_edge_scale),
        cont_edge_scale=cfg.get("cont_edge_scale", defaults.cont_edge_scale),
        enable_lane_departure=cfg.get("enable_lane_departure", defaults.enable_lane_departure),
        lane_gate_enabled=cfg.get("lane_gate_enabled", defaults.lane_gate_enabled),
        lane_near_scale=cfg.get("lane_near_scale", defaults.lane_near_scale),
        lane_wide_scale=cfg.get("lane_wide_scale", defaults.lane_wide_scale),
        lane_cont_scale=cfg.get("lane_cont_scale", defaults.lane_cont_scale),
        max_lat_accel=cfg.get("max_lat_accel", defaults.max_lat_accel),
        lat_accel_scale=cfg.get("lat_accel_scale", defaults.lat_accel_scale),
        enable_overprogress=cfg.get("enable_overprogress", defaults.enable_overprogress),
        overprogress_margin=cfg.get("overprogress_margin", defaults.overprogress_margin),
        overprogress_penalty=cfg.get("overprogress_penalty", defaults.overprogress_penalty),
        stopped_penalty=cfg.get("stopped_penalty", defaults.stopped_penalty),
        underprogress_penalty=cfg.get("underprogress_penalty", defaults.underprogress_penalty),
        underprogress_threshold=cfg.get("underprogress_threshold", defaults.underprogress_threshold),
        progress_norm_scale=cfg.get("progress_norm_scale", defaults.progress_norm_scale),
        reward_mode=cfg.get("reward_mode", defaults.reward_mode),
    )
