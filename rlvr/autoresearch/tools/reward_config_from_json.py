"""Build RewardConfig from a GRPO config JSON file.

Ensures cleaning, visualization, and eval tools use the EXACT same
reward settings as training. Import and call load_reward_config(path).
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

    return RewardConfig(
        w_safety=cfg.get("w_safety", 5.0),
        w_progress=cfg.get("w_progress", 2.0),
        w_smooth=cfg.get("w_smooth", 0.5),
        w_feasibility=cfg.get("w_feasibility", 5.0),
        w_centerline=cfg.get("w_centerline", 5.0),
        near_edge_scale=cfg.get("near_edge_scale", 5.0),
        wide_edge_scale=cfg.get("wide_edge_scale", 0.5),
        cont_edge_scale=cfg.get("cont_edge_scale", 0.0),
        enable_lane_departure=cfg.get("enable_lane_departure", False),
        lane_gate_enabled=cfg.get("lane_gate_enabled", False),
        lane_near_scale=cfg.get("lane_near_scale", 3.0),
        lane_wide_scale=cfg.get("lane_wide_scale", 0.2),
        lane_cont_scale=cfg.get("lane_cont_scale", 0.0),
        max_lat_accel=cfg.get("max_lat_accel", 2.0),
        lat_accel_scale=cfg.get("lat_accel_scale", 5.0),
        enable_overprogress=cfg.get("enable_overprogress", False),
        overprogress_margin=cfg.get("overprogress_margin", 1.0),
        overprogress_penalty=cfg.get("overprogress_penalty", 3.0),
        stopped_penalty=cfg.get("stopped_penalty", 100.0),
        underprogress_penalty=cfg.get("underprogress_penalty", 200.0),
        underprogress_threshold=cfg.get("underprogress_threshold", 0.5),
        progress_norm_scale=cfg.get("progress_norm_scale", 10.0),
        reward_mode=cfg.get("reward_mode", "gate"),
    )
