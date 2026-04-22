"""Build RewardConfig from a GRPO config JSON file.

The JSON can be a serialized ``GRPOConfig`` (as produced by
``GRPOConfig.to_json``) or any reward-relevant JSON subset. Keys that are
not ``RewardConfig`` fields are silently dropped (lets the same file carry
training-only fields like ``learning_rate`` or ``_comment_*``).

Required reward fields (thresholds / scales / weights / gates that change
training behaviour) must be present in the JSON — loading raises
``ValueError`` if any are missing. This prevents evaluation or diagnostic
tools from silently using ``RewardConfig`` defaults that diverge from what
the training run actually used.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from rlvr.reward import RewardConfig

# Legacy field renames — mirrors GRPOConfig.from_json.
_LEGACY_KEYS = {
    "near_edge_scale": "rb_near_scale",
    "wide_edge_scale": "rb_wide_scale",
    "cont_edge_scale": "rb_cont_scale",
}

# Fields that MUST be present in the config JSON. These all change scoring
# semantics — a missing field here would silently fall back to a
# ``RewardConfig`` default that almost certainly disagrees with the training
# run. Serialised configs from ``GRPOConfig.to_json`` always have all of them.
_REQUIRED_REWARD_FIELDS = (
    "reward_mode",
    "rb_gate_enabled",
    "rb_cross_thresh",
    "rb_near_thresh",
    "rb_wide_thresh",
    "rb_cont_thresh",
    "rb_near_scale",
    "rb_wide_scale",
    "rb_cont_scale",
    "w_progress",
    "w_centerline",
    "w_safety",
    "w_smooth",
    "w_feasibility",
    "stopped_penalty",
    "enable_lane_departure",
)


def load_reward_config(config_path: str | Path) -> RewardConfig:
    """Load a RewardConfig from a GRPO/experiment config JSON.

    Uses the intersection of JSON keys and ``RewardConfig`` dataclass fields
    so new reward fields become configurable automatically as soon as they
    exist on both sides. Raises ``ValueError`` if any field in
    ``_REQUIRED_REWARD_FIELDS`` is missing from the JSON.
    """
    with open(config_path) as f:
        raw = json.load(f)

    for old, new in _LEGACY_KEYS.items():
        if old in raw and new not in raw:
            raw[new] = raw[old]

    missing = [k for k in _REQUIRED_REWARD_FIELDS if k not in raw]
    if missing:
        raise ValueError(
            f"Reward config {config_path} is missing required fields: {missing}. "
            "Set them explicitly to match the training run — silent fallback "
            "to RewardConfig defaults would produce scoring that disagrees "
            "with training."
        )

    reward_field_names = {f.name for f in fields(RewardConfig)}
    kwargs = {k: v for k, v in raw.items() if k in reward_field_names}
    return RewardConfig(**kwargs)
