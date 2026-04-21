"""Build RewardConfig from a GRPO config JSON file.

The JSON can be a serialized ``GRPOConfig`` (as produced by
``GRPOConfig.to_json``) or any reward-relevant JSON subset. Keys that are
not ``RewardConfig`` fields are silently dropped (lets the same file carry
training-only fields like ``learning_rate`` or ``_comment_*``).

Fields missing from the JSON fall back to ``RewardConfig`` dataclass
defaults.
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


def load_reward_config(config_path: str | Path) -> RewardConfig:
    """Load a RewardConfig from a GRPO/experiment config JSON.

    Uses the intersection of JSON keys and ``RewardConfig`` dataclass fields
    so new reward fields become configurable automatically as soon as they
    exist on both sides.
    """
    with open(config_path) as f:
        raw = json.load(f)

    for old, new in _LEGACY_KEYS.items():
        if old in raw and new not in raw:
            raw[new] = raw[old]

    reward_field_names = {f.name for f in fields(RewardConfig)}
    kwargs = {k: v for k, v in raw.items() if k in reward_field_names}
    return RewardConfig(**kwargs)
