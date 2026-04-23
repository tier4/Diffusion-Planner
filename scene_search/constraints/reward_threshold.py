"""Constraint: filter replay scenes by per-step drift metrics.

Reads values straight from the index ``entry["metrics"]`` dict populated by
``scene_search.replay_index.load_replay_run`` — NOT from the NPZ. This
avoids duplicating the reward computation (which already lives in
``rlvr.reward`` and was run at sim time by ``scenario_generation.replay``).

Only fires on replay-backed scenes; entries without a metrics dict (i.e.
sidecar-backed scenes from rosbag NPZs) are dropped when this constraint
is enabled.
"""

from __future__ import annotations

from scene_search.constraints.base import BaseConstraint
from scene_search.constraints.registry import register


@register("reward_threshold")
class RewardThresholdConstraint(BaseConstraint):
    name = "Reward Thresholds"
    description = ("Filter replay scenes by live drift metrics (rb / cl / "
                   "lane gate / lane_near_frac). Requires replay runs loaded "
                   "via --replay_runs.")

    def get_params_spec(self) -> dict:
        return {
            "rb_min_dist_max": {"type": "float", "default": 0.6,
                                "label": "Max rb_min_dist (m)  [drift if ≤]",
                                "min": 0.0, "max": 10.0, "step": 0.05},
            "abs_cl_score_min": {"type": "float", "default": 0.5,
                                 "label": "Min |cl_score|  [drift if ≥]",
                                 "min": 0.0, "max": 3.0, "step": 0.05},
            "require_lane_cross": {"type": "float", "default": 0.0,
                                   "label": "Require lane_gate=0? (1=yes, 0=no)",
                                   "min": 0.0, "max": 1.0, "step": 1.0},
            "lane_near_frac_min": {"type": "float", "default": 0.0,
                                   "label": "Min lane_near_frac  [drift if ≥]",
                                   "min": 0.0, "max": 1.0, "step": 0.05},
        }

    def filter(self, npz_path: str, npz_data, params: dict,
               entry: dict | None = None) -> bool:
        # npz_data kept for BaseConstraint signature compatibility but
        # unused here — reward_threshold reads from entry["metrics"] only.
        if entry is None:
            return False  # not a replay entry; drop
        metrics = entry.get("metrics") or {}
        if not metrics:
            return False

        rb = metrics.get("rb_min_dist")
        cl = metrics.get("cl_score")
        gate = metrics.get("lane_gate")
        near = metrics.get("lane_near_frac")

        # A scene passes when it satisfies EVERY enabled (non-trivial) test.
        # "Enabled" = the param is tighter than its no-op default:
        #   rb_min_dist_max < 10.0   → rb must be ≤ max
        #   abs_cl_score_min > 0.0   → |cl| must be ≥ min
        #   require_lane_cross ≥ 0.5 → gate must be 0
        #   lane_near_frac_min > 0.0 → near must be ≥ min
        if params["rb_min_dist_max"] < 10.0:
            if rb is None or rb > params["rb_min_dist_max"]:
                return False
        if params["abs_cl_score_min"] > 0.0:
            if cl is None or abs(cl) < params["abs_cl_score_min"]:
                return False
        if params["require_lane_cross"] >= 0.5:
            if gate is None or gate >= 0.5:
                return False
        if params["lane_near_frac_min"] > 0.0:
            if near is None or near < params["lane_near_frac_min"]:
                return False
        return True
