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
    description = (
        "Filter replay scenes by live drift metrics (rb / cl / "
        "lane gate / lane_near_frac). Requires replay runs loaded "
        "via --replay_runs."
    )

    def get_params_spec(self) -> dict:
        return {
            "rb_min_dist_max": {
                "type": "float",
                "default": 0.6,
                "label": "Max rb_min_dist (m)  [drift if ≤]",
                "min": 0.0,
                "max": 10.0,
                "step": 0.05,
            },
            "abs_cl_score_min": {
                "type": "float",
                "default": 0.5,
                "label": "Min |cl_score|  [drift if ≥]",
                "min": 0.0,
                "max": 3.0,
                "step": 0.05,
            },
            "require_lane_cross": {
                "type": "float",
                "default": 0.0,
                "label": "Require lane_gate=0? (1=yes, 0=no)",
                "min": 0.0,
                "max": 1.0,
                "step": 1.0,
            },
            "lane_near_frac_min": {
                "type": "float",
                "default": 0.0,
                "label": "Min lane_near_frac  [drift if ≥]",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
        }

    def filter(self, npz_path: str, npz_data, params: dict, entry: dict | None = None) -> bool:
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

        # A scene passes when it satisfies EVERY test below. The UI
        # defaults for ``rb_min_dist_max`` (0.6) and ``abs_cl_score_min``
        # (0.5) are already meaningful drift thresholds — treating the
        # "default = no-op" pattern here would require the user to dial
        # them to 10.0 / 0.0 to see the intended drift-only filter, which
        # is surprising. The two fraction-style fields still use a >0
        # gate because their no-op (0.0) IS the "don't filter on this"
        # setting a user would naturally pick.
        if rb is None or rb > params["rb_min_dist_max"]:
            return False
        if cl is None or abs(cl) < params["abs_cl_score_min"]:
            return False
        if params["require_lane_cross"] >= 0.5:
            if gate is None or gate >= 0.5:
                return False
        if params["lane_near_frac_min"] > 0.0:
            if near is None or near < params["lane_near_frac_min"]:
                return False
        return True
