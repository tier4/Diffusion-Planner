# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single [0,1] EPDMS-*like* aggregate from the raw ``planner_metrics`` subscores.

This is an EPDMS-**structured proxy**, NOT a faithful NAVSIM EPDMS port (the faithful
one lives in OnePlanner; see issue #142). It mirrors the canonical shape -- a product
of binary gates times a weighted average of quality terms, bounded to [0,1] -- but it
differs from real EPDMS in several material ways:

    * it scores the **raw predicted waypoints**, not a controller-simulated (LQR +
      kinematic bicycle) rollout, so it is pure open-loop;
    * **DDC** (driving-direction compliance) is omitted -- not computed upstream;
    * there is **no false-positive filtering** against a human reference;
    * **comfort** uses only mean |jerk| (the repo's ``comfort`` subscore), not the full
      lon/lat-accel + yaw-rate + extended-comfort battery;
    * **TTC** is a graded fraction of safe steps, not the binary constant-velocity check;
    * **progress** is normalized against the GT (expert) path length.

So treat it as a closed-loop-correlated *proxy* for ranking/debugging checkpoints, not
as the metric NAVSIM or OnePlanner reports.

The subscores from :func:`planner_metrics.aggregate.compute_subscores_batch` live on
heterogeneous scales, and the penalty-style terms (``safety``, ``comfort``,
``centerline``, ``feasibility``, ``red_light``) follow the reward convention
``0 = perfect, more negative = worse``. EPDMS instead wants a *score* in ``[0,1]``
(``1 = perfect``). This module flips and bounds each term and combines them::

    EPDMS_like = (PROD of binary gates in {0,1}) * (weighted avg of quality terms in [0,1])

Gates (multiplicative, hard-zero the score):
    * NC  -- no at-fault collision         (from ``collision_step``)
    * DAC -- drivable-area compliance      (``rb_crossing_gate``)
    * TLC -- traffic-light compliance      (``red_light`` == 0)
    * KIN -- kinematic feasibility         (``kinematic_gate``)
    * optionally ``lane_crossing_gate`` / ``sc_crossing_gate`` when enabled upstream
      (they are 1.0 when their subscore feature is disabled, so including them is safe).

Quality terms (weighted average, each renormalized to [0,1], higher = better):
    * TTC      -- ``ttc`` (already a fraction of TTC-safe steps in [0,1])
    * progress -- ``progress`` (metres) normalized by the expert's GT path length
    * comfort  -- binary: mean |jerk| (= ``-comfort``) within ``jerk_bound``
    * lane     -- binary: lane usage (= sqrt(-``centerline``)) within ``lane_usage_bound``

NOTE: the thresholds/weights below must be calibrated so ground-truth (expert)
trajectories score near 1.0 -- mirror the calibration note on
``compute_kinematic_gate`` (GT must pass the gates).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EPDMSLikeConfig:
    # --- quality thresholds (binary terms) ---
    # Mean |jerk| (m/s^3) at or below this scores comfort = 1. Calibrate so GT passes
    # (human driving mean |jerk| is typically well under this).
    jerk_bound: float = 2.0
    # Lane usage = |rear-axle lateral offset| / lane half-width. <= 1.0 means the rear
    # axle is within the lane boundary.
    lane_usage_bound: float = 1.0

    # --- quality weights (weighted average -> [0,1]) ---
    w_ttc: float = 5.0
    w_progress: float = 5.0
    w_comfort: float = 2.0
    w_lane: float = 5.0

    # Minimum GT-progress denominator (metres) so near-stationary scenes don't blow up
    # the progress ratio. A scene where the expert barely moves contributes progress_q
    # ~= 1 for any non-regressing plan.
    progress_eps: float = 1.0

    # Multiply in lane_crossing_gate / sc_crossing_gate when present (1.0 when the
    # corresponding feature is disabled upstream, so this is safe to leave on).
    include_optional_gates: bool = True


def _as_gate(subscores: dict, key: str, n: int, device, dtype) -> torch.Tensor:
    """Fetch a {0,1} gate tensor, defaulting to all-ones (pass) when absent."""
    g = subscores.get(key)
    if g is None:
        return torch.ones(n, device=device, dtype=dtype)
    if not torch.is_tensor(g):
        g = torch.as_tensor(g, device=device, dtype=dtype)
    return g.to(device=device, dtype=dtype).reshape(-1)


def _collision_gate(subscores: dict, n: int, device, dtype) -> torch.Tensor:
    """NC gate from ``collision_step`` (list of None|int per scene, or a tensor)."""
    coll = subscores.get("collision_step")
    if coll is None:
        return torch.ones(n, device=device, dtype=dtype)
    if torch.is_tensor(coll):
        # Convention upstream: a negative / sentinel value means "no collision".
        return (coll.reshape(-1) < 0).to(device=device, dtype=dtype)
    return torch.tensor([1.0 if c is None else 0.0 for c in coll], device=device, dtype=dtype)


def epdms_like_aggregate(
    subscores: dict,
    gt_progress: torch.Tensor,
    cfg: EPDMSLikeConfig | None = None,
):
    """Collapse the raw subscore dict into a single [0,1] EPDMS-like score per scene.

    Args:
        subscores: output of ``compute_subscores_batch`` -- batched ``(N,)`` tensors
            for ``ttc``/``progress``/``comfort``/``centerline``/``red_light`` plus the
            gate fields (``rb_crossing_gate``, ``kinematic_gate``, ``collision_step``,
            optionally ``lane_crossing_gate``/``sc_crossing_gate``).
        gt_progress: ``(N,)`` expert (ground-truth) path length in metres, used as the
            progress-normalization reference. See :func:`gt_path_length`.
        cfg: thresholds/weights. Defaults to :class:`EPDMSLikeConfig`.

    Returns:
        ``(score, components)`` where ``score`` is ``(N,)`` in [0,1] and ``components``
        is a dict of the per-scene gate values and normalized quality terms (all
        ``(N,)``) for logging/debugging.
    """
    cfg = cfg or EPDMSLikeConfig()

    ttc = subscores["ttc"].reshape(-1)
    device, dtype = ttc.device, ttc.dtype
    n = ttc.shape[0]

    gt_progress = torch.as_tensor(gt_progress, device=device, dtype=dtype).reshape(-1)

    # ---- gates (binary, multiplicative) ----
    nc = _collision_gate(subscores, n, device, dtype)  # no collision
    dac = _as_gate(subscores, "rb_crossing_gate", n, device, dtype)  # drivable area
    kin = _as_gate(subscores, "kinematic_gate", n, device, dtype)  # feasibility
    red_light = subscores.get("red_light")
    if red_light is None:
        tlc = torch.ones(n, device=device, dtype=dtype)
    else:
        # red_light == 0 -> clean; negative -> violation.
        tlc = (red_light.reshape(-1) >= -1e-6).to(dtype)

    gates = nc * dac * kin * tlc
    if cfg.include_optional_gates:
        gates = (
            gates
            * _as_gate(subscores, "lane_crossing_gate", n, device, dtype)
            * _as_gate(subscores, "sc_crossing_gate", n, device, dtype)
        )

    # ---- quality terms (each renormalized to [0,1]) ----
    ttc_q = ttc.clamp(0.0, 1.0)

    progress = subscores["progress"].reshape(-1)
    denom = gt_progress.clamp(min=cfg.progress_eps)
    progress_q = (progress / denom).clamp(0.0, 1.0)

    # comfort = -mean_abs_jerk  ->  mean_abs_jerk = -comfort
    comfort = subscores["comfort"].reshape(-1)
    comfort_q = ((-comfort) <= cfg.jerk_bound).to(dtype)

    # centerline = -lane_usage^2  ->  lane_usage = sqrt(max(0, -centerline))
    centerline = subscores["centerline"].reshape(-1)
    lane_usage = (-centerline).clamp(min=0.0).sqrt()
    lane_q = (lane_usage <= cfg.lane_usage_bound).to(dtype)

    w_sum = cfg.w_ttc + cfg.w_progress + cfg.w_comfort + cfg.w_lane
    if w_sum <= 0:
        raise ValueError(
            "EPDMSLikeConfig quality weights (w_ttc/w_progress/w_comfort/w_lane) "
            "must sum to a positive value"
        )
    quality = (
        cfg.w_ttc * ttc_q
        + cfg.w_progress * progress_q
        + cfg.w_comfort * comfort_q
        + cfg.w_lane * lane_q
    ) / w_sum

    score = gates * quality

    components = {
        "epdms_like": score,
        "gate_nc": nc,
        "gate_dac": dac,
        "gate_tlc": tlc,
        "gate_kin": kin,
        "q_ttc": ttc_q,
        "q_progress": progress_q,
        "q_comfort": comfort_q,
        "q_lane": lane_q,
        "quality": quality,
    }
    return score, components


def gt_path_length(ego_future_xy: torch.Tensor) -> torch.Tensor:
    """Cumulative planar path length of the expert future, used as the progress
    reference. ``ego_future_xy`` is ``(N, T, >=2)`` (x, y, ...); returns ``(N,)``."""
    xy = ego_future_xy[..., :2]
    steps = torch.linalg.norm(xy[:, 1:, :] - xy[:, :-1, :], dim=-1)  # (N, T-1)
    return steps.sum(dim=-1)


__all__ = ["EPDMSLikeConfig", "epdms_like_aggregate", "gt_path_length"]
