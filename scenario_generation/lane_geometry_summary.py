"""Pooling of the route / center deviation summary keys across segments or groups."""

from __future__ import annotations

import math

# (mean key, unmeasured-rate key) pairs of the per-step lane-geometry metrics; see
# ``reproducer_rollout._lane_geometry_block`` for what each is.
_LANE_GEOMETRY_MEANS = (
    ("mean_route_deviation_m", "route_deviation_unmeasured_rate"),
    ("mean_center_deviation_m", "center_deviation_unmeasured_rate"),
    ("mean_center_deviation_signed_m", "center_deviation_unmeasured_rate"),
)
_LANE_GEOMETRY_RATES = ("route_deviation_unmeasured_rate", "center_deviation_unmeasured_rate")
LANE_GEOMETRY_KEYS = (
    *(m for m, _ in _LANE_GEOMETRY_MEANS),
    *_LANE_GEOMETRY_RATES,
    "max_center_deviation_m",
)


def pool_lane_geometry(items, steps_key: str) -> dict:
    """Pool segment/group summaries' route / center deviation keys into one summary's worth.

    Each mean is weighted by the steps it was actually measured on (``steps_key`` x
    ``1 - unmeasured_rate``), so the pooled mean equals the mean over every measured step.
    Unmeasured rates are step-weighted, the max is the max. ``inf`` = nothing measured.
    """
    items = list(items)
    out: dict = {}
    for mean_key, rate_key in _LANE_GEOMETRY_MEANS:
        num = den = 0.0
        for it in items:
            v = it.get(mean_key)
            w = float(it.get(steps_key, 0) or 0) * (1.0 - float(it.get(rate_key, 0.0) or 0.0))
            if v is not None and math.isfinite(float(v)) and w > 0:
                num += float(v) * w
                den += w
        out[mean_key] = num / den if den else float("inf")
    total = sum(float(it.get(steps_key, 0) or 0) for it in items)
    for rate_key in _LANE_GEOMETRY_RATES:
        out[rate_key] = (
            sum(float(it.get(steps_key, 0) or 0) * float(it.get(rate_key, 0.0) or 0.0) for it in items)
            / total
            if total
            else 0.0
        )
    maxes = [
        float(it["max_center_deviation_m"])
        for it in items
        if it.get("max_center_deviation_m") is not None
        and math.isfinite(float(it["max_center_deviation_m"]))
    ]
    out["max_center_deviation_m"] = max(maxes) if maxes else float("inf")
    return out
