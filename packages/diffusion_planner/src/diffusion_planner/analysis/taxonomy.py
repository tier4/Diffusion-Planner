"""Finer buckets for attention records than the token block alone.

A block tells you a token is a lane; it does not tell you whether that lane is
bounded by a crosswalk or a guard rail, nor whether an agent is ahead of the ego
or behind it. These helpers add those distinctions so trends can be stated about
the things people actually ask about.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.dimensions import LANE_TYPE_DIM

__all__ = [
    "BEARING_BUCKETS",
    "BOUNDARY_BUCKETS",
    "BOUNDARY_TYPE_NAMES",
    "bearing_bucket",
    "lane_boundary_buckets",
    "nearest_route_rank",
    "annotate_records",
    "bucket_shares",
]

BOUNDARY_TYPE_NAMES: tuple[str, ...] = (
    "crosswalk",
    "curbstone",
    "guard_rail",
    "line_thick",
    "line_thin",
    "pedestrian_marking",
    "road_border",
    "road_shoulder",
    "virtual",
    "zebra_marking",
)
"""Boundary types encoded in each half of ``lane_types``."""

BOUNDARY_BUCKETS: dict[str, tuple[str, ...]] = {
    "pedestrian": ("crosswalk", "pedestrian_marking", "zebra_marking"),
    "kerb": ("curbstone", "road_shoulder"),
    "barrier": ("guard_rail", "road_border"),
    "marking": ("line_thick", "line_thin"),
    "virtual": ("virtual",),
}
"""Groupings of boundary type.

These are **boundary** types, not lane semantics: the schema has no "sidewalk
lane" label. ``pedestrian`` is the closest available proxy for pedestrian
infrastructure and should be reported as such. A lane may carry several
boundaries, so the buckets overlap and do not partition lane attention.
"""

BEARING_BUCKETS: tuple[str, ...] = ("ahead", "left", "right", "behind")
"""Direction of a token relative to the ego heading, which is ego-frame +x."""


def bearing_bucket(x_m: float, y_m: float) -> str:
    """Name the quadrant a point falls in, relative to the ego heading."""
    angle = math.degrees(math.atan2(y_m, x_m))
    if abs(angle) <= 45.0:
        return "ahead"
    if abs(angle) > 135.0:
        return "behind"
    return "left" if angle > 0.0 else "right"


def lane_boundary_buckets(lane_types: Any) -> list[set[str]]:
    """Which boundary buckets each lane segment carries.

    ``lane_types`` is ``(L, 20)``: a left-boundary one-hot followed by a
    right-boundary one-hot, each over :data:`BOUNDARY_TYPE_NAMES`.
    """
    types = np.asarray(lane_types, dtype=np.float32)
    if types.shape[-1] != LANE_TYPE_DIM:
        raise ValueError(
            f"lane_types last dimension is {types.shape[-1]}, expected {LANE_TYPE_DIM}"
        )
    half = LANE_TYPE_DIM // 2
    present = (types[..., :half] > 0.0) | (types[..., half:] > 0.0)

    lookup = {
        name: [BOUNDARY_TYPE_NAMES.index(item) for item in members]
        for name, members in BOUNDARY_BUCKETS.items()
    }
    buckets: list[set[str]] = []
    for row in present:
        found = {
            bucket
            for bucket, indices in lookup.items()
            if any(bool(row[index]) for index in indices)
        }
        buckets.append(found)
    return buckets


def annotate_records(
    records: Sequence[dict[str, Any]], frame: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Add bearing and lane-boundary buckets to attention records.

    Returns new dicts; the input records are left alone.
    """
    lane_buckets = lane_boundary_buckets(frame["lane_types"])
    annotated: list[dict[str, Any]] = []
    for record in records:
        entry = dict(record)
        if record["x_m"] is not None and record["y_m"] is not None:
            entry["bearing"] = bearing_bucket(record["x_m"], record["y_m"])
            if record["block"] == "neighbors":
                entry["class_bearing"] = (
                    f"{record.get('agent_class', 'unknown')}·{entry['bearing']}"
                )
        if record["block"] == "lanes":
            index = record["block_index"]
            if 0 <= index < len(lane_buckets):
                for bucket in BOUNDARY_BUCKETS:
                    entry[f"lane_{bucket}"] = bucket in lane_buckets[index]
        annotated.append(entry)
    return annotated


def nearest_route_rank(
    records: Sequence[dict[str, Any]],
) -> tuple[int | None, float | None]:
    """Rank of the route token nearest the ego, among all valid tokens.

    "Attention to the immediate route is consistently first" is a claim about
    rank, which a share-based statistic cannot express.

    Returns:
        The 1-based rank and the token's attention percentage, or ``(None, None)``
        when the frame carries no route token with a position.
    """
    ordered = sorted(records, key=lambda item: -item["attention"])
    route = [
        (index, record)
        for index, record in enumerate(ordered)
        if record["block"] == "route_lanes" and record["distance_m"] is not None
    ]
    if not route:
        return None, None
    index, record = min(route, key=lambda item: item[1]["distance_m"])
    return index + 1, float(record["attention_pct"])


def bucket_shares(
    records: Sequence[dict[str, Any]], flags: Sequence[str], within_block: str
) -> dict[str, float]:
    """Share of one block's attention carried by each boolean flag.

    Flags may overlap, so the shares need not sum to one.
    """
    total = sum(
        record["attention"] for record in records if record["block"] == within_block
    )
    if total <= 0.0:
        return dict.fromkeys(flags, 0.0)
    return {
        flag: sum(
            record["attention"]
            for record in records
            if record["block"] == within_block and record.get(flag)
        )
        / total
        for flag in flags
    }
