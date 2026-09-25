"""Reconstruct the lane the ego is travelling on, from the NPZ lane tensor.

The tensor gives geometry and nothing else: no lanelet ids, no connectivity,
each lanelet a short polyline resampled to a fixed point count regardless of
its real length. A lateral reference valid for a whole horizon therefore has
to be rebuilt -- pick the lanelet the ego is on, then walk forward through the
successors that chain onto it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from planner_metrics.geometry import _point_to_segments_dist

_LANE_POINT_MIN_NORM = 1e-6
_SEGMENT_MIN_LENGTH = 1e-6

# A lanelet only qualifies as the ego's own lane when its tangent at the ego is
# within ~37 deg of the ego heading; a crossing road's turn lanelet that passes
# through the ego position is steeper than that.
_MIN_SOURCE_HEADING_ALIGNMENT = 0.8
# Lanelets whose centerline distance from the ego differs by less than this are
# a tie (typically two branches of a fork starting at the same point).
_SOURCE_DISTANCE_TIE_M = 0.1


@dataclass(frozen=True)
class SourceLane:
    """The ego's own lane, rebuilt as one continuous centerline.

    Attributes:
        path: ``(M, 2)`` chained centerline, the lateral reference itself.
        owners: ``(M,)`` lanelet index per path point.
        index: the lanelet the ego starts on.
        heading_aligned: False when nothing near the ego ran with it and the
            pick fell back to the nearest lanelet regardless -- possibly
            oncoming, which mirrors every lateral reading taken from ``path``.
            Callers are expected to surface this rather than score through it.
    """

    path: torch.Tensor
    owners: torch.Tensor
    index: int
    heading_aligned: bool

    def half_widths_at(self, lanes: torch.Tensor, point: torch.Tensor) -> tuple[float, float]:
        """Return the ``(left, right)`` half width where ``point`` meets the lane.

        Read at a station, since a chain can run through a taper or a merge.
        Nearest SEGMENT, not nearest vertex: a long successor's vertices can be
        metres apart, so past a joint the nearest vertex still belongs to the
        short predecessor.
        """
        point = point.to(self.path).reshape(1, 2)
        seg_p1, seg_p2 = _polyline_segments(self.path)
        segment = int(_point_to_segments_dist(point, seg_p1, seg_p2)[0].argmin())
        # Degenerate segments were dropped, so map the index back through the
        # kept mask; of a segment's two points the later one owns the station.
        kept = torch.nonzero(
            (self.path[1:] - self.path[:-1]).norm(dim=-1) > _SEGMENT_MIN_LENGTH
        ).squeeze(-1)
        return _lane_half_widths(lanes[int(self.owners[kept[segment] + 1])])

    def segments(
        self, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the reference path as ``(p1, p2)`` segments, optionally moved."""
        return _polyline_segments(self.path if device is None else self.path.to(device))


def _valid_point_mask(lane: torch.Tensor) -> torch.Tensor:
    """Return which of a lanelet's ``(P, D)`` rows are real points, not padding."""
    return lane[:, :4].abs().sum(dim=-1) > _LANE_POINT_MIN_NORM


def _polyline_segments(polyline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a polyline's non-degenerate segments as ``(p1, p2)``."""
    p1, p2 = polyline[:-1], polyline[1:]
    keep = (p2 - p1).norm(dim=-1) > _SEGMENT_MIN_LENGTH
    return p1[keep], p2[keep]


def _scene_centerlines(lanes: torch.Tensor) -> dict[int, torch.Tensor]:
    """Return each usable lanelet's centerline ``(K, 2)``, keyed by lanelet index.

    Padded-out lanelets and those without a single real segment are dropped
    here, once per scene, so the lane search below never has to guard for them.
    """
    centerlines = {}
    for index in range(lanes.shape[0]):
        centerline = lanes[index][_valid_point_mask(lanes[index])][:, :2]
        if centerline.shape[0] >= 2 and _polyline_segments(centerline)[0].shape[0] > 0:
            centerlines[index] = centerline
    if not centerlines:
        raise ValueError("no usable lanelet centerline in this scene")
    return centerlines


def _lane_half_widths(lane: torch.Tensor) -> tuple[float, float]:
    """Return the lanelet's ``(left, right)`` half width in meters.

    The lane tensor stores each boundary as an offset from the centerline
    point, so the half width is that offset's length.  The median over the
    lanelet's points keeps a single malformed boundary point from distorting
    the threshold.
    """
    points = lane[_valid_point_mask(lane)]
    return (
        float(points[:, 4:6].norm(dim=-1).median()),
        float(points[:, 6:8].norm(dim=-1).median()),
    )


def _nearest_lane_index(
    point: torch.Tensor, centerlines: dict[int, torch.Tensor]
) -> tuple[int, bool]:
    """Return ``(index, heading_aligned)`` for the lanelet the ego is travelling on at ``point``.

    Candidates are filtered by heading first (in the ego frame that is just the
    tangent's x-component), which rejects oncoming lanelets and the crossing
    road's turn lanelet that sweeps through the ego position at 45-70 deg --
    often closer to the ego than its own lane's centerline. The nearest of the
    survivors wins; a tie within ``_SOURCE_DISTANCE_TIE_M`` goes to the one
    running most along the ego heading, i.e. the straight branch of a fork.
    With no survivors the nearest lanelet is returned as ``heading_aligned=
    False`` for the caller to surface.

    This does NOT disambiguate two parallel lanes: past the midpoint of a lane
    change already in progress, the lane being entered is the nearest one.
    """
    point = point.reshape(1, 2)
    nearest = (float("inf"), -1)
    candidates: list[tuple[float, float, int]] = []  # (distance, ahead alignment, index)
    for index, centerline in centerlines.items():
        seg_p1, seg_p2 = _polyline_segments(centerline)
        # One distance computation for both the ranking and the tangent, so the
        # segment whose heading is tested is the one that made this lanelet the
        # nearest; resampled segments are uneven enough for a separately-picked
        # one to have a different tangent.
        distances = _point_to_segments_dist(point, seg_p1, seg_p2)[0]
        distance = float(distances.min())
        segment = int(distances.argmin())
        nearest = min(nearest, (distance, index))
        tangent = seg_p2[segment] - seg_p1[segment]
        if float(tangent[0] / tangent.norm()) < _MIN_SOURCE_HEADING_ALIGNMENT:
            continue
        ahead = centerline[-1] - seg_p1[segment]  # spans at least the nearest segment
        # Clamped: a zero `ahead` would give NaN, which wins every comparison.
        candidates.append(
            (distance, float(ahead[0] / ahead.norm().clamp_min(_SEGMENT_MIN_LENGTH)), index)
        )

    if not candidates:
        return nearest[1], False
    nearest_distance = min(distance for distance, _, _ in candidates)
    tied = [c for c in candidates if c[0] <= nearest_distance + _SOURCE_DISTANCE_TIE_M]
    return max(tied, key=lambda c: c[1])[2], True


def _arc_length_behind(path: torch.Tensor, point: torch.Tensor) -> float:
    """Return the path's arc length that lies behind ``point``'s projection."""
    seg_p1, seg_p2 = _polyline_segments(path)
    segment = seg_p2 - seg_p1
    lengths = segment.norm(dim=-1)
    offset = point.reshape(1, 2) - seg_p1
    t = ((offset * segment).sum(-1) / (lengths**2)).clamp(0, 1)
    closest = seg_p1 + t.unsqueeze(-1) * segment
    nearest = int((point.reshape(1, 2) - closest).norm(dim=-1).argmin())
    return float(lengths[:nearest].sum() + t[nearest] * lengths[nearest])


def _build_source_lane_path(
    centerlines: dict[int, torch.Tensor],
    source_index: int,
    reference_point: torch.Tensor,
    required_length_m: float,
    chain_tolerance_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend the source lanelet forward through its endpoint-chained successors.

    Two lanelets chain when the successor's first point coincides with the
    current tail -- that is how lanelet2 successor links survive into the NPZ,
    which carries no explicit connectivity. At a fork the branch whose OVERALL
    direction best matches the current heading wins, not the one whose initial
    tangent does: a turn lanelet leaves the fork tangent to the straight one,
    so their first segments are indistinguishable.

    The budget is measured AHEAD OF THE EGO. Lanelets are resampled to a fixed
    point count regardless of length, so the ego often sits deep inside a long
    source lanelet and counting the path behind it would stop chaining while
    the trajectories still run off the end.

    Returns the path and the lanelet each point came from.
    """
    path = centerlines[source_index]
    owners = torch.full((path.shape[0],), source_index, dtype=torch.long, device=path.device)
    # The prefix behind the ego never changes as successors are appended.
    behind_ego = _arc_length_behind(path, reference_point.to(path))

    used = {source_index}
    while float((path[1:] - path[:-1]).norm(dim=-1).sum()) - behind_ego < required_length_m:
        # The last NON-degenerate segment: a centerline may end in a duplicate
        # pair, and reading `path[-1] - path[-2]` would end the chain there.
        tail_p1, tail_p2 = _polyline_segments(path)
        tail_direction = tail_p2[-1] - tail_p1[-1]
        tail_direction = tail_direction / tail_direction.norm()

        best_index, best_alignment = -1, 0.0
        for index, centerline in centerlines.items():
            if index in used or float((centerline[0] - path[-1]).norm()) > chain_tolerance_m:
                continue
            overall = centerline[-1] - centerline[0]
            alignment = float(tail_direction @ overall) / max(
                float(overall.norm()), _SEGMENT_MIN_LENGTH
            )
            if alignment > best_alignment:
                best_alignment, best_index = alignment, index
        if best_index < 0:
            break
        used.add(best_index)
        appended = centerlines[best_index][1:]
        path = torch.cat([path, appended], dim=0)
        owners = torch.cat([owners, torch.full((appended.shape[0],), best_index, dtype=torch.long)])

    return path, owners


def reconstruct_source_lane(
    lanes: torch.Tensor,
    reference_point: torch.Tensor,
    required_length_m: float,
    chain_tolerance_m: float,
) -> SourceLane:
    """Rebuild the lane the ego is on at ``reference_point``, long enough to cover
    ``required_length_m`` ahead of it.

    ``lanes`` is one scene's ``(S, P, D>=8)`` lane tensor.
    """
    centerlines = _scene_centerlines(lanes)
    reference_point = reference_point.to(lanes)
    index, heading_aligned = _nearest_lane_index(reference_point, centerlines)
    path, owners = _build_source_lane_path(
        centerlines, index, reference_point, required_length_m, chain_tolerance_m
    )
    return SourceLane(path=path, owners=owners, index=index, heading_aligned=heading_aligned)


__all__ = ["SourceLane", "reconstruct_source_lane"]
