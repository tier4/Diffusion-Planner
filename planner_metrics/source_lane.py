"""Reconstruct the lane the ego is travelling on, from the NPZ lane tensor.

The tensor gives geometry and nothing else: no lanelet ids, no connectivity,
each lanelet a short polyline resampled to a fixed point count regardless of
its real length. A lateral reference valid for a whole horizon therefore has
to be rebuilt -- pick the lanelet the ego is on, then walk forward through the
successors that chain onto it.
"""

from __future__ import annotations

import math
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
# At a fork, the straight continuation is the branch whose lanes run on
# farthest without a lanelet bending off: one whose overall direction deviates
# more than this from where its predecessor ended (a 90 deg turn lanelet's does
# by ~45 deg; a lanelet following a road's curve, by half its own bend). A turn
# lane can run parallel to the through lane for tens of meters before it bends
# off, so the branches' own directions cannot tell them apart.
_STRAIGHT_MAX_DEVIATION_DEG = 15.0
# How far ahead that reach is followed. Lanes on beyond this make no difference.
_STRAIGHT_REACH_LIMIT_M = 150.0
# Branches whose reaches differ by less than this are a tie (both cut off by the
# edge of the lane tensor, typically), left to the branches' own directions.
_STRAIGHT_REACH_TIE_M = 10.0


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


def _successor_map(
    centerlines: dict[int, torch.Tensor], chain_tolerance_m: float
) -> dict[int, list[int]]:
    """Return each lanelet's endpoint-chained successors (first point on its last)."""
    indices = list(centerlines)
    starts = torch.stack([centerlines[i][0] for i in indices])
    ends = torch.stack([centerlines[i][-1] for i in indices])
    chained = torch.cdist(ends, starts) <= chain_tolerance_m
    return {
        index: [indices[j] for j in torch.nonzero(chained[row]).squeeze(-1).tolist() if j != row]
        for row, index in enumerate(indices)
    }


def _straight_reach(
    centerlines: dict[int, torch.Tensor],
    successors: dict[int, list[int]],
    first: int,
    heading: torch.Tensor,
    used: set[int],
) -> float:
    """Return how far the lanes from the start of ``first`` run on without bending off.

    ``heading`` is the direction the lane arrives at ``first`` in. Followed
    through any chained successors (the longest way wins), stopping at the first
    lanelet whose overall direction deviates more than
    ``_STRAIGHT_MAX_DEVIATION_DEG`` from where its predecessor ended, and at
    ``_STRAIGHT_REACH_LIMIT_M``. A turn lane's reach ends where it bends off; the
    through lane's runs on, through the road's own curves.
    """
    min_alignment = math.cos(math.radians(_STRAIGHT_MAX_DEVIATION_DEG))
    best = 0.0
    stack = [(first, heading, 0.0, frozenset(used | {first}))]
    while stack and best < _STRAIGHT_REACH_LIMIT_M:
        index, arriving, before, seen = stack.pop()
        centerline = centerlines[index]
        chord = centerline[-1] - centerline[0]
        if float(arriving @ chord) < min_alignment * float(chord.norm()) * float(arriving.norm()):
            best = max(best, before)
            continue
        total = before + float((centerline[1:] - centerline[:-1]).norm(dim=-1).sum())
        best = max(best, total)
        if total >= _STRAIGHT_REACH_LIMIT_M:
            continue
        tail_p1, tail_p2 = _polyline_segments(centerline)
        leaving = tail_p2[-1] - tail_p1[-1]
        stack.extend(
            (successor, leaving, total, seen | {successor})
            for successor in successors[index]
            if successor not in seen
        )
    return min(best, _STRAIGHT_REACH_LIMIT_M)


def _longest_reaching(
    candidates: list[int],
    centerlines: dict[int, torch.Tensor],
    successors: dict[int, list[int]],
    heading: torch.Tensor,
    used: set[int],
) -> set[int]:
    """Return the branches of a fork that run on (nearly) farthest along ``heading``."""
    reach = {
        index: _straight_reach(centerlines, successors, index, heading, used)
        for index in candidates
    }
    longest = max(reach.values())
    return {index for index, value in reach.items() if value >= longest - _STRAIGHT_REACH_TIE_M}


def _nearest_lane_index(
    point: torch.Tensor,
    centerlines: dict[int, torch.Tensor],
    successors: dict[int, list[int]] | None = None,
) -> tuple[int, bool]:
    """Return ``(index, heading_aligned)`` for the lanelet the ego is travelling on at ``point``.

    Candidates are filtered by heading first (in the ego frame that is just the
    tangent's x-component), which rejects oncoming lanelets and the crossing
    road's turn lanelet that sweeps through the ego position at 45-70 deg --
    often closer to the ego than its own lane's centerline. The nearest of the
    survivors wins; a tie within ``_SOURCE_DISTANCE_TIE_M`` goes to the
    straight branch of a fork: the one whose lanes run on farthest along the ego
    heading (given ``successors``), then the one running most along it.
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
    if successors is not None and len(tied) > 1:
        reaching = _longest_reaching(
            [c[2] for c in tied], centerlines, successors, torch.tensor([1.0, 0.0]).to(point), set()
        )
        tied = [c for c in tied if c[2] in reaching]
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
    successors: dict[int, list[int]],
    source_index: int,
    reference_point: torch.Tensor,
    required_length_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend the source lanelet forward through its endpoint-chained successors.

    Two lanelets chain when the successor's first point coincides with the
    current tail -- that is how lanelet2 successor links survive into the NPZ,
    which carries no explicit connectivity. At a fork the straight
    continuation wins: the branch whose lanes run on farthest along the current
    heading (``_straight_reach``), then among those the one whose OVERALL
    direction best matches it. Neither the initial tangent nor the first
    lanelet's direction alone tells a turn lane from the through lane: a turn
    lanelet leaves the fork tangent to the straight one, and a turn lane can
    run alongside it for tens of meters before it bends off, its first lanelet
    even pointing closer to the heading. Chaining such a branch would make the
    branch the source lane, and a GT taking it would never leave it.

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

        alignments = {}
        for index in successors[int(owners[-1])]:
            if index in used:
                continue
            centerline = centerlines[index]
            overall = centerline[-1] - centerline[0]
            alignment = float(tail_direction @ overall) / max(
                float(overall.norm()), _SEGMENT_MIN_LENGTH
            )
            if alignment > 0.0:
                alignments[index] = alignment
        if len(alignments) > 1:
            reaching = _longest_reaching(
                list(alignments), centerlines, successors, tail_direction, used
            )
            alignments = {index: a for index, a in alignments.items() if index in reaching}
        best_index = max(alignments, key=alignments.get) if alignments else -1
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
    successors = _successor_map(centerlines, chain_tolerance_m)
    reference_point = reference_point.to(lanes)
    index, heading_aligned = _nearest_lane_index(reference_point, centerlines, successors)
    path, owners = _build_source_lane_path(
        centerlines, successors, index, reference_point, required_length_m
    )
    return SourceLane(path=path, owners=owners, index=index, heading_aligned=heading_aligned)


__all__ = ["SourceLane", "reconstruct_source_lane"]
