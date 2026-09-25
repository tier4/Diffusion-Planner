"""Reconstruct the lane the ego is travelling on, from the NPZ lane tensor.

The tensor gives geometry and nothing else: no lanelet ids, no connectivity,
each lanelet a short polyline resampled to a fixed point count regardless of
its real length.  A metric that needs a lateral reference valid for a whole
8 s horizon therefore has to rebuild the lane itself -- pick the lanelet the
ego is on, then walk forward through the successors that chain onto it.

``lane_change`` is the caller today; the reconstruction is kept separate
because it is a different problem from scoring, with its own failure modes
(an oncoming lanelet passing closer than the ego's own, a fork whose branches
are indistinguishable at their first segment, an ego sitting deep inside a
long lanelet).
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
        owners: ``(M,)`` lanelet index each path point came from, so a caller
            can read lane properties at the station it cares about rather than
            assuming the whole chain matches the first lanelet.
        index: the lanelet the ego starts on.
        heading_aligned: False when no lanelet near the ego ran with it and the
            pick fell back to the nearest one regardless -- possibly oncoming or
            crossing, which mirrors or skews every lateral reading taken from
            ``path``.  Callers are expected to surface this rather than score
            through it.
    """

    path: torch.Tensor
    owners: torch.Tensor
    index: int
    heading_aligned: bool

    def half_widths_at(self, lanes: torch.Tensor, point: torch.Tensor) -> tuple[float, float]:
        """Return the ``(left, right)`` half width where ``point`` meets the lane.

        Read at a station rather than once for the lane: a chain can run through
        a taper or a merge, so the first lanelet's width is not necessarily the
        width where a caller's threshold applies.
        """
        point = point.to(self.path).reshape(1, 2)
        seg_p1, seg_p2 = _polyline_segments(self.path)
        # Nearest SEGMENT, not nearest vertex: lanelets are resampled to a fixed
        # point count, so a long successor's vertices can be metres apart and the
        # nearest vertex still belongs to the short predecessor well past the
        # joint -- which would read the wrong lanelet's width for the whole gap.
        segment = int(_point_to_segments_dist(point, seg_p1, seg_p2)[0].argmin())
        # `_polyline_segments` may have dropped degenerate segments, so map back
        # through the kept mask rather than assuming index alignment.
        kept = torch.nonzero(
            (self.path[1:] - self.path[:-1]).norm(dim=-1) > _SEGMENT_MIN_LENGTH
        ).squeeze(-1)
        # A segment spans two points; the later one owns the station past the joint.
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

    Candidates are first filtered by heading: the centerline tangent nearest
    the ego must be within ``_MIN_SOURCE_HEADING_ALIGNMENT`` of the ego heading
    (the ego frame puts the ego at the origin heading +x, so that is simply the
    tangent's x-component).  This rejects oncoming and crossing lanelets, and
    also a crossing road's turn lanelet that sweeps through the ego position at
    45-70 deg -- common at an intersection, and closer to the ego than its own
    lane's centerline often enough to matter.

    Among the aligned candidates the nearest centerline wins.  Distances within
    ``_SOURCE_DISTANCE_TIE_M`` of the minimum are a tie -- at a fork the
    straight and turning branches start at the same point -- and the tie goes
    to the lanelet whose remaining run (from the ego to its end) points most
    along the ego heading, i.e. the straight branch.

    When no lanelet passes the heading filter the nearest one is returned with
    ``heading_aligned=False``; the scorer surfaces that flag per sample so a
    scene scored against an oncoming or crossing reference lane can be told
    apart from a genuine planner failure.

    This does NOT disambiguate two parallel lanes: once the ego is past the
    midpoint of a lane change already in progress, the lane it is moving INTO
    is the nearest one, and nothing in the geometry says otherwise.
    """
    point = point.reshape(1, 2)
    nearest = (float("inf"), -1)
    candidates: list[tuple[float, float, int]] = []  # (distance, ahead alignment, index)
    for index, centerline in centerlines.items():
        seg_p1, seg_p2 = _polyline_segments(centerline)
        # One distance computation serves both the ranking and the tangent, so
        # the segment whose heading is tested is always the one that made this
        # lanelet the nearest. Segments are very uneven after resampling, and a
        # separately-picked nearest segment (by midpoint, say) can be a
        # different one with a different tangent.
        distances = _point_to_segments_dist(point, seg_p1, seg_p2)[0]
        distance = float(distances.min())
        segment = int(distances.argmin())
        nearest = min(nearest, (distance, index))
        tangent = seg_p2[segment] - seg_p1[segment]
        if float(tangent[0] / tangent.norm()) < _MIN_SOURCE_HEADING_ALIGNMENT:
            continue
        ahead = centerline[-1] - seg_p1[segment]  # spans at least the nearest segment
        # Clamped so a centerline whose last point revisits the nearest segment's
        # start yields 0.0 rather than a NaN, which would win every comparison.
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
    current tail (that is how lanelet2 successor links survive into the NPZ,
    which carries no explicit connectivity).  Where several successors chain --
    a fork -- the one whose overall direction (last point minus first point)
    best matches the current heading wins, so the reconstructed path follows
    the ego's own lane rather than a turning branch.  The overall direction is
    used, not the initial tangent: a turn lanelet leaves the fork tangent to
    the straight one, so their first segments are indistinguishable and the
    pick would come down to floating-point noise.

    Chaining stops once the path reaches ``required_length_m`` AHEAD OF THE
    EGO: lanelets are resampled to a fixed point count regardless of length,
    so the ego often sits deep inside a long source lanelet, and measuring the
    budget against the whole path (including what is behind the ego) would
    stop chaining while the trajectories still run off the end.

    Returns the path and, for each of its points, the lanelet the point came
    from, so a caller can read lane properties (width, say) at the station it
    cares about instead of assuming the whole chain matches the first lanelet.
    """
    path = centerlines[source_index]
    owners = torch.full((path.shape[0],), source_index, dtype=torch.long, device=path.device)
    # The prefix behind the ego never changes as successors are appended, so it
    # is measured once.
    behind_ego = _arc_length_behind(path, reference_point.to(path))

    used = {source_index}
    while float((path[1:] - path[:-1]).norm(dim=-1).sum()) - behind_ego < required_length_m:
        # The last NON-DEGENERATE segment: a centerline is admitted on having one
        # usable segment, so its final two points may be a duplicate pair, and
        # reading those directly would end the chain at the first such lanelet.
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
