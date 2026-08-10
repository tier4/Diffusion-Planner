"""Route stitching, Frenet (s, d) projection, and route QA."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

GAP_DEDUP_THRESHOLD = 0.5
GAP_INTERPOLATION_THRESHOLD = 3.0
PROJECTION_FAIL_THRESHOLD = 5.0
ROUTE_COVERAGE_THRESHOLD = 0.9
MONOTONICITY_TOLERANCE = 0.1

HORIZONS = {"2s": 20, "4s": 40, "8s": 80}


@dataclass
class RouteQA:
    route_valid: bool
    max_segment_gap: float
    total_interpolated_gap: float
    n_valid_segments: int
    route_arc_length: float
    route_coverage_insufficient: bool
    human_max_proj_dist: float
    n_monotonic_violations: int
    frac_planner_proj_fail: float

    def to_dict(self) -> dict[str, float]:
        return {
            "route_valid": int(self.route_valid),
            "max_segment_gap": self.max_segment_gap,
            "total_interpolated_gap": self.total_interpolated_gap,
            "n_valid_segments": self.n_valid_segments,
            "route_arc_length": self.route_arc_length,
            "route_coverage_insufficient": int(self.route_coverage_insufficient),
            "human_max_proj_dist": self.human_max_proj_dist,
            "n_monotonic_violations": self.n_monotonic_violations,
            "frac_planner_proj_fail": self.frac_planner_proj_fail,
        }


@dataclass
class StitchedRoute:
    centerline: np.ndarray  # (L, 2)
    arc_length: np.ndarray  # (L,)
    qa: RouteQA


def stitch_route_lanes(route_lanes: np.ndarray) -> StitchedRoute:
    """Stitch ordered route_lanes segments into one continuous centerline.

    Args:
        route_lanes: (25, 20, 33) or (1, 25, 20, 33). Features[0:2] are (x, y).
    """
    if route_lanes.ndim == 4:
        route_lanes = route_lanes.squeeze(0)

    segments: list[np.ndarray] = []
    for seg in route_lanes:
        xy = seg[:, :2]
        if np.allclose(xy, 0.0):
            continue
        nonzero_mask = ~np.all(xy == 0.0, axis=-1)
        if not nonzero_mask.any():
            continue
        last_valid = np.where(nonzero_mask)[0][-1]
        segments.append(xy[: last_valid + 1].copy())

    n_valid = len(segments)
    if n_valid == 0:
        empty_qa = RouteQA(
            route_valid=False,
            max_segment_gap=float("inf"),
            total_interpolated_gap=0.0,
            n_valid_segments=0,
            route_arc_length=0.0,
            route_coverage_insufficient=True,
            human_max_proj_dist=float("inf"),
            n_monotonic_violations=0,
            frac_planner_proj_fail=1.0,
        )
        return StitchedRoute(
            centerline=np.empty((0, 2)),
            arc_length=np.empty((0,)),
            qa=empty_qa,
        )

    max_gap = 0.0
    total_interp = 0.0
    route_valid = True
    points = [segments[0]]

    for i in range(1, n_valid):
        prev_end = segments[i - 1][-1]
        curr_start = segments[i][0]
        gap = float(np.linalg.norm(curr_start - prev_end))
        max_gap = max(max_gap, gap)

        if gap <= GAP_DEDUP_THRESHOLD:
            points.append(segments[i][1:])
        elif gap <= GAP_INTERPOLATION_THRESHOLD:
            midpoint = (prev_end + curr_start) / 2.0
            points.append(midpoint[np.newaxis])
            points.append(segments[i][1:])
            total_interp += gap
        else:
            route_valid = False
            points.append(segments[i])

    centerline = np.concatenate(points, axis=0)
    diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=-1)
    arc_length = np.zeros(len(centerline))
    arc_length[1:] = np.cumsum(diffs)

    qa = RouteQA(
        route_valid=route_valid,
        max_segment_gap=max_gap,
        total_interpolated_gap=total_interp,
        n_valid_segments=n_valid,
        route_arc_length=float(arc_length[-1]) if len(arc_length) > 0 else 0.0,
        route_coverage_insufficient=False,  # set after human projection
        human_max_proj_dist=0.0,
        n_monotonic_violations=0,
        frac_planner_proj_fail=0.0,
    )
    return StitchedRoute(centerline=centerline, arc_length=arc_length, qa=qa)


def _project_points_to_polyline(
    polyline: np.ndarray,
    arc_length: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (M, 2) points onto a polyline. Returns s, d, proj_dist each (M,)."""
    n_seg = len(polyline) - 1
    if n_seg < 1:
        M = len(points)
        return np.full(M, np.nan), np.full(M, np.nan), np.full(M, np.inf)

    seg_starts = polyline[:-1]  # (n_seg, 2)
    seg_ends = polyline[1:]  # (n_seg, 2)
    seg_vecs = seg_ends - seg_starts  # (n_seg, 2)
    seg_lens = np.linalg.norm(seg_vecs, axis=-1)  # (n_seg,)
    seg_lens_safe = np.maximum(seg_lens, 1e-10)

    M = len(points)
    s_out = np.empty(M)
    d_out = np.empty(M)
    dist_out = np.empty(M)

    for i in range(M):
        p = points[i]
        dp = p - seg_starts  # (n_seg, 2)
        t = np.sum(dp * seg_vecs, axis=-1) / (seg_lens_safe**2)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_starts + t[:, np.newaxis] * seg_vecs  # (n_seg, 2)
        dists = np.linalg.norm(p - proj, axis=-1)  # (n_seg,)
        best = int(np.argmin(dists))

        s_out[i] = arc_length[best] + t[best] * seg_lens[best]
        dist_out[i] = dists[best]

        # Signed lateral offset: cross product gives sign
        tangent = seg_vecs[best]
        to_point = p - proj[best]
        cross = tangent[0] * to_point[1] - tangent[1] * to_point[0]
        d_out[i] = float(np.sign(cross)) * dists[best]

    return s_out, d_out, dist_out


def project_to_route(
    route: StitchedRoute,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto the stitched route centerline.

    Args:
        route: StitchedRoute from stitch_route_lanes.
        points: (T, 2) single trajectory or (N, T, 2) batch of trajectories.

    Returns:
        (s, d, proj_dist) with shape matching input (T,) or (N, T).
    """
    if points.ndim == 2:
        return _project_points_to_polyline(route.centerline, route.arc_length, points)

    N, T, _ = points.shape
    s_all = np.empty((N, T))
    d_all = np.empty((N, T))
    dist_all = np.empty((N, T))
    for n in range(N):
        s_all[n], d_all[n], dist_all[n] = _project_points_to_polyline(
            route.centerline, route.arc_length, points[n]
        )
    return s_all, d_all, dist_all


def update_qa_after_projection(
    route: StitchedRoute,
    human_s: np.ndarray,
    human_proj_dist: np.ndarray,
    samples_proj_dist: np.ndarray,
) -> None:
    """Update route QA fields after projecting human and samples."""
    qa = route.qa
    qa.human_max_proj_dist = (
        float(np.max(human_proj_dist)) if len(human_proj_dist) > 0 else float("inf")
    )

    mono_violations = np.sum(np.diff(human_s) < -MONOTONICITY_TOLERANCE)
    qa.n_monotonic_violations = int(mono_violations)

    if samples_proj_dist.size > 0:
        qa.frac_planner_proj_fail = float((samples_proj_dist > PROJECTION_FAIL_THRESHOLD).mean())
    else:
        qa.frac_planner_proj_fail = 1.0

    if route.qa.route_arc_length > 0:
        qa.route_coverage_insufficient = (
            float(np.max(human_s)) > route.qa.route_arc_length * ROUTE_COVERAGE_THRESHOLD
        )
    else:
        qa.route_coverage_insufficient = True


def frenet_energy_scores(
    human_sd: np.ndarray,
    samples_sd: np.ndarray,
    horizons: dict[str, int] = HORIZONS,
) -> dict[str, float]:
    """Compute longitudinal and lateral Energy Scores separately.

    Args:
        human_sd: (T, 2) [s, d] of the human trajectory.
        samples_sd: (N, T, 2) [s, d] of planner samples.
        horizons: mapping of name -> number of timesteps.
    """
    N = len(samples_sd)
    out: dict[str, float] = {}

    for name, h in horizons.items():
        for comp_idx, comp_name in [(0, "lon"), (1, "lat")]:
            y = human_sd[:h, comp_idx].reshape(1, -1)  # (1, h)
            x = samples_sd[:, :h, comp_idx].reshape(N, -1)  # (N, h)

            obs = float(cdist(x, y).mean())

            pw = cdist(x, x)
            np.fill_diagonal(pw, 0.0)
            div = float(pw.sum() / (N * (N - 1)))

            out[f"es_{comp_name}_{name}"] = obs - 0.5 * div

    return out
