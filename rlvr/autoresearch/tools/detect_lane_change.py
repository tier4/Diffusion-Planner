#!/usr/bin/env python3
"""Detect whether a training scene involves a lane change.

Replaces the centerline-saturation heuristic in
``scenario_generation/tools/classify_replay_steps.py`` (which conflates
"off-route drift" with "in-progress lane change") with a route-geometry
prediction validated against the recorded ego trajectory.

Algorithm
---------

The route is encoded in NPZ field ``route_lanes`` as a sequence of
lane polylines (head-to-tail chained: lane[i].last == lane[i+1].first
for in-lane progress; a LATERAL JUMP between consecutive lanes signals
a lane change in the route plan). Each lane has 20 polyline points
(x, y, ...) in the ego frame.

Predict (PRE — uses only ego_current_state + route_lanes):

    1. Find the route_lane closest to ego_current (perpendicular dist).
    2. Walk the route forward from that lane's last point.
    3. If any consecutive (lane[i].last → lane[i+1].first) transition
       within ``lookahead_m`` arc-length has a LATERAL gap >
       ``lane_change_thresh_m`` (perpendicular to lane[i] tangent),
       predict LANE_CHANGE.

Ground truth (uses ego_agent_future against the unrouted ``lanes``
channel — the full surrounding lane set, not just the planned path):

    1. At each future timestep, find the closest lane in ``lanes`` to
       ego_agent_future[t, :2].
    2. If the lane index changes during the future AND the new lane is
       laterally offset (not just longitudinal continuation) from the
       start lane, the ego ACTUALLY changed lanes.

Outputs
-------

JSON dict per scene with fields:
    - predicted: bool
    - actual:    bool
    - predict_meta: {n_route_lanes, max_lateral_gap_m, ...}
    - actual_meta:  {start_lane_idx, end_lane_idx, max_lat_displacement_m, ...}

Plus a summary confusion matrix.

Usage
-----

    python -m rlvr.autoresearch.tools.detect_lane_change \\
        --scenes path/to/scene_list.json \\
        --n_random 100 \\
        --output predictions.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Polyline helpers
# ---------------------------------------------------------------------------

def _valid_lane_indices(lanes: np.ndarray) -> np.ndarray:
    """Return indices of non-zero lane channels (each lane is (P, F))."""
    norms = np.linalg.norm(lanes, axis=(1, 2))
    return np.where(norms > 1e-3)[0]


def _valid_pts(lane: np.ndarray) -> np.ndarray:
    """Return rows of ``lane`` (P, F) whose first 4 features are non-zero."""
    n = np.linalg.norm(lane[:, :4], axis=-1)
    keep = n > 1e-3
    return lane[keep]


def _polyline_arc_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=-1).sum())


def _point_to_polyline_dist(pt: np.ndarray, poly: np.ndarray) -> tuple[float, int, float]:
    """Min perpendicular distance from ``pt`` (2,) to polyline ``poly`` (P, 2).

    Returns (min_dist, segment_idx, t) where the closest point lies at
    poly[i] + t*(poly[i+1] - poly[i]).
    """
    if poly.shape[0] < 2:
        d = np.linalg.norm(poly[0, :2] - pt[:2])
        return float(d), 0, 0.0
    a = poly[:-1, :2]
    b = poly[1:, :2]
    ab = b - a
    ap = pt[:2] - a
    seg_len_sq = (ab * ab).sum(axis=-1)
    seg_len_sq = np.where(seg_len_sq < 1e-9, 1e-9, seg_len_sq)
    t = (ap * ab).sum(axis=-1) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d = np.linalg.norm(closest - pt[:2], axis=-1)
    i = int(np.argmin(d))
    return float(d[i]), i, float(t[i])


def _closest_lane(pt: np.ndarray, lane_block: np.ndarray) -> tuple[int, float]:
    """Return (idx, perpendicular_distance) for the closest valid lane in
    ``lane_block`` (L, P, F) to ``pt`` (2,)."""
    valid = _valid_lane_indices(lane_block)
    best_idx = -1
    best_d = float("inf")
    for i in valid:
        pts = _valid_pts(lane_block[i])
        if pts.shape[0] == 0:
            continue
        d, _, _ = _point_to_polyline_dist(pt, pts[:, :2])
        if d < best_d:
            best_d = d
            best_idx = int(i)
    return best_idx, best_d


# ---------------------------------------------------------------------------
#  Lane-change PREDICTOR (route_lanes only)
# ---------------------------------------------------------------------------

def predict_lane_change(npz_path: str | Path,
                        lookahead_m: float = 50.0,
                        lane_change_thresh_m: float = 1.0,
                        ego_off_route_thresh_m: float = 2.5,
                        speed_lookahead_horizon_s: float = 8.0,
                        speed_lookahead_buffer_m: float = 5.0) -> dict:
    """Predict lane-change from ego_current_state + route_lanes geometry.

    A lane change is flagged iff EITHER:

      (a) ego is currently laterally offset from the FIRST route lane by
          at least ``lane_change_thresh_m`` (i.e. must merge to the
          route), OR

      (b) somewhere along the next ``effective_lookahead_m`` of the
          route a consecutive (lane_i.last_pt -> lane_{i+1}.first_pt)
          transition has a LATERAL displacement >= ``lane_change_thresh_m``
          (perpendicular to lane_i's tangent at its last point).

    ``effective_lookahead_m`` is the smaller of ``lookahead_m`` and
    (current speed * ``speed_lookahead_horizon_s``) +
    ``speed_lookahead_buffer_m``. This keeps the predictor honest:
    a route lane-change scheduled 80 m ahead is irrelevant when ego is
    only going to travel 30 m in the 8 s window.
    """
    d = np.load(npz_path, allow_pickle=True)
    ego_xy = np.asarray([0.0, 0.0], dtype=np.float32)  # ego frame
    rl = d["route_lanes"]
    valid = _valid_lane_indices(rl)
    speed = float(d["ego_current_state"][4])
    speed_budget = speed * speed_lookahead_horizon_s + speed_lookahead_buffer_m
    effective_lookahead_m = float(min(lookahead_m, speed_budget))

    if len(valid) == 0:
        return {
            "predicted": False,
            "predict_meta": {
                "n_route_lanes": 0,
                "reason": "no_route_lanes",
            },
        }

    # Find ego's current lane (closest of the route).
    cur_lane_idx, cur_dist = _closest_lane(ego_xy, rl)

    # Lateral offset from ego to the FIRST route lane.
    #
    # If the projection lands STRICTLY INSIDE a segment (0 < t < 1),
    # we have an unambiguous perpendicular distance.
    #
    # If it lands at an endpoint (t = 0 of segment 0 or t = 1 of the
    # last segment), ego is longitudinally outside the polyline — set
    # the lateral offset to 0 so we don't mis-flag scenes where the
    # route just hasn't started in front of ego yet (or ended behind
    # ego). The pairwise gap pass below picks up genuine route splits.
    first_pts = _valid_pts(rl[valid[0]])
    ego_lat_to_first_route = 0.0
    if first_pts.shape[0] >= 2:
        d_perp, seg_i, t = _point_to_polyline_dist(ego_xy, first_pts[:, :2])
        endpoint_hit = (
            (seg_i == 0 and t <= 1e-3)
            or (seg_i == first_pts.shape[0] - 2 and t >= 1.0 - 1e-3)
        )
        if not endpoint_hit:
            p0 = first_pts[seg_i, :2]
            p1 = (first_pts[seg_i + 1, :2]
                  if seg_i + 1 < first_pts.shape[0] else p0)
            tan = p1 - p0
            n = np.linalg.norm(tan)
            if n > 1e-6:
                tan = tan / n
                normal = np.array([-tan[1], tan[0]])
                base = p0 + t * (p1 - p0)
                ego_lat_to_first_route = float(
                    abs(np.dot(ego_xy - base, normal)))
            else:
                ego_lat_to_first_route = float(d_perp)

    if len(valid) == 1:
        # Single route lane — only signal is "ego off-route laterally".
        predicted = ego_lat_to_first_route >= ego_off_route_thresh_m
        return {
            "predicted": bool(predicted),
            "predict_meta": {
                "n_route_lanes": 1,
                "current_lane_idx": cur_lane_idx,
                "current_lane_dist_m": cur_dist,
                "ego_lat_to_first_route_m": ego_lat_to_first_route,
                "max_lateral_gap_m": 0.0,
                "lookahead_m": lookahead_m,
                "effective_lookahead_m": effective_lookahead_m,
                "speed_m_s": speed,
                "lane_change_thresh_m": lane_change_thresh_m,
                "reason": ("single_route_lane_offset" if predicted
                           else "single_route_lane"),
            },
        }

    # route_lanes are stored in route order (cf. tensor_converter); we
    # walk consecutive pairs starting at cur_lane_idx.
    valid_after = [i for i in valid if i >= cur_lane_idx]
    if len(valid_after) < 2:
        return {
            "predicted": False,
            "predict_meta": {
                "n_route_lanes": int(len(valid)),
                "current_lane_idx": cur_lane_idx,
                "current_lane_dist_m": cur_dist,
                "max_lateral_gap_m": 0.0,
                "reason": "no_successor_lanes",
            },
        }

    # arc_to_gap[k] = arc-length from ego's projection on lane valid_after[0]
    # forward to the END of lane valid_after[k] (where its successor
    # gap lives).
    max_lat_gap = 0.0
    triggering_pair = None
    arc_used = 0.0  # cumulative arc-length to END of lane k
    # Compute remaining length of the FIRST lane (from ego's projection
    # to its last point) so the budget is honest.
    first = _valid_pts(rl[valid_after[0]])
    if first.shape[0] >= 2:
        d0, seg0, t0 = _point_to_polyline_dist(ego_xy, first[:, :2])
        # arc from ego projection forward along lane to its last pt
        arc_remaining = (1.0 - t0) * float(np.linalg.norm(
            first[seg0 + 1, :2] - first[seg0, :2]
        )) if seg0 + 1 < first.shape[0] else 0.0
        for j in range(seg0 + 1, first.shape[0] - 1):
            arc_remaining += float(np.linalg.norm(
                first[j + 1, :2] - first[j, :2]))
    else:
        arc_remaining = 0.0

    for k in range(len(valid_after) - 1):
        a_idx = valid_after[k]
        b_idx = valid_after[k + 1]
        a_pts = _valid_pts(rl[a_idx])
        b_pts = _valid_pts(rl[b_idx])
        if a_pts.shape[0] < 2 or b_pts.shape[0] < 1:
            continue

        # Tangent of lane a at its last segment.
        tan = a_pts[-1, :2] - a_pts[-2, :2]
        n_tan = np.linalg.norm(tan)
        if n_tan < 1e-6:
            continue
        tan = tan / n_tan
        normal = np.array([-tan[1], tan[0]])  # left perpendicular

        gap_vec = b_pts[0, :2] - a_pts[-1, :2]
        # Decompose into longitudinal (along tan) and lateral (along normal).
        lat = float(np.abs(np.dot(gap_vec, normal)))
        lon = float(np.dot(gap_vec, tan))

        # Arc to THIS gap = arc remaining on first lane (if k == 0) +
        # full lengths of intermediate lanes.
        arc_to_gap = arc_used + (arc_remaining if k == 0
                                 else _polyline_arc_length(a_pts))

        if lat > max_lat_gap:
            max_lat_gap = lat
        if lat >= lane_change_thresh_m and triggering_pair is None:
            if arc_to_gap <= effective_lookahead_m:
                triggering_pair = (int(a_idx), int(b_idx),
                                   lat, lon, float(arc_to_gap))

        # Advance budget.
        if k == 0:
            arc_used = arc_remaining
        else:
            arc_used += _polyline_arc_length(a_pts)
        if arc_used > effective_lookahead_m:
            break

    # Combine triggers: explicit gap in upcoming route OR ego currently
    # off the route's first lane laterally (uses a stricter
    # ``ego_off_route_thresh_m`` because curving merges naturally close
    # a 1-2 m offset within the 8 s window without counting as a lane
    # change).
    ego_off_route = ego_lat_to_first_route >= ego_off_route_thresh_m
    predicted = (triggering_pair is not None) or ego_off_route

    return {
        "predicted": bool(predicted),
        "predict_meta": {
            "n_route_lanes": int(len(valid)),
            "current_lane_idx": cur_lane_idx,
            "current_lane_dist_m": cur_dist,
            "ego_lat_to_first_route_m": ego_lat_to_first_route,
            "max_lateral_gap_m": float(max_lat_gap),
            "lookahead_m": lookahead_m,
            "effective_lookahead_m": effective_lookahead_m,
            "speed_m_s": speed,
            "lane_change_thresh_m": lane_change_thresh_m,
            "ego_off_route_trigger": bool(ego_off_route),
            "trigger": (None if triggering_pair is None
                        else {"from_lane": triggering_pair[0],
                              "to_lane": triggering_pair[1],
                              "lateral_gap_m": triggering_pair[2],
                              "longitudinal_gap_m": triggering_pair[3],
                              "arc_to_trigger_m": triggering_pair[4]}),
        },
    }


# ---------------------------------------------------------------------------
#  Lane-change GROUND TRUTH (ego_agent_future vs. lanes)
# ---------------------------------------------------------------------------

def _lanes_share_endpoint(lane_a: np.ndarray, lane_b: np.ndarray,
                          chain_thresh_m: float = 1.0) -> bool:
    """Return True iff lanes a and b are longitudinally chained.

    Two segments chain when an endpoint of one is essentially equal to
    an endpoint of the other. That happens for splits / merges /
    successor-lane connections in the lanelet2 graph and means a switch
    of "closest lane" between them is NOT a lane change — just driving
    forward across a lane boundary.
    """
    a = _valid_pts(lane_a)
    b = _valid_pts(lane_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return False
    a_first = a[0, :2]
    a_last = a[-1, :2]
    b_first = b[0, :2]
    b_last = b[-1, :2]
    for p, q in ((a_last, b_first), (a_first, b_last),
                 (a_last, b_last), (a_first, b_first)):
        if float(np.linalg.norm(p - q)) <= chain_thresh_m:
            return True
    return False


def _lateral_offset_between_lanes(lane_a: np.ndarray, lane_b: np.ndarray,
                                  pivot_xy: np.ndarray) -> float:
    """Approximate the perpendicular offset between two lanes, measured
    at ``pivot_xy``. Returns the absolute lateral distance from the
    closest point of lane_a to the closest point of lane_b at the same
    longitudinal station as ``pivot_xy``.
    """
    a = _valid_pts(lane_a)
    b = _valid_pts(lane_b)
    if a.shape[0] < 2 or b.shape[0] < 1:
        return 0.0
    # Project pivot onto a; get the corresponding tangent + normal.
    d_a, seg_i, t = _point_to_polyline_dist(pivot_xy, a[:, :2])
    p0 = a[seg_i, :2]
    p1 = a[seg_i + 1, :2] if seg_i + 1 < a.shape[0] else a[seg_i, :2]
    # Closest point on a to pivot:
    base = p0 + t * (p1 - p0)
    # Closest point on b to pivot:
    d_b, _, _ = _point_to_polyline_dist(pivot_xy, b[:, :2])
    # Use that signed lateral via tangent.
    tangent = p1 - p0
    n = np.linalg.norm(tangent)
    if n < 1e-6:
        # Degenerate; just return distance gap.
        return abs(d_a - d_b)
    tangent = tangent / n
    normal = np.array([-tangent[1], tangent[0]])
    # Closest point on b — find vector from base to closest pt on b.
    d_b_pt_dist, b_seg, t_b = _point_to_polyline_dist(base, b[:, :2])
    bp0 = b[b_seg, :2]
    bp1 = b[b_seg + 1, :2] if b_seg + 1 < b.shape[0] else b[b_seg, :2]
    b_closest = bp0 + t_b * (bp1 - bp0)
    delta = b_closest - base
    return float(abs(np.dot(delta, normal)))


def actual_lane_change(npz_path: str | Path,
                       move_thresh_m: float = 5.0,
                       lane_lateral_offset_m: float = 1.5,
                       endpoint_chain_thresh_m: float = 1.0,
                       commit_dwell_steps: int = 15,
                       commit_dist_diff_m: float = 0.6,
                       ego_lateral_drift_m: float = 1.5) -> dict:
    """Reconstruct whether the ego actually changed lanes during the
    recorded future.

    Strategy:

    1. At every future timestep, find the closest lane (in ``lanes``).
    2. Compress to a unique sequence of closest-lane indices.
    3. For each consecutive pair (l_i -> l_{i+1}) in that sequence,
       classify the transition:
         * "chained" iff the two lane polylines share an endpoint
           (longitudinal continuation, NOT a lane change).
         * Otherwise compute the perpendicular offset between l_i and
           l_{i+1} at the transition point. Offset >=
           ``lane_lateral_offset_m`` => candidate lane change.
    4. Ghost-transition filters:
         * COMMITTED: the new lane must become at least
           ``commit_dist_diff_m`` closer than the old lane within
           ``commit_dwell_steps`` future steps.
         * EGO_DRIFT: the ego must itself drift at least
           ``ego_lateral_drift_m`` perpendicular to its initial
           heading. Diverging parallel lanes alone (where ego barely
           moves laterally but the lane geometry spreads apart) do not
           count.
    5. A scene is flagged actual=True iff at least one such transition
       qualifies AND the ego_drift filter passes.
    """
    d = np.load(npz_path, allow_pickle=True)
    fut = d["ego_agent_future"]  # (80, 3)
    lanes = d["lanes"]            # (140, 20, 33)

    total_move = float(np.linalg.norm(fut[-1, :2] - fut[0, :2]))
    if total_move < move_thresh_m:
        return {
            "actual": False,
            "actual_meta": {
                "reason": "stopped",
                "total_motion_m": total_move,
            },
        }

    # Ego self-drift in INITIAL ego frame: project every future xy onto
    # the initial heading and compute the maximum perpendicular
    # excursion. ego_agent_future is already in ego-current frame, so
    # the initial heading is +x and the lateral axis is +y.
    init_heading = float(fut[0, 2])
    cos_h = float(np.cos(-init_heading))
    sin_h = float(np.sin(-init_heading))
    # Rotate so initial heading is +x: lateral coordinate = (-sin) x + cos y.
    lat_in_init = (-sin_h) * fut[:, 0] + cos_h * fut[:, 1]
    max_ego_lateral_drift = float(np.max(np.abs(lat_in_init - lat_in_init[0])))

    # Closest-lane index per timestep, compressed to unique sequence.
    seq: list[int] = []
    seq_t: list[int] = []
    for t in range(fut.shape[0]):
        li, _ = _closest_lane(fut[t, :2].astype(np.float32), lanes)
        if li < 0:
            continue
        if not seq or seq[-1] != li:
            seq.append(li)
            seq_t.append(t)

    if not seq:
        return {
            "actual": False,
            "actual_meta": {
                "reason": "no_lanes",
                "total_motion_m": total_move,
            },
        }

    transitions: list[dict] = []
    actual = False
    for k in range(len(seq) - 1):
        a_idx = seq[k]
        b_idx = seq[k + 1]
        t_at_b = seq_t[k + 1]
        chained = _lanes_share_endpoint(lanes[a_idx], lanes[b_idx],
                                        chain_thresh_m=endpoint_chain_thresh_m)
        pivot = fut[t_at_b, :2].astype(np.float32)
        offset = _lateral_offset_between_lanes(lanes[a_idx], lanes[b_idx],
                                               pivot)

        # "Committed" check: ego must EVENTUALLY drift more strongly
        # toward b than toward a. We measure the maximum value of
        # (d_a - d_b) reached within ``commit_dwell_steps`` steps after
        # the transition; if it never crosses ``commit_dist_diff_m``,
        # this is a ghost transition (ego sits between two parallel
        # polylines). A genuine LC widens the gap monotonically.
        a_pts = _valid_pts(lanes[a_idx])
        b_pts = _valid_pts(lanes[b_idx])
        max_diff = -float("inf")
        end_tau = min(t_at_b + commit_dwell_steps, fut.shape[0])
        for tau in range(t_at_b, end_tau):
            xy = fut[tau, :2].astype(np.float32)
            d_a, _, _ = _point_to_polyline_dist(xy, a_pts[:, :2])
            d_b, _, _ = _point_to_polyline_dist(xy, b_pts[:, :2])
            diff = d_a - d_b
            if diff > max_diff:
                max_diff = diff
        committed = (max_diff >= commit_dist_diff_m)

        is_lc = ((not chained)
                 and (offset >= lane_lateral_offset_m)
                 and committed
                 and (max_ego_lateral_drift >= ego_lateral_drift_m))
        transitions.append({
            "t": t_at_b,
            "from_lane": a_idx,
            "to_lane": b_idx,
            "chained": bool(chained),
            "lateral_offset_m": offset,
            "committed": bool(committed),
            "qualifies_as_lane_change": bool(is_lc),
        })
        if is_lc:
            actual = True

    return {
        "actual": bool(actual),
        "actual_meta": {
            "start_lane_idx": int(seq[0]),
            "end_lane_idx": int(seq[-1]),
            "n_unique_closest_lanes": int(len(seq)),
            "transitions": transitions,
            "total_motion_m": float(total_move),
            "max_ego_lateral_drift_m": float(max_ego_lateral_drift),
            "lane_lateral_offset_m": lane_lateral_offset_m,
            "endpoint_chain_thresh_m": endpoint_chain_thresh_m,
            "ego_lateral_drift_m": ego_lateral_drift_m,
        },
    }


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

def evaluate_scene(npz_path: str | Path,
                   lookahead_m: float,
                   lane_change_thresh_m: float,
                   ego_off_route_thresh_m: float,
                   move_thresh_m: float,
                   lane_lateral_offset_m: float,
                   endpoint_chain_thresh_m: float) -> dict:
    out = {"scene": str(npz_path)}
    try:
        out.update(predict_lane_change(
            npz_path,
            lookahead_m=lookahead_m,
            lane_change_thresh_m=lane_change_thresh_m,
            ego_off_route_thresh_m=ego_off_route_thresh_m,
        ))
        out.update(actual_lane_change(
            npz_path,
            move_thresh_m=move_thresh_m,
            lane_lateral_offset_m=lane_lateral_offset_m,
            endpoint_chain_thresh_m=endpoint_chain_thresh_m,
        ))
    except Exception as e:
        out["error"] = repr(e)
    return out


def confusion(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    skipped_stopped = 0
    skipped_error = 0
    for r in rows:
        if "error" in r:
            skipped_error += 1
            continue
        if "actual" not in r:
            skipped_error += 1
            continue
        if r.get("actual_meta", {}).get("reason") == "stopped":
            skipped_stopped += 1
            continue
        p = r["predicted"]
        a = r["actual"]
        if p and a:
            tp += 1
        elif p and not a:
            fp += 1
        elif (not p) and a:
            fn += 1
        else:
            tn += 1
    n_eval = tp + fp + tn + fn
    acc = (tp + tn) / max(n_eval, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n_total_rows": len(rows),
        "n_skipped_stopped": skipped_stopped,
        "n_skipped_error": skipped_error,
        "n_evaluated": n_eval,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True,
                   help="JSON file containing list of NPZ paths.")
    p.add_argument("--output", required=True,
                   help="Output JSON file with per-scene predictions + summary.")
    p.add_argument("--n_random", type=int, default=100,
                   help="Sample N random scenes from --scenes (0 = use all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookahead_m", type=float, default=50.0,
                   help="Predict only lane changes scheduled within this arc-length.")
    p.add_argument("--lane_change_thresh_m", type=float, default=1.0,
                   help="Consecutive route lanes with lateral gap >= this "
                        "trigger a lane-change prediction.")
    p.add_argument("--ego_off_route_thresh_m", type=float, default=2.5,
                   help="If ego is laterally offset from the route's "
                        "first lane by at least this much (and the "
                        "projection is interior, not at an endpoint), "
                        "predict a lane change. Stricter than "
                        "lane_change_thresh_m because curving merges "
                        "naturally close a small offset.")
    p.add_argument("--move_thresh_m", type=float, default=5.0,
                   help="GT skips scenes where ego barely moved.")
    p.add_argument("--lane_lateral_offset_m", type=float, default=1.5,
                   help="GT lane change requires the new closest-lane to "
                        "be at least this far perpendicular from the "
                        "previous closest-lane at the transition point.")
    p.add_argument("--endpoint_chain_thresh_m", type=float, default=1.0,
                   help="If two consecutive closest-lanes share an "
                        "endpoint within this radius, the transition is "
                        "treated as a longitudinal continuation, not a "
                        "lane change.")
    args = p.parse_args()

    with open(args.scenes) as f:
        all_scenes = json.load(f)
    if not isinstance(all_scenes, list):
        raise SystemExit("--scenes must contain a JSON list of NPZ paths")

    if args.n_random and len(all_scenes) > args.n_random:
        rng = random.Random(args.seed)
        scenes = rng.sample(all_scenes, args.n_random)
    else:
        scenes = list(all_scenes)
    print(f"Evaluating {len(scenes)} scenes (seed={args.seed})")

    rows: list[dict] = []
    for i, s in enumerate(scenes):
        row = evaluate_scene(
            s,
            lookahead_m=args.lookahead_m,
            lane_change_thresh_m=args.lane_change_thresh_m,
            ego_off_route_thresh_m=args.ego_off_route_thresh_m,
            move_thresh_m=args.move_thresh_m,
            lane_lateral_offset_m=args.lane_lateral_offset_m,
            endpoint_chain_thresh_m=args.endpoint_chain_thresh_m,
        )
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(scenes)}")

    summary = confusion(rows)
    print("\nConfusion matrix (excluding stopped & errored scenes):")
    print(f"  TP={summary['tp']:3d}  FP={summary['fp']:3d}")
    print(f"  FN={summary['fn']:3d}  TN={summary['tn']:3d}")
    print(f"  evaluated={summary['n_evaluated']} "
          f"(skipped {summary['n_skipped_stopped']} stopped, "
          f"{summary['n_skipped_error']} errored)")
    print(f"  accuracy={summary['accuracy']:.3f}  "
          f"precision={summary['precision']:.3f}  "
          f"recall={summary['recall']:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "config": {
                "scenes": args.scenes,
                "n_random": args.n_random,
                "seed": args.seed,
                "lookahead_m": args.lookahead_m,
                "lane_change_thresh_m": args.lane_change_thresh_m,
                "ego_off_route_thresh_m": args.ego_off_route_thresh_m,
                "move_thresh_m": args.move_thresh_m,
                "lane_lateral_offset_m": args.lane_lateral_offset_m,
                "endpoint_chain_thresh_m": args.endpoint_chain_thresh_m,
            },
            "summary": summary,
            "rows": rows,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
