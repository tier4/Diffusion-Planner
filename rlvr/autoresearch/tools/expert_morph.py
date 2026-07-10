#!/usr/bin/env python3
"""Frenet det->GT morph repair candidate for expert_disagreement scenes.

For ``expert_disagreement`` scenes the disagreement is a PROGRESS / timing
mismatch between the model's deterministic plan and the logged human expert.
This module builds a single extra repair candidate by blending the model
deterministic plan (``det_traj``) toward the logged expert future
(``expert_traj``) in the route Frenet frame, under a smooth quintic onset
weight (zero deformation AND zero speed-jump at t=0) plus longitudinal
acceleration / jerk feasibility clamps.

Everything is ego-frame numpy in/out. The route Frenet frame is built by
REUSING ``scenario_generation.tools._heatmap_common`` (no hand-rolled
projection geometry).
"""

from __future__ import annotations

import math

import numpy as np

from scenario_generation.tools._heatmap_common import project_points_to_polyline

# Longitudinal-clamp bookkeeping: a step "fired" when the feasibility clamp had
# to alter the desired acceleration by more than this tolerance.
_CLAMP_TOL = 1e-6
# A candidate is rejected if the feasible longitudinal signal strays this far
# (metres, terminal progress) from the intended blended signal.
_TERMINAL_PROGRESS_TOL_M = 2.0
# ...or if the accel/jerk clamps had to fire on more than this fraction of steps.
_CLAMP_FIRE_FRACTION = 0.5
# Route reliability: reject if a det/expert endpoint overshoots the polyline end
# (or precedes its start) by more than this many metres along the terminal tangent.
_ROUTE_OVERSHOOT_MARGIN_M = 3.0
# Lateral rate cap (m/s) — bounds |d[i]-d[i-1]|/dt so the morph cannot teleport
# sideways.
_DEFAULT_LATERAL_RATE_CAP_MPS = 2.0
_EPS = 1e-9


def _cumulative_arc(pts: np.ndarray) -> np.ndarray:
    """Cumulative segment-length arc for a polyline.

    Same formula ``_heatmap_common.build_route_polyline`` /
    ``reproducer_danger_scorer`` use for the reference arc.
    """
    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg * seg).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float64)


def _quintic_smoothstep(u: np.ndarray, w_max: float) -> np.ndarray:
    """Quintic smoothstep w(u) = w_max*(6u^5 - 15u^4 + 10u^3).

    w(0)=0 and w'(0)=0 -> zero deformation AND zero speed-jump at t=0.
    """
    u = np.clip(u, 0.0, 1.0)
    return w_max * (6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3)


def _endpoint_overshoot(pt: np.ndarray, pts: np.ndarray) -> float:
    """Signed metres a point lies BEYOND the polyline (start or end).

    Returns the larger of the start-side overshoot (point precedes the first
    vertex along the initial tangent) and the end-side overshoot (point lies
    past the last vertex along the terminal tangent). Non-positive when the
    point projects onto the interior.
    """
    head_tan = pts[1] - pts[0]
    head_len = float(np.linalg.norm(head_tan))
    tail_tan = pts[-1] - pts[-2]
    tail_len = float(np.linalg.norm(tail_tan))
    over_start = 0.0
    over_end = 0.0
    if head_len > _EPS:
        over_start = -float(np.dot(pt - pts[0], head_tan / head_len))
    if tail_len > _EPS:
        over_end = float(np.dot(pt - pts[-1], tail_tan / tail_len))
    return max(over_start, over_end)


def _feasible_longitudinal(
    s_target: np.ndarray,
    v0: float,
    *,
    max_accel: float,
    max_jerk: float,
    dt: float,
) -> tuple[np.ndarray, int]:
    """Produce a monotone, accel/jerk-limited arc signal tracking ``s_target``.

    Integrates a speed profile forward with v[0]=v0 (so t=0 matches the ego),
    clamping per-step acceleration to [-max_accel, max_accel] and jerk to
    [-max_jerk, max_jerk], with speed kept >= 0 (monotone non-decreasing s).

    Returns (s_feasible, n_clamps_fired).
    """
    T = int(s_target.shape[0])
    s = np.empty(T, dtype=np.float64)
    s[0] = float(s_target[0])
    if T == 1:
        return s, 0
    # Desired per-step speed of the intended (blended) signal.
    v_des = np.diff(s_target) / dt  # (T-1,)
    v_prev = float(v0)
    a_prev = 0.0
    n_fired = 0
    for i in range(1, T):
        a_desired = (v_des[i - 1] - v_prev) / dt
        # Jerk clamp (relative to previous accel), then accel clamp.
        a_clamped = np.clip(a_desired, a_prev - max_jerk * dt, a_prev + max_jerk * dt)
        a_clamped = float(np.clip(a_clamped, -max_accel, max_accel))
        if abs(a_clamped - a_desired) > _CLAMP_TOL:
            n_fired += 1
        v_i = v_prev + a_clamped * dt
        if v_i < 0.0:
            v_i = 0.0
        s[i] = s[i - 1] + v_i * dt
        v_prev = v_i
        a_prev = a_clamped
    return s, n_fired


def _feasible_lateral(d_target: np.ndarray, *, rate_cap: float, dt: float) -> np.ndarray:
    """Bound the lateral signal's per-step rate to |d[i]-d[i-1]|/dt <= rate_cap."""
    T = int(d_target.shape[0])
    d = np.empty(T, dtype=np.float64)
    d[0] = float(d_target[0])
    max_step = rate_cap * dt
    for i in range(1, T):
        step = float(d_target[i] - d[i - 1])
        step = float(np.clip(step, -max_step, max_step))
        d[i] = d[i - 1] + step
    return d


def _arc_interp(
    s_ref: np.ndarray, pts: np.ndarray, s_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate polyline position + unit tangent at arc positions s_query.

    Returns (base_xy (M,2), unit_tangent (M,2)).
    """
    s_max = float(s_ref[-1])
    sq = np.clip(s_query, 0.0, s_max)
    idx = np.searchsorted(s_ref, sq, side="right") - 1
    idx = np.clip(idx, 0, len(s_ref) - 2)
    seg_len = s_ref[idx + 1] - s_ref[idx]
    frac = (sq - s_ref[idx]) / np.maximum(seg_len, _EPS)
    seg_vec = pts[idx + 1] - pts[idx]
    base = pts[idx] + frac[:, None] * seg_vec
    tan_len = np.linalg.norm(seg_vec, axis=1, keepdims=True)
    unit_tan = seg_vec / np.maximum(tan_len, _EPS)
    return base, unit_tan


def build_expert_morph_candidate(
    det_traj: np.ndarray,
    expert_traj: np.ndarray,
    route_xy: np.ndarray,
    *,
    w_max: float = 1.0,
    max_accel: float = 2.0,
    max_jerk: float = 4.0,
    dt: float = 0.1,
) -> np.ndarray | None:
    """Blend a deterministic plan toward the logged expert in route Frenet frame.

    Args:
        det_traj: (T,4) [x,y,cos,sin] ego-frame model deterministic plan.
        expert_traj: (T,4) [x,y,cos,sin] ego-frame logged expert future.
        route_xy: (P,2) ego-frame route centerline polyline.
        w_max: peak onset weight (1.0 -> reach the expert by the horizon end).
        max_accel: longitudinal accel cap (m/s^2).
        max_jerk: longitudinal jerk cap (m/s^3).
        dt: timestep (s).

    Returns:
        (T,4) [x,y,cos,sin] ego-frame morphed trajectory, or None when the
        route projection is unreliable or the feasibility clamps had to distort
        the intended blend too much (see module-level thresholds).
    """
    det = np.asarray(det_traj, dtype=np.float64)
    expert = np.asarray(expert_traj, dtype=np.float64)
    route = np.asarray(route_xy, dtype=np.float64)

    if det.ndim != 2 or det.shape[1] < 2:
        raise ValueError(f"det_traj must be (T,>=2), got {det.shape}")
    if expert.ndim != 2 or expert.shape[1] < 2:
        raise ValueError(f"expert_traj must be (T,>=2), got {expert.shape}")

    # Route reliability: need at least a segment.
    if route.ndim != 2 or route.shape[1] < 2 or route.shape[0] < 2:
        return None
    # Drop consecutive duplicate route points (zero-length segments break arc).
    keep = np.concatenate([[True], np.linalg.norm(np.diff(route, axis=0), axis=1) > _EPS])
    route = route[keep]
    if route.shape[0] < 2:
        return None

    T = int(min(det.shape[0], expert.shape[0]))
    if T < 2:
        return None
    det = det[:T]
    expert = expert[:T]

    det_xy = det[:, :2]
    expert_xy = expert[:, :2]

    # Reject if the trajectory extends beyond the route arc by a margin.
    for pt in (det_xy[0], det_xy[-1], expert_xy[0], expert_xy[-1]):
        if _endpoint_overshoot(pt, route) > _ROUTE_OVERSHOOT_MARGIN_M:
            return None

    s_ref = _cumulative_arc(route)
    if s_ref[-1] < _EPS:
        return None

    det_proj = project_points_to_polyline(det_xy, route, s_ref)
    gt_proj = project_points_to_polyline(expert_xy, route, s_ref)
    s_det, d_det = det_proj[:, 0], det_proj[:, 1]
    s_gt, d_gt = gt_proj[:, 0], gt_proj[:, 1]

    # Smooth quintic onset weight.
    u = np.arange(T, dtype=np.float64) / float(T - 1)
    w = _quintic_smoothstep(u, w_max)

    # Blend in Frenet.
    s_blend = s_det + w * (s_gt - s_det)
    d_blend = d_det + w * (d_gt - d_det)

    # Monotone non-decreasing longitudinal target (no reversing).
    s_blend = np.maximum.accumulate(s_blend)

    # Initial ego speed from the det plan (keep v[0] continuous with the ego).
    v0 = float(np.linalg.norm(det_xy[1] - det_xy[0]) / dt)

    s_feasible, n_fired = _feasible_longitudinal(
        s_blend, v0, max_accel=max_accel, max_jerk=max_jerk, dt=dt
    )

    # Feasibility failure: clamps distorted the intended progress too much.
    terminal_err = abs(float(s_feasible[-1] - s_blend[-1]))
    fired_fraction = n_fired / float(max(T - 1, 1))
    if terminal_err > _TERMINAL_PROGRESS_TOL_M or fired_fraction > _CLAMP_FIRE_FRACTION:
        return None

    d_feasible = _feasible_lateral(d_blend, rate_cap=_DEFAULT_LATERAL_RATE_CAP_MPS, dt=dt)

    # Reconstruct ego-frame xy from (s,d): route point at arc s, offset by d
    # along the route left-normal (matches project_to_polyline's sign
    # convention: +lateral == left of travel).
    base, unit_tan = _arc_interp(s_ref, route, s_feasible)
    left_normal = np.stack([-unit_tan[:, 1], unit_tan[:, 0]], axis=1)
    recon_xy = base + d_feasible[:, None] * left_normal

    # Heading from finite differences of the reconstructed path (naturally folds
    # in the route tangent plus the small correction from the lateral rate);
    # fall back to the route tangent when nearly stationary.
    tan_angle = np.arctan2(unit_tan[:, 1], unit_tan[:, 0])
    heading = np.empty(T, dtype=np.float64)
    dxy = np.diff(recon_xy, axis=0)
    step_norm = np.linalg.norm(dxy, axis=1)
    for t in range(T - 1):
        if step_norm[t] > 1e-3:
            heading[t] = math.atan2(dxy[t, 1], dxy[t, 0])
        else:
            heading[t] = tan_angle[t]
    heading[T - 1] = heading[T - 2]

    out = np.empty((T, 4), dtype=np.float32)
    out[:, 0] = recon_xy[:, 0]
    out[:, 1] = recon_xy[:, 1]
    out[:, 2] = np.cos(heading)
    out[:, 3] = np.sin(heading)
    return out
