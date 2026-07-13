#!/usr/bin/env python3
"""det->GT morph repair candidate for expert_disagreement scenes.

For ``expert_disagreement`` scenes the disagreement is a PROGRESS / timing
mismatch between the model's deterministic plan and the logged human expert.
The morph is deliberately simple:

1. **Re-time the det plan to the expert's timing** — blend the two cumulative
   DISTANCE SCHEDULES (not instantaneous speeds) under a quintic onset weight
   (zero speed-jump at t=0), track the result with an accel/jerk-limited
   profile, and interpolate the det plan's own positions AND headings at the
   resulting arcs. The det plan is already a feasible, kinematically valid
   trajectory; driving the same path slower (or stopping on it) stays valid
   by construction — a stop is a fixed point on the path with its fixed
   heading.
2. **Stopped expert => stop-arc cap** — when the expert ends stopped (or its
   log-truncated held tail reads as stopped), the arc schedule is capped at a
   stop point: by default the expert trajectory's final point, or the
   caller-supplied ``stop_anchor_xy`` (e.g. the recorded expert's actual stop
   position). The cap anticipates with a braking parabola and is floored at
   the tracker's shortest feasible stop from the det plan's initial speed.
3. **Lateral merge toward the expert** — the expert's lateral offset at the
   morph's end arc, measured in the det path's own Frenet frame (projection
   reuses ``_heatmap_common``; no hand-rolled geometry), applied as ONE
   analytic quintic ease in arc space whose magnitude is pre-scaled so the
   crab-angle slope bound holds without pointwise clipping; the offset only
   changes while the vehicle moves — no lateral drift at standstill.

Everything is ego-frame numpy in/out.
"""

from __future__ import annotations

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
# Lateral slope cap (m of offset per m of arc) — |Δd| <= slope_cap * Δs, so the
# lateral offset freezes when the vehicle stops (~30° max path-crab angle).
_DEFAULT_LATERAL_SLOPE_CAP = 0.58
# Reject when the expert strays this far laterally from the det path — the two
# trajectories no longer describe the same maneuver corridor and a lateral
# merge is meaningless.
_PATHS_DIVERGED_M = 5.0
# Expert counts as stopped at the horizon end below this speed (m/s); then the
# morph's arc is capped at the expert's stop location plus this overshoot.
_STOPPED_EXPERT_SPEED_MPS = 0.5
_STOP_ARC_OVERSHOOT_M = 0.5
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


def _det_headings(det: np.ndarray) -> np.ndarray:
    """Unwrapped per-step heading of the det plan.

    Prefers the plan's own (cos, sin) channels — they are valid even where the
    plan is stationary. Falls back to the path tangent for 2-column input.
    """
    if det.shape[1] >= 4:
        return np.unwrap(np.arctan2(det[:, 3], det[:, 2]))
    d = np.zeros_like(det[:, :2])
    d[1:-1] = det[2:, :2] - det[:-2, :2]
    d[0] = det[1, :2] - det[0, :2]
    d[-1] = det[-1, :2] - det[-2, :2]
    ang = np.arctan2(d[:, 1], d[:, 0])
    # Hold heading through stationary stretches (no tangent to read off).
    moving = np.linalg.norm(d, axis=1) > _EPS
    if moving.any():
        first = int(np.argmax(moving))
        ang[:first] = ang[first]
        for i in range(first + 1, len(ang)):
            if not moving[i]:
                ang[i] = ang[i - 1]
    return np.unwrap(ang)


def build_expert_morph_candidate(
    det_traj: np.ndarray,
    expert_traj: np.ndarray,
    *,
    w_max: float = 1.0,
    max_accel: float = 2.0,
    max_jerk: float = 4.0,
    dt: float = 0.1,
    stop_anchor_xy: np.ndarray | None = None,
    return_diag: bool = False,
) -> np.ndarray | None | tuple[np.ndarray | None, dict]:
    """Re-time the det plan to the expert's speed profile + lateral merge.

    Args:
        det_traj: (T,4) [x,y,cos,sin] ego-frame model deterministic plan.
        expert_traj: (T,4) [x,y,cos,sin] ego-frame logged expert future.
        w_max: peak onset weight (1.0 -> fully adopt the expert by horizon end).
        max_accel: longitudinal accel cap (m/s^2).
        max_jerk: longitudinal jerk cap (m/s^3).
        dt: timestep (s).
        stop_anchor_xy: optional (2,) ego-frame point overriding WHERE a
            stopped-expert morph stops: its projection onto the det path
            becomes the stop arc (e.g. the recorded expert's actual stop
            position). Default: the pseudo-target's final point, i.e. the
            expert's remaining travel distance applied from the ego's pose.
        return_diag: when True, return ``(morph, diag)`` where ``diag`` records
            why synthesis failed (``stage``) plus the clamp statistics — the
            repair pipeline persists it so morph losses are diagnosable.

    Returns:
        (T,4) [x,y,cos,sin] ego-frame morphed trajectory, or None when the
        blend is infeasible (see module-level thresholds). With ``return_diag``
        a ``(morph, diag)`` tuple instead.
    """

    def _ret(morph: np.ndarray | None, stage: str, **extra):
        if not return_diag:
            return morph
        return morph, {"stage": stage, **extra}

    det = np.asarray(det_traj, dtype=np.float64)
    expert = np.asarray(expert_traj, dtype=np.float64)
    if det.ndim != 2 or det.shape[1] < 2:
        raise ValueError(f"det_traj must be (T,>=2), got {det.shape}")
    if expert.ndim != 2 or expert.shape[1] < 2:
        raise ValueError(f"expert_traj must be (T,>=2), got {expert.shape}")

    T = int(min(det.shape[0], expert.shape[0]))
    if T < 2:
        return _ret(None, "too_short_horizon")
    det = det[:T]
    expert = expert[:T].copy()

    # The logged expert future can be zero-padded past its valid length. Those
    # trailing (0,0) pads would read as teleports, so hold the last valid pose.
    # NOTE: an all-zero expert is a genuinely STOPPED expert, not padding.
    expert_valid = np.abs(expert[:, :2]).sum(axis=1) > _EPS
    if expert_valid.any() and not expert_valid.all():
        last_valid = int(np.max(np.flatnonzero(expert_valid)))
        if last_valid < T - 1:
            expert[last_valid + 1 :] = expert[last_valid]

    det_xy = det[:, :2]
    expert_xy = expert[:, :2]

    # --- 1. Re-time: blend the distance-traveled schedules. ---
    # s_det_sched / s_gt_sched are each trajectory's own cumulative distance
    # over time. Blending the SCHEDULES (not the instantaneous speeds) makes
    # the morph's terminal arc converge to the expert's traveled distance —
    # "drive the det path with the expert's timing".
    v_det = np.linalg.norm(np.diff(det_xy, axis=0), axis=1) / dt  # (T-1,)
    s_det_sched = np.concatenate([[0.0], np.cumsum(v_det * dt)])
    v_gt = np.linalg.norm(np.diff(expert_xy, axis=0), axis=1) / dt  # (T-1,)
    s_gt_sched = np.concatenate([[0.0], np.cumsum(v_gt * dt)])
    u = np.arange(T, dtype=np.float64) / float(T - 1)
    w = _quintic_smoothstep(u, w_max)
    s_target = s_det_sched + w * (s_gt_sched - s_det_sched)
    # Monotone non-decreasing (a time-varying blend of monotone schedules can
    # locally reverse; the vehicle cannot).
    s_target = np.maximum.accumulate(s_target)

    s_det = _cumulative_arc(det_xy)
    th_det = _det_headings(det)

    # --- 1b. Stopped expert => stop AT the expert's stop location (projected
    # onto the det path), not merely after the expert's traveled DISTANCE: the
    # (diverged, ahead) ego covering the same distance can land mid-curve or on
    # a road border where the det plan was only passing through — and a target
    # that PARKS there is unsafe. The cap anticipates with the braking parabola
    # v <= sqrt(2 a_eff (s_cap - s)) so the demand stays trackable, and is
    # floored at the tracker's own shortest feasible stop (the ego may already
    # be past the expert's spot).
    expert_speed_end = float(np.linalg.norm(expert_xy[-1] - expert_xy[-2]) / dt)
    if expert_speed_end < _STOPPED_EXPERT_SPEED_MPS:
        keep0 = np.concatenate([[True], np.linalg.norm(np.diff(det_xy, axis=0), axis=1) > _EPS])
        stop_pts = det_xy[keep0]
        if stop_pts.shape[0] >= 2:
            # Where should the morph stop? Default: the (re-anchored)
            # pseudo-target's final point = the expert's REMAINING distance
            # applied from the ego's own pose. With stop_anchor_xy: the
            # caller-supplied point — e.g. the recorded expert's actual stop
            # POSITION — for "stop where the driver stopped" semantics (a
            # diverged-ahead ego then stops earlier, bounded by the floor).
            stop_point = expert_xy[-1:] if stop_anchor_xy is None else stop_anchor_xy[None, :2]
            stop_proj = project_points_to_polyline(
                np.asarray(stop_point, dtype=np.float64), stop_pts, _cumulative_arc(stop_pts)
            )
            a_stop = float(stop_proj[0, 0]) + _STOP_ARC_OVERSHOOT_M
            s_floor, _ = _feasible_longitudinal(
                np.full(T, float(s_target[0])),
                float(v_det[0]),
                max_accel=max_accel,
                max_jerk=max_jerk,
                dt=dt,
            )
            s_cap = max(a_stop, float(s_floor[-1]) + 0.5)
            a_eff = 0.7 * max_accel
            s_capped = np.empty_like(s_target)
            s_capped[0] = min(float(s_target[0]), s_cap)
            for i in range(1, T):
                ds_i = float(s_target[i] - s_target[i - 1])
                v_allow = np.sqrt(max(2.0 * a_eff * (s_cap - s_capped[i - 1]), 0.0))
                s_capped[i] = min(s_capped[i - 1] + min(ds_i, v_allow * dt), s_cap)
            s_target = s_capped

    # --- 1c. Feasibility: accel/jerk-limited tracking of the demand. ---
    s_feasible, n_fired = _feasible_longitudinal(
        s_target, float(v_det[0]), max_accel=max_accel, max_jerk=max_jerk, dt=dt
    )
    terminal_err = abs(float(s_feasible[-1] - s_target[-1]))
    fired_fraction = n_fired / float(max(T - 1, 1))
    if terminal_err > _TERMINAL_PROGRESS_TOL_M or fired_fraction > _CLAMP_FIRE_FRACTION:
        return _ret(
            None,
            "infeasible_deceleration",
            terminal_err_m=terminal_err,
            clamp_fire_fraction=fired_fraction,
            v0_mps=float(v_det[0]),
        )
    # Numerical standstill hygiene: the tracker's jerk-limited decay leaves a
    # long TAIL of crumb-sized steps after a stop. Positions/headings are
    # interpolated from the arc, so crumb-sized arc motion becomes crumb-sized
    # pose noise. Freeze the suffix with < 1 cm of REMAINING motion so a stop
    # is bit-exact (identical arcs -> identical interpolated poses -> exact
    # zero yaw and speed, like logged data). Defining the tail by remaining
    # motion — not by per-step size — keeps slow-creep profiles intact (a
    # 0.09 m/s queue creep is all sub-centimetre steps but must not park).
    remaining = s_feasible[-1] - s_feasible
    tail = remaining < 0.01
    if tail.any():
        tail_start = int(np.argmax(tail))
        s_feasible[tail_start:] = s_feasible[tail_start]

    # --- 2. Sample the det plan's own positions + headings at the new arcs. ---
    # np.interp needs an increasing grid; s_det has TIES where the det plan is
    # stationary. Tied arcs share a position, but the det heading channel may
    # rotate in place, so which tie np.interp lands on would be arbitrary —
    # dedup to the FIRST sample of each tied arc group for a deterministic grid.
    grid = np.concatenate([[True], np.diff(s_det) > _EPS])
    s_grid = s_det[grid]
    x_grid = det_xy[grid, 0]
    y_grid = det_xy[grid, 1]
    th_grid = th_det[grid]
    base_x = np.interp(s_feasible, s_grid, x_grid)
    base_y = np.interp(s_feasible, s_grid, y_grid)
    th = np.interp(s_feasible, s_grid, th_grid)
    # Beyond the det path's end (expert faster than det), continue along the
    # det plan's terminal heading.
    over = s_feasible > s_grid[-1]
    if over.any():
        extra = s_feasible[over] - float(s_grid[-1])
        base_x[over] = x_grid[-1] + extra * np.cos(th_grid[-1])
        base_y[over] = y_grid[-1] + extra * np.sin(th_grid[-1])
        th[over] = th_grid[-1]
    base = np.stack([base_x, base_y], axis=1)

    # --- 3. Lateral merge toward the expert, in the det path's Frenet frame. ---
    # ONE smooth ease from the det path toward the expert's lateral offset at
    # the morph's end arc: d(s) = d_end * quintic(arc fraction). The magnitude
    # is limited analytically so the lateral slope never exceeds the crab-angle
    # bound — no pointwise clipping, hence no slope discontinuities for the
    # kinematic gate's SG-smoothed yaw to trip on. A quintic's peak slope is
    # 1.875/span, so scale = slope_cap*span / (1.875*|d_end|), capped at 1.
    d_end = 0.0
    keep = np.concatenate([[True], np.linalg.norm(np.diff(det_xy, axis=0), axis=1) > _EPS])
    ref_pts = det_xy[keep]
    if ref_pts.shape[0] >= 2:
        ref_arc = _cumulative_arc(ref_pts)
        proj = project_points_to_polyline(expert_xy, ref_pts, ref_arc)
        a_gt, d_gt = proj[:, 0], proj[:, 1]
        # Only projections landing in the path INTERIOR are lateral offsets; an
        # expert that overruns the det path's end projects onto the endpoint
        # and its longitudinal overhang would masquerade as a lateral gap.
        interior = (a_gt > _EPS) & (a_gt < float(ref_arc[-1]) - _EPS)
        if interior.any():
            a_int, d_int = a_gt[interior], d_gt[interior]
            if float(np.max(np.abs(d_int))) > _PATHS_DIVERGED_M:
                return _ret(None, "paths_diverged", max_lateral_gap_m=float(np.max(np.abs(d_int))))
            # Expert lateral at the arc where the morph ends.
            a_mono = np.maximum.accumulate(a_int)
            d_end = float(np.interp(s_feasible[-1], a_mono, d_int))
    arc_span = max(float(s_feasible[-1] - s_feasible[0]), _EPS)
    if abs(d_end) > _EPS:
        # Peak slope of d_end*scale*quintic(u, w_max) is 1.875*w_max*scale*|d_end|/span,
        # so w_max belongs in the denominator — without it the crab-angle bound
        # is violated for w_max > 1.
        scale = min(1.0, _DEFAULT_LATERAL_SLOPE_CAP * arc_span / (1.875 * w_max * abs(d_end)))
    else:
        scale = 0.0
    u_arc = (s_feasible - s_feasible[0]) / arc_span
    d = d_end * scale * _quintic_smoothstep(u_arc, w_max)

    left_normal = np.stack([-np.sin(th), np.cos(th)], axis=1)
    recon_xy = base + d[:, None] * left_normal
    # Heading correction for the lateral slope; d is a smooth function of the
    # arc, so beta is smooth and vanishes wherever the arc stops advancing.
    ds = np.concatenate([[0.0], np.diff(s_feasible)])
    dd = np.concatenate([[0.0], np.diff(d)])
    beta = np.where(ds > _EPS, np.arctan2(dd, np.maximum(ds, _EPS)), 0.0)
    heading = th + beta

    out = np.empty((T, 4), dtype=np.float32)
    out[:, 0] = recon_xy[:, 0]
    out[:, 1] = recon_xy[:, 1]
    out[:, 2] = np.cos(heading)
    out[:, 3] = np.sin(heading)
    return _ret(
        out,
        "ok",
        terminal_err_m=terminal_err,
        clamp_fire_fraction=fired_fraction,
        v0_mps=float(v_det[0]),
    )
