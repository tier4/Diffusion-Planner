"""Validity checks for trajectory augmentation, independent of how the
candidate trajectories were produced.

An augmenter's job is to propose displaced trajectories; deciding whether a
proposal is *usable* is a separate question, and the same question for every
augmenter:

  * where can the ego go sideways at all — :func:`border_lateral_bounds` and
    :func:`neighbor_lateral_bounds` give the free lateral interval per timestep,
    which is what a sampler needs BEFORE it has a trajectory;
  * is a proposed trajectory physically drivable — :func:`kinematic_feasibility`
    screens finished polylines on lateral acceleration, jerk, yaw rate and
    Ackermann steering;
  * does it actually hit anything — :func:`veto_overlapping` re-checks winners
    with the canonical exact signed-OBB clearance from ``planner_metrics``.

Everything here is batched over the training batch and takes plain tensors, so
the quintic and bridge augmenters (or a new one) can use the same checks the
frenet augmenter does. Nothing in this module knows what a Frenet frame is.

Ego positions are base_link (rear axle) throughout, matching the dataset and
``planner_metrics``; neighbor boxes are centroid-referenced.
"""

from __future__ import annotations

import math

import torch

from planner_metrics.geometry import build_road_border_segments
from planner_metrics.subscores import compute_ego_neighbor_signed_clearance

DT = 0.1

# Feasibility limits: candidates violating these are REJECTED (never clipped).
LIMITS = {
    "lat_acc": 3.0,  # m/s^2
    "yaw_rate": 0.6,  # rad/s  (~34 deg/s)
    # Comfort jerk, applied to the FUTURE segment only - that is what the model
    # learns to output. (Peak lateral jerk of a quintic ~ 60*dy/T^3: a comfortable
    # recovery from dy=0.75 m needs T >= 2.8 s, so at M=2 s most realistic offsets
    # are correctly REJECTED. The production 2 s quintic bridge teaches ~6 m/s^3.)
    "jerk": 2.0,  # m/s^3
    # The HISTORY segment is context, not a target: it only has to be a plausible
    # (possibly uncomfortable) way to have arrived at the perturbed pose. The 3 s
    # past window would otherwise cap |dy| at 27*jerk/60 ~ 0.9 m.
    "jerk_history": 5.0,  # m/s^3
    # Ackermann / kinematic-bicycle feasibility (needs wheelbase from ego_shape):
    # steering angle delta = atan(WB * kappa), kappa = yaw_rate / speed.
    "steer": 0.61,  # rad (~35 deg), typical passenger-car lock
}

# the steering test runs in tan space (see kinematic_feasibility)
_TAN_STEER = math.tan(LIMITS["steer"])

# lateral bound when nothing constrains the ego; also the distance beyond which
# a border cannot tighten the bounds, hence the segment-pruning reach below.
UNCONSTRAINED_M = 20.0
_BORDER_REACH_M = UNCONSTRAINED_M + 5.0


def ddt(x: torch.Tensor, dim: int) -> torch.Tensor:
    """d/dt along `dim` on the fixed DT grid.

    Same arithmetic as torch.gradient(x, spacing=DT, dim=dim) — central
    differences inside, one-sided at the ends — written out because the library
    call costs ~7x more on the candidate tensors (3.5 ms vs 0.5 ms for the three
    derivatives at B=512). Verified bit-identical to torch.gradient.
    """
    lo = x.narrow(dim, 0, 1)
    lo2 = x.narrow(dim, 1, 1)
    hi2 = x.narrow(dim, x.shape[dim] - 2, 1)
    hi = x.narrow(dim, x.shape[dim] - 1, 1)
    mid = (x.narrow(dim, 2, x.shape[dim] - 2) - x.narrow(dim, 0, x.shape[dim] - 2)) / (2 * DT)
    return torch.cat([(lo2 - lo) / DT, mid, (hi - hi2) / DT], dim=dim)


def time_aligned_neighbor_tracks(
    past: torch.Tensor, fut: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recorded neighbor states on the same time grid as the ego polyline.

    Past for t <= 0, recorded future for t > 0. Futures come in two layouts:
    4-col (x, y, cos, sin) or the canonical 3-col (x, y, heading) — train_epoch
    only converts to cos/sin AFTER augmentation, so both are handled.

    Args:
        past: (B, N, P, >=8) neighbor history; cols 6, 7 are width, length.
        fut: (B, N, F, 3 or 4) recorded neighbor futures.

    Returns:
        state: (B, N, T, 4) x, y, cos, sin with T = P + F.
        valid: (B, N, T) True where the slot holds a real observation.
    """
    if fut.shape[-1] >= 4:
        fut4 = fut[..., :4]
    else:
        fut4 = torch.cat([fut[..., :2], fut[..., 2:3].cos(), fut[..., 2:3].sin()], dim=-1)
        # keep padded rows padded: an all-zero 3-col row must not become
        # (0, 0, 1, 0) after cos/sin
        fut4 = fut4 * (fut.abs().sum(-1, keepdim=True) > 0)
    state = torch.cat([past[..., :4], fut4], dim=2)  # (B, N, T, 4)
    return state, state.abs().sum(-1) > 0


def unconstrained_bounds(B: int, T: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Lateral bounds before any obstacle cuts them.

    NOTE: do not try to pre-shrink these to a sampler's nominal offset. For a
    quintic the offset at t=0 is not a bound on the profile — it overshoots when
    the draw has an initial heading slope (measured |L| = 3.39 m for a 1.98 m
    offset) — and a bound like that silently rejects feasible perturbations.
    """
    lo = torch.full((B, T), -UNCONSTRAINED_M, device=device, dtype=dtype)
    hi = torch.full((B, T), UNCONSTRAINED_M, device=device, dtype=dtype)
    return lo, hi


def border_lateral_bounds(
    line_strings: torch.Tensor,
    xy: torch.Tensor,
    nrm: torch.Tensor,
    half_w: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tighten lateral bounds where a road border blocks the way.

    Casts a ray along the path normal at every timestep and solves for the
    intersection with each border segment in closed form, batched over
    (B, T, segments). Unlike a nearest-distance query this keeps the two sides
    apart, which is what a bound needs: a curb 1 m left and one 1.5 m right must
    not both collapse to "1 m away".

    Args:
        line_strings: (B, L, P, >=4) map polylines; channel 3 flags road border.
        xy: (B, T, 2) ego path (base_link).
        nrm: (B, T, 2) unit path normal.
        half_w: (B,) ego half width, including whatever margin the caller wants.
        lo, hi: (B, T) bounds to tighten.

    Returns:
        the tightened (lo, hi).
    """
    if line_strings.shape[-1] < 4:
        raise ValueError(
            f"line_strings has {line_strings.shape[-1]} channels; the corridor needs "
            "channel 3 (road-border flag)"
        )
    # a border further than the current bound from every path point can never
    # tighten it, so drop those before the ray math (~180 of 1140 slots survive
    # on the pipeline set); the builder pads to the batch maximum, so the
    # surviving bounds are unchanged.
    a, e, sv = build_road_border_segments(line_strings, xy, reach=_BORDER_REACH_M)
    d = e - a
    nx, ny = nrm[..., 0:1], nrm[..., 1:2]  # (B, T, 1)
    dx = d[:, None, :, 0]  # (B, 1, S)
    dy_ = d[:, None, :, 1]
    det = nx * (-dy_) - ny * (-dx)  # (B, T, S)
    rx = a[:, None, :, 0] - xy[..., 0:1]
    ry = a[:, None, :, 1] - xy[..., 1:2]
    s = (rx * (-dy_) - ry * (-dx)) / det
    u = (nx * ry - ny * rx) / det
    ok = torch.isfinite(s) & (u >= 0) & (u <= 1) & sv[:, None, :]
    pos = torch.where(ok & (s > 0), s, torch.full_like(s, torch.inf))
    neg = torch.where(ok & (s < 0), s, torch.full_like(s, -torch.inf))
    return (
        torch.maximum(lo, neg.amax(-1) + half_w[:, None]),
        torch.minimum(hi, pos.amin(-1) - half_w[:, None]),
    )


def neighbor_lateral_bounds(
    state: torch.Tensor,
    valid: torch.Tensor,
    shapes_wl: torch.Tensor,
    xy: torch.Tensor,
    tan: torch.Tensor,
    nrm: torch.Tensor,
    half_l: torch.Tensor,
    half_w: torch.Tensor,
    wb: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tighten lateral bounds where a recorded neighbor blocks the way.

    A neighbor cuts the corridor on the side it sits, over the timesteps where
    it overlaps the ego's longitudinal window.

    Args:
        state, valid: from :func:`time_aligned_neighbor_tracks`.
        shapes_wl: (B, N, 2) neighbor width, length.
        xy, tan, nrm: (B, T, 2) ego path, tangent, normal.
        half_l, half_w, wb: (B,) ego half length, half width, wheel base.
        lo, hi: (B, T) bounds to tighten.

    Returns:
        tightened (lo, hi) and ``near`` (B, N) — neighbors close enough
        longitudinally to be worth an exact overlap check later.
    """
    # Only slots holding a real track can cut the corridor, and they are ~31 of
    # the 320 padded slots. Flatten to the valid (scene, neighbor) pairs so the
    # projections below run over those alone, then reduce back per scene;
    # padding contributed +-inf, i.e. nothing.
    near_any = torch.zeros(valid.shape[:2], dtype=torch.bool, device=valid.device)
    vb, vn = torch.nonzero(valid.any(-1), as_tuple=True)  # (M,)
    if vb.numel() == 0:
        return lo, hi, near_any

    valid_m = valid[vb, vn]  # (M, T)
    w_n = shapes_wl[vb, vn, 0][:, None]  # (M, 1)
    l_n = shapes_wl[vb, vn, 1][:, None]
    c = state[vb, vn, :, :2]  # (M, T, 2)
    axi = state[vb, vn, :, 2:4]
    per = torch.stack([-axi[..., 1], axi[..., 0]], dim=-1)
    tan_m, nrm_m = tan[vb], nrm[vb]  # (M, T, 2)
    half_l_m, half_w_m, wb_m = half_l[vb, None], half_w[vb, None], wb[vb, None]
    rel = c - xy[vb]  # (M, T, 2)
    # ego xy is base_link (rear axle); the footprint CENTER sits wb/2 ahead
    # along the tangent, so shift the longitudinal window there — otherwise a
    # transient neighbor overlapping only the front bumper imposes no cut.
    lon = (rel * tan_m).sum(-1)  # rear-axle window (validated semantics)
    lat = (rel * nrm_m).sum(-1)
    ext_lon = ((tan_m * axi).sum(-1).abs() * l_n + (tan_m * per).sum(-1).abs() * w_n) / 2
    ext_lat = ((nrm_m * axi).sum(-1).abs() * l_n + (nrm_m * per).sum(-1).abs() * w_n) / 2
    near = valid_m & (lon.abs() <= half_l_m + ext_lon)
    # Same test, padded, for the exact-OBB veto: the pad covers the ego's
    # longitudinal half-extent growing under a heading change
    # (half_l*cos + half_w*sin <= half_l + half_w) plus the wb/2 centre shift,
    # so a pair outside it cannot overlap and skipping it changes no verdict.
    near_any[vb, vn] = (valid_m & (lon.abs() <= half_l_m + ext_lon + half_w_m + wb_m / 2)).any(-1)

    cut_hi = torch.where(
        near & (lat > 0), lat - ext_lat - half_w_m, torch.full_like(lat, torch.inf)
    )
    cut_lo = torch.where(
        near & (lat < 0), lat + ext_lat + half_w_m, torch.full_like(lat, -torch.inf)
    )
    idx = vb[:, None].expand_as(cut_hi)
    return (
        lo.scatter_reduce(0, idx, cut_lo, reduce="amax"),
        hi.scatter_reduce(0, idx, cut_hi, reduce="amin"),
        near_any,
    )


def path_metrics(poly_xy: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Lateral accel, lateral jerk, yaw rate and 1/speed of a polyline.

    Finite-differenced on the fixed DT grid, so these are the trajectory's real
    kinematics — including the coupling with the reference path's own curvature,
    which a check on a lateral-offset profile alone would miss.

    Args:
        poly_xy: (..., T, 2) positions at DT spacing.

    Returns:
        lat_a, lat_j, yaw_rate, inv_speed — each (..., T).
    """
    v = ddt(poly_xy, -2)
    # speed is clamped away from zero, so one reciprocal serves the divisions
    # below (a multiply is cheaper than a divide at candidate-tensor sizes)
    inv_sp = v.norm(dim=-1).clamp(min=0.5).reciprocal()
    a = ddt(v, -2)
    lat_a = (v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]) * inv_sp
    jk = ddt(a, -2)
    lat_j = (v[..., 0] * jk[..., 1] - v[..., 1] * jk[..., 0]) * inv_sp
    return lat_a, lat_j, lat_a * inv_sp, inv_sp


def kinematic_feasibility(
    cand_xy: torch.Tensor,
    scene: torch.Tensor,
    ref_xy: torch.Tensor,
    wb: torch.Tensor,
    n_past: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Is each candidate polyline physically drivable?

    Limits are GT-RELATIVE: where the recorded drive itself exceeds a limit
    (data glitches happen), a candidate only has to stay within the recording's
    own maximum. The comfort jerk limit applies to the future segment, which is
    what the model learns to output; the history segment only has to be a
    plausible way to have arrived, so it gets a looser limit.

    Args:
        cand_xy: (M, T, 2) candidate polylines, any provenance.
        scene: (M,) index of the scene each candidate belongs to.
        ref_xy: (B, T, 2) the recorded path per scene.
        wb: (B,) wheel base.
        n_past: number of history timesteps; index T < n_past is history.

    Returns:
        ok: (M,) bool, passes every limit.
        jerk_future_peak: (M,) peak |lateral jerk| over the future segment,
            useful as a tie-break between feasible candidates.
    """
    laC, ljC, yrC, ivC = path_metrics(cand_xy)  # (M, T)
    laG, ljG, yrG, ivG = path_metrics(ref_xy)  # (B, T)
    # Steering compared in tan space: atan(x) <= max(atan(g), STEER) is
    # equivalent to x <= max(g, tan(STEER)) since atan is monotonic, which
    # avoids an atan over the whole candidate tensor.
    stC = (wb[scene, None] * yrC * ivC).abs()
    stG = (wb[:, None] * yrG * ivG).abs()

    def allow(gt_max, limit):
        return torch.clamp(gt_max, min=limit)[scene]  # (M,)

    peak = ljC[:, n_past:].abs().amax(-1)
    a_ok = laC.abs().amax(-1) <= allow(laG.abs().amax(-1), LIMITS["lat_acc"])
    j_ok = (peak <= allow(ljG[:, n_past:].abs().amax(-1), LIMITS["jerk"])) & (
        ljC[:, :n_past].abs().amax(-1)
        <= allow(ljG[:, :n_past].abs().amax(-1), LIMITS["jerk_history"])
    )
    y_ok = yrC.abs().amax(-1) <= allow(yrG.abs().amax(-1), LIMITS["yaw_rate"])
    s_ok = stC.amax(-1) <= allow(stG.amax(-1), _TAN_STEER)
    return a_ok & j_ok & y_ok & s_ok, peak


def veto_overlapping(
    rows: torch.Tensor,
    ego_traj: torch.Tensor,
    ego_shape: torch.Tensor,
    state: torch.Tensor,
    valid: torch.Tensor,
    shapes_wl: torch.Tensor,
    near: torch.Tensor,
) -> torch.Tensor:
    """Clear rows whose footprint truly overlaps a recorded neighbor.

    Lateral bounds are a fast approximation — measured to accept ~1.4% of
    winners whose TRUE footprint overlaps a neighbor box — so the accepted
    trajectory is re-checked with the canonical exact signed-OBB clearance and
    the row is dropped back to plain ground truth if it collides.

    Road borders are deliberately NOT vetoed this way: the recorded drives
    themselves touch the mapped borders (GT corner distance reaches 0.0), so an
    OBB border veto would reject ground-truth-like data.

    Args:
        rows: (B,) bool, which scenes currently intend to use their candidate.
        ego_traj: (B, T, 4) candidate x, y, cos, sin (base_link).
        ego_shape: (B, 3) wheel_base, length, width — one per scene.
        state, valid: from :func:`time_aligned_neighbor_tracks`.
        shapes_wl: (B, N, 2) neighbor width, length.
        near: (B, N) neighbors worth checking, from
            :func:`neighbor_lateral_bounds`.

    Returns:
        ``rows`` with overlapping scenes set False.
    """
    if not bool(rows.any()):
        return rows
    # One paired call for the whole batch: per-scene calls were one tiny kernel
    # each, and their launch overhead dominated training (228 ms/batch at
    # B=512) to reject the ~1.4% of winners that truly overlap.
    bi, ni = torch.nonzero(rows[:, None] & near, as_tuple=True)
    if bi.numel() == 0:
        return rows
    clr = compute_ego_neighbor_signed_clearance(
        ego_traj[bi],
        ego_shape[bi],
        state[bi, ni],
        shapes_wl[bi, ni],
        valid[bi, ni],
        paired=True,
        overlap_only=True,  # a veto only asks whether they overlap
    )  # (M, T)
    rows[bi[clr.amin(-1) < 0.0]] = False
    return rows
