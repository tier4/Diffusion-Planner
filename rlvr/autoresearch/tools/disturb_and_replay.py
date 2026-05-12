#!/usr/bin/env python3
"""Manufacture training-input NPZs by perturbing existing warm scenes.

Three perturbation kinds are applied per scene:

  (1) Subtle pose perturbation (|x|,|y| <= 1.5m, |yaw| <= 10 deg).
      Implemented as: shift ``ego_current_state[0:4]`` to encode a non-zero
      pose. ``ego_agent_past`` is also rigid-shifted by the same transform so
      the past traces lead consistently into the perturbed current pose.

  (2) Parallel-offset history shift. Find the route centerline tangent at
      ego's current arc_s (use the closest valid route_lanes point as the
      anchor) and compute the perpendicular vector. Apply a lateral offset:
      shift ``ego_current_state[0:2]`` and every step of
      ``ego_agent_past[:, 0:2]`` by ``offset * perp``. Heading stays parallel.

  (3) Random pose+history jitter combo. Same as (1) plus per-step noise on
      ``ego_agent_past`` to simulate noisy history estimates.

Outputs:

  * ``<output_dir>/<source_stem>_var<NN>.npz`` — perturbed NPZs ready to
    feed into ranked-SFT (or any tool that accepts a list of NPZ paths).
    Map + neighbor + static + goal fields are transformed into the NEW
    ego-baselink frame (perturbed ego at origin) so reward.py and the model
    see a self-consistent frame. This was added 2026-05-12 — earlier
    versions only shifted ``ego_current_state`` and ``ego_agent_past``,
    leaving lanes/route_lanes/line_strings/polygons/neighbors/static/goal
    in the OLD frame. The model output is ego-current-pose-relative, so
    when reward.py compared ego_trajs to the unshifted map data, the
    distance was computed in mismatched frames and was wrong by the
    perturbation magnitude. That bug let trajectories crossing the road
    border pass the RB gate (rank-1 winners visually exited the lane
    while ``top1_rb_cross=False``). See git log for details.
    ``ego_agent_future`` is **inherited from the source NPZ unchanged**
    by default — ranked-SFT ignores ``ego_agent_future`` and synthesizes
    its own SFT target from the K-best ranked trajectory at training
    time, so writing into it is unnecessary.

  * ``<output_dir>/manifest.json`` — per-output metadata including
    ``dx, dy, dtheta_deg, dv, lateral_offset_m, longitudinal_offset_m,
    source_scene, kind, variant_name``. Downstream visualization /
    filtering tools key off this.

  * ``<output_scene_list>`` — flat JSON list of kept perturbed NPZ paths.

Variants are rejected when the perturbed pose is out-of-lane at t=0 (we use
``compute_lane_departure_penalty`` so the geometry exactly matches training).

Optional / legacy: passing ``--base_model`` enables a deprecated codepath
that runs the named model in deterministic inference and overwrites
``ego_agent_future`` with its 80-step prediction. This is kept for
backward compat only — the rewrite is wasted compute since ranked-SFT
ignores ``ego_agent_future``, and using a stale baseline's prediction
risks encoding the wrong recovery target. Skip this flag for normal use.

Usage::

    python -m rlvr.autoresearch.tools.disturb_and_replay \
        --scenes /path/to/warm.json \
        --output_dir /path/to/aug_dir \
        --output_scene_list /path/to/aug_list.json \
        --base_model /path/to/<base_model_dir>/best_model.pth \
        [--n_per_scene 5] [--offsets 0.3,0.5,0.8] \
        [--reject_out_of_lane] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from preference_optimization.utils import load_npz_data
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data
from rlvr.reward import compute_lane_departure_penalty


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wrap_angle(rad: float) -> float:
    """Wrap radians to (-pi, pi]."""
    return float((rad + math.pi) % (2 * math.pi) - math.pi)


def _rotate_past_about_pivot(
    past: np.ndarray, pivot_x: float, pivot_y: float, yaw_rad: float
) -> np.ndarray:
    """Rotate ``ego_agent_past`` (T, 3) about (pivot_x, pivot_y) by ``yaw_rad``.

    Mirrors ``recovery_test.apply_yaw_perturbation``. Past format is
    ``(x, y, heading_rad)`` so heading is bumped by ``yaw_rad``.
    """
    out = past.copy()
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    x_off = out[:, 0] - pivot_x
    y_off = out[:, 1] - pivot_y
    out[:, 0] = pivot_x + c * x_off - s * y_off
    out[:, 1] = pivot_y + s * x_off + c * y_off
    out[:, 2] = np.array(
        [_wrap_angle(float(h) + yaw_rad) for h in out[:, 2]], dtype=np.float32
    )
    return out


def _apply_yaw_to_state(
    state: np.ndarray, yaw_rad: float, set_position: tuple[float, float] | None = None
) -> np.ndarray:
    """Rotate ego_current_state heading + velocity + acceleration by ``yaw_rad``.

    Layout: [x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate].

    If ``set_position`` is given (e.g. for combined parallel+yaw), the (x, y)
    channels are overwritten BEFORE the rotation is applied. Otherwise the
    position is left untouched (yaw rotates about the current ego position).
    """
    out = state.copy()
    if set_position is not None:
        out[0] = float(set_position[0])
        out[1] = float(set_position[1])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_old, sin_old = float(out[2]), float(out[3])
    out[2] = c * cos_old - s * sin_old
    out[3] = s * cos_old + c * sin_old
    vx_old, vy_old = float(out[4]), float(out[5])
    out[4] = c * vx_old - s * vy_old
    out[5] = s * vx_old + c * vy_old
    ax_old, ay_old = float(out[6]), float(out[7])
    out[6] = c * ax_old - s * ay_old
    out[7] = s * ax_old + c * ay_old
    return out


def _route_tangent_at_origin(route_lanes: np.ndarray) -> np.ndarray:
    """Return the route centerline unit tangent vector closest to the ego origin.

    Falls back to ``(1, 0)`` (ego forward) when no valid route point is found.
    Route lane channels [0:2] are positions and [2:4] are tangent dx/dy in
    ego frame (see ``scenario_generation/tensor_converter.py::_build_lanes``).
    """
    # route_lanes: (N_lanes, P, D)
    pts = route_lanes[..., :2]  # (N, P, 2)
    tan = route_lanes[..., 2:4]  # (N, P, 2)
    valid = np.abs(route_lanes[..., :8]).sum(axis=-1) > 1e-3
    if not valid.any():
        return np.array([1.0, 0.0], dtype=np.float32)
    flat_pts = pts.reshape(-1, 2)
    flat_tan = tan.reshape(-1, 2)
    flat_valid = valid.reshape(-1)
    flat_pts = flat_pts[flat_valid]
    flat_tan = flat_tan[flat_valid]
    if flat_pts.size == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    dists = np.linalg.norm(flat_pts, axis=1)
    j = int(np.argmin(dists))
    t = flat_tan[j]
    n = float(np.linalg.norm(t))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (t / n).astype(np.float32)


def _build_segments(route_lanes: np.ndarray) -> np.ndarray:
    """Build [N_seg, 2, 2] consecutive valid centerline segments per lane.

    Used for measuring lateral distance from a point to the nearest route
    centerline polyline (same convention as ``recovery_test``).
    """
    if route_lanes.ndim == 2:
        route_lanes = route_lanes[None]
    rl = route_lanes  # [S, P, D]
    S, P, _ = rl.shape
    segs = []
    for s in range(S):
        for p in range(P - 1):
            a = rl[s, p, 0:2]
            b = rl[s, p + 1, 0:2]
            if np.linalg.norm(a) < 1e-3 or np.linalg.norm(b) < 1e-3:
                continue
            segs.append([a, b])
    if not segs:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.array(segs, dtype=np.float64)


def _point_to_segments_dist(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Per-point min perpendicular distance to any of the supplied segments.

    Args:
        points: [T, 2]
        segs: [N, 2, 2]
    Returns:
        [T] distances; ``nan`` if ``segs`` is empty.
    """
    if segs.shape[0] == 0:
        return np.full((points.shape[0],), np.nan)
    a = segs[:, 0, :]  # [N, 2]
    b = segs[:, 1, :]  # [N, 2]
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=-1).clip(min=1e-9)
    ap = points[:, None, :] - a[None, :, :]
    dot = np.sum(ap * ab[None, :, :], axis=-1)
    t = (dot / ab_len_sq[None, :]).clip(0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    diff = points[:, None, :] - proj
    dist = np.linalg.norm(diff, axis=-1)
    return dist.min(axis=-1)


def _apply_rigid_to_past(
    past: np.ndarray, dx: float, dy: float, dtheta: float
) -> np.ndarray:
    """Shift ``ego_agent_past`` so past steps lead into the perturbed current pose.

    Past stays rigidly attached to the ego: every (px, py, ph) gets rotated by
    ``dtheta`` around the new origin and translated by (dx, dy). Heading is
    bumped by ``dtheta`` (then wrapped).

    Args:
        past: (T, 3) array of (x, y, heading_rad).
    Returns:
        Same shape, perturbed.
    """
    out = past.copy()
    c, s = math.cos(dtheta), math.sin(dtheta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = out[:, :2] @ rot.T  # rotate around origin
    xy[:, 0] += dx
    xy[:, 1] += dy
    out[:, :2] = xy
    out[:, 2] = np.array(
        [_wrap_angle(float(h) + dtheta) for h in out[:, 2]], dtype=np.float32
    )
    return out


def _set_ego_current_state(
    state: np.ndarray, dx: float, dy: float, dtheta: float, dv: float = 0.0
) -> np.ndarray:
    """Encode a perturbed pose in ``ego_current_state``.

    Layout: [x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate].
    We modify only [0:4] and [4] (longitudinal velocity).
    """
    out = state.copy()
    out[0] = dx
    out[1] = dy
    out[2] = math.cos(dtheta)
    out[3] = math.sin(dtheta)
    if dv != 0.0:
        out[4] = max(0.0, float(out[4]) + dv)
    return out


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str  # short tag, e.g. "base", "subtle_L", "parallel_R_0.5", "yaw_L_10"
    dx: float
    dy: float
    dtheta_deg: float
    dv: float = 0.0
    history_jitter_std: float = 0.0  # per-step xy gaussian std (m), 0 = none
    # Mode selects how the perturbation is applied. Options:
    #   "shift" - existing kind 1/2 behavior: set position to (dx,dy), set
    #             heading to (cos(dtheta), sin(dtheta)) (overwrites). Velocity
    #             and acceleration are NOT rotated. Past is rigid-shifted then
    #             yaw-rotated about the new origin.
    #   "yaw"   - pure rotation about the current ego pose (0,0). dx/dy are
    #             ignored (must be 0). Velocity/accel rotate by dtheta.
    #             Past is rotated about (0,0) by dtheta.
    #   "combo" - parallel shift then yaw rotation about the SHIFTED pose.
    #             Position becomes (dx,dy); heading/vel/accel rotate by dtheta;
    #             past is rigid-shifted by (dx,dy) then yaw-rotated about
    #             (dx, dy).
    mode: str = "shift"


def _make_subtle_variant(
    rng: np.random.Generator, sign: float, side: str,
    perp: np.ndarray, tangent: np.ndarray,
) -> Variant:
    trans_mag = float(rng.uniform(0.5, 1.5))
    yaw_mag = float(rng.uniform(5.0, 10.0))
    long_frac = float(rng.uniform(-0.5, 0.5))
    d = sign * trans_mag * perp + long_frac * tangent
    dv = float(rng.uniform(-0.5, 0.5))
    return Variant(
        name=f"subtle_{side}",
        dx=float(d[0]),
        dy=float(d[1]),
        dtheta_deg=sign * yaw_mag,
        dv=dv,
        mode="shift",
    )


def _make_parallel_variant(
    rng: np.random.Generator, sign: float, side: str,
    perp: np.ndarray, offset_choices: list[float],
) -> Variant:
    mag = float(rng.choice(offset_choices))
    d = sign * mag * perp
    return Variant(
        name=f"parallel_{side}_{mag:.1f}",
        dx=float(d[0]),
        dy=float(d[1]),
        dtheta_deg=0.0,
        mode="shift",
    )


def _make_yaw_variant(
    rng: np.random.Generator, sign: float, side: str,
    yaw_choices: list[float],
) -> Variant:
    yaw_mag = float(rng.choice(yaw_choices))
    return Variant(
        name=f"yaw_{side}_{yaw_mag:.1f}",
        dx=0.0,
        dy=0.0,
        dtheta_deg=sign * yaw_mag,
        mode="yaw",
    )


def _make_combined_variant(
    rng: np.random.Generator, sign: float, side: str,
    perp: np.ndarray, offset_choices: list[float], yaw_choices: list[float],
) -> Variant:
    mag = float(rng.choice(offset_choices))
    yaw_mag = float(rng.choice(yaw_choices))
    d = sign * mag * perp
    return Variant(
        name=f"combo_{side}_{mag:.1f}_{yaw_mag:.1f}",
        dx=float(d[0]),
        dy=float(d[1]),
        dtheta_deg=sign * yaw_mag,
        mode="combo",
    )


def _build_variants(
    rng: np.random.Generator, offsets: list[float], n_per_scene: int,
    tangent: np.ndarray, kind: str = "default",
    yaw_degs: list[float] | None = None,
) -> list[Variant]:
    """Return a fixed-order list of variants for one scene.

    ``kind`` selects the recipe:

      - ``default``: 1 baseline + 2 subtle (shift) + 2 parallel (shift); extra
        slots become jitter combos. Original tool behavior.
      - ``parallel_only``: 1 baseline + (n-1) parallel variants split L/R
        across the supplied ``offsets``. Used for v4_highmag.
      - ``yaw_only``: 1 baseline + (n-1) yaw variants split L/R across
        ``yaw_degs``. Used for v5_yaw.
      - ``combined``: 1 baseline + 2 subtle + 2 parallel + 2 yaw + 2 combo
        (parallel+yaw). Used for v6_combined.

    For ``parallel_only`` and ``yaw_only``, when ``n_per_scene-1`` does not
    divide evenly across L/R variants, magnitudes are cycled so the order is
    deterministic given the seed.
    """
    perp = np.array([-tangent[1], tangent[0]], dtype=np.float32)  # left of tangent
    offset_choices = list(offsets) if offsets else [0.5]
    yaw_choices = list(yaw_degs) if yaw_degs else [5.0, 10.0, 15.0]

    variants: list[Variant] = [Variant("base", 0.0, 0.0, 0.0)]

    if kind == "default":
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            variants.append(_make_subtle_variant(rng, sign, side, perp, tangent))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            variants.append(_make_parallel_variant(rng, sign, side, perp, offset_choices))

        # If n_per_scene > 5 we add kind-3 jitter combos.
        extra = max(0, n_per_scene - len(variants))
        for k in range(extra):
            sign = +1.0 if (k % 2 == 0) else -1.0
            side = "L" if sign > 0 else "R"
            trans_mag = float(rng.uniform(0.4, 1.0))
            yaw_mag = float(rng.uniform(3.0, 8.0))
            d = sign * trans_mag * perp
            variants.append(
                Variant(
                    name=f"jitter_{side}_{k}",
                    dx=float(d[0]),
                    dy=float(d[1]),
                    dtheta_deg=sign * yaw_mag,
                    history_jitter_std=0.10,
                )
            )

    elif kind == "parallel_only":
        # Distribute (n-1) variants L/R, cycling through magnitudes. For
        # n_per_scene=4 with offsets=[1.0, 1.5] this yields:
        #   L_1.0, R_1.0, L_1.5, R_1.5
        slots = max(0, n_per_scene - 1)
        for i in range(slots):
            mag_idx = i // 2
            mag = offset_choices[mag_idx % len(offset_choices)]
            sign = +1.0 if (i % 2 == 0) else -1.0
            side = "L" if sign > 0 else "R"
            d = sign * mag * perp
            variants.append(
                Variant(
                    name=f"parallel_{side}_{mag:.1f}",
                    dx=float(d[0]),
                    dy=float(d[1]),
                    dtheta_deg=0.0,
                    mode="shift",
                )
            )

    elif kind == "yaw_only":
        slots = max(0, n_per_scene - 1)
        for i in range(slots):
            yaw_idx = i // 2
            yaw_mag = yaw_choices[yaw_idx % len(yaw_choices)]
            sign = +1.0 if (i % 2 == 0) else -1.0
            side = "L" if sign > 0 else "R"
            variants.append(
                Variant(
                    name=f"yaw_{side}_{yaw_mag:.1f}",
                    dx=0.0,
                    dy=0.0,
                    dtheta_deg=sign * yaw_mag,
                    mode="yaw",
                )
            )

    elif kind == "combined":
        # 2 subtle + 2 parallel + 2 yaw + 2 combo, capped to n_per_scene-1.
        recipe = []
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("subtle", sign, side))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("parallel", sign, side))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("yaw", sign, side))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("combo", sign, side))

        slots = max(0, n_per_scene - 1)
        for tag, sign, side in recipe[:slots]:
            if tag == "subtle":
                variants.append(_make_subtle_variant(rng, sign, side, perp, tangent))
            elif tag == "parallel":
                variants.append(_make_parallel_variant(rng, sign, side, perp, offset_choices))
            elif tag == "yaw":
                variants.append(_make_yaw_variant(rng, sign, side, yaw_choices))
            elif tag == "combo":
                variants.append(
                    _make_combined_variant(rng, sign, side, perp, offset_choices, yaw_choices)
                )

    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    if len(variants) > n_per_scene:
        variants = variants[:n_per_scene]
    return variants


def _apply_inverse_rigid_to_spatial(
    out: dict[str, np.ndarray], dx: float, dy: float, dtheta: float
) -> None:
    """Re-anchor every spatial NPZ field to the NEW ego pose (perturbed pose
    becomes the origin (0, 0) with heading along +x).

    The transform is the inverse rigid of (dx, dy, dtheta): rotate by
    ``-dtheta`` about origin, then translate by ``(-dx, -dy)``. Direction
    channels (cos/sin headings, lane direction vectors, neighbor velocity)
    rotate by ``-dtheta``. Yaw channels subtract ``dtheta`` and wrap.

    Fields handled (anchored to OLD ego baselink frame in source NPZ):
      ego_agent_past, ego_agent_future, neighbor_agents_past,
      neighbor_agents_future, static_objects, lanes, route_lanes,
      polygons, line_strings, goal_pose.

    NOT touched (caller handles, or semantic):
      ego_current_state (caller resets to origin), lanes_speed_limit,
      route_lanes_speed_limit, *_has_speed_limit, turn_indicators,
      version.

    Mutates ``out`` in place.
    """
    c, s = math.cos(-dtheta), math.sin(-dtheta)
    # ranks of arrays vary — write helpers that work on the last 2 dims.

    def _xy_inv(arr: np.ndarray, ix: int = 0, iy: int = 1) -> None:
        x = arr[..., ix] - np.float32(dx)
        y = arr[..., iy] - np.float32(dy)
        arr[..., ix] = (c * x - s * y).astype(arr.dtype)
        arr[..., iy] = (s * x + c * y).astype(arr.dtype)

    def _dir_inv(arr: np.ndarray, ix: int, iy: int) -> None:
        x = arr[..., ix].copy()
        y = arr[..., iy].copy()
        arr[..., ix] = (c * x - s * y).astype(arr.dtype)
        arr[..., iy] = (s * x + c * y).astype(arr.dtype)

    def _wrap_yaw_inplace(arr: np.ndarray, idx: int) -> None:
        # Subtract dtheta then wrap to (-pi, pi].
        a = arr[..., idx] - np.float32(dtheta)
        # vectorised wrap
        a = (a + np.pi) % (2 * np.pi) - np.pi
        arr[..., idx] = a.astype(arr.dtype)

    # --- ego_agent_past (T, 3) [x, y, yaw] ---
    if "ego_agent_past" in out:
        past = out["ego_agent_past"]
        # All steps are valid; no zero-padding to worry about
        _xy_inv(past, 0, 1)
        _wrap_yaw_inplace(past, 2)

    # --- ego_agent_future (T, 3) [x, y, yaw] — zeros mark invalid steps ---
    if "ego_agent_future" in out:
        fut = out["ego_agent_future"]
        valid = (fut[:, 0] != 0) | (fut[:, 1] != 0)
        _xy_inv(fut, 0, 1)
        _wrap_yaw_inplace(fut, 2)
        # Restore zeros for originally-invalid slots
        fut[~valid] = 0

    # --- neighbor_agents_past (N, T, 11) ---
    #   channels: [x, y, cos_h, sin_h, vx, vy, w, l, label_oh×3]
    if "neighbor_agents_past" in out:
        nap = out["neighbor_agents_past"]
        valid = (nap[..., 0] != 0) | (nap[..., 1] != 0) | (nap[..., 2] != 0) | (nap[..., 3] != 0)
        _xy_inv(nap, 0, 1)
        _dir_inv(nap, 2, 3)
        if nap.shape[-1] >= 6:
            _dir_inv(nap, 4, 5)  # neighbor velocity (baselink-frame vx, vy)
        # Zero out invalid slots (per parse_rosbag.py the inactive slots are all zero)
        nap[~valid] = 0

    # --- neighbor_agents_future (N, T, 3) [x, y, yaw] ---
    if "neighbor_agents_future" in out:
        naf = out["neighbor_agents_future"]
        valid = (naf[..., 0] != 0) | (naf[..., 1] != 0)
        _xy_inv(naf, 0, 1)
        _wrap_yaw_inplace(naf, 2)
        naf[~valid] = 0

    # --- static_objects (5, 10) ---
    #   channels: [x, y, cos_h, sin_h, w, l, label_oh×4]
    if "static_objects" in out:
        so = out["static_objects"]
        valid = (so[..., 0] != 0) | (so[..., 1] != 0) | (so[..., 2] != 0) | (so[..., 3] != 0)
        _xy_inv(so, 0, 1)
        _dir_inv(so, 2, 3)
        so[~valid] = 0

    # --- lanes (S, P, 33): [x, y, dx, dy, ...other features] per point ---
    if "lanes" in out:
        lanes = out["lanes"]
        valid = (lanes[..., 0] != 0) | (lanes[..., 1] != 0) | (lanes[..., 2] != 0) | (lanes[..., 3] != 0)
        _xy_inv(lanes, 0, 1)
        _dir_inv(lanes, 2, 3)
        # Zero invalid points to preserve "valid = nonzero" convention
        invalid_mask = ~valid
        if invalid_mask.any():
            inv_idx = np.where(invalid_mask)
            lanes[inv_idx[0], inv_idx[1], :] = 0

    # --- route_lanes (S, P, 33): same layout as lanes ---
    if "route_lanes" in out:
        rl = out["route_lanes"]
        valid = (rl[..., 0] != 0) | (rl[..., 1] != 0) | (rl[..., 2] != 0) | (rl[..., 3] != 0)
        _xy_inv(rl, 0, 1)
        _dir_inv(rl, 2, 3)
        invalid_mask = ~valid
        if invalid_mask.any():
            inv_idx = np.where(invalid_mask)
            rl[inv_idx[0], inv_idx[1], :] = 0

    # --- polygons (10, 40, 3) [x, y, intersection_oh] — xy only ---
    if "polygons" in out:
        poly = out["polygons"]
        valid = (poly[..., 0] != 0) | (poly[..., 1] != 0)
        _xy_inv(poly, 0, 1)
        invalid_mask = ~valid
        if invalid_mask.any():
            inv_idx = np.where(invalid_mask)
            poly[inv_idx[0], inv_idx[1], :2] = 0

    # --- line_strings (60, 20, 4) [x, y, stop_oh, rb_oh] — xy only ---
    if "line_strings" in out:
        ls = out["line_strings"]
        valid = (ls[..., 0] != 0) | (ls[..., 1] != 0)
        _xy_inv(ls, 0, 1)
        invalid_mask = ~valid
        if invalid_mask.any():
            inv_idx = np.where(invalid_mask)
            ls[inv_idx[0], inv_idx[1], :2] = 0

    # --- goal_pose (3,) or (4,) [x, y, yaw] (3,) or [x, y, cos, sin] (4,) ---
    if "goal_pose" in out:
        gp = out["goal_pose"].copy()
        if gp.size >= 2:
            x = float(gp[0]) - dx
            y = float(gp[1]) - dy
            gp[0] = c * x - s * y
            gp[1] = s * x + c * y
            if gp.size == 3:
                gp[2] = _wrap_angle(float(gp[2]) - dtheta)
            elif gp.size >= 4:
                # cos/sin form
                cg, sg = float(gp[2]), float(gp[3])
                gp[2] = c * cg - s * sg
                gp[3] = s * cg + c * sg
        out["goal_pose"] = gp


def _apply_variant(
    npz: dict[str, np.ndarray], variant: Variant, rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return a perturbed copy of ``npz`` per ``variant``.

    NOTE: ``ego_agent_future`` is INTENTIONALLY NOT modified here. It will be
    overwritten later with the baseline model's prediction.
    """
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in npz.items()}

    if variant.name == "base":
        return out

    dtheta = math.radians(variant.dtheta_deg)
    mode = getattr(variant, "mode", "shift")
    # All modes converge to "the new ego pose is (dx, dy, dtheta) in the OLD
    # baselink frame, and after this function the NPZ is anchored to that NEW
    # pose so the model sees ego_current_state at (0, 0, 0) — in-distribution."

    if mode == "yaw":
        # Pure rotation about (0, 0). dx = dy = 0 by construction.
        assert variant.dx == 0.0 and variant.dy == 0.0
    # mode "shift" and "combo" both encode the new ego pose in (variant.dx,
    # variant.dy, dtheta) — same downstream transform.

    # Apply the inverse-rigid to every map/neighbor/static/goal field so they
    # are now in the NEW ego-baselink frame.
    _apply_inverse_rigid_to_spatial(out, variant.dx, variant.dy, dtheta)

    # ego_agent_past: rotate xy by -dtheta about origin and translate by
    # (-dx, -dy) so the trail leads correctly into the NEW origin.
    # _apply_inverse_rigid_to_spatial already handled ego_agent_past; only
    # need history jitter on top for the "shift" mode.
    if mode == "shift" and variant.history_jitter_std > 0.0:
        past = out["ego_agent_past"].copy()
        noise = rng.normal(
            loc=0.0, scale=variant.history_jitter_std, size=(past.shape[0], 2)
        ).astype(np.float32)
        past[:, :2] = past[:, :2] + noise
        out["ego_agent_past"] = past

    # ego_current_state: reset to the canonical origin pose (0, 0, heading
    # along +x). Body-frame velocity / accel / steering / yaw_rate stay
    # (they describe the ego's motion relative to its own body — perturbing
    # the ego in space doesn't change them). Optional dv applied per
    # variant.
    ecs = out["ego_current_state"].copy()
    ecs[0] = 0.0
    ecs[1] = 0.0
    ecs[2] = 1.0  # cos(0)
    ecs[3] = 0.0  # sin(0)
    if variant.dv != 0.0 and ecs.size >= 5:
        ecs[4] = max(0.0, float(ecs[4]) + variant.dv)
    out["ego_current_state"] = ecs

    return out


# ---------------------------------------------------------------------------
# Lane membership filter
# ---------------------------------------------------------------------------

@torch.no_grad()
def _is_in_lane(
    perturbed_npz_path: Path, dx: float, dy: float, dtheta: float,
    device: torch.device, threshold: float = 0.15,
) -> tuple[bool, float]:
    """Run ``compute_lane_departure_penalty`` on a 2-step traj at the perturbed pose."""
    data = load_npz_data(str(perturbed_npz_path), device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)

    traj = torch.zeros(1, 2, 4, device=device)
    traj[0, :, 0] = dx
    traj[0, :, 1] = dy
    traj[0, :, 2] = math.cos(dtheta)
    traj[0, :, 3] = math.sin(dtheta)

    gate, near, wide, _, _ = compute_lane_departure_penalty(traj, ego_shape, data)
    if gate.item() < 0.5:
        return False, 0.0
    if near.item() > 0:
        clearance = 0.10
    elif wide.item() > 0:
        clearance = 0.30
    else:
        clearance = 0.50
    return clearance >= threshold, clearance


# ---------------------------------------------------------------------------
# Baseline model inference
# ---------------------------------------------------------------------------

def _load_base_model(
    base_model_path: Path, device: torch.device,
) -> tuple[Diffusion_Planner, Config]:
    """Load the LoRA-less base model + its training config.

    Mirrors the pattern in ``recovery_test._load_model``.
    """
    model_dir = base_model_path.parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(
            f"Could not locate args.json next to {base_model_path}"
        )

    model_args = Config(str(args_path))
    model_args.device = device
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(str(base_model_path), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, model_args


@torch.no_grad()
def _baseline_predict_future(
    model: Diffusion_Planner,
    model_args: Config,
    perturbed_npz_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Run a single deterministic forward pass and return the predicted future.

    Returns:
        ``[T, 3]`` numpy array of ``(x, y, heading_rad)`` in ego frame, suitable
        for direct assignment to ``ego_agent_future``.
    """
    data = load_npz_data(str(perturbed_npz_path), device)
    batch = _stack_scene_data([data], device)
    norm_batch = _normalize_batch(batch, model_args)

    B = norm_batch["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    norm_batch["sampled_trajectories"] = torch.zeros(
        B, P, future_len + 1, 4, device=device
    )

    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    saved_fn = getattr(decoder, "_guidance_fn", None)
    decoder._guidance_fn = None
    try:
        _, outputs = model(norm_batch)
    finally:
        decoder._guidance_fn = saved_fn

    pred = outputs["prediction"][0, 0].detach().cpu().numpy()  # [T, 4] (x,y,cos,sin)
    if pred.shape[0] != future_len:
        # Defensive: clip / pad to expected length
        T = future_len
        out = np.zeros((T, 4), dtype=np.float32)
        n = min(T, pred.shape[0])
        out[:n] = pred[:n]
        pred = out

    # Convert (x, y, cos, sin) -> (x, y, heading_rad)
    fut = np.zeros((pred.shape[0], 3), dtype=np.float32)
    fut[:, 0] = pred[:, 0]
    fut[:, 1] = pred[:, 1]
    fut[:, 2] = np.arctan2(pred[:, 3], pred[:, 2]).astype(np.float32)
    return fut


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_offsets(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, help="JSON list of input warm NPZ paths")
    parser.add_argument("--output_dir", required=True, help="Directory for augmented NPZs")
    parser.add_argument(
        "--output_scene_list",
        required=True,
        help="JSON output: list of resulting NPZ paths",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="(Optional, BUGGY) baseline model. When provided, runs forward "
             "inference and overwrites ego_agent_future — ranked-SFT IGNORES "
             "ego_agent_future, so this codepath is dead weight (per handoff). "
             "Leave unset to dump perturbed NPZs as-is for K=8 winner-recovery "
             "filtering downstream.",
    )
    parser.add_argument("--n_per_scene", type=int, default=5)
    parser.add_argument(
        "--kind",
        type=str,
        default="default",
        choices=["default", "parallel_only", "yaw_only", "combined"],
        help="Recipe for variants per scene. 'default' = 1 base + 2 subtle + "
             "2 parallel (existing). 'parallel_only' = 1 base + (n-1) parallel "
             "split L/R cycling --offsets. 'yaw_only' = 1 base + (n-1) yaw "
             "perturbations cycling --yaw_degs. 'combined' = 1 base + 2 subtle "
             "+ 2 parallel + 2 yaw + 2 (parallel+yaw).",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default="0.3,0.5,0.8",
        help="Comma-separated lateral offsets for parallel-shift kinds (m).",
    )
    parser.add_argument(
        "--yaw_degs",
        type=str,
        default="5,10,15",
        help="Comma-separated yaw magnitudes (deg) for yaw_only / combined kinds.",
    )
    parser.add_argument(
        "--reject_out_of_lane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop variants whose perturbed pose is outside (or within "
             "--reject_threshold of) the lane boundary.",
    )
    parser.add_argument("--reject_threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ego_shape", type=str, required=True,
        help="Ego dimensions as 'WHEEL_BASE,LENGTH,WIDTH' in metres. REQUIRED "
             "— there is no default to fall back to. The values are written "
             "into every output NPZ's `ego_shape` field so downstream "
             "reward.py sees the correct footprint (the previous silent "
             "default undersized the gate by ~3 m on larger platforms).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    _ego_shape_parts = [float(x) for x in args.ego_shape.split(",")]
    if len(_ego_shape_parts) != 3 or any(v <= 0 for v in _ego_shape_parts):
        raise SystemExit(
            f"--ego_shape must be 'WB,LEN,WIDTH' with 3 positive values; "
            f"got {args.ego_shape!r}"
        )
    ego_shape_np = np.array(_ego_shape_parts, dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_list_path = Path(args.output_scene_list)
    out_list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scenes = json.load(f)
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"--scenes must be a non-empty JSON list; got {type(scenes)}")

    offsets = _parse_offsets(args.offsets)
    yaw_degs = _parse_offsets(args.yaw_degs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print(f"[disturb_and_replay] device={device} base_model={args.base_model}")
    if args.base_model is not None:
        base_model, base_args = _load_base_model(Path(args.base_model), device)
    else:
        base_model, base_args = None, None
        print(
            "[disturb_and_replay] --base_model not provided — skipping baseline "
            "inference; ego_agent_future left as in source NPZ."
        )

    written: list[str] = []
    manifest: list[dict] = []  # per-output {npz, source_scene, variant_name, dx, dy, dtheta_deg, dv, kind}
    n_attempted = 0
    n_rejected = 0
    n_inference_failed = 0
    rejection_log: list[dict] = []
    by_kind: dict[str, dict[str, int]] = {}

    # Sanity tracking — measure if the predicted future actually closes the
    # perturbation gap across the dataset.
    recovery_diag: list[dict] = []

    for scene_idx, scene_path in enumerate(scenes):
        scene_path = str(scene_path)
        try:
            with np.load(scene_path) as raw:
                npz = {k: raw[k].copy() for k in raw.files}
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {scene_idx} {scene_path}: load failed ({e})")
            continue

        # Verify the source NPZ's ego_shape matches --ego_shape if present;
        # always write our value into the output (no silent inheritance).
        if "ego_shape" in npz:
            src_es = np.asarray(npz["ego_shape"]).reshape(-1)[:3]
            if not np.allclose(src_es, ego_shape_np, atol=1e-2):
                raise SystemExit(
                    f"Source NPZ {scene_path} has ego_shape={src_es.tolist()} "
                    f"but --ego_shape={ego_shape_np.tolist()}. Pass the matching "
                    f"value or fix the source NPZ — refusing to silently "
                    f"override conflicting dims."
                )
        npz["ego_shape"] = ego_shape_np.copy()

        if "route_lanes" not in npz:
            print(f"[skip] {scene_idx} {scene_path}: no route_lanes")
            continue
        tangent = _route_tangent_at_origin(npz["route_lanes"])
        # Pre-compute centerline segments once per scene for diag.
        scene_segs = _build_segments(npz["route_lanes"])

        scene_stem = Path(scene_path).stem
        variants = _build_variants(
            rng, offsets, args.n_per_scene, tangent,
            kind=args.kind, yaw_degs=yaw_degs,
        )

        for var_idx, variant in enumerate(variants):
            n_attempted += 1
            kind = variant.name.split("_")[0]
            by_kind.setdefault(kind, {"attempted": 0, "kept": 0, "rejected": 0})
            by_kind[kind]["attempted"] += 1

            perturbed = _apply_variant(npz, variant, rng)
            # Carry forward ego_shape (perturbation doesn't change dimensions).
            perturbed["ego_shape"] = ego_shape_np.copy()

            out_path = output_dir / f"{scene_stem}_var{var_idx:02d}.npz"
            np.savez(out_path, **perturbed)

            if args.reject_out_of_lane and variant.name != "base":
                ok, clearance = _is_in_lane(
                    out_path,
                    variant.dx,
                    variant.dy,
                    math.radians(variant.dtheta_deg),
                    device,
                    args.reject_threshold,
                )
                if not ok:
                    n_rejected += 1
                    by_kind[kind]["rejected"] += 1
                    rejection_log.append(
                        {
                            "scene": scene_path,
                            "variant": variant.name,
                            "dx": variant.dx,
                            "dy": variant.dy,
                            "dtheta_deg": variant.dtheta_deg,
                            "clearance": clearance,
                        }
                    )
                    out_path.unlink(missing_ok=True)
                    continue

            if base_model is not None:
                # Legacy baseline-inference codepath. KNOWN-BAD: ranked-SFT
                # ignores ego_agent_future, so the rewrite is wasted and can
                # encode the wrong recovery target. Kept gated for backward
                # compat only.
                try:
                    fut = _baseline_predict_future(base_model, base_args, out_path, device)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[infer-fail] {scene_idx} {variant.name}: {e}; "
                        f"dropping variant."
                    )
                    n_inference_failed += 1
                    by_kind[kind]["rejected"] += 1
                    out_path.unlink(missing_ok=True)
                    continue

                # Sanity: lateral distance to centerline at t=0 (perturbed pose) vs t=79 (end of recovery).
                try:
                    pt0 = np.array([[variant.dx, variant.dy]], dtype=np.float64)
                    d0 = float(_point_to_segments_dist(pt0, scene_segs)[0])
                    pt79 = np.array([fut[-1, :2]], dtype=np.float64)
                    d79 = float(_point_to_segments_dist(pt79, scene_segs)[0])
                    recovery_diag.append({
                        "scene": Path(scene_path).stem,
                        "variant": variant.name,
                        "lat_t0": d0,
                        "lat_t79": d79,
                        "delta": d0 - d79,
                    })
                except Exception:
                    pass

                perturbed["ego_agent_future"] = fut.astype(np.float32)
                np.savez(out_path, **perturbed)
            else:
                # Skip baseline inference. Source NPZ's ego_agent_future
                # carries through unchanged (zeros for sim-source scenes).
                np.savez(out_path, **perturbed)

            written.append(str(out_path))
            manifest.append({
                "npz": str(out_path),
                "source_scene": scene_path,
                "variant_name": variant.name,
                "kind": kind,
                "dx": float(variant.dx),
                "dy": float(variant.dy),
                "dtheta_deg": float(variant.dtheta_deg),
                "dv": float(variant.dv),
                # Lateral offset magnitude (signed, m). Positive = left of tangent
                # (perp = (-ty, tx)), negative = right. Computed by projecting
                # (dx, dy) onto the centerline normal at the source pose.
                "lateral_offset_m": float(variant.dx * (-tangent[1]) + variant.dy * tangent[0]),
                "longitudinal_offset_m": float(variant.dx * tangent[0] + variant.dy * tangent[1]),
            })
            by_kind[kind]["kept"] += 1

        if (scene_idx + 1) % 10 == 0:
            print(
                f"  Processed {scene_idx + 1}/{len(scenes)} scenes — "
                f"{len(written)} kept / {n_rejected} rejected / "
                f"{n_inference_failed} infer-failed so far"
            )

    with open(out_list_path, "w") as f:
        json.dump(written, f, indent=2)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}  ({len(manifest)} entries)")

    # Recovery diagnostic summary
    if recovery_diag:
        deltas = np.array([r["delta"] for r in recovery_diag])
        recovery_summary = {
            "n_diag": len(recovery_diag),
            "mean_lat_t0": float(np.mean([r["lat_t0"] for r in recovery_diag])),
            "mean_lat_t79": float(np.mean([r["lat_t79"] for r in recovery_diag])),
            "mean_delta": float(np.mean(deltas)),
            "p50_delta": float(np.median(deltas)),
            "p95_delta": float(np.percentile(deltas, 95)),
            "frac_recovered": float((deltas > 0.0).mean()),
        }
    else:
        recovery_summary = {}

    summary = {
        "n_input_scenes": len(scenes),
        "n_per_scene_target": args.n_per_scene,
        "n_attempted": n_attempted,
        "n_kept": len(written),
        "n_rejected": n_rejected,
        "n_inference_failed": n_inference_failed,
        "by_kind": by_kind,
        "offsets": offsets,
        "reject_out_of_lane": bool(args.reject_out_of_lane),
        "reject_threshold": args.reject_threshold,
        "seed": args.seed,
        "base_model": str(args.base_model),
        "output_dir": str(output_dir),
        "output_scene_list": str(out_list_path),
        "recovery_summary": recovery_summary,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    rej_path = output_dir / "rejections.json"
    with open(rej_path, "w") as f:
        json.dump(rejection_log, f, indent=2)
    diag_path = output_dir / "recovery_diag.json"
    with open(diag_path, "w") as f:
        json.dump(recovery_diag, f, indent=2)

    print(
        f"\nDone. {len(written)}/{n_attempted} variants kept "
        f"({n_rejected} rejected, {n_inference_failed} infer-failed) "
        f"from {len(scenes)} scenes."
    )
    print("Per-kind:")
    for k, v in by_kind.items():
        print(f"  {k:>10}: kept {v['kept']:>4d} / rejected {v['rejected']:>4d}")
    if recovery_summary:
        rs = recovery_summary
        print(
            "Recovery diagnostic: "
            f"lat_t0={rs['mean_lat_t0']:.3f}m  lat_t79={rs['mean_lat_t79']:.3f}m  "
            f"mean_delta={rs['mean_delta']:+.3f}m  "
            f"frac_recovered={rs['frac_recovered']:.2f}"
        )
    print(f"Wrote scene list: {out_list_path}")
    print(f"Wrote summary  : {summary_path}")


if __name__ == "__main__":
    main()
