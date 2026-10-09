"""The object an object_avoidance window avoids, found from the recording alone.

Closed loop only. Used by ``geometry.score_object_avoidance`` for reported values; the
verdict does not read it.

Agent tracks. The recorded frames carry no track ids. Each frame's agents are chained to
the previous frame's by position: an agent's second-to-last ``neighbor_agents_past`` point
(the previous frame's time) taken to the world frame must lie within ``TRACK_MATCH_M`` of
a previous-frame agent's current position (nearest only, each previous agent used once).
Unlike ``conflict.agent_tracks`` (every ``TRACK_STRIDE`` frames, positions only) this
keeps every frame's box -- heading, width, length -- which the clearances need.

Swerve. The human's signed lateral offset from the route lanes (all valid
``route_lanes`` points of the frame joined into one polyline) peaks somewhere in the
scored stretch; its sign is the swerve side. A frame is "swerved" when the offset towards
that side is at least ``SWERVE_MIN_M``.

Target: an agent that, in some swerved frame of the stretch, sits ``AHEAD_MIN_M ..
AHEAD_MAX_M`` ahead of the human (its ego frame) and ``BAND_MIN_M .. BAND_MAX_M`` from the
route-lane centre away from the swerve side (the kerb), and is slow there: stopped
(a vehicle that moved at most ``STOPPED_DISP_M`` over at least ``STOPPED_HISTORY_S`` of
history and ``STOPPED_SPEED_MPS`` over the last second) or slower than
``max(SLOW_MIN_MPS, SLOW_HUMAN_RATIO * human speed)`` (pedestrians and bicycles: slower
than ``max(SLOW_MIN_MPS, human speed)``). Agents the human overtook (their arc along the
route lane went from ahead of the human to behind while in that band) come first, then
the one the human passed closest (OBB clearance over the stretch). None when no agent
qualifies.

Clearances are the rollout's OBB definition (``scenario_generation.metrics.object``):
closest-point distance between the ego box (from ``ego_shape``, rear-axle pose) and the
agent box, 0 on overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from diffusion_planner.model.guidance.collision import batch_signed_distance_rect

from planner_metrics.geometry import _closest_points_between_rects
from scenario_generation.metrics.object import _ego_neighbor_obb
from scenario_generation.scenario_metrics.base import ClosedLoopScenarioInput, project_onto_path

REC_DT_S = 0.1
TRACK_MATCH_M = 0.3
SWERVE_MIN_M = 1.0
AHEAD_MIN_M, AHEAD_MAX_M = -8.0, 30.0
BAND_MIN_M, BAND_MAX_M = -1.5, 2.5
STOPPED_HISTORY_S = 2.0
STOPPED_DISP_M = 1.0
STOPPED_SPEED_MPS = 0.3
SLOW_MIN_MPS = 1.0
SLOW_HUMAN_RATIO = 0.5
TYPES = ("vehicle", "pedestrian", "bicycle")


@dataclass(frozen=True)
class AvoidTarget:
    agent_type: str
    stopped: bool  # stopped in some qualifying frame (else only slow)
    overtaken: bool  # the human overtook it within the stretch
    human_clearance_m: float  # the human's min OBB clearance to it over the stretch
    boxes: dict[int, np.ndarray]  # recorded frame -> world (x, y, heading, width, length)


def obb_clearance(boxes_local: np.ndarray, ego_shape: np.ndarray) -> np.ndarray:
    """(M,) OBB clearance (0 on overlap) from an ego at the origin heading +x to boxes
    ``(x, y, heading, width, length)`` in its frame."""
    b = np.asarray(boxes_local, dtype=np.float64).reshape(-1, 5)
    if not len(b):
        return np.zeros(0)
    nb = np.zeros((len(b), 11))
    nb[:, 0], nb[:, 1] = b[:, 0], b[:, 1]
    nb[:, 2], nb[:, 3] = np.cos(b[:, 2]), np.sin(b[:, 2])
    nb[:, 6], nb[:, 7] = b[:, 3], b[:, 4]
    ego, npc, _ = _ego_neighbor_obb(nb, np.asarray(ego_shape, dtype=np.float64), "cpu")
    with torch.no_grad():
        p1, p2 = _closest_points_between_rects(ego, npc)
        clearance = (p1 - p2).norm(dim=-1).numpy().astype(np.float64)
        overlap = (batch_signed_distance_rect(ego, npc) < 0).numpy()
    return np.where(overlap, 0.0, clearance)


def to_pose_frame(boxes_world: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """World boxes ``(x, y, heading, width, length)`` into the frame of pose (x, y, yaw)."""
    b = np.asarray(boxes_world, dtype=np.float64).reshape(-1, 5)
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = b[:, 0] - x, b[:, 1] - y
    return np.stack([dx * c + dy * s, -dx * s + dy * c, b[:, 2] - yaw, b[:, 3], b[:, 4]], 1)


def _agents(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(past (N, T, 4) x/y/cos/sin, width (N,), length (N,), type index (N,))."""
    past = np.asarray(frame["neighbor_agents_past"], dtype=np.float64)
    if past.shape[-1] >= 11:  # legacy layout: width, length, one-hot type in the columns
        width, length, label = past[:, -1, 6], past[:, -1, 7], past[:, -1, 8:11]
    else:
        n = past.shape[0]
        shape = np.asarray(frame.get("agent_shape", np.zeros((n, 2))), dtype=np.float64)
        width, length = shape[:, 0], shape[:, 1]
        label = np.asarray(frame.get("agent_label", np.zeros((n, 3))), dtype=np.float64)
    return past[..., :4], np.abs(width), np.abs(length), label.argmax(axis=1)


def _route_polyline(frame: dict) -> np.ndarray | None:
    lanes = frame.get("route_lanes")
    if lanes is None:
        return None
    lanes = np.asarray(lanes, dtype=np.float64)[..., :2]
    valid = np.abs(lanes).sum(axis=-1) > 0
    pts = [lanes[i][valid[i]] for i in range(len(lanes)) if valid[i].any()]
    if not pts:
        return None
    p = np.concatenate(pts)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-3]]
    return p if len(p) >= 2 else None


def _chain(prev_xy: np.ndarray, prev_ids: np.ndarray, query: np.ndarray, ok: np.ndarray):
    """Track id per query point (-1 = new): nearest previous agent within TRACK_MATCH_M,
    each previous agent taken once, in query order."""
    out = -np.ones(len(query), dtype=np.int64)
    if not len(prev_xy) or not len(query):
        return out
    d = np.linalg.norm(query[:, None] - prev_xy[None], axis=-1)
    j = d.argmin(axis=1)
    used = set()
    for r in range(len(query)):
        if ok[r] and d[r, j[r]] < TRACK_MATCH_M and j[r] not in used:
            out[r] = prev_ids[j[r]]
            used.add(j[r])
    return out


def find_avoid_target(
    inp: ClosedLoopScenarioInput, first: int, last: int, ego_shape: np.ndarray, until: int
) -> AvoidTarget | None:
    """The avoided agent over recorded frames ``first .. last`` (see module docstring).

    Tracks are followed to frame ``until`` (>= ``last``) so the returned boxes cover every
    frame the scored sim steps replayed.
    """
    boxes: dict[int, dict[int, np.ndarray]] = {}  # track -> frame -> box
    types: dict[int, int] = {}
    rows = []  # (frame, track, ahead x, arc rel., lane offset, speed|nan, stopped, type, clr)
    offset = np.full(last - first + 1, np.nan)
    prev_xy, prev_ids = np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    next_id = 0
    for f in range(first, until + 1):
        try:
            frame = inp.load_frame(f)
        except KeyError:
            prev_xy, prev_ids = np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
            continue
        past, width, length, typ = _agents(frame)
        n_hist = past.shape[1]
        idx = np.flatnonzero(np.abs(past[:, -1, :2]).sum(axis=1) > 0)
        cur = past[idx, -1]
        world = inp.to_world(cur[:, :2], f)
        before = past[idx, -2, :2] if n_hist > 1 else np.zeros((len(idx), 2))
        ids = _chain(prev_xy, prev_ids, inp.to_world(before, f), np.abs(before).sum(axis=1) > 0)
        for r in np.flatnonzero(ids < 0):
            ids[r] = next_id
            next_id += 1
        heading = inp.rec_yaw[f] + np.arctan2(cur[:, 3], cur[:, 2])
        for r, i in enumerate(idx):
            t = int(ids[r])
            types.setdefault(t, int(typ[i]))
            boxes.setdefault(t, {})[f] = np.array(
                [world[r, 0], world[r, 1], heading[r], width[i], length[i]]
            )
        prev_xy, prev_ids = world, ids
        if f > last:
            continue  # past the stretch: boxes only, for the scored steps
        poly = _route_polyline(frame)
        if poly is not None:
            s_h, d_h = project_onto_path(np.zeros((1, 2)), poly)
            offset[f - first] = d_h[0]
        if not len(idx):
            continue
        local = np.stack(
            [cur[:, 0], cur[:, 1], np.arctan2(cur[:, 3], cur[:, 2]), width[idx], length[idx]], 1
        )
        clr = obb_clearance(local, ego_shape)
        if poly is not None:
            s_a, d_a = project_onto_path(cur[:, :2], poly)
            s_rel, lane = s_a - s_h[0], d_a
        else:
            s_rel, lane = cur[:, 0], np.full(len(idx), np.nan)
        hist = past[idx]
        seen = np.abs(hist[..., :2]).sum(axis=-1) > 0
        k0 = seen.argmax(axis=1)
        hist_s = (n_hist - 1 - k0) * REC_DT_S
        disp = np.linalg.norm(hist[:, -1, :2] - hist[np.arange(len(idx)), k0, :2], axis=1)
        sec = min(11, n_hist)
        last_ok = seen[:, -sec:].all(axis=1)
        speed = np.linalg.norm(hist[:, -1, :2] - hist[:, -sec, :2], axis=1) / ((sec - 1) * REC_DT_S)
        stopped = (
            (typ[idx] == 0)
            & (hist_s >= STOPPED_HISTORY_S)
            & (disp <= STOPPED_DISP_M)
            & last_ok
            & (speed <= STOPPED_SPEED_MPS)
        )
        speed = np.where(last_ok, speed, np.nan)
        for r in range(len(idx)):
            rows.append(
                (f, int(ids[r]), cur[r, 0], s_rel[r], lane[r], speed[r], stopped[r], clr[r])
            )
    if not rows or not np.isfinite(offset).any():
        return None
    side = float(np.sign(offset[int(np.nanargmax(np.abs(offset)))]))
    if side == 0.0:
        return None
    cands: dict[int, dict] = {}
    human_clr: dict[int, float] = {}
    for f, t, x, s_rel, lane, speed, stop, clr in rows:
        human_clr[t] = min(human_clr.get(t, np.inf), float(clr))
        kerb = -side * lane
        if not (np.isfinite(kerb) and BAND_MIN_M <= kerb <= BAND_MAX_M):
            continue
        if not AHEAD_MIN_M <= x <= AHEAD_MAX_M:
            continue
        c = cands.setdefault(t, {"s_max": -np.inf, "s_min": np.inf, "slow": False, "stop": False})
        c["s_max"], c["s_min"] = max(c["s_max"], s_rel), min(c["s_min"], s_rel)
        if offset[f - first] * side < SWERVE_MIN_M or not np.isfinite(speed):
            continue
        human_v = float(inp.rec_speed[f])
        ratio = SLOW_HUMAN_RATIO if types[t] == 0 else 1.0
        if stop or speed < max(SLOW_MIN_MPS, ratio * human_v):
            c["slow"] = True
            c["stop"] |= bool(stop)
    found = [(t, c) for t, c in cands.items() if c["slow"]]
    if not found:
        return None

    def rank(item):
        t, c = item
        return (not (c["s_max"] > 0 and c["s_min"] < 0), human_clr[t])

    t, c = min(found, key=rank)
    return AvoidTarget(
        agent_type=TYPES[types[t]],
        stopped=c["stop"],
        overtaken=bool(c["s_max"] > 0 and c["s_min"] < 0),
        human_clearance_m=float(human_clr[t]),
        boxes=boxes[t],
    )
