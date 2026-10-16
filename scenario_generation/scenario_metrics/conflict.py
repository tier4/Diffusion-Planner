"""Who a yield is for, where its path meets the ego's, and the post-encroachment time.

Closed loop only. Used by ``progress.yield_conflict``.

Agent tracks. The recorded frames carry no track ids, but each frame's
``neighbor_agents_past`` holds every agent's last second (and more), ego-centric at
that frame. Every ``TRACK_STRIDE`` frames the last ``TRACK_STRIDE + 1`` points are
taken to the world frame and chained to the previous sample's agents by position at
the shared time (within ``TRACK_MATCH_M``). Only the past is read: NPZ windows are
loaded without the neighbors' future.

Conflicts, on the recorded ego path (arc ``s``, signed lateral ``d``):

- crossing: the agent's center comes within ego half width + agent half width +
  margin (``VEHICLE_MARGIN_M``; ``VRU_MARGIN_M`` for pedestrians and bicycles, whose
  whole crossing a driver waits for) while moving (``MIN_SPEED_MPS``) at
  ``MIN_CROSSING_ANGLE_DEG`` or more to the path. Conflict point = median arc while
  inside the band; the conflict lasts while it stays inside. A crossing longer than
  ``MAX_CROSSING_S`` is not one (a lead vehicle or a parked car on the path);
- merging: a vehicle or bicycle that enters the band from outside and then stays on
  the path longer than ``MAX_CROSSING_S`` (the ego waits to turn into its road).
  Conflict point = where it entered; it has cleared it ``MERGE_CLEAR_M`` further on.

Yield target: a conflict ``0 .. TARGET_MAX_DIST_M`` ahead of the human's front at the
anchor, active within the anchor's span (+-1 s); the last one the human let go first,
else the one with the shortest human PET. None means no agent crosses or merges into
the ego's path near the anchor: the anchor is likely not a yield.

PET (post-encroachment time) is the time from the agent clearing the conflict point to
the ego's front reaching it; when the ego's rear clears it before the agent arrives,
it is negative (the ego went first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario_generation.scenario_metrics.base import ClosedLoopScenarioInput, project_onto_path

REC_DT_S = 0.1
TRACK_STRIDE = 10
TRACK_MATCH_M = 0.5
VEHICLE_MARGIN_M = 0.3
VRU_MARGIN_M = 2.0
MIN_SPEED_MPS = 0.3
MIN_CROSSING_ANGLE_DEG = 20.0
MAX_CROSSING_S = 8.0
MERGE_CLEAR_M = 3.0
TARGET_MAX_DIST_M = 20.0
SPAN_SLACK_FRAMES = 10
# Ego width when the frames carry no ``ego_shape``.
DEFAULT_EGO_WIDTH_M = 2.0
TYPES = ("vehicle", "pedestrian", "bicycle")


@dataclass(frozen=True)
class Conflict:
    agent_type: str
    kind: str  # "cross" | "merge"
    s_m: float  # conflict point, arc along the recorded path
    half_w_m: float  # agent half width counted into the conflict point
    first_frame: int  # agent enters the conflict
    last_frame: int  # agent has cleared it at the next frame


@dataclass(frozen=True)
class Pet:
    seconds: float | None
    order: str  # "agent_first" | "ego_first" | "overlap" | "ego_never_reached" | "agent_never_replayed"


def ego_offsets(frame: dict) -> tuple[float, float, float]:
    """(front, rear, width) of the ego from the rear-axle pose, from ``ego_shape``
    ``(wheelbase, length, width)``; zero offsets without it."""
    shape = frame.get("ego_shape")
    if shape is None:
        return 0.0, 0.0, DEFAULT_EGO_WIDTH_M
    wb, length, width = (float(v) for v in np.asarray(shape, dtype=np.float64).reshape(-1)[:3])
    return (wb + length) / 2.0, (length - wb) / 2.0, width


def _agent_attrs(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(past positions (N, T, 2), type index (N,), half width (N,)) of one frame."""
    past = np.asarray(frame["neighbor_agents_past"], dtype=np.float64)
    if past.shape[-1] >= 11:  # legacy layout: width, length, one-hot type in the columns
        label, width = past[:, -1, 8:11], past[:, -1, 6]
    else:
        n = past.shape[0]
        label = np.asarray(frame.get("agent_label", np.zeros((n, 3))), dtype=np.float64)
        width = np.asarray(frame.get("agent_shape", np.zeros((n, 2))), dtype=np.float64)[:, 0]
    return past[..., :2], label.argmax(axis=1), np.abs(width) / 2.0


def agent_tracks(inp: ClosedLoopScenarioInput) -> list[dict]:
    """``[{"type", "half_w", "pos": {frame: xy}}]`` over the window's recorded frames."""
    tracks: list[dict] = []
    last: dict[int, np.ndarray] = {}  # track -> world xy at the current sample frame
    samples = list(range(TRACK_STRIDE, inp.n_frames, TRACK_STRIDE))
    if inp.n_frames - 1 not in samples:
        samples.append(inp.n_frames - 1)
    for f in samples:
        try:
            frame = inp.load_frame(f)
        except KeyError:
            continue
        past, types, half_w = _agent_attrs(frame)
        c, s = np.cos(inp.rec_yaw[f]), np.sin(inp.rec_yaw[f])
        rot = np.array([[c, -s], [s, c]])
        k = min(TRACK_STRIDE + 1, past.shape[1])
        nxt: dict[int, np.ndarray] = {}
        for i in np.flatnonzero(np.abs(past[:, -1]).sum(axis=1) > 0):
            pts = past[i, -k:]
            ok = np.abs(pts).sum(axis=1) > 0
            world = pts @ rot.T + inp.rec_xy[f]
            frames = np.arange(f - k + 1, f + 1)
            start = world[0] if ok[0] else None
            best, best_d = None, TRACK_MATCH_M
            if start is not None:
                for t, xy in last.items():
                    d = float(np.linalg.norm(xy - start))
                    if d < best_d:
                        best, best_d = t, d
            if best is None:
                tracks.append({"type": TYPES[int(types[i])], "half_w": float(half_w[i]), "pos": {}})
                best = len(tracks) - 1
            pos = tracks[best]["pos"]
            for fr, xy, valid in zip(frames, world, ok):
                if valid and fr >= 0:
                    pos[int(fr)] = xy
            nxt[best] = world[-1]
        last = nxt
    return tracks


def conflicts(
    tracks: list[dict], path_xy: np.ndarray, path_s_end: float, ego_width_m: float
) -> list[Conflict]:
    out: list[Conflict] = []
    for tr in tracks:
        fr = np.array(sorted(tr["pos"]))
        if len(fr) < 5:
            continue
        xy = np.array([tr["pos"][f] for f in fr])
        s, lat = project_onto_path(xy, path_xy)
        seg = np.diff(path_xy, axis=0)
        # Path heading at each point's arc: the segment the arc falls on.
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
        tang = np.arctan2(seg[j, 1], seg[j, 0])
        v = np.gradient(xy, fr * REC_DT_S, axis=0) if len(fr) > 1 else np.zeros_like(xy)
        speed = np.linalg.norm(v, axis=1)
        angle = np.abs((np.arctan2(v[:, 1], v[:, 0]) - tang + np.pi / 2) % np.pi - np.pi / 2)
        margin = VEHICLE_MARGIN_M if tr["type"] == "vehicle" else VRU_MARGIN_M
        inside = (np.abs(lat) <= ego_width_m / 2 + tr["half_w"] + margin) & (s > 0.5)
        inside &= s < path_s_end - 0.5
        moving = speed >= MIN_SPEED_MPS
        crossing = inside & moving & (angle >= np.radians(MIN_CROSSING_ANGLE_DEG))
        contiguous = np.r_[False, np.diff(fr) == 1]
        if crossing.any():
            lo = hi = int(np.flatnonzero(crossing)[0])
            while lo > 0 and inside[lo - 1] and contiguous[lo]:
                lo -= 1
            while hi + 1 < len(fr) and inside[hi + 1] and contiguous[hi + 1]:
                hi += 1
            long_stay = (fr[hi] - fr[lo]) * REC_DT_S > MAX_CROSSING_S
            if not long_stay:
                out.append(
                    Conflict(
                        tr["type"],
                        "cross",
                        float(np.median(s[lo : hi + 1])),
                        tr["half_w"],
                        int(fr[lo]),
                        int(fr[hi]),
                    )
                )
                continue
            if tr["type"] == "pedestrian" or lo == 0:
                continue
            entry = lo
        else:
            if tr["type"] == "pedestrian":
                continue
            entries = np.flatnonzero(inside[1:] & ~inside[:-1] & moving[1:]) + 1
            if not len(entries):
                continue
            entry = int(entries[0])
        s0 = float(s[entry])
        past = np.flatnonzero(s[entry:] >= s0 + MERGE_CLEAR_M)
        if len(past):
            out.append(
                Conflict(tr["type"], "merge", s0, 0.0, int(fr[entry]), int(fr[entry + past[0]]))
            )
    return out


def pet(
    front_s: np.ndarray,
    rear_s: np.ndarray,
    agent_in: int | None,
    agent_out: int | None,
    c: Conflict,
    dt: float,
) -> Pet:
    """PET on one clock: ego front/rear arc per index, the index the agent enters the
    conflict and the first index after it cleared it (None if never)."""
    reach = np.flatnonzero(front_s >= c.s_m - c.half_w_m)
    if not len(reach):
        return Pet(None, "ego_never_reached")
    k_reach = int(reach[0])
    clear = np.flatnonzero(rear_s >= c.s_m + c.half_w_m)
    if agent_in is not None and len(clear) and clear[0] <= agent_in:
        return Pet(float(clear[0] - agent_in) * dt, "ego_first")
    if agent_out is not None and agent_out <= k_reach:
        return Pet(float(k_reach - agent_out) * dt, "agent_first")
    if agent_in is None:
        return Pet(None, "agent_never_replayed")
    return Pet(0.0, "overlap")


def find_target(
    inp: ClosedLoopScenarioInput,
) -> tuple[Conflict, Pet, int, tuple[float, float, np.ndarray, float]] | None:
    """The yield target of ``inp``'s anchor and span, from the recording alone.

    Returns ``(target, human PET, number of candidate conflicts, (ego front offset,
    rear offset, recorded path, human front arc at the anchor))``, or None when no
    agent crosses or merges into the recorded path near the anchor. Reads only the
    recorded frames, so open loop can use it with the same window.
    """
    try:
        anchor = inp.load_frame(inp.anchor_frame)
    except KeyError:
        anchor = {}
    front, rear, width = ego_offsets(anchor)
    path = inp.rec_xy
    rec_s = np.maximum.accumulate(project_onto_path(path, path)[0])
    first, last = inp.span_frames if inp.span_frames is not None else (inp.anchor_frame,) * 2
    anchor_front = float(rec_s[inp.anchor_frame] + front)
    found = [
        c
        for c in conflicts(agent_tracks(inp), path, float(rec_s[-1]), width)
        if 0.0 <= c.s_m - anchor_front <= TARGET_MAX_DIST_M
        and c.first_frame <= last + SPAN_SLACK_FRAMES
        and c.last_frame >= first - SPAN_SLACK_FRAMES
    ]
    if not found:
        return None
    human = [
        (c, pet(rec_s + front, rec_s - rear, c.first_frame, c.last_frame + 1, c, REC_DT_S))
        for c in found
    ]
    let_go = [(c, h) for c, h in human if h.order == "agent_first"]
    if let_go:
        target, human_pet = max(let_go, key=lambda x: x[0].last_frame)
    else:
        target, human_pet = min(human, key=lambda x: (x[1].seconds is None, x[1].seconds or 0.0))
    return target, human_pet, len(found), (front, rear, path, anchor_front)
