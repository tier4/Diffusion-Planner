"""nuPlan/EPDMS-style scoring of one closed-loop window's realized trajectory.

Inputs are the per-tick ``rollout.jsonl`` rows a windowed ``render_segment`` wrote (live ego
pose / speed / collision flags on the truth clock) and the recorded NPZ frames of the same
ticks (map, route, signals, neighbors -- all in the recorded-ego frame, moved here to the
world frame).  Nothing is re-simulated.

score = NC * DAC * DDC * TLC * MP * (5*EP + 5*TTC + 4*SL + 2*C + 2*LK) / 18

Multiplicative terms (0/1 unless noted):

``NC``   no at-fault collision (per-track first-contact classification, ``metrics.at_fault``).
``DAC``  drivable area: no road-border overlap tick (``road_border`` distance rule).
``DDC``  driving direction (navsim): displacement while the ego centre is outside every
         on-route lane polygon, max over a 1 s window: < 2 m -> 1, < 6 m -> 0.5, else 0.
``TLC``  no red-light entry (``metrics.red_light``); ``tl_measured_frac`` reports how many
         ticks carried a resolved signal at all.
``MP``   making progress: ``EP >= 0.2`` unless the recorded ego itself barely moved.

Weighted terms (nuPlan closed-loop weights 5/5/4/2 + EPDMS lane keeping 2):

``EP``   progress along the recorded route polyline relative to the recorded ego.
``TTC``  time-to-collision (navsim port: constant-velocity projection over 1 s, agent ahead).
``SL``   speed-limit compliance: ``1 - mean_overspeed / 2.23 m/s`` over ticks whose route lane
         carries a limit (nuPlan ``max_overspeed_value_threshold``).
``C``    comfort: every nuPlan bound (lon/lat accel, jerk, yaw rate/accel) holds -> 1 else 0.
``LK``   lane keeping (EPDMS spec): |lateral offset to nearest route centreline| > 0.5 m for
         >= 2 s -> 0.  Exempt inside intersections, while queued, and on ticks where the
         RECORDED ego was itself > 0.5 m off the centreline (the human left the lane too --
         e.g. passing a parked car), so avoidance the log also shows is not punished.

Validity: the evaluated span ends at the first NOT-at-fault contact (a replayed follower
rear-ending the ego, a side brush) because the non-reactive replay is no longer a valid
world after physical contact; the cut and its reason are reported.  A window whose valid
span is shorter than ``min_valid_s`` is marked invalid and excluded from the aggregate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from planner_metrics.pdms_navsim import (
    LK_D_MAX,
    LK_MAX_VIOLATION_S,
    LK_QUEUE_PROGRESS_M,
    LK_QUEUE_RELEASE_S,
    LK_QUEUE_SPEED,
    LK_QUEUE_WINDOW_S,
    ego_is_comfortable,
    states_from_poses,
    time_to_collision,
)
from scenario_generation.tools._heatmap_common import project_points_to_polyline

DT = 0.1
WEIGHTS = {"ep": 5.0, "ttc": 5.0, "sl": 4.0, "comfort": 2.0, "lk": 2.0}
SPEED_LIMIT_MAX_OVERSPEED_MPS = 2.23  # nuPlan speed_limit_compliance
MAKING_PROGRESS_MIN = 0.2  # nuPlan ego_is_making_progress
DDC_COMPLIANCE_M = 2.0  # navsim / nuPlan driving_direction thresholds
DDC_VIOLATION_M = 6.0
DDC_WINDOW_S = 1.0
# lanes[..., 8:13] traffic-light one-hot (scenario_generation.traffic_light): GREEN 0, YELLOW 1,
# RED 2, WHITE 3 (= present but unresolved), NONE 4.  Only 0..2 count as a measured signal.
TL_MEASURED_STATES = (0, 1, 2)
COMFORT_NAMES = ("lon_accel", "lat_accel", "jerk_mag", "lon_jerk", "yaw_accel", "yaw_rate")


@dataclass(frozen=True)
class ScoreConfig:
    min_recorded_progress_m: float = 5.0
    min_valid_s: float = 3.0
    truncate_on_ghost_contact: bool = True
    # Optional speed-aware divergence validity flag (reported, never gating by default).
    divergence_flag_min_m: float = 5.0
    divergence_flag_headway_s: float = 2.0
    polyline_margin_frames: int = 300
    # Lane-keeping exemption zone around recorded avoidance (arc length along the route).
    lk_zone_back_m: float = 15.0
    lk_zone_fwd_m: float = 5.0
    # Object-aware exemption: a stopped agent on the route within this arc-length ahead.
    lk_obstacle_ahead_m: float = 30.0
    lk_obstacle_stopped_mps: float = 0.5
    # Optional clearance term (0 by default: nuPlan-compatible weights only).
    clearance_weight: float = 0.0
    clearance_full_m: float = 0.5
    # Recorded operational stop (bus stop / roadside): stopped this long while off-centre or
    # off the route lanes and not at a red signal.  Tagged, and DDC/LK are exempt in the zone.
    recorded_stop_min_s: float = 3.0
    recorded_stop_speed_mps: float = 0.5
    recorded_stop_offset_m: float = 1.5  # off-centre this much (~lane half width) counts as a bay
    # Recorded ego outside its route lanes for more than this share of the window (parked at
    # the route start / end, depot): bucketed as ``recorded_off_route``.
    recorded_off_route_max_frac: float = 0.5


# --------------------------------------------------------------------------- #
# geometry from recorded frames
# --------------------------------------------------------------------------- #
def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _to_world(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64) @ _rot(float(pose[2])).T + np.asarray(pose[:2])


def _valid_xy(pts: np.ndarray) -> np.ndarray:
    return np.abs(pts[..., :2]).sum(axis=-1) > 1e-6


def map_frame_index(tl, xy: np.ndarray, lo: int, hi: int, margin: int, fallback: int) -> int:
    """Recorded frame whose map window best covers a live ego at ``xy``.

    Route lanes / intersections / borders in a frame are the window around the RECORDED ego
    of that frame.  A live ego that lags behind (or runs ahead of) the recorded one on the
    truth clock can fall outside that window, so map terms read the frame recorded nearest
    to the live position instead -- restricted to ``[lo - margin, hi + margin)`` so a later
    pass over the same road is not picked.  Agents always come from the truth-clock frame.
    """
    i = int(tl.nearest(np.asarray(xy, dtype=np.float64)))
    return i if (lo - margin) <= i < (hi + margin) else int(fallback)


def frame_geometry(npz: dict, pose: np.ndarray) -> dict:
    """World-frame route rings/centrelines, borders, intersections, agents of one frame."""
    route = np.asarray(npz["route_lanes"], dtype=np.float64)
    limits = np.asarray(npz.get("route_lanes_speed_limit", np.zeros((len(route), 1)))).reshape(-1)
    has_limit = np.asarray(
        npz.get("route_lanes_has_speed_limit", np.zeros((len(route), 1), dtype=bool))
    ).reshape(-1)
    rings, centers, ring_limits, tl_measured = [], [], [], False
    for r, lane in enumerate(route):
        valid = _valid_xy(lane) & (np.abs(lane[:, :8]).sum(axis=-1) > 1e-6)
        if valid.sum() < 2:
            continue
        center = lane[valid, :2]
        left = center + lane[valid, 4:6]
        right = center + lane[valid, 6:8]
        rings.append(_to_world(np.concatenate([left, right[::-1]], axis=0), pose))
        centers.append(_to_world(center, pose))
        ring_limits.append(float(limits[r]) if bool(has_limit[r]) else None)
        if lane.shape[1] >= 13 and int(np.argmax(lane[valid][0, 8:13])) in TL_MEASURED_STATES:
            tl_measured = True
    borders = []
    ls = np.asarray(npz.get("line_strings", np.zeros((0, 1, 4))), dtype=np.float64)
    for row in ls:
        if row.shape[1] < 4 or not (row[:, 3] > 0.5).any():
            continue
        pts = row[_valid_xy(row), :2]
        if len(pts) >= 2:
            borders.append(_to_world(pts, pose))
    inters = []
    for poly in np.asarray(npz.get("polygons", np.zeros((0, 1, 3))), dtype=np.float64):
        pts = poly[_valid_xy(poly), :2]
        if len(pts) >= 3:
            inters.append(_to_world(pts, pose))
    nb = np.asarray(npz["neighbor_agents_past"], dtype=np.float64)[:, -1, :]
    alive = np.abs(nb[:, :6]).sum(axis=1) > 0
    nb = nb[alive]
    boxes = np.zeros((len(nb), 9), dtype=np.float64)
    if len(nb):
        boxes[:, :2] = _to_world(nb[:, :2], pose)
        boxes[:, 3] = np.where(np.abs(nb[:, 6]) > 1e-3, np.abs(nb[:, 6]), 2.0)  # width
        boxes[:, 4] = np.where(np.abs(nb[:, 7]) > 1e-3, np.abs(nb[:, 7]), 4.5)  # length
        boxes[:, 5] = 1.5
        boxes[:, 6] = np.arctan2(nb[:, 3], nb[:, 2]) + float(pose[2])
        vel = nb[:, 4:6] @ _rot(float(pose[2])).T
        boxes[:, 7:9] = vel
    rec_lat = _recorded_lat_offset(npz)
    return {
        "route_rings": rings,
        "route_centerlines": centers,
        "route_limits": ring_limits,
        "tl_measured": tl_measured,
        "borders": borders,
        "intersections": inters,
        "agent_boxes": boxes,
        "recorded_lat_offset_m": rec_lat,
    }


def _recorded_on_route(npz: dict, center_offset: float) -> bool:
    """Whether the recorded ego footprint centre lies inside one of its own route lanes."""
    rings = []
    for lane in np.asarray(npz["route_lanes"], dtype=np.float64):
        valid = _valid_xy(lane) & (np.abs(lane[:, :8]).sum(axis=-1) > 1e-6)
        if valid.sum() >= 2:
            center = lane[valid, :2]
            rings.append(
                np.concatenate([center + lane[valid, 4:6], (center + lane[valid, 6:8])[::-1]])
            )
    return _contains(rings, np.array([center_offset, 0.0])) >= 0


def _recorded_red_on_route(npz: dict) -> bool:
    route = np.asarray(npz["route_lanes"], dtype=np.float64)
    for lane in route:
        valid = _valid_xy(lane)
        if valid.sum() and lane.shape[1] >= 13 and int(np.argmax(lane[valid][0, 8:13])) == 2:
            return True
    return False


def _recorded_lat_offset(npz: dict) -> float:
    """Lateral offset of the recorded ego (frame origin) from the nearest route centreline."""
    best = math.inf
    for lane in np.asarray(npz["route_lanes"], dtype=np.float64):
        valid = _valid_xy(lane)
        if valid.sum() >= 2:
            c = lane[valid, :2]
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
            best = min(best, float(project_points_to_polyline(np.zeros((1, 2)), c, s)[0, 2]))
    return best


def _contains(rings: list, xy: np.ndarray):
    """Index of the first ring containing ``xy`` (or -1)."""
    from shapely.geometry import Point, Polygon

    p = Point(float(xy[0]), float(xy[1]))
    for i, ring in enumerate(rings):
        if len(ring) >= 3 and Polygon(ring).contains(p):
            return i
    return -1


def _min_dist_to_lines(xy: np.ndarray, lines: list) -> float:
    best = math.inf
    for ln in lines:
        if len(ln) < 2:
            continue
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ln, axis=0), axis=1))])
        best = min(best, float(project_points_to_polyline(xy.reshape(1, 2), ln, s)[0, 2]))
    return best


# --------------------------------------------------------------------------- #
# rollout rows
# --------------------------------------------------------------------------- #
def read_rollout(rollout_jsonl: Path) -> list[dict]:
    rows = []
    with open(rollout_jsonl) as f:
        for line in f:
            row = json.loads(line)
            if "event" not in row:
                rows.append(row)
    return rows


def valid_span(rows: list[dict], cfg: ScoreConfig) -> tuple[int, dict]:
    """Ticks to score and why the span ended."""
    n = len(rows)
    cut = {"reason": "window_end", "tick": n}
    if cfg.truncate_on_ghost_contact:
        for r in rows:
            types = r.get("collision_types") or []
            if types and not r.get("at_fault", False):
                cut = {"reason": "ghost_contact", "tick": int(r["k"]), "collision_types": types}
                break
    return int(cut["tick"]), cut


# --------------------------------------------------------------------------- #
# per-window score
# --------------------------------------------------------------------------- #
def score_window_epdms(
    tl,
    lo: int,
    hi: int,
    rows: list[dict],
    ego_shape: np.ndarray,
    cfg: ScoreConfig,
    *,
    at_fault_block: dict | None = None,
    terminated: str = "max_steps",
) -> dict:
    wheelbase, length, width = (float(v) for v in np.asarray(ego_shape).reshape(-1)[:3])
    center_offset = 0.5 * wheelbase
    n_cut, cut = valid_span(rows, cfg)
    rows = rows[:n_cut]
    T = len(rows)
    out: dict = {
        "valid_ticks": T,
        "valid_span": cut,
        "invalid": T * DT < cfg.min_valid_s,
    }
    if T == 0:
        out.update({"score": None, "terms": {}})
        return out

    poses = np.array(
        [[r["ego"][0], r["ego"][1], math.cos(r["yaw"]), math.sin(r["yaw"])] for r in rows],
        dtype=np.float64,
    )
    speeds = np.array([float(r["speed"]) for r in rows])
    centres = poses[:, :2] + center_offset * poses[:, 2:4]
    truth = [frame_geometry(tl.npz(lo + t), tl.poses[lo + t]) for t in range(T)]
    map_idx = [
        map_frame_index(tl, centres[t], lo, hi, cfg.polyline_margin_frames, lo + t)
        for t in range(T)
    ]
    cache: dict[int, dict] = {}
    geoms = []
    for t in range(T):
        i = map_idx[t]
        if i == lo + t:
            geoms.append(truth[t])
            continue
        if i not in cache:
            cache[i] = frame_geometry(tl.npz(i), tl.poses[i])
        geoms.append(cache[i])

    # ---- progress / divergence on the recorded polyline ----------------------------------
    n_route = len(tl.poses)
    a, b = max(0, lo - cfg.polyline_margin_frames), min(n_route, hi + cfg.polyline_margin_frames)
    pts = np.asarray(tl.poses[a:b, :2], dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    rec_arc = s[lo - a : hi - a]
    proj = project_points_to_polyline(poses[:, :2], pts, s)
    lon = proj[:, 0] - rec_arc[:T]
    lat = proj[:, 1]
    recorded = float(rec_arc[min(T, len(rec_arc)) - 1] - rec_arc[0])
    realized = float(np.max(proj[:, 0]) - rec_arc[0])
    low_recorded = recorded < cfg.min_recorded_progress_m
    ep = 1.0 if low_recorded else float(np.clip(realized / max(recorded, 1e-6), 0.0, 1.0))
    mp = 1.0 if (low_recorded or ep >= MAKING_PROGRESS_MIN) else 0.0
    rec_speed = np.gradient(rec_arc[:T], DT) if T > 1 else np.zeros(T)
    div_thresh = np.maximum(cfg.divergence_flag_min_m, cfg.divergence_flag_headway_s * rec_speed)
    div_flag = np.abs(lon) > div_thresh

    # ---- NC / DAC / TLC from the rollout rows --------------------------------------------
    at_fault_any = any(bool(r.get("at_fault", False)) for r in rows)
    nc = 0.0 if at_fault_any else 1.0
    rb = np.array([r["rb_dist_m"] if r.get("rb_dist_m") is not None else np.inf for r in rows])
    dac = 0.0 if bool((rb < 0.1).any()) else 1.0
    red = any(bool(r.get("red_light_violation", False)) for r in rows)
    tlc = 0.0 if red else 1.0
    tl_measured_frac = float(np.mean([g["tl_measured"] for g in geoms]))

    # ---- TTC / comfort (navsim ports on the realized trajectory) --------------------------
    states = states_from_poses(poses[None], DT)[0]
    ttc = float(
        time_to_collision(
            states, [g["agent_boxes"] for g in truth], length, width, DT, center_offset
        )
    )
    comfort_ok = ego_is_comfortable(states[None], np.arange(T) * DT)[0]
    comfort = 1.0 if bool(comfort_ok.all()) else 0.0

    # ---- lane keeping with intersection / queue / recorded-offset exemptions ---------------
    d_lat = np.array(
        [_min_dist_to_lines(c, g["route_centerlines"]) for c, g in zip(centres, geoms)]
    )
    in_inter = np.array([_contains(g["intersections"], c) >= 0 for c, g in zip(centres, geoms)])
    # Recorded-avoidance zone: arc-length intervals where the RECORDED ego was off-centre
    # (looked up over the window plus margin so an avoidance the human started before the
    # window still counts), widened backwards/forwards so an earlier or wider avoidance by
    # the ego is not punished.  Ego ticks whose projected arc falls in a zone are exempt.
    zone_lo, zone_hi = (
        max(0, lo - cfg.polyline_margin_frames),
        min(n_route, hi + cfg.polyline_margin_frames),
    )
    zones, route_zones = [], []
    rec_stop_run, rec_stop = 0, False
    rec_off_route_ticks = 0
    for i in range(zone_lo, zone_hi):
        frame = tl.npz(i)
        rec_offset = _recorded_lat_offset(frame)
        off_centre = rec_offset > LK_D_MAX
        off_route = not _recorded_on_route(frame, center_offset)
        in_bay = off_route or rec_offset > cfg.recorded_stop_offset_m
        if lo <= i < hi and off_route:
            rec_off_route_ticks += 1
        arc_i = float(s[i - a]) if a <= i < b else None
        if arc_i is not None and (off_centre or off_route):
            zones.append((arc_i - cfg.lk_zone_back_m, arc_i + cfg.lk_zone_fwd_m))
            if off_route:
                route_zones.append((arc_i - cfg.lk_zone_back_m, arc_i + cfg.lk_zone_fwd_m))
        # Operational stop of the recorded ego inside the window itself.
        if lo <= i < hi:
            stopped = float(tl.speeds[i]) <= cfg.recorded_stop_speed_mps
            if stopped and in_bay and not _recorded_red_on_route(frame):
                rec_stop_run += 1
                if rec_stop_run * DT >= cfg.recorded_stop_min_s:
                    rec_stop = True
            else:
                rec_stop_run = 0
    ego_arc = proj[:, 0]
    rec_off = np.array([any(z0 <= arc_t <= z1 for z0, z1 in zones) for arc_t in ego_arc])
    rec_off_route = np.array(
        [any(z0 <= arc_t <= z1 for z0, z1 in route_zones) for arc_t in ego_arc]
    )
    # Obstacle-aware exemption: a stopped agent on a route lane, ahead of the ego along the
    # route within lk_obstacle_ahead_m (parked car the human did not avoid cleanly either).
    obstacle_ahead = np.zeros(T, dtype=bool)
    for t in range(T):
        boxes = truth[t]["agent_boxes"]
        if len(boxes) == 0:
            continue
        stopped = np.hypot(boxes[:, 7], boxes[:, 8]) <= cfg.lk_obstacle_stopped_mps
        if not stopped.any():
            continue
        arcs = project_points_to_polyline(boxes[stopped, :2], pts, s)
        ahead = (arcs[:, 0] > ego_arc[t]) & (arcs[:, 0] - ego_arc[t] <= cfg.lk_obstacle_ahead_m)
        if not ahead.any():
            continue
        for xy in boxes[stopped][ahead][:, :2]:
            if _contains(geoms[t]["route_rings"], xy) >= 0:
                obstacle_ahead[t] = True
                break
    lk_exempt = rec_off | obstacle_ahead

    # ---- DDC / route adherence / speed limit ----------------------------------------------
    on_route_idx = [_contains(g["route_rings"], c) for g, c in zip(geoms, centres)]
    on_route = np.array([i >= 0 for i in on_route_idx])
    step = np.zeros(T)
    step[1:] = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    off_disp = np.where(on_route | rec_off_route, 0.0, step)
    win = max(1, int(round(DDC_WINDOW_S / DT)))
    worst = max((float(off_disp[max(0, t - win + 1) : t + 1].sum()) for t in range(T)), default=0.0)
    ddc = 1.0 if worst < DDC_COMPLIANCE_M else (0.5 if worst < DDC_VIOLATION_M else 0.0)
    over = []
    for g, i, v in zip(geoms, on_route_idx, speeds):
        limit = g["route_limits"][i] if i >= 0 else None
        if limit is not None and limit > 0:
            over.append(max(0.0, float(v) - limit))
    sl = 1.0 - min(1.0, float(np.mean(over)) / SPEED_LIMIT_MAX_OVERSPEED_MPS) if over else 1.0

    qwin = max(1, int(LK_QUEUE_WINDOW_S / DT))
    queue = np.array(
        [
            speeds[t] <= LK_QUEUE_SPEED
            and step[max(0, t - qwin) : t + 1].sum() <= LK_QUEUE_PROGRESS_M
            for t in range(T)
        ]
    )
    release = np.zeros(T, dtype=bool)
    last_q, grace = -(10**9), int(LK_QUEUE_RELEASE_S / DT)
    for t in range(T):
        if queue[t]:
            last_q = t
        elif t - last_q <= grace:
            release[t] = True
    violation = (d_lat > LK_D_MAX) & ~in_inter & ~queue & ~release & ~lk_exempt
    lk, run = 1.0, 0
    for t in range(T):
        run = run + 1 if violation[t] else 0
        if run * DT >= LK_MAX_VIOLATION_S:
            lk = 0.0
            break

    clr = np.array([r["clearance_m"] if r.get("clearance_m") is not None else np.inf for r in rows])
    moving = speeds > LK_QUEUE_SPEED
    clr_min = (
        float(np.min(clr[moving])) if moving.any() and np.isfinite(clr[moving]).any() else None
    )
    clearance = 1.0 if clr_min is None else float(np.clip(clr_min / cfg.clearance_full_m, 0.0, 1.0))
    weights = dict(WEIGHTS)
    if cfg.clearance_weight > 0:
        weights["clearance"] = float(cfg.clearance_weight)
    values = {"ep": ep, "ttc": ttc, "sl": sl, "comfort": comfort, "lk": lk, "clearance": clearance}
    weighted = sum(w * values[k] for k, w in weights.items()) / sum(weights.values())
    multiplicative = nc * dac * ddc * tlc * mp
    out["recorded_stop"] = bool(rec_stop)
    rec_off_route_frac = rec_off_route_ticks / max(1, hi - lo)
    out["recorded_off_route"] = bool(rec_off_route_frac > cfg.recorded_off_route_max_frac)
    out["bucket"] = (
        "recorded_off_route"
        if out["recorded_off_route"]
        else ("recorded_stop" if rec_stop else "main")
    )
    out.update(
        {
            "score": float(multiplicative * weighted),
            "multiplicative": float(multiplicative),
            "weighted": float(weighted),
            "terms": {
                "nc": nc,
                "dac": dac,
                "ddc": ddc,
                "tlc": tlc,
                "mp": mp,
                "ep": ep,
                "ttc": ttc,
                "sl": sl,
                "comfort": comfort,
                "lk": lk,
                "clearance": clearance,
            },
            "weights": weights,
            "detail": {
                "progress_ratio": ep,
                "low_recorded_progress": bool(low_recorded),
                "realized_progress_m": realized,
                "recorded_progress_m": recorded,
                "ade_lon_m": float(np.mean(np.abs(lon))),
                "ade_lat_m": float(np.mean(np.abs(lat))),
                "lon_final_m": float(lon[-1]),
                "lon_max_behind_m": float(-np.min(np.minimum(lon, 0.0))),
                "lon_max_ahead_m": float(np.max(np.maximum(lon, 0.0))),
                "divergence_flag_ticks": int(div_flag.sum()),
                "ddc_worst_offroute_m": float(worst),
                "route_adherence_frac": float(on_route.mean()),
                "tl_measured_frac": tl_measured_frac,
                "overspeed_mean_mps": float(np.mean(over)) if over else None,
                "overspeed_max_mps": float(np.max(over)) if over else None,
                "speed_limit_ticks": int(len(over)),
                "comfort_failed": [n for n, ok in zip(COMFORT_NAMES, comfort_ok) if not ok],
                "lk_offset_p95_m": float(np.percentile(d_lat[np.isfinite(d_lat)], 95))
                if np.isfinite(d_lat).any()
                else None,
                "lk_exempt_recorded_off_frac": float(rec_off.mean()),
                "lk_exempt_obstacle_frac": float(obstacle_ahead.mean()),
                "ddc_exempt_recorded_off_route_frac": float(rec_off_route.mean()),
                "recorded_off_route_frac": float(rec_off_route_frac),
                "clearance_min_moving_m": clr_min,
                "lk_exempt_intersection_frac": float(in_inter.mean()),
                "map_frame_follows_ego_frac": float(
                    np.mean([i != lo + t for t, i in enumerate(map_idx)])
                ),
                "at_fault": at_fault_block,
                "terminated": terminated,
            },
        }
    )
    return out
