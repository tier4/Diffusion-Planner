"""Closed-loop scenario metrics: stop_arrival family.

Owns labels: traffic_light_stop, obstacle_stop, temporal_stop, arrival.

Closed-loop mirrors of ``planner_metrics/stop_overshoot.py`` and
``planner_metrics/arrival.py``, scored on the live ego's realized path after the
anchor instead of one predicted trajectory. Positions are arc lengths along the
recorded path (``project_onto_path``, extended past its end), so "past the stop
point" means further along the road, not further in xy.

``temporal_stop`` (a stop-line stop the human holds briefly before moving on) is scored as
a stop too: its event span runs from the approach to the human's stop at the line, so the
question is the same as at a red light. Open loop still scores it as a yield and has no
stop tolerance for it, so its tolerance is the closed-loop ``TEMPORAL_STOP_TOLERANCE_M``.
Unlike a red light the human departs within the window, so the goal radius rarely ends
the rollout before the stop.

The rollout ends when the live ego comes within ``GOAL_REACH_M`` of the window's last
recorded pose (``terminated == "goal"``). Both metrics must live with that: it hides
the last few metres before the endpoint, which is exactly where an arrival settles and,
when the recording ends with the human still stopped, where a stop happens.
Windows extended until the human departs (``build_scenario_manifest
--extend_until_departure``) put the goal past the stop: the stop metric then sees the
ego's stop as is, and the arrival metric scores the ego's stop at the human's arrival
stop instead of the closest approach to the window's end.

Open-loop reference values (``ol_*``, reported only, never part of the verdict or the
reason) say what the open-loop definition measures on the realized trajectory:

- stop: ``planner_metrics/stop_overshoot.py`` as is. Both paths are expressed in the
  anchor frame's ego frame and projected onto that frame's chained route-lane centerlines
  (``route_lanes``, ``lanes`` as fallback); each stop is the median s of the *final*
  sustained stop within the interval, or its terminal s without one. The interval is the
  open-loop 8 s (``OL_STOP_HORIZON_S``): recorded frames ``anchor_frame`` + 1 .. + 80 for
  the human (speed from 0.1 s pose deltas, as open loop), sim steps ``anchor_step`` + 1 ..
  + ``round(8 / dt)`` for the ego (speed as logged by the rollout), cut short where the
  window or the trace ends -- a goal termination then reads the terminal s ``GOAL_REACH_M``
  short of a stop at the window's end. Omitted when the anchor frame has no usable route
  lane or either interval is empty.
- arrival: the ego's *final* pose in the trace vs the recorded endpoint (FDE and wrapped
  heading error). Biased by the goal radius: a goal termination ends up to
  ``GOAL_REACH_M`` short, which alone exceeds the 2 m open-loop tolerance.

``ol_passed`` (0/1) is the open-loop rule with the same tolerance(s).

traffic_light_stop and (extended) arrival split the stop into reacting and holding. The
label's verdict is the reaction: where the ego first stopped, comparable with open loop.
Holding is closed loop only and dominated by the model creeping at standstill; it is
reported as ``hold_passed`` (0/1, its mean the hold pass rate), not part of the verdict:
the ego's ``hold_creep_m`` against the human's own creep while holding
(``human_hold_creep_m``) plus ``HOLD_CREEP_MARGIN_M``. Only a stop that answers the label
counts as the first stop: for traffic_light_stop one held long enough
(``TRAFFIC_LIGHT_MIN_STOP_S``) and not far before the line (``STOP_FOR_LINE_MAX_SHORT_M``),
for arrival one near the arrival point (``ARRIVAL_STOP_SEARCH_M``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planner_metrics.stop_overshoot import (
    _current_route_segments,
    _project_along_route,
    _speed_mps,
    _stop_position_s,
)
from scenario_generation.scenario_metrics.base import (
    ClosedLoopScenarioInput,
    ScenarioResult,
    path_arclength,
    project_onto_path,
    wrap_angle,
)
from scenario_generation.scenario_metrics.conflict import _agent_attrs
from scenario_generation.scenario_metrics.geometry import _to_local
from scenario_generation.scenario_metrics.registry import register
from scenario_generation.scenario_metrics.shared_config import open_loop_parameters

# Closed-loop-only constants (no open-loop counterpart); the tolerances shared with open
# loop come from ``ScenarioOpenLoopConfig`` through ``StopParams`` / ``ArrivalParams``.

# Goal radius the closed-loop evaluator runs windows with (``closed_loop_eval.py``,
# ``goal_reach_m=5.0``); not in the trace, so mirrored here.
GOAL_REACH_M = 5.0
# Recorded frames are 10 Hz (``loader.SIM_DT_S``); ``rec_speed`` is per frame.
REC_DT_S = 0.1
# The recorded path is extended straight past its last pose by this much before the
# live ego is projected on it; otherwise a window ending at the human's stop point (or
# endpoint) would clamp any overshoot to zero.
PATH_EXTENSION_M = 200.0
# Closed loop only: open loop scores temporal_stop as a yield, so its config has no stop
# tolerance. Human labels failed stops 0.24 m past the line and up; 0.15 m leaves room
# for pose noise only.
TEMPORAL_STOP_TOLERANCE_M = 0.15
# Closed loop only: the shortest ego stop that counts as a temporal stop. Human labels
# called a 0.5 s stop a rolling stop (a fail), so ``sustained_stop_s`` alone is too short.
# Measured as the sustained-stop run itself, which the net-displacement window of
# ``_ego_stop_speed`` makes about 0.3 s shorter than the time spent at rest. Once the ego
# has held such a stop, it is free to move on (see ``score_stop``).
TEMPORAL_STOP_MIN_HOLD_S = 1.0
# Closed loop only: how far the ego may creep on from its first stop while the human holds
# (``hold_passed``), beyond the human's own creep. Humans creep a little while holding,
# e.g. a queue moving up; the margin is a user decision.
HOLD_CREEP_MARGIN_M = 1.0
# The human holds from its first still frame after the anchor (net speed over
# ``HUMAN_STILL_WINDOW_S`` below ``HUMAN_STILL_SPEED_MPS``, i.e. under 5 cm in 0.5 s) to
# its last still frame before it departs (net speed above ``HUMAN_DEPART_SPEED_MPS``).
HUMAN_STILL_WINDOW_S = 0.5
HUMAN_STILL_SPEED_MPS = 0.1
HUMAN_DEPART_SPEED_MPS = 1.0
# Labels whose stop is at a stop line (vse ``stop_line_classifier``): they are judged by
# the ego's front against that line when the human's stop frame has one.
STOP_LINE_LABELS = ("traffic_light_stop", "temporal_stop")
# The stop line is searched from this far behind the human's front (humans stop up to
# ~0.5 m past it) to this far ahead of it.
STOP_LINE_SEARCH_BEHIND_M = 2.0
STOP_LINE_SEARCH_AHEAD_M = 10.0
# Length of the open-loop GT future / prediction ``stop_overshoot`` reads; the interval
# of the ``ol_*`` stop values.
OL_STOP_HORIZON_S = 8.0
# A vehicle stands before the stop line the human stopped for (the human was queued, see
# ``queue_values``) when, at the human's stop frame, its centre is within this lateral
# offset of the recorded path (a lane is ~3 m wide, so the next lane stays out) ...
QUEUE_CORRIDOR_M = 1.5
# ... between the human's front and this far past the stop line (a lead vehicle stopped
# with its front over the line) ...
QUEUE_PAST_LINE_M = 2.0
# ... and nearly stopped: net speed over the last ``QUEUE_SPEED_WINDOW_S`` of its past
# track at most this (speed is ignored when that part of the track is missing).
QUEUE_MAX_SPEED_MPS = 1.0
QUEUE_SPEED_WINDOW_S = 0.5
# traffic_light_stop: the shortest ego stop (sustained-stop run, as
# ``TEMPORAL_STOP_MIN_HOLD_S``) that counts, capped by the human's own stop duration. A
# rolling stop is not a stop; but if the human barely stopped, the light turned green on
# its arrival (the replayed light follows the recording), so a lawful ego need not stop
# longer.
TRAFFIC_LIGHT_MIN_STOP_S = 1.0
# traffic_light_stop: a stop with the ego's front further than this before the stop line
# (or the human's stop without one) is behind a lead vehicle, not for the line (about one
# car length), and does not count. A stop past the line counts (and fails the tolerance).
STOP_FOR_LINE_MAX_SHORT_M = 5.0
# Extended arrival: only ego stops within this distance of the human's arrival stop count;
# earlier ones further away are queueing behind other vehicles.
ARRIVAL_STOP_SEARCH_M = 5.0
QUEUED_REASON = "human was not at the head of the queue (a vehicle stood before the stop line)"


@dataclass(frozen=True)
class StopParams:
    # Open-loop config: ``scenario_{obstacle,traffic_light}_stop_tolerance_m``;
    # ``TEMPORAL_STOP_TOLERANCE_M`` for temporal_stop.
    tolerance_m: float
    # Closed loop only; ``_STOP_SPEED_THRESHOLD_MPS`` / ``_SUSTAINED_STOP_DURATION_S``
    # (stop_overshoot.py) are module constants there, not config fields.
    stop_speed_mps: float = 0.5
    sustained_stop_s: float = 0.5

    @classmethod
    def from_config(cls, label: str, config=None) -> StopParams:
        if label == "temporal_stop":
            return cls(tolerance_m=TEMPORAL_STOP_TOLERANCE_M)
        return cls(tolerance_m=float(open_loop_parameters(label, config)["tolerance_m"]))


@dataclass(frozen=True)
class ArrivalParams:
    # Open-loop config: ``scenario_arrival_{position,heading}_tolerance_*``.
    position_tolerance_m: float
    heading_tolerance_deg: float
    # Radius within which the endpoint counts as reached. The rollout stops the ego at
    # ``GOAL_REACH_M``, so any tighter radius would be decided by termination alone.
    reach_m: float

    @classmethod
    def from_config(cls, config=None) -> ArrivalParams:
        p = open_loop_parameters("arrival", config)
        position_tolerance_m = float(p["position_tolerance_m"])
        return cls(
            position_tolerance_m=position_tolerance_m,
            heading_tolerance_deg=float(p["heading_tolerance_deg"]),
            reach_m=max(position_tolerance_m, GOAL_REACH_M),
        )


# The sustained-stop rule alone (its tolerance is not read), for the arrival stops.
_STOP_RULE = StopParams(tolerance_m=0.0)


def _stop_runs(speed: np.ndarray, dt: float, p: StopParams) -> list[tuple[int, int]]:
    """``[start, end)`` runs with speed <= threshold lasting >= the sustained duration."""
    width = max(1, round(p.sustained_stop_s / dt))
    stopped = np.r_[False, np.asarray(speed) <= p.stop_speed_mps, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(stopped))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= width]


def _stop_steps(run: tuple[int, int], k0: int, dt: float, p: StopParams) -> np.ndarray:
    """Steps of the first ``sustained_stop_s`` of the stop ``run`` at/after ``k0``.

    The window that makes it a stop: where the ego came to rest, before any creep that
    the speed threshold still calls stopped.
    """
    start = max(run[0], k0)
    end = min(run[1], start + round(p.sustained_stop_s / dt))
    return np.arange(start, max(start + 1, end))


def _first_stop_steps(
    live: list[tuple[int, int]], k0: int, dt: float, p: StopParams
) -> np.ndarray | None:
    """``_stop_steps`` of the ego's first stop at/after ``k0``; None without a stop."""
    return _stop_steps(live[0], k0, dt, p) if live else None


def _hold_creep_m(s_ego: np.ndarray, first: np.ndarray, judged: np.ndarray, first_s: float):
    """Furthest arc over the ``judged`` steps from the first stop on, minus that stop."""
    later = judged[judged >= first[0]]
    return float(s_ego[later].max()) - first_s if len(later) else np.nan


def _front_offset_m(frame: dict) -> float | None:
    """Rear axle (the pose) to the front face: ``(wheelbase + length) / 2``, as the
    evaluator's box (centre half a wheelbase ahead of the axle)."""
    shape = frame.get("ego_shape")
    if shape is None:
        return None
    shape = np.asarray(shape, dtype=np.float64).reshape(-1)
    return float(shape[0] + shape[1]) / 2.0


def _stop_line_arc(
    inp: ClosedLoopScenarioInput, frame_idx: int, human_s: float
) -> tuple[float, float] | None:
    """``(arc of the stop line, front offset)`` the human stopped for, or None.

    ``human_s`` is the human's stop (rear-axle arc) and ``frame_idx`` a frame of it.

    The stop lines of the human's stop frame (``stop_lines``, ego-centric segments) are
    taken to the world frame; the line the human stopped for is the first one crossing
    the recorded path within ``STOP_LINE_SEARCH_BEHIND_M`` behind to
    ``STOP_LINE_SEARCH_AHEAD_M`` ahead of the human's front.
    """
    try:
        frame = inp.load_frame(frame_idx)
    except KeyError:  # frame not available
        return None
    lines, front = frame.get("stop_lines"), _front_offset_m(frame)
    if lines is None or front is None:
        return None
    human_front_s = human_s + front
    lines = np.asarray(lines, dtype=np.float64)
    lines = lines[np.abs(lines).sum(axis=(1, 2)) > 0]
    path = _extended_path(inp)
    crossings = []
    for seg in lines:
        arc, lat = project_onto_path(inp.to_world(seg, frame_idx), path)
        if lat[0] * lat[1] > 0 or lat[0] == lat[1]:
            continue  # the segment does not cross the recorded path
        s_cross = arc[0] + (arc[1] - arc[0]) * lat[0] / (lat[0] - lat[1])
        if -STOP_LINE_SEARCH_BEHIND_M <= s_cross - human_front_s <= STOP_LINE_SEARCH_AHEAD_M:
            crossings.append(float(s_cross))
    return (min(crossings), front) if crossings else None


def _lead_vehicle_before_line(
    inp: ClosedLoopScenarioInput, frame_idx: int, human_front_s: float, line_s: float
) -> bool | None:
    """Whether a vehicle stood between the human's front and the stop line at ``frame_idx``.

    Reads the frame's ``neighbor_agents_past[:, -1]`` (ego-centric at that frame, taken to
    the world frame): a vehicle (``agent_label`` index 0; legacy columns 8:11) whose centre
    is within ``QUEUE_CORRIDOR_M`` of the recorded path, with its arc after
    ``human_front_s`` and at most ``line_s + QUEUE_PAST_LINE_M``, and nearly stopped
    (``QUEUE_MAX_SPEED_MPS``). None when the frame has no neighbor data.
    """
    try:
        frame = inp.load_frame(frame_idx)
    except KeyError:  # frame not available
        return None
    if frame.get("neighbor_agents_past") is None:
        return None
    past, types, _ = _agent_attrs(frame)
    w = max(1, round(QUEUE_SPEED_WINDOW_S / REC_DT_S))
    for i in np.flatnonzero((np.abs(past[:, -1]).sum(axis=1) > 0) & (types == 0)):
        arc, lat = project_onto_path(inp.to_world(past[i, -1], frame_idx), _extended_path(inp))
        if (
            abs(lat[0]) > QUEUE_CORRIDOR_M
            or not human_front_s < arc[0] <= line_s + QUEUE_PAST_LINE_M
        ):
            continue
        if past.shape[1] > w and np.abs(past[i, -1 - w]).sum() > 0:
            speed = np.linalg.norm(past[i, -1] - past[i, -1 - w]) / (w * REC_DT_S)
            if speed > QUEUE_MAX_SPEED_MPS:
                continue
        return True
    return False


def queue_values(
    inp: ClosedLoopScenarioInput, frame_idx: int, human_front_s: float, line_s: float
) -> dict[str, float]:
    """``human_front_to_line_m`` (the gap from the human's front to the stop line) and
    ``queued`` (1.0 if a vehicle stood in it, ``_lead_vehicle_before_line``; absent
    without neighbor data).

    A queued human did not stop for the line, and the position-keyed replay pulls the
    lead vehicle on with an ego that creeps, so the line says nothing about the ego.
    """
    values = {"human_front_to_line_m": line_s - human_front_s}
    lead = _lead_vehicle_before_line(inp, frame_idx, human_front_s, line_s)
    if lead is not None:
        values["queued"] = float(lead)
    return values


def _net_speed(xy: np.ndarray, dt: float, window_s: float) -> np.ndarray:
    """Net displacement over a centred ``window_s`` window, per second."""
    n, w = len(xy), max(1, round(window_s / dt))
    a = np.clip(np.arange(n) - w // 2, 0, max(n - 1 - w, 0))
    b = np.minimum(a + w, n - 1)
    span = np.maximum(b - a, 1) * dt
    return np.linalg.norm(xy[b] - xy[a], axis=1) / span


def _ego_stop_speed(inp: ClosedLoopScenarioInput, p: StopParams) -> np.ndarray:
    """Live ego speed for stop detection: net displacement over ``sustained_stop_s``.

    A stopped closed-loop ego jitters by a few centimetres per step, so the rollout's
    per-step ``speed`` (and per-step pose deltas) swing across the stop threshold
    while it stands still and no sustained stop is ever seen. The net displacement over
    a centred window of the stop duration is what "standing still" means here.
    """
    return _net_speed(inp.ego_xy, inp.dt, p.sustained_stop_s)


def _human_hold_creep_m(inp: ClosedLoopScenarioInput) -> float:
    """How far the human advanced along the recorded path while holding.

    Arc from its first still frame at/after the anchor to its last still frame before it
    departs (the first frame after that moving faster than ``HUMAN_DEPART_SPEED_MPS``;
    the window end if none). NaN if the human is never still after the anchor.
    """
    v = _net_speed(inp.rec_xy, REC_DT_S, HUMAN_STILL_WINDOW_S)
    frames = np.arange(inp.n_frames)
    still = np.flatnonzero((v < HUMAN_STILL_SPEED_MPS) & (frames >= inp.anchor_frame))
    if not len(still):
        return np.nan
    departs = np.flatnonzero((v > HUMAN_DEPART_SPEED_MPS) & (frames > still[0]))
    last = still[still < departs[0]][-1] if len(departs) else still[-1]
    s_rec = path_arclength(inp.rec_xy)
    return float(s_rec[last] - s_rec[still[0]])


def _hold_values(inp: ClosedLoopScenarioInput, hold_creep_m: float) -> dict[str, float]:
    """``human_hold_creep_m``, ``max_hold_creep_m`` and, when both creeps are known,
    ``hold_passed`` (1.0 if the ego's ``hold_creep_m`` is within the maximum)."""
    human = _human_hold_creep_m(inp)
    values = {"human_hold_creep_m": human, "max_hold_creep_m": human + HOLD_CREEP_MARGIN_M}
    if np.isfinite(human) and np.isfinite(hold_creep_m):
        values["hold_passed"] = float(hold_creep_m <= values["max_hold_creep_m"])
    return values


def _wait(inp: ClosedLoopScenarioInput, h0: int, h1: int) -> tuple[float, float, int | None]:
    """``(human's stop duration, time the replay spent inside it, step it left)``.

    The replay cursor follows the ego, so an ego that edges on past the human's stop
    pulls the recording forward to the human's departure (as in ``yield_wait``).
    """
    entered = np.flatnonzero(inp.rec_idx >= h0)
    left = np.flatnonzero(inp.rec_idx >= h1)
    k_in = int(entered[0]) if len(entered) else inp.n_steps
    k_out = int(left[0]) if len(left) else inp.n_steps
    return (h1 - h0) * REC_DT_S, max(0, k_out - k_in) * inp.dt, k_out if len(left) else None


def _extended_path(inp: ClosedLoopScenarioInput) -> np.ndarray:
    yaw = inp.rec_yaw[-1]
    tip = inp.rec_xy[-1] + PATH_EXTENSION_M * np.array([np.cos(yaw), np.sin(yaw)])
    return np.vstack([inp.rec_xy, tip])


def _ol_stop_values(inp: ClosedLoopScenarioInput, p: StopParams) -> dict[str, float]:
    """Open-loop ``stop_overshoot`` of the realized ego vs the human (see module doc)."""
    a, k0 = inp.anchor_frame, inp.anchor_step
    try:
        frame = inp.load_frame(a)
    except KeyError:  # frame not available
        return {}
    lanes = frame.get("route_lanes", frame.get("lanes"))
    if lanes is None:
        return {}
    try:
        segments = _current_route_segments(np.asarray(lanes, dtype=np.float64))
    except ValueError:  # no usable centerline
        return {}
    gt = _to_local(inp, inp.rec_xy[a + 1 : a + 1 + round(OL_STOP_HORIZON_S / REC_DT_S)], a)
    steps = slice(k0 + 1, k0 + 1 + round(OL_STOP_HORIZON_S / inp.dt))
    ego = _to_local(inp, inp.ego_xy[steps], a)
    if not len(gt) or not len(ego):
        return {}
    gt_s, gt_stopped = _stop_position_s(gt, _speed_mps(gt), segments)
    # Same rule as ``_stop_position_s`` (final sustained stop, else terminal s), with the
    # rollout's speed and step length instead of 0.1 s pose deltas.
    ego_pos = _project_along_route(ego, segments)
    runs = _stop_runs(_ego_stop_speed(inp, p)[steps], inp.dt, p)
    ego_s = float(np.median(ego_pos[runs[-1][0] : runs[-1][1]])) if runs else float(ego_pos[-1])
    overshoot = max(0.0, ego_s - gt_s)
    return {
        "ol_gt_stop_s_m": gt_s,
        "ol_ego_stop_s_m": ego_s,
        "ol_stop_overshoot_m": overshoot,
        "ol_gt_sustained_stop": float(gt_stopped),
        "ol_ego_sustained_stop": float(bool(runs)),
        "ol_passed": float(overshoot <= p.tolerance_m),
    }


def red_hold_values(inp: ClosedLoopScenarioInput) -> dict[str, float] | None:
    """Where the ego held while the human waited at the red before ``anchor_frame``.

    For traffic_light_go, whose anchor is the light turning green. The human's wait is
    the last sustained stop of the recording that starts before the anchor; the stop
    line is the one the human stopped for (``_stop_line_arc``), compared with the ego's
    front. Without a stop line the human's stop is the reference, compared with the
    ego's rear axle. Over the sim steps from the replay entering the human's stop to
    ``anchor_step``:

    - ``furthest_past_line_m``: the furthest the ego got, past the reference;
    - ``first_stop_past_line_m``: where its first sustained stop was, past the reference
      (NaN without a stop);
    - ``hold_creep_m``: how far it moved on from that stop (NaN without a stop).

    With a stop line, also ``queue_values`` at the human's stop frame (``queued``,
    ``human_front_to_line_m``).

    None when the human did not stop before the anchor, the replay never reached the
    human's stop, or the anchor was never replayed.
    """
    p = StopParams(tolerance_m=0.0)
    k_anchor = inp.anchor_step
    human = [r for r in _stop_runs(inp.rec_speed, REC_DT_S, p) if r[0] < inp.anchor_frame]
    if k_anchor is None or not human:
        return None
    h0, h1 = human[-1]
    entered = np.flatnonzero(inp.rec_idx >= h0)
    if not len(entered) or entered[0] > k_anchor:
        return None
    k0 = int(entered[0])
    path = _extended_path(inp)
    human_s = float(np.median(project_onto_path(inp.rec_xy[h0:h1], path)[0]))
    line = _stop_line_arc(inp, (h0 + h1) // 2, human_s)
    ref = line[0] - line[1] if line is not None else human_s
    ego_s = np.maximum.accumulate(project_onto_path(inp.ego_xy, path)[0])
    values = {
        "stop_line_found": float(line is not None),
        "furthest_past_line_m": float(ego_s[k0 : k_anchor + 1].max()) - ref,
        "first_stop_past_line_m": np.nan,
        "hold_creep_m": np.nan,
    }
    if line is not None:
        values.update(queue_values(inp, (h0 + h1) // 2, human_s + line[1], line[0]))
    live = [
        r for r in _stop_runs(_ego_stop_speed(inp, p), inp.dt, p) if r[1] > k0 and r[0] <= k_anchor
    ]
    first = _first_stop_steps(live, k0, inp.dt, p)
    if first is not None:
        first_s = float(np.median(ego_s[first]))
        values["first_stop_past_line_m"] = first_s - ref
        values["hold_creep_m"] = _hold_creep_m(ego_s, first, np.arange(k0, k_anchor + 1), first_s)
    return values


def _no_anchor(metric: str, inp: ClosedLoopScenarioInput) -> ScenarioResult:
    return ScenarioResult(
        metric,
        None,
        details={"terminated": inp.terminated},
        reason=f"anchor frame {inp.anchor_frame} never replayed (terminated={inp.terminated})",
    )


def score_stop(inp: ClosedLoopScenarioInput, p: StopParams) -> ScenarioResult:
    """Realized stop position must not pass the human's by more than ``tolerance_m``.

    Human stop: the first sustained stop of the recording still ongoing at/after the
    anchor (the open-loop scorer takes the final one within its 8 s GT future; here the
    window runs on for 20 s, past the human's release and possibly a later stop, so the
    first one is the stop the label is about -- a human creeping up in two stops is
    measured at the first). Its position is the run's median arc length.

    Reference: for ``STOP_LINE_LABELS`` the stop line the human stopped for (from the
    human's stop frame, see ``_stop_line_arc``), compared with the ego's *front*; humans
    stop with their front within about -0.5..+2 m of it, while the ego is driven by its
    rear axle and stops some metres past the human's own position. Without a stop line
    (or for obstacle_stop) the human's stop, compared with the rear axle, as before.
    ``overshoot_m`` / ``undershoot_m`` are against the reference; ``past_human_stop_m``
    is always against the human's stop.

    Live stop: judged while the replay is still inside the human's stop (sim steps from
    ``anchor_step`` whose replayed frame is before the human's departure). The ego must
    make a sustained stop then, and the furthest it gets meanwhile must not pass
    ``human + tolerance``: creeping on after stopping counts, and a brief stop short of
    the line followed by rolling through fails. Judging by where the ego stopped *first*
    would pass both. Stopping short passes and reports the undershoot; driving on after
    the human's release is not penalized. Without a stop in that interval the ego fails
    once it passed the limit, with the overshoot measured at its first stop after it,
    or its furthest point if it never stops.

    temporal_stop asks only for a full stop at the line before moving on: an ego stop
    counts only once it lasts ``TEMPORAL_STOP_MIN_HOLD_S`` (a shorter one is a rolling
    stop), and the furthest point is taken up to the end of the ego's first such stop
    instead of the human's departure. Moving on after the ego left that stop is its
    departure, not overshoot, even while the human still holds (the human holds about a
    second, and the replay fast-forwards once the ego is ahead).

    traffic_light_stop is judged by the reaction alone: the ego's first stop at/after the
    anchor must not pass ``reference + tolerance`` (``first_stop_overshoot_m <=
    tolerance_m``). Only a stop for the line counts: one lasting ``required_stop_s`` (``min(
    TRAFFIC_LIGHT_MIN_STOP_S, human_wait_s)``, as a sustained-stop run) with the ego no more
    than ``STOP_FOR_LINE_MAX_SHORT_M`` short of the reference; a brief stop, or one behind
    a lead vehicle, is skipped. An ego that passes the limit without such a stop fails
    ("passed the stop line without stopping at it"). Creeping on
    while the red lasts is holding, reported as ``hold_passed`` (see the module doc), as
    is the share of the human's wait the replay spent inside it (``wait_ratio``; an ego
    that edges on pulls the replay to the green).

    traffic_light_stop is not scored either when the human was not at the head of the
    queue (``queue_values``: a vehicle stood between its front and the stop line); the
    values are kept, with ``queued`` and ``human_front_to_line_m``.

    Not scored: the human never stops after the anchor; or the trace ends (goal,
    max_steps, abort) before the ego either stops or passes the limit. The latter is
    the common case when the recording ends with the human still stopped: the goal is
    then the stop point and the rollout ends ``GOAL_REACH_M`` short of it, still moving.

    Reported only, splitting the judged position into reacting and holding:

    - ``first_stop_overshoot_m``: the ego's first stop at/after the anchor (median arc
      over its first ``sustained_stop_s``; for temporal_stop a held stop, for
      traffic_light_stop a stop for the line, see above) minus the reference, signed
      (negative short of it); NaN if the ego never stops. traffic_light_stop also reports
      that stop's ``first_stop_duration_s`` and the ``required_stop_s``.
      Reacting to the scene: comparable with open loop.
    - ``hold_creep_m``: the furthest arc from that stop on over the steps the verdict
      judges (before the human's departure; for temporal_stop up to the end of the
      ego's held stop) minus the first stop; NaN without a stop or such a step.
      Holding: closed loop only, and dominated by the model's creep at standstill,
      which the position-keyed replay also turns into a skipped wait.

    For traffic_light_stop ``overshoot_m`` is ``max(0, first_stop_overshoot_m)``. For the
    other labels, when a stop while the human holds decides the verdict, it is
    ``max(0, first_stop_overshoot_m + hold_creep_m)``.
    """
    metric = "stop_overshoot"
    if inp.anchor_step is None:
        return _no_anchor(metric, inp)
    s_rec = path_arclength(inp.rec_xy)
    s_ego, _ = project_onto_path(inp.ego_xy, _extended_path(inp))
    a, k0 = inp.anchor_frame, inp.anchor_step
    red = np.asarray(inp.red_light_violation[k0:], dtype=bool)
    details: dict = {"terminated": inp.terminated}
    values: dict[str, float] = {"tolerance_m": p.tolerance_m, **_ol_stop_values(inp, p)}
    if inp.label == "traffic_light_stop":
        # Reported only; red light is its own generic metric.
        values["red_light_violation_steps"] = float(red.sum())
        details["first_red_light_violation_step"] = int(k0 + np.argmax(red)) if red.any() else None

    human = [r for r in _stop_runs(inp.rec_speed, REC_DT_S, p) if r[1] > a]
    if not human:
        return ScenarioResult(
            metric, None, values, details, reason="human never stops after the anchor"
        )
    h0, h1 = human[0]
    human_s = float(np.median(s_rec[h0:h1]))
    values["human_stop_s_m"] = human_s
    details["human_stop_frames"] = [h0, h1]
    # Reference stop position (rear-axle arc): the stop line minus the front offset when
    # the label stops at a line and the human's stop frame has it, else the human's stop.
    line = (
        _stop_line_arc(inp, (h0 + h1 - 1) // 2, human_s) if inp.label in STOP_LINE_LABELS else None
    )
    if line is not None:
        line_s, front = line
        ref_s = line_s - front
        values["stop_line_s_m"] = line_s
        values["human_front_past_line_m"] = human_s + front - line_s
        details["stop_reference"] = "stop_line"
        if inp.label == "traffic_light_stop":
            values.update(queue_values(inp, (h0 + h1 - 1) // 2, human_s + front, line_s))
    else:
        ref_s = human_s
        details["stop_reference"] = "human_stop"
    limit = ref_s + p.tolerance_m

    live = [r for r in _stop_runs(_ego_stop_speed(inp, p), inp.dt, p) if r[1] > k0]
    if inp.label == "temporal_stop":
        min_hold = max(1, round(TEMPORAL_STOP_MIN_HOLD_S / inp.dt))
        live = [r for r in live if r[1] - r[0] >= min_hold]
    if inp.label == "traffic_light_stop":
        human_wait_s, waited_s, k_out = _wait(inp, h0, h1)
        required_s = min(TRAFFIC_LIGHT_MIN_STOP_S, human_wait_s)
        values.update(
            {
                "human_wait_s": human_wait_s,
                "waited_s": waited_s,
                "wait_ratio": waited_s / human_wait_s,
                "required_stop_s": required_s,
            }
        )
        details["step_left_wait"] = k_out
        # Stops for the line only (see the docstring); 1e-6 absorbs float steps.
        live = [
            r
            for r in live
            if (r[1] - r[0]) * inp.dt >= required_s - 1e-6
            and float(np.median(s_ego[_stop_steps(r, k0, inp.dt, p)])) - ref_s
            >= -STOP_FOR_LINE_MAX_SHORT_M
        ]
    beyond = np.flatnonzero(s_ego[k0:] > limit)
    k_cross = k0 + int(beyond[0]) if len(beyond) else None
    values["max_s_after_anchor_m"] = float(s_ego[k0:].max())
    # Steps whose replayed frame is before the human's departure: the light is still red.
    holding = np.flatnonzero((np.arange(inp.n_steps) >= k0) & (inp.rec_idx < h1))
    held = [r for r in live if len(holding) and r[0] <= holding[-1]]
    # Steps the furthest point is taken over: for temporal_stop up to the end of the
    # ego's first held stop, as what follows is its departure.
    judged = holding[holding < held[0][1]] if held and inp.label == "temporal_stop" else holding
    first = _first_stop_steps(live, k0, inp.dt, p)
    if first is None:
        values["first_stop_overshoot_m"] = values["hold_creep_m"] = np.nan
    else:
        first_s = float(np.median(s_ego[first]))
        values["first_stop_overshoot_m"] = first_s - ref_s
        values["hold_creep_m"] = _hold_creep_m(s_ego, first, judged, first_s)
    reason = ""
    if inp.label == "traffic_light_stop":
        values["first_stop_duration_s"] = (live[0][1] - live[0][0]) * inp.dt if live else np.nan
        values.update(_hold_values(inp, values["hold_creep_m"]))
    if inp.label == "traffic_light_stop" and first is not None:
        r0, r1 = live[0]
        stop_s, stopped = float(np.median(s_ego[first])), True
        passed = values["first_stop_overshoot_m"] <= p.tolerance_m
        reason = "" if passed else "ego's first stop was past the stop limit"
    elif inp.label == "traffic_light_stop" and k_cross is not None:
        r0 = r1 = None
        stop_s, stopped = values["max_s_after_anchor_m"], False
        passed, reason = False, "ego passed the stop line without stopping at it"
    elif held and inp.label != "traffic_light_stop":
        r0, r1 = held[0]
        stop_s = float(s_ego[judged].max())
        passed, stopped = stop_s <= limit, True
    elif k_cross is not None:
        after = [r for r in live if r[0] >= k_cross]
        if after:
            r0, r1 = after[0]
            stop_s, stopped = float(np.median(s_ego[r0:r1])), True
        else:
            r0 = r1 = None
            stop_s, stopped = values["max_s_after_anchor_m"], False
        passed = False
    else:
        values["final_s_m"] = float(s_ego[-1])
        values["final_speed_mps"] = float(inp.ego_speed[-1])
        values["shortfall_to_human_stop_m"] = human_s - float(s_ego[-1])
        return _unless_queued(
            ScenarioResult(
                metric,
                None,
                values,
                details,
                reason=f"trace ended ({inp.terminated}) before the ego stopped or passed the human stop",
            )
        )
    values.update(
        {
            "ego_stop_s_m": stop_s,
            # Past the reference (the stop line, by the ego's front, or the human's stop).
            "overshoot_m": max(0.0, stop_s - ref_s),
            "undershoot_m": max(0.0, ref_s - stop_s),
            "past_human_stop_m": stop_s - human_s,
            "ego_sustained_stop": float(stopped),
        }
    )
    details["ego_stop_steps"] = None if r0 is None else [r0, r1]
    details["first_step_past_limit"] = k_cross
    return _unless_queued(ScenarioResult(metric, passed, values, details, reason))


def _unless_queued(r: ScenarioResult) -> ScenarioResult:
    """Not applicable (all values kept) when the human was queued (``queue_values``)."""
    if r.values.get("queued") == 1.0:
        return ScenarioResult(r.metric, None, r.values, r.details, QUEUED_REASON)
    return r


def _score_arrival_stop(
    inp: ClosedLoopScenarioInput, p: ArrivalParams, human: tuple[int, int]
) -> ScenarioResult:
    """The ego's first stop near the human's arrival stop must be within
    ``position_tolerance_m`` of it, heading within ``heading_tolerance_deg``.

    Used when the window runs on past the human's stop (``extend_until_departure``), so
    the goal radius no longer hides it. The arrival point is the middle of the human's
    first sustained stop at/after the anchor; the ego's stop is its first sustained stop
    after ``anchor_step`` within ``ARRIVAL_STOP_SEARCH_M`` of it (earlier stops further
    away are queueing), at the median pose over its first ``sustained_stop_s``
    (``first_stop_distance_m``, ``stop_distance_m``; heading at the middle of it).
    Without such a stop the ego's first stop at all is reported and fails ("stopped away
    from the arrival point"); driving past without a stop fails.

    The verdict is the reaction: where the ego first stopped, comparable with open loop.
    Staying is holding, closed loop only and dominated by the model's creep, reported
    only: ``hold_creep_m`` (the furthest arc from the first stop on while the replay is
    before the human's departure, minus the first stop; NaN without a stop) against the
    human's own creep as ``hold_passed`` (see the module doc), and the share of the
    human's dwell the replay spent inside it (``wait_ratio``; as in ``yield_wait``, an ego
    that edges on pulls the recording forward to the human's departure). The open-loop
    reference values compare the final pose with the window's end and say nothing here,
    so they are left out.
    """
    metric = "arrival"
    k0 = inp.anchor_step
    h0, h1 = human
    i_arr = (h0 + h1 - 1) // 2
    s_rec = path_arclength(inp.rec_xy)
    s_ego, lat = project_onto_path(inp.ego_xy, _extended_path(inp))
    s_arr = float(s_rec[i_arr])
    values: dict[str, float] = {
        "arrival_point_s_m": s_arr,
        "position_tolerance_m": p.position_tolerance_m,
        "heading_tolerance_deg": p.heading_tolerance_deg,
        "max_s_after_anchor_m": float(s_ego[k0:].max()),
    }
    details: dict = {
        "terminated": inp.terminated,
        "arrival_mode": "stop_at_arrival_point",
        "human_stop_frames": [h0, h1],
    }
    live = [
        r for r in _stop_runs(_ego_stop_speed(inp, _STOP_RULE), inp.dt, _STOP_RULE) if r[1] > k0
    ]

    def distance(r: tuple[int, int]) -> float:
        xy = np.median(inp.ego_xy[_stop_steps(r, k0, inp.dt, _STOP_RULE)], axis=0)
        return float(np.linalg.norm(xy - inp.rec_xy[i_arr]))

    near_stops = [r for r in live if distance(r) <= ARRIVAL_STOP_SEARCH_M]
    live = near_stops or live
    first = _first_stop_steps(live, k0, inp.dt, _STOP_RULE)
    if first is None:
        values["first_stop_distance_m"] = values["hold_creep_m"] = np.nan
    else:
        dwelling = np.flatnonzero((np.arange(inp.n_steps) >= k0) & (inp.rec_idx < h1))
        first_xy = np.median(inp.ego_xy[first], axis=0)
        first_s = float(np.median(s_ego[first]))
        values["first_stop_distance_m"] = float(np.linalg.norm(first_xy - inp.rec_xy[i_arr]))
        values["hold_creep_m"] = _hold_creep_m(s_ego, first, dwelling, first_s)
    dwell_s, waited_s, k_out = _wait(inp, h0, h1)
    values.update(
        {"human_dwell_s": dwell_s, "waited_s": waited_s, "wait_ratio": waited_s / dwell_s}
    )
    details["step_left_dwell"] = k_out
    values.update(_hold_values(inp, values["hold_creep_m"]))
    if first is None:
        if (
            inp.terminated != "goal"
            and values["max_s_after_anchor_m"] < s_arr - p.position_tolerance_m
        ):
            return ScenarioResult(
                metric,
                None,
                values,
                details,
                reason=f"trace ended ({inp.terminated}) before the ego reached the arrival point",
            )
        return ScenarioResult(
            metric, False, values, details, reason="ego never stopped after the anchor"
        )
    k = int(first[len(first) // 2])
    heading_err = float(np.degrees(abs(wrap_angle(inp.ego_yaw[k] - inp.rec_yaw[i_arr]))))
    values.update(
        {
            "ego_stop_s_m": first_s,
            "stop_distance_m": values["first_stop_distance_m"],
            "longitudinal_offset_m": first_s - s_arr,
            "lateral_offset_m": float(np.median(lat[first])),
            "heading_error_deg": heading_err,
        }
    )
    details["ego_stop_steps"] = list(live[0])
    near = values["stop_distance_m"] <= p.position_tolerance_m
    heading_ok = heading_err <= p.heading_tolerance_deg
    details["position_within_tolerance"] = near
    details["heading_within_tolerance"] = heading_ok
    if not near:
        reason = "ego stopped away from the arrival point"
    elif not heading_ok:
        reason = "heading off at the arrival point"
    else:
        reason = ""
    return ScenarioResult(metric, near and heading_ok, values, details, reason)


def score_arrival(inp: ClosedLoopScenarioInput, p: ArrivalParams) -> ScenarioResult:
    """Closest approach to the recorded endpoint, and pose there, within tolerance.

    When the window runs on more than ``GOAL_REACH_M`` past the human's first sustained
    stop at/after the anchor (a window extended until the bus moves on), the arrival is
    that stop instead and ``_score_arrival_stop`` decides. Otherwise:

    Open loop compares the final predicted pose to the GT endpoint. Here the rollout
    stops the ego ``GOAL_REACH_M`` (> the 2 m tolerance) from the endpoint, so the
    final pose is short by construction and its distance says nothing. The pose is
    instead split at the ego's closest approach (after the anchor): the ego passes when
    it got within ``reach_m`` of the endpoint, laterally within ``position_tolerance_m``
    of the recorded path there, and heading within ``heading_tolerance_deg`` of the
    recorded heading at the same arc length. The last ``reach_m`` of longitudinal
    approach is unobservable; with ``GOAL_REACH_M <= position_tolerance_m`` this
    reduces to the open-loop endpoint check.
    """
    metric = "arrival"
    if inp.anchor_step is None:
        return _no_anchor(metric, inp)
    human = [r for r in _stop_runs(inp.rec_speed, REC_DT_S, _STOP_RULE) if r[1] > inp.anchor_frame]
    if human:
        s_rec = path_arclength(inp.rec_xy)
        i_arr = (human[0][0] + human[0][1] - 1) // 2
        if s_rec[-1] - s_rec[i_arr] > GOAL_REACH_M:
            return _score_arrival_stop(inp, p, human[0])
    k0 = inp.anchor_step
    end_xy, end_yaw = inp.rec_xy[-1], inp.rec_yaw[-1]
    dist = np.linalg.norm(inp.ego_xy[k0:] - end_xy, axis=1)
    k = k0 + int(np.argmin(dist))
    s_rec = path_arclength(inp.rec_xy)
    s_ego, lat = project_onto_path(inp.ego_xy[k], _extended_path(inp))
    # Recorded heading where the ego is along the path (first frame at that arc length).
    i_ref = min(int(np.searchsorted(s_rec, s_ego[0])), inp.n_frames - 1)
    heading_err = float(np.degrees(abs(wrap_angle(inp.ego_yaw[k] - inp.rec_yaw[i_ref]))))
    values = {
        "closest_distance_m": float(dist.min()),
        "final_distance_m": float(dist[-1]),
        "longitudinal_shortfall_m": float(s_rec[-1] - s_ego[0]),
        "lateral_offset_m": float(lat[0]),
        "heading_error_deg": heading_err,
        "endpoint_heading_error_deg": float(np.degrees(abs(wrap_angle(inp.ego_yaw[k] - end_yaw)))),
        "reach_m": p.reach_m,
        "position_tolerance_m": p.position_tolerance_m,
        "heading_tolerance_deg": p.heading_tolerance_deg,
    }
    ol_heading_err = float(np.degrees(abs(wrap_angle(inp.ego_yaw[-1] - end_yaw))))
    values.update(
        {
            "ol_final_displacement_error_m": values["final_distance_m"],
            "ol_final_heading_error_deg": ol_heading_err,
            "ol_passed": float(
                values["final_distance_m"] <= p.position_tolerance_m
                and ol_heading_err <= p.heading_tolerance_deg
            ),
        }
    )
    reached = values["closest_distance_m"] <= p.reach_m
    lateral_ok = abs(values["lateral_offset_m"]) <= p.position_tolerance_m
    heading_ok = heading_err <= p.heading_tolerance_deg
    details = {
        "terminated": inp.terminated,
        "closest_step": k,
        "reference_frame": i_ref,
        "reached": reached,
        "lateral_within_tolerance": lateral_ok,
        "heading_within_tolerance": heading_ok,
    }
    return ScenarioResult(metric, reached and lateral_ok and heading_ok, values, details)


@register("traffic_light_stop", "obstacle_stop", "temporal_stop")
def _stop(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    return score_stop(inp, StopParams.from_config(inp.label, config))


@register("arrival")
def _arrival(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    return score_arrival(inp, ArrivalParams.from_config(config))
