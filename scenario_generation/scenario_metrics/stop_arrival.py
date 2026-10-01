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
# tolerance. The same 0.5 m as the open-loop red-light and obstacle stops.
TEMPORAL_STOP_TOLERANCE_M = 0.5
# Length of the open-loop GT future / prediction ``stop_overshoot`` reads; the interval
# of the ``ol_*`` stop values.
OL_STOP_HORIZON_S = 8.0


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


def _stop_runs(speed: np.ndarray, dt: float, p: StopParams) -> list[tuple[int, int]]:
    """``[start, end)`` runs with speed <= threshold lasting >= the sustained duration."""
    width = max(1, round(p.sustained_stop_s / dt))
    stopped = np.r_[False, np.asarray(speed) <= p.stop_speed_mps, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(stopped))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= width]


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
    runs = _stop_runs(inp.ego_speed[steps], inp.dt, p)
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

    Live stop: the ego's last sustained stop (ongoing at/after ``anchor_step``) before
    it first passes ``human + tolerance``. Stopping short passes and reports the
    undershoot; stopping and later driving on (after the release) is not penalized.
    Passing the limit without a stop before it fails, with the overshoot measured at
    the ego's first stop after it, or its furthest point if it never stops.

    Not scored: the human never stops after the anchor; or the trace ends (goal,
    max_steps, abort) before the ego either stops or passes the limit. The latter is
    the common case when the recording ends with the human still stopped: the goal is
    then the stop point and the rollout ends ``GOAL_REACH_M`` short of it, still moving.
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
    limit = human_s + p.tolerance_m
    values["human_stop_s_m"] = human_s
    details["human_stop_frames"] = [h0, h1]

    live = [r for r in _stop_runs(inp.ego_speed, inp.dt, p) if r[1] > k0]
    beyond = np.flatnonzero(s_ego[k0:] > limit)
    k_cross = k0 + int(beyond[0]) if len(beyond) else None
    before = [r for r in live if k_cross is None or r[0] < k_cross]
    values["max_s_after_anchor_m"] = float(s_ego[k0:].max())
    if before:
        r0, r1 = before[-1]
        stop_s = float(np.median(s_ego[r0:r1]))
        passed, stopped = True, True
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
        return ScenarioResult(
            metric,
            None,
            values,
            details,
            reason=f"trace ended ({inp.terminated}) before the ego stopped or passed the human stop",
        )
    values.update(
        {
            "ego_stop_s_m": stop_s,
            "overshoot_m": max(0.0, stop_s - human_s),
            "undershoot_m": max(0.0, human_s - stop_s),
            "ego_sustained_stop": float(stopped),
        }
    )
    details["ego_stop_steps"] = None if r0 is None else [r0, r1]
    details["first_step_past_limit"] = k_cross
    return ScenarioResult(metric, passed, values, details)


def score_arrival(inp: ClosedLoopScenarioInput, p: ArrivalParams) -> ScenarioResult:
    """Closest approach to the recorded endpoint, and pose there, within tolerance.

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
