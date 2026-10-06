"""Closed-loop scenario metrics: progress family.

Owns labels: departure, traffic_light_go, pedestrian_yield, vehicle_yield.

``pedestrian_yield`` / ``vehicle_yield`` anchors that carry their event span are scored
by ``yield_wait`` (closed loop only, see its docstring); the fixed-horizon
``yield_progress`` below scores the rest and is reported alongside.

The open-loop scorers (``planner_metrics/departure.py``, ``planner_metrics/yield_progress.py``)
read one predicted trajectory; here the same thresholds (read from
``ScenarioOpenLoopConfig``, see ``shared_config``) are applied to the ego's *realized*
trajectory from ``anchor_step`` on. Progress is the arc length gained along
the recorded path (``project_onto_path``), not Euclidean displacement, so a swerve or a
sideways drift at standstill does not count as progress. A path that passes near itself
(a loop) could snap a point onto the wrong pass; windows are 30 s of forward driving, so
that is ignored.

The horizon is ``round(horizon_s / dt)`` sim steps after ``anchor_step``, mirroring the
open-loop prediction whose first point is 0.1 s after the current pose. When the trace
ends before that:

- a verdict the available steps already decide stands (maximum progress only grows);
- ``terminated == "goal"`` (ego within the goal radius of the window's last recorded
  pose) passes departure -- progress was made. For yield it fails only if the ego got
  past the human's waiting point (its pose at the anchor frame, plus the tolerance);
  otherwise it is not applicable: when the human holds until the window's end, the
  goal *is* the waiting point and the goal radius ends the rollout before the ego
  shows whether it would have held;
- otherwise the available steps are scored and ``horizon_truncated`` is recorded (no
  step after the anchor at all -> not applicable).

traffic_light_go must also have held behind the stop line while the human waited at the
red: it fails, whatever the progress afterwards, when its front got past the stop line
(``stop_arrival.red_hold_values``: the furthest point from the replay entering the
human's stop to the anchor). The replay is position-keyed, so an ego creeping on the red
pulls the scene to the green, and the rollout's red-light check never sees it red.
When a vehicle stood between the human and that line, the human was queued, not stopped
by the line, and the window is not scored (the replay pulls the lead vehicle on with a
creeping ego). ``replay_ahead_s`` (reported for both labels) is how much earlier than the
human the ego reached the anchor (``anchor_frame * REC_DT_S - anchor_step * dt``;
positive = the replay was fast-forwarded).

Open-loop reference values (``ol_*``, reported only, never part of the verdict): the
open-loop quantity measured on the same realized steps (``anchor_step`` + 1 .. the horizon
or the trace's end), with the ego pose at ``anchor_step`` standing in for the open-loop
current pose. Departure reports ``ol_max_displacement_m`` (max Euclidean distance from
that pose), yield ``ol_max_forward_progress_m`` (max progress along that pose's heading,
the open-loop ego +x); ``ol_passed`` (0/1) is the open-loop rule with the same threshold.
Both read a swerve as movement where arc progress does not. Omitted without a step
after the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario_generation.scenario_metrics import conflict, stop_arrival
from scenario_generation.scenario_metrics.base import (
    ClosedLoopScenarioInput,
    ScenarioResult,
    path_arclength,
    project_onto_path,
)
from scenario_generation.scenario_metrics.registry import register
from scenario_generation.scenario_metrics.shared_config import open_loop_parameters

DEPARTURE_LABELS = ("departure", "traffic_light_go")
# temporal_stop is a stop-line stop, scored by the stop family (``stop_arrival``); open
# loop still scores it as a yield, so ``yield_progress`` keeps reading its config.
YIELD_LABELS = ("pedestrian_yield", "vehicle_yield")
# Yield labels whose event span is the human's wait.
WAIT_SPAN_LABELS = YIELD_LABELS

# Closed loop only: departure's horizon, longer than open loop's 3 s. A closed-loop ego
# starts from its own standstill, and human labels passed every departure, including
# ones that took 3-5 s to gain the 2 m (0.94 agreement at 5 s, against 0.62 at 3 s).
# traffic_light_go keeps the open-loop horizon.
CLOSED_LOOP_DEPARTURE_HORIZON_S = 5.0

# Recorded frames are 10 Hz; span lengths are in recorded frames.
REC_DT_S = 0.1
# Closed loop only: how far past the stop line a traffic_light_go ego's front may get while
# the human waits at the red (the model typically stops 0-2 m short and then creeps
# 1.0-1.6 m on). Humans stop with the front more than 0.5 m past the line in only 2-5% of
# red-light and stop-sign stops, and relabeled windows failed every ego 0.45 m or more past
# it (agreement 0.88, against 0.85 with no tolerance). The same 0.5 m as traffic_light_stop.
TRAFFIC_LIGHT_GO_STOP_LINE_TOLERANCE_M = 0.5
# Closed loop only (no open-loop counterpart): how much further than the human the ego
# may get along the road while the replay is inside the human's wait (``yield_wait``).
# Human labels passed every yield up to 2.98 m past the human's progress; how long the
# ego stayed in the span (``wait_ratio``, reported) disagreed with them.
YIELD_MAX_EXCESS_PROGRESS_M = 3.0
# Closed loop only: the shortest post-encroachment time to the yielded-to agent that
# still passes (``yield_conflict``). Human labels accepted every yield down to 0.5 s; this
# floor is "nearly touching", not a comfort margin.
YIELD_MIN_PET_S = 0.5


@dataclass(frozen=True)
class DepartureParams:
    horizon_s: float
    minimum_progress_m: float

    @classmethod
    def from_config(cls, label: str, config=None) -> DepartureParams:
        """From the open-loop config (``scenario_<label>_*``, see ``shared_config``),
        except departure's horizon (``CLOSED_LOOP_DEPARTURE_HORIZON_S``)."""
        p = open_loop_parameters(label, config)
        return cls(
            horizon_s=(
                CLOSED_LOOP_DEPARTURE_HORIZON_S
                if label == "departure"
                else float(p["horizon_seconds"])
            ),
            # Open loop's displacement threshold, measured here as arc length gained.
            minimum_progress_m=float(p["minimum_displacement_m"]),
        )


@dataclass(frozen=True)
class YieldParams:
    horizon_s: float
    maximum_forward_progress_m: float

    @classmethod
    def from_config(cls, label: str, config=None) -> YieldParams:
        """Both from the open-loop config (``scenario_<label>_*``, see ``shared_config``)."""
        p = open_loop_parameters(label, config)
        return cls(
            horizon_s=float(p["horizon_seconds"]),
            maximum_forward_progress_m=float(p["maximum_forward_progress_m"]),
        )


@dataclass(frozen=True)
class _Progress:
    """Forward progress over the post-anchor horizon (index 0 = ``anchor_step``)."""

    progress: np.ndarray  # (A+1,) arc gained along the recorded path since anchor_step
    horizon_steps: int
    truncated: bool
    reference_m: float  # the human's progress over the same horizon from the anchor frame
    # Open-loop definitions over the same steps (nan without a step after the anchor).
    ol_max_displacement_m: float  # Euclidean, from the anchor_step pose
    ol_max_forward_progress_m: float  # along the anchor_step heading

    @property
    def available_steps(self) -> int:
        return len(self.progress) - 1

    @property
    def max_m(self) -> float:
        return float(self.progress[1:].max()) if self.available_steps else 0.0


def _progress(inp: ClosedLoopScenarioInput, horizon_s: float) -> _Progress:
    assert inp.anchor_step is not None
    h = int(round(horizon_s / inp.dt))
    k0 = inp.anchor_step
    k1 = min(k0 + h, inp.n_steps - 1)
    arc, _ = project_onto_path(inp.ego_xy[k0 : k1 + 1], inp.rec_xy)
    rec_arc = path_arclength(inp.rec_xy)
    rel = inp.ego_xy[k0 + 1 : k1 + 1] - inp.ego_xy[k0]
    forward = rel @ np.array([np.cos(inp.ego_yaw[k0]), np.sin(inp.ego_yaw[k0])])
    # The human's reference uses recorded frames (0.1 s each), which match sim steps only at dt=0.1.
    ref_end = min(inp.anchor_frame + int(round(horizon_s / 0.1)), inp.n_frames - 1)
    return _Progress(
        progress=arc - arc[0],
        horizon_steps=h,
        truncated=k0 + h > inp.n_steps - 1,
        reference_m=float(rec_arc[ref_end] - rec_arc[inp.anchor_frame]),
        ol_max_displacement_m=float(np.linalg.norm(rel, axis=1).max()) if len(rel) else np.nan,
        ol_max_forward_progress_m=float(forward.max()) if len(rel) else np.nan,
    )


def _first_time_s(mask: np.ndarray, dt: float) -> float | None:
    hit = np.flatnonzero(mask)
    return float(hit[0] * dt) if len(hit) else None


def _details(inp: ClosedLoopScenarioInput, p: _Progress) -> dict:
    return {
        "anchor_step": inp.anchor_step,
        "horizon_steps": p.horizon_steps,
        "available_steps": p.available_steps,
        "horizon_truncated": p.truncated,
        "terminated": inp.terminated,
    }


def _yield_goal_verdict(
    inp: ClosedLoopScenarioInput, tol: float, values: dict[str, float], details: dict
) -> ScenarioResult:
    """Yield verdict for a rollout the goal radius ended before the horizon did.

    Fails only if the ego got past the human's waiting point: its furthest arc along the
    recorded path reached the human's anchor-frame arc plus ``tol``. Short of that the
    rollout ended too early to tell, so the verdict is None rather than a pass.
    """
    ego_max_arc = float(project_onto_path(inp.ego_xy, inp.rec_xy)[0].max())
    human_anchor_arc = float(path_arclength(inp.rec_xy)[inp.anchor_frame])
    values = {**values, "ego_max_arc_m": ego_max_arc, "human_anchor_arc_m": human_anchor_arc}
    if ego_max_arc >= human_anchor_arc + tol:
        return ScenarioResult(
            "yield_progress",
            False,
            values,
            details,
            reason="goal reached before the horizon; ego got past the human's waiting point",
        )
    return ScenarioResult(
        "yield_progress",
        None,
        values,
        details,
        reason="goal reached before the horizon; ego stopped short of the human's waiting point (rollout goal radius)",
    )


@register(*DEPARTURE_LABELS)
def departure_progress(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """The ego must gain ``minimum_progress_m`` along the recorded path within the horizon.

    traffic_light_go also fails, before the progress check, when the ego's front got past
    the stop line while the human waited at the red (``red_hold_values``), and is not
    applicable (values kept) when the human was not at the head of the queue there
    (``queued``, see ``stop_arrival.queue_values``).
    Anchor never reached -> not applicable, even on goal: the ego never saw the scene it
    should depart from, so reaching the end says nothing about departing from it.
    """
    params = DepartureParams.from_config(inp.label, config)
    metric = "departure_progress"
    if inp.anchor_step is None:
        return ScenarioResult(
            metric,
            None,
            reason="anchor frame never replayed",
            details={"terminated": inp.terminated},
        )
    p = _progress(inp, params.horizon_s)
    departed = p.max_m >= params.minimum_progress_m
    details = _details(inp, p)
    details["time_to_threshold_s"] = _first_time_s(p.progress >= params.minimum_progress_m, inp.dt)
    values = {
        "progress_m": p.max_m,
        "threshold_m": params.minimum_progress_m,
        "horizon_s": params.horizon_s,
        "reference_progress_m": p.reference_m,
        # Rounded so that whole-frame differences compare exactly against the tolerance.
        "replay_ahead_s": round(inp.anchor_frame * REC_DT_S - inp.anchor_step * inp.dt, 6),
    }
    red = stop_arrival.red_hold_values(inp) if inp.label == "traffic_light_go" else None
    if red is not None:
        values.update(red)
        values["stop_line_tolerance_m"] = TRAFFIC_LIGHT_GO_STOP_LINE_TOLERANCE_M
    if p.available_steps:
        values["ol_max_displacement_m"] = p.ol_max_displacement_m
        values["ol_passed"] = float(p.ol_max_displacement_m >= params.minimum_progress_m)
    reason = ""
    if red is not None and red.get("queued") == 1.0:
        departed, reason = None, stop_arrival.QUEUED_REASON
    elif red is not None and red["furthest_past_line_m"] > TRAFFIC_LIGHT_GO_STOP_LINE_TOLERANCE_M:
        departed = False
        reason = "ego passed the stop line on red"
    elif not departed and p.truncated and inp.terminated == "goal":
        departed, reason = True, "goal reached before the horizon ended"
    elif not departed and p.truncated:
        reason = f"trace ended ({inp.terminated}) before the horizon; scored on {p.available_steps} steps"
        if not p.available_steps:
            departed = None
    return ScenarioResult(
        metric,
        departed,
        values=values,
        details=details,
        reason=reason,
    )


def yield_progress(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """The ego must not gain more than ``maximum_forward_progress_m`` within the horizon.

    A goal termination before the verdict is decided (also before the anchor was
    replayed) is judged by ``_yield_goal_verdict``.
    """
    params = YieldParams.from_config(inp.label, config)
    metric = "yield_progress"
    tol = params.maximum_forward_progress_m
    base_values = {"threshold_m": tol, "horizon_s": params.horizon_s}
    if inp.anchor_step is None:
        if inp.terminated == "goal":
            return _yield_goal_verdict(
                inp, tol, base_values, {"anchor_step": None, "terminated": inp.terminated}
            )
        return ScenarioResult(
            metric,
            None,
            reason="anchor frame never replayed",
            details={"terminated": inp.terminated},
        )
    p = _progress(inp, params.horizon_s)
    yielded = p.max_m <= tol
    details = _details(inp, p)
    details["time_exceeded_s"] = _first_time_s(p.progress > tol, inp.dt)
    values = {"progress_m": p.max_m, **base_values, "reference_progress_m": p.reference_m}
    if p.available_steps:
        values["ol_max_forward_progress_m"] = p.ol_max_forward_progress_m
        values["ol_passed"] = float(p.ol_max_forward_progress_m <= tol)
    reason = ""
    if yielded and p.truncated and inp.terminated == "goal":
        return _yield_goal_verdict(inp, tol, values, details)
    if yielded and p.truncated:
        reason = f"trace ended ({inp.terminated}) before the horizon; scored on {p.available_steps} steps"
        if not p.available_steps:
            yielded = None
    return ScenarioResult(
        metric,
        yielded,
        values=values,
        details=details,
        reason=reason,
    )


def yield_wait(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """While the replay is inside the human's wait, the ego must not get more than
    ``YIELD_MAX_EXCESS_PROGRESS_M`` further along the road than the human did.

    The human's wait is the anchor's event span (``inp.span_frames``). The scored steps
    run from the first step that replays a span frame to the first step past the span;
    the ego's progress over them (arc along the recorded path) is compared with the
    human's over the span. Pushing on into the yielded-to agent's path reads as a large
    excess.

    The replay cursor follows the live ego's position, so an ego that creeps on pulls
    the recorded scene forward with it and leaves the span early. How long it stayed is
    reported as ``wait_ratio`` (sim time inside the span over its recorded duration) but
    is not part of the verdict: human labels accepted yields that left the span well
    before the human's wait ended, as long as the ego did not push on.

    The fixed-horizon ``yield_progress`` values (``progress_m``, ``reference_progress_m``,
    ...) are kept for reference; ``fixed_horizon_passed`` is its verdict as 0/1.

    Not scored: the replay never enters the span; or the trace ends inside it before
    the ego has waited long enough to pass.
    """
    assert inp.span_frames is not None
    metric = "yield_wait"
    first, last = inp.span_frames
    span_s = (last + 1 - first) * REC_DT_S
    values: dict[str, float] = {
        "human_wait_s": span_s,
        "max_excess_progress_m": YIELD_MAX_EXCESS_PROGRESS_M,
    }
    details: dict = {"span_frames": [first, last], "terminated": inp.terminated}
    if inp.anchor_step is not None:
        fixed = yield_progress(inp, config)
        values.update(fixed.values)
        if fixed.passed is not None:
            values["fixed_horizon_passed"] = float(fixed.passed)
    entered = np.flatnonzero(inp.rec_idx >= first)
    if not len(entered):
        return ScenarioResult(metric, None, values, details, reason="span never replayed")
    k_in = int(entered[0])
    left = np.flatnonzero(inp.rec_idx > last)
    k_out = int(left[0]) if len(left) else inp.n_steps
    waited_s = (k_out - k_in) * inp.dt
    in_span = inp.rec_idx[k_in:k_out]
    rec_arc = path_arclength(inp.rec_xy)
    ego_arc, _ = project_onto_path(inp.ego_xy[k_in : max(k_out, k_in + 1)], inp.rec_xy)
    values.update(
        {
            "wait_ratio": waited_s / span_s,
            "waited_s": waited_s,
            "progress_in_span_m": float(ego_arc.max() - ego_arc[0]),
            "human_progress_in_span_m": float(rec_arc[last] - rec_arc[first]),
        }
    )
    values["excess_progress_m"] = values["progress_in_span_m"] - values["human_progress_in_span_m"]
    details.update(
        {
            "step_entered_span": k_in,
            "step_left_span": k_out if len(left) else None,
            # Largest cursor jump inside the span: the replay skipping ahead with the ego.
            "max_replay_jump_frames": int(np.diff(in_span).max()) if len(in_span) > 1 else 0,
        }
    )
    held = values["excess_progress_m"] <= YIELD_MAX_EXCESS_PROGRESS_M
    if not len(left) and held:
        return ScenarioResult(
            metric,
            None,
            values,
            details,
            reason=f"trace ended ({inp.terminated}) inside the span",
        )
    reason = "" if held else "ego pushed on past the human while yielding"
    return ScenarioResult(metric, held, values, details, reason)


def yield_conflict(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """``yield_wait``, plus the agent yielded to (``conflict``): no such agent -> not
    applicable; the ego going first or a post-encroachment time under
    ``YIELD_MIN_PET_S`` -> fail.

    The target is found on the recording: an agent crossing or merging into the
    recorded ego path near the anchor, active in the human's wait (see ``conflict``).
    Without one the anchor is likely not a yield -- human labels marked 19 of 22 such
    pedestrian windows as invalid or unclear -- so it is not scored.

    The closed-loop PET is on the sim clock: the agent is in the conflict while the
    replayed frame (``rec_idx``) is inside its recorded conflict interval, and the
    live ego's front/rear arcs are measured on the recorded path. An ego that edges
    on pulls the replay forward, so the agent clears just ahead of it: a short PET.
    The human's PET on the recording is reported alongside. An ego that never reaches
    the conflict point keeps the ``yield_wait`` verdict.

    A collision while the agent is around the conflict is reported
    (``collision_near_target``) but not judged: the rollout's collision flag does not
    say which agent was hit, and collisions are other metrics' concern.
    """
    metric = "yield_conflict"
    wait = yield_wait(inp, config)
    values = dict(wait.values)
    details = dict(wait.details)
    if "wait_ratio" not in values:  # the span was never replayed
        return ScenarioResult(metric, wait.passed, values, details, wait.reason)
    values["min_pet_s"] = YIELD_MIN_PET_S
    found = conflict.find_target(inp)
    if found is None:
        return ScenarioResult(
            metric,
            None,
            values,
            details,
            "no agent crosses or merges into the ego's path near the anchor (likely not a yield)",
        )
    target, human_pet, n_conflicts, (front, rear, path, anchor_front) = found
    slack = conflict.SPAN_SLACK_FRAMES
    ego_s = np.maximum.accumulate(project_onto_path(inp.ego_xy, path)[0])
    k_in = np.flatnonzero(inp.rec_idx >= target.first_frame)
    k_out = np.flatnonzero(inp.rec_idx > target.last_frame)
    cl = conflict.pet(
        ego_s + front,
        ego_s - rear,
        int(k_in[0]) if len(k_in) else None,
        int(k_out[0]) if len(k_out) else None,
        target,
        inp.dt,
    )
    near = (inp.rec_idx >= target.first_frame - slack) & (inp.rec_idx <= target.last_frame + slack)
    collided = bool(inp.collision[near].any())
    values.update(
        {
            "target_dist_m": target.s_m - anchor_front,
            "human_pet_s": np.nan if human_pet.seconds is None else human_pet.seconds,
            "pet_s": np.nan if cl.seconds is None else cl.seconds,
            "collision_near_target": float(collided),
        }
    )
    details.update(
        {
            "target_type": target.agent_type,
            "target_kind": target.kind,
            "target_frames": [target.first_frame, target.last_frame],
            "human_order": human_pet.order,
            "order": cl.order,
            "n_conflicts": n_conflicts,
        }
    )
    if cl.order == "ego_first":
        return ScenarioResult(
            metric, False, values, details, "ego took the conflict point before the agent"
        )
    if cl.seconds is not None and cl.seconds < YIELD_MIN_PET_S:
        return ScenarioResult(
            metric, False, values, details, f"post-encroachment time under {YIELD_MIN_PET_S} s"
        )
    return ScenarioResult(metric, wait.passed, values, details, wait.reason)


@register(*YIELD_LABELS)
def _yield(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """``yield_conflict`` when the anchor's span is the human's wait, else ``yield_progress``."""
    if inp.label in WAIT_SPAN_LABELS and inp.span_frames is not None:
        return yield_conflict(inp, config)
    return yield_progress(inp, config)
