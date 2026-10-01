"""Closed-loop scenario metrics: progress family.

Owns labels: departure, traffic_light_go, pedestrian_yield, vehicle_yield.

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


@dataclass(frozen=True)
class DepartureParams:
    horizon_s: float
    minimum_progress_m: float

    @classmethod
    def from_config(cls, label: str, config=None) -> DepartureParams:
        """Both from the open-loop config (``scenario_<label>_*``, see ``shared_config``)."""
        p = open_loop_parameters(label, config)
        return cls(
            horizon_s=float(p["horizon_seconds"]),
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
    }
    if p.available_steps:
        values["ol_max_displacement_m"] = p.ol_max_displacement_m
        values["ol_passed"] = float(p.ol_max_displacement_m >= params.minimum_progress_m)
    reason = ""
    if not departed and p.truncated and inp.terminated == "goal":
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


@register(*YIELD_LABELS)
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
