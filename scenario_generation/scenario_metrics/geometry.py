"""Closed-loop scenario metrics: geometry family.

Owns labels: simple_turn, centerline, lane_change, object_avoidance, lane_follow.

Each scorer mirrors its open-loop counterpart in ``planner_metrics`` but reads the
ego's *realized* trajectory after the anchor instead of one predicted trajectory.
The open-loop horizon (8 s of prediction) becomes 8 s of the *recorded* drive after
the anchor frame, and the ego is scored over the sim steps that cover that stretch:
from ``anchor_step`` until the position-keyed cursor reaches the stretch's last
recorded frame. Tying the window to road position rather than sim time keeps
lateral scores independent of speed, the same decoupling the open-loop lane-change
metric makes; a trace that ends first (``max_steps``, abort) is scored on what it
has, and the progress check below says whether that covered the stretch.

A closed-loop ego can also *stall*, which a prediction cannot: an ego that stops
at the anchor has near-zero lateral error and no collision. ``simple_turn``,
``centerline`` and ``object_avoidance`` therefore also require the ego to cover
``MIN_PROGRESS_RATIO`` of the human's arc length over the stretch.

Open-loop ``simple_turn``/``centerline`` report errors only, with no pass line;
``MAX_LATERAL_ERROR_M`` is new here, and the raw errors are kept in ``values`` so
another threshold can be applied offline.

Event spans. When the anchor carries its event span (``inp.span_frames``, from the
registry's ``span_frame_start``/``span_frame_stop``), the recorded stretch is the span
itself -- turn start .. turn complete, lateral move .. settled in the new lane, lateral
move .. back in lane + 1 s -- instead of the fixed 8 s after the anchor, and it starts at
the span's first frame (the anchor, for these labels). Without a span (or with an empty
one) nothing changes. ``values["span_used"]`` says which interval was scored.

``lane_follow`` (span required; the builder cuts strict lane-follow runs into windows
whose anchor is the span start) scores the lateral deviation of the realized path from
the route-lane centerline over the whole span. A span is longer than one frame's route
lanes reach, so each sim step is measured against the ``route_lanes`` of the recorded
frame the cursor replayed, taken every ``LANE_FOLLOW_LANE_FRAME_STEP`` frames. Passing
needs max deviation <= ``LANE_FOLLOW_MAX_LATERAL_ERROR_M`` (p95 and the human's own
deviation are reported alongside), no road-border contact (the rollout's ``rb_dist_m``
below the evaluator's contact threshold) and ``LANE_FOLLOW_MIN_PROGRESS_RATIO`` of the
human's progress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from planner_metrics.centerline import compute_centerline_error_components_batch
from planner_metrics.gt_lateral_deviation import compute_gt_lateral_deviation_batch
from planner_metrics.lane_change import lane_change_decision
from planner_metrics.source_lane import reconstruct_source_lane
from scenario_generation.scenario_metrics.base import (
    ClosedLoopScenarioInput,
    ScenarioResult,
    path_arclength,
    project_onto_path,
)
from scenario_generation.scenario_metrics.registry import register
from scenario_generation.scenario_metrics.shared_config import open_loop_parameters

_REC_DT_S = 0.1  # one recorded frame per 0.1 s (see base)

# From the open-loop config (``shared_config``): the simple_turn / centerline /
# lane_change horizons (``scenario_<label>_horizon_seconds``) and lane_change's
# ``minimum_lateral_shift_m`` / ``chain_tolerance_m``. Everything below is closed loop
# only, or an open-loop module constant that is not a config field.

# object_avoidance has no open-loop horizon field; open loop scores the whole 8 s prediction.
OBJECT_AVOIDANCE_HORIZON_S = 8.0
# New in closed loop: the closest the ego may pass a neighbor without a collision.
# Human labels failed passes at 0.06-0.45 m and accepted 0.53 m and up.
OBJECT_AVOIDANCE_MIN_CLEARANCE_M = 0.5
# planner_metrics/lane_change.py::_SOURCE_PATH_MARGIN_M.
LANE_CHANGE_SOURCE_PATH_MARGIN_M = 20.0
# New in closed loop: how much recorded drive past the human's settling the ego gets to
# finish its lane change. Human labels failed only lane changes that came clearly late;
# 2 s matched them best (0.85 agreement, against 0.75 with no grace).
LANE_CHANGE_GRACE_S = 2.0
# New in closed loop (no open-loop threshold). 1.0 m is lane_change's default minimum
# lateral shift: a smaller deviation cannot have put the ego in another lane.
MAX_LATERAL_ERROR_M = 1.0
# New in closed loop: the anti-stall check. A human arc below the floor is "did not
# move", for which any ego progress counts as covered.
MIN_PROGRESS_RATIO = 0.9
MIN_REC_PROGRESS_M = 1.0
# lane_follow (closed loop only). 0.5 m max deviation is a user decision; progress is
# looser than the other labels' 0.9 because a lane-follow span has no event to reach.
LANE_FOLLOW_MAX_LATERAL_ERROR_M = 0.5
LANE_FOLLOW_MIN_PROGRESS_RATIO = 0.8
LANE_FOLLOW_HORIZON_S = 30.0  # fallback without a span (builder's max window length)
LANE_FOLLOW_LANE_FRAME_STEP = 20  # recorded frames (2 s) per route-lanes lookup
# reproducer_rollout.RB_COLLISION_THRESH_M: road-border contact below this distance.
ROAD_BORDER_CONTACT_M = 0.1


@dataclass(frozen=True)
class _Window:
    """The sim steps scored for one anchor and the recorded stretch they cover."""

    steps: np.ndarray  # sim steps k, from start_step on
    start_frame: int  # first recorded frame of the stretch (anchor, or span start)
    start_step: int  # first sim step whose replayed frame is at/past start_frame
    end_frame: int  # last recorded frame of the stretch
    span_used: bool  # the stretch is the anchor's event span, not a fixed horizon
    path_xy: np.ndarray  # rec_xy[start_frame : end_frame + 1]
    reached_end: bool  # the cursor reached end_frame
    ego_progress_m: float  # furthest arc length of the scored steps along path_xy
    rec_progress_m: float  # the human's arc length over path_xy
    # The trace ended inside the stretch on the window's own goal (within 5 m of the
    # window's last recorded pose), which can fall short of the ratio on a clipped stretch.
    goal_inside: bool
    min_progress_ratio: float = MIN_PROGRESS_RATIO

    @property
    def progress_ratio(self) -> float:
        if self.rec_progress_m < MIN_REC_PROGRESS_M:
            return 1.0
        return self.ego_progress_m / self.rec_progress_m

    @property
    def covered(self) -> bool:
        return self.progress_ratio >= self.min_progress_ratio or self.goal_inside

    def values(self, dt: float) -> dict[str, float]:
        return {
            "window_duration_s": float(len(self.steps) * dt),
            "eval_interval_s": float((self.end_frame - self.start_frame) * _REC_DT_S),
            "span_used": float(self.span_used),
            "ego_progress_m": self.ego_progress_m,
            "rec_progress_m": self.rec_progress_m,
            "progress_ratio": self.progress_ratio,
            "reached_end": float(self.reached_end),
        }


def _span(inp: ClosedLoopScenarioInput) -> tuple[int, int] | None:
    """The anchor's event span clipped to the window, or None (absent or empty)."""
    if inp.span_frames is None:
        return None
    first, last = max(int(inp.span_frames[0]), 0), min(int(inp.span_frames[1]), inp.n_frames - 1)
    return (first, last) if last > first else None


def _window(
    inp: ClosedLoopScenarioInput,
    horizon_s: float,
    *,
    min_progress_ratio: float = MIN_PROGRESS_RATIO,
    extend_s: float = 0.0,
) -> _Window | None:
    """Scored steps over the anchor's span, else ``horizon_s`` of recorded drive after the
    anchor, either stretched by ``extend_s`` of recorded drive; None if the live ego
    never reached the stretch's start."""
    span = _span(inp)
    if span is None:
        start_frame = inp.anchor_frame
        end_frame = min(inp.anchor_frame + int(round(horizon_s / _REC_DT_S)), inp.n_frames - 1)
    else:
        start_frame, end_frame = span
    end_frame = min(end_frame + int(round(extend_s / _REC_DT_S)), inp.n_frames - 1)
    reached = np.flatnonzero(inp.rec_idx >= start_frame)
    if not len(reached) or (span is None and inp.anchor_step is None):
        return None
    start_step = int(reached[0]) if span is not None else int(inp.anchor_step)
    after = np.arange(start_step, inp.n_steps)
    hit = np.flatnonzero(inp.rec_idx[after] >= end_frame)
    steps = after[: hit[0] + 1] if len(hit) else after
    path = inp.rec_xy[start_frame : end_frame + 1]
    arc, _ = project_onto_path(inp.ego_xy[steps], path)
    return _Window(
        steps=steps,
        start_frame=start_frame,
        start_step=start_step,
        end_frame=end_frame,
        span_used=span is not None,
        path_xy=path,
        reached_end=bool(len(hit)),
        ego_progress_m=float(arc.max()),
        rec_progress_m=float(path_arclength(path)[-1]),
        goal_inside=inp.terminated == "goal" and not len(hit),
        min_progress_ratio=min_progress_ratio,
    )


def _horizon_s(label: str, config) -> float:
    return float(open_loop_parameters(label, config)["horizon_seconds"])


def _to_local(inp: ClosedLoopScenarioInput, xy: np.ndarray, frame: int) -> np.ndarray:
    """Inverse of ``inp.to_world``: world points into recorded frame ``frame``'s ego frame."""
    c, s = np.cos(inp.rec_yaw[frame]), np.sin(inp.rec_yaw[frame])
    d = np.asarray(xy, dtype=np.float64) - inp.rec_xy[frame]
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1)


def _not_reached(metric: str) -> ScenarioResult:
    return ScenarioResult(metric=metric, passed=None, reason="anchor never reached")


def _base_details(inp: ClosedLoopScenarioInput, w: _Window) -> dict:
    return {
        "anchor_frame": inp.anchor_frame,
        "anchor_step": inp.anchor_step,
        "start_frame": w.start_frame,
        "start_step": w.start_step,
        "end_frame": w.end_frame,
        "last_step": int(w.steps[-1]),
        "terminated": inp.terminated,
    }


def _lateral_result(
    metric: str, inp: ClosedLoopScenarioInput, w: _Window, components: dict[str, torch.Tensor]
) -> ScenarioResult:
    """Shared verdict of the two lateral-deviation labels."""
    lateral = components["lateral_error_m"][0].numpy()
    longitudinal = components["longitudinal_error_m"][0].numpy()
    values = {
        "average_lateral_error_m": float(lateral.mean()),
        "final_lateral_error_m": float(lateral[-1]),
        "max_lateral_error_m": float(lateral.max()),
        "max_longitudinal_error_m": float(longitudinal.max()),
        **w.values(inp.dt),
    }
    covered = w.covered
    within = values["max_lateral_error_m"] <= MAX_LATERAL_ERROR_M
    if not covered:
        reason = "ego did not cover the recorded stretch"
    elif not within:
        reason = f"lateral error above {MAX_LATERAL_ERROR_M} m"
    else:
        reason = ""
    return ScenarioResult(
        metric=metric,
        passed=covered and within,
        values=values,
        details={**_base_details(inp, w), "covered": covered},
        reason=reason,
    )


@register("simple_turn")
def score_simple_turn(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """Lateral deviation of the realized path from the recorded human path over the turn.

    Mirrors ``planner_metrics/gt_lateral_deviation.py`` (the open-loop ``simple_turn``
    scorer) with the recorded stretch after the anchor as the GT path, both expressed
    in the anchor frame so the torch helper runs unchanged.
    """
    metric = "gt_lateral_deviation"
    w = _window(inp, _horizon_s("simple_turn", config))
    if w is None:
        return _not_reached(metric)
    if w.rec_progress_m < MIN_REC_PROGRESS_M:
        return ScenarioResult(
            metric=metric,
            passed=None,
            values=w.values(inp.dt),
            reason="recorded ego did not move after the anchor",
        )
    ego = _to_local(inp, inp.ego_xy[w.steps], w.start_frame)
    gt = _to_local(inp, w.path_xy, w.start_frame)
    components = compute_gt_lateral_deviation_batch(
        torch.from_numpy(ego)[None], {"ego_agent_future": torch.from_numpy(gt)[None]}
    )
    return _lateral_result(metric, inp, w, components)


@register("centerline")
def score_centerline(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """Lateral deviation of the realized path from the route-lane centerline after the anchor.

    Mirrors ``planner_metrics/centerline.py`` on the anchor frame's ``route_lanes``
    (``lanes`` as fallback, same precedence). One frame suffices: route lanes reach
    ~100 m ahead, more than 8 s of recorded drive at urban speeds.
    """
    metric = "centerline"
    w = _window(inp, _horizon_s("centerline", config))
    if w is None:
        return _not_reached(metric)
    frame = inp.load_frame(w.start_frame)
    lanes = {
        k: torch.from_numpy(np.asarray(frame[k], dtype=np.float64))
        for k in ("route_lanes", "lanes")
        if k in frame
    }
    ego = _to_local(inp, inp.ego_xy[w.steps], w.start_frame)
    try:
        components = compute_centerline_error_components_batch(torch.from_numpy(ego)[None], lanes)
    except ValueError as exc:  # no route lane / no valid centerline segment
        return ScenarioResult(metric=metric, passed=None, values=w.values(inp.dt), reason=str(exc))
    return _lateral_result(metric, inp, w, components)


@register("lane_change")
def score_lane_change(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """Did the realized trajectory complete the human's lane change?

    Mirrors ``planner_metrics/lane_change.py::evaluate_lane_change_with_details``: the
    source lane is rebuilt from the anchor frame's ``lanes`` at the recorded anchor pose
    (``source_lane.reconstruct_source_lane``), and every lateral reading is taken
    against it. Passing needs the ego's final offset past the source lane's boundary on
    the human's side AND within the lane tolerance of the human's final offset -- the
    decision is open loop's own ``lane_change_decision``, not a copy of it.

    Unlike open loop -- which counts them as failures -- a window whose human drove no
    lane change, or whose source lane is not heading-aligned, is not applicable. No
    progress check: as in open loop, a slow ego still gets credit for its lateral move.

    The stretch runs ``LANE_CHANGE_GRACE_S`` of recorded drive past the span's end (the
    human settled in the new lane), so a lane change a little later than the human's
    still passes; one that has not finished by then is late.
    """
    metric = "lane_change"
    params = open_loop_parameters("lane_change", config)
    min_lateral_shift_m = float(params["minimum_lateral_shift_m"])
    w = _window(inp, float(params["horizon_seconds"]), extend_s=LANE_CHANGE_GRACE_S)
    if w is None:
        return _not_reached(metric)
    frame = inp.load_frame(w.start_frame)
    lanes_np = frame.get("lanes", frame.get("route_lanes"))
    if lanes_np is None or lanes_np.shape[-1] < 8:
        return ScenarioResult(
            metric=metric, passed=None, reason="anchor frame has no lanes with boundary columns"
        )
    lanes = torch.from_numpy(np.asarray(lanes_np, dtype=np.float64))

    ego = _to_local(inp, inp.ego_xy[w.steps], w.start_frame)
    # Open-loop GT is ego_agent_future, which starts one frame after the anchor.
    gt = _to_local(inp, inp.rec_xy[w.start_frame + 1 : w.end_frame + 1], w.start_frame)
    if len(gt) < 2:
        return ScenarioResult(metric=metric, passed=None, reason="window ends at the anchor")
    required = (
        float(np.linalg.norm(np.concatenate([ego, gt]), axis=1).max())
        + LANE_CHANGE_SOURCE_PATH_MARGIN_M
    )
    try:
        source = reconstruct_source_lane(
            lanes, torch.zeros(2), required, float(params["chain_tolerance_m"])
        )
    except ValueError as exc:
        return ScenarioResult(metric=metric, passed=None, reason=str(exc))
    path = source.path.numpy()
    _, pred = project_onto_path(ego, path)
    _, gt_lat = project_onto_path(gt, path)
    left, right = source.half_widths_at(lanes, torch.from_numpy(gt[-1]))

    # The decision is open loop's own, run as a batch of one. Its verdicts come back
    # ungated; the two preconditions are turned into "not applicable" below.
    decision = lane_change_decision(
        torch.from_numpy(pred)[None],
        torch.from_numpy(gt_lat)[None],
        torch.tensor([left], dtype=torch.float64),
        torch.tensor([right], dtype=torch.float64),
        torch.tensor([source.heading_aligned]),
        min_lateral_shift_m,
        # Sim time from the anchor step to the first step past the boundary; the whole
        # window when it never got there (open loop reports the horizon likewise).
        torch.from_numpy((w.steps - w.start_step) * inp.dt),
        len(w.steps) * inp.dt,
    )
    detected = bool(decision["gt_lane_change_detected"][0])
    crossed = bool(decision["left_source_lane"][0])
    reached = bool(decision["reached_gt_lane"][0])
    values = {
        key: float(decision[key][0])
        for key in (
            "completion_ratio",
            "final_lateral_offset_error_m",
            "lane_change_time_s",
            "predicted_lateral_shift_m",
            "gt_lateral_shift_m",
            "initial_lateral_offset_m",
            "gt_direction",
            "lane_tolerance_m",
        )
    }
    values.update(
        {"left_source_lane": float(crossed), "reached_gt_lane": float(reached), **w.values(inp.dt)}
    )
    details = {
        **_base_details(inp, w),
        "source_lane_index": source.index,
        "source_lane_heading_aligned": source.heading_aligned,
        "lane_tolerance_from_map": bool(decision["lane_tolerance_from_map"][0]),
        "lane_half_width_left_m": left,
        "lane_half_width_right_m": right,
        "gt_lane_change_detected": detected,
    }
    if not source.heading_aligned:
        return ScenarioResult(
            metric, None, values, details, "source lane not heading-aligned with the recorded ego"
        )
    if not detected:
        return ScenarioResult(
            metric, None, values, details, "recorded ego performed no lane change after the anchor"
        )
    if not crossed:
        reason = "did not leave the source lane towards the recorded side"
    elif not reached:
        reason = "ended outside the recorded target lane"
    else:
        reason = ""
    return ScenarioResult(metric, crossed and reached, values, details, reason)


@register("object_avoidance")
def score_object_avoidance(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """No collision after the anchor, a wide enough berth, and the ego actually got past
    the obstacle.

    Mirrors ``planner_metrics/object_avoidance.py`` (collision = OBB clearance <= 0
    against any neighbor, no actor-type filter) on the rollout's own per-step
    collision/clearance, plus two closed-loop checks: the minimum clearance must reach
    ``OBJECT_AVOIDANCE_MIN_CLEARANCE_M`` (a scrape-by is not an avoidance), and the
    anti-stall progress check: stopping short of the obstacle forever would otherwise
    "avoid" it. As in open loop, a window with no neighbor at all is not applicable.
    """
    metric = "object_avoidance"
    w = _window(inp, OBJECT_AVOIDANCE_HORIZON_S)
    if w is None:
        return _not_reached(metric)
    clearance = inp.clearance_m[w.steps]
    collided = bool(inp.collision[w.steps].any() or (clearance <= 0.0).any())
    values = {
        "collision": float(collided),
        "min_clearance_m": float(clearance.min()),
        "min_clearance_threshold_m": OBJECT_AVOIDANCE_MIN_CLEARANCE_M,
        **w.values(inp.dt),
    }
    details = _base_details(inp, w)
    if not np.isfinite(clearance).any():
        return ScenarioResult(metric, None, values, details, "no neighbor after the anchor")
    covered = w.covered
    details["covered"] = covered
    wide = bool(clearance.min() >= OBJECT_AVOIDANCE_MIN_CLEARANCE_M)
    if collided:
        first = int(w.steps[np.flatnonzero(inp.collision[w.steps] | (clearance <= 0.0))[0]])
        details["first_collision_step"] = first
        reason = "collision after the anchor"
    elif not wide:
        reason = f"passed a neighbor closer than {OBJECT_AVOIDANCE_MIN_CLEARANCE_M} m"
    elif not covered:
        reason = "ego did not get past the recorded stretch"
    else:
        reason = ""
    return ScenarioResult(metric, not collided and wide and covered, values, details, reason)


def _route_lane_lateral(
    inp: ClosedLoopScenarioInput, w: _Window, xy: np.ndarray, frames: np.ndarray
) -> np.ndarray:
    """|Lateral offset| of each point from the route-lane centerline of its recorded frame.

    ``frames[j]`` is the recorded frame point ``j`` belongs to; lanes are read every
    ``LANE_FOLLOW_LANE_FRAME_STEP`` frames from ``w.start_frame`` (nan where a frame has no
    usable route lane).
    """
    out = np.full(len(xy), np.nan)
    base = (
        w.start_frame
        + (frames - w.start_frame) // LANE_FOLLOW_LANE_FRAME_STEP * LANE_FOLLOW_LANE_FRAME_STEP
    )
    for b in np.unique(base):
        rows = np.flatnonzero(base == b)
        frame = inp.load_frame(int(b))
        lanes = {
            k: torch.from_numpy(np.asarray(frame[k], dtype=np.float64))
            for k in ("route_lanes", "lanes")
            if k in frame
        }
        local = _to_local(inp, xy[rows], int(b))
        try:
            lateral = compute_centerline_error_components_batch(
                torch.from_numpy(local)[None], lanes
            )
        except ValueError:
            continue
        out[rows] = np.abs(lateral["lateral_error_m"][0].numpy())
    return out


@register("lane_follow")
def score_lane_follow(inp: ClosedLoopScenarioInput, config=None) -> ScenarioResult:
    """Stay on the route-lane centerline over the lane-follow span, without stalling or
    touching a road border.

    Closed loop only (open loop's nearest counterpart is ``centerline``). See the module
    docstring for the per-frame route lanes, thresholds and progress rule.
    """
    metric = "lane_follow"
    w = _window(inp, LANE_FOLLOW_HORIZON_S, min_progress_ratio=LANE_FOLLOW_MIN_PROGRESS_RATIO)
    if w is None:
        return _not_reached(metric)
    ego_frames = np.clip(inp.rec_idx[w.steps], w.start_frame, w.end_frame)
    ego = _route_lane_lateral(inp, w, inp.ego_xy[w.steps], ego_frames)
    rec_frames = np.arange(w.start_frame, w.end_frame + 1)
    rec = _route_lane_lateral(inp, w, inp.rec_xy[rec_frames], rec_frames)
    measured = np.isfinite(ego)
    values = w.values(inp.dt)
    details = _base_details(inp, w)
    if not measured.any():
        return ScenarioResult(metric, None, values, details, "no route lane over the span")
    lat = ego[measured]
    values.update(
        {
            "max_lateral_error_m": float(lat.max()),
            "p95_lateral_error_m": float(np.percentile(lat, 95)),
            "average_lateral_error_m": float(lat.mean()),
            "measured_ratio": float(measured.mean()),
            "collision": float(inp.collision[w.steps].any()),
        }
    )
    if np.isfinite(rec).any():
        values["rec_max_lateral_error_m"] = float(np.nanmax(rec))
        values["rec_p95_lateral_error_m"] = float(np.nanpercentile(rec, 95))
    border_contact = False
    if inp.road_border_m is not None:
        rb = inp.road_border_m[w.steps]
        finite = np.isfinite(rb)
        if finite.any():
            values["min_road_border_m"] = float(rb[finite].min())
            border_contact = bool((rb[finite] < ROAD_BORDER_CONTACT_M).any())
    details["road_border_checked"] = inp.road_border_m is not None
    covered = w.covered
    within = values["max_lateral_error_m"] <= LANE_FOLLOW_MAX_LATERAL_ERROR_M
    details["covered"] = covered
    if not covered:
        reason = "ego did not cover the recorded stretch"
    elif border_contact:
        reason = "road-border contact"
    elif not within:
        reason = f"lateral error above {LANE_FOLLOW_MAX_LATERAL_ERROR_M} m"
    else:
        reason = ""
    return ScenarioResult(
        metric, covered and within and not border_contact, values, details, reason
    )
