"""Full-route closed-loop rollout with per-step map-area tagging for grouped eval."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.utils.scene_skip import is_skipped

from scenario_generation.closed_loop_eval import build_mp4
from scenario_generation.metrics.centerline_deviation import lateral_offset_from_route_lanes
from scenario_generation.metrics.safety_clearance import score_safety_step
from scenario_generation.metrics.turn_indicator import score_turn_indicator_step
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import (
    DT,
    _advance_step,
    _closed_loop_replan_step,
    _draw_step,
    _feed_turn_indicator,
    _hold_turn_indicator,
    _pre_step,
    _seed_state,
)
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.scenario_classification import AreaEpisode, area_at_idx


@dataclass
class StepRecord:
    k: int
    rec_idx: int
    area: str | None
    centerline_m: float | None = None
    turn_match: bool | None = None
    clearance_m: float = float("inf")
    collision: bool = False
    neighbor_violation: bool = False
    rb_violation: bool = False
    png_path: Path | None = None


@dataclass
class RouteRolloutResult:
    route_key: str
    sequence: str
    steps: list[StepRecord] = field(default_factory=list)
    png_dir: Path | None = None
    n_steps_run: int = 0
    terminated: str = ""
    timers: Timers | None = None


def _percentile(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, q))


@torch.no_grad()
def run_full_route_rollout(
    model,
    model_args,
    tl: RouteTimeline,
    route_key: str,
    sequence: str,
    segments: list[AreaEpisode],
    area_to_metric_group: dict[str, str],
    png_dir: Path | None,
    *,
    device: str = "cuda",
    near_miss_thresh: float = 0.3,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    unstick_after: int = 300,
    unstick_advance_m: float = 2.5,
    draw_every: int = 8,
    replan_interval: int = 10,
    tracker_mode: str = "mpc",
    profile: bool = False,
    profile_sync_gpu: bool = False,
) -> RouteRolloutResult:
    """One closed-loop pass over the entire route; tag each step by reproducer ``rec_idx`` area."""
    n = len(tl)
    if n < 2:
        return RouteRolloutResult(route_key=route_key, sequence=sequence)

    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    cap = max(3 * n, n)
    timers = Timers() if profile else None
    rollout_timers = timers or Timers()
    s = _seed_state(
        tl,
        0,
        n,
        search_radius,
        warmup_steps,
        near_miss_thresh,
        goal_reach_m=5.0,
        max_stuck_steps=0,
        timers=rollout_timers,
        max_steps=cap,
        unstick_after=unstick_after,
        unstick_advance_m=unstick_advance_m,
        neighbor_history_mode="recorded",
        goal_mode="route",
        tracker_mode=tracker_mode,
    )

    steps: list[StepRecord] = []
    plan_world = None

    while not s.done:
        if timers is not None:
            with timers("pre_step"):
                pre = _pre_step(s)
        else:
            pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx, slot_uuids, _wbu = pre
        area = area_at_idx(segments, int(idx))
        metric_group = area_to_metric_group.get(area) if area else None

        turn_match = None
        pred_cur, plan_world, override, outputs = _closed_loop_replan_step(
            int(s.k),
            replan_interval,
            plan_world,
            s.live_pose,
            np_dict,
            model,
            model_args,
            device,
            timers=timers,
            profile_sync_gpu=profile_sync_gpu,
        )
        if outputs is not None:
            _feed_turn_indicator(s, outputs)
        else:
            _hold_turn_indicator(s)

        if tracker_mode == "perfect" and override is None:
            tx, ty, th = (
                float(plan_world[0][0, 0]),
                float(plan_world[0][0, 1]),
                float(plan_world[1][0]),
            )
            spd = float(np.hypot(tx - s.live_pose[0], ty - s.live_pose[1]) / DT)
            override = (np.array([tx, ty, th], dtype=np.float64), spd)

        def _score_step_metrics():
            nonlocal turn_match
            if outputs is not None and metric_group is not None:
                ti = score_turn_indicator_step(outputs, metric_group)
                if ti["turn_match"] is not None:
                    turn_match = bool(ti["turn_match"])
            centerline_m = None
            if metric_group in ("straight", "curve"):
                centerline_m = lateral_offset_from_route_lanes(np_dict["route_lanes"])
            safety = score_safety_step(
                neighbors_live,
                s.ego_shape,
                np_dict,
                device,
                margin_m=near_miss_thresh,
            )
            return centerline_m, safety

        if timers is not None:
            with timers("score_safety"):
                centerline_m, safety = _score_step_metrics()
        else:
            centerline_m, safety = _score_step_metrics()

        png_path = None
        if png_dir is not None and s.k % draw_every == 0:
            out_path = png_dir / f"{s.k:05d}.png"
            nids = tl.neighbor_ids(idx) if slot_uuids is None else slot_uuids
            if timers is not None:
                with timers("draw"):
                    _draw_step(
                        np_dict,
                        pred_cur,
                        s.ego_shape,
                        out_path,
                        neighbor_ids=nids,
                        step=s.k,
                        total=cap,
                    )
            else:
                _draw_step(
                    np_dict,
                    pred_cur,
                    s.ego_shape,
                    out_path,
                    neighbor_ids=nids,
                    step=s.k,
                    total=cap,
                )
            png_path = out_path

        steps.append(
            StepRecord(
                k=int(s.k),
                rec_idx=int(idx),
                area=area,
                centerline_m=centerline_m,
                turn_match=turn_match,
                clearance_m=float(safety["clearance_m"]),
                collision=bool(safety["collision"]),
                neighbor_violation=bool(safety["neighbor_violation"]),
                rb_violation=bool(safety["rb_violation"]),
                png_path=png_path,
            )
        )

        snaps_before = s.n_snaps
        _advance_step(s, pred_cur, idx, device, rollout_timers, override=override)
        if s.n_snaps > snaps_before:
            plan_world = None

    if profile and timers is not None:
        timers.merge(rollout_timers)

    return RouteRolloutResult(
        route_key=route_key,
        sequence=sequence,
        steps=steps,
        png_dir=png_dir,
        n_steps_run=int(s.k),
        terminated=s.terminated,
        timers=timers,
    )


def aggregate_area_metrics(
    steps: list[StepRecord],
    area_name: str,
    metric_group: str,
    route_key: str,
    bag_name: str,
    tl: RouteTimeline,
    *,
    labeled_ranges: list[list[int]],
    video_start_idx: int,
    video_end_idx: int,
    span_index: int,
    near_miss_thresh: float = 0.3,
) -> dict | None:
    """Aggregate metrics on labeled, non-skipped frames only."""
    from scenario_generation.scenario_classification import idx_in_labeled_ranges

    area_steps = [
        st
        for st in steps
        if st.area == area_name
        and idx_in_labeled_ranges(st.rec_idx, labeled_ranges)
        and not is_skipped(tl.npz_paths[st.rec_idx])
    ]
    if not area_steps:
        return None

    idxs = [st.rec_idx for st in area_steps]
    f_start = int(tl.frame_indices[video_start_idx])
    f_end = int(tl.frame_indices[min(video_end_idx - 1, len(tl.frame_indices) - 1)])

    cl = np.array(
        [st.centerline_m for st in area_steps if st.centerline_m is not None],
        dtype=np.float32,
    )
    tm = np.array(
        [st.turn_match for st in area_steps if st.turn_match is not None],
        dtype=bool,
    )
    clearances = np.array([st.clearance_m for st in area_steps], dtype=np.float32)
    collisions = [st.collision for st in area_steps]

    base = {
        "metric_group": metric_group,
        "area_name": area_name,
        "sequence": route_key,
        "bag": bag_name,
        "span_index": span_index,
        "list_start_idx": video_start_idx,
        "list_end_idx": video_end_idx,
        "labeled_ranges": labeled_ranges,
        "segment": f"[{f_start},{f_end}]",
        "n_steps_run": len(area_steps),
        "terminated": "area_span",
        "min_clearance": float(clearances[np.isfinite(clearances)].min())
        if clearances.size
        else float("inf"),
        "mean_clearance": float(clearances[np.isfinite(clearances)].mean())
        if clearances.size
        else float("inf"),
        "n_collision_steps": int(sum(collisions)),
        "n_near_miss_steps": int(np.sum(clearances <= near_miss_thresh)),
        "n_snaps": 0,
        "centerline_mean_m": float(cl[np.isfinite(cl)].mean()) if cl.size else None,
        "centerline_p95_m": _percentile(cl, 95) if cl.size else None,
        "centerline_max_m": float(cl[np.isfinite(cl)].max()) if cl.size else None,
        "turn_match_rate": float(tm.mean()) if tm.size else None,
        "neighbor_violation_steps": int(sum(st.neighbor_violation for st in area_steps)),
        "rb_violation_steps": int(sum(st.rb_violation for st in area_steps)),
        "collision_steps": int(sum(collisions)),
        "min_clearance_m": float(clearances[np.isfinite(clearances)].min())
        if clearances.size
        else float("inf"),
        "min_rb_dist_m": None,
        "video_path": "",
    }
    return base


def build_area_video(
    rollout: RouteRolloutResult,
    out_mp4: Path,
    fps: float,
    *,
    video_start_idx: int,
    video_end_idx: int,
) -> bool:
    """Copy PNGs for a continuous video span (includes unlabeled stopping gaps)."""
    if rollout.png_dir is None:
        return False
    video_steps = [
        st
        for st in rollout.steps
        if st.png_path is not None and video_start_idx <= st.rec_idx < video_end_idx
    ]
    if not video_steps:
        return False

    staging = out_mp4.parent / f"_staging_{out_mp4.stem}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for st in sorted(video_steps, key=lambda x: x.k):
        if st.png_path and st.png_path.is_file():
            shutil.copy2(st.png_path, staging / f"{st.png_path.name}")

    if not any(staging.glob("*.png")):
        shutil.rmtree(staging, ignore_errors=True)
        return False

    build_mp4(staging, out_mp4, fps)
    shutil.rmtree(staging, ignore_errors=True)
    return True
