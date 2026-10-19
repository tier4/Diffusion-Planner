"""Native-H5 closed-loop evaluation of every segment at once on the e2e-devkit batched simulator, scored, traced and drawn by the reproducer's own code."""

from __future__ import annotations

import json
import math
import time
import zlib
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO

import h5py
import hdf5plugin  # noqa: F401 - registers the zstd HDF5 filter
import numpy as np
import torch
from t4_e2e_devkit.sim.batched.executors import PlacedTrackingExecutor
from t4_e2e_devkit.sim.batched.frames import FrameSelectorConfig
from t4_e2e_devkit.sim.batched.poses import local_to_world, world_to_local
from t4_e2e_devkit.sim.batched.simulator import (
    DIVERGED,
    TERMINATIONS,
    BatchedReplaySimulator,
    ReplayEpisodes,
    SimulatorConfig,
    SimulatorStep,
)
from t4_e2e_devkit.sim.batched.window import WindowedFrames

from scenario_generation.closed_loop_eval import build_mp4
from scenario_generation.closed_loop_evaluation import (
    FullRouteRouteJob,
    JobRunResult,
    RolloutParams,
)
from scenario_generation.metrics.object import score_object_step_batched
from scenario_generation.perf_timer import Timers
from scenario_generation.plan_optimizer import PlanOptimizer
from scenario_generation.reproducer_rollout import (
    _GT_DEV_WINDOW_FRAMES,
    DT,
    STUCK_SPEED_MPS,
    GTTracking,
    _draw_step,
    _EgoDyn,
    _finalize,
    _optimize_plan,
    _score_into,
    _score_turn_indicator,
    _seed_state,
    _SegState,
    _world_plan_to_ego,
    _world_pose_to_ego,
    gt_deviation_viz,
    score_gt_tracking,
    trace_start,
    trace_step,
    trace_terminated,
)

from .closed_loop import (
    NativeH5FullRouteClosedLoopEvaluation,
    NativeH5RouteTimeline,
    TorchNativeNormalizer,
)
from .metric_compat import legacy_lanes, legacy_route_lanes
from .schema import MODEL_INPUT_NAMES

BINARY_KEYS = (
    "lane_traffic_light_past",
    "route_traffic_light_past",
    "lane_traffic_light_future",
    "route_traffic_light_future",
    "lane_types",
    "route_lane_types",
    "agent_label",
)
OBJECT_KEYS = ("neighbor_agents_past",)
PLAN_BATCH = 128
MAX_SEGMENTS = 256
MAX_QUEUED_DRAWINGS = 256
VIDEO_NAME = "segment.mp4"
WINDOW_FRAMES = 64
READ_CHUNK = 32
READ_WORKERS = 4
FIRST_TURN_CODE = 1
SUPPORTED = {
    "tracker_mode": "perfect",
    "neighbor_history_mode": "recorded",
    "timeline_progress_mode": "pose",
    "goal_mode": "segment",
    "warmup_steps": 0,
    "max_stuck_steps": 0,
    "abort_max_snaps": 0,
    "window": None,
}


def simulator_config(params: RolloutParams) -> SimulatorConfig:
    """The devkit simulator settings that reproduce ``render_segment`` under ``params``; raises on a setting it does not reproduce."""
    unsupported = {
        name: getattr(params, name)
        for name, value in SUPPORTED.items()
        if getattr(params, name) != value
    }
    if unsupported:
        raise ValueError(
            f"the batched closed-loop backend reproduces only {SUPPORTED}; got {unsupported}"
        )
    return SimulatorConfig(
        dt=DT,
        plan_steps=80,
        replan_steps=params.replan_interval,
        agent_shift_steps=0,
        goal_reach_m=params.goal_reach_m,
        goal_pass_window_m=None,
        route_end_stop_steps=None,
        end_without_route_lanes=False,
        diverge_m=params.abort_deviation_m if params.abort_deviation_m > 0 else None,
        diverge_after=params.abort_after,
        deviation_window=_GT_DEV_WINDOW_FRAMES,
        stuck_speed_mps=STUCK_SPEED_MPS,
        unstick_after=params.unstick_after,
        unstick_radius_mult=params.unstick_radius_mult,
        unstick_teleport_after=params.unstick_teleport_after,
        unstick_advance_m=params.unstick_advance_m,
        moving_speed_mps=0.0,
        yaw_rate_clamp_rps=math.inf,
        frames=FrameSelectorConfig(
            search_radius_m=params.search_radius,
            heading_gate_rad=math.pi / 2 if params.yaw_gate else math.inf,
        ),
    )


class FrameReader:
    """Frames ``start:stop`` of each segment's model inputs, with the objects removed for no-object segments."""

    def __init__(self, segments: list[tuple[str, int, bool]]) -> None:
        """Bind ``(h5_path, first H5 row, drop_objects)`` per segment."""
        self.segments = segments
        self.handles: dict[str, h5py.File] = {}

    def __call__(self, segment: int, start: int, stop: int) -> dict[str, np.ndarray]:
        """Read frames ``start:stop`` of ``segment``."""
        path, first, drop = self.segments[segment]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        frames = self.handles[path]["frames"]
        rows = slice(first + start, first + stop)
        out = {key: np.asarray(frames[key][rows], dtype=np.float32) for key in MODEL_INPUT_NAMES}
        if drop:
            for key in OBJECT_KEYS:
                out[key] = np.zeros_like(out[key])
        return out

    def __getstate__(self) -> dict:
        """Pickle without the open files."""
        return {"segments": self.segments, "handles": {}}


@dataclass
class ReplayCursor:
    """The frame-selection state the reproducer's trace and summary read, filled from the simulator."""

    max_idx_reached: int = 0
    state: str = "normal"
    state_run_steps: int = 0
    normal_steps: int = 0
    repeat_steps: int = 0


@dataclass
class Segment:
    """One segment rolled out in the batch: its evaluator, job, timeline span, scoring state, trace and optimizer."""

    evaluator: BatchedNativeH5Evaluation
    job: FullRouteRouteJob
    timeline: NativeH5RouteTimeline
    span: tuple[int, int]
    png_dir: Path
    state: _SegState
    trace: IO[str]
    optimizer: PlanOptimizer | None
    pending: dict | None = None
    teleports: int = 0
    reseed_turns: bool = False
    drawn: list = field(default_factory=list)
    ended: bool = False
    diverged: bool = False

    @property
    def seed(self) -> int:
        """The sampler seed of this segment's first tick, the same whatever else runs in the batch."""
        return zlib.crc32(f"{self.job.route_key}:{self.span[0]}".encode())


@dataclass
class Tick:
    """What one tick scored for one row and the plan, drawing and trace steps read."""

    row: int
    segment: Segment
    idx: int
    np_dict: dict
    gt: GTTracking


class BatchedRollout:
    """All segments of all evaluators rolled out together on the devkit simulator, one tick at a time."""

    def __init__(
        self,
        model,
        segments: list[Segment],
        params: RolloutParams,
        draw_pool,
        timers: Timers,
    ) -> None:
        """Build the episodes of ``segments`` and the simulator that replays them."""
        self.model = model
        self.segments = segments
        self.params = params
        self.draw_pool = draw_pool
        self.timers = timers
        self.queued: deque[Future] = deque()
        self.normalize = TorchNativeNormalizer()
        self.device = torch.device(params.device)
        lengths = [end - start for start, end in (segment.span for segment in segments)]
        longest = max(lengths)
        poses = torch.full((len(segments), longest, 3), 1e9, dtype=torch.float64)
        speeds = torch.zeros(len(segments), longest, dtype=torch.float64)
        for row, segment in enumerate(segments):
            start, end = segment.span
            poses[row, : end - start] = torch.as_tensor(segment.timeline.poses[start:end])
            speeds[row, : end - start] = torch.as_tensor(segment.timeline.speeds[start:end])
        counts = torch.as_tensor(lengths, device=self.device)
        steps = torch.as_tensor(
            [segment.state.max_steps for segment in segments], device=self.device
        )
        reader = FrameReader(
            [
                (
                    str(s.timeline.h5_path),
                    int(s.timeline.frame_indices[s.span[0]]),
                    s.evaluator.config.params.drop_objects,
                )
                for s in segments
            ]
        )
        self.frames = WindowedFrames(
            reader, lengths, WINDOW_FRAMES, READ_CHUNK, BINARY_KEYS, self.device, READ_WORKERS
        )
        self.sim = BatchedReplaySimulator(
            ReplayEpisodes(
                self.frames,
                poses.to(self.device),
                speeds.to(self.device),
                counts,
                counts - 1,
                steps,
            ),
            PlacedTrackingExecutor(dt=DT),
            simulator_config(params),
        )

    def run(self) -> None:
        """Roll every segment out to its end, writing traces and drawing frames, and leave each segment's metrics in ``segment.state``."""
        step = self.sim.reset()
        try:
            while True:
                self._end(step)
                if not bool(step.active.any()):
                    break
                with self.timers("score_tick"):
                    ticks = self._score(step)
                with self.timers("plan"):
                    plans, logits, adopt = self._plan(step, ticks)
                with self.timers("draw_submit"):
                    self._draw(step, ticks, plans, adopt)
                speed = self.sim.speed.clone()
                with self.timers("simulate"):
                    step = self.sim.step(plans, logits, adopt)
                self._advance(ticks, speed)
            self._finish()
        finally:
            self.frames.close()

    def _end(self, step: SimulatorStep) -> None:
        """Close the segments that stopped running: score a diverged one's last tick and log it as ``render_segment`` does, wait for its drawings and encode its video, so neither the queue nor the frames on disk outgrow the running segments."""
        stopped = torch.nonzero(~step.active).flatten().tolist()
        for row in stopped:
            segment = self.segments[row]
            if segment.ended:
                continue
            segment.ended = True
            if int(self.sim.termination[row]) == DIVERGED:
                self._log_divergence(row, segment)
            for future in segment.drawn:
                future.result()
            segment.drawn.clear()
            if any(segment.png_dir.glob("*.png")):
                build_mp4(
                    segment.png_dir, segment.png_dir / VIDEO_NAME, segment.evaluator.config.fps
                )

    def _log_divergence(self, row: int, segment: Segment) -> None:
        """Score the GT deviation of the tick a segment diverged on and write the reproducer's ``diverged`` trace line."""
        s = segment.state
        s.k = int(self.sim.ticks[row])
        s.live_pose = self.sim.pose[row].cpu().numpy().astype(np.float64)
        gt = score_gt_tracking(s, segment.span[0] + int(self.sim.index[row]))
        line = {
            "event": "diverged",
            "k": s.k,
            "gt_deviation_m": round(gt.deviation.dist_m, 3),
            "deviation_streak": int(self.params.abort_after),
            "snap_count": int(self.sim.teleports[row]),
        }
        segment.trace.write(json.dumps(line) + "\n")
        segment.diverged = True

    def _score(self, step: SimulatorStep) -> list[Tick]:
        """Score the current tick of every running segment, as ``render_segment`` does before its replan."""
        sim = self.sim
        active = torch.nonzero(step.active).flatten()
        rows = active.tolist()
        scene = {key: step.scene[key][active].float().cpu().numpy() for key in MODEL_INPUT_NAMES}
        pose = sim.pose[active].cpu().numpy()
        history = (
            torch.cat(
                (
                    sim.history[active],
                    sim.history_speed[active, :, None],
                    sim.history_yaw_rate[active, :, None],
                ),
                -1,
            )
            .cpu()
            .numpy()
        )
        state = {
            "index": sim.index[active].cpu().numpy(),
            "reached": sim.selector.reached[active].cpu().numpy(),
            "repeated": sim.selector.repeated[active].cpu().numpy(),
            "stuck": sim.stuck[active].cpu().numpy(),
            "expands": sim.expands[active].cpu().numpy(),
            "teleports": sim.teleports[active].cpu().numpy(),
            "normal": sim.normal_steps[active].cpu().numpy(),
            "repeat": sim.repeat_steps[active].cpu().numpy(),
            "ticks": sim.ticks[active].cpu().numpy(),
            "speed": sim.speed[active].cpu().numpy(),
        }
        objects = score_object_step_batched(
            [
                neighbors_live(scene, i, self.segments[row].evaluator.config.params.drop_objects)
                for i, row in enumerate(rows)
            ],
            [self.segments[row].state.ego_shape for row in rows],
            self.params.device,
        )
        return [
            self._score_row(row, i, scene, pose[i], history[i], state, objects[i])
            for i, row in enumerate(rows)
        ]

    def _score_row(self, row, i, scene, pose, history, state, objects) -> Tick:
        """Fill segment ``row``'s state for this tick and score it."""
        segment = self.segments[row]
        s = segment.state
        idx = segment.span[0] + int(state["index"][i])
        if segment.reseed_turns:
            turn = s.tl.recorded_turn_indicator(idx)
            s.last_turn_indicator = turn
            s.turn_indicator_prev_scored_gt = s.turn_indicator_prev_scored_pred = turn
            s.prev_max_idx, s.stuck = idx, 0
            segment.reseed_turns = False
        s.k = int(state["ticks"][i])
        s.sim_time = s.k * DT
        s.live_pose = pose.astype(np.float64)
        s.ego_hist = history.astype(np.float64)
        s.dyn = _EgoDyn(
            speed=float(state["speed"][i]), accel=s.dyn.accel, yaw_rate=float(history[-1, 4])
        )
        update_cursor(s.cursor, segment.span[0], state, i)
        if s.cursor.max_idx_reached > s.prev_max_idx:
            s.prev_max_idx, s.stuck = s.cursor.max_idx_reached, 0
        else:
            s.stuck += 1
        s.ego_stuck = int(state["stuck"][i])
        s.expand_count = int(state["expands"][i])
        s.snap_count = int(state["teleports"][i])
        np_dict = scoring_inputs(scene, i, s)
        cl, col, rear, _, _ = objects
        _score_into(
            s,
            None,
            self.params.device,
            self.timers,
            np_dict,
            object_cl=cl,
            object_col=col,
            object_rear_col=rear,
        )
        gt = score_gt_tracking(s, idx)
        segment.pending = trace_step(s, idx, gt)
        return Tick(row, segment, idx, np_dict, gt)

    def _plan(self, step: SimulatorStep, ticks: list[Tick]):
        """Plan for the rows due for a replan, pass the plans through the optimizer when it is on, and score their turn indicators."""
        rows = torch.nonzero(step.replan).flatten()
        if not len(rows):
            return None, None, None
        plans, logits = [], []
        for part in rows.split(PLAN_BATCH):
            data = self.normalize({key: step.scene[key][part].float() for key in MODEL_INPUT_NAMES})
            seeds = [
                (self.segments[row].seed + self.model.seed + tick) % 2**31
                for row, tick in zip(part.tolist(), self.sim.ticks[part].tolist(), strict=True)
            ]
            _, out = self.model(data, seeds)
            plans.append(out["prediction"][:, 0].float().to(self.device))
            logits.append(out["turn_indicator_logit"][:, 1:4].float().to(self.device))
        plans, logits = torch.cat(plans), torch.cat(logits)
        by_row = {tick.row: tick for tick in ticks}
        codes = (logits.argmax(-1) + FIRST_TURN_CODE).tolist()
        adopt = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        for j, row in enumerate(rows.tolist()):
            tick = by_row[row]
            s = tick.segment.state
            s.last_turn_indicator = codes[j]
            _score_turn_indicator(s, tick.idx)
            if tick.segment.optimizer is not None:
                with self.timers("plan_optimizer"):
                    plans[j], adopt[j] = self._optimize(
                        tick, plans[j], bool(self.sim.has_plan[row])
                    )
        return plans, logits, adopt

    def _optimize(
        self, tick: Tick, plan: torch.Tensor, has_plan: bool
    ) -> tuple[torch.Tensor, bool]:
        """Pass one plan through the segment's optimizer; a failed solve keeps the current plan, or the raw one before the first."""
        s = tick.segment.state
        pose = torch.as_tensor(s.live_pose, device=self.device)
        world = local_to_world(plan.double(), pose[None]).cpu().numpy()
        solved = _optimize_plan(
            tick.segment.optimizer, (world[:, :2], world[:, 2]), s, tick.np_dict
        )
        if solved is None:
            return plan, not has_plan
        local = _world_plan_to_ego(solved[0], solved[1], *s.live_pose)
        return torch.as_tensor(local, device=self.device), True

    def _draw(self, step: SimulatorStep, ticks: list[Tick], plans, adopt) -> None:
        """Hand this tick's frame of every segment due for a drawing to the draw pool, as ``render_segment`` does."""
        every = self.params.draw_every
        if every is None:
            return
        fresh = {}
        if plans is not None:
            for j, row in enumerate(torch.nonzero(step.replan).flatten().tolist()):
                if bool(adopt[j]):
                    fresh[row] = plans[j].cpu().numpy()
        for tick in ticks:
            s = tick.segment.state
            if s.k % every:
                continue
            plan = fresh.get(tick.row)
            if plan is None:
                plan = self._current_plan(tick.row)
            while len(self.queued) >= MAX_QUEUED_DRAWINGS:
                self.queued.popleft().result()
            self.queued.append(
                self.draw_pool.submit(
                    _draw_step,
                    tick.np_dict,
                    plan,
                    s.ego_shape,
                    tick.segment.png_dir / f"{s.k:05d}.png",
                    neighbor_ids=None,
                    step=s.k,
                    total=s.max_steps,
                    title_prefix=self.params.title_prefix,
                    distance_label_offset_m=self.params.distance_label_offset_m,
                    view_half_m=self.params.view_half_m,
                    reproducer_ego=_world_pose_to_ego(s.tl.poses[tick.idx], s.live_pose),
                    gt_deviation_viz=gt_deviation_viz(s.tl, tick.gt.deviation, s.live_pose),
                    turn_indicator_pred=int(s.last_turn_indicator),
                    turn_indicator_gt=s.tl.recorded_turn_indicator(tick.idx),
                )
            )
            tick.segment.drawn.append(self.queued[-1])

    def _current_plan(self, row: int) -> np.ndarray:
        """The rest of the plan row ``row`` is executing, in its current ego frame."""
        sim = self.sim
        rest = sim.plan[row, min(int(sim.plan_step[row]), sim.plan.shape[1] - 1) :]
        return world_to_local(rest, sim.pose[row][None]).float().cpu().numpy()

    def _advance(self, ticks: list[Tick], speed_before: torch.Tensor) -> None:
        """Record the acceleration each segment's step produced and write its trace line, as ``render_segment`` does after its advance."""
        rows = torch.as_tensor([tick.row for tick in ticks], device=self.device)
        speed = self.sim.speed[rows].cpu().numpy()
        before = speed_before[rows].cpu().numpy()
        teleports = self.sim.teleports[rows].cpu().numpy()
        for i, tick in enumerate(ticks):
            segment, s = tick.segment, tick.segment.state
            teleported = int(teleports[i]) != segment.teleports
            segment.teleports = int(teleports[i])
            accel = float((speed[i] - before[i]) / DT)
            s.accels[s.k] = np.nan if teleported else accel
            s.dyn = _EgoDyn(speed=float(speed[i]), accel=0.0 if teleported else accel)
            segment.reseed_turns = teleported
            row = segment.pending
            row["brake_accel_mps2"] = None if teleported else accel
            segment.trace.write(json.dumps(row) + "\n")

    def _finish(self) -> None:
        """Copy each segment's end state from the simulator into its scoring state and close its trace."""
        sim = self.sim
        final = {
            "ticks": sim.ticks,
            "termination": sim.termination,
            "reached": sim.selector.reached,
            "normal": sim.normal_steps,
            "repeat": sim.repeat_steps,
            "expands": sim.expands,
            "teleports": sim.teleports,
        }
        final = {key: value.cpu().numpy() for key, value in final.items()}
        final_pose = sim.pose.detach().cpu().numpy()
        for row, segment in enumerate(self.segments):
            s = segment.state
            s.k = int(final["ticks"][row])
            s.terminated = TERMINATIONS[int(final["termination"][row])]
            s.live_pose = final_pose[row].astype(np.float64)
            s.cursor.max_idx_reached = segment.span[0] + int(final["reached"][row])
            s.cursor.normal_steps = int(final["normal"][row])
            s.cursor.repeat_steps = int(final["repeat"][row])
            s.expand_count = int(final["expands"][row])
            s.snap_count = int(final["teleports"][row])
            if not segment.diverged:
                segment.trace.write(json.dumps(trace_terminated(s)) + "\n")
            segment.trace.close()


def neighbors_live(scene: dict[str, np.ndarray], i: int, drop_objects: bool) -> np.ndarray:
    """Row ``i``'s current agents in the reproducer's ``(A, 11)`` scoring layout, all zero without objects as the reproducer scores them."""
    agents = scene["neighbor_agents_past"][i, :, -1]
    out = np.zeros((agents.shape[0], 11), dtype=np.float32)
    if drop_objects:
        return out
    out[:, :4] = agents[:, :4]
    out[:, 6:8] = scene["agent_shape"][i]
    out[:, 8:11] = scene["agent_label"][i]
    return out


def scoring_inputs(scene: dict[str, np.ndarray], i: int, s: _SegState) -> dict:
    """Row ``i``'s scoring and drawing inputs, built as ``build_input_np`` builds them for a native-H5 frame."""
    native = {key: np.array(scene[key][i]) for key in MODEL_INPUT_NAMES}
    lines = np.zeros((*native["road_borders"].shape[:-1], 4), dtype=np.float32)
    lines[..., :2] = native["road_borders"]
    lines[..., 3] = (np.linalg.norm(lines[..., :2], axis=-1) > 0).astype(np.float32)
    out = {key: value[None] for key, value in native.items()}
    out["ego_current_state"] = np.array(
        [[0.0, 0.0, 1.0, 0.0, s.dyn.speed, 0.0, s.dyn.accel, 0.0, s.dyn.steering, s.dyn.yaw_rate]],
        dtype=np.float32,
    )
    out["line_strings"] = lines[None]
    out["metric_lanes"] = legacy_lanes(native)[None]
    out["metric_route_lanes"] = legacy_route_lanes(native)[None]
    out["metric_ego_shape"] = s.ego_shape[None]
    return out


def update_cursor(cursor: ReplayCursor, start: int, state: dict, i: int) -> None:
    """Copy row ``i``'s frame-selection state into ``cursor``."""
    name = "repeat" if bool(state["repeated"][i]) else "normal"
    cursor.state_run_steps = cursor.state_run_steps + 1 if name == cursor.state else 1
    cursor.state = name
    cursor.max_idx_reached = start + int(state["reached"][i])
    cursor.normal_steps = int(state["normal"][i])
    cursor.repeat_steps = int(state["repeat"][i])


class BatchedNativeH5Evaluation(NativeH5FullRouteClosedLoopEvaluation):
    """Native-H5 evaluator whose segments were rolled out by ``run_batched``; it records them as the standard evaluator does."""

    def __init__(self, *args, **kwargs) -> None:
        """Set up as the standard evaluator, with no rolled-out segments yet."""
        super().__init__(*args, **kwargs)
        self.rolled_out: dict[str, list[tuple[tuple[int, int], Path, dict]]] = {}

    def run_job(
        self,
        job,
        *,
        segments_file=None,
        digest_file=None,
        draw_pool=None,
        frames_root: Path | None = None,
    ) -> JobRunResult:
        """Record the segments ``run_batched`` rolled out for ``job``."""
        result = JobRunResult()
        for span, png_dir, metrics in self.rolled_out.pop(job.job_id):
            self.record_segment(
                result,
                job,
                span,
                png_dir,
                metrics,
                segments_file,
                digest_file,
                png_dir / VIDEO_NAME,
            )
        return result


def segment_state(
    timeline: NativeH5RouteTimeline, span: tuple[int, int], params: RolloutParams
) -> _SegState:
    """The reproducer's scoring state of one segment, with a cursor the simulator fills."""
    start, end = span
    s = _seed_state(
        tl=timeline,
        start=start,
        end=end,
        search_radius=params.search_radius,
        warmup_steps=params.warmup_steps,
        near_miss_thresh=params.near_miss_thresh,
        goal_reach_m=params.goal_reach_m,
        max_stuck_steps=params.max_stuck_steps,
        timers=Timers(),
        max_steps=params.max_steps if params.max_steps is not None else 3 * (end - start),
        unstick_after=params.unstick_after,
        unstick_advance_m=params.unstick_advance_m,
        unstick_radius_mult=params.unstick_radius_mult,
        unstick_teleport_after=params.unstick_teleport_after,
        neighbor_history_mode=params.neighbor_history_mode,
        tracker_mode=params.tracker_mode,
        replay_mode=params.timeline_progress_mode,
        goal_mode=params.goal_mode,
        strong_brake_mps2=params.strong_brake_mps2,
        yaw_gate=params.yaw_gate,
        deviation_collision_thresh_m=params.deviation_collision_thresh_m,
    )
    s.cursor = ReplayCursor(max_idx_reached=start)
    return s


def plan_optimizer(params: RolloutParams, s: _SegState) -> PlanOptimizer | None:
    """The segment's plan optimizer, set up as ``render_segment`` sets it up, or None when it is off."""
    if params.plan_optimizer is None:
        return None
    wheelbase, length, width = map(float, s.ego_shape)
    box = {"dimensions": {"x": length, "y": width}, "center": {"x": wheelbase / 2.0}}
    return PlanOptimizer(params.plan_optimizer, box, [])


def job_segments(
    evaluator: BatchedNativeH5Evaluation,
    key: str,
    job: FullRouteRouteJob,
    frames_root: Path,
    timelines: list[NativeH5RouteTimeline],
    timers: Timers,
) -> list[Segment]:
    """Open ``job``'s route, appending its timeline to ``timelines``, and set up each of its segments with a started trace."""
    params = evaluator.config.params
    route = evaluator._routes_by_id[job.job_id]
    timeline = NativeH5RouteTimeline(
        route["h5_path"],
        int(route.get("frame_start", 0)),
        route.get("frame_stop"),
        timers=timers,
        vehicle_params=evaluator.vehicle_params,
    )
    timelines.append(timeline)
    evaluator.rolled_out[job.job_id] = []
    segments = []
    spans = list(timeline.iter_segments(job.seg_len))
    if len(spans) > 1:
        raise ValueError(
            f"{job.route_key} splits into {len(spans)} segments of {job.seg_len} frames; the batched "
            "backend replays whole routes, so set closed_loop_seg_len to at least the route length"
        )
    for span in spans:
        png_dir = frames_root / key / f"{job.route_key}_{span[0]}_{span[1]}"
        png_dir.mkdir(parents=True, exist_ok=True)
        s = segment_state(timeline, span, params)
        trace = (png_dir / "rollout.jsonl").open("w", encoding="utf-8")
        trace.write(json.dumps(trace_start(s)) + "\n")
        segments.append(
            Segment(evaluator, job, timeline, span, png_dir, s, trace, plan_optimizer(params, s))
        )
    return segments


def shared_params(evaluators: dict[str, BatchedNativeH5Evaluation]) -> RolloutParams:
    """The rollout settings every evaluator shares, objects aside; raises when they differ."""
    shared = [replace(e.config.params, drop_objects=False) for e in evaluators.values()]
    if any(other != shared[0] for other in shared):
        raise ValueError("one batch replays every input with the same rollout settings but objects")
    return shared[0]


def hand_over(segment: Segment) -> None:
    """Give a rolled-out segment's metrics, with its optimizer summary when one ran, to its evaluator."""
    metrics = _finalize(segment.state)
    if segment.optimizer is not None:
        metrics["plan_optimizer"] = segment.optimizer.summary()
    segment.evaluator.rolled_out[segment.job.job_id].append(
        (segment.span, segment.png_dir, metrics)
    )


def run_batched(
    model,
    assignments: dict[str, list[FullRouteRouteJob]],
    evaluators: dict[str, BatchedNativeH5Evaluation],
    frames_root: Path,
    draw_pool,
) -> dict:
    """Roll out every assigned segment of every evaluator, at most ``MAX_SEGMENTS`` routes at a time, and hand each evaluator its rolled-out segments; returns the stage timings."""
    timers = Timers()
    started = time.perf_counter()
    work = [(key, job) for key, jobs in assignments.items() for job in jobs]
    if work:
        params = shared_params(evaluators)
        for first in range(0, len(work), MAX_SEGMENTS):
            roll_out_jobs(
                model,
                work[first : first + MAX_SEGMENTS],
                evaluators,
                params,
                frames_root,
                draw_pool,
                timers,
            )
    return {"segments": len(work), "elapsed_sec": time.perf_counter() - started, **timers.as_dict()}


def roll_out_jobs(model, work, evaluators, params, frames_root, draw_pool, timers) -> None:
    """Roll out one batch of ``(evaluator key, job)`` routes, hand their metrics over and release their files."""
    timelines: list[NativeH5RouteTimeline] = []
    segments: list[Segment] = []
    try:
        for key, job in work:
            segments += job_segments(evaluators[key], key, job, frames_root, timelines, timers)
        BatchedRollout(model, segments, params, draw_pool, timers).run()
        for segment in segments:
            hand_over(segment)
    finally:
        for segment in segments:
            segment.trace.close()
        for timeline in timelines:
            timeline.close()
