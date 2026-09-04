"""OpenSCENARIO-driven closed-loop rollout for the Diffusion-Planner validator.

The ``scenario_sim`` sibling of the NPZ-replay path: it drives the C++ OpenSCENARIO
interpreter (``openscenario_python.HeadlessRunner``, built with SSV2_HEADLESS_EGO) and runs
the live Diffusion-Planner as the ego on every tick -- read sim truth, build a
:class:`~scenario_generation.scene_context.SceneContext` snapshot, infer, inject the plan,
``step()``, repeat.

Invariant: the Python scene is never advanced. The C++ ``step()`` is the sole integrator for
both the ego and the NPCs; Python only reads truth and injects a plan. Advancing the Python
scene as well would double-simulate and diverge from sim truth.

Frame contract: the sim reports map-frame poses, which the SceneContext stores as-is;
``to_model_tensors`` re-centers onto the ego, and the ego-frame plan is mapped back with the
current ego pose before injection.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from concurrent.futures import Executor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.metrics.object import score_object_step
from scenario_generation.perf_timer import Timers
from scenario_generation.render_pool import render_pool
from scenario_generation.reproducer_rollout import _world_plan_to_ego
from scenario_generation.scenario_sim_metrics import build_segment_row
from scenario_generation.scenario_sim_route import resolve_route
from scenario_generation.scenario_sim_scene import (
    _HISTORY_LEN,
    DT,
    TURN_INDICATOR_DISABLE,
    HistoryBuffers,
    SceneConfig,
    baselink_xyh,
    build_scene,
    ego_metric_box,
    resolve_ego_name,
    update_history,
)
from scenario_generation.simulate import (
    _ego_to_world,
    _predict_batch,
    resolve_keep_turn_indicator,
)
from scenario_generation.tensor_converter import _build_neighbor_agents_past
from scenario_generation.tools.eval_cl_trajectory import (
    border_segments_from_map,
    evaluate_trajectory,
)
from scenario_generation.transforms import _rotation_matrix

# Same default the reproducer path uses, so the strong_brake metric is comparable.
STRONG_BRAKE_MPS2 = -2.5

# Used only when the sim never reported an ego state, i.e. the scenario ended before the
# first tick and the trajectory log is empty, so the value cannot affect any metric.
_FALLBACK_EGO_BOX = (4.0, 1.8, 2.6)


@dataclass
class RolloutConfig:
    """Tuning for a single scenario_sim rollout."""

    # Must satisfy fps == 1 / DT: the sim step and the model timestep are the same tick.
    fps: float = 10.0
    # 1 = every tick = 10 Hz, matching the production node's planning_frequency_hz. Values
    # > 1 consume a cached plan open-loop in between: cheaper, less reactive.
    replan_interval: int = 1
    max_steps: int = 300
    # Ticks of real history required before a frame is scored. The buffer is seeded by
    # repeating the spawn pose, which is truthful -- the ego does start at rest -- so this is
    # not about waiting for fabricated rows to scroll out, but about how much of the
    # pull-away transient belongs in the measurement.
    warmup_steps: int = 5
    near_miss_thresh: float = 1.0
    find_route_min_len_m: float = 120.0
    # Coordinate-contract check: after the first stepped tick the ego's realized pose must
    # land within this tolerance (m) of the injected plan's first future point. A gross frame
    # mismatch blows past it.
    coord_check_tol_m: float = 2.0
    # The position check alone cannot see a rotated plan: after one tick both the realized
    # pose and the plan's first point sit within v*dt of the spawn pose, so any heading
    # convention error stays inside the metre tolerance at ordinary speeds and vanishes
    # entirely from rest. Heading is compared separately because it is not speed-scaled.
    coord_check_tol_rad: float = 0.1
    # A PNG every N ticks; None renders nothing. Metrics do not depend on it.
    draw_every: int | None = None
    scene: SceneConfig = field(default_factory=SceneConfig)

    def __post_init__(self) -> None:
        # DT drives the plan-point speeds, SceneContext.dt and the acceleration first
        # difference, while fps drives the simulator. A mismatch produces a complete row with
        # silently wrong speeds and brake counts, so it has to fail here.
        if abs(self.fps * DT - 1.0) > 1e-9:
            raise ValueError(f"fps={self.fps} does not match the model timestep DT={DT}")


@torch.no_grad()
def _predict_ego_plan(model, model_args, scene, device, ego_name: str) -> tuple[np.ndarray, int]:
    """Run the model as ego -> (ego-frame plan ``(future_len, 4)``, turn-indicator class).

    No ``map_cache`` is passed: the cache only pays off across steps that share one
    ``map_data``, and this loop rebuilds it around the ego every tick.
    """
    preds, tis = _predict_batch(
        model,
        model_args,
        scene,
        [ego_name],
        device,
        return_turn_indicators=True,
    )
    # A model without a turn-indicator head returns no class at all. Falling back to 0 puts
    # NO_COMMAND into a history that is in report space, where 0 is not a value, and
    # resolve_keep_turn_indicator carries it forward from then on.
    return preds[ego_name], int(tis.get(ego_name, TURN_INDICATOR_DISABLE))


def _ego_plan_to_map_trajectory(
    plan_ego: np.ndarray, ex: float, ey: float, eh: float
) -> np.ndarray:
    """Ego-frame plan -> map-frame ``[N, 4]`` of (x, y, yaw, longitudinal v) for
    ``set_ego_trajectory``, using the current ego pose as the frame origin.

    Each speed is the step onto its own point, so the first one is measured from the ego
    rather than from the point after it -- the tracker locates itself at the start of a tick
    and consumes that first value, so getting it from the wrong pair of points offsets the
    commanded speed by roughly a quarter of the plan's acceleration.
    """
    world_xy, world_h = _ego_to_world(
        plan_ego[:, :2], plan_ego[:, 2:4], ex, ey, eh, dtype=np.float64
    )
    seg = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
    first = math.hypot(world_xy[0, 0] - ex, world_xy[0, 1] - ey)
    speeds = np.concatenate([[first], seg]) / DT
    return np.column_stack([world_xy, world_h, speeds])


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _score_neighbors(scene, ego_state: dict, device: str, ego_name: str) -> tuple[float, bool]:
    """Instantaneous (min_clearance, collision) from raw ego-frame neighbour OBBs."""
    ex, ey, eh = baselink_xyh(ego_state)
    R = _rotation_matrix(eh)
    neighbors_live = _build_neighbor_agents_past(
        scene, ego_name, R, np.array([ex, ey], dtype=np.float64), eh
    )[0, :, -1, :]
    # (box_wheelbase, length, width) -- metric geometry, so ego_metric_box and not entity_shape.
    ego_shape = np.array(ego_metric_box(ego_state), dtype=np.float32)[[2, 0, 1]]
    min_clr, coll, _ = score_object_step(neighbors_live, ego_shape, device)
    return min_clr, coll


def _traj_entry(step: int, ego_state: dict, goal_xy: np.ndarray) -> dict:
    """One trajectory_log row (world pose, speed, goal distance) for post-hoc metrics."""
    x, y, h = baselink_xyh(ego_state)
    tw = ego_state["twist"]
    return {
        "step": step,
        "x": x,
        "y": y,
        "heading": h,
        "speed": float(math.hypot(tw["linear_x"], tw["linear_y"])),
        "goal_d": float(math.hypot(x - goal_xy[0], y - goal_xy[1])),
    }


def _start_sim(runner, osc_path: str | Path) -> None:
    """Bring the scenario up to "active", which is when it has a map.

    A scenario the interpreter rejects at parse time leaves ``configure()`` at "unconfigured",
    and calling ``get_entity_states()`` on an unconfigured core dereferences a null pointer, so
    a bad lifecycle transition has to fail here rather than be carried into the loop.

    Activation is what builds the simulator's configuration -- and with it resolves and loads
    the map -- so nothing may ask which map is running until this returns.
    """
    st_cfg = runner.configure()
    if st_cfg != "inactive":
        raise RuntimeError(
            f"configure() did not reach 'inactive' (got '{st_cfg}') -- scenario "
            f"rejected by the interpreter at parse/configure time: {osc_path}"
        )
    st_act = runner.activate()
    if st_act != "active":
        raise RuntimeError(f"activate() did not reach 'active' (got '{st_act}'): {osc_path}")


def _resolve_route_for_ego(
    runner, builder: LaneletSceneBuilder, osc_path: str | Path, cfg: RolloutConfig, verbose: bool
) -> tuple[str, list[int], np.ndarray]:
    """The ego's name and route, off the entities the running scenario reports."""
    ego_name = resolve_ego_name(runner.get_entity_states())

    x0, y0, h0 = baselink_xyh(runner.get_ego_state(ego_ref=ego_name))
    ego0_xy = np.array([x0, y0], dtype=np.float32)
    ego_route_ids, goal_pose = resolve_route(
        builder, ego0_xy, h0, osc_path, ego_name, min_len_m=cfg.find_route_min_len_m
    )
    if not ego_route_ids:
        raise RuntimeError("Empty ego route -- cannot build SceneContext")
    if verbose:
        print(
            f"  [scenario_sim] route={len(ego_route_ids)} lanelets, "
            f"start_ll={ego_route_ids[0]}, goal_ll={ego_route_ids[-1]}"
        )
    return ego_name, ego_route_ids, goal_pose


def _scored_accels(speeds: np.ndarray, start: int, n: int) -> np.ndarray:
    """Realized acceleration over the ``n`` scored frames from ``start``.

    Differenced from the sample before the window when there is one, so every returned value
    is a real first difference. Only a window starting at tick 0 has to open with a zero.
    """
    sp = np.asarray(speeds, dtype=np.float64)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if start > 0:
        return np.diff(sp[start - 1 : start + n]) / DT
    accels = np.zeros(n, dtype=np.float64)
    accels[1:] = np.diff(sp[:n]) / DT
    return accels


def _write_rollout_trace(
    output_dir: Path,
    *,
    trajectory_log: list[dict],
    clearances: list[float],
    collisions: list[bool],
    rb_dists: np.ndarray,
    scored_from: int,
    terminated_reason: str,
) -> None:
    """Write ``rollout.jsonl`` next to the PNGs: one line per sim step, plus a terminated line.

    The schema is the reproducer's, because the readers are shared: ``trajectory_colormap``
    takes ``ego`` (required) plus ``speed`` / ``clearance_m`` / ``collision`` / ``rb_dist_m``,
    and skips any line carrying an ``event`` key. Road-border distance only exists after the
    rollout -- it is computed over the whole trajectory at once -- so the trace is written here
    rather than streamed, which is the one place this path differs from the reproducer's.

    A quantity this path never observed is left out entirely rather than written as its safe
    value: the readers default a missing key to "no event", which a colormap would paint as a
    measurement that was taken and found nothing. ``red_light_violation`` is always absent
    (traffic lights are not read), and ``rb_dist_m`` is absent on a map with no road borders.
    """
    # Scoring runs every tick from ``scored_from`` to the last one logged. A trace whose samples
    # were off by a tick would still be a complete, plausible picture, so the alignment the
    # padding below relies on is asserted rather than assumed.
    if scored_from + len(clearances) != len(trajectory_log):
        raise ValueError(
            f"clearance series does not span ticks {scored_from}..{len(trajectory_log)}: "
            f"got {len(clearances)} samples"
        )
    # The row's blocks are counted from ``scored_from`` on, but the trace colours the whole
    # driven path, so the metric series is head-padded back to tick 0 instead. An unscored
    # warmup tick has no clearance rather than a clearance of zero.
    clearances = [float("nan")] * scored_from + list(clearances)
    collisions = [False] * scored_from + list(collisions)
    have_rb = len(rb_dists) == len(trajectory_log)
    with (output_dir / "rollout.jsonl").open("w", encoding="utf-8") as f:
        for k, entry in enumerate(trajectory_log):
            clr = clearances[k]
            line = {
                "k": k,
                "ego": [round(float(entry["x"]), 3), round(float(entry["y"]), 3)],
                "yaw": round(float(entry["heading"]), 4),
                "dist_goal": round(float(entry["goal_d"]), 3),
                "speed": round(float(entry["speed"]), 3),
                "clearance_m": round(float(clr), 4) if np.isfinite(clr) else None,
                "collision": bool(collisions[k]),
            }
            if have_rb:
                rb = float(rb_dists[k])
                line["rb_dist_m"] = round(rb, 4) if np.isfinite(rb) else None
            f.write(json.dumps(line) + "\n")
        f.write(
            json.dumps(
                {"event": "terminated", "k": len(trajectory_log), "reason": terminated_reason}
            )
            + "\n"
        )


def _finalize_row(
    output_dir: Path,
    *,
    trajectory_log: list[dict],
    ego_state: dict | None,
    cfg: RolloutConfig,
    clearances: list[float],
    collisions: list[bool],
    scored_from: int | None,
    terminated_reason: str,
    result_kind: str,
    coord_err: float,
    yaw_err: float,
    borders: list[np.ndarray],
) -> dict:
    """Dump the trajectory, compute post-hoc road-border metrics and build the row.

    Clearance and collision are only observed once the ego has accumulated ``warmup_steps`` of
    real history, so the pull-away from rest does not dominate what is measured. The
    road-border and braking series come off the whole
    trajectory, so they are trimmed to the same window here -- otherwise the row's blocks
    would count over denominators that differ by the warmup, and strong_brake would score the
    pull-away from rest.

    Acceleration is differenced from one sample before the window, so the first scored frame
    carries a real first difference instead of a zero standing in for one.
    """
    (output_dir / "trajectory_log.json").write_text(json.dumps(trajectory_log))

    ego_len, ego_w, ego_wb = (
        ego_metric_box(ego_state) if ego_state is not None else _FALLBACK_EGO_BOX
    )
    rb, series = evaluate_trajectory(trajectory_log, borders, ego_len, ego_w, ego_wb)
    # None when the run ended before the history was warm: nothing was scored, so every block
    # is built from an empty series rather than from frames the object block never saw.
    start = len(trajectory_log) if scored_from is None else scored_from
    stop = start + len(clearances)
    # The trace is per-tick over the whole run, so it takes the untrimmed series and the tick
    # the trimming starts at, not the trimmed arrays the row's blocks are built from.
    _write_rollout_trace(
        output_dir,
        trajectory_log=trajectory_log,
        clearances=clearances,
        collisions=collisions,
        rb_dists=series["rb_dists"],
        scored_from=start,
        terminated_reason=terminated_reason,
    )

    row = build_segment_row(
        n_steps_run=len(clearances),
        terminated=terminated_reason,
        result_kind=result_kind,
        clearances=clearances,
        collisions=collisions,
        rb_dists=series["rb_dists"][start:stop],
        accels=_scored_accels(series["speeds"], start, len(clearances)),
        near_miss_thresh=cfg.near_miss_thresh,
        strong_brake_mps2=STRONG_BRAKE_MPS2,
        progress_m=rb["progress_m"],
    )
    # scenario_sim-only diagnostics, kept flat (outside the shared category blocks) so
    # aggregate never sees them as a metric category.
    return {
        **row,
        # A sim tick, not an index into the trimmed series, so it names a frame of the run.
        "worst_step": int(start + np.argmin(clearances)) if clearances else -1,
        "n_ticks_run": len(trajectory_log),
        "rb_has_data": rb["rb_has_data"],
        "coord_check_ok": bool(
            coord_err <= cfg.coord_check_tol_m and yaw_err <= cfg.coord_check_tol_rad
        ),
        "coord_check_err_m": coord_err,
        "coord_check_yaw_err_rad": yaw_err,
    }


def run_scenario_sim_rollout(
    model,
    model_args,
    osc_path: str | Path,
    output_dir: str | Path,
    map_path: str | Path | None = None,
    *,
    config: RolloutConfig | None = None,
    device: str = "cpu",
    verbose: bool = True,
    timers: Timers | None = None,
    builder: LaneletSceneBuilder | None = None,
    draw_pool: Executor | None = None,
) -> dict:
    """Run one closed-loop OpenSCENARIO rollout and return an aggregate-ready row.

    ``model`` / ``model_args`` follow the ``run_closed_loop_eval`` contract
    (``model(data) -> (_, outputs)`` with ``outputs["prediction"]``; ``model_args`` provides
    ``observation_normalizer`` / ``predicted_neighbor_num`` / ``future_len``). ``builder`` lets
    a caller that outlives one scenario reuse a parsed map, which is per-map work a
    per-scenario process would otherwise pay per scenario.
    """
    # Participants in one DDS domain all discover each other, so processes sharing a domain cost
    # N^2 of discovery. 101 is the last domain whose RTPS base ports clear Linux's ephemeral
    # range; setdefault, because a caller that knows its slot assigns better than pid modulo.
    os.environ.setdefault("ROS_DOMAIN_ID", str(os.getpid() % 101))

    import openscenario_python as osp  # requires the SSV2_HEADLESS_EGO overlay

    cfg = config or RolloutConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timers = timers or Timers()
    _t_rollout = time.perf_counter()
    buffers = HistoryBuffers()
    trajectory_log: list[dict] = []
    clearances: list[float] = []
    collisions: list[bool] = []
    # Tick the object series starts at. Every other series is trimmed to it so all the row's
    # blocks are counted over the same frames.
    scored_from: int | None = None

    cached_plan_ego: np.ndarray | None = None  # (future_len, 4) ego frame
    # The map-frame plan the sim is tracking, outliving the replan that set it.
    pts: np.ndarray | None = None
    # The planner is the only source of this signal -- the simulator relays what it is given --
    # so the history the model reads is the one resolved here, held between replans.
    ti_report = TURN_INDICATOR_DISABLE
    ti_hist = deque([ti_report] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
    # NaN until the first stepped tick; every comparison against the tolerance is then False,
    # so a rollout that never stepped reports the check as failed rather than as passed.
    coord_err: float = float("nan")
    yaw_err: float = float("nan")
    ego_state: dict | None = None  # last tick's ego truth, for the finalize ego shape
    terminated_reason = "max_steps"

    # The interpreter's JUnit result is the only place the reason a scenario was rejected
    # survives -- on_configure reports failure as a lifecycle state and nothing else. write_to()
    # silently does nothing when its directory is missing, so it has to exist before the run.
    osp_out = output_dir / "osp_out"
    osp_out.mkdir(parents=True, exist_ok=True)

    # Gated: the renderer pulls in matplotlib and the whole replay module.
    save_step_figure = None
    if cfg.draw_every:
        from scenario_generation.replay import save_step_figure

    frames: list = []

    # Measured across the ``with`` header: opening the sim is the constructor plus __enter__,
    # which no context manager of ours can wrap.
    _t = time.perf_counter()
    # A pool passed in outlives the scenario, so a caller running many pays the spawn once. The
    # fallback pool starts its process on the first submitted frame, so drawing off spawns none.
    with (
        osp.HeadlessRunner(
            osc_path=str(osc_path),
            output_directory=str(osp_out),
            local_frame_rate=cfg.fps,
            # The DT invariant is over the simulator's integration step, which is
            # real_time_factor / frame_rate -- not frame_rate alone. Passed explicitly so the
            # invariant does not rest on a default defined on the other side of the binding.
            local_real_time_factor=1.0,
        ) as runner,
        nullcontext(draw_pool) if draw_pool is not None else render_pool(1) as pool,
    ):
        draw = pool if save_step_figure is not None else None
        timers.add("sim_open", time.perf_counter() - _t)
        with timers("sim_start"):
            _start_sim(runner, osc_path)
        # Asked of the interpreter rather than re-derived from the scenario: it has already
        # applied its own attribute reader to RoadNetwork/LogicFile and chosen a file for a
        # directory path, and any second implementation of that is a way for the two ends to
        # load different maps. Pass ``map_path`` only to test a substitute.
        if map_path is None:
            map_path = runner.lanelet2_map_path()
        with timers("map_build"):
            if builder is None:
                builder = LaneletSceneBuilder(str(map_path))
            elif Path(builder.lanelet_path).resolve() != Path(map_path).resolve():
                # Route, centreline and road-border geometry all come off the builder's map. A
                # builder carried over from another scenario would compute every one of them
                # against a map the interpreter did not load.
                raise ValueError(
                    f"builder holds {builder.lanelet_path}, but this scenario loaded {map_path}"
                )
        # Derived once: the drawing reads it every tick and the road-border metrics read it again.
        borders = border_segments_from_map(builder._lanelet_map)
        with timers("route_resolve"):
            ego_name, ego_route_ids, goal_pose = _resolve_route_for_ego(
                runner, builder, osc_path, cfg, verbose
            )
        goal_xy = goal_pose[:2]
        # Already in the scene's map frame. A polyline because save_step_figure accepts a
        # lanelet id list but does not read it.
        route_polylines = (
            [
                builder._cache[ll_id].raw_centerline[:, :2]
                for ll_id in ego_route_ids
                if ll_id in builder._cache
            ]
            if cfg.draw_every
            else None
        )

        for step in range(cfg.max_steps):
            with timers("sim_get_states"):
                states = runner.get_entity_states()
            if ego_name not in states:
                raise RuntimeError(f"Sim stopped reporting the ego entity '{ego_name}'")
            ego_state = states[ego_name]
            ex, ey, eh = baselink_xyh(ego_state)
            # Logged before stepping, so row k is the state clearance and collision are
            # measured on. Recording the post-step pose instead pairs every object sample
            # with a road-border and speed sample one tick later.
            trajectory_log.append(_traj_entry(step, ego_state, goal_xy))

            with timers("scene_build"):
                update_history(buffers, states, ego_name)
                ti_hist.append(ti_report)
                scene = build_scene(
                    states,
                    buffers,
                    builder,
                    ego_route_ids,
                    goal_pose,
                    cfg.scene,
                    ego_name,
                    np.fromiter(ti_hist, dtype=np.int32, count=_HISTORY_LEN),
                )

            # Replan every ``replan_interval`` ticks; consume the cached plan in between.
            if cached_plan_ego is None or step % cfg.replan_interval == 0:
                # The first inference is cold (lazy allocation, kernel autotuning), so it is
                # timed as its own stage and kept out of the steady-state ms/call.
                with timers("predict_cold" if cached_plan_ego is None else "predict"):
                    cached_plan_ego, ti_model = _predict_ego_plan(
                        model, model_args, scene, device, ego_name
                    )
                ti_report = resolve_keep_turn_indicator(ti_model, ti_report)

                with timers("sim_set_traj"):
                    # Anchored once, at the pose it was planned from. The tracker locates
                    # itself on the trajectory by closest point, so it advances along a
                    # map-frame plan on its own; re-anchoring on a cached tick would carry
                    # the whole plan along with the ego and it would never make progress.
                    pts = _ego_plan_to_map_trajectory(cached_plan_ego, ex, ey, eh)
                    runner.set_ego_trajectory(pts, ego_ref=ego_name)
                    runner.set_ego_turn_indicator(int(ti_report), ego_ref=ego_name)

            if buffers.age[ego_name] >= cfg.warmup_steps:
                if scored_from is None:
                    scored_from = step
                with timers("score_objects"):
                    clr, coll = _score_neighbors(scene, ego_state, device, ego_name)
                clearances.append(clr)
                collisions.append(coll)

            if draw is not None and pts is not None and step % cfg.draw_every == 0:
                with timers("draw_submit"):
                    # The plan the sim is tracking, viewed from the current pose rather than
                    # re-anchored there: cached_plan_ego would start every frame at the ego's
                    # nose and hide a plan it fails to progress along. Safe to hand to another
                    # process: both arguments are this tick's own objects.
                    frames.append(
                        draw.submit(
                            save_step_figure,
                            scene,
                            {ego_name: _world_plan_to_ego(pts[:, :2], pts[:, 2], ex, ey, eh)},
                            output_dir / f"{step:05d}.png",
                            step,
                            cfg.max_steps,
                            route_polylines=route_polylines,
                            road_border_polylines=borders,
                            sim_time=step * DT,
                        )
                    )

            # step() is the sole integrator: it advances BOTH the ego and the NPCs.
            with timers("sim_step"):
                outcome = runner.step()

            if step == 0:  # verify the frame contract on the first stepped tick
                ax, ay, ah = baselink_xyh(runner.get_ego_state(ego_ref=ego_name))
                coord_err = float(math.hypot(ax - pts[0, 0], ay - pts[0, 1]))
                yaw_err = abs(_wrap_pi(ah - pts[0, 2]))
                if verbose:
                    ok = coord_err <= cfg.coord_check_tol_m and yaw_err <= cfg.coord_check_tol_rad
                    print(
                        f"  [scenario_sim] coord check: err={coord_err:.3f} m "
                        f"(tol {cfg.coord_check_tol_m}), yaw={yaw_err:.3f} rad "
                        f"(tol {cfg.coord_check_tol_rad}) -> {'OK' if ok else 'FAIL'}"
                    )

            if outcome == "terminated":
                terminated_reason = "sim_terminated"
                break

        result_kind = runner.result_kind()

    # The trace and the encoder both read this directory; neither may see a partial sequence.
    with timers("draw_join"):
        for f in frames:
            f.result()

    with timers("finalize"):
        row = _finalize_row(
            output_dir,
            trajectory_log=trajectory_log,
            ego_state=ego_state,
            cfg=cfg,
            clearances=clearances,
            collisions=collisions,
            scored_from=scored_from,
            terminated_reason=terminated_reason,
            result_kind=result_kind,
            coord_err=coord_err,
            yaw_err=yaw_err,
            borders=borders,
        )
    # Teardown happens in HeadlessRunner.__exit__, so it is not a stage of its own: it lands in
    # rollout_total minus the parts.
    timers.add("rollout_total", time.perf_counter() - _t_rollout)
    row["map_path"] = str(map_path)
    row["timing"] = timers.as_dict()
    return row
