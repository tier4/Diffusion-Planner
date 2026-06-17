"""Closed-loop Perception-Reproducer rollout over one route segment, scored with
the canonical metrics OBB for collision / near-miss mining.

Per sim tick (10 Hz):

1. Cursor picks the recorded frame to reproduce (``PerceptionReproducer``).
2. That frame's baked model-input tensors (neighbors, lanes, route, polygons,
   line_strings, traffic, goal) are re-centered from the *recorded* ego onto the
   *live* ego with one rigid transform (``world_to_ego_frame``) — no lanelet map.
3. The live ego's own history / dynamics overwrite ego_agent_past + current.
4. The model predicts the ego trajectory; ``PerfectTracker`` advances the ego one
   step along it (perfect tracking).
5. The realized ego footprint is scored against the reproduced neighbors with the
   canonical OBB (``batch_signed_distance_rect`` / ``center_rect_to_points`` /
   ``_build_ego_bbox_corners``) — min clearance, collision, near-miss.

No rendering here (``--no_render`` is the mining default); every stage is timed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from diffusion_planner.dimensions import INPUT_T, MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.metrics.config import RewardConfig
from diffusion_planner.metrics.subscores import compute_static_collision_penalty

from scenario_generation.perception_reproducer import PerceptionReproducer
from scenario_generation.perf_timer import Timers
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.transforms import _rotation_matrix, world_to_ego_frame

DT = 0.1
PAST = INPUT_T + 1  # 31


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def _heading_to_cos_sin(h: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(h), np.sin(h)], axis=-1).astype(np.float32)


def _ego_pred_to_world(pred_xy, pred_cos_sin, ex, ey, eyaw):
    c, s = math.cos(eyaw), math.sin(eyaw)
    wx = ex + pred_xy[..., 0] * c - pred_xy[..., 1] * s
    wy = ey + pred_xy[..., 0] * s + pred_xy[..., 1] * c
    wh = np.arctan2(pred_cos_sin[..., 1], pred_cos_sin[..., 0]) + eyaw
    return np.stack([wx, wy], axis=-1).astype(np.float32), wh.astype(np.float32)


def _rel_pose(recorded_pose: np.ndarray, live_pose: np.ndarray) -> tuple[float, float, float]:
    """Live ego pose expressed in the recorded-ego frame (dx, dy, dyaw)."""
    R = _rotation_matrix(float(recorded_pose[2]))  # rotates world delta by -recorded_yaw
    d = R @ (live_pose[:2] - recorded_pose[:2])
    dyaw = float(live_pose[2] - recorded_pose[2])
    return float(d[0]), float(d[1]), dyaw


# --------------------------------------------------------------------------- #
# model input
# --------------------------------------------------------------------------- #
def _npz_to_model_base(npz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """NPZ training arrays -> batched [1,...] model-input dict (un-normalized).

    Ego heading + goal are widened to cos/sin (4-col) to match the model layout
    that ``world_to_ego_frame`` and the normalizer expect.
    """

    def b(a):
        return np.asarray(a)[None].astype(np.float32)

    ep = np.asarray(npz["ego_agent_past"]).astype(np.float32)
    if ep.shape[-1] == 3:
        ep = np.concatenate([ep[:, :2], _heading_to_cos_sin(ep[:, 2])], axis=-1)
    out = {
        "ego_agent_past": ep[None],
        "ego_current_state": b(npz["ego_current_state"].reshape(-1)[:10]),
        "neighbor_agents_past": b(npz["neighbor_agents_past"]),
        "lanes": b(npz["lanes"]),
        "lanes_speed_limit": b(npz["lanes_speed_limit"]),
        "lanes_has_speed_limit": np.asarray(npz["lanes_has_speed_limit"])[None].astype(bool),
        "route_lanes": b(npz["route_lanes"]),
        "route_lanes_speed_limit": b(npz["route_lanes_speed_limit"]),
        "route_lanes_has_speed_limit": np.asarray(npz["route_lanes_has_speed_limit"])[None].astype(
            bool
        ),
        "polygons": b(npz["polygons"]),
        "line_strings": b(npz["line_strings"]),
        "static_objects": b(npz["static_objects"]),
        "ego_shape": b(npz["ego_shape"].reshape(-1)[:3]),
        "turn_indicators": np.asarray(npz["turn_indicators"]).reshape(-1)[None].astype(np.int64),
    }
    goal = np.asarray(npz["goal_pose"]).reshape(-1).astype(np.float32)
    if goal.shape[0] == 3:
        goal = np.concatenate([goal[:2], _heading_to_cos_sin(goal[2:3]).reshape(2)])
    out["goal_pose"] = goal[None]
    return out


def _live_ego_past(ego_hist_world: np.ndarray, live_pose: np.ndarray) -> np.ndarray:
    """(1, PAST, 4) live ego history in the current live-ego frame [x,y,cos,sin]."""
    R = _rotation_matrix(float(live_pose[2]))
    exy = live_pose[:2]
    n = ego_hist_world.shape[0]
    out = np.zeros((PAST, 4), dtype=np.float32)
    for t in range(PAST):
        src = ego_hist_world[max(0, n - PAST + t)]
        d = R @ (src[:2] - exy)
        h = float(src[2] - live_pose[2])
        out[t] = [d[0], d[1], math.cos(h), math.sin(h)]
    return out[None]


@dataclass
class _EgoDyn:
    speed: float
    accel: float = 0.0
    yaw_rate: float = 0.0
    steering: float = 0.0


def _live_ego_current(dyn: _EgoDyn) -> np.ndarray:
    """(1,10) ego_current_state in live-ego frame: ego at origin, heading +x."""
    return np.array(
        [[0.0, 0.0, 1.0, 0.0, dyn.speed, 0.0, dyn.accel, 0.0, dyn.steering, dyn.yaw_rate]],
        dtype=np.float32,
    )


def build_input_np(
    tl: RouteTimeline,
    idx: int,
    live_pose: np.ndarray,
    ego_hist_world: np.ndarray,
    dyn: _EgoDyn,
) -> tuple[dict, np.ndarray]:
    """CPU-only per-segment input build (numpy, no torch/normalize).

    Returns (recentered [1,...] numpy model-input dict, neighbors_live (320,11) for
    scoring). This is the threadable half: ``np.load`` and ``world_to_ego_frame``
    are numpy/IO and release the GIL, so many segments build concurrently; the
    torch conversion + normalization happen once for the whole batch afterwards
    (see ``_to_torch_batch``).
    """
    base = _npz_to_model_base(tl.npz(idx))
    dx, dy, dyaw = _rel_pose(tl.poses[idx], live_pose)
    recen = world_to_ego_frame(base, dx, dy, dyaw)  # re-center recorded frame on live ego
    # Swap in the live ego's own history + dynamics (closed-loop truth).
    recen["ego_agent_past"] = _live_ego_past(ego_hist_world, live_pose)
    recen["ego_current_state"] = _live_ego_current(dyn)
    neighbors_live = recen["neighbor_agents_past"][0, :, -1, :].copy()  # (320,11) for scoring
    return recen, neighbors_live


def _to_torch_batch(np_dicts: list[dict], model_args, device: str) -> dict:
    """Concat N single-sample numpy dicts -> one batched, normalized torch dict.

    Does the work that used to be per-segment (host->device copy + normalization)
    ONCE for the whole batch: N concatenations, one H2D transfer per key, one
    normalizer call.
    """
    N = len(np_dicts)
    data = {}
    for k in np_dicts[0]:
        arr = np.concatenate([d[k] for d in np_dicts], axis=0)
        if k in ("lanes_has_speed_limit", "route_lanes_has_speed_limit"):
            data[k] = torch.from_numpy(arr).to(device)
        elif k == "turn_indicators":
            data[k] = torch.from_numpy(arr).long().to(device)
        else:
            data[k] = torch.from_numpy(arr.astype(np.float32)).to(device)
    data["delay"] = torch.zeros((N,), dtype=torch.long, device=device)
    data["sampled_trajectories"] = torch.zeros(
        (N, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM), dtype=torch.float32, device=device
    )
    return model_args.observation_normalizer(data)


# --------------------------------------------------------------------------- #
# scoring (canonical OBB)
# --------------------------------------------------------------------------- #
def score_step(
    neighbors_live: np.ndarray,
    ego_shape: np.ndarray,
    ego_speed: float,
    device: str,
    config: RewardConfig | None = None,
) -> tuple[float, bool, int]:
    """Min ego-neighbor clearance (m), collision flag, and #valid neighbors.

    Uses the SAME function the avoidance (RSFT) reward / eval use:
    ``compute_static_collision_penalty`` (metrics.subscores) — its
    ``per_timestep_min`` is the canonical sc OBB clearance (SAT penetration for
    overlap + closest-point Euclidean distance for separated boxes). We feed the
    instantaneous reproduced neighbor snapshot as a 2-frame "stopped" obstacle set
    (replicated, so v0=0 → stopped-mask passes) and read the t=0 clearance.

    neighbors_live: (320, 11) in live-ego frame [x,y,cos,sin,vx,vy,w,l,type...].
    """
    config = config or RewardConfig()
    valid = np.abs(neighbors_live[:, :6]).sum(axis=1) > 0
    if not valid.any():
        return float("inf"), False, 0
    nb = neighbors_live[valid]
    M = nb.shape[0]

    # Ego at origin, heading +x; 2 identical frames so ego_moving etc. are defined.
    ego_trajs = torch.zeros((1, 2, 4), dtype=torch.float32, device=device)
    ego_trajs[0, :, 2] = 1.0
    ego_shape_t = torch.tensor(ego_shape[:3], dtype=torch.float32, device=device)

    nb_xycs = torch.tensor(nb[:, :4], dtype=torch.float32, device=device)  # x,y,cos,sin
    neighbor_futures = nb_xycs[:, None, :].expand(M, 2, 4).contiguous()  # (M,2,4) static
    neighbor_shapes = torch.tensor(
        np.stack([nb[:, 6], nb[:, 7]], axis=-1), dtype=torch.float32, device=device
    )  # (M,2) [width, length]
    neighbor_valid = torch.ones((M, 2), dtype=torch.bool, device=device)

    out = compute_static_collision_penalty(
        ego_trajs, ego_shape_t, neighbor_futures, neighbor_shapes, neighbor_valid, config
    )
    clr = float(out["per_timestep_min"][0, 0])
    collision = bool((ego_speed > config.sc_ego_min_speed) and (clr < config.sc_cross_thresh))
    return clr, collision, M


# --------------------------------------------------------------------------- #
# rollout — per-segment state so many segments can run in lock-step on the GPU
# --------------------------------------------------------------------------- #
@dataclass
class SegmentResult:
    metrics: dict
    clearances: np.ndarray
    collisions: np.ndarray
    timers: Timers = field(default_factory=Timers)


@dataclass
class _SegState:
    tl: RouteTimeline
    start: int
    end: int
    near_miss_thresh: float
    warmup_steps: int
    goal_reach_m: float
    max_stuck_steps: int
    cursor: PerceptionReproducer
    tracker: object
    live_pose: np.ndarray
    ego_hist: np.ndarray
    dyn: _EgoDyn
    ego_shape: np.ndarray
    goal_xy: np.ndarray
    clearances: np.ndarray
    collisions: np.ndarray
    sim_time: float = 0.0
    stuck: int = 0
    prev_max_idx: int = 0
    terminated: str = "max_steps"
    k: int = 0
    done: bool = False
    max_steps: int = (
        0  # sim-step cap (decoupled from segment length so a slow ego can still finish)
    )


def _seed_state(
    tl,
    start,
    end,
    search_radius,
    warmup_steps,
    near_miss_thresh,
    goal_reach_m,
    max_stuck_steps,
    timers,
    max_steps=None,
) -> _SegState:
    from scenario_generation.mpc_tracker import PerfectTracker

    # Step cap: defaults to the segment length, but can exceed it so a slow ego
    # (e.g. one that waited out a long red light) can still drive to the segment end.
    cap = int(max_steps) if max_steps is not None else (end - start)

    cursor = PerceptionReproducer(tl, search_radius=search_radius, timers=timers)
    cursor.reset(start)
    live_pose = tl.poses[start].copy()
    npz0 = tl.npz(start)
    ep0 = np.asarray(npz0["ego_agent_past"]).astype(np.float32)  # (31,3) recorded-ego frame
    c0, s0 = math.cos(live_pose[2]), math.sin(live_pose[2])
    hist_xy = np.stack(
        [
            live_pose[0] + ep0[:, 0] * c0 - ep0[:, 1] * s0,
            live_pose[1] + ep0[:, 0] * s0 + ep0[:, 1] * c0,
        ],
        axis=-1,
    )
    ego_hist = np.column_stack([hist_xy, ep0[:, 2] + live_pose[2]]).astype(np.float64)
    ec0 = np.asarray(npz0["ego_current_state"]).reshape(-1)
    return _SegState(
        tl=tl,
        start=start,
        end=end,
        near_miss_thresh=near_miss_thresh,
        warmup_steps=warmup_steps,
        goal_reach_m=goal_reach_m,
        max_stuck_steps=max_stuck_steps,
        cursor=cursor,
        tracker=PerfectTracker(dt=DT),
        live_pose=live_pose,
        ego_hist=ego_hist,
        dyn=_EgoDyn(speed=float(np.hypot(ec0[4], ec0[5]))),
        ego_shape=np.asarray(npz0["ego_shape"]).reshape(-1)[:3].astype(np.float32),
        goal_xy=tl.poses[end - 1, :2],
        clearances=np.full(cap, np.inf, dtype=np.float32),
        collisions=np.zeros(cap, dtype=bool),
        prev_max_idx=cursor.max_idx_reached,
        max_steps=cap,
    )


def _pre_step(s: _SegState):
    """Advance the cursor + build this segment's NUMPY model input, or terminate it.

    Returns (np_dict, neighbors_live, idx) when the segment should run this tick, or
    None when it just terminated (s.done set, s.terminated explains why). CPU-only /
    threadable — torch conversion happens once per batch in the caller."""
    if s.done:
        return None
    if s.k >= s.max_steps:
        s.terminated, s.done = "max_steps", True
        return None
    if float(np.linalg.norm(s.live_pose[:2] - s.goal_xy)) < s.goal_reach_m:
        s.terminated, s.done = "goal", True
        return None
    idx = s.cursor.step(s.live_pose[:2], s.dyn.speed, s.sim_time)
    if s.cursor.max_idx_reached > s.prev_max_idx:
        s.prev_max_idx, s.stuck = s.cursor.max_idx_reached, 0
    else:
        s.stuck += 1
    if s.max_stuck_steps > 0 and s.stuck >= s.max_stuck_steps:
        s.terminated, s.done = "stuck", True
        return None
    np_dict, neighbors_live = build_input_np(s.tl, idx, s.live_pose, s.ego_hist, s.dyn)
    return np_dict, neighbors_live, idx


def _post_step(s: _SegState, pred: np.ndarray, neighbors_live, idx, device, timers):
    """Score this step and advance the ego (perfect tracking of the prediction)."""
    from scenario_generation.mpc_tracker import postprocess_reference

    with timers("score"):
        cl, col, _ = score_step(neighbors_live, s.ego_shape, s.dyn.speed, device)
        s.clearances[s.k] = cl
        s.collisions[s.k] = col
    with timers("advance"):
        if s.k < s.warmup_steps:
            tgt = min(idx + 1, len(s.tl) - 1)
            new_pose = s.tl.poses[tgt].copy()
            new_speed = float(s.tl.speeds[tgt])
        else:
            wxy, wh = _ego_pred_to_world(
                pred[:, :2], pred[:, 2:4], s.live_pose[0], s.live_pose[1], s.live_pose[2]
            )
            ref = postprocess_reference(wxy, wh, dt=DT)
            x0 = np.array(
                [s.live_pose[0], s.live_pose[1], s.live_pose[2], s.dyn.speed], dtype=np.float64
            )
            new_pos, new_speed = s.tracker.track(x0, ref)
            new_pose = np.asarray(new_pos, dtype=np.float64)
        prev_speed = s.dyn.speed
        s.dyn = _EgoDyn(
            speed=float(new_speed),
            accel=float((new_speed - prev_speed) / DT),
            yaw_rate=float(getattr(s.tracker, "last_yaw_rate", 0.0)),
            steering=float(getattr(s.tracker, "last_steering", 0.0)),
        )
        s.live_pose = new_pose
        s.ego_hist = np.vstack([s.ego_hist[1:], s.live_pose[None]])
        s.sim_time += DT
        s.k += 1


def _finalize(s: _SegState, timers: Timers) -> SegmentResult:
    valid_cl = s.clearances[: s.k][np.isfinite(s.clearances[: s.k])]
    progress = float(np.linalg.norm(s.live_pose[:2] - s.tl.poses[s.start, :2]))
    metrics = {
        "segment": [int(s.start), int(s.end)],
        "n_steps_run": int(s.k),
        "terminated": s.terminated,
        "min_clearance": float(valid_cl.min()) if valid_cl.size else float("inf"),
        "mean_clearance": float(valid_cl.mean()) if valid_cl.size else float("inf"),
        "n_collision_steps": int(s.collisions[: s.k].sum()),
        "n_near_miss_steps": int(np.sum(valid_cl <= s.near_miss_thresh)),
        "worst_step": int(
            np.argmin(np.where(np.isfinite(s.clearances[: s.k]), s.clearances[: s.k], np.inf))
        )
        if valid_cl.size
        else -1,
        "progress_m": progress,
    }
    return SegmentResult(
        metrics=metrics, clearances=s.clearances, collisions=s.collisions, timers=timers
    )


@torch.no_grad()
def run_segment(
    model,
    model_args,
    tl: RouteTimeline,
    start: int,
    end: int,
    device: str = "cuda",
    near_miss_thresh: float = 0.5,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    goal_reach_m: float = 5.0,
    max_stuck_steps: int = 100,
    timers: Timers | None = None,
) -> SegmentResult:
    """Single-segment closed-loop reproducer rollout over recorded frames [start, end)."""
    timers = timers or Timers()
    s = _seed_state(
        tl,
        start,
        end,
        search_radius,
        warmup_steps,
        near_miss_thresh,
        goal_reach_m,
        max_stuck_steps,
        timers,
    )
    while not s.done:
        with timers("input_build"):
            pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx = pre
        with timers("to_torch"):
            data = _to_torch_batch([np_dict], model_args, device)
        with timers("model_forward"):
            _, outputs = model(data)
            pred = outputs["prediction"][0, 0].cpu().numpy()
        _post_step(s, pred, neighbors_live, idx, device, timers)
    return _finalize(s, timers)


# --------------------------------------------------------------------------- #
# rendering (off the mining hot path) — draw the live-ego-frame scene per step
# --------------------------------------------------------------------------- #
def _draw_step(np_dict, neighbors_live, pred, ego_shape, near_miss_thresh, title, path):
    """Save a PNG of one reproducer step using the perfect-tracker sim viz style.

    Rebuilds a SceneContext (ego + reproduced neighbors + map) in the live-ego
    frame from the un-normalized input dict and renders it with the shared
    ``draw_scene`` (same look as scenario_generation.simulate), then overlays the
    ego plan (blue) and a red line on the closest ego-neighbor pair when within
    ``near_miss_thresh``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffusion_planner.metrics.geometry import (
        _build_ego_bbox_corners,
        _closest_points_between_rects,
    )
    from diffusion_planner.model.guidance.collision import center_rect_to_points

    from scenario_generation import npz_loader as nl
    from scenario_generation.scene_context import SceneContext
    from scenario_generation.visualize import draw_scene, draw_trajectory

    # Un-batch the [1,...] model-input dict and reuse the NPZ loader's extractors.
    data = {k: np.asarray(v)[0] for k, v in np_dict.items()}
    es = np.asarray(ego_shape).reshape(-1)
    ego = nl._extract_ego_agent(data, float(es[0]), float(es[1]), float(es[2]))
    scene = SceneContext(
        agents=[ego] + nl._extract_neighbors(data),
        map_data=nl._extract_map_data(data),
        ego_agent_id="ego",
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_scene(ax, scene, "ego")

    # Ego plan (model prediction), (T,3) [x, y, heading].
    plan = np.column_stack([pred[:, 0], pred[:, 1], np.arctan2(pred[:, 3], pred[:, 2])])
    draw_trajectory(ax, plan, "#3366cc", label="plan", lw=2, zorder=25)

    # Closest ego-neighbor pair → red line + min clearance in the title.
    et = torch.zeros((1, 1, 4))
    et[0, 0, 2] = 1.0
    ego_c = _build_ego_bbox_corners(et, torch.tensor(es[:3]))[:, :1].reshape(1, 4, 2)
    valid = np.abs(neighbors_live[:, :6]).sum(1) > 0
    nb = neighbors_live[valid]
    min_clr = float("inf")
    if len(nb):
        rects = torch.tensor(
            np.stack([nb[:, 0], nb[:, 1], nb[:, 2], nb[:, 3], nb[:, 7], nb[:, 6]], axis=-1),
            dtype=torch.float32,
        )
        corns = center_rect_to_points(rects)  # (M,4,2)
        p1, p2 = _closest_points_between_rects(ego_c.expand(len(nb), 4, 2), corns)
        d = (p1 - p2).norm(dim=-1)
        j = int(d.argmin())
        min_clr = float(d[j])
        if min_clr <= near_miss_thresh:
            ax.plot(
                [float(p1[j, 0]), float(p2[j, 0])],
                [float(p1[j, 1]), float(p2[j, 1])],
                "-",
                color="red",
                lw=2,
                zorder=30,
            )

    ax.set_aspect("equal")
    ax.set_xlim(-25, 45)
    ax.set_ylim(-30, 30)
    ax.set_title(f"{title}  min_clr={min_clr:.2f}m", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


@torch.no_grad()
def render_segment(
    model,
    model_args,
    tl: RouteTimeline,
    start: int,
    end: int,
    out_dir,
    device: str = "cuda",
    near_miss_thresh: float = 0.5,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    window: tuple[int, int] | None = None,
    max_steps: int | None = None,
    goal_reach_m: float = 5.0,
    max_stuck_steps: int = 0,
) -> dict:
    """Re-run one segment with per-step PNG rendering (live-ego frame).

    ``window`` = (lo, hi) step range to render (default: all). ``max_steps``
    overrides the step cap so a slow ego can drive to the segment end (it stops
    on goal-reach). ``max_stuck_steps``=0 (default) disables the stuck cutoff so
    the full run is rendered. Returns the SegmentResult metrics.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timers = Timers()
    s = _seed_state(
        tl,
        start,
        end,
        search_radius,
        warmup_steps,
        near_miss_thresh,
        goal_reach_m,
        max_stuck_steps,
        timers,
        max_steps=max_steps,
    )
    route = out_dir.name
    while not s.done:
        k = s.k
        pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx = pre
        data = _to_torch_batch([np_dict], model_args, device)
        _, outputs = model(data)
        pred = outputs["prediction"][0, 0].cpu().numpy()
        if window is None or (window[0] <= k <= window[1]):
            cl, col, _ = score_step(neighbors_live, s.ego_shape, s.dyn.speed, device)
            tag = "COLLISION" if col else ""
            _draw_step(
                np_dict,
                neighbors_live,
                pred,
                s.ego_shape,
                near_miss_thresh,
                f"{route} step {k:04d} rec={idx} {tag}",
                out_dir / f"{k:05d}.png",
            )
        _post_step(s, pred, neighbors_live, idx, device, timers)
    return _finalize(s, timers).metrics


@torch.no_grad()
def run_segments_batched(
    model,
    model_args,
    work_units: list[tuple],
    device: str = "cuda",
    batch_size: int = 16,
    near_miss_thresh: float = 0.5,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    goal_reach_m: float = 5.0,
    max_stuck_steps: int = 100,
    n_build_threads: int = 8,
    timers: Timers | None = None,
) -> list[SegmentResult]:
    """Run many segments in lock-step: ONE batched model forward per tick.

    work_units: list of (RouteTimeline, start, end). Processed in chunks of
    ``batch_size`` (bound GPU memory). Segments in a chunk terminate raggedly
    (goal/stuck/max); finished ones drop out while the rest continue.

    Two amortizations per tick: (1) the per-segment NUMPY input build (np.load +
    world_to_ego_frame, GIL-releasing) runs across ``n_build_threads`` threads;
    (2) the torch conversion + normalization + model.forward run ONCE on the
    stacked batch. So both the CPU build and the GPU forward are shared by the
    whole active set instead of paid per segment.
    """
    from concurrent.futures import ThreadPoolExecutor

    timers = timers or Timers()
    results: list[SegmentResult] = []
    pool = ThreadPoolExecutor(max_workers=max(1, n_build_threads))
    try:
        for c0 in range(0, len(work_units), batch_size):
            chunk = work_units[c0 : c0 + batch_size]
            states = [
                _seed_state(
                    tl,
                    start,
                    end,
                    search_radius,
                    warmup_steps,
                    near_miss_thresh,
                    goal_reach_m,
                    max_stuck_steps,
                    timers,
                )
                for (tl, start, end) in chunk
            ]
            active = list(states)
            while active:
                with timers("input_build"):
                    pre_list = list(pool.map(_pre_step, active))
                built = [(s, *pre) for s, pre in zip(active, pre_list) if pre is not None]
                if built:
                    with timers("to_torch"):
                        data = _to_torch_batch([b[1] for b in built], model_args, device)
                    with timers("model_forward"):
                        _, outputs = model(data)
                        preds = outputs["prediction"][:, 0].cpu().numpy()  # (B,80,4)
                    for i, (s, _np, nb, idx) in enumerate(built):
                        _post_step(s, preds[i], nb, idx, device, timers)
                active = [s for s in active if not s.done]
            results.extend(_finalize(s, timers) for s in states)
    finally:
        pool.shutdown(wait=True)
    return results
