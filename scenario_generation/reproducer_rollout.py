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
from diffusion_planner.dimensions import INPUT_T, POSE_DIM
from diffusion_planner.model.guidance.collision import (
    batch_signed_distance_rect,
    center_rect_to_points,
)

from planner_metrics.geometry import (
    _build_ego_bbox_corners,
    _closest_points_between_rects,
)
from scenario_generation.perception_reproducer import PerceptionReproducer
from scenario_generation.perf_timer import Timers
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.tensor_converter import _heading_to_cos_sin
from scenario_generation.transforms import _rotation_matrix, world_to_ego_frame

DT = 0.1
PAST = INPUT_T + 1  # 31


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
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
    # Match to_model_tensors: P = 1 ego + predicted neighbors, T = future_len (+1).
    n_agents = 1 + model_args.predicted_neighbor_num
    data["sampled_trajectories"] = torch.zeros(
        (N, n_agents, model_args.future_len + 1, POSE_DIM), dtype=torch.float32, device=device
    )
    return model_args.observation_normalizer(data)


# --------------------------------------------------------------------------- #
# scoring (canonical OBB)
# --------------------------------------------------------------------------- #
def _ego_neighbor_obb(neighbors_live: np.ndarray, ego_shape: np.ndarray, device: str):
    """Build ego corners (at origin) + valid-neighbor corners; return (ego_b, npc_corners, M).

    Canonical OBB geometry (``_build_ego_bbox_corners`` + ``center_rect_to_points``).
    Returns (None, None, 0) if there are no valid neighbors.
    """
    valid = np.abs(neighbors_live[:, :6]).sum(axis=1) > 0
    if not valid.any():
        return None, None, 0
    nb = neighbors_live[valid]
    M = nb.shape[0]
    et = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
    et[0, 0, 2] = 1.0
    ego_c = _build_ego_bbox_corners(
        et, torch.tensor(ego_shape[:3], dtype=torch.float32, device=device)
    )[:, :1].reshape(1, 4, 2)
    rects = torch.tensor(
        np.stack([nb[:, 0], nb[:, 1], nb[:, 2], nb[:, 3], nb[:, 7], nb[:, 6]], axis=-1),
        dtype=torch.float32,
        device=device,
    )  # x, y, cos, sin, length, width
    return ego_c.expand(M, 4, 2), center_rect_to_points(rects), M


def score_step(
    neighbors_live: np.ndarray,
    ego_shape: np.ndarray,
    ego_speed: float,
    device: str,
) -> tuple[float, bool, int]:
    """Min ego-neighbor clearance (m), collision flag, and #valid neighbors.

    RAW oriented-bounding-box check against EVERY valid neighbor — moving and
    static alike, with NO direction/rear-end or ego-speed gating. Collision =
    the ego box overlaps any neighbor box (canonical ``batch_signed_distance_rect``
    < 0); clearance = exact closest-point distance to the nearest neighbor
    (``_closest_points_between_rects``). This deliberately differs from the
    avoidance reward's ``compute_static_collision_penalty``, which only scores
    *stopped* neighbors and filters out rear-end hits — for mining we want to
    catch collisions with moving neighbors AND the ego being struck from behind.

    neighbors_live: (320, 11) in live-ego frame [x,y,cos,sin,vx,vy,w,l,type...].
    ``ego_speed`` is unused (kept for signature stability); collisions are counted
    regardless of ego speed.
    """
    ego_b, npc_corners, M = _ego_neighbor_obb(neighbors_live, ego_shape, device)
    if M == 0:
        return float("inf"), False, 0
    p1, p2 = _closest_points_between_rects(ego_b, npc_corners)
    clr = (p1 - p2).norm(dim=-1)  # exact closest-point distance per neighbor
    signed = batch_signed_distance_rect(ego_b, npc_corners)  # < 0 => overlap
    return float(clr.min()), bool((signed < 0).any()), M


def score_step_batched(
    neighbors_list: list[np.ndarray],
    ego_shapes: list[np.ndarray],
    device: str,
) -> list[tuple[float, bool, int]]:
    """``score_step`` for many segments at once: ONE batched OBB pass over all pairs.

    The OBB primitives (``_closest_points_between_rects`` / ``batch_signed_distance_rect``)
    are per-pair independent, so we concatenate every segment's (ego, neighbor) box
    pairs into one big batch, run the geometry once, then slice the result back per
    segment. Bit-identical to calling ``score_step`` per segment (no cross-segment
    interaction), but collapses N tiny GPU launches per tick into one — including the
    box construction: ONE host->device transfer + ONE ``center_rect_to_points`` for
    every neighbor across all segments (ego corners built once when shapes match, else
    per segment and repeat-interleaved). Returns a list aligned to the inputs:
    (min_clearance, collision, n_valid_neighbors)."""
    valids = [np.abs(nb[:, :6]).sum(axis=1) > 0 for nb in neighbors_list]
    counts = [int(v.sum()) for v in valids]
    if sum(counts) == 0:
        return [(float("inf"), False, 0) for _ in neighbors_list]

    # All valid neighbors across all segments -> one transfer -> one corner build.
    nb_all = np.concatenate(
        [nb[v] for nb, v in zip(neighbors_list, valids) if v.any()], axis=0
    )  # (K, 11)
    rects = torch.tensor(
        np.stack(
            [nb_all[:, 0], nb_all[:, 1], nb_all[:, 2], nb_all[:, 3], nb_all[:, 7], nb_all[:, 6]],
            axis=-1,
        ),
        dtype=torch.float32,
        device=device,
    )  # (K, 6) x, y, cos, sin, length, width
    npc_all = center_rect_to_points(rects)  # (K, 4, 2)

    # Ego box at origin (heading +x), one per segment, repeated per its neighbors.
    # K from the CPU list (avoids a per-tick GPU->CPU sync that int(counts_t.sum()) forces).
    K = sum(counts)
    et = torch.zeros((len(neighbors_list), 1, 4), dtype=torch.float32, device=device)
    et[:, 0, 2] = 1.0
    if all(np.array_equal(ego_shapes[0], sh) for sh in ego_shapes):
        ego1 = _build_ego_bbox_corners(
            et[:1], torch.tensor(ego_shapes[0][:3], dtype=torch.float32, device=device)
        ).reshape(1, 4, 2)
        ego_all = ego1.expand(K, 4, 2)
    else:
        ego_each = torch.stack(
            [
                _build_ego_bbox_corners(
                    et[i : i + 1], torch.tensor(sh[:3], dtype=torch.float32, device=device)
                ).reshape(4, 2)
                for i, sh in enumerate(ego_shapes)
            ],
            dim=0,
        )  # (B, 4, 2)
        counts_t = torch.tensor(counts, device=device)
        ego_all = ego_each.repeat_interleave(counts_t, dim=0)  # (K, 4, 2)

    p1, p2 = _closest_points_between_rects(ego_all, npc_all)
    clr_all = (p1 - p2).norm(dim=-1)  # (K,)
    signed_all = batch_signed_distance_rect(ego_all, npc_all)  # (K,)
    out: list[tuple[float, bool, int]] = []
    off = 0
    for m in counts:
        if m == 0:
            out.append((float("inf"), False, 0))
        else:
            out.append(
                (
                    float(clr_all[off : off + m].min()),
                    bool((signed_all[off : off + m] < 0).any()),
                    m,
                )
            )
            off += m
    return out


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
    # Unstick: when the ego makes no forward progress for ``unstick_after`` steps
    # (e.g. it stops at a yellow light and never proceeds), snap it forward to the
    # recorded GT ego pose ~``unstick_advance_m`` ahead instead of stalling forever.
    unstick_after: int = 0
    unstick_advance_m: float = 5.0
    ego_stuck: int = 0
    n_snaps: int = 0
    # One-pass collision-scene save (set when run_segments_batched gets save_dir).
    # save_buf rolls the last save_max_scenes+1 (k, idx, live_pose, np_dict) snapshots
    # (deep enough for the min-movement window extension); it is CLEARED on an unstick
    # teleport so a saved window never crosses the jump.
    save_buf: object = None
    saved_collision: bool = False
    last_snap_step: int | None = None
    save_out_dir: object = None


def _ego_state_from_frame(tl: RouteTimeline, idx: int) -> tuple[np.ndarray, np.ndarray, "_EgoDyn"]:
    """Build (live_pose, ego_hist (31,3 world), dyn) from recorded frame ``idx``.

    Reconstructs the live ego world pose + recent history + speed from the frame's
    recorded ego pose and ego_agent_past. Used to seed a segment and to snap the
    ego back onto the recorded GT pose when it gets stuck."""
    pose = tl.poses[idx].copy()
    ep = np.asarray(tl.npz(idx)["ego_agent_past"]).astype(np.float32)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    hist_xy = np.stack(
        [pose[0] + ep[:, 0] * c - ep[:, 1] * s, pose[1] + ep[:, 0] * s + ep[:, 1] * c],
        axis=-1,
    )
    ego_hist = np.column_stack([hist_xy, ep[:, 2] + pose[2]]).astype(np.float64)
    return pose, ego_hist, _EgoDyn(speed=float(tl.speeds[idx]))


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
    unstick_after=0,
    unstick_advance_m=5.0,
) -> _SegState:
    from scenario_generation.mpc_tracker import PerfectTracker

    # Step cap: defaults to the segment length, but can exceed it so a slow ego
    # (e.g. one that waited out a long red light) can still drive to the segment end.
    cap = int(max_steps) if max_steps is not None else (end - start)

    cursor = PerceptionReproducer(tl, search_radius=search_radius, timers=timers)
    cursor.reset(start)
    live_pose, ego_hist, dyn = _ego_state_from_frame(tl, start)
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
        dyn=dyn,
        ego_shape=np.asarray(tl.npz(start)["ego_shape"]).reshape(-1)[:3].astype(np.float32),
        goal_xy=tl.poses[end - 1, :2],
        clearances=np.full(cap, np.inf, dtype=np.float32),
        collisions=np.zeros(cap, dtype=bool),
        prev_max_idx=cursor.max_idx_reached,
        max_steps=cap,
        unstick_after=int(unstick_after),
        unstick_advance_m=float(unstick_advance_m),
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


def _score_into(s: _SegState, neighbors_live, device, timers):
    """Score this step's ego↔neighbor clearance/collision into the segment state."""
    with timers("score"):
        cl, col, _ = score_step(neighbors_live, s.ego_shape, s.dyn.speed, device)
        s.clearances[s.k] = cl
        s.collisions[s.k] = col


def _advance_step(s: _SegState, pred: np.ndarray, idx, device, timers):
    """Advance the ego one step (perfect tracking of the prediction) + unstick."""
    from scenario_generation.mpc_tracker import postprocess_reference

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

        # Unstick: if the ego has been (near-)stopped for too long (e.g. it halted
        # at a yellow light and won't proceed), snap it forward onto the recorded
        # GT ego pose ~unstick_advance_m ahead so the rollout continues.
        if s.unstick_after > 0:
            s.ego_stuck = s.ego_stuck + 1 if s.dyn.speed < 0.5 else 0
            if s.ego_stuck >= s.unstick_after:
                n = len(s.tl)
                tgt = min(max(s.cursor.max_idx_reached, 0) + 1, n - 1)
                while tgt < n - 1 and (
                    float(np.linalg.norm(s.tl.poses[tgt, :2] - s.live_pose[:2]))
                    < s.unstick_advance_m
                ):
                    tgt += 1
                s.live_pose, s.ego_hist, s.dyn = _ego_state_from_frame(s.tl, tgt)
                s.cursor.reset(tgt)
                s.prev_max_idx = s.cursor.max_idx_reached
                s.ego_stuck = 0
                s.n_snaps += 1


def _post_step(s: _SegState, pred: np.ndarray, neighbors_live, idx, device, timers):
    """Score this step and advance the ego (sequential path: run_segment + extractor)."""
    _score_into(s, neighbors_live, device, timers)
    _advance_step(s, pred, idx, device, timers)


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
        "n_snaps": int(s.n_snaps),
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
    max_stuck_steps: int = 0,
    max_steps: int | None = None,
    unstick_after: int = 300,
    unstick_advance_m: float = 5.0,
    timers: Timers | None = None,
) -> SegmentResult:
    """Single-segment closed-loop reproducer rollout over recorded frames [start, end).

    Unstick is on by default (snap the ego forward onto the recorded GT pose after
    ``unstick_after`` steps of no progress). The only timeout is the step cap
    ``max_steps`` (default 3*(end-start)); the hard stuck-cutoff is off.
    """
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
        max_steps=max_steps if max_steps is not None else 3 * (end - start),
        unstick_after=unstick_after,
        unstick_advance_m=unstick_advance_m,
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
def _build_neighbor_interp(tl: RouteTimeline, lo: int, hi: int, eps: float = 0.1) -> dict:
    """Per-track world-trajectory anchors for temporal interpolation.

    Scans recorded frames [lo, hi) and, for each track UUID, collects its world
    pose then keeps only the *fresh* samples (drops held/stale repeats — frames
    where the perception position didn't move > eps). Returns
    ``{uuid: (idx_arr, xy_arr (N,2), heading_arr (N,))}``. Querying between two
    anchors (``_interp_pose``) linearly interpolates across the held gaps, turning
    the freeze-then-jump stutter into smooth motion. Uses the sidecar track IDs to
    associate the same car across frames.
    """
    raw: dict[str, list] = {}
    for idx in range(lo, hi):
        ids = tl.neighbor_ids(idx)
        if not ids:
            continue
        pose = tl.poses[idx]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        nb = tl.npz(idx)["neighbor_agents_past"][:, -1]  # (320, 11) ego frame
        for slot in range(min(len(ids), nb.shape[0])):
            row = nb[slot]
            if np.abs(row[:6]).sum() == 0:
                continue
            wx = pose[0] + row[0] * c - row[1] * s
            wy = pose[1] + row[0] * s + row[1] * c
            wh = math.atan2(row[3], row[2]) + pose[2]
            raw.setdefault(ids[slot], []).append((idx, wx, wy, wh))
    interp: dict[str, tuple] = {}
    for u, lst in raw.items():
        kept = [lst[0]]
        for samp in lst[1:]:
            if math.hypot(samp[1] - kept[-1][1], samp[2] - kept[-1][2]) > eps:
                kept.append(samp)
        if kept[-1][0] != lst[-1][0]:
            kept.append(lst[-1])  # keep the final sample so interp reaches the end
        interp[u] = (
            np.array([k[0] for k in kept]),
            np.array([[k[1], k[2]] for k in kept], dtype=np.float64),
            np.unwrap(np.array([k[3] for k in kept])),
        )
    return interp


def _interp_pose(anchors: tuple, idx: int) -> tuple[float, float, float]:
    """Linear world pose (x, y, heading) at recorded-frame ``idx`` from fresh anchors."""
    idxs, xy, hd = anchors
    if idx <= idxs[0]:
        return float(xy[0, 0]), float(xy[0, 1]), float(hd[0])
    if idx >= idxs[-1]:
        return float(xy[-1, 0]), float(xy[-1, 1]), float(hd[-1])
    j = int(np.searchsorted(idxs, idx))  # idxs[j-1] <= idx <= idxs[j]
    i0, i1 = j - 1, j
    t = (idx - idxs[i0]) / (idxs[i1] - idxs[i0])
    return (
        float(xy[i0, 0] + t * (xy[i1, 0] - xy[i0, 0])),
        float(xy[i0, 1] + t * (xy[i1, 1] - xy[i0, 1])),
        float(hd[i0] + t * (hd[i1] - hd[i0])),
    )


def _apply_neighbor_interp(np_dict, neighbor_ids, live_pose, idx, interp):
    """Replace each neighbor's current pose with its interpolated world pose.

    Mutates ``np_dict`` neighbor current (x, y, cos, sin) in the live-ego frame.
    """
    nb = np_dict["neighbor_agents_past"][0]  # (320, 31, 11) live-ego frame
    ex, ey, eyaw = float(live_pose[0]), float(live_pose[1]), float(live_pose[2])
    c, s = math.cos(eyaw), math.sin(eyaw)
    for slot in range(min(nb.shape[0], len(neighbor_ids))):
        row = nb[slot, -1]
        if np.abs(row[:6]).sum() == 0:
            continue
        anchors = interp.get(neighbor_ids[slot])
        if anchors is None:
            continue
        wx, wy, wh = _interp_pose(anchors, idx)
        dxw, dyw = wx - ex, wy - ey  # world -> live-ego
        nb[slot, -1, 0] = dxw * c + dyw * s
        nb[slot, -1, 1] = -dxw * s + dyw * c
        lh = wh - eyaw
        nb[slot, -1, 2] = math.cos(lh)
        nb[slot, -1, 3] = math.sin(lh)


def _polylines_from_tensor(t: np.ndarray, border_only: bool = False) -> list[np.ndarray]:
    """Extract (P,2) xy polylines (live-ego frame) from a lane/line_string tensor."""
    out = []
    for seg in t:
        v = np.abs(seg[:, :2]).sum(1) > 0.1
        if v.sum() < 2:
            continue
        if border_only and seg[:, 3].max() <= 0.5:  # line_strings channel 3 = road border
            continue
        out.append(seg[v, :2].astype(np.float64))
    return out


def _draw_step(np_dict, pred, ego_shape, path, neighbor_ids=None, step=0, total=1):
    """Save a PNG of one reproducer step with the EXACT perfect-tracker sim renderer.

    Rebuilds a SceneContext (ego + reproduced neighbors + map) in the live-ego
    frame and calls ``replay.save_step_figure`` — the same function the route sim
    uses. That gives the fixed viewport + fixed tick spacing (no per-frame
    rescale), traffic-light-colored lanes, the road-border distance line, the
    ego↔nearest static-NPC (red) and moving-NPC (blue) distance lines, the ego
    plan overlay, and stable colors (it hashes ``agent.id``).

    ``neighbor_ids``: per-slot track UUIDs from the sidecar. When given, neighbor
    agents are renamed to their UUID so the sim's own ``_stable_color`` keeps one
    color per track across frames (vs the flickering distance-sorted slot colors).
    """
    from pathlib import Path

    from scenario_generation import npz_loader as nl
    from scenario_generation.replay import save_step_figure
    from scenario_generation.scene_context import SceneContext

    data = {k: np.asarray(v)[0] for k, v in np_dict.items()}
    es = np.asarray(ego_shape).reshape(-1)
    ego = nl._extract_ego_agent(data, float(es[0]), float(es[1]), float(es[2]))
    neighbors = nl._extract_neighbors(data)

    # Rename neighbors to their track UUID so save_step_figure's _stable_color
    # gives one stable color per track across frames.
    if neighbor_ids:
        for a in neighbors:
            slot = int(a.id.rsplit("_", 1)[1])
            if slot < len(neighbor_ids):
                a.id = f"nb_{str(neighbor_ids[slot])[:8]}"

    scene = SceneContext(
        agents=[ego] + neighbors, map_data=nl._extract_map_data(data), ego_agent_id="ego"
    )
    save_step_figure(
        scene,
        {"ego": pred},  # ego-frame (80,4) prediction -> drawn as the ego plan
        Path(path),
        step,
        total,
        route_polylines=_polylines_from_tensor(data["route_lanes"]),
        road_border_polylines=_polylines_from_tensor(data["line_strings"], border_only=True),
    )


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
    color_by_uuid: bool = True,
    unstick_after: int = 300,
    unstick_advance_m: float = 5.0,
    interpolate: bool = True,
) -> dict:
    """Re-run one segment with per-step PNG rendering (live-ego frame).

    Runs until the ego reaches the segment end (within ``goal_reach_m``) or the
    step cap (``max_steps``, default 3*(end-start) — the only timeout). Unstick is
    on: after ``unstick_after`` (~30 s) of no progress the ego is snapped onto the
    recorded GT pose ~``unstick_advance_m`` ahead.

    ``interpolate``: smooth stale recorded neighbor positions by linearly
    interpolating each track between its real detections (uses the sidecar track
    UUIDs) — removes the freeze-then-jump perception stutter. ``color_by_uuid``:
    stable per-track colors. ``window`` = (lo, hi) step range to render (all).
    Returns the SegmentResult metrics.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = max_steps if max_steps is not None else 3 * (end - start)
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
        max_steps=cap,
        unstick_after=unstick_after,
        unstick_advance_m=unstick_advance_m,
    )
    # Build per-track interpolation anchors over the frames this render visits.
    # The cursor maps sim steps to recorded frames in ~[start, end]; a small
    # margin covers any overrun without scanning the whole route.
    interp = _build_neighbor_interp(tl, start, min(end + 100, len(tl))) if interpolate else {}
    while not s.done:
        k = s.k
        pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx = pre
        data = _to_torch_batch([np_dict], model_args, device)
        _, outputs = model(data)
        pred = outputs["prediction"][0, 0].cpu().numpy()
        nids = tl.neighbor_ids(idx) if (color_by_uuid or interpolate) else None
        if interpolate and nids and interp:
            _apply_neighbor_interp(np_dict, nids, s.live_pose, idx, interp)
        if window is None or (window[0] <= k <= window[1]):
            _draw_step(
                np_dict,
                pred,
                s.ego_shape,
                out_dir / f"{k:05d}.png",
                neighbor_ids=nids if color_by_uuid else None,
                step=k,
                total=cap,
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
    max_stuck_steps: int = 0,
    unstick_after: int = 300,
    unstick_advance_m: float = 5.0,
    max_steps_mult: int = 3,
    n_build_threads: int = 8,
    prefetch_ahead: int = 2,
    timers: Timers | None = None,
    save_dir=None,
    save_pre_steps: int = 80,
    save_thresh: float | None = None,
    save_pre_arc_m: float = 1.0,
    save_max_scenes: int = 160,
    save_min_post_snap_frames: int = 30,
    route_keys: list[str] | None = None,
) -> list[SegmentResult]:
    """Run many segments in lock-step: ONE batched model forward per tick.

    work_units: list of (RouteTimeline, start, end). Processed in chunks of
    ``batch_size`` (bound GPU memory). Segments terminate raggedly (goal / step
    cap); finished ones drop out while the rest continue.

    ONE-PASS collision-scene save: when ``save_dir`` is given, each segment keeps a
    rolling buffer of its last ``save_pre_steps`` scene snapshots and, on the FIRST
    step within ``save_thresh`` m of a neighbor, dumps that window (+ manifest) to
    ``save_dir/<route>_<start>_<end>/``. The scenes come from THIS run — the same one
    that detected the collision — so they always match the hit (the legacy two-pass
    ``extract_collision_scenes`` re-ran the rollout, which is batch-sensitive and so
    could anchor a different/empty window). The buffer is cleared on an unstick
    teleport so a saved window never spans the jump. ``route_keys`` (aligned to
    ``work_units``) names the output dirs; if None it is derived from each timeline.

    Unstick is on by default (snap the ego forward onto the recorded GT pose after
    ``unstick_after`` steps of no progress), so a segment isn't bailed out at a
    yellow-light stall. The only timeout is the step cap = ``max_steps_mult`` *
    segment length (default 3x → 1800 for a 600-frame segment); the hard
    stuck-cutoff is off.

    Two amortizations per tick: (1) the per-segment NUMPY input build (np.load +
    world_to_ego_frame, GIL-releasing) runs across ``n_build_threads`` threads;
    (2) the torch conversion + normalization + model.forward run ONCE on the
    stacked batch. (3) I/O overlap: while the GPU runs the forward (CPU otherwise
    idle), background threads prefetch the next ``prefetch_ahead`` recorded frames
    of each active segment into the npz cache, so the following tick's input build
    is a cache hit instead of paying the decompress on the critical path. The cursor
    is ~monotonic, so frame ``max_idx_reached + 1`` is almost always the next one
    consumed. Set ``prefetch_ahead=0`` to disable (A/B; results are identical either
    way — prefetch only warms the cache).
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
                    max_steps=max_steps_mult * (end - start),
                    unstick_after=unstick_after,
                    unstick_advance_m=unstick_advance_m,
                )
                for (tl, start, end) in chunk
            ]
            if save_dir is not None:
                from collections import deque
                from pathlib import Path

                for off, s in enumerate(states):
                    key = route_keys[c0 + off] if route_keys else _route_key(s.tl)
                    s.save_buf = deque(maxlen=save_max_scenes + 1)
                    s.save_out_dir = Path(save_dir) / f"{key}_{s.start}_{s.end}"
            active = list(states)
            while active:
                with timers("input_build"):
                    pre_list = list(pool.map(_pre_step, active))
                built = [(s, *pre) for s, pre in zip(active, pre_list) if pre is not None]
                if built:
                    with timers("to_torch"):
                        data = _to_torch_batch([b[1] for b in built], model_args, device)
                    with timers("model_forward"):
                        # Fire-and-forget prefetch of each segment's upcoming frames
                        # so they decompress on background threads while the GPU is
                        # busy below (overlaps the npz I/O with the forward).
                        if prefetch_ahead > 0:
                            # Only the segments actually running this tick (built); ones
                            # that terminated in _pre_step won't consume more frames.
                            for s, *_ in built:
                                nxt = s.cursor.max_idx_reached + 1
                                pool.submit(s.tl.prefetch, range(nxt, nxt + prefetch_ahead))
                        _, outputs = model(data)
                        preds = outputs["prediction"][:, 0].cpu().numpy()  # (B,80,4)
                    # Score ALL segments in one batched OBB pass, then advance each.
                    with timers("score"):
                        score_list = score_step_batched(
                            [b[2] for b in built], [b[0].ego_shape for b in built], device
                        )
                    for (s, _np, nb, idx), (cl, col, _M) in zip(built, score_list):
                        s.clearances[s.k] = cl
                        s.collisions[s.k] = col
                        # One-pass save: buffer this step, then dump the window on the
                        # FIRST collision — from THIS run, so the scenes match the hit.
                        if s.save_buf is not None:
                            s.save_buf.append((s.k, idx, s.live_pose.copy(), _np))
                            if (
                                not s.saved_collision
                                and save_thresh is not None
                                and cl <= save_thresh
                            ):
                                mani = _dump_precollision_window(
                                    s.save_out_dir,
                                    s.tl,
                                    model_args,
                                    s.k,
                                    list(s.save_buf),
                                    s.last_snap_step,
                                    save_pre_steps,
                                    save_thresh,
                                    s.start,
                                    s.end,
                                    pre_arc_m=save_pre_arc_m,
                                    max_scenes=save_max_scenes,
                                    min_post_snap_frames=save_min_post_snap_frames,
                                )
                                # Consume the segment's one save only on an ACTUAL write; a
                                # snap-skipped hit stays open for a later, settled collision.
                                if mani is not None:
                                    s.saved_collision = True
                    for i, (s, _np, nb, idx) in enumerate(built):
                        prev_snaps = s.n_snaps
                        _advance_step(s, preds[i], idx, device, timers)
                        # Clear the buffer on an unstick teleport: pre-jump frames belong
                        # to a different ego path and must never enter a saved window.
                        if s.save_buf is not None and s.n_snaps > prev_snaps:
                            s.save_buf.clear()
                            s.last_snap_step = s.k
                active = [s for s in active if not s.done]
            results.extend(_finalize(s, timers) for s in states)
    finally:
        pool.shutdown(wait=True)
    return results


# --------------------------------------------------------------------------- #
# Collision-scene extractor
# --------------------------------------------------------------------------- #
def _min_clearance_any(neighbors_live: np.ndarray, ego_shape: np.ndarray, device: str) -> float:
    """Min OBB clearance ego(at origin) to ANY valid neighbor (m). inf if none.

    Raw distance to the nearest neighbor of any kind (moving or static, any
    direction) — the collision trigger for extraction ("<= thresh m to a
    neighbor"). Same all-neighbor geometry score_step uses.
    """
    ego_b, npc_corners, M = _ego_neighbor_obb(neighbors_live, ego_shape, device)
    if M == 0:
        return float("inf")
    p1, p2 = _closest_points_between_rects(ego_b, npc_corners)
    return float((p1 - p2).norm(dim=-1).min())


def _future_to_4col(arr: np.ndarray) -> np.ndarray:
    """Heading future -> cos/sin future: (..., 3) [x, y, heading] -> (..., 4)
    [x, y, cos, sin]. Already-4-col input passes through. Zero (invalid) rows stay zero.
    The trainable / reward schema is ALWAYS 4-col for futures — never save 3-col."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] == 4:
        return arr
    mask = np.abs(arr[..., :2]).sum(-1) == 0
    h = arr[..., 2]
    out = np.concatenate(
        [arr[..., :2], np.cos(h)[..., None], np.sin(h)[..., None]], axis=-1
    ).astype(np.float32)
    out[mask] = 0.0
    return out


def _recenter_neighbor_future(naf: np.ndarray, dx: float, dy: float, dyaw: float) -> np.ndarray:
    """Re-center a recorded neighbor_agents_future onto the live ego.

    naf: (Pn, T, 3) [x, y, heading] (or (Pn, T, 4) [x,y,cos,sin]) in the recorded-ego
    frame. (dx, dy, dyaw) is the live ego pose in the recorded-ego frame. Returns
    (Pn, T, 4) [x, y, cos, sin] in the live-ego frame (the trainable/reward schema —
    never 3-col); invalid (zero) entries stay zero.
    """
    from scenario_generation.transforms import transform_positions

    naf = np.asarray(naf, dtype=np.float32)
    head = np.arctan2(naf[..., 3], naf[..., 2]) if naf.shape[-1] == 4 else naf[..., 2]
    mask = np.abs(naf[..., :2]).sum(-1) == 0
    R = _rotation_matrix(dyaw)
    xy = transform_positions(naf[..., :2], R, np.array([dx, dy], dtype=np.float64))
    h = head - dyaw
    out = np.concatenate(
        [xy.astype(np.float32), np.cos(h)[..., None], np.sin(h)[..., None]], axis=-1
    ).astype(np.float32)
    out[mask] = 0.0
    return out


def _world_pose_to_ego(world_pose: np.ndarray, ref_pose: np.ndarray) -> tuple[float, float, float]:
    """Express a world ego pose in the live-ego frame of ``ref_pose`` -> (x, y, heading)."""
    R = _rotation_matrix(float(ref_pose[2]))
    d = R @ (world_pose[:2] - ref_pose[:2])
    return float(d[0]), float(d[1]), float(world_pose[2] - ref_pose[2])


def _scene_npz_from_np_dict(np_dict: dict) -> dict:
    """Squeeze a [1,...] live-ego model-input dict into an un-batched training NPZ
    dict (drops the batch dim; keeps every key)."""
    return {k: np.asarray(v)[0] for k, v in np_dict.items()}


def _precollision_window_start(
    t_c: int,
    pre_steps: int,
    last_snap_step: int | None,
    poses_by_step: dict | None = None,
    pre_arc_m: float = 0.0,
    max_scenes: int | None = None,
) -> int:
    """First step of the pre-collision window.

    Baseline: ``t_c - pre_steps`` (>= ``pre_steps`` frames; may be negative -> the
    backtrack path reads recorded frames before the segment start). Never crosses an
    unstick snap (clamped to ``last_snap_step``).

    MIN-MOVEMENT EXTEND: if ``pre_arc_m > 0`` and the ego's cumulative arc length over
    the baseline window is below ``pre_arc_m`` (slow creep, e.g. queueing into a stopped
    car), the window is extended further back — frame by frame — until the ego has
    travelled ``pre_arc_m`` of arc length OR the window hits ``max_scenes`` frames (incl.
    t_c) OR the snap / buffer start. ``poses_by_step`` maps step -> world pose (from the
    live buffer) and supplies the arc length."""
    base = t_c - pre_steps
    if last_snap_step is not None:
        base = max(base, last_snap_step)
    # Cap to max_scenes frames even on the no-extend path, so we never request a step
    # that was already evicted from the (max_scenes+1)-deep buffer (which would silently
    # splice recorded GT frames into a window that should be all-live).
    if max_scenes is not None:
        base = max(base, t_c - (max_scenes - 1))
    if not poses_by_step or pre_arc_m <= 0:
        return base
    live_ks = sorted(k for k in poses_by_step if k <= t_c)
    if not live_ks:
        return base
    floor = live_ks[0]
    if max_scenes is not None:
        floor = max(floor, t_c - (max_scenes - 1))
    if last_snap_step is not None:
        floor = max(floor, last_snap_step)
    if base <= floor:  # can't extend (early collision / snap clamp already binds)
        return base

    def _cumarc(a: int) -> float:
        ks = [k for k in live_ks if a <= k <= t_c]
        s = 0.0
        for i in range(len(ks) - 1):
            p, q = poses_by_step[ks[i]], poses_by_step[ks[i + 1]]
            s += float(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5)
        return s

    start_k = base
    while start_k > floor and _cumarc(start_k) < pre_arc_m:
        start_k -= 1
    return start_k


def _route_key(tl: RouteTimeline) -> str:
    """Route key from a timeline's NPZ names, e.g. '16-24-07_00000000_00000038' -> '16-24-07_00000000'."""
    from pathlib import Path

    return "_".join(Path(tl.npz_paths[0]).stem.split("_")[:2])


def _dump_precollision_window(
    out_dir,
    tl: RouteTimeline,
    model_args,
    t_c: int,
    buf,
    last_snap_step: int | None,
    pre_steps: int,
    collision_thresh: float,
    seg_start: int,
    seg_end: int,
    pre_arc_m: float = 0.0,
    max_scenes: int | None = None,
    min_post_snap_frames: int = 0,
) -> dict | None:
    """Write the scenes before collision step ``t_c`` from a live buffer.

    ``buf`` is an iterable of ``(k, idx, live_pose, np_dict)`` snapshots captured DURING
    the rollout that detected the collision (so the saved scenes match that exact run —
    no re-simulation). Steps before the segment start are filled from the recorded NPZs.
    The window is >= ``pre_steps`` frames, extended backward to cover ``pre_arc_m`` of ego
    arc length when the ego barely moved (capped at ``max_scenes``), and never crosses an
    unstick teleport (``last_snap_step``). Returns the manifest, or None if skipped.

    SKIP rule: if an unstick teleport fired fewer than ``min_post_snap_frames`` steps
    before the collision, the ego had too little settled history after the jump — the
    contact is likely teleport-induced, so nothing is saved (returns None).
    """
    import json
    from pathlib import Path

    if (
        min_post_snap_frames > 0
        and last_snap_step is not None
        and (t_c - last_snap_step) < min_post_snap_frames
    ):
        print(
            f"  [save] SKIP collision@{t_c}: only {t_c - last_snap_step} frames "
            f"({(t_c - last_snap_step) * DT:.1f}s) of history since the unstick snap"
        )
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live_by_step = {rec[0]: rec for rec in buf}
    poses_by_step = {rec[0]: rec[2] for rec in buf}  # step k -> world pose (for ego_future)
    fut_len = int(model_args.future_len)
    saved: list[int] = []

    start_k = _precollision_window_start(
        t_c, pre_steps, last_snap_step, poses_by_step, pre_arc_m, max_scenes
    )
    if start_k < t_c - pre_steps:
        print(
            f"  [save] window extended {t_c - pre_steps} -> {start_k} "
            f"(slow ego: {t_c - start_k} frames to cover {pre_arc_m} m arc)"
        )
    elif start_k > t_c - pre_steps:
        print(f"  [save] window clamped {t_c - pre_steps} -> {start_k} (unstick snap in window)")
    # Inclusive of t_c: the LAST saved frame (collision+00000) IS the collision step, so the
    # window ends exactly on the contact (its current neighbor box is the one within thresh).
    for step_k in range(start_k, t_c + 1):
        if step_k >= 0 and step_k in live_by_step:
            _, idx, live_pose, np_dict = live_by_step[step_k]
            scene = _scene_npz_from_np_dict(np_dict)
            ep = scene["ego_agent_past"]
            scene["ego_agent_past"] = np.column_stack(
                [ep[:, 0], ep[:, 1], np.arctan2(ep[:, 3], ep[:, 2])]
            ).astype(np.float32)
            g = scene["goal_pose"]
            scene["goal_pose"] = np.array([g[0], g[1], math.atan2(g[3], g[2])], dtype=np.float32)
            with np.load(tl.npz_paths[idx], allow_pickle=True) as z:
                naf = z["neighbor_agents_future"] if "neighbor_agents_future" in z.files else None
            if naf is not None:
                dx, dy, dyaw = _rel_pose(tl.poses[idx], live_pose)
                scene["neighbor_agents_future"] = _recenter_neighbor_future(naf, dx, dy, dyaw)
            eaf = np.zeros((fut_len, 4), dtype=np.float32)
            for j in range(1, fut_len + 1):
                fk = step_k + j
                if fk > t_c or fk not in poses_by_step:
                    break
                ex, ey, eh = _world_pose_to_ego(poses_by_step[fk], live_pose)
                eaf[j - 1] = (ex, ey, math.cos(eh), math.sin(eh))
            scene["ego_agent_future"] = eaf
            scene["origin"] = np.array("live")
        else:
            frame = seg_start + step_k
            if frame < 0 or frame >= len(tl):
                continue
            with np.load(tl.npz_paths[frame], allow_pickle=True) as z:
                scene = {key: z[key] for key in z.files if key not in ("map_name", "token")}
            for fk in ("neighbor_agents_future", "ego_agent_future"):
                if fk in scene:
                    scene[fk] = _future_to_4col(scene[fk])
            scene["origin"] = np.array("recorded")
        token = f"{step_k - t_c:+06d}"
        np.savez_compressed(out_dir / f"collision{token}.npz", **scene)
        saved.append(int(step_k))

    manifest = {
        "segment": [int(seg_start), int(seg_end)],
        "collision_step": int(t_c),
        "collision_thresh": float(collision_thresh),
        "n_scenes": len(saved),
        "steps_saved": saved,
        "n_live": sum(1 for k in saved if k >= 0),
        "n_recorded": sum(1 for k in saved if k < 0),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def extract_collision_scenes(
    model,
    model_args,
    tl: RouteTimeline,
    start: int,
    end: int,
    out_dir,
    device: str = "cuda",
    collision_thresh: float = 0.2,
    pre_steps: int = 80,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    unstick_after: int = 300,
    unstick_advance_m: float = 5.0,
    max_steps: int | None = None,
    pre_arc_m: float = 1.0,
    max_scenes: int = 160,
    min_post_snap_frames: int = 30,
) -> dict | None:
    """Mine the batch of scenes leading up to the FIRST collision in a segment.

    Runs the closed-loop reproducer over [start, end); on the first sim-step T_c
    whose ego↔any-neighbor OBB clearance <= ``collision_thresh`` (default 0.2 m),
    saves the ``pre_steps`` (default 80) scenes BEFORE the collision as training
    NPZs. Each saved scene is a full model-input snapshot centered on the ego at
    that step, with:
      * ego_agent_past + ego_current_state (backtracked live history),
      * neighbor_agents_past (recorded neighbors re-centered),
      * neighbor_agents_future (recorded GT, re-centered),
      * the full map (lanes/route/borders/polygons/goal/ego_shape/turn_indicators),
      * ego_agent_future = the realized live ego path truncated AT the collision
        (zeros for timesteps after T_c).

    When the collision is early (T_c < pre_steps), the window reaches before the
    segment start: those earlier scenes are taken straight from the RECORDED NPZs
    of frames before ``start`` (real GT context — ego/neighbor history + GT
    futures), which the full-route timeline still holds. So the batch always has
    ``pre_steps`` scenes with real, continuous ego history (subject to the route
    start). Returns a manifest dict, or None if no collision occurred.

    NOTE: this is the legacy SECOND-PASS path — it re-runs the rollout to reconstruct
    the window, so the collision it finds can differ from the one the miner detected
    (the closed-loop rollout is sensitive to GPU batch composition). Prefer the
    ONE-PASS save in ``run_segments_batched`` (``save_dir=...``), which dumps the
    buffer from the SAME run that detected the collision. Kept for backward compat.
    """
    from collections import deque
    from pathlib import Path

    out_dir = Path(out_dir)
    cap = max_steps if max_steps is not None else 3 * (end - start)
    timers = Timers()
    s = _seed_state(
        tl,
        start,
        end,
        search_radius,
        warmup_steps,
        0.5,
        5.0,
        0,
        timers,
        max_steps=cap,
        unstick_after=unstick_after,
        unstick_advance_m=unstick_advance_m,
    )
    # Rolling buffer of the last max_scenes+1 live steps; each: (k, idx, live_pose, np_dict).
    buf: deque = deque(maxlen=max_scenes + 1)
    t_c = None
    # Step right after the most recent unstick snap (None = no snap yet). The pre-collision
    # window must not cross a snap, or a saved scene's realized ego_future/ego_past would
    # span the ~5 m teleport.
    last_snap_step = None
    while not s.done:
        k = s.k
        pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx = pre
        data = _to_torch_batch([np_dict], model_args, device)
        _, outputs = model(data)
        pred = outputs["prediction"][0, 0].cpu().numpy()
        clr = _min_clearance_any(neighbors_live, s.ego_shape, device)
        buf.append((k, idx, s.live_pose.copy(), np_dict))
        if clr <= collision_thresh:
            t_c = k
            break
        prev_snaps = s.n_snaps
        _post_step(s, pred, neighbors_live, idx, device, timers)
        if s.n_snaps > prev_snaps:  # an unstick snap fired advancing k -> s.k
            last_snap_step = s.k
    if t_c is None:
        return None

    # buf's last entry is the collision step t_c (appended before the break), so it
    # already carries the t_c world pose needed for the late scenes' ego_future.
    return _dump_precollision_window(
        out_dir,
        tl,
        model_args,
        t_c,
        list(buf),
        last_snap_step,
        pre_steps,
        collision_thresh,
        start,
        end,
        pre_arc_m=pre_arc_m,
        max_scenes=max_scenes,
        min_post_snap_frames=min_post_snap_frames,
    )
