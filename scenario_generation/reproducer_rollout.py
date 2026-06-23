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
from scenario_generation.simulate import decode_turn_indicator
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


def build_input_raw(
    tl: RouteTimeline,
    idx: int,
    live_pose: np.ndarray,
    ego_hist_world: np.ndarray,
    dyn: _EgoDyn,
) -> tuple:
    """gpu_transform variant of :func:`build_input_np`: skip the numpy
    ``world_to_ego_frame`` (it runs once, batched, on the GPU in ``_to_torch_batch_gpu``)
    and return the UN-transformed recorded base + the relative pose + the live-ego arrays.
    Still threadable (only ``np.load`` + cheap live-ego construction)."""
    base = _npz_to_model_base(tl.npz(idx))
    dxyz = _rel_pose(tl.poses[idx], live_pose)
    live_past = _live_ego_past(ego_hist_world, live_pose)  # (1, PAST, 4), already ego-frame
    live_cur = _live_ego_current(dyn)  # (1, 10), already ego-frame
    return base, dxyz, live_past, live_cur, idx


def _arrays_to_device(arrays: dict, device: str) -> dict:
    """H2D each model-input array with the correct dtype: the speed-limit has-flags stay
    bool, ``turn_indicators`` is long, everything else is float32. Shared by both batch
    builders so the two paths can't drift on dtype handling."""
    out: dict = {}
    for k, arr in arrays.items():
        if k in ("lanes_has_speed_limit", "route_lanes_has_speed_limit"):
            out[k] = torch.from_numpy(arr).to(device)
        elif k == "turn_indicators":
            out[k] = torch.from_numpy(arr).long().to(device)
        else:
            out[k] = torch.from_numpy(arr.astype(np.float32)).to(device)
    return out


def _add_static_inputs(data: dict, model_args, n: int, device: str) -> None:
    """Add the per-batch ``delay`` + zero ``sampled_trajectories`` the model expects
    (P = 1 ego + predicted neighbors, T = future_len + 1). In place."""
    data["delay"] = torch.zeros((n,), dtype=torch.long, device=device)
    n_agents = 1 + model_args.predicted_neighbor_num
    data["sampled_trajectories"] = torch.zeros(
        (n, n_agents, model_args.future_len + 1, POSE_DIM), dtype=torch.float32, device=device
    )


def _to_torch_batch(np_dicts: list[dict], model_args, device: str) -> dict:
    """Concat N single-sample numpy dicts -> one batched, normalized torch dict.

    Does the work that used to be per-segment (host->device copy + normalization)
    ONCE for the whole batch: N concatenations, one H2D transfer per key, one
    normalizer call.
    """
    N = len(np_dicts)
    arrays = {k: np.concatenate([d[k] for d in np_dicts], axis=0) for k in np_dicts[0]}
    data = _arrays_to_device(arrays, device)
    _add_static_inputs(data, model_args, N, device)
    return model_args.observation_normalizer(data)


def _to_torch_batch_gpu(raw_payloads: list[tuple], model_args, device: str, want_np_dicts: bool):
    """gpu_transform build: stack the UN-transformed recorded frames, H2D once, run
    ``world_to_ego_frame_torch`` on the whole batch on-device, swap in the live ego, then
    normalize — so the per-segment numpy ``world_to_ego_frame`` is replaced by ONE batched
    GPU op. Returns ``(data, neighbors_live_list, np_dict_list_or_None)`` so the caller's
    score/save/advance path is byte-for-byte the same as the CPU build (the saved scenes
    and ``neighbors_live`` are extracted here, pre-normalization, exactly as build_input_np
    produced them — only the float ordering of the transform differs, ~1e-5).

    ``want_np_dicts`` materializes per-segment un-normalized dicts for the save buffer; it
    forces a D2H of the full model input each step, so it is only requested when saving.
    """
    from scenario_generation.transforms import world_to_ego_frame_torch

    bases = [p[0] for p in raw_payloads]
    N = len(bases)
    batch = _arrays_to_device(
        {k: np.concatenate([b[k] for b in bases], axis=0) for k in bases[0]}, device
    )

    dx = torch.tensor([p[1][0] for p in raw_payloads], dtype=torch.float32, device=device)
    dy = torch.tensor([p[1][1] for p in raw_payloads], dtype=torch.float32, device=device)
    dyaw = torch.tensor([p[1][2] for p in raw_payloads], dtype=torch.float32, device=device)
    world_to_ego_frame_torch(batch, dx, dy, dyaw)

    # Swap in the live ego (already in the live-ego frame, NOT transformed) — matches
    # build_input_np, which overwrites these AFTER world_to_ego_frame.
    batch["ego_agent_past"] = torch.from_numpy(
        np.concatenate([p[2] for p in raw_payloads], axis=0).astype(np.float32)
    ).to(device)
    batch["ego_current_state"] = torch.from_numpy(
        np.concatenate([p[3] for p in raw_payloads], axis=0).astype(np.float32)
    ).to(device)

    # Corrected neighbor context: replace the transformed recorded neighbor block with the
    # simulated (shown-motion) one, already built in the live-ego frame — matches the CPU
    # override in _pre_step. p[5] is None in recorded mode (payload is a 5-tuple).
    if len(raw_payloads[0]) > 5 and raw_payloads[0][5] is not None:
        batch["neighbor_agents_past"] = torch.from_numpy(
            np.concatenate([p[5] for p in raw_payloads], axis=0).astype(np.float32)
        ).to(device)

    # Extract scoring inputs (+ optional save dicts) BEFORE normalization, so they match
    # the un-normalized arrays build_input_np returns.
    nb_all = batch["neighbor_agents_past"][:, :, -1, :].detach().cpu().numpy()  # (N,320,11)
    neighbors_live = [nb_all[i].copy() for i in range(N)]
    np_dicts = None
    if want_np_dicts:
        np_dicts = [
            {k: batch[k][i : i + 1].detach().cpu().numpy() for k in batch} for i in range(N)
        ]

    _add_static_inputs(batch, model_args, N, device)
    return model_args.observation_normalizer(batch), neighbors_live, np_dicts


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
    # Closed-loop turn-indicator history (INPUT_T+1,). Seeded from the recorded frame, then
    # each step the MODEL's predicted turn indicator is fed back in (recorded seed phases out
    # within PAST steps, exactly like ego_hist) — so the model context + saved npz never carry
    # the recorded driver's signals, only the sim's own predictions.
    turn_hist: np.ndarray = None
    # Per-collision-episode save state. An episode runs while clearance <= thresh and ends on
    # clearing; ``last_collision_uuid`` is the colliding UUID of the last SAVED collision (a new
    # episode is distinct only if its UUID differs). ``episode_eligible`` is set once per episode
    # (distinct?), ``episode_saved`` latches after the episode's one window is written.
    last_collision_uuid: object = None
    in_episode: bool = False
    episode_eligible: bool = False
    episode_saved: bool = False
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
    # Corrected neighbor context (neighbor_history_mode="sim"): rebuild neighbor_agents_past
    # each step from the SIMULATED shown motion (velocity from shown deltas, frozen->v~0).
    nbr_tracker: object = None


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
    neighbor_history_mode="recorded",
) -> _SegState:
    from scenario_generation.mpc_tracker import PerfectTracker

    # Step cap: defaults to the segment length, but can exceed it so a slow ego
    # (e.g. one that waited out a long red light) can still drive to the segment end.
    cap = int(max_steps) if max_steps is not None else (end - start)

    cursor = PerceptionReproducer(tl, search_radius=search_radius, timers=timers)
    cursor.reset(start)
    live_pose, ego_hist, dyn = _ego_state_from_frame(tl, start)
    # Corrected neighbor context: one timeline sample per 0.1 s sim step (real time on a
    # step=1 corpus). Raises if the corpus has no neighbor tracks.
    nbr_tracker = (
        SimNeighborTracker(tl, start, max_rec_advance=1.0)
        if neighbor_history_mode == "sim"
        else None
    )
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
        turn_hist=np.asarray(tl.npz(start)["turn_indicators"]).reshape(-1).astype(np.int64),
        ego_shape=np.asarray(tl.npz(start)["ego_shape"]).reshape(-1)[:3].astype(np.float32),
        goal_xy=tl.poses[end - 1, :2],
        clearances=np.full(cap, np.inf, dtype=np.float32),
        collisions=np.zeros(cap, dtype=bool),
        prev_max_idx=cursor.max_idx_reached,
        max_steps=cap,
        unstick_after=int(unstick_after),
        unstick_advance_m=float(unstick_advance_m),
        nbr_tracker=nbr_tracker,
    )


def _pre_step(s: _SegState, gpu_transform: bool = False):
    """Advance the cursor + build this segment's model input, or terminate it.

    Returns (np_dict, neighbors_live, idx) normally, or — when ``gpu_transform`` — the raw
    payload tuple (base, pose, live_past, live_cur, idx) for a batched on-device transform
    (see ``_to_torch_batch_gpu``). None when the segment just terminated (s.done set).
    CPU-only / threadable; torch conversion happens once per batch in the caller."""
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
    sim_nb = None
    slot_uuids = None  # slot -> UUID for the sim neighbor block (sim mode only)
    world_by_uuid = None  # UUID -> current shown world pose (for sim-future assembly)
    if s.nbr_tracker is not None:
        s.nbr_tracker.step(idx, s.live_pose[:2])
        sim_nb, slot_uuids, world_by_uuid = s.nbr_tracker.build(
            s.live_pose
        )  # (1,320,31,11) live-ego
    if gpu_transform:
        # 8-tuple (..., sim_nb, slot_uuids, world_by_uuid); sim_nb overrides the recorded
        # neighbor block AFTER the batched world_to_ego transform (None = recorded mode).
        base, dxyz, live_past, live_cur, ridx = build_input_raw(
            s.tl, idx, s.live_pose, s.ego_hist, s.dyn
        )
        if s.turn_hist is not None:
            base["turn_indicators"] = s.turn_hist[None].astype(
                np.int64
            )  # closed-loop, not recorded
        return (base, dxyz, live_past, live_cur, ridx, sim_nb, slot_uuids, world_by_uuid)
    np_dict, neighbors_live = build_input_np(s.tl, idx, s.live_pose, s.ego_hist, s.dyn)
    if sim_nb is not None:
        np_dict["neighbor_agents_past"] = sim_nb
        neighbors_live = sim_nb[0, :, -1, :].copy()
    if s.turn_hist is not None:
        np_dict["turn_indicators"] = s.turn_hist[None].astype(np.int64)  # closed-loop, not recorded
    return np_dict, neighbors_live, idx, slot_uuids, world_by_uuid


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
                # Re-seed the sim machinery at the teleport target so post-snap recording is
                # correct, not stale: the neighbor tracker's rec_t is capped at 1.0/step, so
                # without this it would lag many steps behind the jumped ego (stale neighbors);
                # turn_hist would carry pre-snap predictions for a different ego path. Both
                # restart from the recorded state at tgt and phase out again (like rollout start).
                # The save buffer is cleared on this snap (caller), so no window mixes the jump.
                if s.nbr_tracker is not None:
                    s.nbr_tracker = SimNeighborTracker(s.tl, tgt, max_rec_advance=1.0)
                if s.turn_hist is not None:
                    s.turn_hist = (
                        np.asarray(s.tl.npz(tgt)["turn_indicators"]).reshape(-1).astype(np.int64)
                    )
                s.last_collision_uuid = None  # teleported -> next contact is a fresh collision
                s.in_episode = False
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
    neighbor_history_mode: str = "recorded",
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
        neighbor_history_mode=neighbor_history_mode,
    )
    while not s.done:
        with timers("input_build"):
            pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx, _suuid, _wbu = pre
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


# --------------------------------------------------------------------------- #
# Simulated neighbor history (corrected closed-loop neighbor context)
# --------------------------------------------------------------------------- #
def _build_nbr_world_tracks(tl: RouteTimeline, lo: int, hi: int, eps: float = 0.05):
    """Per-UUID recorded WORLD trajectory + box attrs + array-index span.

    Scans recorded frames and, keyed by the sidecar ``neighbor_ids`` (track UUIDs),
    collects each track's world pose, box attributes (width, length, is_veh, is_ped,
    is_bike) and the (first, last) array index where it appears. Returns
    ``(interp, attrs, span)`` (interp[uuid] = (idx_arr, xy_arr (N,2), heading_arr (N,))).
    """
    raw: dict[str, list] = {}
    attrs: dict[str, np.ndarray] = {}
    for idx in range(lo, hi):
        ids = tl.neighbor_ids(idx)
        if not ids:
            continue
        pose = tl.poses[idx]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        nb = tl.neighbor_last(idx)  # (320, 11) recorded-ego frame — single-key load (fast)
        for slot in range(min(len(ids), nb.shape[0])):
            row = nb[slot]
            if np.abs(row[:6]).sum() == 0:
                continue
            u = ids[slot]
            wx = pose[0] + row[0] * c - row[1] * s
            wy = pose[1] + row[0] * s + row[1] * c
            wh = math.atan2(row[3], row[2]) + pose[2]
            raw.setdefault(u, []).append((idx, wx, wy, wh))
            if u not in attrs:
                attrs[u] = row[6:11].astype(np.float32)  # width,length,is_veh,is_ped,is_bike
    interp: dict[str, tuple] = {}
    span: dict[str, tuple] = {}
    for u, lst in raw.items():
        if len(lst) < 2:
            continue
        kept = [lst[0]]
        for samp in lst[1:]:
            if math.hypot(samp[1] - kept[-1][1], samp[2] - kept[-1][2]) > eps:
                kept.append(samp)
        if kept[-1][0] != lst[-1][0]:
            kept.append(lst[-1])
        interp[u] = (
            np.array([k[0] for k in kept]),
            np.array([[k[1], k[2]] for k in kept], dtype=np.float64),
            np.unwrap(np.array([k[3] for k in kept])),
        )
        span[u] = (int(lst[0][0]), int(lst[-1][0]))
    return interp, attrs, span


def _route_nbr_tracks(tl: RouteTimeline):
    """Per-route UUID world tracks, built once and cached on the timeline."""
    cached = getattr(tl, "_nbr_tracks", None)
    if cached is None:
        cached = _build_nbr_world_tracks(tl, 0, len(tl))
        tl._nbr_tracks = cached
    return cached


class SimNeighborTracker:
    """Build the model's neighbor context from the SIMULATED (shown) neighbor motion.

    Recorded mode copies each cursor frame's own 31-step history verbatim, so a
    cursor-frozen car still reads its recorded velocity (e.g. a moving car held in
    place because the ego crept still shows ~11 m/s while it visibly never moves —
    input and replay disagree, producing phantom collisions).

    This tracker follows each neighbor by track UUID, advances a continuous
    recorded-time cursor ``rec_t`` toward the position-keyed cursor's target frame
    (capped at ``max_rec_advance`` array-indices per 0.1 s sim step → interpolates
    between recorded anchors), and keeps a rolling per-sim-step world history per
    UUID. ``neighbor_agents_past`` is rebuilt from that shown history each step:
    velocity is the finite difference of the shown positions, so a frozen neighbor
    reads v approx 0 (a static obstacle) and a moving one its true speed. Step 0 is
    seeded from the recorded history so the first frame equals the original context.
    """

    def __init__(self, tl: RouteTimeline, start: int, max_rec_advance: float = 1.0):
        self.tl = tl
        self.interp, self.attrs, self.span = _route_nbr_tracks(tl)
        if not self.interp:
            raise ValueError(
                "SimNeighborTracker: no neighbor tracks (sidecar neighbor_ids empty). Reconvert "
                "the corpus with populated neighbor_ids, or use neighbor_history_mode=recorded."
            )
        self.rec_t = float(start)
        self.max_adv = float(max_rec_advance)
        self.hist: dict[str, list] = {}  # uuid -> rolling list[(wx,wy,wh)], len <= PAST
        self._seed_start(start)

    def _present(self, u: str, t: float) -> bool:
        lo, hi = self.span[u]
        return lo - 0.5 <= t <= hi + 0.5

    def _seed_start(self, start: int) -> None:
        """Seed each track present at ``start`` with the recorded 0.1 s history leading
        up to ``start`` (from its world anchors) so step 0 reproduces the original context."""
        for u in self.interp:
            if not self._present(u, start):
                continue
            self.hist[u] = [
                _interp_pose(self.interp[u], start - (PAST - 1) + k) for k in range(PAST - 1)
            ]

    def _frac_target(self, target_idx: int, live_xy) -> float:
        """Refine the integer position-cursor frame to a FRACTIONAL recorded index by
        projecting the live ego onto the recorded ego path around ``target_idx``.

        The cursor returns the nearest *integer* recorded frame, so chasing it with an
        integer-capped step makes ``rec_t`` snap to integers and ``_interp_pose`` never
        actually interpolates — a slow live ego then holds a neighbor for several steps and
        jumps a whole 0.1 s of recorded motion at once (the visible jank). Projecting the
        live ego onto the recorded ego polyline segment around ``target_idx`` yields a
        sub-frame fraction, so ``rec_t`` advances smoothly and the neighbor interpolates."""
        if live_xy is None:
            return float(target_idx)
        poses = self.tl.poses
        n = len(poses)
        live = np.asarray(live_xy, dtype=np.float64)[:2]
        best, best_d = float(target_idx), float("inf")
        for i in (int(target_idx) - 1, int(target_idx)):
            if i < 0 or i + 1 >= n:
                continue
            a = poses[i, :2].astype(np.float64)
            ab = poses[i + 1, :2].astype(np.float64) - a
            l2 = float(ab @ ab)
            if l2 < 1e-9:
                continue
            tc = min(max(float((live - a) @ ab / l2), 0.0), 1.0)
            d = float(np.hypot(*(live - (a + tc * ab))))
            if d < best_d:
                best_d, best = d, i + tc
        return best

    def step(self, target_idx: int, live_xy=None) -> None:
        """Advance ``rec_t`` toward the (fractional) cursor target (capped, never backward)
        and push the current interpolated world pose of every present track into its rolling
        history. ``live_xy`` enables the sub-frame fractional advance (smooth interpolation)."""
        target = self._frac_target(target_idx, live_xy)
        self.rec_t += min(max(target - self.rec_t, 0.0), self.max_adv)
        for u in self.interp:
            if not self._present(u, self.rec_t):
                continue
            p = _interp_pose(self.interp[u], self.rec_t)
            dq = self.hist.get(u)
            if dq is None:
                dq = [p] * (PAST - 1)  # newly appeared -> no motion history yet (v approx 0)
                self.hist[u] = dq
            dq.append(p)
            if len(dq) > PAST:
                del dq[0]

    def build(self, live_pose: np.ndarray) -> tuple[np.ndarray, list, dict]:
        """(1, 320, 31, 11) neighbor_agents_past in the live-ego frame, from shown history."""
        ex, ey, eyaw = float(live_pose[0]), float(live_pose[1]), float(live_pose[2])
        R = _rotation_matrix(eyaw)  # world delta -> ego frame (rotates by -eyaw)
        present = [u for u in self.hist if len(self.hist[u]) > 0 and self._present(u, self.rec_t)]

        def _cur_d2(u):
            wx, wy, _ = self.hist[u][-1]
            d = R @ np.array([wx - ex, wy - ey])
            return float(d[0] * d[0] + d[1] * d[1])

        present.sort(key=_cur_d2)  # nearest-first, mirroring the recorded slot order
        out = np.zeros((320, PAST, 11), dtype=np.float32)
        slot_uuids: list[str] = []  # slot -> track UUID (for sim-future assembly across frames)
        world_by_uuid: dict[str, tuple] = {}  # UUID -> current shown world pose (wx, wy, wh)
        for slot, u in enumerate(present[:320]):
            slot_uuids.append(u)
            world_by_uuid[u] = self.hist[u][-1]
            dq = self.hist[u]
            n = len(dq)
            poses = ([dq[0]] * (PAST - n)) + list(dq) if n < PAST else list(dq)
            world = np.asarray(poses, dtype=np.float64)  # (PAST, 3) wx, wy, wh
            d = (world[:, :2] - np.array([ex, ey])) @ R.T  # (PAST, 2) ego-frame xy
            lh = world[:, 2] - eyaw
            out[slot, :, 0] = d[:, 0]
            out[slot, :, 1] = d[:, 1]
            out[slot, :, 2] = np.cos(lh)
            out[slot, :, 3] = np.sin(lh)
            vw = np.zeros((PAST, 2))
            if PAST > 1:
                vw[1:] = np.diff(world[:, :2], axis=0) / DT
                vw[0] = vw[1]
            ve = vw @ R.T  # world velocity -> ego frame
            out[slot, :, 4] = ve[:, 0]
            out[slot, :, 5] = ve[:, 1]
            out[slot, :, 6:11] = self.attrs[u]
        return out[None], slot_uuids, world_by_uuid


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
        np_dict, neighbors_live, idx, _suuid, _wbu = pre
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
    save_min_pre_frames: int = 30,
    save_min_ego_speed: float = 0.5,
    route_keys: list[str] | None = None,
    gpu_transform: bool = False,
    neighbor_history_mode: str = "recorded",
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
    if save_dir is not None and save_max_scenes < save_pre_steps + 1:
        # The buffer is save_max_scenes+1 deep and the window is >= save_pre_steps frames
        # plus the collision step. A smaller cap would silently truncate the window — fail
        # loudly rather than save shorter-than-requested batches.
        raise ValueError(
            f"save_max_scenes ({save_max_scenes}) must be >= save_pre_steps + 1 "
            f"({save_pre_steps + 1}); otherwise the saved window is silently truncated."
        )
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
                    neighbor_history_mode=neighbor_history_mode,
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
                    pre_list = list(pool.map(lambda s: _pre_step(s, gpu_transform), active))
                live = [(s, pre) for s, pre in zip(active, pre_list) if pre is not None]
                if live:
                    if gpu_transform:
                        # ONE batched on-device world_to_ego_frame; downstream identical.
                        raw_payloads = [pre for _s, pre in live]
                        with timers("to_torch"):
                            data, nb_list, npd_list = _to_torch_batch_gpu(
                                raw_payloads, model_args, device, want_np_dicts=save_dir is not None
                            )
                        built = [
                            (
                                s,
                                npd_list[i] if npd_list is not None else None,
                                nb_list[i],
                                raw_payloads[i][4],  # idx
                                raw_payloads[i][6],  # slot_uuids (sim mode; None otherwise)
                                raw_payloads[i][7],  # world_by_uuid
                            )
                            for i, (s, _pre) in enumerate(live)
                        ]
                    else:
                        built = [(s, *pre) for s, pre in live]
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
                        # Model's predicted turn indicator per segment, decoded with the SAME
                        # C++-style keep-bias logic as the perfect-tracker sim (reused helper),
                        # then fed back into turn_hist below (closed-loop, no recorded leak).
                        ti_pred = decode_turn_indicator(outputs["turn_indicator_logit"], 0.25)
                    # Score ALL segments in one batched OBB pass, then advance each.
                    with timers("score"):
                        score_list = score_step_batched(
                            [b[2] for b in built], [b[0].ego_shape for b in built], device
                        )
                    for (s, _np, nb, idx, suuid, wbu), (cl, col, _M) in zip(built, score_list):
                        s.clearances[s.k] = cl
                        s.collisions[s.k] = col
                        # One-pass save: buffer this step, then dump the window on the
                        # FIRST collision — from THIS run, so the scenes match the hit.
                        if s.save_buf is not None:
                            s.save_buf.append((s.k, idx, s.live_pose.copy(), _np, suuid, wbu))
                            # Per-EPISODE saving. A contact EPISODE runs while clearance <= thresh
                            # and ends when it clears (> thresh) — so a NEW distinct collision needs
                            # collided -> NOT collided -> collided (the clear gap). An episode is
                            # ELIGIBLE only if its colliding vehicle's UUID differs from the last
                            # SAVED collision's (a same-vehicle re-contact after a brief jitter-clear
                            # is the SAME collision, not a new one; UUID can change for one physical
                            # vehicle, so this is a heuristic). Within an eligible episode we retry
                            # every step until the FIRST clean-start window saves (the contact onset
                            # may have <80 clear steps before it, but a slightly-later step in the
                            # same episode often has a clean 80-step approach), then stop for that
                            # episode. The colliding (nearest) neighbor is slot 0 (build() sorts
                            # nearest-first), so its UUID is suuid[0]. Each save -> its own per-
                            # episode dir tagged by the save step. Gates (t0-clean / ego-moved /
                            # min-pre-frames) still apply, and the UUID is only consumed on an ACTUAL
                            # save (a dropped degenerate contact must not block a later savable one).
                            in_contact = save_thresh is not None and cl <= save_thresh
                            if not in_contact:
                                s.in_episode = False  # episode ended; the next contact is new
                            else:
                                colliding_uuid = suuid[0] if suuid else None
                                if not s.in_episode:  # entering contact from a clear -> new episode
                                    s.in_episode = True
                                    s.episode_saved = False
                                    s.episode_eligible = (
                                        colliding_uuid is None
                                        or colliding_uuid != s.last_collision_uuid
                                    )
                                # Cheap pre-check before the (expensive) save attempt: only try
                                # when the window's nominal start frame is CLEAR (> save_thresh).
                                # In a sustained contact the lookback start is still in-contact, so
                                # t0-clean would drop it anyway — skipping the _dump (npz-load +
                                # OBB clearance) here avoids a per-step save-spam that starved the
                                # GPU. When the prior contact scrolls out of the lookback the start
                                # clears and we attempt (so a later clean window in the same episode
                                # is still caught). Buffer/snap floor matches _precollision_window_start.
                                ws = s.k - save_pre_steps
                                if s.last_snap_step is not None:
                                    ws = max(ws, s.last_snap_step)
                                start_clear = ws < 0 or s.clearances[ws] > save_thresh
                                if s.episode_eligible and not s.episode_saved and start_clear:
                                    episode_dir = Path(f"{s.save_out_dir}_tc{s.k:05d}")
                                    mani = _dump_precollision_window(
                                        episode_dir,
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
                                        min_pre_frames=save_min_pre_frames,
                                        min_ego_speed=save_min_ego_speed,
                                    )
                                    if mani is not None:
                                        s.episode_saved = True
                                        s.last_collision_uuid = colliding_uuid
                    for i, (s, _np, nb, idx, _suuid, _wbu) in enumerate(built):
                        prev_snaps = s.n_snaps
                        _advance_step(s, preds[i], idx, device, timers)
                        # Feed the model's predicted turn indicator back into the rolling
                        # history (recorded seed scrolls out within PAST steps) — the saved
                        # context then carries the sim's own signals, never the recorded ones.
                        if s.turn_hist is not None:
                            s.turn_hist = np.append(s.turn_hist[1:], np.int64(ti_pred[i]))
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

    Baseline: ``t_c - pre_steps``, then clamped UP to the earliest live buffer step
    (``min(poses_by_step)``, >= 0) — recorded backfill is disabled, so the window is
    all-live and may be shorter than ``pre_steps`` for an early contact. Never crosses an
    unstick snap (clamped to ``last_snap_step``).

    MIN-MOVEMENT EXTEND: if ``pre_arc_m > 0`` and the ego's cumulative arc length over
    the baseline window is below ``pre_arc_m`` (slow creep, e.g. queueing into a stopped
    car), the window is extended further back — frame by frame — until the ego has
    travelled ``pre_arc_m`` of arc length OR the window hits ``max_scenes`` frames (incl.
    t_c) OR the snap / buffer start. ``poses_by_step`` maps step -> world pose (from the
    live buffer) and supplies the arc length."""
    base = t_c - pre_steps
    # NEVER backfill recorded frames: clamp to the earliest LIVE step held in the buffer
    # (>= 0; after an unstick teleport the buffer was cleared, so its min step is the
    # post-snap floor). An early contact therefore yields a SHORTER all-live window rather
    # than a recorded prefix that doesn't physically connect to the live rollout.
    live_floor = min(poses_by_step) if poses_by_step else 0
    base = max(base, live_floor)
    if last_snap_step is not None:
        base = max(base, last_snap_step)
    # Cap to max_scenes frames even on the no-extend path, so we never request a step
    # that was already evicted from the (max_scenes+1)-deep buffer.
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
    """Route key from a timeline, using the SAME derivation as group_routes/the miner
    (``route_timeline.route_prefix`` — strip the trailing ``_<frameidx>``), so the
    output dir name matches the hits.jsonl ``route`` field and distinct routes that
    only share a leading token are not collapsed together."""
    from pathlib import Path

    from scenario_generation.route_timeline import route_prefix

    return route_prefix(Path(tl.npz_paths[0]))


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
    min_pre_frames: int = 30,
    min_ego_speed: float = 0.5,
) -> dict | None:
    """Write the scenes before collision step ``t_c`` from a live buffer.

    ``buf`` is an iterable of ``(k, idx, live_pose, np_dict)`` snapshots captured DURING
    the rollout that detected the collision (so the saved scenes match that exact run —
    no re-simulation). The window is ALL-LIVE: it spans at most ``pre_steps`` frames before
    the contact, extended backward to cover ``pre_arc_m`` of ego arc length when the ego
    barely moved (capped at ``max_scenes``), clamped to the live buffer and never crossing
    an unstick teleport (``last_snap_step``). Recorded frames before the rollout start are
    NOT backfilled — splicing the recorded ego/perception onto the model-driven live state
    produced a discontinuous clearance jump at the seam, so an early contact just yields a
    shorter all-live window. Returns the manifest, or None if skipped.

    SKIP rules (return None):
    - an unstick teleport fired fewer than ``min_post_snap_frames`` steps before the
      collision (too little settled history; the contact is likely teleport-induced);
    - fewer than ``min_pre_frames`` live frames precede the contact (too short an approach
      to be a useful pre-collision scene now that recorded backfill is disabled).
    """
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    # Clear any prior batch in this dir FIRST — before the skip early-return — so that a
    # re-mine which now SKIPS this segment (or writes a shorter window) never leaves stale
    # older-offset collision*.npz behind to be mistaken for a fresh save. Don't create the
    # dir on a skip (avoid littering empty dirs); only clear if it already exists.
    if out_dir.exists():
        for stale in out_dir.glob("collision*.npz"):
            stale.unlink()

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

    live_by_step = {rec[0]: rec for rec in buf}
    poses_by_step = {rec[0]: rec[2] for rec in buf}  # step k -> world pose (for ego_future)
    fut_len = int(model_args.future_len)
    saved: list[int] = []

    start_k = _precollision_window_start(
        t_c, pre_steps, last_snap_step, poses_by_step, pre_arc_m, max_scenes
    )
    # All-live window: never backfill recorded frames (start_k is clamped to the live
    # floor). Skip the hit if too few live frames precede the contact.
    n_pre = t_c - start_k
    if n_pre < min_pre_frames:
        print(
            f"  [save] SKIP collision@{t_c}: only {n_pre} live pre-frames "
            f"(< {min_pre_frames}); not backfilling recorded frames"
        )
        return None
    # t0-clean gate: a valid pre-collision scene must START clear of the neighbor and approach
    # INTO contact. If the window's first frame is already within collision_thresh, the ego is
    # already in/through the neighbor (it collided earlier, then crept while the position cursor
    # barely advanced) — that's not a recoverable approach, so drop it (nothing to learn).
    first_np = live_by_step[start_k][3]
    nb0 = np.asarray(first_np["neighbor_agents_past"])[0, :, -1, :]
    es0 = np.asarray(first_np["ego_shape"]).reshape(-1)[:3].astype(np.float32)
    c0 = _min_clearance_any(nb0, es0, "cpu")
    if c0 <= collision_thresh:
        print(
            f"  [save] SKIP collision@{t_c}: window starts already in contact "
            f"(t0 clearance {c0:.2f}m <= {collision_thresh}m) — ego already through the neighbor"
        )
        return None
    # ego-moved gate (replaces the instantaneous speed-at-contact gate): the EGO must have
    # driven across the approach — total ego path over [start_k, t_c] > min_ego_speed * window
    # seconds * 0.3. A model creeping into a car (~0.31 m/s -> ~2.5 m over 8 s) PASSES (it's a
    # real avoidance failure); a stopped ego rear-ended by a moving neighbor (~0 m path) is
    # DROPPED. Uses the realized live ego poses (poses_by_step), so it's a sim quantity.
    ks = sorted(k for k in poses_by_step if start_k <= k <= t_c)
    ego_arc = sum(
        float(np.hypot(*(poses_by_step[ks[i + 1]][:2] - poses_by_step[ks[i]][:2])))
        for i in range(len(ks) - 1)
    )
    min_arc = min_ego_speed * (n_pre * DT) * 0.3
    if ego_arc < min_arc:
        print(
            f"  [save] SKIP collision@{t_c}: ego barely moved over the approach "
            f"(arc {ego_arc:.2f}m < {min_arc:.2f}m) — stopped/rear-ended, not an ego-caused approach"
        )
        return None
    # All gates passed — only NOW create the output dir, so rejected retries (t0-clean /
    # ego-moved / too-short) never litter empty per-episode dirs.
    out_dir.mkdir(parents=True, exist_ok=True)
    if start_k < t_c - pre_steps:
        print(
            f"  [save] window extended {t_c - pre_steps} -> {start_k} "
            f"(slow ego: {t_c - start_k} frames to cover {pre_arc_m} m arc)"
        )
    elif start_k > t_c - pre_steps:
        print(
            f"  [save] window shortened {t_c - pre_steps} -> {start_k} "
            f"(all-live: no recorded backfill / unstick floor)"
        )
    # Inclusive of t_c: the LAST saved frame (collision+00000) IS the collision step, so the
    # window ends exactly on the contact (its current neighbor box is the one within thresh).
    for step_k in range(start_k, t_c + 1):
        # The window is clamped all-live (start_k >= the live buffer floor; no recorded
        # backfill). Every step in [start_k, t_c] must therefore be in the buffer — a miss
        # means the clamp/buffer invariant regressed, so fail loudly rather than silently
        # splice recorded frames back in (the discontinuous seam this change removed).
        if step_k < 0 or step_k not in live_by_step:
            raise AssertionError(
                f"all-live pre-collision window expected step {step_k} in the live buffer "
                f"[{start_k}, {t_c}] but it is absent (window-clamp / buffer regression)"
            )
        _, idx, live_pose, np_dict, slot_uuids, _wbu = live_by_step[step_k]
        scene = _scene_npz_from_np_dict(np_dict)
        ep = scene["ego_agent_past"]
        scene["ego_agent_past"] = np.column_stack(
            [ep[:, 0], ep[:, 1], np.arctan2(ep[:, 3], ep[:, 2])]
        ).astype(np.float32)
        g = scene["goal_pose"]
        scene["goal_pose"] = np.array([g[0], g[1], math.atan2(g[3], g[2])], dtype=np.float32)
        # neighbor_agents_future: read out of the SIMULATION's own shown future (the realized
        # neighbor world poses at the subsequent rollout steps), UUID-matched and slot-aligned
        # with neighbor_agents_past, expressed in this frame's live-ego frame. This keeps the
        # target consistent with the (sim) past — a held-static neighbor stays static instead
        # of teleporting to its recorded log. Recorded mode (no tracker) keeps the recorded GT.
        if slot_uuids is not None:
            naf_sim = np.zeros((320, fut_len, 4), dtype=np.float32)
            ex0, ey0, eh0 = float(live_pose[0]), float(live_pose[1]), float(live_pose[2])
            R = _rotation_matrix(eh0)  # world delta -> live-ego frame (matches build())
            uuid_slots = list(enumerate(slot_uuids[:320]))
            for j in range(1, fut_len + 1):
                fk = step_k + j
                if fk > t_c or fk not in live_by_step:
                    break  # rollout ended at contact; no shown future beyond t_c
                wbu_fk = live_by_step[fk][5] or {}
                slots, wx, wy, wh = [], [], [], []
                for slot, u in uuid_slots:
                    wp = wbu_fk.get(u)
                    if wp is not None:
                        slots.append(slot)
                        wx.append(wp[0])
                        wy.append(wp[1])
                        wh.append(wp[2])
                if not slots:
                    continue
                d = (np.column_stack([wx, wy]) - np.array([ex0, ey0])) @ R.T  # (m,2) ego xy
                h = np.asarray(wh) - eh0
                naf_sim[slots, j - 1, 0] = d[:, 0]
                naf_sim[slots, j - 1, 1] = d[:, 1]
                naf_sim[slots, j - 1, 2] = np.cos(h)
                naf_sim[slots, j - 1, 3] = np.sin(h)
            scene["neighbor_agents_future"] = naf_sim
        else:
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
        token = f"{step_k - t_c:+06d}"
        np.savez_compressed(out_dir / f"collision{token}.npz", **scene)
        saved.append(int(step_k))

    manifest = {
        "segment": [int(seg_start), int(seg_end)],
        "collision_step": int(t_c),
        "collision_thresh": float(collision_thresh),
        "n_scenes": len(saved),
        "steps_saved": saved,
        "n_live": len(saved),  # all-live by construction (no recorded backfill)
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
    min_pre_frames: int = 30,
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

    When the collision is early (T_c < pre_steps) the window is SHORTER (all-live): it
    shares ``_dump_precollision_window``, which no longer backfills recorded frames
    before the rollout start (splicing the recorded ego/perception onto the live state
    produced a discontinuous clearance seam) and skips a hit with fewer than
    ``min_pre_frames`` (default 30) live frames before the contact. Returns a manifest
    dict, or None if no collision occurred / the window was too short.

    NOTE: this is the legacy SECOND-PASS path — it re-runs the rollout to reconstruct
    the window, so the collision it finds can differ from the one the miner detected
    (the closed-loop rollout is sensitive to GPU batch composition). Prefer the
    ONE-PASS save in ``run_segments_batched`` (``save_dir=...``), which dumps the
    buffer from the SAME run that detected the collision. Kept for backward compat.
    """
    from collections import deque
    from pathlib import Path

    if max_scenes < pre_steps + 1:
        # Same invariant the one-pass saver enforces: the window is >= pre_steps frames
        # plus the collision step, so a smaller cap would silently truncate it.
        raise ValueError(
            f"max_scenes ({max_scenes}) must be >= pre_steps + 1 ({pre_steps + 1}); "
            "otherwise the saved window is silently truncated."
        )
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
        np_dict, neighbors_live, idx, _suuid, _wbu = pre
        data = _to_torch_batch([np_dict], model_args, device)
        _, outputs = model(data)
        pred = outputs["prediction"][0, 0].cpu().numpy()
        clr = _min_clearance_any(neighbors_live, s.ego_shape, device)
        # 6-tuple to match the batched buffer / _dump_precollision_window unpacking; this
        # legacy extractor path is recorded-mode (no sim tracker) -> slot_uuids/world None.
        buf.append((k, idx, s.live_pose.copy(), np_dict, None, None))
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
        min_pre_frames=min_pre_frames,
    )
