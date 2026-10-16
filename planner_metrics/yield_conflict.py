"""Yield post-encroachment time: does the predicted ego let the agent it yields to go first?

Shared by the ``pedestrian_yield`` and ``vehicle_yield`` scenario labels. Their
samples are decision frames: the yielded-to agent is still in the ego's way and clears
within the prediction horizon (the scenario selector picks the frame, e.g. 2 s before
it clears). Everything is read from that one frame, in the shared ego-relative frame
(see ``planner_metrics/scene_format.py``); index ``k`` of a trajectory is ``0.1 * k`` s
after the frame, index 0 being the current pose.

Conflicts, on the human's future path (``ego_agent_future`` from the current pose; arc
``s`` and signed lateral ``d`` of each point projected onto it):

- crossing: the agent's center comes within ego half width + agent half width +
  margin (``VEHICLE_MARGIN_M``; ``VRU_MARGIN_M`` for pedestrians and cyclists, whose
  whole crossing a driver waits for) while moving (``MIN_SPEED_MPS``) at
  ``MIN_CROSSING_ANGLE_DEG`` or more to the path. Conflict point = median arc while in
  the band; the conflict lasts while it stays in. A crossing longer than
  ``MAX_CROSSING_S`` is not one (a lead vehicle or a parked car on the path);
- merging: a vehicle or cyclist that enters the band from outside and then stays on
  the path (the ego waits to turn into its road). Conflict point = where it entered;
  it has cleared it ``MERGE_CLEAR_M`` further on.

Target: a conflict ``0 .. maximum_conflict_distance_m`` ahead of the ego's front now;
the last one the human lets go first, else the one with the shortest human PET.

PET (post-encroachment time) is the time from the agent clearing the conflict point to
the ego's front reaching it; when the ego's rear clears it before the agent arrives it
is negative (the ego went first). A sample fails when the predicted ego goes first,
reaches the conflict point while the agent is still on it, or has a PET under
``minimum_pet_seconds``. A prediction that does not reach the conflict point within
the horizon passes. A sample without a target also passes, and is counted in
``target_found_rate_percent`` so a selection that misses its agent shows up.

The closed-loop scenario metric for the same labels uses the same rules on the whole
recorded window; keep the two in step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from planner_metrics.evaluation import MetricEvaluation

DT_S = 0.1
VEHICLE_MARGIN_M = 0.3
VRU_MARGIN_M = 2.0
MIN_SPEED_MPS = 0.3
MIN_CROSSING_ANGLE_DEG = 20.0
MAX_CROSSING_S = 8.0
MERGE_CLEAR_M = 3.0
# The human's 8 s future is the rear axle's; its front reaches a conflict point the axle
# never gets to, and a waiting human barely moves. The path is extended straight on from
# its last pose by this much (along the last motion, else the current heading).
PATH_EXTENSION_M = 20.0
VEHICLE, PEDESTRIAN, CYCLIST = 0, 1, 2

# Detail ``order`` codes.
ORDERS = (
    "no_target",
    "agent_first",
    "ego_first",
    "overlap",
    "ego_never_reached",
    "agent_not_cleared",
)


@dataclass(frozen=True)
class Conflict:
    agent_type: int  # VEHICLE / PEDESTRIAN / CYCLIST
    kind: str  # "cross" | "merge"
    s_m: float  # conflict point, arc along the path
    half_w_m: float  # agent half width counted into the conflict point
    first: int  # index the agent enters the conflict
    last: int  # last index it is in it


def project_onto_path(
    points: np.ndarray, path: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arc length, signed lateral offset (left +) and path heading at each point.

    Each point goes to its nearest segment (clamped); zero-length segments are skipped.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    path = np.asarray(path, dtype=np.float64)
    seg = np.diff(path, axis=0)
    length = np.linalg.norm(seg, axis=1)
    keep = length > 1e-9
    if not keep.any():
        d = np.linalg.norm(pts - path[0], axis=1)
        return np.zeros(len(pts)), d, np.zeros(len(pts))
    a, seg, length = path[:-1][keep], seg[keep], length[keep]
    cum = np.concatenate([[0.0], np.cumsum(length)])[:-1]
    rel = pts[:, None, :] - a[None]
    t = np.clip((rel * seg[None]).sum(-1) / length[None] ** 2, 0.0, 1.0)
    foot = a[None] + t[..., None] * seg[None]
    j = np.linalg.norm(pts[:, None, :] - foot, axis=-1).argmin(axis=1)
    k = np.arange(len(pts))
    off = pts - foot[k, j]
    side = np.sign(seg[j, 0] * off[:, 1] - seg[j, 1] * off[:, 0])
    return (
        cum[j] + t[k, j] * length[j],
        side * np.linalg.norm(off, axis=1),
        np.arctan2(seg[j, 1], seg[j, 0]),
    )


def find_conflict(
    xy: np.ndarray,
    valid: np.ndarray,
    agent_type: int,
    half_w_m: float,
    path: np.ndarray,
    ego_width_m: float,
) -> Conflict | None:
    """The agent's crossing or merge of ``path``; ``xy``/``valid`` per index ``k``."""
    idx = np.flatnonzero(valid)
    if len(idx) < 5:
        return None
    pts = xy[idx]
    s, lat, tangent = project_onto_path(pts, path)
    path_end = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    v = np.gradient(pts, idx * DT_S, axis=0)
    speed = np.linalg.norm(v, axis=1)
    angle = np.abs((np.arctan2(v[:, 1], v[:, 0]) - tangent + np.pi / 2) % np.pi - np.pi / 2)
    margin = VEHICLE_MARGIN_M if agent_type == VEHICLE else VRU_MARGIN_M
    inside = (np.abs(lat) <= ego_width_m / 2 + half_w_m + margin) & (s > 0.5) & (s < path_end - 0.5)
    moving = speed >= MIN_SPEED_MPS
    crossing = inside & moving & (angle >= np.radians(MIN_CROSSING_ANGLE_DEG))
    contiguous = np.r_[False, np.diff(idx) == 1]
    if crossing.any():
        lo = hi = int(np.flatnonzero(crossing)[0])
        while lo > 0 and inside[lo - 1] and contiguous[lo]:
            lo -= 1
        while hi + 1 < len(idx) and inside[hi + 1] and contiguous[hi + 1]:
            hi += 1
        if (idx[hi] - idx[lo]) * DT_S <= MAX_CROSSING_S:
            return Conflict(
                agent_type,
                "cross",
                float(np.median(s[lo : hi + 1])),
                half_w_m,
                int(idx[lo]),
                int(idx[hi]),
            )
        if agent_type == PEDESTRIAN or lo == 0:
            return None
        entry = lo
    else:
        if agent_type == PEDESTRIAN:
            return None
        entries = np.flatnonzero(inside[1:] & ~inside[:-1] & moving[1:]) + 1
        if not len(entries):
            return None
        entry = int(entries[0])
    s0 = float(s[entry])
    past = np.flatnonzero(s[entry:] >= s0 + MERGE_CLEAR_M)
    if not len(past):
        return None
    return Conflict(agent_type, "merge", s0, 0.0, int(idx[entry]), int(idx[entry + past[0]]))


def post_encroachment_time(
    front_s: np.ndarray, rear_s: np.ndarray, conflict: Conflict, n_agent_steps: int
) -> tuple[float | None, str]:
    """``(PET in s or None, order)`` of an ego with arcs per index against ``conflict``."""
    reach = np.flatnonzero(front_s >= conflict.s_m - conflict.half_w_m)
    if not len(reach):
        return None, "ego_never_reached"
    k_reach = int(reach[0])
    clear = np.flatnonzero(rear_s >= conflict.s_m + conflict.half_w_m)
    if len(clear) and clear[0] <= conflict.first:
        return float(clear[0] - conflict.first) * DT_S, "ego_first"
    k_out = conflict.last + 1
    if k_out < n_agent_steps and k_out <= k_reach:
        return float(k_reach - k_out) * DT_S, "agent_first"
    if k_out >= n_agent_steps:
        return None, "agent_not_cleared"  # still on the conflict at the horizon's end
    return 0.0, "overlap"


def extend_path(path: np.ndarray, extension_m: float = PATH_EXTENSION_M) -> np.ndarray:
    """``path`` with a straight piece of ``extension_m`` appended after its last point."""
    steps = np.diff(path, axis=0)
    moved = np.flatnonzero(np.linalg.norm(steps, axis=1) > 1e-3)
    direction = steps[moved[-1]] if len(moved) else np.array([1.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    return np.vstack([path, path[-1] + extension_m * direction])


def _ego_arcs(
    traj_xy: np.ndarray, path: np.ndarray, front: float, rear: float
) -> tuple[np.ndarray, np.ndarray]:
    s = np.maximum.accumulate(project_onto_path(np.vstack([[0.0, 0.0], traj_xy]), path)[0])
    return s + front, s - rear


def _sample(value: torch.Tensor, index: int, ndim: int) -> np.ndarray:
    value = value[index] if value.ndim == ndim + 1 else value
    return value.detach().cpu().numpy().astype(np.float64)


def evaluate_yield_conflict_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Score each sample's predicted ego against the agent the human yields to."""
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (B, T, D>=2), got {tuple(ego_trajs.shape)}")
    min_pet = float(parameters["minimum_pet_seconds"])
    max_dist = float(parameters["maximum_conflict_distance_m"])
    for key in ("ego_agent_future", "neighbor_agents_past", "neighbor_agents_future", "ego_shape"):
        if key not in data:
            raise ValueError(f"yield_conflict needs {key}")
    batch = ego_trajs.shape[0]
    out = {
        "target_found": np.zeros(batch, dtype=bool),
        "target_type": np.full(batch, -1, dtype=np.int64),
        "target_is_merge": np.zeros(batch, dtype=bool),
        "conflict_distance_m": np.full(batch, np.nan),
        "pet_seconds": np.full(batch, np.nan),
        "human_pet_seconds": np.full(batch, np.nan),
        "order": np.zeros(batch, dtype=np.int64),
        "human_order": np.zeros(batch, dtype=np.int64),
        "yielded": np.ones(batch, dtype=bool),
    }
    for b in range(batch):
        gt = _sample(data["ego_agent_future"], b, 2)
        gt_valid = np.abs(gt[:, :2]).sum(axis=1) > 0
        if not gt_valid.any():
            continue
        # Zero rows pad a recording that ends early; a human waiting at the origin also
        # reads (0, 0), so only the rows after the last non-zero one are padding.
        human_xy = gt[: int(np.flatnonzero(gt_valid)[-1]) + 1, :2]
        path = extend_path(np.vstack([[0.0, 0.0], human_xy]))
        wb, length, width = _sample(data["ego_shape"], b, 1)[:3]
        front, rear = (wb + length) / 2.0, (length - wb) / 2.0
        past = _sample(data["neighbor_agents_past"], b, 3)
        future = _sample(data["neighbor_agents_future"], b, 3)
        n_steps = 1 + future.shape[1]
        candidates = []
        for i in range(past.shape[0]):
            xy = np.vstack([past[i, -1:, :2], future[i, :, :2]])
            valid = np.abs(xy).sum(axis=1) > 0
            if not valid.any():
                continue
            agent_type = int(np.argmax(past[i, -1, 8:11])) if past.shape[-1] >= 11 else VEHICLE
            half_w = abs(float(past[i, -1, 6])) / 2.0 if past.shape[-1] >= 11 else 0.9
            c = find_conflict(xy, valid, agent_type, half_w, path, width)
            if c is not None and 0.0 <= c.s_m - front <= max_dist:
                candidates.append(c)
        if not candidates:
            continue
        human_front, human_rear = _ego_arcs(human_xy, path, front, rear)
        human = [
            (c, *post_encroachment_time(human_front, human_rear, c, n_steps)) for c in candidates
        ]
        let_go = [h for h in human if h[2] == "agent_first"]
        if let_go:
            target, human_pet, human_order = max(let_go, key=lambda h: h[0].last)
        else:
            target, human_pet, human_order = min(human, key=lambda h: (h[1] is None, h[1] or 0.0))
        pred_front, pred_rear = _ego_arcs(
            ego_trajs[b, :, :2].detach().cpu().numpy(), path, front, rear
        )
        pet, order = post_encroachment_time(pred_front, pred_rear, target, n_steps)
        out["target_found"][b] = True
        out["target_type"][b] = target.agent_type
        out["target_is_merge"][b] = target.kind == "merge"
        out["conflict_distance_m"][b] = target.s_m - front
        out["pet_seconds"][b] = np.nan if pet is None else pet
        out["human_pet_seconds"][b] = np.nan if human_pet is None else human_pet
        out["order"][b] = ORDERS.index(order)
        out["human_order"][b] = ORDERS.index(human_order)
        out["yielded"][b] = not (
            order in ("ego_first", "overlap")
            or (pet is not None and order == "agent_first" and pet < min_pet)
        )
    device, dtype = ego_trajs.device, ego_trajs.dtype
    t = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
    return MetricEvaluation(
        scores={
            "success_rate_percent": t["yielded"].to(dtype) * 100.0,
            "target_found_rate_percent": t["target_found"].to(dtype) * 100.0,
        },
        details={
            "yield_conflict": {
                **{k: (v.to(dtype) if v.dtype == torch.float64 else v) for k, v in t.items()},
                "minimum_pet_seconds": torch.full((batch,), min_pet, dtype=dtype, device=device),
            }
        },
    )


__all__ = [
    "Conflict",
    "ORDERS",
    "evaluate_yield_conflict_with_details",
    "extend_path",
    "find_conflict",
    "post_encroachment_time",
    "project_onto_path",
]
