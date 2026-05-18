"""Filter a path_list.json by removing samples whose GT ego trajectory collides with
static_objects, neighbor_agents_future, or road_border line strings.

All data in each npz is in the ego-centric frame at t=0, so no transform is required.

Usage:
    python ros_scripts/filter_collision_free_npz.py path_list.json \\
        --save_path filtered_list.json \\
        --num_workers 8
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path, help="Input path_list.json")
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument(
        "--static_object_margin",
        type=float,
        default=0.0,
        help="Inflate static-object rect by this margin (m). Collision if ego rect overlaps inflated target.",
    )
    parser.add_argument(
        "--neighbor_margin",
        type=float,
        default=0.0,
        help="Inflate neighbor rect by this margin (m).",
    )
    parser.add_argument(
        "--road_border_margin",
        type=float,
        default=0.0,
        help="Collision if any ego rect edge point comes within this distance of a road border segment.",
    )
    parser.add_argument(
        "--time_stride",
        type=int,
        default=1,
        help="Check every N-th timestep of ego_agent_future (1 = all 80 steps).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
    )
    return parser.parse_args()


def load_path_list(input_json: Path) -> list[str]:
    """Accept both legacy list format and sampling dict format {"seed": ..., "files": [...]}."""
    with open(input_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data["files"])
    return list(data)


def compute_rect_corners(
    cx: np.ndarray,
    cy: np.ndarray,
    cos_h: np.ndarray,
    sin_h: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    """Return (..., 4, 2) corners in CCW order for an oriented rectangle centred at (cx, cy)."""
    half_l = length / 2.0
    half_w = width / 2.0
    # Local corners: FR, FL, RL, RR (CCW when viewed from +z)
    local_x = np.stack([half_l, half_l, -half_l, -half_l], axis=-1)
    local_y = np.stack([half_w, -half_w, -half_w, half_w], axis=-1)
    rot_x = cos_h[..., None] * local_x - sin_h[..., None] * local_y
    rot_y = sin_h[..., None] * local_x + cos_h[..., None] * local_y
    out_x = cx[..., None] + rot_x
    out_y = cy[..., None] + rot_y
    return np.stack([out_x, out_y], axis=-1)  # (..., 4, 2)


def compute_ego_corners(
    ego_future: np.ndarray,
    ego_shape: np.ndarray,
) -> np.ndarray:
    """Return (T, 4, 2). Matches loss.compute_ego_bbox_corners: ego_future xy is rear axle;
    box centre is offset forward by wheelbase / 2 along heading.
    """
    wheel_base = ego_shape[0]
    length = ego_shape[1]
    width = ego_shape[2]
    x = ego_future[:, 0]
    y = ego_future[:, 1]
    h = ego_future[:, 2]
    cos_h = np.cos(h)
    sin_h = np.sin(h)
    cog_offset = 0.5 * wheel_base
    cx = x + cos_h * cog_offset
    cy = y + sin_h * cog_offset
    T = ego_future.shape[0]
    length_arr = np.full(T, length, dtype=np.float32)
    width_arr = np.full(T, width, dtype=np.float32)
    return compute_rect_corners(cx, cy, cos_h, sin_h, length_arr, width_arr)


def rect_rect_overlap_sat(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
) -> np.ndarray:
    """SAT overlap check for oriented rectangles.

    corners_a: (..., 4, 2), corners_b: (..., 4, 2) — broadcast-compatible.
    Returns bool (...) — True iff the two rects overlap.
    """
    # Edge vectors from corner 0->1 and 1->2 give two orthogonal axes; for two rects we need 4 axes.
    def edge_normals(corners: np.ndarray) -> np.ndarray:
        e0 = corners[..., 1, :] - corners[..., 0, :]  # (..., 2)
        e1 = corners[..., 2, :] - corners[..., 1, :]
        n0 = np.stack([-e0[..., 1], e0[..., 0]], axis=-1)
        n1 = np.stack([-e1[..., 1], e1[..., 0]], axis=-1)
        return np.stack([n0, n1], axis=-2)  # (..., 2, 2)

    axes = np.concatenate(
        [edge_normals(corners_a), edge_normals(corners_b)], axis=-2
    )  # (..., 4, 2)
    # Project corners onto each axis: (... , 4_corner, 4_axis)
    proj_a = np.einsum("...cd,...ad->...ca", corners_a, axes)
    proj_b = np.einsum("...cd,...ad->...ca", corners_b, axes)
    min_a = proj_a.min(axis=-2)
    max_a = proj_a.max(axis=-2)
    min_b = proj_b.min(axis=-2)
    max_b = proj_b.max(axis=-2)
    separated_axis = (max_a < min_b) | (max_b < min_a)  # (..., 4)
    return ~separated_axis.any(axis=-1)


def point_to_segment_distance_np(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """p: (..., 2), a/b: (..., 2). Broadcast-compatible. Returns (...,)."""
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(-1)
    safe_denom = np.where(denom < 1e-8, np.float32(1e-8), denom)
    t = (ap * ab).sum(-1) / safe_denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    diff = p - closest
    return np.sqrt((diff * diff).sum(-1) + 1e-12)


def check_static_object_collision(
    ego_corners: np.ndarray,
    static_objects: np.ndarray,
    margin: float,
) -> bool:
    """ego_corners: (T, 4, 2). static_objects: (N, 10) with channels [x, y, cos, sin, w, l, type×4]."""
    valid_mask = np.abs(static_objects[:, :4]).sum(axis=-1) > 1e-6
    if not valid_mask.any():
        return False
    objs = static_objects[valid_mask]  # (M, 10)
    M = objs.shape[0]

    cx = objs[:, 0]
    cy = objs[:, 1]
    cos_h = objs[:, 2]
    sin_h = objs[:, 3]
    width = objs[:, 4] + 2.0 * margin
    length = objs[:, 5] + 2.0 * margin
    obj_corners = compute_rect_corners(cx, cy, cos_h, sin_h, length, width)  # (M, 4, 2)

    # Broadcast over (T, M)
    T = ego_corners.shape[0]
    ego_b = ego_corners[:, None, :, :]  # (T, 1, 4, 2)
    obj_b = obj_corners[None, :, :, :]  # (1, M, 4, 2)
    ego_bb = np.broadcast_to(ego_b, (T, M, 4, 2))
    obj_bb = np.broadcast_to(obj_b, (T, M, 4, 2))
    overlap = rect_rect_overlap_sat(ego_bb, obj_bb)  # (T, M)
    return bool(overlap.any())


def check_neighbor_collision(
    ego_corners: np.ndarray,
    neighbor_future: np.ndarray,
    neighbor_past: np.ndarray,
    margin: float,
) -> bool:
    """ego_corners: (T, 4, 2). neighbor_future: (N, T, 3) [x, y, heading]. neighbor_past: (N, P, 11)."""
    # neighbor i valid iff its past last frame is non-zero
    last_past = neighbor_past[:, -1, :]
    valid_mask = np.abs(last_past[:, :4]).sum(axis=-1) > 1e-6
    if not valid_mask.any():
        return False
    nf = neighbor_future[valid_mask]  # (M, T, 3)
    nw = last_past[valid_mask, 6] + 2.0 * margin  # width (M,)
    nl = last_past[valid_mask, 7] + 2.0 * margin  # length (M,)
    nw = np.maximum(nw, 1e-3)
    nl = np.maximum(nl, 1e-3)
    M, T, _ = nf.shape

    # Per-timestep validity (skip frames where position is zero — padding)
    step_valid = np.abs(nf[..., :2]).sum(axis=-1) > 1e-6  # (M, T)
    if not step_valid.any():
        return False

    nx = nf[..., 0]
    ny = nf[..., 1]
    nh = nf[..., 2]
    n_cos = np.cos(nh)
    n_sin = np.sin(nh)
    nl_bt = np.broadcast_to(nl[:, None], (M, T))
    nw_bt = np.broadcast_to(nw[:, None], (M, T))
    nb_corners = compute_rect_corners(nx, ny, n_cos, n_sin, nl_bt, nw_bt)  # (M, T, 4, 2)

    # Align with ego: ego_corners (T, 4, 2) -> (M, T, 4, 2)
    ego_b = np.broadcast_to(ego_corners[None, :, :, :], (M, T, 4, 2))
    overlap = rect_rect_overlap_sat(ego_b, nb_corners)  # (M, T)
    overlap = overlap & step_valid
    return bool(overlap.any())


def check_road_border_collision(
    ego_corners: np.ndarray,
    line_strings: np.ndarray,
    margin: float,
) -> bool:
    """ego_corners: (T, 4, 2). line_strings: (N, P, 4); channel 3 > 0.5 marks road border."""
    # Build (Sx2) segment list filtered to road border, valid endpoints.
    a = line_strings[:, :-1, :]  # (N, P-1, 4)
    b = line_strings[:, 1:, :]
    seg_valid = (np.abs(a[..., :2]).sum(-1) > 1e-6) & (np.abs(b[..., :2]).sum(-1) > 1e-6)
    is_border = (line_strings[..., 3] > 0.5).any(axis=-1, keepdims=True)  # (N, 1)
    seg_valid = seg_valid & is_border  # (N, P-1)
    if not seg_valid.any():
        return False

    seg_a = a[..., :2][seg_valid]  # (S, 2)
    seg_b = b[..., :2][seg_valid]  # (S, 2)
    S = seg_a.shape[0]
    T = ego_corners.shape[0]

    # 1) Edge-point distance: min distance from ego corners to any border segment.
    p = ego_corners.reshape(T * 4, 1, 2)
    aa = seg_a[None, :, :]
    bb = seg_b[None, :, :]
    dist = point_to_segment_distance_np(p, aa, bb)  # (T*4, S)
    if (dist < margin).any():
        return True

    # 2) Edge-segment intersection: catch the case where ego edge crosses a border segment
    #    without any ego corner being close.
    edge_a = ego_corners  # (T, 4, 2)
    edge_b = np.roll(ego_corners, -1, axis=1)
    return _segments_intersect_any(
        edge_a.reshape(T * 4, 2),
        edge_b.reshape(T * 4, 2),
        seg_a,
        seg_b,
    )


def _segments_intersect_any(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> bool:
    """Returns True if any segment (p1[i], p2[i]) intersects any segment (p3[j], p4[j])."""
    # Broadcast to (I, J)
    p1b = p1[:, None, :]
    p2b = p2[:, None, :]
    p3b = p3[None, :, :]
    p4b = p4[None, :, :]
    d1 = _cross(p4b - p3b, p1b - p3b)
    d2 = _cross(p4b - p3b, p2b - p3b)
    d3 = _cross(p2b - p1b, p3b - p1b)
    d4 = _cross(p2b - p1b, p4b - p1b)
    cond_general = ((d1 * d2) < 0) & ((d3 * d4) < 0)
    return bool(cond_general.any())


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def check_one(args_tuple):
    path, opts = args_tuple
    data = np.load(path, allow_pickle=True)
    ego_future = data["ego_agent_future"][:: opts["time_stride"]]  # (T, 3)
    ego_shape = data["ego_shape"]
    ego_corners = compute_ego_corners(ego_future, ego_shape)

    reasons = []
    if check_static_object_collision(
        ego_corners, data["static_objects"], opts["static_object_margin"]
    ):
        reasons.append("static_object")
    neighbor_future = data["neighbor_agents_future"][:, :: opts["time_stride"], :]
    if check_neighbor_collision(
        ego_corners,
        neighbor_future,
        data["neighbor_agents_past"],
        opts["neighbor_margin"],
    ):
        reasons.append("neighbor")
    if check_road_border_collision(
        ego_corners, data["line_strings"], opts["road_border_margin"]
    ):
        reasons.append("road_border")
    return (path, reasons)


def main() -> None:
    args = parse_args()
    paths = load_path_list(args.input_json)
    print(f"Loaded {len(paths)} paths from {args.input_json}")

    opts = {
        "static_object_margin": args.static_object_margin,
        "neighbor_margin": args.neighbor_margin,
        "road_border_margin": args.road_border_margin,
        "time_stride": args.time_stride,
    }
    job_args = [(p, opts) for p in paths]

    kept: list[str] = []
    dropped: list[tuple[str, list[str]]] = []
    counts = {"static_object": 0, "neighbor": 0, "road_border": 0}

    if args.num_workers <= 1:
        iterator = (check_one(a) for a in job_args)
    else:
        pool = Pool(processes=args.num_workers)
        iterator = pool.imap_unordered(check_one, job_args, chunksize=16)

    for path, reasons in tqdm(iterator, total=len(paths)):
        if len(reasons) == 0:
            kept.append(path)
        else:
            dropped.append((path, reasons))
            for r in reasons:
                counts[r] += 1

    if args.num_workers > 1:
        pool.close()
        pool.join()

    kept.sort()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(kept, f, indent=4)

    log_path = args.save_path.with_suffix(".log")
    with open(log_path, "w") as f:
        f.write(f"input_json: {args.input_json}\n")
        f.write(f"save_path: {args.save_path}\n")
        f.write(f"static_object_margin: {args.static_object_margin}\n")
        f.write(f"neighbor_margin: {args.neighbor_margin}\n")
        f.write(f"road_border_margin: {args.road_border_margin}\n")
        f.write(f"time_stride: {args.time_stride}\n")
        f.write(f"total: {len(paths)}\n")
        f.write(f"kept:  {len(kept)}\n")
        f.write(f"dropped: {len(dropped)}\n")
        f.write(f"  by static_object: {counts['static_object']}\n")
        f.write(f"  by neighbor:      {counts['neighbor']}\n")
        f.write(f"  by road_border:   {counts['road_border']}\n")
        f.write("\n--- dropped paths ---\n")
        for p, rs in dropped:
            f.write(f"{','.join(rs)}\t{p}\n")

    print(f"kept {len(kept)} / {len(paths)} (dropped {len(dropped)})")
    print(f"  static_object: {counts['static_object']}")
    print(f"  neighbor:      {counts['neighbor']}")
    print(f"  road_border:   {counts['road_border']}")
    print(f"Saved {args.save_path} (and log {log_path})")


if __name__ == "__main__":
    main()
