"""Geometry primitives for the metrics subscores: ego bbox, polygon / point-to-segment
distance, lane-polygon builders, and Savitzky-Golay kernels.

Dependency-free (math / numpy / torch + guidance.collision only) — must not import
from ``rlvr``, so the base-SFT validation loop can use it without pulling in RLVR.
"""

from __future__ import annotations

import numpy as np
import torch


def _build_ego_bbox_corners(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
) -> torch.Tensor:
    """Build oriented bounding box corners for ego trajectories.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.

    Returns:
        (N, T, 4, 2) corner points in global frame.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    dtype = ego_trajs.dtype

    heading = ego_trajs[..., 2:4]  # (N, T, 2)
    heading_unit = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ego_xy = ego_trajs[..., :2]

    wheel_base = ego_shape[0]
    ego_length = ego_shape[1]
    ego_width = ego_shape[2]

    cog_to_rear = 0.5 * wheel_base
    ego_center_xy = ego_xy + heading_unit * cog_to_rear

    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    half_sizes = torch.tensor([half_length, half_width], device=device, dtype=dtype).expand(N, T, 2)

    corner_signs = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    local_corners = corner_signs[None, None, :, :] * half_sizes[:, :, None, :]  # (N, T, 4, 2)

    rot = torch.stack(
        [
            heading_unit[..., 0],
            -heading_unit[..., 1],
            heading_unit[..., 1],
            heading_unit[..., 0],
        ],
        dim=-1,
    ).reshape(N, T, 2, 2)

    rotated_corners = torch.einsum("btij,btkj->btki", rot, local_corners)
    return ego_center_xy[:, :, None, :] + rotated_corners  # (N, T, 4, 2)


_LN_X, _LN_Y = 0, 1
_LN_DX, _LN_DY = 2, 3
_LN_LBX, _LN_LBY = 4, 5
_LN_RBX, _LN_RBY = 6, 7
_LN_MAX_DIST = 30.0


def _point_in_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Ray casting point-in-polygon test.

    Args:
        points: (M, 2) query points.
        polygon: (V, 2) polygon vertices (no need to close — last edge connects
            vertex V-1 back to vertex 0 automatically).

    Returns:
        (M,) bool tensor — True if the point is inside the polygon.
    """
    px, py = points[:, 0:1], points[:, 1:2]  # (M, 1)
    v1 = polygon  # (V, 2)
    v2 = torch.roll(polygon, -1, dims=0)  # (V, 2)

    y1, y2 = v1[:, 1], v2[:, 1]  # (V,)
    x1, x2 = v1[:, 0], v2[:, 0]

    # Does horizontal ray from (px, py) cross edge (v1, v2)?
    cond_y = (y1[None, :] > py) != (y2[None, :] > py)  # (M, V)
    dy = y2 - y1  # (V,) — can be negative, must NOT clamp
    safe_dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy), dy)
    ix = x1[None, :] + (py - y1[None, :]) * (x2[None, :] - x1[None, :]) / safe_dy[None, :]
    cond_x = px < ix
    return ((cond_y & cond_x).sum(dim=1) % 2) == 1  # (M,)


def _points_in_polygons_batched(
    points: torch.Tensor,
    polygons_v1: torch.Tensor,
    polygons_v2: torch.Tensor,
    poly_valid: torch.Tensor,
) -> torch.Tensor:
    """Batched ray casting: check M points against P polygons simultaneously.

    Args:
        points: (M, 2) query points.
        polygons_v1: (P, V, 2) start vertices of each polygon edge.
        polygons_v2: (P, V, 2) end vertices of each polygon edge.
        poly_valid: (P, V) bool — which edges are real (not padding).

    Returns:
        (M, P) bool — True if point m is inside polygon p.
    """
    M = points.shape[0]
    P, V, _ = polygons_v1.shape

    px = points[:, 0:1, None]  # (M, 1, 1)
    py = points[:, 1:2, None]  # (M, 1, 1)

    y1 = polygons_v1[:, :, 1]  # (P, V)
    y2 = polygons_v2[:, :, 1]
    x1 = polygons_v1[:, :, 0]
    x2 = polygons_v2[:, :, 0]

    # (M, P, V)
    cond_y = (y1[None] > py) != (y2[None] > py)
    dy = y2 - y1  # (P, V)
    safe_dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy), dy)
    ix = x1[None] + (py - y1[None]) * (x2[None] - x1[None]) / safe_dy[None]
    cond_x = px < ix

    # Mask out padding edges
    valid = poly_valid[None, :, :]  # (1, P, V)
    crossings = (cond_y & cond_x & valid).sum(dim=2)  # (M, P)
    return (crossings % 2) == 1


def _build_lane_polygons(
    lanes: torch.Tensor,
) -> list[torch.Tensor]:
    """Build closed polygons from lane segment boundaries.

    Each lane segment becomes a polygon: left boundary points forward,
    then right boundary points reversed.

    Args:
        lanes: (S, P, 33) lane tensor.

    Returns:
        List of (V, 2) polygon vertex tensors (only segments with ≥3 valid
        points are included).
    """
    polys: list[torch.Tensor] = []
    for seg_idx in range(lanes.shape[0]):
        pts = lanes[seg_idx, :, :2]
        lb = lanes[seg_idx, :, 4:6]
        rb = lanes[seg_idx, :, 6:8]
        valid = pts.abs().sum(dim=-1) > 0.1
        if valid.sum() < 3:
            continue
        left = (pts + lb)[valid]  # (K, 2)
        right = (pts + rb)[valid]  # (K, 2)
        poly = torch.cat([left, right.flip(0)], dim=0)  # (2K, 2)
        polys.append(poly)
    return polys


@torch.no_grad()
def _ego_on_road_polygon(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    lane_polys: list[torch.Tensor],
) -> torch.Tensor:
    """Check if the ego vehicle is on-road using polygon containment.

    For each timestep, builds the 4 ego bounding-box corners and checks
    whether every corner lies inside at least one lane polygon (ray casting).

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        lane_polys: list of (V, 2) polygon tensors from _build_lane_polygons.

    Returns:
        (N, T) bool tensor — True where the ego is fully on-road.
    """
    if not lane_polys:
        return torch.ones(
            ego_trajs.shape[0], ego_trajs.shape[1], dtype=torch.bool, device=ego_trajs.device
        )

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    half_l = float(ego_shape[1]) / 2
    half_w = float(ego_shape[2]) / 2

    cos_h = ego_trajs[:, :, 2]  # (N, T)
    sin_h = ego_trajs[:, :, 3]
    cx = ego_trajs[:, :, 0]
    cy = ego_trajs[:, :, 1]

    # Sample points along the ego rectangle perimeter for higher resolution.
    # 4 corners + 20 points per side = 84 sample points total.
    _PTS_PER_SIDE = 20
    local_pts: list[tuple[float, float]] = []
    # Front edge (left to right)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((half_l, half_w * (1 - 2 * t)))
    # Right edge (front to rear)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((half_l * (1 - 2 * t), -half_w))
    # Rear edge (right to left)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((-half_l, -half_w * (1 - 2 * t)))
    # Left edge (rear to front)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((-half_l * (1 - 2 * t), half_w))

    local_pts_t = torch.tensor(local_pts, device=device, dtype=ego_trajs.dtype)  # (K, 2)
    K = local_pts_t.shape[0]

    # Rotate + translate all sample points: (N, T, K, 2)
    rx = (
        local_pts_t[:, 0][None, None, :] * cos_h[:, :, None]
        - local_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    )
    ry = (
        local_pts_t[:, 0][None, None, :] * sin_h[:, :, None]
        + local_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    )
    pts_x = cx[:, :, None] + rx  # (N, T, K)
    pts_y = cy[:, :, None] + ry

    all_pts = torch.stack([pts_x, pts_y], dim=-1).reshape(-1, 2)  # (N*T*K, 2)

    # Batch all polygons: pad to same vertex count and run one vectorized check
    traj_center = ego_trajs[:, :, :2].reshape(-1, 2)
    traj_min = traj_center.min(dim=0).values - 10
    traj_max = traj_center.max(dim=0).values + 10

    # Filter nearby polygons by bounding box
    nearby_polys = []
    for poly in lane_polys:
        pmin = poly.min(dim=0).values
        pmax = poly.max(dim=0).values
        if (
            pmax[0] < traj_min[0]
            or pmin[0] > traj_max[0]
            or pmax[1] < traj_min[1]
            or pmin[1] > traj_max[1]
        ):
            continue
        nearby_polys.append(poly)

    if nearby_polys:
        max_v = max(p.shape[0] for p in nearby_polys)
        P = len(nearby_polys)
        padded_v1 = torch.zeros(P, max_v, 2, device=device)
        padded_v2 = torch.zeros(P, max_v, 2, device=device)
        poly_valid = torch.zeros(P, max_v, dtype=torch.bool, device=device)
        for i, poly in enumerate(nearby_polys):
            V = poly.shape[0]
            padded_v1[i, :V] = poly
            padded_v2[i, :V] = torch.roll(poly, -1, dims=0)
            poly_valid[i, :V] = True

        # (M, P) — True if point is inside polygon
        inside_matrix = _points_in_polygons_batched(all_pts, padded_v1, padded_v2, poly_valid)
        inside_any = inside_matrix.any(dim=1)  # (M,)
    else:
        inside_any = torch.zeros(all_pts.shape[0], dtype=torch.bool, device=device)

    # At least 95% of perimeter points must be inside a lane polygon.
    # Requiring 100% is too strict — a few points can protrude 1-2cm past
    # a lane boundary at polygon seams without the ego being truly offroad.
    inside_any = inside_any.reshape(N, T, K)
    _ON_ROAD_THRESHOLD = 0.95
    inside_frac = inside_any.float().mean(dim=-1)  # (N, T)
    on_road = inside_frac >= _ON_ROAD_THRESHOLD  # (N, T)

    # Also compute fraction of points outside for a soft proximity penalty:
    # fraction_outside = 0 means fully on-road, >0 means partially protruding.
    fraction_outside = 1.0 - inside_any.float().mean(dim=-1)  # (N, T)

    # Edge proximity check: sample points on a rectangle EXPANDED by 25cm.
    # If an expanded point is OUTSIDE all lane polygons, the lane boundary
    # is closer than 25cm to the ego at that location.
    _EDGE_MARGIN = 0.25  # metres
    margin_pts: list[tuple[float, float]] = []
    outer_half_l = half_l + _EDGE_MARGIN
    outer_half_w = half_w + _EDGE_MARGIN
    # Front edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((outer_half_l, outer_half_w * (1 - 2 * t)))
    # Right edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((outer_half_l * (1 - 2 * t), -outer_half_w))
    # Rear edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((-outer_half_l, -outer_half_w * (1 - 2 * t)))
    # Left edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((-outer_half_l * (1 - 2 * t), outer_half_w))

    margin_pts_t = torch.tensor(margin_pts, device=device, dtype=ego_trajs.dtype)

    mrx = (
        margin_pts_t[:, 0][None, None, :] * cos_h[:, :, None]
        - margin_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    )
    mry = (
        margin_pts_t[:, 0][None, None, :] * sin_h[:, :, None]
        + margin_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    )
    mpts_x = cx[:, :, None] + mrx
    mpts_y = cy[:, :, None] + mry
    all_margin_pts = torch.stack([mpts_x, mpts_y], dim=-1).reshape(-1, 2)

    if nearby_polys:
        margin_inside_matrix = _points_in_polygons_batched(
            all_margin_pts,
            padded_v1,
            padded_v2,
            poly_valid,
        )
        margin_outside = ~margin_inside_matrix.any(dim=1)
    else:
        margin_outside = torch.ones(all_margin_pts.shape[0], dtype=torch.bool, device=device)

    # Fraction of expanded points that are OUTSIDE = fraction of ego perimeter
    # where the lane boundary is closer than 25cm
    margin_outside = margin_outside.reshape(N, T, K)
    near_edge_penalty = margin_outside.float().mean(dim=-1)  # (N, T)
    # 0 = well inside (all expanded points inside lanes = boundary >25cm away)
    # 1 = entire perimeter near edge (all expanded points outside = boundary <25cm)

    # Second wider margin at 40cm for stronger penalty when ego is very close
    _WIDE_MARGIN = 0.40
    wide_pts: list[tuple[float, float]] = []
    wide_half_l = half_l + _WIDE_MARGIN
    wide_half_w = half_w + _WIDE_MARGIN
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((wide_half_l, wide_half_w * (1 - 2 * t)))
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((wide_half_l * (1 - 2 * t), -wide_half_w))
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((-wide_half_l, -wide_half_w * (1 - 2 * t)))
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((-wide_half_l * (1 - 2 * t), wide_half_w))

    wide_pts_t = torch.tensor(wide_pts, device=device, dtype=ego_trajs.dtype)
    wrx = (
        wide_pts_t[:, 0][None, None, :] * cos_h[:, :, None]
        - wide_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    )
    wry = (
        wide_pts_t[:, 0][None, None, :] * sin_h[:, :, None]
        + wide_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    )
    wpts_x = cx[:, :, None] + wrx
    wpts_y = cy[:, :, None] + wry
    all_wide_pts = torch.stack([wpts_x, wpts_y], dim=-1).reshape(-1, 2)

    if nearby_polys:
        wide_inside = _points_in_polygons_batched(
            all_wide_pts,
            padded_v1,
            padded_v2,
            poly_valid,
        )
        wide_outside = ~wide_inside.any(dim=1)
    else:
        wide_outside = torch.ones(all_wide_pts.shape[0], dtype=torch.bool, device=device)

    wide_outside = wide_outside.reshape(N, T, K)
    wide_edge_penalty = wide_outside.float().mean(dim=-1)  # (N, T)

    return on_road, fraction_outside, near_edge_penalty, wide_edge_penalty


def _build_sg_diff_kernel(
    window: int = 11, poly: int = 3, deriv: int = 3, delta: float = 0.1
) -> torch.Tensor:
    """Build Savitzky-Golay differentiation kernel (precomputed, cached).

    Returns a 1D convolution kernel that computes the deriv-th derivative
    using a local polynomial fit over `window` points.
    Pure numpy implementation — no scipy dependency.
    """
    # SG coefficients via least-squares polynomial fitting
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    # Build Vandermonde matrix
    A = np.vander(x, N=poly + 1, increasing=True)  # [window, poly+1]
    # Pseudo-inverse gives the coefficient extraction matrix
    pinv = np.linalg.pinv(A)  # [poly+1, window]
    # The deriv-th row of pinv gives smoothing coefficients for the deriv-th derivative
    import math as _math

    coeffs = pinv[deriv] * _math.factorial(deriv) / (delta**deriv)
    # Reverse to match convolution convention (scipy savgol_coeffs convention)
    return torch.tensor(coeffs.copy(), dtype=torch.float32).flip(0)  # flip for conv1d


def _closest_points_between_rects(
    rect1: torch.Tensor,
    rect2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest-point pair between two rectangles, vectorised, pure PyTorch.

    For each pair checks all 32 vertex-to-edge queries (16 rect1 corners →
    rect2 edges + 16 rect2 corners → rect1 edges) and returns the vertex +
    its foot on the nearest edge. Exact for non-overlapping rectangles,
    approximate (points on the nearest edges) for overlapping ones — fine
    for visualisation.

    Args:
        rect1: (B, 4, 2) corners in CCW or CW order.
        rect2: (B, 4, 2) corners.

    Returns:
        pt1: (B, 2) closest point on rect1.
        pt2: (B, 2) closest point on rect2.
    """
    B = rect1.shape[0]
    device = rect1.device
    dtype = rect1.dtype

    # Build two vertex-to-edge configurations:
    #   Config A (rect1 vertices → rect2 edges): 4 verts × 4 edges = 16 queries per pair.
    #   Config B (rect2 vertices → rect1 edges): 4 verts × 4 edges = 16 queries per pair.
    # Concatenated → 32 queries per pair, shaped (B, 32, 2) for q/sa/sb.
    r1_edges_a = rect1  # (B, 4, 2)
    r1_edges_b = torch.roll(rect1, -1, dims=1)  # (B, 4, 2)
    r2_edges_a = rect2
    r2_edges_b = torch.roll(rect2, -1, dims=1)

    # Config A flat: 4 verts × 4 edges = 16 (B, 16, 2/2/2)
    vA_q = rect1.unsqueeze(2).expand(-1, -1, 4, -1).reshape(B, 16, 2)  # query pt
    vA_sa = r2_edges_a.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)  # seg start
    vA_sb = r2_edges_b.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)
    # Config B flat
    vB_q = rect2.unsqueeze(2).expand(-1, -1, 4, -1).reshape(B, 16, 2)
    vB_sa = r1_edges_a.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)
    vB_sb = r1_edges_b.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)

    q = torch.cat([vA_q, vB_q], dim=1)  # (B, 32, 2)
    sa = torch.cat([vA_sa, vB_sa], dim=1)
    sb = torch.cat([vA_sb, vB_sb], dim=1)

    # Foot of perpendicular from q onto segment (sa, sb), clamped to [0, 1].
    seg = sb - sa  # (B, 32, 2)
    seg_len2 = (seg * seg).sum(dim=-1).clamp_min(1e-12)
    t = ((q - sa) * seg).sum(dim=-1) / seg_len2
    t = t.clamp(0.0, 1.0)
    foot = sa + t.unsqueeze(-1) * seg  # (B, 32, 2)
    d = (q - foot).norm(dim=-1)  # (B, 32)

    # Best index per pair; in first 16 q is on rect1, in last 16 q is on rect2.
    best = d.argmin(dim=-1)  # (B,)
    arange = torch.arange(B, device=device)
    q_best = q[arange, best]  # (B, 2)
    foot_best = foot[arange, best]  # (B, 2)

    # If best < 16, q is on rect1, foot is on rect2. Else swap.
    is_on_r1 = best < 16
    pt1 = torch.where(is_on_r1.unsqueeze(-1), q_best, foot_best)
    pt2 = torch.where(is_on_r1.unsqueeze(-1), foot_best, q_best)
    return pt1.to(dtype), pt2.to(dtype)


# NOTE: this is the SECOND `_build_lane_polygons` in this module and it
# intentionally shadows the earlier list-of-vertices variant above — this
# edge-returning version is the one bound at module scope and used by
# `compute_lane_departure_penalty`. The shadowing is preserved verbatim from the
# original rlvr.reward (issue #130 pure move). The earlier definition is
# effectively dead; removing it is left to a follow-up cleanup PR so this PR
# stays a behavior-identical move.
def _build_lane_polygons(
    lanes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build lane polygon edges from lane tensor. Vectorized, no python loops.

    Each lane polygon = left boundary edges + right boundary edges (reversed winding)
    + two closing edges connecting the ends. Boundary = center + offset.

    Args:
        lanes: (S, P, D) lane tensor.

    Returns:
        edge_v1: (E, 2) polygon edge start vertices (all polygons concatenated)
        edge_v2: (E, 2) polygon edge end vertices
        edge_poly_id: (E,) int — which polygon each edge belongs to
        n_polys: total number of polygons
    """
    S, P, D = lanes.shape
    device = lanes.device

    center = lanes[..., :2]  # (S, P, 2)
    valid = center.norm(dim=-1) > 1e-3  # (S, P)
    left_pts = center + lanes[..., 4:6]
    right_pts = center + lanes[..., 6:8]

    n_valid = valid.sum(dim=1)
    has_poly = n_valid >= 2

    if not has_poly.any():
        z = torch.zeros(0, 2, device=device)
        return z, z, torch.zeros(0, dtype=torch.int32, device=device), 0

    poly_id_per_lane = torch.cumsum(has_poly.int(), dim=0) - 1  # (S,)
    n_polys = int(has_poly.sum().item())

    # Consecutive boundary edges (left forward, right reversed for winding)
    valid_pair = valid[:, :-1] & valid[:, 1:]  # (S, P-1)
    lane_ids_pair = torch.arange(S, device=device).unsqueeze(1).expand(S, P - 1)
    idx = torch.where(valid_pair.reshape(-1))[0]

    if len(idx) == 0:
        z = torch.zeros(0, 2, device=device)
        return z, z, torch.zeros(0, dtype=torch.int32, device=device), 0

    l_v1 = left_pts[:, :-1].reshape(-1, 2)[idx]
    l_v2 = left_pts[:, 1:].reshape(-1, 2)[idx]
    r_v1 = right_pts[:, 1:].reshape(-1, 2)[idx]  # reversed winding
    r_v2 = right_pts[:, :-1].reshape(-1, 2)[idx]
    edge_pid = poly_id_per_lane[lane_ids_pair.reshape(-1)[idx]]

    # Closing edges: connect left end→right end and right start→left start
    pl = torch.where(has_poly)[0]
    fv = valid.float().argmax(dim=1)[pl]
    lv = P - 1 - valid.flip(1).float().argmax(dim=1)[pl]
    c1_v1 = left_pts[pl, lv]
    c1_v2 = right_pts[pl, lv]
    c2_v1 = right_pts[pl, fv]
    c2_v2 = left_pts[pl, fv]
    c_pid = poly_id_per_lane[pl].int()

    all_v1 = torch.cat([l_v1, r_v1, c1_v1, c2_v1])
    all_v2 = torch.cat([l_v2, r_v2, c1_v2, c2_v2])
    all_pid = torch.cat([edge_pid, edge_pid, c_pid, c_pid]).int()

    return all_v1, all_v2, all_pid, n_polys


def _point_in_polygons(
    points: torch.Tensor,
    edge_v1: torch.Tensor,
    edge_v2: torch.Tensor,
    edge_poly_id: torch.Tensor,
    n_polys: int,
) -> torch.Tensor:
    """GPU-parallel point-in-polygon via ray casting. No python loops.

    Args:
        points: (Q, 2) query points.
        edge_v1, edge_v2: (E, 2) polygon edge endpoints.
        edge_poly_id: (E,) which polygon each edge belongs to.
        n_polys: total number of polygons.

    Returns:
        inside: (Q,) bool — True if inside ANY polygon.
    """
    Q = points.shape[0]
    E = edge_v1.shape[0]
    device = points.device

    if E == 0 or n_polys == 0:
        return torch.zeros(Q, dtype=torch.bool, device=device)

    px = points[:, 0]
    py = points[:, 1]
    v1x, v1y = edge_v1[:, 0], edge_v1[:, 1]
    v2x, v2y = edge_v2[:, 0], edge_v2[:, 1]

    # Prefilter: discard edges that can't be crossed by any query point's +x ray
    keep = (
        (torch.maximum(v1x, v2x) >= px.min())
        & (torch.maximum(v1y, v2y) >= py.min())
        & (torch.minimum(v1y, v2y) <= py.max())
    )

    if not keep.any():
        return torch.zeros(Q, dtype=torch.bool, device=device)

    v1x = v1x[keep]
    v1y = v1y[keep]
    v2x = v2x[keep]
    v2y = v2y[keep]
    edge_poly_id = edge_poly_id[keep]
    E = v1x.shape[0]

    # Chunk over query points when Q×E is large to avoid OOM
    _MAX_QE = 10_000_000  # ~200 MB accounting for multiple intermediates (bool, float, int64 index)
    chunk_size = max(1, _MAX_QE // E) if E > 0 else Q

    if chunk_size >= Q:
        return _pip_core(px, py, v1x, v1y, v2x, v2y, edge_poly_id, E, n_polys, device)

    results = []
    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        results.append(
            _pip_core(
                px[start:end],
                py[start:end],
                v1x,
                v1y,
                v2x,
                v2y,
                edge_poly_id,
                E,
                n_polys,
                device,
            )
        )
    return torch.cat(results)


def _pip_core(
    px: torch.Tensor,
    py: torch.Tensor,
    v1x: torch.Tensor,
    v1y: torch.Tensor,
    v2x: torch.Tensor,
    v2y: torch.Tensor,
    edge_poly_id: torch.Tensor,
    E: int,
    n_polys: int,
    device: torch.device,
) -> torch.Tensor:
    """Core ray-casting kernel for a chunk of query points."""
    Q = px.shape[0]
    py_exp = py[:, None]
    above1 = v1y[None, :] > py_exp
    above2 = v2y[None, :] > py_exp
    straddles = above1 != above2

    dy = (v2y - v1y)[None, :]
    dy_safe = dy.clone()
    dy_safe[dy_safe.abs() < 1e-10] = 1.0
    t = (py_exp - v1y[None, :]) / dy_safe
    x_int = v1x[None, :] + t * (v2x - v1x)[None, :]

    crossing = straddles & (x_int > px[:, None])

    counts = torch.zeros(Q, n_polys, dtype=torch.int32, device=device)
    counts.scatter_add_(1, edge_poly_id[None, :].expand(Q, E).long(), crossing.int())

    inside_any = ((counts % 2) == 1).any(dim=1)
    return inside_any


def _point_to_segments_dist(
    points: torch.Tensor,
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
) -> torch.Tensor:
    """Distance from each point to each segment. Fully parallel on GPU.

    Args:
        points: (Q, 2)
        seg_p1, seg_p2: (E, 2)

    Returns:
        dist: (Q, E) distance matrix.
    """
    seg = seg_p2 - seg_p1
    seg_len2 = (seg**2).sum(-1).clamp(min=1e-10)
    diff = points[:, None, :] - seg_p1[None, :, :]
    t = ((diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]).clamp(0, 1)
    closest = seg_p1[None, :, :] + t[:, :, None] * seg[None, :, :]
    return (points[:, None, :] - closest).norm(dim=-1)


def _point_to_segments_min_dist(
    points: torch.Tensor,
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
) -> torch.Tensor:
    """Min distance from each point to nearest segment. Chunks to avoid OOM.

    Like _point_to_segments_dist but only returns (Q,) min distances
    instead of the full (Q, E) matrix. Chunks over query points when
    Q×E > 10M elements.

    Args:
        points: (Q, 2)
        seg_p1, seg_p2: (E, 2)

    Returns:
        min_dist: (Q,) min distance per point.
    """
    Q = points.shape[0]
    E = seg_p1.shape[0]
    _MAX_QE = 10_000_000
    chunk_size = max(1, _MAX_QE // E) if E > 0 else Q

    if chunk_size >= Q:
        return _point_to_segments_dist(points, seg_p1, seg_p2).min(dim=1).values

    results = []
    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        d = _point_to_segments_dist(points[start:end], seg_p1, seg_p2)
        results.append(d.min(dim=1).values)
    return torch.cat(results)


def _points_inside_intersection_areas(
    points: torch.Tensor,
    polygons_tensor: torch.Tensor,
) -> torch.Tensor:
    """Test whether each point lies inside ANY intersection_area polygon.

    Uses horizontal-ray casting, fully batched.

    Args:
        points: (Q, 2).
        polygons_tensor: (Np, P, 2+K) per-scene polygons (from NPZ `polygons`).
            Per-point validity is derived from ||xy|| > 1e-3. Polygons with
            fewer than 3 valid points are ignored.

    Returns:
        (Q,) bool — True if the point is inside at least one polygon.
    """
    Q = points.shape[0]
    device = points.device
    inside_any = torch.zeros(Q, dtype=torch.bool, device=device)
    if polygons_tensor.shape[-1] < 2:
        return inside_any
    pg_xy = polygons_tensor[..., :2]  # (Np, P, 2)
    pg_valid = pg_xy.norm(dim=-1) > 1e-3  # (Np, P)
    Np = pg_xy.shape[0]
    for p_idx in range(Np):
        mask = pg_valid[p_idx]
        if mask.sum() < 3:
            continue
        verts = pg_xy[p_idx][mask]  # (Pv, 2)
        v1 = verts
        v2 = torch.roll(verts, -1, dims=0)
        px = points[:, 0:1]
        py = points[:, 1:2]
        y1 = v1[:, 1][None, :]
        y2 = v2[:, 1][None, :]
        x1 = v1[:, 0][None, :]
        x2 = v2[:, 0][None, :]
        cond_y = (y1 > py) != (y2 > py)
        denom = y2 - y1
        safe_denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
        x_intersect = x1 + (py - y1) * (x2 - x1) / safe_denom
        cond = cond_y & (x_intersect > px)
        crossings = cond.sum(dim=-1)
        inside_p = (crossings % 2) == 1
        inside_any = inside_any | inside_p
    return inside_any


def _point_to_segments_signed_min_dist(
    points: torch.Tensor,
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
    seg_outward: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each point, find nearest segment and return unsigned + signed distance.

    Signed distance: positive if point is on the outward side of its nearest
    segment, negative if inside. Magnitude equals unsigned distance.

    Fully batched on GPU, chunked to stay under ~10M Q×E elements.

    Args:
        points: (Q, 2).
        seg_p1, seg_p2: (E, 2) segment endpoints.
        seg_outward: (E, 2) outward unit vector per segment (perpendicular
            to segment direction, pointing away from the lane interior).

    Returns:
        unsigned_dist: (Q,) min distance per point.
        signed_dist: (Q,) (query - closest_on_segment) · seg_outward[argmin_seg].
    """
    Q = points.shape[0]
    E = seg_p1.shape[0]
    if E == 0:
        return (
            torch.full((Q,), 100.0, device=points.device, dtype=points.dtype),
            torch.full((Q,), -100.0, device=points.device, dtype=points.dtype),
        )

    seg = seg_p2 - seg_p1  # (E, 2)
    seg_len2 = (seg**2).sum(-1).clamp(min=1e-10)  # (E,)

    _MAX_QE = 10_000_000
    chunk_size = max(1, _MAX_QE // E)

    unsigned_all = []
    signed_all = []

    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        chunk = points[start:end]
        diff = chunk[:, None, :] - seg_p1[None, :, :]
        t_raw = (diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]
        is_unclamped = (t_raw > 0.0) & (t_raw < 1.0)
        t = t_raw.clamp(0, 1)
        closest = seg_p1[None, :, :] + t[:, :, None] * seg[None, :, :]
        to_query = chunk[:, None, :] - closest
        dist = to_query.norm(dim=-1)

        # Find the actually-nearest segment per query (clamped or not).
        min_dist, min_idx = dist.min(dim=1)

        # If the nearest segment's projection is CLAMPED (foot lies at an
        # endpoint), the query is past the segment's endpoint — don't flag as
        # crossing. Falling back to some other distant unclamped segment would
        # produce a spurious outward-projection reading because "outward" is
        # only meaningful perpendicular to the segment.
        nearest_unclamped = is_unclamped.gather(1, min_idx[:, None]).squeeze(-1)

        gathered_to_query = to_query.gather(1, min_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
        outward_for_min = seg_outward[min_idx]
        signed_raw = (gathered_to_query * outward_for_min).sum(-1)
        signed = torch.where(nearest_unclamped, signed_raw, torch.full_like(signed_raw, -100.0))

        unsigned_all.append(min_dist)
        signed_all.append(signed)

    return torch.cat(unsigned_all), torch.cat(signed_all)


def _classify_outer_boundaries(
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
    seg_dir: torch.Tensor,
    seg_lane: torch.Tensor,
    edge_v1: torch.Tensor,
    edge_v2: torch.Tensor,
    edge_poly_id: torch.Tensor,
    n_polys: int,
    nudge: float = 0.05,
    gap_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify boundary segments as outer (road edge) via midpoint nudge + containment.

    For each segment, nudge its midpoint outward (perpendicular to lane direction).
    If the nudged point lands inside any lane polygon → shared boundary.
    If outside but close to a different lane's boundary → junction gap (shared).
    Otherwise → road edge (outer).

    Segments alternate left/right per lane: even=left boundary, odd=right boundary.

    Args:
        seg_p1, seg_p2: (M, 2) boundary segment endpoints.
        seg_dir: (M, 2) unit lane direction at each segment.
        seg_lane: (M,) lane index.
        edge_v1, edge_v2: polygon edge vertices for containment check.
        edge_poly_id: polygon IDs for edges.
        n_polys: total polygon count.
        nudge: outward nudge distance in meters.
        gap_threshold: max distance to different-lane segment to be a junction gap.

    Returns:
        is_outer: (M,) bool.
        outward: (M, 2) outward unit vector per segment (away from lane interior).
    """
    M = seg_p1.shape[0]
    device = seg_p1.device

    # Midpoint of each segment
    mid = (seg_p1 + seg_p2) / 2

    # Outward normal from lane direction: left_normal = (-dy, dx)
    left_normal = torch.stack([-seg_dir[:, 1], seg_dir[:, 0]], dim=-1)

    # Even indices = left boundary → outward = left normal
    # Odd indices = right boundary → outward = -left normal (right normal)
    is_left = torch.arange(M, device=device) % 2 == 0
    outward = torch.where(is_left[:, None], left_normal, -left_normal)
    outward = outward / outward.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    nudged = mid + nudge * outward

    # Check if nudged point is inside any polygon
    inside = _point_in_polygons(nudged, edge_v1, edge_v2, edge_poly_id, n_polys)

    # Inside → shared. Outside → candidate road edge.
    candidate_outer = ~inside

    # At intersections, nudged point may land in gap between polygons.
    # If close to a different lane's boundary segment → junction gap, not road edge.
    if candidate_outer.any():
        nudged_outer = nudged[candidate_outer]
        d = _point_to_segments_dist(nudged_outer, seg_p1, seg_p2)  # (n_cand, M)
        # Mask out same-lane segments
        outer_lane = seg_lane[candidate_outer]
        same_lane_mask = outer_lane[:, None] == seg_lane[None, :]
        d[same_lane_mask] = 999.0
        # Close to different-lane segment → junction gap
        min_d = d.min(dim=1).values
        is_junction_gap = min_d < gap_threshold
        outer_indices = torch.where(candidate_outer)[0]
        candidate_outer[outer_indices[is_junction_gap]] = False

    return candidate_outer, outward


__all__ = [
    "_build_ego_bbox_corners",
    "_LN_X",
    "_LN_Y",
    "_LN_DX",
    "_LN_DY",
    "_LN_LBX",
    "_LN_LBY",
    "_LN_RBX",
    "_LN_RBY",
    "_LN_MAX_DIST",
    "_point_in_polygon",
    "_points_in_polygons_batched",
    "_build_lane_polygons",
    "_ego_on_road_polygon",
    "_build_sg_diff_kernel",
    "_closest_points_between_rects",
    "_point_in_polygons",
    "_pip_core",
    "_point_to_segments_dist",
    "_point_to_segments_min_dist",
    "_points_inside_intersection_areas",
    "_point_to_segments_signed_min_dist",
    "_classify_outer_boundaries",
]
