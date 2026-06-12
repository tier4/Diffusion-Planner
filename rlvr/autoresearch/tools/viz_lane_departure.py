"""Visualize lane departure and road border distance checks.

Shows lane polygons, outer boundary segments, road border polylines, ego
footprint with perimeter points colored by distance, and a line from the worst
point to the nearest boundary.

Modes:
  --mode lane   (default) Lane departure: polygon containment + outer boundary distance
  --mode rb     Road border: point-to-segment distance to border polylines
  --mode both   Side-by-side comparison

Usage:
    python -m rlvr.autoresearch.tools.viz_lane_departure \
        --scenes val_v4_100.json \
        --indices 10 44 46 2 22 \
        --output_dir ~/Pictures/lane_departure_viz \
        [--step auto] [--mode lane]
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon

from preference_optimization.utils import load_npz_data
from rlvr.reward import (
    _LANE_PTS_PER_SIDE,
    RewardConfig,
    _build_lane_polygons,
    _classify_outer_boundaries,
    _point_in_polygons,
    _point_to_segments_dist,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_scene_base(scene_path, step=None):
    """Load scene data and auto-pick step. Shared by lane and rb modes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_npz_data(scene_path, device)
    with np.load(scene_path) as raw:
        gt_raw = raw["ego_agent_future"].copy()

    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width_val = ego_shape[2].item()
    ro = (length - wb) / 2
    half_w = width_val / 2

    # Auto-pick step at curve apex if not specified
    if step is None:
        dh = np.diff(gt_raw[:, 2])
        dh = np.arctan2(np.sin(dh), np.cos(dh))
        cum_yaw = np.cumsum(np.abs(dh))
        total_yaw = cum_yaw[-1]
        step = max(1, min(int(np.searchsorted(cum_yaw, total_yaw * 0.5)), 78))

    cos_h = float(np.cos(gt_raw[step, 2]))
    sin_h = float(np.sin(gt_raw[step, 2]))
    cx, cy = float(gt_raw[step, 0]), float(gt_raw[step, 1])

    dh = np.diff(gt_raw[:, 2])
    total_yaw = np.degrees(np.sum(np.abs(np.arctan2(np.sin(dh), np.cos(dh)))))

    return {
        "scene_path": scene_path,
        "step": step,
        "total_yaw": total_yaw,
        "gt_raw": gt_raw,
        "data": data,
        "device": device,
        "cx": cx,
        "cy": cy,
        "cos_h": cos_h,
        "sin_h": sin_h,
        "length": length,
        "ro": ro,
        "half_w": half_w,
        "width_val": width_val,
        "ego_shape": ego_shape,
    }


def _build_ego_perimeter(base, pts_per_side, skip_corner_dupes=False):
    """Build ego perimeter points at step. Returns world_pts tensor + numpy."""
    device = base["device"]
    ro, length, half_w, width_val = base["ro"], base["length"], base["half_w"], base["width_val"]
    cx, cy, cos_h, sin_h = base["cx"], base["cy"], base["cos_h"], base["sin_h"]

    lp_list = []
    for j in range(pts_per_side):
        f = j / (pts_per_side - 1)
        lp_list.append((-ro + f * length, -half_w))
        lp_list.append((-ro + f * length, half_w))
        if skip_corner_dupes and (f == 0 or f == 1):
            continue
        lp_list.append((-ro, -half_w + f * width_val))
        lp_list.append((length - ro, -half_w + f * width_val))
    local_pts = torch.tensor(lp_list, device=device, dtype=torch.float32)

    rot_m = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], device=device, dtype=torch.float32)
    world_pts = (rot_m @ local_pts.T).T + torch.tensor([cx, cy], device=device, dtype=torch.float32)
    return world_pts, local_pts


def _ego_box_corners(base):
    """Return 4 corners of ego bounding box as numpy array."""
    cx, cy = base["cx"], base["cy"]
    cos_h, sin_h = base["cos_h"], base["sin_h"]
    l, ro, hw = base["length"], base["ro"], base["half_w"]
    return np.array(
        [
            [cx + (l - ro) * cos_h - hw * sin_h, cy + (l - ro) * sin_h + hw * cos_h],
            [cx + (l - ro) * cos_h + hw * sin_h, cy + (l - ro) * sin_h - hw * cos_h],
            [cx - ro * cos_h + hw * sin_h, cy - ro * sin_h - hw * cos_h],
            [cx - ro * cos_h - hw * sin_h, cy - ro * sin_h + hw * cos_h],
        ]
    )


# ---------------------------------------------------------------------------
# Lane departure check
# ---------------------------------------------------------------------------


def check_scene_lane(scene_path, step=None):
    """Run lane departure check using reward.py methods."""
    base = _load_scene_base(scene_path, step)
    device = base["device"]
    data = base["data"]

    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]

    edge_v1, edge_v2, edge_poly_id, n_polys = _build_lane_polygons(lanes)

    center = lanes[..., :2]
    direction = lanes[..., 2:4]
    lb_offset = lanes[..., 4:6]
    rb_offset = lanes[..., 6:8]
    valid = center.norm(dim=-1) > 1e-3
    S = lanes.shape[0]

    sp1_l = []
    sp2_l = []
    sd_l = []
    sl_l = []
    for s_idx in range(S):
        idx = torch.where(valid[s_idx])[0]
        if len(idx) < 2:
            continue
        lp = center[s_idx, idx] + lb_offset[s_idx, idx]
        rp = center[s_idx, idx] + rb_offset[s_idx, idx]
        dirs = direction[s_idx, idx].clone()
        for i in range(1, len(idx)):
            if dirs[i].norm() < 1e-6:
                dirs[i] = dirs[i - 1]
        if dirs[0].norm() < 1e-6 and len(dirs) > 1:
            dirs[0] = dirs[1]
        for i in range(len(idx) - 1):
            md = (dirs[i] + dirs[i + 1]) / 2
            dn = md.norm()
            if dn > 1e-6:
                md = md / dn
            sp1_l.append(lp[i])
            sp2_l.append(lp[i + 1])
            sd_l.append(md)
            sl_l.append(s_idx)
            sp1_l.append(rp[i])
            sp2_l.append(rp[i + 1])
            sd_l.append(md)
            sl_l.append(s_idx)

    has_segs = bool(sp1_l)
    if has_segs:
        seg_p1 = torch.stack(sp1_l)
        seg_p2 = torch.stack(sp2_l)
        seg_dir = torch.stack(sd_l)
        seg_lane = torch.tensor(sl_l, device=device, dtype=torch.int64)

        is_outer, _ = _classify_outer_boundaries(
            seg_p1,
            seg_p2,
            seg_dir,
            seg_lane,
            edge_v1,
            edge_v2,
            edge_poly_id,
            n_polys,
        )
        outer_p1 = seg_p1[is_outer]
        outer_p2 = seg_p2[is_outer]
    else:
        outer_p1 = torch.zeros(0, 2, device=device)
        outer_p2 = torch.zeros(0, 2, device=device)
        is_outer = torch.zeros(0, dtype=torch.bool, device=device)
        seg_p1 = torch.zeros(0, 2, device=device)

    # Ego perimeter (lane uses _LANE_PTS_PER_SIDE with corner dedup)
    world_pts, _ = _build_ego_perimeter(base, _LANE_PTS_PER_SIDE, skip_corner_dupes=True)

    inside = _point_in_polygons(world_pts, edge_v1, edge_v2, edge_poly_id, n_polys)

    if outer_p1.shape[0] > 0:
        d_full = _point_to_segments_dist(world_pts, outer_p1, outer_p2)
        pt_dists = d_full.min(dim=1).values.cpu().numpy()
        worst = int(pt_dists.argmin())
        closest_seg_idx = int(d_full[worst].argmin().item())
        seg_vec = outer_p2[closest_seg_idx] - outer_p1[closest_seg_idx]
        seg_len2 = (seg_vec**2).sum().clamp(min=1e-10)
        t_param = ((world_pts[worst] - outer_p1[closest_seg_idx]) * seg_vec).sum() / seg_len2
        t_param = t_param.clamp(0, 1)
        closest_boundary_pt = (outer_p1[closest_seg_idx] + t_param * seg_vec).cpu().numpy()
    else:
        pt_dists = np.full(world_pts.shape[0], 100.0)
        worst = 0
        closest_boundary_pt = np.array([base["cx"], base["cy"]])

    return {
        **base,
        "mode": "lane",
        "world_pts": world_pts.cpu().numpy(),
        "pt_dists": pt_dists,
        "pt_in_any": inside.cpu().numpy(),
        "worst": worst,
        "wp": world_pts[worst].cpu().numpy(),
        "closest_boundary_pt": closest_boundary_pt,
        "outer_p1": outer_p1.cpu().numpy(),
        "outer_p2": outer_p2.cpu().numpy(),
        "n_outer": int(is_outer.sum()) if has_segs else 0,
        "n_total_segs": seg_p1.shape[0],
    }


# ---------------------------------------------------------------------------
# Road border check
# ---------------------------------------------------------------------------


def check_scene_rb(scene_path, step=None, config=None):
    """Run road border distance check using point-to-segment distance.

    If step is None, auto-picks the timestep where the ego is closest to the
    road border (or first enters the near/wide zone), rather than the curve apex.
    """
    if config is None:
        config = RewardConfig(enable_overprogress=True)

    # First pass: load scene and compute per-timestep min distance to find best step
    base_init = _load_scene_base(scene_path, step=1)  # dummy step
    device = base_init["device"]
    data = base_init["data"]
    gt_raw = base_init["gt_raw"]

    ls = data.get("line_strings")
    has_borders = False
    seg_p1 = torch.zeros(0, 2, device=device)
    seg_p2 = torch.zeros(0, 2, device=device)
    border_polylines = []

    if ls is not None:
        if ls.dim() == 4:
            ls = ls[0]
        if ls.shape[-1] >= 4:
            border_flag = ls[..., 3]
            border_xy = ls[..., :2]
            is_border = border_flag > 0.5
            has_coords = border_xy.norm(dim=-1) > 1e-3
            valid_mask = is_border & has_coords

            valid_pair = valid_mask[:, :-1] & valid_mask[:, 1:]
            seg_idx = torch.where(valid_pair.reshape(-1))[0]
            if seg_idx.shape[0] > 0:
                seg_p1 = border_xy[:, :-1].reshape(-1, 2)[seg_idx]
                seg_p2 = border_xy[:, 1:].reshape(-1, 2)[seg_idx]
                has_borders = True

            for i in range(ls.shape[0]):
                mask = valid_mask[i]
                if mask.sum() > 1:
                    border_polylines.append(border_xy[i, mask].cpu().numpy())

    # Auto-pick step: find the timestep where ego is closest to road border
    if step is None and has_borders:
        T = gt_raw.shape[0]
        ro = base_init["ro"]
        length = base_init["length"]
        half_w = base_init["half_w"]
        width_val = base_init["width_val"]

        # Build perimeter at every timestep along GT
        _PTS = 20
        lp_list = []
        for j in range(_PTS):
            f = j / (_PTS - 1)
            lp_list.append((-ro + f * length, -half_w))
            lp_list.append((-ro + f * length, half_w))
            lp_list.append((-ro, -half_w + f * width_val))
            lp_list.append((length - ro, -half_w + f * width_val))
        local_pts = torch.tensor(lp_list, device=device, dtype=torch.float32)
        K = local_pts.shape[0]

        # GT positions
        gt_t = torch.tensor(gt_raw, device=device, dtype=torch.float32)
        cos_h = torch.cos(gt_t[:, 2])
        sin_h = torch.sin(gt_t[:, 2])
        rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(T, 2, 2)
        rotated = torch.einsum("tij,kj->tki", rot, local_pts)  # (T, K, 2)
        world_all = gt_t[:, :2].unsqueeze(1) + rotated  # (T, K, 2)

        # Min distance per timestep (skip t=0)
        from rlvr.reward import _point_to_segments_min_dist

        flat = world_all.reshape(T * K, 2)
        dists = _point_to_segments_min_dist(flat, seg_p1, seg_p2).reshape(T, K)
        per_ts = dists.min(dim=1).values  # (T,)
        per_ts[0] = 999.0  # skip t=0
        step = int(per_ts.argmin().item())
        step = max(1, min(step, T - 2))

        # Also store per-timestep distances for the full trajectory
        per_ts_np = per_ts.cpu().numpy()
    elif step is None:
        # Fallback: curve apex
        dh = np.diff(gt_raw[:, 2])
        dh = np.arctan2(np.sin(dh), np.cos(dh))
        cum_yaw = np.cumsum(np.abs(dh))
        total_yaw = cum_yaw[-1]
        step = max(1, min(int(np.searchsorted(cum_yaw, total_yaw * 0.5)), 78))
        per_ts_np = None
    else:
        per_ts_np = None

    # Now rebuild base with the correct step
    base = _load_scene_base(scene_path, step=step)
    base["data"] = data  # reuse already-loaded data

    # Ego perimeter at the chosen step (20 per side, no dedup — matches reward.py)
    world_pts, _ = _build_ego_perimeter(base, 20, skip_corner_dupes=False)

    if has_borders:
        d_full = _point_to_segments_dist(world_pts, seg_p1, seg_p2)
        pt_dists = d_full.min(dim=1).values.cpu().numpy()
        worst = int(pt_dists.argmin())
        closest_seg_idx = int(d_full[worst].argmin().item())
        seg_vec = seg_p2[closest_seg_idx] - seg_p1[closest_seg_idx]
        seg_len2 = (seg_vec**2).sum().clamp(min=1e-10)
        t_param = ((world_pts[worst] - seg_p1[closest_seg_idx]) * seg_vec).sum() / seg_len2
        t_param = t_param.clamp(0, 1)
        closest_border_pt = (seg_p1[closest_seg_idx] + t_param * seg_vec).cpu().numpy()
    else:
        pt_dists = np.full(world_pts.shape[0], 100.0)
        worst = 0
        closest_border_pt = np.array([base["cx"], base["cy"]])

    return {
        **base,
        "mode": "rb",
        "config": config,
        "world_pts": world_pts.cpu().numpy(),
        "pt_dists": pt_dists,
        "worst": worst,
        "wp": world_pts[worst].cpu().numpy(),
        "closest_boundary_pt": closest_border_pt,
        "border_polylines": border_polylines,
        "seg_p1": seg_p1.cpu().numpy(),
        "seg_p2": seg_p2.cpu().numpy(),
        "n_segments": seg_p1.shape[0],
        "per_ts_min": per_ts_np,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_common(ax, r, zoom=None):
    """Draw shared elements: lane boundaries, road borders, GT trajectory, ego box."""
    data = r["data"]

    # Lane boundary lines (always show filled polygons for context)
    if "lanes" in data:
        lanes_raw = data["lanes"]
        if lanes_raw.dim() == 4:
            lanes_raw = lanes_raw[0]
        lanes_raw = lanes_raw.cpu().numpy()
        for s in range(lanes_raw.shape[0]):
            c = lanes_raw[s, :, :2]
            v = np.linalg.norm(c, axis=-1) > 1e-3
            if v.sum() < 2:
                continue
            lx = lanes_raw[s, v, 0] + lanes_raw[s, v, 4]
            ly = lanes_raw[s, v, 1] + lanes_raw[s, v, 5]
            rx = lanes_raw[s, v, 0] + lanes_raw[s, v, 6]
            ry = lanes_raw[s, v, 1] + lanes_raw[s, v, 7]
            ax.fill(
                np.concatenate([lx, rx[::-1]]),
                np.concatenate([ly, ry[::-1]]),
                color="lightgreen",
                alpha=0.15,
                zorder=1,
            )
            ax.plot(lx, ly, "b-", linewidth=0.4, alpha=0.25, zorder=4)
            ax.plot(rx, ry, "b-", linewidth=0.4, alpha=0.25, zorder=4)

    # Road borders from line_strings
    if "line_strings" in data:
        ls_d = data["line_strings"]
        if ls_d.dim() == 4:
            ls_d = ls_d[0]
        ls_d = ls_d.cpu().numpy()
        for j in range(ls_d.shape[0]):
            pts = ls_d[j]
            if ls_d.shape[-1] >= 4:
                v = (pts[:, 3] > 0.5) & (np.abs(pts[:, :2]).sum(axis=-1) > 0.01)
            else:
                v = np.abs(pts[:, :2]).sum(axis=-1) > 0.01
            if v.sum() > 1:
                ax.plot(pts[v, 0], pts[v, 1], "r-", linewidth=3, alpha=0.8, zorder=6)

    # GT trajectory
    ax.plot(r["gt_raw"][:, 0], r["gt_raw"][:, 1], "g--", linewidth=2.5, zorder=7)

    # Ego box
    corners = _ego_box_corners(r)
    ax.add_patch(
        MplPolygon(corners, closed=True, fill=False, edgecolor="darkred", linewidth=2.5, zorder=8)
    )

    # Axes
    ax.set_aspect("equal")
    cx, cy = r["cx"], r["cy"]
    if zoom is not None:
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
    else:
        gt_x, gt_y = r["gt_raw"][:, 0], r["gt_raw"][:, 1]
        xc = (gt_x.min() + gt_x.max()) / 2
        yc = (gt_y.min() + gt_y.max()) / 2
        span = max(gt_x.max() - gt_x.min(), gt_y.max() - gt_y.min()) / 2 + 8
        ax.set_xlim(xc - span, xc + span)
        ax.set_ylim(yc - span, yc + span)
    ax.grid(True, alpha=0.2)


def _draw_perimeter_points(ax, r, cmap, norm_v, mark_outside_lane=False, point_size=20):
    """Draw ego perimeter points colored by distance."""
    config = r.get("config") or RewardConfig(enable_overprogress=True)
    for p in range(len(r["pt_dists"])):
        col = cmap(norm_v(r["pt_dists"][p]))
        sz = point_size
        mk = "o"
        if mark_outside_lane and not r["pt_in_any"][p]:
            col = "red"
            mk = "X"
            sz = point_size * 2
        elif not mark_outside_lane:
            if r["pt_dists"][p] < config.rb_cross_thresh:
                col = "red"
                mk = "X"
                sz = point_size * 2
        ax.scatter(
            r["world_pts"][p, 0],
            r["world_pts"][p, 1],
            c=[col],
            s=sz,
            marker=mk,
            zorder=9,
            edgecolors="black",
            linewidths=0.3,
        )


def _draw_distance_line(ax, r, marker_size=12, line_width=2):
    """Draw worst-point-to-boundary distance line with annotation."""
    wp = r["wp"]
    cbp = r["closest_boundary_pt"]
    min_dist = r["pt_dists"][r["worst"]]
    ax.plot(wp[0], wp[1], "kX", markersize=marker_size, markeredgewidth=2, zorder=10)
    ax.plot([wp[0], cbp[0]], [wp[1], cbp[1]], "k-", linewidth=line_width, zorder=10)
    ax.plot(cbp[0], cbp[1], "ko", markersize=max(4, marker_size // 2), zorder=10)
    return min_dist


def draw_scene_lane(result, output_path, zoom=None):
    """Draw lane departure visualization."""
    r = result
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    _draw_common(ax, r, zoom)

    # Outer boundary segments
    for i in range(r["outer_p1"].shape[0]):
        ax.plot(
            [r["outer_p1"][i, 0], r["outer_p2"][i, 0]],
            [r["outer_p1"][i, 1], r["outer_p2"][i, 1]],
            "r-",
            linewidth=1.0,
            alpha=0.4,
            zorder=5,
        )

    cmap_c = cmx.RdYlGn
    norm_v = plt.Normalize(vmin=0, vmax=2.0)
    _draw_perimeter_points(ax, r, cmap_c, norm_v, mark_outside_lane=True)
    min_dist = _draw_distance_line(ax, r)

    ax.annotate(
        f"Dist to road edge: {min_dist:.3f}m\n"
        f"All in lane: {r['pt_in_any'].all()}\n"
        f"Outer segs: {r['n_outer']}/{r['n_total_segs']}",
        xy=(r["wp"][0], r["wp"][1]),
        xytext=(r["wp"][0] + 0.5, r["wp"][1] + 2),
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=2),
        bbox=dict(facecolor="yellow", alpha=0.9),
        zorder=11,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_v)
    plt.colorbar(sm, ax=ax, shrink=0.4, label="Distance to road edge (m)")

    ax.legend(
        handles=[
            Patch(facecolor="lightgreen", alpha=0.3, label="Lane polygon"),
            plt.Line2D([0], [0], color="r", linewidth=1.5, alpha=0.6, label="Outer boundary seg"),
            plt.Line2D([0], [0], color="r", linewidth=3, label="Road border"),
            plt.Line2D([0], [0], color="g", ls="--", linewidth=2.5, label="GT trajectory"),
        ],
        fontsize=10,
        loc="upper left",
    )

    scene_name = os.path.basename(r["scene_path"]).replace(".npz", "")
    ax.set_title(
        f"{scene_name} — step {r['step']}, {r['total_yaw']:.0f} deg curve\n"
        f"Lane departure | Outer segs: {r['n_outer']}/{r['n_total_segs']}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rb_zone(min_dist, config):
    """Classify distance into zone string."""
    if min_dist < config.rb_cross_thresh:
        return "CROSSING"
    elif min_dist < config.rb_near_thresh:
        return "NEAR"
    elif min_dist < config.rb_wide_thresh:
        return "WIDE"
    return "SAFE"


def _draw_rb_panel(ax, r, zoom=None, is_zoomed=False):
    """Draw a single RB panel (used for both overview and zoomed views)."""
    config = r.get("config", RewardConfig(enable_overprogress=True))
    _draw_common(ax, r, zoom)

    # Border segments (thin orange)
    for i in range(r["seg_p1"].shape[0]):
        ax.plot(
            [r["seg_p1"][i, 0], r["seg_p2"][i, 0]],
            [r["seg_p1"][i, 1], r["seg_p2"][i, 1]],
            "-",
            color="orange",
            linewidth=0.8 if not is_zoomed else 1.5,
            alpha=0.3 if not is_zoomed else 0.5,
            zorder=5,
        )

    cmap_c = cmx.RdYlGn
    norm_v = plt.Normalize(vmin=0, vmax=2.0)
    pt_sz = 20 if not is_zoomed else 60
    _draw_perimeter_points(ax, r, cmap_c, norm_v, mark_outside_lane=False, point_size=pt_sz)
    mk_sz = 12 if not is_zoomed else 18
    lw = 2 if not is_zoomed else 3
    min_dist = _draw_distance_line(ax, r, marker_size=mk_sz, line_width=lw)

    zone = _rb_zone(min_dist, config)

    # Annotation — position differently for zoomed view
    if is_zoomed:
        ax.annotate(
            f"Dist: {min_dist:.3f}m | {zone}\n"
            f"cross<{config.rb_cross_thresh}m  near<{config.rb_near_thresh}m  wide<{config.rb_wide_thresh}m",
            xy=(r["wp"][0], r["wp"][1]),
            xytext=(r["wp"][0] + 0.3, r["wp"][1] + 1.0),
            fontsize=12,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=2),
            bbox=dict(facecolor="yellow", alpha=0.9),
            zorder=11,
        )
    else:
        ax.annotate(
            f"Min dist: {min_dist:.3f}m | Zone: {zone}\n"
            f"Thresholds: cross<{config.rb_cross_thresh}m, near<{config.rb_near_thresh}m, wide<{config.rb_wide_thresh}m\n"
            f"Segments: {r['n_segments']}",
            xy=(r["wp"][0], r["wp"][1]),
            xytext=(r["wp"][0] + 0.5, r["wp"][1] + 2),
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=2),
            bbox=dict(facecolor="yellow", alpha=0.9),
            zorder=11,
        )

    return min_dist, zone, cmap_c, norm_v


def draw_scene_rb(result, output_path, zoom=None):
    """Draw road border visualization: overview + zoomed ROI side by side."""
    r = result
    config = r.get("config", RewardConfig(enable_overprogress=True))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 13))

    # Left: overview
    min_dist, zone, cmap_c, norm_v = _draw_rb_panel(ax1, r, zoom=zoom, is_zoomed=False)

    # Right: zoomed to ROI (worst point +/- 4m)
    roi_zoom = 4.0
    # Override center to worst point for the zoom
    r_zoomed = dict(r)
    wp = r["wp"]
    r_zoomed["cx"] = float(wp[0])
    r_zoomed["cy"] = float(wp[1])
    _draw_rb_panel(ax2, r_zoomed, zoom=roi_zoom, is_zoomed=True)

    ax1.set_title("Overview", fontsize=13)
    ax2.set_title(f"Zoomed ROI ({roi_zoom:.0f}m)", fontsize=13)

    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_v)
    plt.colorbar(sm, ax=[ax1, ax2], shrink=0.4, label="Distance to road border (m)")

    ax1.legend(
        handles=[
            Patch(facecolor="lightgreen", alpha=0.15, label="Lane polygon"),
            plt.Line2D([0], [0], color="r", linewidth=3, label="Road border"),
            plt.Line2D([0], [0], color="g", ls="--", linewidth=2.5, label="GT trajectory"),
            plt.Line2D([0], [0], color="orange", linewidth=0.8, alpha=0.3, label="Border segments"),
        ],
        fontsize=9,
        loc="upper left",
    )

    scene_name = os.path.basename(r["scene_path"]).replace(".npz", "")
    fig.suptitle(
        f"{scene_name} — step {r['step']}, {r['total_yaw']:.0f} deg curve\n"
        f"Road border: {min_dist:.3f}m ({zone}) | {r['n_segments']} segments",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_scene_both(lane_result, rb_result, output_path, zoom=None):
    """Draw side-by-side lane departure + road border comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 14))
    cmap_c = cmx.RdYlGn
    norm_v = plt.Normalize(vmin=0, vmax=2.0)

    # Left: lane departure
    r = lane_result
    _draw_common(ax1, r, zoom)
    for i in range(r["outer_p1"].shape[0]):
        ax1.plot(
            [r["outer_p1"][i, 0], r["outer_p2"][i, 0]],
            [r["outer_p1"][i, 1], r["outer_p2"][i, 1]],
            "r-",
            linewidth=1.0,
            alpha=0.4,
            zorder=5,
        )
    _draw_perimeter_points(ax1, r, cmap_c, norm_v, mark_outside_lane=True)
    lane_dist = _draw_distance_line(ax1, r)
    ax1.set_title(
        f"Lane departure — dist={lane_dist:.3f}m, in_lane={r['pt_in_any'].all()}\n"
        f"Outer segs: {r['n_outer']}/{r['n_total_segs']}",
        fontsize=13,
    )

    # Right: road border
    r = rb_result
    config = r.get("config", RewardConfig(enable_overprogress=True))
    _draw_common(ax2, r, zoom)
    for i in range(r["seg_p1"].shape[0]):
        ax2.plot(
            [r["seg_p1"][i, 0], r["seg_p2"][i, 0]],
            [r["seg_p1"][i, 1], r["seg_p2"][i, 1]],
            "-",
            color="orange",
            linewidth=0.8,
            alpha=0.3,
            zorder=5,
        )
    _draw_perimeter_points(ax2, r, cmap_c, norm_v, mark_outside_lane=False)
    rb_dist = _draw_distance_line(ax2, r)
    if rb_dist < config.rb_cross_thresh:
        zone = "CROSSING"
    elif rb_dist < config.rb_near_thresh:
        zone = "NEAR"
    elif rb_dist < config.rb_wide_thresh:
        zone = "WIDE"
    else:
        zone = "SAFE"
    ax2.set_title(
        f"Road border — dist={rb_dist:.3f}m, zone={zone}\n"
        f"{r['n_segments']} segments | thresholds: cross<{config.rb_cross_thresh}, near<{config.rb_near_thresh}, wide<{config.rb_wide_thresh}m",
        fontsize=13,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_v)
    plt.colorbar(sm, ax=[ax1, ax2], shrink=0.4, label="Distance to boundary (m)")

    scene_name = os.path.basename(lane_result["scene_path"]).replace(".npz", "")
    fig.suptitle(
        f"{scene_name} — step {lane_result['step']}, {lane_result['total_yaw']:.0f} deg curve",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy API (backwards compat)
# ---------------------------------------------------------------------------


def check_scene(scene_path, step=None):
    """Legacy API — runs lane departure check."""
    return check_scene_lane(scene_path, step)


def draw_scene(result, output_path, zoom=None):
    """Legacy API — draws lane departure."""
    draw_scene_lane(result, output_path, zoom)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize lane departure / road border check")
    parser.add_argument("--scenes", type=str, required=True, help="Path to scene list JSON")
    parser.add_argument(
        "--indices", type=int, nargs="+", default=None, help="Scene indices to visualize"
    )
    parser.add_argument(
        "--n_scenes", type=int, default=5, help="Number of scenes (if indices not given)"
    )
    parser.add_argument("--output_dir", type=str, default="~/Pictures/lane_departure_viz")
    parser.add_argument("--step", type=int, default=None, help="Timestep (None = auto curve apex)")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom to ego +/- meters")
    parser.add_argument(
        "--mode",
        type=str,
        default="lane",
        choices=["lane", "rb", "both"],
        help="Visualization mode: lane (default), rb (road border), both (side-by-side)",
    )
    args = parser.parse_args()

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        indices = args.indices
    else:
        indices = list(range(min(args.n_scenes, len(scenes))))

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for idx in indices:
        sp = scenes[idx]
        print(f"Processing scene {idx} ({os.path.basename(sp)})...")
        try:
            if args.mode == "lane":
                result = check_scene_lane(sp, step=args.step)
                fname = os.path.join(out_dir, f"scene{idx:03d}_lane.png")
                draw_scene_lane(result, fname, zoom=args.zoom)
                status = "IN" if result["pt_in_any"].all() else "OUT"
                print(
                    f"  {status}, dist={result['pt_dists'][result['worst']]:.3f}m, "
                    f"outer={result['n_outer']}/{result['n_total_segs']}, "
                    f"yaw={result['total_yaw']:.0f} deg"
                )
            elif args.mode == "rb":
                result = check_scene_rb(sp, step=args.step)
                fname = os.path.join(out_dir, f"scene{idx:03d}_rb.png")
                draw_scene_rb(result, fname, zoom=args.zoom)
                min_d = result["pt_dists"][result["worst"]]
                config = result.get("config", RewardConfig(enable_overprogress=True))
                if min_d < config.rb_cross_thresh:
                    zone = "CROSSING"
                elif min_d < config.rb_near_thresh:
                    zone = "NEAR"
                elif min_d < config.rb_wide_thresh:
                    zone = "WIDE"
                else:
                    zone = "SAFE"
                print(
                    f"  {zone}, dist={min_d:.3f}m, segments={result['n_segments']}, "
                    f"yaw={result['total_yaw']:.0f} deg"
                )
            else:  # both
                lane_r = check_scene_lane(sp, step=args.step)
                rb_r = check_scene_rb(sp, step=args.step)
                fname = os.path.join(out_dir, f"scene{idx:03d}_both.png")
                draw_scene_both(lane_r, rb_r, fname, zoom=args.zoom)
                lane_d = lane_r["pt_dists"][lane_r["worst"]]
                rb_d = rb_r["pt_dists"][rb_r["worst"]]
                print(f"  lane={lane_d:.3f}m, rb={rb_d:.3f}m, yaw={lane_r['total_yaw']:.0f} deg")
            print(f"  Saved: {fname}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
