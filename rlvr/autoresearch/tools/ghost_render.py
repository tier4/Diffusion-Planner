"""Ghost-racing dual-ego renderer.

Reads two completed `replay.py` outputs (ours = "primary", e.g. the model under
test, plus one "ghost" overlay, e.g. baseline) and re-renders each step with both
egos drawn in the same world frame on a fixed viewport. The primary sim
provides the world-state context (lanes, road borders, route, NPCs); the
ghost ego is overlaid at its own world pose at that step.

Both runs must use the same route + spawn config (same seed, same
max_active_npcs). Per-step NPZs are read from the primary sim.

Output: per-step PNG with the same numbering as the source frames + a
single trajectory_log_diff.json summarising the lateral gap each step,
plus an optional WebM via ffmpeg.

Usage:
    python -m rlvr.autoresearch.tools.ghost_render \
        --primary_dir <path/to/route_sim_perfect_mt> \
        --ghost_dir   <path/to/baseline_perfect_mt> \
        --output_dir  <path/to/dual_ego_perfect_mt> \
        --primary_label model_a --ghost_label baseline \
        --view_half_m 40 --webm
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


# Fixed colours for the two egos. RGB hex.
PRIMARY_COLOR = "#1f77b4"  # blue, fully opaque box for the primary sim
GHOST_COLOR = "#d62728"    # red, semi-transparent for the ghost overlay
ROUTE_COLOR = "#ffaa00"
LANE_COLOR = "#888888"
BORDER_COLOR = "#cc0000"


def _world_from_ego(ego_pts: np.ndarray, world_x: float, world_y: float,
                    world_heading: float) -> np.ndarray:
    """ego_pts: (N, 2) in ego frame -> (N, 2) in world frame."""
    c = math.cos(world_heading)
    s = math.sin(world_heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return ego_pts @ rot.T + np.array([world_x, world_y], dtype=np.float64)


def _draw_oriented_box(ax, x: float, y: float, heading: float,
                        length: float, width: float, color: str,
                        alpha: float = 1.0, lw: float = 2.0,
                        zorder: int = 20, label: str | None = None) -> None:
    """Draw an oriented rectangle centered on (x, y) (NOT rear axle)."""
    c = math.cos(heading)
    s = math.sin(heading)
    half_l = length / 2
    half_w = width / 2
    corners_local = np.array([
        [-half_l, -half_w], [half_l, -half_w],
        [half_l, half_w], [-half_l, half_w], [-half_l, -half_w],
    ])
    rot = np.array([[c, -s], [s, c]])
    corners = corners_local @ rot.T + np.array([x, y])
    ax.fill(corners[:, 0], corners[:, 1], color=color, alpha=alpha * 0.35,
            zorder=zorder)
    ax.plot(corners[:, 0], corners[:, 1], color=color, lw=lw, alpha=alpha,
            zorder=zorder + 1, label=label)
    arrow_len = max(length * 0.6, 2.0)
    ax.annotate("",
        xy=(x + arrow_len * c, y + arrow_len * s), xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=12, alpha=alpha),
        zorder=zorder + 2)


def _draw_polylines(ax, polys_world: list[np.ndarray], color: str,
                    lw: float = 1.0, alpha: float = 0.7,
                    zorder: int = 4) -> None:
    for pl in polys_world:
        if pl.shape[0] < 2:
            continue
        ax.plot(pl[:, 0], pl[:, 1], "-", color=color, lw=lw, alpha=alpha,
                zorder=zorder)


def _extract_polylines_from_tensor(tensor: np.ndarray, valid_ch: int = 3,
                                    xy_cols=(0, 1)) -> list[np.ndarray]:
    """tensor: (N, 20, C). Returns list of (M, 2) ego-frame polylines."""
    polys = []
    for i in range(tensor.shape[0]):
        pts = tensor[i, :, list(xy_cols)].T  # (20, 2)
        if valid_ch is not None:
            valid = tensor[i, :, valid_ch] > 0.5
            valid &= np.abs(pts).sum(axis=1) > 0.01
        else:
            valid = np.abs(pts).sum(axis=1) > 0.01
        if valid.sum() < 2:
            continue
        polys.append(pts[valid].astype(np.float64))
    return polys


def _polyline_lane_or_route(tensor: np.ndarray) -> list[np.ndarray]:
    """Lane / route_lane tensors have NO explicit valid channel — strip zeros."""
    polys = []
    for i in range(tensor.shape[0]):
        pts = tensor[i, :, :2]
        valid = np.abs(pts).sum(axis=1) > 0.01
        if valid.sum() < 2:
            continue
        polys.append(pts[valid].astype(np.float64))
    return polys


def _polyline_route_with_tl(tensor: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """Like _polyline_lane_or_route but also returns the dominant TL channel
    (0=GREEN / 1=YELLOW / 2=RED / 4=NONE) per polyline, read from
    channels [8:13] of the lane tensor (TL one-hot block written by
    ``TrafficLightController.tick``)."""
    polys: list[np.ndarray] = []
    tl_states: list[int] = []
    for i in range(tensor.shape[0]):
        pts = tensor[i, :, :2]
        valid = np.abs(pts).sum(axis=1) > 0.01
        if valid.sum() < 2:
            continue
        polys.append(pts[valid].astype(np.float64))
        if tensor.shape[2] >= 13:
            tl_block = tensor[i, valid, 8:13]
            # Argmax across timesteps → most-common channel for this polyline.
            ch_counts = tl_block.sum(axis=0)
            tl_states.append(int(np.argmax(ch_counts)))
        else:
            tl_states.append(4)  # TL_NONE
    return polys, tl_states


# Map TL channel index → hex colour (mirrors traffic_light.TL_HEX).
_TL_COLOUR = {0: "#22bb22", 1: "#ddaa00", 2: "#dd2222", 3: "#aaaaaa", 4: None}


def _render_one_step(
    primary_npz_path: Path,
    primary_pose: dict,
    ghost_pose: dict,
    out_path: Path,
    step: int,
    n_steps: int,
    view_half_m: float,
    primary_label: str,
    ghost_label: str,
    ego_length: float,
    ego_width: float,
    center_mode: str = "midpoint",
    adaptive_view: bool = True,
    max_view_half_m: float = 120.0,
) -> dict:
    """Render one frame and return per-step diff metrics."""
    # Load primary scene NPZ (ego-frame).
    npz = np.load(primary_npz_path, allow_pickle=False)
    lanes_ego = npz["lanes"]
    route_ego = npz["route_lanes"]
    line_strings_ego = npz["line_strings"]

    # Primary world pose.
    px = float(primary_pose["x"])
    py = float(primary_pose["y"])
    ph = float(primary_pose["heading"])

    # Ghost world pose.
    gx = float(ghost_pose["x"])
    gy = float(ghost_pose["y"])
    gh = float(ghost_pose["heading"])

    # Transform ego-frame polylines to world frame using PRIMARY ego pose.
    lane_polys_ego, lane_tl_states = _polyline_route_with_tl(lanes_ego)
    route_polys_ego, route_tl_states = _polyline_route_with_tl(route_ego)
    border_polys_ego = _extract_polylines_from_tensor(
        line_strings_ego, valid_ch=3, xy_cols=(0, 1)
    )
    lane_polys_world = [_world_from_ego(p, px, py, ph) for p in lane_polys_ego]
    route_polys_world = [_world_from_ego(p, px, py, ph) for p in route_polys_ego]
    border_polys_world = [
        _world_from_ego(p, px, py, ph) for p in border_polys_ego
    ]

    # Fixed-size figure with FIXED axes margins (NOT tight_layout) to avoid
    # per-frame grid-size jitter. Figure size in inches × dpi = pixel size.
    fig = Figure(figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor("#f8f8f8")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.92)
    ax = fig.add_subplot(1, 1, 1)

    _draw_polylines(ax, lane_polys_world, color=LANE_COLOR, lw=0.7,
                    alpha=0.55, zorder=2)
    _draw_polylines(ax, border_polys_world, color=BORDER_COLOR, lw=1.6,
                    alpha=0.85, zorder=3)
    _draw_polylines(ax, route_polys_world, color=ROUTE_COLOR, lw=2.4,
                    alpha=0.55, zorder=4)

    # Overlay TL state on lanes that carry one (dominant channel != NONE).
    # Tinted segment + filled circle at lane mid for a clear signal marker.
    for poly, tl in zip(lane_polys_world, lane_tl_states):
        col = _TL_COLOUR.get(tl)
        if col is None:
            continue
        ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=2.2, alpha=0.85, zorder=6)
        mid = poly[len(poly) // 2]
        ax.scatter([mid[0]], [mid[1]], s=70, c=col, edgecolors="black",
                    linewidths=0.6, zorder=7)
    # Same for the route (signals along the ego's planned path).
    for poly, tl in zip(route_polys_world, route_tl_states):
        col = _TL_COLOUR.get(tl)
        if col is None:
            continue
        ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=3.2, alpha=0.95, zorder=6)

    # Trail: last 30 steps of each ego (if available — caller passes them in
    # the pose dicts as `trail`).
    primary_trail = primary_pose.get("trail")
    ghost_trail = ghost_pose.get("trail")
    if primary_trail is not None and len(primary_trail) > 1:
        pt = np.asarray(primary_trail)
        ax.plot(pt[:, 0], pt[:, 1], "-", color=PRIMARY_COLOR,
                lw=1.4, alpha=0.5, zorder=5)
    if ghost_trail is not None and len(ghost_trail) > 1:
        gt = np.asarray(ghost_trail)
        ax.plot(gt[:, 0], gt[:, 1], "--", color=GHOST_COLOR,
                lw=1.4, alpha=0.5, zorder=5)

    # Both ego boxes.
    _draw_oriented_box(ax, px, py, ph, ego_length, ego_width,
                       color=PRIMARY_COLOR, alpha=0.95, lw=2.2,
                       zorder=20, label=primary_label)
    _draw_oriented_box(ax, gx, gy, gh, ego_length, ego_width,
                       color=GHOST_COLOR, alpha=0.55, lw=1.8,
                       zorder=19, label=ghost_label)

    # Connecting line between the two ego centers.
    ax.plot([px, gx], [py, gy], "k:", lw=1.0, alpha=0.55, zorder=18)

    # Choose viewport center.
    if center_mode == "primary":
        cx, cy = px, py
    elif center_mode == "ghost":
        cx, cy = gx, gy
    else:  # midpoint
        cx, cy = (px + gx) / 2.0, (py + gy) / 2.0

    # Choose viewport size. With --adaptive_view, snap to discrete steps so
    # the camera doesn't wobble per-frame.
    if adaptive_view:
        # Need to fit both egos with margin. Half-extent from center to ego.
        max_dx = max(abs(px - cx), abs(gx - cx))
        max_dy = max(abs(py - cy), abs(gy - cy))
        needed = max(max_dx, max_dy) + 25.0  # 25 m margin
        # Snap up to nearest 20 m step; clamp to [view_half_m, max_view_half_m].
        snap = math.ceil(needed / 20.0) * 20.0
        view_h = float(min(max(view_half_m, snap), max_view_half_m))
    else:
        view_h = view_half_m

    ax.set_xlim(cx - view_h, cx + view_h)
    ax.set_ylim(cy - view_h, cy + view_h)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)

    # Diff metrics.
    dx = gx - px
    dy = gy - py
    dist = math.hypot(dx, dy)
    # Lateral component = projection onto primary's right vector.
    right = np.array([math.sin(ph), -math.cos(ph)])
    lateral = float(dx * right[0] + dy * right[1])  # positive = ghost on primary's right
    longitudinal = float(dx * math.cos(ph) + dy * math.sin(ph))  # positive = ahead of primary
    speed_p = float(primary_pose.get("speed", 0.0))
    speed_g = float(ghost_pose.get("speed", 0.0))
    title = (
        f"step {step:04d}/{n_steps}  t={step * 0.1:.1f}s   "
        f"{primary_label} vs {ghost_label}\n"
        f"sep={dist:.2f} m   lat={lateral:+.2f} m   lon={longitudinal:+.2f} m   "
        f"v_p={speed_p:.1f} m/s   v_g={speed_g:.1f} m/s"
    )
    ax.set_title(title, fontsize=10, family="monospace")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.savefig(out_path, dpi=100)
    fig.clf()

    return {
        "step": step,
        "sep_m": dist,
        "lat_m": lateral,
        "lon_m": longitudinal,
        "p_x": px, "p_y": py, "p_h": ph, "p_v": speed_p,
        "g_x": gx, "g_y": gy, "g_h": gh, "g_v": speed_g,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary_dir", type=Path, required=True,
                    help="Path to the primary route_sim dir (must contain "
                         "trajectory_log.json + npz/replay_step_*.npz)")
    ap.add_argument("--ghost_dir", type=Path, required=True,
                    help="Path to the ghost route_sim dir (only "
                         "trajectory_log.json is needed)")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--primary_label", type=str, default="primary")
    ap.add_argument("--ghost_label", type=str, default="ghost")
    ap.add_argument("--view_half_m", type=float, default=40.0)
    ap.add_argument("--center", type=str, default="midpoint",
                    choices=["primary", "ghost", "midpoint"],
                    help="Where to center the camera. 'midpoint' adapts the "
                         "viewport so both egos stay visible.")
    ap.add_argument("--adaptive_view", action="store_true",
                    help="Grow viewport from --view_half_m up to --max_view_half_m "
                         "in discrete 20m steps to keep both egos in frame.")
    ap.add_argument("--max_view_half_m", type=float, default=120.0)
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--end_step", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="Render every Nth frame (default 1).")
    ap.add_argument("--ego_length", type=float, default=7.2369)
    ap.add_argument("--ego_width", type=float, default=2.29156)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--webm", action="store_true",
                    help="Encode dual_ego.webm via ffmpeg after rendering.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    p_traj = json.loads((args.primary_dir / "trajectory_log.json").read_text())
    g_traj = json.loads((args.ghost_dir / "trajectory_log.json").read_text())
    p_by_step = {int(e["step"]): e for e in p_traj}
    g_by_step = {int(e["step"]): e for e in g_traj}

    # Frames common to both runs (by step index).
    common_steps = sorted(set(p_by_step) & set(g_by_step))
    if args.end_step is not None:
        common_steps = [s for s in common_steps if s <= args.end_step]
    common_steps = [s for s in common_steps if s >= args.start_step]
    common_steps = common_steps[::args.stride]
    if not common_steps:
        raise RuntimeError("No common steps between primary and ghost.")
    n_steps = max(common_steps) + 1
    print(f"Rendering {len(common_steps)} steps "
          f"(primary={len(p_traj)}, ghost={len(g_traj)}, "
          f"common range=[{common_steps[0]}, {common_steps[-1]}])")

    # Build trail arrays once.
    p_world = np.array([[e["x"], e["y"]] for e in p_traj], dtype=np.float64)
    g_world = np.array([[e["x"], e["y"]] for e in g_traj], dtype=np.float64)
    p_step_idx = {int(e["step"]): i for i, e in enumerate(p_traj)}
    g_step_idx = {int(e["step"]): i for i, e in enumerate(g_traj)}

    npz_dir = args.primary_dir / "npz"
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"Missing npz dir: {npz_dir}")

    diff_log: list[dict] = []

    def _job(step):
        primary_pose = dict(p_by_step[step])
        ghost_pose = dict(g_by_step[step])
        # Trails (last 30 steps).
        pi = p_step_idx[step]
        gi = g_step_idx[step]
        primary_pose["trail"] = p_world[max(0, pi - 30):pi + 1]
        ghost_pose["trail"] = g_world[max(0, gi - 30):gi + 1]
        npz_path = npz_dir / f"replay_step_{step:04d}.npz"
        if not npz_path.exists():
            return None
        out_path = args.output_dir / f"step_{step:04d}.png"
        return _render_one_step(
            npz_path, primary_pose, ghost_pose, out_path,
            step, n_steps, args.view_half_m,
            args.primary_label, args.ghost_label,
            args.ego_length, args.ego_width,
            center_mode=args.center,
            adaptive_view=args.adaptive_view,
            max_view_half_m=args.max_view_half_m,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(_job, common_steps)):
            if res is not None:
                diff_log.append(res)
            if i % 200 == 0:
                print(f"  rendered {i}/{len(common_steps)}")

    diff_path = args.output_dir / "trajectory_log_diff.json"
    diff_path.write_text(json.dumps(diff_log))
    print(f"Wrote diff log: {diff_path} ({len(diff_log)} entries)")
    if diff_log:
        sep = np.array([e["sep_m"] for e in diff_log])
        lat = np.array([e["lat_m"] for e in diff_log])
        print(f"  sep_m  mean={sep.mean():.2f}  p50={np.percentile(sep,50):.2f}  "
              f"p95={np.percentile(sep,95):.2f}  max={sep.max():.2f}")
        print(f"  lat_m  mean={lat.mean():+.2f}  p5={np.percentile(lat,5):+.2f}  "
              f"p95={np.percentile(lat,95):+.2f}  abs_max={np.abs(lat).max():.2f}")

    if args.webm:
        webm_path = args.output_dir / "dual_ego.webm"
        cmd = [
            "ffmpeg", "-y", "-framerate", "10",
            "-i", str(args.output_dir / "step_%04d.png"),
            "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p",
            str(webm_path),
        ]
        print("Encoding WebM:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Wrote {webm_path}")


if __name__ == "__main__":
    main()
