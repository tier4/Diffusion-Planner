"""Render REALIZED closed-loop RB-crossing regions, baseline vs model, dual-ego.

Extracts both bags' realized ego world trajectories, projects to the route arc,
finds arc windows where EITHER model's ego perimeter crosses a road border
(reward distance < rb_cross_thresh), and renders an arc-synced clip per window:
both ego footprints (red OUTLINE on the frames where that model is crossing) over
the actual world road borders + route centerline. Same footprint+borders style as
the perfect-track ghost sims. WebM (VP9) per crossing region.

Reuses: _extract_poses_from_bag, build_route_polyline/project_to_polyline,
LaneletSceneBuilder.road_border_polylines, _obb_corners, reward._point_to_segments_min_dist.
"""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from scenario_generation.tools._heatmap_common import build_route_polyline, load_route, project_to_polyline
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder, _obb_corners
from scenario_generation.tools.heatmap_route_deviation import _extract_poses_from_bag
from rlvr.reward import _point_to_segments_min_dist, RewardConfig

BASE_C, MODEL_C, CROSS_C, NB_BORDER, ROUTE_C = "#1f77b4", "#d62728", "#ff0000", "#cc0000", "#9467bd"


def _peri(c, n=8):
    o = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        for t in np.linspace(0, 1, n, endpoint=False): o.append(a * (1 - t) + b * t)
    return np.array(o, dtype=np.float32)


def _series(bag, seg1, seg2, pts, arc, shape, stride):
    WB, L, W = shape
    poses = _extract_poses_from_bag(Path(bag))[::stride]
    out = []
    for x, y, yaw, _ in poses:
        c = _obb_corners(float(x), float(y), float(yaw), L, W, wheelbase=WB)
        d = _point_to_segments_min_dist(torch.tensor(_peri(c)), seg1, seg2).min().item()
        a = float(project_to_polyline(np.array([float(x), float(y)]), pts, arc)[0])
        out.append((float(x), float(y), float(yaw), a, d, c))
    return out


def _draw_box(ax, corners, color, crossing, z):
    poly = np.vstack([corners, corners[:1]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.35, zorder=z)
    ax.plot(poly[:, 0], poly[:, 1], "-", color=(CROSS_C if crossing else color),
            lw=(3 if crossing else 1.5), zorder=z + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", required=True)
    ap.add_argument("--baseline_bag", required=True)
    ap.add_argument("--model_bag", required=True)
    ap.add_argument("--model_label", default="FINAL")
    ap.add_argument("--ego_shape", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--front_cut", type=float, default=50.0, help="skip first N m of route (ends not in-bounds)")
    ap.add_argument("--tail_cut", type=float, default=50.0, help="skip last N m of route")
    ap.add_argument("--view_half", type=float, default=22.0)
    ap.add_argument("--pad_m", type=float, default=60.0, help="context around each crossing window")
    ap.add_argument("--lane_clip_pad", type=float, default=50.0, help="spatial pad (m) when pre-filtering lanes/borders to crossing regions")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    shape = [float(x) for x in args.ego_shape.split(",")]
    thresh = RewardConfig().rb_cross_thresh
    route = load_route(Path(args.route))
    b = LaneletSceneBuilder(str(route.map_path))
    s1, s2 = [], []
    for pl in b.road_border_polylines():
        pl = np.asarray(pl)[:, :2]
        if pl.shape[0] >= 2: s1.append(pl[:-1]); s2.append(pl[1:])
    if not s1:
        raise SystemExit(f"map {route.map_path} has no road-border polylines (>=2 points) — cannot render RB crossings")
    seg1 = torch.tensor(np.concatenate(s1), dtype=torch.float32)
    seg2 = torch.tensor(np.concatenate(s2), dtype=torch.float32)
    borders = [np.asarray(pl)[:, :2] for pl in b.road_border_polylines()]
    # map lane centerlines (all loaded lanelets, world frame)
    lanes = []
    for ll in b.lanelet_ids():
        cl = np.asarray(b.raw_centerline(ll))[:, :2]
        if cl.shape[0] >= 2:
            lanes.append(cl)
    pts, arc = build_route_polyline(route)
    amax = float(arc.max())

    base = _series(args.baseline_bag, seg1, seg2, pts, arc, shape, args.stride)
    mod = _series(args.model_bag, seg1, seg2, pts, arc, shape, args.stride)

    # crossing arcs (in-bounds) from EITHER model -> merge into windows
    xa = sorted([r[3] for r in base + mod if r[4] < thresh and args.front_cut <= r[3] <= amax - args.tail_cut])
    if not xa:
        print("no in-bounds crossings to render"); return
    windows = []
    lo = xa[0]; prev = xa[0]
    for a in xa[1:]:
        if a - prev > 120: windows.append((lo, prev)); lo = a
        prev = a
    windows.append((lo, prev))

    # pre-filter lanes/borders to the crossing regions only (perf: avoid per-frame scan of all map lanelets)
    region_xy = np.array([[r[0], r[1]] for r in mod
                          if any(wlo - args.pad_m <= r[3] <= whi + args.pad_m for wlo, whi in windows)])
    if len(region_xy):
        xmin, ymin = region_xy.min(0) - args.lane_clip_pad
        xmax, ymax = region_xy.max(0) + args.lane_clip_pad
        inbox = lambda pl: pl.shape[0] >= 2 and (
            (pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) & (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)).any()
        lanes = [pl for pl in lanes if inbox(pl)]
        borders = [pl for pl in borders if inbox(pl)]

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for wi, (wlo, whi) in enumerate(windows):
        a0, a1 = wlo - args.pad_m, whi + args.pad_m
        # arc-synced frames: step the model's poses in the window
        frames = [r for r in mod if a0 <= r[3] <= a1]
        if not frames:
            print(f"window {wi}: arc {int(wlo)}-{int(whi)}m has no model poses in [{a0:.0f},{a1:.0f}]m — skipping")
            continue
        wd = out / f"win{wi}_arc{int(wlo)}-{int(whi)}"; wd.mkdir(exist_ok=True)
        for fi, mr in enumerate(frames):
            ma = mr[3]
            br = min(base, key=lambda r: abs(r[3] - ma))  # nearest-arc baseline pose
            cx, cy = mr[0], mr[1]
            fig = Figure(figsize=(11, 11)); ax = fig.add_subplot(111); fig.patch.set_facecolor("#f8f8f8")
            inview = lambda pl: pl.shape[0] >= 2 and (
                (pl[:, 0] >= cx - 40) & (pl[:, 0] <= cx + 40) & (pl[:, 1] >= cy - 40) & (pl[:, 1] <= cy + 40)).any()
            ll = [pl for pl in lanes if inview(pl)]
            if ll: ax.add_collection(LineCollection(ll, colors="#999999", linewidths=0.8, alpha=0.45, zorder=2))
            bl = [pl for pl in borders if inview(pl)]
            if bl: ax.add_collection(LineCollection(bl, colors=NB_BORDER, linewidths=2.2, alpha=0.9, zorder=5))
            m = (arc >= ma - 40) & (arc <= ma + 40)
            ax.plot(pts[m, 0], pts[m, 1], "-", color=ROUTE_C, lw=2.0, alpha=0.5, zorder=3)
            _draw_box(ax, br[5], BASE_C, br[4] < thresh, 10)
            _draw_box(ax, mr[5], MODEL_C, mr[4] < thresh, 14)
            ax.plot([], [], "-", color=BASE_C, lw=2, label=f"baseline  border={br[4]:.2f}m")
            ax.plot([], [], "-", color=MODEL_C, lw=2, label=f"{args.model_label}  border={mr[4]:.2f}m")
            ax.set_xlim(cx - args.view_half, cx + args.view_half); ax.set_ylim(cy - args.view_half, cy + args.view_half)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.15); ax.legend(fontsize=10, loc="upper left")
            ax.set_title(f"RB crossing region  arc~{int(ma)}m   (red outline = ego crossing border, thresh {thresh:.2f}m)", fontsize=11)
            fig.tight_layout(); fig.savefig(wd / f"f{fi:04d}.png", dpi=100); fig.clf()
        webm = out / f"win{wi}_arc{int(wlo)}-{int(whi)}.webm"
        r = subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(wd / "f%04d.png"),
                            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1",
                            "-pix_fmt", "yuv420p", str(webm)], capture_output=True)
        if r.returncode != 0 or not webm.exists():
            raise RuntimeError(
                f"ffmpeg failed for window {wi} (rc={r.returncode}); no WebM written.\n"
                f"{r.stderr.decode(errors='replace')[-2000:]}")
        print(f"window {wi}: arc {int(wlo)}-{int(whi)}m, {len(frames)} frames -> {webm.name}")


if __name__ == "__main__":
    main()
