#!/usr/bin/env python3
"""Centerline-tracking heatmap comparison between two psim closed-loop runs
(rosbag→npz). Reuses the official reward-function lateral metric so the
numbers match what GRPO training sees.

Why a separate tool from ``eval_centerline_metrics.py``:
  * That tool *generates* trajectories from a model and scores them on each
    npz's own ``route_lanes`` field.
  * This tool consumes *already-rolled-out* psim trajectories (logged ego
    poses across an entire run) and projects them onto a reference route
    centerline built from the rosbag's published ``LaneletRoute`` + the
    lanelet2 map. The npz ``route_lanes`` field is unreliable in psim teleport
    bags (40-75% of frames have empty route lanes — the route message is
    published once before teleport and the converter cannot re-anchor it).

Lateral metric:
  * Calls :func:`rlvr.autoresearch.tools.eval_centerline_metrics
    .lat_offset_and_naive_score` per frame with the ego at the origin in
    ego frame and the GT polyline projected into ego frame as a synthetic
    ``route_lanes`` tensor.
  * Default ``usage_mode='baselink'`` matches the current reward default
    (since 2026-04-27); ``ego_lat`` is the signed metric distance from
    base_link to the nearest centerline point.

Outputs (in ``--out_dir``):
  <baseline_label>_offsets.npz, <comparison_label>_offsets.npz,
  route_polyline.npy, heatmap_combined.png, heatmap_centerline_xy.png,
  heatmap_centerline_diff.png, heatmap_zoom*.png, histogram_offsets.png,
  timeseries.png, progress_vs_offset.png, summary.txt

Usage:
    source .venv/bin/activate
    python -m rlvr.autoresearch.tools.eval_psim_centerline \\
        --baseline_dir <path>/npz/baseline \\
        --prism_dir <path>/npz/<comparison_run> \\
        --osm /path/to/lanelet2_map.osm \\
        --route_json /path/to/route.json \\
        --out_dir <path>/analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlvr.autoresearch.tools._psim_centerline_common import (
    arc_bin_diff,
    build_route_polyline,
    crop_run_by_offset,
    polyline_cumulative_arclength,
    project_point_to_polyline_arclength,
    stats_line,
    world_polyline_to_ego_route_lanes,
)
from rlvr.autoresearch.tools.eval_centerline_metrics import (
    lat_offset_and_naive_score,
)
from scenario_generation.transforms import yaw_from_quat


# ---------------------------------------------------------------------------
# Per-run aggregation.
# ---------------------------------------------------------------------------

def collect_run_trajectory_log(
    traj_log_path: Path,
    polyline: np.ndarray,
    ego_half_w: float,
    device: str = "cpu",
) -> dict:
    """Aggregate per-frame lateral metrics from a closed-loop replay's
    ``trajectory_log.json`` (produced by ``scenario_generation.replay``).

    The log is a list of dicts with ``step / x / y / heading / speed``. We
    treat ``step`` as the timestamp (1 step ≈ 0.1 s in the replay) and use
    the world-frame ``(x, y, heading)`` to compute lateral offset to the
    reference polyline via the official reward helper.
    """
    log = json.loads(Path(traj_log_path).read_text())
    print(f"  {traj_log_path.parent.name}: {len(log)} replay steps")
    n = len(log)
    ts = np.zeros(n, dtype=np.int64)
    world_xy = np.zeros((n, 2), dtype=np.float64)
    lat = np.full(n, np.nan, dtype=np.float64)
    lat_signed = np.full(n, np.nan, dtype=np.float64)
    lon = np.full(n, np.nan, dtype=np.float64)
    speed = np.full(n, np.nan, dtype=np.float64)

    traj = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)

    for i, entry in enumerate(log):
        x = float(entry["x"])
        y = float(entry["y"])
        yaw = float(entry["heading"])
        # Convert replay step index to nanoseconds (replay is at 10 Hz / 0.1 s
        # per step) so the same time-axis code that handles rosbag-derived
        # nanosecond timestamps works for trajectory_log inputs.
        ts[i] = int(entry["step"]) * 100_000_000
        world_xy[i] = (x, y)
        speed[i] = float(entry.get("speed", 0.0))

        route_lanes = world_polyline_to_ego_route_lanes(
            polyline, (x, y), yaw
        ).to(device)
        out = lat_offset_and_naive_score(
            traj=traj, data={"route_lanes": route_lanes},
            ego_half_w=ego_half_w, usage_mode="baselink",
        )
        if out is None:
            continue
        lat[i] = float(out["lat_offset_m"][0])
        lat_signed[i] = float(out["signed_lat_offset_m"][0])
        lon[i] = project_point_to_polyline_arclength(polyline, x, y)

    order = np.argsort(ts)
    return {
        "ts": ts[order], "world_xy": world_xy[order],
        "lat": lat_signed[order], "abs_lat": lat[order],
        "lon": lon[order], "speed": speed[order],
    }


def collect_run(
    npz_dir: Path,
    polyline: np.ndarray,
    ego_half_w: float,
    device: str = "cpu",
) -> dict:
    files = sorted(glob.glob(str(npz_dir / "*.npz")))
    print(f"  {npz_dir.name}: {len(files)} npz files")
    n = len(files)
    ts = np.zeros(n, dtype=np.int64)
    world_xy = np.zeros((n, 2), dtype=np.float64)
    lat = np.full(n, np.nan, dtype=np.float64)
    lat_signed = np.full(n, np.nan, dtype=np.float64)
    lon = np.full(n, np.nan, dtype=np.float64)
    speed = np.full(n, np.nan, dtype=np.float64)

    traj = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)  # (T=1, 4)

    for i, fp in enumerate(files):
        js_path = fp.replace(".npz", ".json")
        if not os.path.exists(js_path):
            continue
        with open(js_path) as f:
            j = json.load(f)
        x = float(j["x"])
        y = float(j["y"])
        yaw = yaw_from_quat(j["qx"], j["qy"], j["qz"], j["qw"])
        ts[i] = j.get("timestamp", 0)
        world_xy[i] = (x, y)
        with np.load(fp) as d:
            ego = d["ego_current_state"]
        speed[i] = float(np.linalg.norm(ego[4:6]))

        route_lanes = world_polyline_to_ego_route_lanes(
            polyline, (x, y), yaw
        ).to(device)
        out = lat_offset_and_naive_score(
            traj=traj,
            data={"route_lanes": route_lanes},
            ego_half_w=ego_half_w,
            usage_mode="baselink",
        )
        if out is None:
            continue
        lat[i] = float(out["lat_offset_m"][0])
        lat_signed[i] = float(out["signed_lat_offset_m"][0])
        lon[i] = project_point_to_polyline_arclength(polyline, x, y)

    order = np.argsort(ts)
    return {
        "ts": ts[order],
        "world_xy": world_xy[order],
        "lat": lat_signed[order],
        "abs_lat": lat[order],
        "lon": lon[order],
        "speed": speed[order],
    }


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=Path, default=None,
                    help="Directory of npz files (rosbag→npz pipeline). "
                         "Mutually exclusive with --baseline_traj_log.")
    ap.add_argument("--prism_dir", type=Path, default=None,
                    help="Directory of npz files. Mutually exclusive with "
                         "--prism_traj_log.")
    ap.add_argument("--baseline_traj_log", type=Path, default=None,
                    help="trajectory_log.json from a closed-loop replay run. "
                         "Use this for scenario_generation.replay output.")
    ap.add_argument("--prism_traj_log", type=Path, default=None,
                    help="trajectory_log.json from a closed-loop replay run.")
    ap.add_argument("--baseline_label", default="baseline",
                    help="Label for the baseline run, used in plot titles, "
                         "histogram legends, summary lines, and the output "
                         "<label>_offsets.npz filename.")
    ap.add_argument("--prism_label", default="comparison",
                    help="Label for the comparison run. The CLI flag name "
                         "is `--prism_label` for backward-compat with earlier "
                         "PRiSM-vs-baseline workflows; the run does not have "
                         "to be a PRiSM-trained checkpoint.")
    ap.add_argument("--osm", type=Path, required=True,
                    help="lanelet2_map.osm matching the rosbag map.")
    ap.add_argument("--route_json", type=Path, required=True,
                    help="LaneletRoute exported as JSON (see "
                         "extract_route.py in the route-rosbag.)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_offset_m", type=float, default=10.0,
                    help="Drop frames whose |lateral offset| exceeds this "
                         "(off-route detours, teleport boundaries).")
    ap.add_argument("--ego_half_w", type=float, default=0.85,
                    help="Half ego width (m). Only consumed when usage_mode is "
                         "'body'; the tool currently hardcodes 'baselink' so "
                         "this is ignored. Kept for forward-compat in case a "
                         "--usage_mode flag is added later.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    polyline = build_route_polyline(args.osm, args.route_json)
    np.save(args.out_dir / "route_polyline.npy", polyline)

    if (args.baseline_dir is None) == (args.baseline_traj_log is None):
        raise SystemExit("Provide exactly one of --baseline_dir / --baseline_traj_log")
    if (args.prism_dir is None) == (args.prism_traj_log is None):
        raise SystemExit("Provide exactly one of --prism_dir / --prism_traj_log")

    print(f"Loading {args.baseline_label}...")
    if args.baseline_traj_log is not None:
        base = collect_run_trajectory_log(
            args.baseline_traj_log, polyline, args.ego_half_w, args.device,
        )
    else:
        base = collect_run(args.baseline_dir, polyline, args.ego_half_w, args.device)
    print(f"Loading {args.prism_label}...")
    if args.prism_traj_log is not None:
        prism = collect_run_trajectory_log(
            args.prism_traj_log, polyline, args.ego_half_w, args.device,
        )
    else:
        prism = collect_run(args.prism_dir, polyline, args.ego_half_w, args.device)

    np.savez(args.out_dir / f"{args.baseline_label}_offsets.npz", **base)
    np.savez(args.out_dir / f"{args.prism_label}_offsets.npz", **prism)

    base_c = crop_run_by_offset(base, args.max_offset_m)
    prism_c = crop_run_by_offset(prism, args.max_offset_m)
    if len(base_c["lat"]) == 0 or len(prism_c["lat"]) == 0:
        empty = []
        if len(base_c["lat"]) == 0:
            empty.append(args.baseline_label)
        if len(prism_c["lat"]) == 0:
            empty.append(args.prism_label)
        raise SystemExit(
            f"No frames remain after cropping by --max_offset_m="
            f"{args.max_offset_m} for run(s): {', '.join(empty)}. "
            "This can happen if the threshold is too small or if the metric "
            "failed on every frame. Increase --max_offset_m or inspect the "
            "input data."
        )

    summary_lines = [
        "psim centerline-tracking comparison (lateral via "
        "rlvr.reward.compute_centerline_score_batch / lat_offset_and_naive_score)",
        f"Reference: {args.route_json.name} ({len(polyline)} centerline points)",
        f"Map: {args.osm}",
        f"Frames cropped where |lateral offset| > {args.max_offset_m} m.",
        "Sign: + = left of route direction, − = right.",
        "",
        stats_line(args.baseline_label, base_c, base),
        stats_line(args.prism_label, prism_c, prism),
        "",
    ]

    b_abs = np.abs(base_c["lat"])
    p_abs = np.abs(prism_c["lat"])
    delta_label = f"{args.prism_label} − {args.baseline_label}"

    def _delta_line(metric: str, p_val: float, b_val: float) -> str:
        delta = p_val - b_val
        if abs(b_val) <= 1e-12:
            pct = "pct n/a; baseline ≈ 0"
        else:
            pct = f"{100 * delta / b_val:+.2f}%"
        return f"Δ |lat| {metric:<6s} ({delta_label}): {delta:+.4f} m  ({pct})"

    summary_lines.append(_delta_line("mean", float(np.mean(p_abs)), float(np.mean(b_abs))))
    summary_lines.append(_delta_line("median", float(np.median(p_abs)), float(np.median(b_abs))))
    summary_lines.append(
        _delta_line("p95", float(np.percentile(p_abs, 95)), float(np.percentile(b_abs, 95)))
    )
    for thr in (0.3, 0.5, 1.0):
        summary_lines.append(
            f"Frames with |lat| > {thr}m:  "
            f"{args.baseline_label}={(b_abs>thr).sum()}/{len(b_abs)} ({100*(b_abs>thr).mean():.2f}%) | "
            f"{args.prism_label}={(p_abs>thr).sum()}/{len(p_abs)} ({100*(p_abs>thr).mean():.2f}%)"
        )
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary + "\n")

    # ---- Scatter heatmaps and combined panel ----
    all_xy = np.concatenate([base_c["world_xy"], prism_c["world_xy"]], axis=0)
    x_min, x_max = all_xy[:, 0].min() - 20, all_xy[:, 0].max() + 20
    y_min, y_max = all_xy[:, 1].min() - 20, all_xy[:, 1].max() + 20
    vmax = float(np.percentile(np.concatenate([np.abs(base_c["lat"]), np.abs(prism_c["lat"])]), 95))
    vmax = max(vmax, 0.5)

    def scatter_panel(ax, run, ttl):
        ax.plot(polyline[:, 0], polyline[:, 1], "-", color="0.7", lw=2.5,
                alpha=0.5, zorder=1, label="route centerline")
        sc = ax.scatter(
            run["world_xy"][:, 0], run["world_xy"][:, 1],
            c=np.abs(run["lat"]), cmap="inferno", vmin=0.0, vmax=vmax,
            s=18, alpha=0.9, edgecolors="none", zorder=2,
        )
        ax.set_title(
            f"{ttl}\nmean|lat|={np.mean(np.abs(run['lat'])):.3f}m  "
            f"p95={np.percentile(np.abs(run['lat']),95):.3f}m  "
            f"max={np.max(np.abs(run['lat'])):.3f}m"
        )
        ax.set_xlabel("MGRS x [m]")
        ax.set_ylabel("MGRS y [m]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        return sc

    arc_bins, arc_centers, diff_arc = arc_bin_diff(base_c, prism_c, bin_m=5.0)
    poly_arc, _ = polyline_cumulative_arclength(polyline)
    poly_bin = np.clip(np.digitize(poly_arc, arc_bins) - 1, 0, len(arc_centers) - 1)
    poly_diff = diff_arc[poly_bin]
    valid_diff = np.isfinite(poly_diff)
    finite = diff_arc[np.isfinite(diff_arc)]
    vlim = max(float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0, 0.3)

    # Two-panel side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
    sc_b = scatter_panel(axes[0], base_c, args.baseline_label)
    sc_p = scatter_panel(axes[1], prism_c, args.prism_label)
    plt.colorbar(sc_b, ax=axes[0], fraction=0.04, pad=0.02, label="|lat| [m]")
    plt.colorbar(sc_p, ax=axes[1], fraction=0.04, pad=0.02, label="|lat| [m]")
    plt.suptitle("Centerline-tracking heatmap (lat from compute_centerline_score_batch)", y=0.98)
    plt.tight_layout()
    plt.savefig(args.out_dir / "heatmap_centerline_xy.png", dpi=140)
    plt.close()

    # Diff-only
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.plot(polyline[:, 0], polyline[:, 1], "-", color="0.85", lw=2.0, zorder=1)
    sc = ax.scatter(
        polyline[valid_diff, 0], polyline[valid_diff, 1],
        c=poly_diff[valid_diff], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        s=80, edgecolors="none", zorder=2,
    )
    ax.set_title(
        f"Δ mean |lateral offset|  ({args.prism_label} − {args.baseline_label}), "
        f"5 m arc-length bins\n"
        f"blue = {args.prism_label} better   red = {args.prism_label} worse"
    )
    ax.set_xlabel("MGRS x [m]")
    ax.set_ylabel("MGRS y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Δ |lat| [m]")
    plt.tight_layout()
    plt.savefig(args.out_dir / "heatmap_centerline_diff.png", dpi=140)
    plt.close()

    # Combined 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(22, 11), sharex=True, sharey=True)
    sc_b = scatter_panel(axes[0], base_c, args.baseline_label)
    sc_p = scatter_panel(axes[1], prism_c, args.prism_label)
    plt.colorbar(sc_b, ax=axes[0], fraction=0.04, pad=0.02, label="|lat| [m]")
    plt.colorbar(sc_p, ax=axes[1], fraction=0.04, pad=0.02, label="|lat| [m]")
    ax = axes[2]
    ax.plot(polyline[:, 0], polyline[:, 1], "-", color="0.85", lw=2.0, zorder=1)
    sc = ax.scatter(
        polyline[valid_diff, 0], polyline[valid_diff, 1],
        c=poly_diff[valid_diff], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        s=80, edgecolors="none", zorder=2,
    )
    ax.set_title(
        f"Δ |lat|  ({args.prism_label} − {args.baseline_label})\n"
        f"blue = {args.prism_label} better   red = {args.prism_label} worse\n"
        f"mean Δ|lat| = {np.nanmean(diff_arc):+.3f} m"
    )
    ax.set_xlabel("MGRS x [m]")
    ax.set_ylabel("MGRS y [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Δ |lat| [m]")
    plt.suptitle(
        f"Centerline-tracking comparison — {args.baseline_label} vs {args.prism_label} (psim)",
        y=0.99, fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "heatmap_combined.png", dpi=140)
    plt.close()

    # Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-2.0, 2.0, 161)
    ax.hist(base_c["lat"], bins=bins, alpha=0.55, color="C0", density=True,
            label=f"{args.baseline_label} (n={len(base_c['lat'])}, "
                  f"mean|lat|={np.mean(np.abs(base_c['lat'])):.3f}m)")
    ax.hist(prism_c["lat"], bins=bins, alpha=0.55, color="C3", density=True,
            label=f"{args.prism_label} (n={len(prism_c['lat'])}, "
                  f"mean|lat|={np.mean(np.abs(prism_c['lat'])):.3f}m)")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("signed lateral offset [m]   (+ left, − right)")
    ax.set_ylabel("density")
    ax.set_title("Lateral offset distribution to route centerline")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "histogram_offsets.png", dpi=140)
    plt.close()

    # Time series
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for run, color, label in [(base_c, "C0", args.baseline_label), (prism_c, "C3", args.prism_label)]:
        if len(run["ts"]) == 0:
            continue
        t = (run["ts"] - run["ts"][0]) / 1e9
        axes[0].plot(t, run["lat"], color=color, lw=0.7, label=label)
        axes[1].plot(t, np.abs(run["lat"]), color=color, lw=0.7, label=label)
    axes[0].set_ylabel("signed lat [m]")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Lateral offset to route centerline vs simulation time")
    axes[1].set_ylabel("|lat| [m]")
    axes[1].set_xlabel("time [s] from run start")
    plt.tight_layout()
    plt.savefig(args.out_dir / "timeseries.png", dpi=140)
    plt.close()

    # |lat| vs longitudinal route progress
    fig, ax = plt.subplots(figsize=(11, 5))
    for run, color, label in [(base_c, "C0", args.baseline_label), (prism_c, "C3", args.prism_label)]:
        order = np.argsort(run["lon"])
        ax.plot(run["lon"][order], np.abs(run["lat"])[order], color=color, lw=0.7,
                alpha=0.5, label=label)
    arc_max = max(base_c["lon"].max(), prism_c["lon"].max())
    bins = np.linspace(0, arc_max, 50)
    for run, color, label in [(base_c, "C0", args.baseline_label), (prism_c, "C3", args.prism_label)]:
        idx = np.digitize(run["lon"], bins)
        mu = np.array([
            np.mean(np.abs(run["lat"])[idx == i]) if (idx == i).any() else np.nan
            for i in range(1, len(bins))
        ])
        ax.plot(0.5 * (bins[1:] + bins[:-1]), mu, color=color, lw=2.0,
                label=f"{label} (binned mean)")
    ax.set_xlabel("longitudinal arc-length along route centerline [m]")
    ax.set_ylabel("|lateral offset| [m]")
    ax.set_title("Lateral tracking error vs route progress")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "progress_vs_offset.png", dpi=140)
    plt.close()

    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
