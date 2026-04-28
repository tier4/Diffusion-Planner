#!/usr/bin/env python3
"""N-way centerline-tracking heatmap comparison across closed-loop psim runs.

Reuses :func:`rlvr.autoresearch.tools.eval_psim_centerline.collect_run`
(npz directory) and ``collect_run_trajectory_log`` (replay JSON) so the
lateral metric is identical to ``compute_centerline_score_batch`` /
``lat_offset_and_naive_score`` (the GRPO training reward function).

Each run is described by a ``--run`` argument:

  --run name=<label>,kind=<npz|trajlog>,path=<dir or .json>

At least two runs must be given. Outputs (in ``--out_dir``):
  summary.txt
  heatmap_nway_xy.png        N-panel scatter colored by |lat|
  heatmap_nway_diff.png      (run_i − run_0) projected onto the route polyline
  histogram_offsets.png      stacked density histograms
  progress_vs_offset.png     binned mean |lat| vs route arc-length
  per-run npz dumps + route_polyline.npy

Usage:
    python -m rlvr.autoresearch.tools.eval_psim_centerline_nway \\
        --run name=psim_new,kind=npz,path=<npz_dir> \\
        --run name=perfect,kind=trajlog,path=<replay>/trajectory_log.json \\
        --run name=mpc,kind=trajlog,path=<replay>/trajectory_log.json \\
        --osm <map>.osm --route_json <route>.json --out_dir <out>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlvr.autoresearch.tools._psim_centerline_common import (
    build_route_polyline,
    crop_run_by_offset,
    polyline_cumulative_arclength,
    stats_line,
)
from rlvr.autoresearch.tools.eval_psim_centerline import (
    collect_run,
    collect_run_trajectory_log,
)


def _parse_run(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    if "name" not in out or "kind" not in out or "path" not in out:
        raise SystemExit(f"--run must include name=, kind=, path= (got {spec!r})")
    if out["kind"] not in ("npz", "trajlog"):
        raise SystemExit(f"--run kind must be 'npz' or 'trajlog' (got {out['kind']})")
    return out


def _load(run: dict, polyline: np.ndarray, ego_half_w: float, device: str) -> dict:
    p = Path(run["path"])
    if run["kind"] == "npz":
        return collect_run(p, polyline, ego_half_w, device)
    return collect_run_trajectory_log(p, polyline, ego_half_w, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="Repeat. Format: name=<label>,kind=<npz|trajlog>,path=<...>")
    ap.add_argument("--osm", type=Path, required=True)
    ap.add_argument("--route_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_offset_m", type=float, default=10.0)
    ap.add_argument("--ego_half_w", type=float, default=0.85,
                    help="Half ego width (m). Forwarded to "
                         "lat_offset_and_naive_score; currently consumed only "
                         "in 'body' usage_mode and the underlying tool "
                         "hardcodes 'baselink', so this is effectively "
                         "ignored. Kept for forward-compat.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [_parse_run(s) for s in args.run]
    if len(runs) < 2:
        raise SystemExit("Need at least 2 runs.")

    polyline = build_route_polyline(args.osm, args.route_json)
    np.save(args.out_dir / "route_polyline.npy", polyline)

    print(f"Loaded route polyline ({len(polyline)} points). Loading {len(runs)} runs...")
    raw = {}
    cropped = {}
    for r in runs:
        print(f"-> {r['name']} ({r['kind']}: {r['path']})")
        d = _load(r, polyline, args.ego_half_w, args.device)
        raw[r["name"]] = d
        cropped[r["name"]] = crop_run_by_offset(d, args.max_offset_m)
        np.savez(args.out_dir / f"{r['name']}_offsets.npz", **d)

    summary_lines = [
        f"N-way centerline-tracking comparison (lateral via "
        f"compute_centerline_score_batch / lat_offset_and_naive_score)",
        f"Reference: {args.route_json.name} ({len(polyline)} centerline points)",
        f"Map: {args.osm}",
        f"Frames cropped where |lateral offset| > {args.max_offset_m} m",
        "Sign: + = left of route direction, − = right",
        "",
    ]
    # Abort early if any run has no kept frames — the rest of the pipeline
    # (stats, scatter, arc-length binning) all reduce over potentially-empty
    # arrays and would crash.
    empty_runs = [name for name, d in cropped.items() if len(d["lat"]) == 0]
    if empty_runs:
        raise SystemExit(
            f"No frames remain after cropping by --max_offset_m="
            f"{args.max_offset_m} for run(s): {', '.join(empty_runs)}. "
            "Increase --max_offset_m or inspect the input data."
        )

    for r in runs:
        summary_lines.append(stats_line(r["name"], cropped[r["name"]], raw[r["name"]]))
    summary_lines.append("")
    base_name = runs[0]["name"]
    base_abs = np.abs(cropped[base_name]["lat"])
    base_mean = float(np.mean(base_abs))
    summary_lines.append(f"Δ baseline = {base_name}")
    for r in runs[1:]:
        a = np.abs(cropped[r["name"]]["lat"])
        d_mean = float(np.mean(a)) - base_mean
        d_p95 = float(np.percentile(a, 95)) - float(np.percentile(base_abs, 95))
        d_med = float(np.median(a)) - float(np.median(base_abs))
        # Guard against a perfectly-on-centerline baseline (mean ≈ 0).
        if abs(base_mean) <= 1e-12:
            pct = "pct n/a"
        else:
            pct = f"{100 * d_mean / base_mean:+.1f}%"
        summary_lines.append(
            f"  {r['name']} vs {base_name}: "
            f"Δmean={d_mean:+.3f}m ({pct})  "
            f"Δmed={d_med:+.3f}m  Δp95={d_p95:+.3f}m"
        )
    summary_lines.append("")
    for thr in (0.3, 0.5, 1.0):
        line = f"Frames |lat| > {thr}m:"
        for r in runs:
            a = np.abs(cropped[r["name"]]["lat"])
            line += f"  {r['name']}={(a>thr).sum()}/{len(a)} ({100*(a>thr).mean():.1f}%)"
        summary_lines.append(line)
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary + "\n")

    # ---- N-panel scatter ----
    all_xy = np.concatenate([cropped[r["name"]]["world_xy"] for r in runs], axis=0)
    x_min, x_max = all_xy[:, 0].min() - 20, all_xy[:, 0].max() + 20
    y_min, y_max = all_xy[:, 1].min() - 20, all_xy[:, 1].max() + 20
    vmax = float(np.percentile(
        np.concatenate([np.abs(cropped[r["name"]]["lat"]) for r in runs]), 95
    ))
    vmax = max(vmax, 0.5)

    N = len(runs)
    fig, axes = plt.subplots(1, N, figsize=(7 * N, 9), sharex=True, sharey=True)
    if N == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        d = cropped[r["name"]]
        ax.plot(polyline[:, 0], polyline[:, 1], "-", color="0.7", lw=2.5,
                alpha=0.5, zorder=1, label="route centerline")
        sc = ax.scatter(
            d["world_xy"][:, 0], d["world_xy"][:, 1],
            c=np.abs(d["lat"]), cmap="inferno", vmin=0.0, vmax=vmax,
            s=18, alpha=0.9, edgecolors="none", zorder=2,
        )
        ax.set_title(
            f"{r['name']}\nmean|lat|={np.mean(np.abs(d['lat'])):.3f}m  "
            f"p95={np.percentile(np.abs(d['lat']),95):.3f}m  "
            f"max={np.max(np.abs(d['lat'])):.3f}m  n={len(d['lat'])}"
        )
        ax.set_xlabel("MGRS x [m]")
        ax.set_ylabel("MGRS y [m]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="|lat| [m]")
    plt.suptitle("Centerline-tracking heatmap (each dot = one frame)", y=0.99)
    plt.tight_layout()
    plt.savefig(args.out_dir / "heatmap_nway_xy.png", dpi=140)
    plt.close()

    # ---- Δ map: each non-base run binned along arc-length and re-projected ----
    arc_max = max(cropped[r["name"]]["lon"].max() for r in runs)
    arc_bins = np.arange(0, arc_max + 5.0, 5.0)
    arc_centers = 0.5 * (arc_bins[1:] + arc_bins[:-1])

    def bin_arc(d: dict) -> np.ndarray:
        idx = np.clip(np.digitize(d["lon"], arc_bins) - 1, 0, len(arc_centers) - 1)
        sums = np.zeros(len(arc_centers))
        cnts = np.zeros(len(arc_centers))
        np.add.at(sums, idx, np.abs(d["lat"]))
        np.add.at(cnts, idx, 1.0)
        return np.divide(sums, cnts, out=np.full_like(sums, np.nan), where=cnts > 0)

    arcs = {r["name"]: bin_arc(cropped[r["name"]]) for r in runs}
    poly_arc, _ = polyline_cumulative_arclength(polyline)
    poly_bin = np.clip(np.digitize(poly_arc, arc_bins) - 1, 0, len(arc_centers) - 1)

    diff_panels = [(r["name"], arcs[r["name"]] - arcs[base_name]) for r in runs[1:]]
    if diff_panels:
        fig, axes = plt.subplots(1, len(diff_panels), figsize=(8 * len(diff_panels), 9),
                                 sharex=True, sharey=True)
        if len(diff_panels) == 1:
            axes = [axes]
        all_finite = np.concatenate([d[np.isfinite(d)] for _, d in diff_panels])
        vlim = max(float(np.percentile(np.abs(all_finite), 98)) if all_finite.size else 1.0, 0.3)
        for ax, (name, diff_arc) in zip(axes, diff_panels):
            poly_diff = diff_arc[poly_bin]
            valid = np.isfinite(poly_diff)
            ax.plot(polyline[:, 0], polyline[:, 1], "-", color="0.85", lw=2.0, zorder=1)
            sc = ax.scatter(
                polyline[valid, 0], polyline[valid, 1],
                c=poly_diff[valid], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                s=80, edgecolors="none", zorder=2,
            )
            ax.set_title(
                f"Δ |lat| ({name} − {base_name})\n"
                f"blue = {name} better   red = {name} worse\n"
                f"mean Δ = {np.nanmean(diff_arc):+.3f} m"
            )
            ax.set_xlabel("MGRS x [m]")
            ax.set_ylabel("MGRS y [m]")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Δ |lat| [m]")
        plt.tight_layout()
        plt.savefig(args.out_dir / "heatmap_nway_diff.png", dpi=140)
        plt.close()

    # ---- Stacked histograms ----
    colors = ["C0", "C3", "C2", "C4", "C1", "C5"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-2.0, 2.0, 161)
    for i, r in enumerate(runs):
        d = cropped[r["name"]]
        ax.hist(d["lat"], bins=bins, alpha=0.5, density=True,
                color=colors[i % len(colors)],
                label=f"{r['name']} (n={len(d['lat'])}, mean|lat|={np.mean(np.abs(d['lat'])):.3f}m)")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("signed lateral offset [m]   (+ left, − right)")
    ax.set_ylabel("density")
    ax.set_title("Lateral offset distribution to route centerline")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "histogram_offsets.png", dpi=140)
    plt.close()

    # ---- Progress vs |lat| ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, r in enumerate(runs):
        d = cropped[r["name"]]
        c = colors[i % len(colors)]
        ax.plot(arc_centers, arcs[r["name"]], color=c, lw=2.0,
                label=f"{r['name']} (binned mean)")
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
