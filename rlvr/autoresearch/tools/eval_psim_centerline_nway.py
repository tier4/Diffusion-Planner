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
from scenario_generation.canvas import Figure
from scenario_generation.colormaps import qualitative


def _parse_run(spec: str) -> dict:
    """Parse a single ``--run name=<>,kind=<>,path=<>`` spec.

    Validates: all three keys present, all three values non-empty, kind in
    {npz, trajlog}. Caller (``main``) further enforces that ``name`` is
    unique across runs — duplicate names would silently overwrite the
    same ``<name>_offsets.npz`` output and confuse downstream plots."""
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    for key in ("name", "kind", "path"):
        if key not in out or not out[key]:
            raise SystemExit(f"--run must include non-empty name=, kind=, path= (got {spec!r})")
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
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeat. Format: name=<label>,kind=<npz|trajlog>,path=<...>",
    )
    ap.add_argument("--osm", type=Path, required=True)
    ap.add_argument("--route_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_offset_m", type=float, default=10.0)
    ap.add_argument(
        "--ego_half_w",
        type=float,
        default=0.85,
        help="Half ego width (m). Forwarded to "
        "lat_offset_and_naive_score; currently consumed only "
        "in 'body' usage_mode and the underlying tool "
        "hardcodes 'baselink', so this is effectively "
        "ignored. Kept for forward-compat.",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [_parse_run(s) for s in args.run]
    if len(runs) < 2:
        raise SystemExit("Need at least 2 runs.")
    seen: set[str] = set()
    for r in runs:
        if r["name"] in seen:
            raise SystemExit(
                f"Duplicate run name {r['name']!r}. Run names must be unique — "
                "they appear in plot legends and the output "
                "<name>_offsets.npz filename, so duplicates would silently "
                "overwrite each other."
            )
        seen.add(r["name"])

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
            line += f"  {r['name']}={(a > thr).sum()}/{len(a)} ({100 * (a > thr).mean():.1f}%)"
        summary_lines.append(line)
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary + "\n")

    # ---- N-panel scatter ----
    all_xy = np.concatenate([cropped[r["name"]]["world_xy"] for r in runs], axis=0)
    x_min, x_max = all_xy[:, 0].min() - 20, all_xy[:, 0].max() + 20
    y_min, y_max = all_xy[:, 1].min() - 20, all_xy[:, 1].max() + 20
    vmax = float(
        np.percentile(np.concatenate([np.abs(cropped[r["name"]]["lat"]) for r in runs]), 95)
    )
    vmax = max(vmax, 0.5)

    N = len(runs)
    fig = Figure((7 * N, 9), dpi=140)
    (axes,) = fig.grid(1, N, equal=True)
    for ax, r in zip(axes, runs):
        d = cropped[r["name"]]
        ax.lines(
            [polyline[:, :2]],
            "#b3b3b3",
            width=2.5,
            alpha=0.5,
            z=1,
            label="route centerline",
        )
        ax.markers(
            d["world_xy"][:, :2],
            "o",
            size=np.sqrt(18),
            values=np.abs(d["lat"]),
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            alpha=0.9,
            z=2,
        )
        ax.title(
            f"{r['name']}\nmean|lat|={np.mean(np.abs(d['lat'])):.3f}m  "
            f"p95={np.percentile(np.abs(d['lat']), 95):.3f}m  "
            f"max={np.max(np.abs(d['lat'])):.3f}m  n={len(d['lat'])}"
        )
        ax.xlabel("MGRS x [m]")
        ax.ylabel("MGRS y [m]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend("upper right")
        ax.colorbar("inferno", 0.0, vmax, "|lat| [m]")
    fig.suptitle("Centerline-tracking heatmap (each dot = one frame)")
    fig.save(args.out_dir / "heatmap_nway_xy.png")

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
        fig = Figure((8 * len(diff_panels), 9), dpi=140)
        (axes,) = fig.grid(1, len(diff_panels), equal=True)
        all_finite = np.concatenate([d[np.isfinite(d)] for _, d in diff_panels])
        vlim = max(float(np.percentile(np.abs(all_finite), 98)) if all_finite.size else 1.0, 0.3)
        for ax, (name, diff_arc) in zip(axes, diff_panels):
            poly_diff = diff_arc[poly_bin]
            valid = np.isfinite(poly_diff)
            ax.lines([polyline[:, :2]], "#d9d9d9", width=2.0, z=1)
            ax.markers(
                polyline[valid, :2],
                "o",
                size=np.sqrt(80),
                values=poly_diff[valid],
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=vlim,
                z=2,
            )
            ax.title(
                f"Δ |lat| ({name} − {base_name})\n"
                f"blue = {name} better   red = {name} worse\n"
                f"mean Δ = {np.nanmean(diff_arc):+.3f} m"
            )
            ax.xlabel("MGRS x [m]")
            ax.ylabel("MGRS y [m]")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.colorbar("RdBu_r", -vlim, vlim, "Δ |lat| [m]")
        fig.save(args.out_dir / "heatmap_nway_diff.png")

    # ---- Stacked histograms ----
    colors = [qualitative("tab10", i) for i in (0, 3, 2, 4, 1, 5)]
    fig = Figure((10, 5), dpi=140)
    ax = fig.panel()
    bins = np.linspace(-2.0, 2.0, 161)
    for i, r in enumerate(runs):
        d = cropped[r["name"]]
        ax.hist(
            d["lat"],
            bins=bins,
            alpha=0.5,
            density=True,
            color=colors[i % len(colors)],
            label=f"{r['name']} (n={len(d['lat'])}, mean|lat|={np.mean(np.abs(d['lat'])):.3f}m)",
        )
    ax.vline(0, color="#000000", width=0.7)
    ax.xlabel("signed lateral offset [m]   (+ left, − right)")
    ax.ylabel("density")
    ax.title("Lateral offset distribution to route centerline")
    ax.legend("upper right")
    fig.save(args.out_dir / "histogram_offsets.png")

    # ---- Progress vs |lat| ----
    fig = Figure((12, 5), dpi=140)
    ax = fig.panel()
    for i, r in enumerate(runs):
        c = colors[i % len(colors)]
        ax.lines(
            [np.stack([arc_centers, arcs[r["name"]]], 1)],
            c,
            width=2.0,
            label=f"{r['name']} (binned mean)",
        )
    ax.xlabel("longitudinal arc-length along route centerline [m]")
    ax.ylabel("|lateral offset| [m]")
    ax.title("Lateral tracking error vs route progress")
    ax.legend("upper right")
    ax.grid(0.3)
    fig.save(args.out_dir / "progress_vs_offset.png")

    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
