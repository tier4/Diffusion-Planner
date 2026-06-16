"""Separate stopped / moving obstacle-clearance heatmaps along a route.

Consumes the ``clearance_log.json`` emitted by ``scenario_generation.replay``
(per-step ego world pose + realized nearest distance to road border / stopped
NPC / moving NPC) and the route pickle, and produces:

  * ``heatmap_stopped.png`` — route coloured by the worst (min) ego↔stopped-NPC
    clearance seen in each arc-length bin (red = close, green = safe), plus a
    min-clearance-vs-arc line chart.
  * ``heatmap_moving.png``  — same for ego↔moving-NPC clearance.
  * ``clearance_arc.npz``   — binned arrays for downstream reuse.
  * ``bad_areas_stopped/`` and ``bad_areas_moving/`` — one representative step
    PNG (the worst step in the bin) copied per arc bin whose min clearance
    falls below the per-type threshold. → one image per bad avoidance area.
  * ``summary.json``        — counts + global minima per obstacle type.

The clearances are the realized geometric distances measured live during the
sim (the same numbers drawn on each step PNG), NOT the model's predicted
future — so the heatmap shows where the ego actually drove dangerously close.

Usage:
    python -m scenario_generation.tools.heatmap_npc_clearance \
        --clearance_log <run_dir>/clearance_log.json \
        --route <route.pkl> \
        --output_dir <run_dir>/clearance_heatmaps \
        --label m2t [--bin_m 5.0] \
        [--bad_thresh_stopped 1.0] [--bad_thresh_moving 2.0]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scenario_generation.tools._heatmap_common import (
    bin_scalar_by_arc,
    build_route_polyline,
    load_route,
    plot_route_heatmap,
    project_to_polyline,
    segments_from_polyline,
)

# Colour cap (m) for each obstacle type's heatmap. Clearances at/above this
# are fully "safe" green; 0 is fully red. Stopped obstacles are evaluated at a
# tighter scale than moving ones (a parked car at 1 m is alarming; a passing
# car at 1 m is alarming too, but the moving channel sweeps a wider range).
_VMAX = {"stopped": 3.0, "moving": 5.0}


def _load_records(clearance_log: Path) -> tuple[list[dict], str]:
    with open(clearance_log) as f:
        payload = json.load(f)
    return payload["records"], payload.get("png_dir", str(clearance_log.parent))


def _collect(records: list[dict], pts: np.ndarray, s: np.ndarray, key: str):
    """Project records with a non-null ``key`` distance onto the route.

    Returns (arc (M,), dist (M,), recs (list)) parallel arrays for samples that
    have a finite distance for this obstacle type.
    """
    arc, dist, recs = [], [], []
    for r in records:
        d = r.get(key)
        if d is None or not math.isfinite(d):
            continue
        s_arc, _, _ = project_to_polyline(
            np.array([r["ego_x"], r["ego_y"]], dtype=np.float64),
            pts,
            s,
        )
        arc.append(s_arc)
        dist.append(float(d))
        recs.append(r)
    return np.asarray(arc), np.asarray(dist), recs


def _render_heatmap(
    out_png: Path,
    pts: np.ndarray,
    s_max: float,
    bin_m: float,
    arc: np.ndarray,
    dist: np.ndarray,
    vmax: float,
    threshold: float,
    title: str,
):
    n_bins = max(1, int(math.ceil(s_max / bin_m)))
    bin_s_mid, mean_val, min_val = bin_scalar_by_arc(arc, dist, s_max, bin_m)
    segs = segments_from_polyline(pts, _arc_lengths(pts), bin_m, n_bins)

    fig, (ax_map, ax_line) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    cmap = plt.get_cmap("RdYlGn")
    plot_route_heatmap(
        ax_map,
        pts,
        segs,
        min_val,
        title,
        vmin=0.0,
        vmax=vmax,
        cmap=cmap,
    )
    ax_map.autoscale_view()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, fraction=0.035, pad=0.02)
    cb.set_label("min clearance (m)  — red=close, green=safe")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")

    # Min-clearance vs arc line chart.
    valid = ~np.isnan(min_val)
    ax_line.plot(
        bin_s_mid[valid], min_val[valid], "-", color="#cc0000", lw=1.5, label="min clearance / bin"
    )
    ax_line.plot(
        bin_s_mid[valid],
        mean_val[valid],
        "--",
        color="#888888",
        lw=1.0,
        label="mean clearance / bin",
    )
    ax_line.axhline(
        threshold, color="black", ls=":", lw=1.0, label=f"bad threshold {threshold:.1f} m"
    )
    ax_line.set_xlabel("route arc-length (m)")
    ax_line.set_ylabel("clearance (m)")
    # Keep the threshold line visible even when it sits above the colour cap.
    ax_line.set_ylim(0, max(vmax, threshold) * 1.1)
    ax_line.legend(fontsize=8, loc="upper right")
    ax_line.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return bin_s_mid, mean_val, min_val


def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg * seg).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def _export_bad_areas(
    out_dir: Path,
    png_dir: Path,
    bin_m: float,
    threshold: float,
    arc: np.ndarray,
    dist: np.ndarray,
    recs: list[dict],
    n_bins: int,
    max_areas: int,
) -> list[dict]:
    """Copy one representative (worst) step PNG per arc bin below threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_idx = np.clip((arc // bin_m).astype(int), 0, n_bins - 1)
    bad: list[dict] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        d_bin = dist[mask]
        if d_bin.min() >= threshold:
            continue
        local = int(np.argmin(d_bin))
        rec = [r for r, m in zip(recs, mask) if m][local]
        bad.append(
            {
                "arc_m": float((b + 0.5) * bin_m),
                "min_clearance_m": float(d_bin.min()),
                "step": int(rec["step"]),
                "png": rec["png"],
            }
        )
    # Worst first; cap the number of copied images.
    bad.sort(key=lambda x: x["min_clearance_m"])
    bad = bad[:max_areas]
    for entry in bad:
        src = png_dir / entry["png"]
        dst = out_dir / (
            f"arc{int(entry['arc_m']):04d}m_d{entry['min_clearance_m']:.2f}"
            f"_step{entry['step']:04d}.png"
        )
        if src.exists():
            shutil.copyfile(src, dst)
            entry["copied_to"] = dst.name
        else:
            entry["copied_to"] = None
            print(f"  [WARN] source PNG missing: {src}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clearance_log", type=Path, required=True)
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--png_dir",
        type=Path,
        default=None,
        help="Dir with the step_NNNN.png files. Defaults to the "
        "png_dir recorded in the clearance log.",
    )
    p.add_argument("--label", default="run")
    p.add_argument("--bin_m", type=float, default=5.0)
    p.add_argument(
        "--bad_thresh_stopped",
        type=float,
        default=1.0,
        help="Arc bins with min stopped-clearance below this are flagged as bad avoidance areas.",
    )
    p.add_argument("--bad_thresh_moving", type=float, default=2.0)
    p.add_argument(
        "--max_bad_areas",
        type=int,
        default=40,
        help="Cap on copied representative PNGs per obstacle type.",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, log_png_dir = _load_records(args.clearance_log)
    png_dir = args.png_dir or Path(log_png_dir)
    print(f"Loaded {len(records)} clearance records; png_dir={png_dir}")

    route = load_route(args.route)
    pts, s = build_route_polyline(route)
    s_max = float(s[-1])
    n_bins = max(1, int(math.ceil(s_max / args.bin_m)))
    print(f"Route arc-length {s_max:.1f} m, {n_bins} bins of {args.bin_m} m")

    summary: dict = {
        "label": args.label,
        "route_arc_m": s_max,
        "bin_m": args.bin_m,
        "n_records": len(records),
        "types": {},
    }
    npz_out: dict = {"route_pts": pts, "bin_s_mid": (np.arange(n_bins) + 0.5) * args.bin_m}

    for key, kind, thresh in (
        ("stopped_dist", "stopped", args.bad_thresh_stopped),
        ("moving_dist", "moving", args.bad_thresh_moving),
    ):
        arc, dist, recs = _collect(records, pts, s, key)
        if len(arc) == 0:
            print(f"[{kind}] no samples within range — skipping heatmap.")
            summary["types"][kind] = {"n_samples": 0, "bad_areas": []}
            continue
        vmax = _VMAX[kind]
        bin_s_mid, mean_val, min_val = _render_heatmap(
            args.output_dir / f"heatmap_{kind}.png",
            pts,
            s_max,
            args.bin_m,
            arc,
            dist,
            vmax,
            thresh,
            f"{args.label}: ego↔{kind}-NPC min clearance (m)",
        )
        bad = _export_bad_areas(
            args.output_dir / f"bad_areas_{kind}",
            png_dir,
            args.bin_m,
            thresh,
            arc,
            dist,
            recs,
            n_bins,
            args.max_bad_areas,
        )
        npz_out[f"{kind}_min"] = min_val
        npz_out[f"{kind}_mean"] = mean_val
        summary["types"][kind] = {
            "n_samples": int(len(arc)),
            "global_min_clearance_m": float(dist.min()),
            "bad_threshold_m": thresh,
            "n_bad_areas": len(bad),
            "bad_areas": bad,
        }
        print(
            f"[{kind}] {len(arc)} samples, global min {dist.min():.2f} m, "
            f"{len(bad)} bad areas (< {thresh} m) → "
            f"{args.output_dir / f'bad_areas_{kind}'}"
        )

    np.savez(args.output_dir / "clearance_arc.npz", **npz_out)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote heatmaps + summary to {args.output_dir}")


if __name__ == "__main__":
    main()
