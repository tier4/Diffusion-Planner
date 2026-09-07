"""Render a top-down ego trajectory colored by a per-step risk metric.

Reads the per-step trace ``reproducer_rollout.render_segment`` already writes to
``rollout.jsonl`` next to a segment's PNGs (ego pose + ``clearance_m`` + ``collision``,
one line per step) and draws a colored polyline (risk metric -> color) with a colorbar
legend, similar in spirit to a routing app's "road risk" heatmap overlay.

Used by both the W&B representative-case image (one PNG per group, ``wandb.Image``,
worst-case route) and the local video/colormap output — same renderer, same look,
so a human comparing the two is looking at the same visualization either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

METRIC_CHOICES = (
    "clearance",
    "collision",
    "near_miss",
    "speed",
    "road_border",
    "red_light",
    "strong_brake",
)

# Fixed sim step (must match reproducer_rollout.DT) -- rollout.jsonl rows are one per sim step
# regardless of draw_every/replan_interval, so consecutive "speed" samples are always DT apart.
# Kept as a local constant (not imported from reproducer_rollout) so this module stays free of
# reproducer_rollout's heavier deps (torch, etc.).
_DT = 0.1

# Binary metrics only ever take 2 colors (event happened / didn't) — a colorbar with 2 ticks
# adds clutter without adding information the title ("metric: collision") doesn't already
# give, so it's skipped for these.
_BINARY_METRICS = ("collision", "near_miss", "red_light", "strong_brake")

# The trace key each metric is derived from. A producer that never observed a quantity omits its
# key entirely, and every accessor below defaults to the safe value -- which would paint a
# uniform "no event" picture indistinguishable from a measured zero. Checking for the key lets
# an unobserved metric be skipped instead, for any producer and for runs already on disk.
_METRIC_TRACE_KEYS = {
    "clearance": "clearance_m",
    "collision": "collision",
    "near_miss": "clearance_m",
    "speed": "speed",
    "road_border": "rb_dist_m",
    "red_light": "red_light_violation",
    "strong_brake": "speed",
}

# Short colorbar axis label. The ticks themselves (see _risk_and_ticks) carry the actual
# units/values — qualitative low/medium/high/very high labels alone aren't legible without
# knowing the metric's scale, so every tick is now a real number instead.
_METRIC_AXIS_LABELS = {
    "clearance": "clearance to nearest obstacle (m)",
    "collision": "",
    "near_miss": "",
    "speed": "ego speed (m/s)",
    "road_border": "distance to road border (m)",
    "red_light": "",
    "strong_brake": "",
}


def load_step_trace(png_dir: str | Path) -> list[dict[str, Any]]:
    """Read per-step rows from ``rollout.jsonl`` next to a segment's PNGs/video.

    Skips the "start"/"terminated"/"diverged" marker lines (identified by an
    ``"event"`` key) — only per-step rows (carrying an ``"ego"`` pose) are returned,
    in step order. Returns ``[]`` if the trace file doesn't exist (e.g. an older run
    from before this file was written, or a 0-frame segment).
    """
    path = Path(png_dir) / "rollout.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "event" in row or "ego" not in row:
                continue
            rows.append(row)
    return rows


def _risk_and_ticks(
    rows: list[dict[str, Any]],
    metric: str,
    near_miss_thresh: float,
    strong_brake_mps2: float = -2.5,
) -> tuple[np.ndarray, list[float], list[str]]:
    """Map each step's raw metric to a [0, 1] risk scalar (0=safe, 1=very high risk) for
    coloring, plus (tick_positions, tick_labels) carrying the actual real-world value at
    each tick — so the colorbar reads e.g. "0.50m" / "1.20 m/s" instead of an unlabeled
    "high"/"low" that means nothing without knowing the metric's scale.
    """
    if metric == "clearance":
        # Clamp at 2x near_miss_thresh so "safe" steps aren't all squeezed into one color
        # band at the low end — the interesting range is near/below the near-miss threshold.
        cap = max(near_miss_thresh * 2.0, 1e-6)
        vals = np.array(
            [r.get("clearance_m") if r.get("clearance_m") is not None else cap for r in rows],
            dtype=np.float64,
        )
        risk = np.clip(1.0 - vals / cap, 0.0, 1.0)
        ticks = [0.0, 0.33, 0.66, 1.0]
        labels = [f"{cap * (1.0 - t):.2f}m" for t in ticks]
        return risk, ticks, labels
    if metric == "collision":
        risk = np.array([1.0 if r.get("collision") else 0.0 for r in rows], dtype=np.float64)
        return risk, [0.0, 1.0], ["no collision", "collision"]
    if metric == "near_miss":
        vals = np.array(
            [r.get("clearance_m") if r.get("clearance_m") is not None else np.inf for r in rows],
            dtype=np.float64,
        )
        risk = (vals <= near_miss_thresh).astype(np.float64)
        return (
            risk,
            [0.0, 1.0],
            [f"clearance > {near_miss_thresh:.2f}m", f"clearance <= {near_miss_thresh:.2f}m"],
        )
    if metric == "speed":
        vals = np.array([r.get("speed", 0.0) for r in rows], dtype=np.float64)
        cap = max(float(vals.max()), 1e-6) if vals.size else 1.0
        risk = np.clip(vals / cap, 0.0, 1.0)
        ticks = [0.0, 0.33, 0.66, 1.0]
        labels = [f"{cap * t:.2f}m/s" for t in ticks]
        return risk, ticks, labels
    if metric == "road_border":
        # Same clamp-at-2x-margin scheme as "clearance" — rb_dist_m is None for steps whose
        # frame carried no lane geometry, treated as "safe" (cap) rather than skewing the scale.
        cap = max(near_miss_thresh * 2.0, 1e-6)
        vals = np.array(
            [r.get("rb_dist_m") if r.get("rb_dist_m") is not None else cap for r in rows],
            dtype=np.float64,
        )
        risk = np.clip(1.0 - vals / cap, 0.0, 1.0)
        ticks = [0.0, 0.33, 0.66, 1.0]
        labels = [f"{cap * (1.0 - t):.2f}m" for t in ticks]
        return risk, ticks, labels
    if metric == "red_light":
        risk = np.array(
            [1.0 if r.get("red_light_violation") else 0.0 for r in rows], dtype=np.float64
        )
        return risk, [0.0, 1.0], ["no violation", "red light violation"]
    if metric == "strong_brake":
        # accel isn't logged per-step directly -- derive it the same way reproducer_rollout
        # does (accel = d(speed)/DT between consecutive steps), then reapply the same
        # two-consecutive-frame rule as scenario_generation.metrics.strong_brake.strong_brake_mask
        # (inlined rather than imported -- that module drags in scenario_generation.metrics'
        # package __init__, which pulls in torch/diffusion_planner.model transitively, and this
        # module stays import-light for wandb_closed_loop_workspace.py's standalone CLI).
        speeds = np.array([r.get("speed", 0.0) for r in rows], dtype=np.float64)
        accels = np.diff(speeds) / _DT if speeds.size > 1 else np.zeros(0, dtype=np.float64)
        accels = np.append(accels, accels[-1] if accels.size else 0.0)  # align length with rows
        raw = accels <= float(strong_brake_mps2)
        risk = np.zeros_like(raw, dtype=np.float64)
        risk[1:] = (raw[:-1] & raw[1:]).astype(np.float64)
        return (
            risk,
            [0.0, 1.0],
            ["no strong brake", f"accel <= {strong_brake_mps2:.2f} m/s²"],
        )
    raise ValueError(f"Unknown colormap metric: {metric!r} (choices: {METRIC_CHOICES})")


def render_trajectory_colormap(
    png_dir: str | Path,
    out_png: str | Path,
    *,
    metric: str = "clearance",
    near_miss_thresh: float = 0.5,
    strong_brake_mps2: float = -2.5,
    title: str | None = None,
    dpi: int = 110,
) -> Path | None:
    """Draw the ego's top-down trajectory colored by ``metric`` risk, with a colorbar.

    ``metric``: ``"clearance"`` (default; small clearance -> red/"very high"),
    ``"collision"`` (binary), ``"near_miss"`` (binary, clearance <= near_miss_thresh),
    ``"speed"``, ``"road_border"`` (distance to the nearest road/lane border, same scale
    scheme as ``"clearance"``; ``None`` on frames with no lane geometry -> treated as safe),
    ``"red_light"`` (binary), ``"strong_brake"`` (binary, accel <= ``strong_brake_mps2``).
    Returns ``out_png`` on success, or ``None`` if there was no per-step trace to draw
    (e.g. a 0-frame segment, or an old run predating the trace fields), or if the run never
    observed the quantity ``metric`` is derived from -- an unobserved metric has to be absent
    rather than drawn as a flat "no event", which reads as a measurement that was taken.
    """
    rows = load_step_trace(png_dir)
    if len(rows) < 2:
        return None
    if not any(_METRIC_TRACE_KEYS[metric] in r for r in rows):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    xy = np.array([r["ego"] for r in rows], dtype=np.float64)
    risk, tick_positions, tick_labels = _risk_and_ticks(
        rows, metric, near_miss_thresh, strong_brake_mps2
    )

    segments = np.concatenate([xy[:-1, None, :], xy[1:, None, :]], axis=1)
    seg_risk = (risk[:-1] + risk[1:]) / 2.0

    # Fixed figure size + fixed axes positions (NOT plt.subplots + bbox_inches="tight"): every
    # metric's PNG must come out at IDENTICAL pixel dimensions, else W&B's gallery panel (which
    # stacks a group's 5 metric images in one panel) warns "Images sizes do not match" and lays
    # them out wrong. The colorbar area on the right is always reserved; binary metrics
    # (collision/near_miss) simply leave it blank instead of shrinking the plot.
    fig = plt.figure(figsize=(6.4, 4.8), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.06, 0.03, 0.70, 0.84])
    ax.set_facecolor("white")
    lc = LineCollection(segments, cmap="turbo", linewidths=4, capstyle="round")
    lc.set_array(seg_risk)
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)

    if metric in _BINARY_METRICS:
        # A thin line briefly changing color is easy to miss for a rare, instantaneous event —
        # overlay an oversized marker at every step where it actually happened.
        event_xy = xy[risk >= 0.5]
        if event_xy.size:
            ax.scatter(
                event_xy[:, 0],
                event_xy[:, 1],
                c="red",
                s=110,
                zorder=4,
                edgecolors="black",
                linewidths=0.8,
            )

    # Start/end markers + text labels so the direction of travel is obvious at a glance
    # without having to trace the (often subtle) color gradient along the line.
    ax.scatter(xy[0, 0], xy[0, 1], c="black", s=60, zorder=5, edgecolors="white")
    ax.scatter(xy[-1, 0], xy[-1, 1], c="black", marker="s", s=60, zorder=5, edgecolors="white")
    # Anchored to the RIGHT of the point (grows inward/rightward) so a start/end at the far-left
    # edge of the path doesn't push its label off the figure and get clipped. Vertically, a point
    # in the upper half of the trajectory gets its label placed BELOW instead of above — otherwise
    # a point near the very top of the plot pushes the label up into the title text above the axes
    # (a tall, narrow trajectory puts its topmost point close enough to the title for this to bite).
    y_mid = (xy[:, 1].min() + xy[:, 1].max()) / 2.0
    for label, point in (("start", xy[0]), ("end", xy[-1])):
        upper_half = point[1] > y_mid
        ax.annotate(
            label,
            point,
            xytext=(7, -7 if upper_half else 7),
            textcoords="offset points",
            ha="left",
            va="top" if upper_half else "bottom",
            color="black",
            fontsize=9,
            fontweight="bold",
            zorder=6,
            annotation_clip=False,
        )

    pad = 5.0
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(f"{title}\nmetric: {metric}", color="black", fontsize=10)

    if metric not in _BINARY_METRICS:
        cax = fig.add_axes([0.80, 0.12, 0.025, 0.68])  # fixed colorbar slot (reserved always)
        cbar = fig.colorbar(lc, cax=cax)
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.yaxis.set_tick_params(color="black")
        plt.setp(cbar.ax.get_yticklabels(), color="black")
        cbar.outline.set_edgecolor("black")
        axis_label = _METRIC_AXIS_LABELS.get(metric, metric)
        if axis_label:
            cbar.set_label(axis_label, color="black", fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")  # no bbox_inches -> uniform dimensions across metrics
    plt.close(fig)
    return out_png


def render_trajectory_colormaps(
    png_dir: str | Path,
    out_dir: str | Path,
    stem: str,
    *,
    metrics: tuple[str, ...] = METRIC_CHOICES,
    near_miss_thresh: float = 0.5,
    strong_brake_mps2: float = -2.5,
    title: str | None = None,
    dpi: int = 110,
) -> dict[str, Path]:
    """Render one trajectory-colormap PNG per metric in ``metrics`` (default: all of
    :data:`METRIC_CHOICES`), named ``{out_dir}/{stem}_trajcolormap_{metric}.png``.

    Used to feed W&B's per-metric keys for the one representative episode — one trajectory,
    several colorings, instead of forcing a single metric choice up front. Returns a dict of
    only the metrics that actually rendered (a metric can legitimately be skipped, e.g. no
    per-step trace at all -> every metric skipped; this never raises).
    """
    out: dict[str, Path] = {}
    for metric in metrics:
        rendered = render_trajectory_colormap(
            png_dir,
            Path(out_dir) / f"{stem}_trajcolormap_{metric}.png",
            metric=metric,
            near_miss_thresh=near_miss_thresh,
            strong_brake_mps2=strong_brake_mps2,
            title=title,
            dpi=dpi,
        )
        if rendered is not None:
            out[metric] = rendered
    return out
