#!/usr/bin/env python3
"""Static-collision clearance heatmap along a route.

Scores each sim step's model prediction against stopped neighbours via
``rlvr.reward.compute_static_collision_penalty`` and projects the
per-step min OBB clearance onto the route arc-length to produce a heatmap.

Supports single-run and two-run (A vs B) comparison modes.

Input: dumped NPZs from ``scenario_generation.replay`` with
``parked_vehicles_yaml`` or ``static_npc_count`` enabled, plus a model
checkpoint and reward config.

Usage:
    python -m scenario_generation.tools.heatmap_static_collision \\
        --route /path/to/route.pkl \\
        --run_a /path/to/run_a_dir \\
        --model_a /path/to/model_a.pth \\
        --config /path/to/reward_config.json \\
        --output /path/to/heatmap.png

    # Two-run comparison:
    python -m scenario_generation.tools.heatmap_static_collision \\
        --route /path/to/route.pkl \\
        --run_a /path/to/run_a_dir --model_a /path/to/model_a.pth \\
        --run_b /path/to/run_b_dir --model_b /path/to/model_b.pth \\
        --config /path/to/reward_config.json \\
        --output /path/to/heatmap.png
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_generation.tools._heatmap_common import (
    bin_scalar_by_arc,
    build_route_polyline,
    load_route,
    recover_ego_world_pose_from_goal,
    segments_from_polyline,
)

_NPZ_RE = re.compile(r"replay_step_(\d+)\.npz$")


def _cluster_encounters(
    arcs: np.ndarray, clrs: np.ndarray, gap: float = 20.0,
) -> list[dict]:
    """Group consecutive arc positions into encounters."""
    if len(arcs) == 0:
        return []
    order = np.argsort(arcs)
    arcs, clrs = arcs[order], clrs[order]
    encounters: list[dict] = []
    start = 0
    for i in range(1, len(arcs)):
        if arcs[i] - arcs[i - 1] > gap:
            encounters.append({
                "arc_mean": float(np.mean(arcs[start:i])),
                "arc_lo": float(arcs[start]),
                "arc_hi": float(arcs[i - 1]),
                "min": float(np.min(clrs[start:i])),
                "mean": float(np.mean(clrs[start:i])),
                "n": i - start,
            })
            start = i
    encounters.append({
        "arc_mean": float(np.mean(arcs[start:])),
        "arc_lo": float(arcs[start]),
        "arc_hi": float(arcs[-1]),
        "min": float(np.min(clrs[start:])),
        "mean": float(np.mean(clrs[start:])),
        "n": len(arcs) - start,
    })
    return encounters


def _match_encounters(
    enc_a: list[dict], enc_b: list[dict],
) -> list[tuple[dict | None, dict | None]]:
    """Match encounters from two runs by arc overlap."""
    matched: list[tuple[dict | None, dict | None]] = []
    used_b: set[int] = set()
    for ea in enc_a:
        best_j, best_ov = -1, -1.0
        for j, eb in enumerate(enc_b):
            if j in used_b:
                continue
            ov = min(ea["arc_hi"], eb["arc_hi"]) - max(ea["arc_lo"], eb["arc_lo"])
            if ov > best_ov:
                best_ov, best_j = ov, j
        if best_j >= 0 and best_ov > 0:
            matched.append((ea, enc_b[best_j]))
            used_b.add(best_j)
        else:
            matched.append((ea, None))
    for j, eb in enumerate(enc_b):
        if j not in used_b:
            matched.append((None, eb))
    return matched


def _score_run_ego_actual(
    run_dir: Path,
    route,
    pts: np.ndarray,
    s: np.ndarray,
    stride: int = 1,
    max_steps: int | None = None,
    relevance_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score based on actual ego pose vs parked neighbors. No model needed.

    Only records steps where the closest parked vehicle is within
    ``relevance_m`` — distant parked vehicles are irrelevant and skipped.
    """
    from diffusion_planner.model.guidance.collision import batch_signed_distance_rect

    from rlvr.reward import _closest_points_between_rects
    from scenario_generation.tools._heatmap_common import project_to_polyline

    def _build_obb_corners(cx, cy, cos_h, sin_h, length, width, wheelbase):
        """Build (1, 4, 2) OBB corners from center pose + dims."""
        rear_overhang = (length - wheelbase) / 2.0
        x0, x1 = -rear_overhang, length - rear_overhang
        y0, y1 = -width / 2.0, width / 2.0
        local = torch.tensor([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=torch.float32)
        R = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], dtype=torch.float32)
        world = (R @ local.T).T + torch.tensor([cx, cy], dtype=torch.float32)
        return world.unsqueeze(0)

    npz_dir = run_dir / "npz"
    if not npz_dir.exists():
        npz_dir = run_dir
    npz_paths = sorted(
        p for p in npz_dir.glob("*.npz") if _NPZ_RE.search(p.name)
    )
    if stride > 1:
        npz_paths = npz_paths[::stride]
    if max_steps:
        npz_paths = npz_paths[:max_steps]
    if not npz_paths:
        raise SystemExit(f"No replay_step_*.npz under {npz_dir}")
    print(f"  {len(npz_paths)} steps to score (ego actual)")

    arc_positions = []
    min_clearances = []

    for i, path in enumerate(npz_paths):
        with np.load(path, allow_pickle=True) as raw:
            data_np = {k: raw[k] for k in raw.files if k != "version"}

        ex, ey, _ = recover_ego_world_pose_from_goal(data_np["goal_pose"], route)
        s_arc, _, _ = project_to_polyline(np.array([ex, ey]), pts, s)

        es = data_np["ego_shape"]
        if es.ndim == 2:
            es = es[0]
        wb, ego_len, ego_w = float(es[0]), float(es[1]), float(es[2])

        nb_past = data_np["neighbor_agents_past"]
        if nb_past.ndim == 3:
            nb_last = nb_past[:, -1, :]
        else:
            nb_last = nb_past[0, :, -1, :]
        valid = np.abs(nb_last[:, :2]).sum(axis=-1) > 1e-6
        if not valid.any():
            continue

        nb_xy = nb_last[valid, :2]
        nb_cos = nb_last[valid, 2]
        nb_sin = nb_last[valid, 3]
        if nb_last.shape[-1] < 8:
            raise ValueError(
                f"neighbor_agents_past has {nb_last.shape[-1]} columns, "
                "expected >= 8 (x, y, cos, sin, vx, vy, width, length)"
            )
        nb_w = nb_last[valid, 6]
        nb_l = nb_last[valid, 7]

        # Quick centre-distance pre-filter: skip neighbors whose centre
        # is further than relevance_m + max possible half-diagonal.
        centre_dists = np.sqrt(nb_xy[:, 0] ** 2 + nb_xy[:, 1] ** 2)
        nearby_mask = centre_dists < relevance_m + 15.0
        if not nearby_mask.any():
            continue

        ego_corners = _build_obb_corners(0.0, 0.0, 1.0, 0.0, ego_len, ego_w, wb)

        best_d = float("inf")
        for j in np.where(nearby_mask)[0]:
            nx, ny = float(nb_xy[j, 0]), float(nb_xy[j, 1])
            nc, ns_ = float(nb_cos[j]), float(nb_sin[j])
            nw, nl = float(nb_w[j]), float(nb_l[j])
            if nw < 0.1 or nl < 0.1:
                raise ValueError(
                    f"Neighbor {j} has invalid dimensions (w={nw}, l={nl})"
                )
            npc_corners = _build_obb_corners(nx, ny, nc, ns_, nl, nw, 0.0)

            pt1, pt2 = _closest_points_between_rects(ego_corners, npc_corners)
            d_val = float(torch.norm(pt1[0] - pt2[0]))

            sd = batch_signed_distance_rect(ego_corners, npc_corners)
            if float(sd[0]) < 0:
                d_val = float(sd[0])

            if d_val < best_d:
                best_d = d_val

        if best_d > relevance_m:
            continue

        arc_positions.append(s_arc)
        min_clearances.append(best_d)

        if (i + 1) % 200 == 0:
            print(f"    scored {i+1}/{len(npz_paths)}")

    return np.array(arc_positions), np.array(min_clearances)


def _score_run(
    run_dir: Path,
    model_path: Path,
    route,
    reward_config_path: Path,
    pts: np.ndarray,
    s: np.ndarray,
    device: str,
    stride: int = 1,
    max_steps: int | None = None,
    inference_delay: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a run. Returns (arc_positions (M,), min_clearance (M,))."""
    from rlvr.autoresearch.tools.audit_static_collision import (
        _score_prediction,
    )
    from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
    from scenario_generation.npz_loader import from_npz
    from scenario_generation.simulate import _predict_batch, load_model
    from scenario_generation.tools._heatmap_common import project_to_polyline

    reward_cfg = load_reward_config(str(reward_config_path))
    if not reward_cfg.static_collision_enabled:
        raise SystemExit(
            f"reward config {reward_config_path} has static_collision_enabled=false. "
            "Set it to true."
        )

    npz_dir = run_dir / "npz"
    if not npz_dir.exists():
        npz_dir = run_dir
    npz_paths = sorted(
        p for p in npz_dir.glob("*.npz")
        if _NPZ_RE.search(p.name)
    )
    if stride > 1:
        npz_paths = npz_paths[::stride]
    if max_steps:
        npz_paths = npz_paths[:max_steps]
    if not npz_paths:
        raise SystemExit(f"No replay_step_*.npz under {npz_dir}")
    print(f"  {len(npz_paths)} steps to score")

    model, model_args = load_model(str(model_path), device)

    arc_positions = []
    min_clearances = []

    for i, path in enumerate(npz_paths):
        with np.load(path, allow_pickle=True) as raw:
            data_np = {k: raw[k] for k in raw.files if k != "version"}

        ex, ey, eyaw = recover_ego_world_pose_from_goal(data_np["goal_pose"], route)

        scene = from_npz(str(path))
        preds = _predict_batch(
            model, model_args, scene, [scene.ego_agent_id], device,
            inference_delay=inference_delay,
        )
        ego_pred = preds.get(scene.ego_agent_id)
        if ego_pred is None:
            continue

        result = _score_prediction(data_np, ego_pred, reward_cfg, device)

        s_arc, _, _ = project_to_polyline(np.array([ex, ey]), pts, s)
        arc_positions.append(s_arc)
        min_clearances.append(result["sc_min_dist"])

        if (i + 1) % 100 == 0:
            print(f"    scored {i+1}/{len(npz_paths)}")

    return np.array(arc_positions), np.array(min_clearances)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--run_a", type=Path, required=True)
    p.add_argument("--model_a", type=Path, default=None)
    p.add_argument("--run_b", type=Path, default=None)
    p.add_argument("--model_b", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None,
                   help="Reward config JSON (required for --mode predicted)")
    p.add_argument("--mode", choices=["predicted", "ego_actual"], default="ego_actual",
                   help="'ego_actual' scores the real ego pose at each step (no model). "
                        "'predicted' scores the model's 80-step prediction (needs --model_a/b + --config).")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bin_m", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--safe_thresh", type=float, default=1.0,
                   help="Above this clearance both models are considered safe — "
                        "no winner is highlighted. Default 1.0m.")
    p.add_argument("--relevance_m", type=float, default=3.0,
                   help="Only record steps where closest parked vehicle is within "
                        "this distance (m). Default 3.")
    p.add_argument("--min_arc_m", type=float, default=None)
    p.add_argument("--max_arc_m", type=float, default=None)
    p.add_argument("--inference_delay", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading route {args.route}")
    route = load_route(args.route)
    pts, s = build_route_polyline(route)
    s_max = float(s[-1])
    print(f"Route polyline: {len(pts)} pts, arc length {s_max:.1f} m")

    if args.mode == "predicted":
        if args.model_a is None or args.config is None:
            raise SystemExit("--mode predicted requires --model_a and --config")

    print(f"[{args.label_a}] scoring {args.run_a} (mode={args.mode}, relevance={args.relevance_m}m)")
    if args.mode == "ego_actual":
        arc_a, clr_a = _score_run_ego_actual(
            args.run_a, route, pts, s, args.stride, args.max_steps,
            args.relevance_m,
        )
    else:
        arc_a, clr_a = _score_run(
            args.run_a, args.model_a, route, args.config,
            pts, s, device, args.stride, args.max_steps, args.inference_delay,
        )
    if len(arc_a) == 0:
        raise SystemExit("No steps had a parked vehicle within relevance range.")
    print(f"  {len(arc_a)} relevant steps, clearance min={clr_a.min():.3f} "
          f"mean={clr_a.mean():.3f} p5={np.percentile(clr_a, 5):.3f}")

    has_b = args.run_b is not None
    if has_b:
        print(f"[{args.label_b}] scoring {args.run_b} (mode={args.mode}, relevance={args.relevance_m}m)")
        if args.mode == "ego_actual":
            arc_b, clr_b = _score_run_ego_actual(
                args.run_b, route, pts, s, args.stride, args.max_steps,
                args.relevance_m,
            )
        else:
            if args.model_b is None:
                raise SystemExit("--mode predicted with --run_b requires --model_b")
            arc_b, clr_b = _score_run(
                args.run_b, args.model_b, route, args.config,
                pts, s, device, args.stride, args.max_steps, args.inference_delay,
            )
        print(f"  {len(arc_b)} scored steps, clearance min={clr_b.min():.3f} "
              f"mean={clr_b.mean():.3f} p5={np.percentile(clr_b, 5):.3f}")

    if args.min_arc_m is not None:
        mask_a = arc_a >= args.min_arc_m
        arc_a, clr_a = arc_a[mask_a], clr_a[mask_a]
        if has_b:
            mask_b = arc_b >= args.min_arc_m
            arc_b, clr_b = arc_b[mask_b], clr_b[mask_b]

    if args.max_arc_m is not None:
        mask_a = arc_a <= args.max_arc_m
        arc_a, clr_a = arc_a[mask_a], clr_a[mask_a]
        if has_b:
            mask_b = arc_b <= args.max_arc_m
            arc_b, clr_b = arc_b[mask_b], clr_b[mask_b]

    safe_thresh = args.safe_thresh

    enc_a = _cluster_encounters(arc_a, clr_a)
    enc_b = _cluster_encounters(arc_b, clr_b) if has_b else []
    matched = _match_encounters(enc_a, enc_b) if has_b else [(e, None) for e in enc_a]
    n_enc = len(matched)

    if n_enc == 0:
        raise SystemExit("No encounters found — no parked vehicles within relevance range.")

    labels, min_a_v, mean_a_v, min_b_v, mean_b_v = [], [], [], [], []
    for ea, eb in matched:
        arc = ea["arc_mean"] if ea else eb["arc_mean"]
        labels.append(f"#{len(labels)+1}\n{arc:.0f}m")
        min_a_v.append(ea["min"] if ea else np.nan)
        mean_a_v.append(ea["mean"] if ea else np.nan)
        min_b_v.append(eb["min"] if eb else np.nan)
        mean_b_v.append(eb["mean"] if eb else np.nan)
    min_a_v, mean_a_v = np.array(min_a_v), np.array(mean_a_v)
    min_b_v, mean_b_v = np.array(min_b_v), np.array(mean_b_v)

    fig = plt.figure(figsize=(max(14, n_enc * 2.0), 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.08)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])

    ax_map.set_aspect("equal")
    ax_map.plot(pts[:, 0], pts[:, 1], color="#cccccc", lw=2.0, zorder=1)
    for i, (ea, eb) in enumerate(matched):
        arc = ea["arc_mean"] if ea else eb["arc_mean"]
        idx = min(int(np.searchsorted(s, arc)), len(pts) - 1)
        tightest = min(
            ea["min"] if ea else 99.0, eb["min"] if eb else 99.0,
        )
        c = "#cc0000" if tightest < 0.4 else "#ff8800" if tightest < safe_thresh else "#228B22"
        ax_map.plot(pts[idx, 0], pts[idx, 1], "o", color=c, ms=8, zorder=10,
                    markeredgecolor="black", markeredgewidth=0.8)
        ax_map.annotate(
            f"{i+1}", (pts[idx, 0], pts[idx, 1]),
            fontsize=7, fontweight="bold", ha="center", va="bottom",
            xytext=(0, 7), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="black", lw=0.4),
        )
    ax_map.set_title("Route", fontsize=9)
    ax_map.tick_params(labelsize=6)

    x = np.arange(n_enc)
    w = 0.18
    ax_bars.bar(x - 1.5 * w, min_a_v, w, label=f"{args.label_a} min",
                color="#d62728", alpha=0.9, edgecolor="black", lw=0.4)
    ax_bars.bar(x - 0.5 * w, mean_a_v, w, label=f"{args.label_a} mean",
                color="#d62728", alpha=0.35, edgecolor="black", lw=0.4)
    if has_b:
        ax_bars.bar(x + 0.5 * w, min_b_v, w, label=f"{args.label_b} min",
                    color="#2166ac", alpha=0.9, edgecolor="black", lw=0.4)
        ax_bars.bar(x + 1.5 * w, mean_b_v, w, label=f"{args.label_b} mean",
                    color="#2166ac", alpha=0.35, edgecolor="black", lw=0.4)

    for xi, va, vb in zip(x, min_a_v, min_b_v if has_b else [np.nan] * n_enc):
        if not np.isnan(va):
            ax_bars.text(xi - 1.5 * w, va + 0.02, f"{va:.2f}", ha="center",
                         va="bottom", fontsize=6.5, color="#d62728", fontweight="bold")
        if has_b and not np.isnan(vb):
            ax_bars.text(xi + 0.5 * w, vb + 0.02, f"{vb:.2f}", ha="center",
                         va="bottom", fontsize=6.5, color="#2166ac", fontweight="bold")

    ax_bars.axhspan(0, 0.2, alpha=0.07, color="red")
    ax_bars.axhspan(0.2, 0.4, alpha=0.04, color="orange")
    ax_bars.axhline(0.2, color="#cc0000", lw=0.6, ls="--", alpha=0.5)
    ax_bars.axhline(0.4, color="#ff8800", lw=0.6, ls="--", alpha=0.5)
    ax_bars.axhline(safe_thresh, color="#228B22", lw=0.6, ls="--", alpha=0.3)

    if has_b:
        for i in range(n_enc):
            ma, mb = min_a_v[i], min_b_v[i]
            if np.isnan(ma) or np.isnan(mb):
                continue
            y_top = max(mean_a_v[i], mean_b_v[i]) + 0.12
            if ma >= safe_thresh and mb >= safe_thresh:
                ax_bars.text(x[i], y_top, "both safe", ha="center", fontsize=6,
                             color="#228B22", fontstyle="italic")
            elif mb > ma:
                ax_bars.text(x[i], y_top, args.label_b, ha="center", fontsize=6,
                             color="#2166ac", fontweight="bold")
            elif ma > mb:
                ax_bars.text(x[i], y_top, args.label_a, ha="center", fontsize=6,
                             color="#d62728", fontweight="bold")

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, fontsize=7)
    ax_bars.set_ylabel("Clearance (m)", fontsize=10)
    ax_bars.legend(fontsize=7, ncol=4, loc="upper right")
    ax_bars.grid(axis="y", alpha=0.15)
    all_v = np.concatenate(
        [v[~np.isnan(v)] for v in [min_a_v, mean_a_v, min_b_v, mean_b_v]]
    )
    ax_bars.set_ylim(0, min(3.5, float(all_v.max()) * 1.2))

    tight = sum(
        1 for i in range(n_enc)
        if not np.isnan(min_a_v[i])
        and (not has_b or not np.isnan(min_b_v[i]))
        and (min_a_v[i] < safe_thresh or (has_b and min_b_v[i] < safe_thresh))
    )
    title_parts = [
        f"{args.label_a}" + (f" vs {args.label_b}" if has_b else ""),
        f"{n_enc} encounters",
        f"{tight} tight (< {safe_thresh}m)",
    ]
    fig.suptitle(
        "Parked vehicle avoidance: " + ", ".join(title_parts),
        fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.90)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")

    np.savez(
        args.output.with_suffix(".npz"),
        arc_a=arc_a, clr_a=clr_a,
        **({"arc_b": arc_b, "clr_b": clr_b} if has_b else {}),
        route_pts=pts, route_s=s,
    )
    print(f"Saved {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
