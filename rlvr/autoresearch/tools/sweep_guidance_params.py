#!/usr/bin/env python3
"""Per-scene guidance-parameter grid sweep on a FROZEN model.

For each scene, generates one deterministic trajectory per grid combo of
(eta_lat, eta_col[, stretch]) — all combos in a single batched forward pass —
scores them with the canonical reward, and records the best combo. The zero
combo (eta_lat=0, eta_col=0, stretch=1) IS the unguided det baseline, so the
sweep answers two questions at once:

  1. Feasibility: can ANY guidance params make this model avoid each scene?
  2. Labels: per-scene best params for supervised explorer regression.

Guidance functions used (all support per-sample [K] tensors):
  lateral                  -- PlannerRFT Eq.2, target offset = lambda_lat * eta_lat
  collision_swerve_batched -- signed proximity-gated swerve (+ = left)
  speed_stretch_batched    -- displacement stretch (< 1 = slow down), optional

Usage:
    python -m rlvr.autoresearch.tools.sweep_guidance_params \
        --model_path <model.pth> --scenes <scenes.json> \
        --config <reward_config.json> --ego_shape WB,L,W \
        --output_dir <dir> \
        [--eta_lat_grid="-1,-0.5,0,0.5,1"] [--eta_col_grid="-1,-0.5,0,0.5,1"] \
        [--stretch_grid 1.0] [--lambda_lat 2.5] [--col_scale 2.0] \
        [--col_range 8.0] [--guidance_scale 0.5] [--audit_scenes 0]

Outputs:
    <output_dir>/sweep_labels.json   per-scene combo table + best params + summary
    <output_dir>/audit/<scene>.png   trajectory-fan renders (first --audit_scenes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401 -- registers batched guidance
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from exploration_policy.utils import generate_reference_trajectory
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.reward import compute_reward_batch


def build_grid(eta_lat_grid, eta_col_grid, stretch_grid):
    """Cartesian grid -> three [K] float lists. Asserts the zero combo exists."""
    combos = [
        (el, ec, st)
        for el in eta_lat_grid
        for ec in eta_col_grid
        for st in stretch_grid
    ]
    zero = (0.0, 0.0, 1.0)
    if zero not in combos:
        raise SystemExit(
            "Grid must contain the zero combo (eta_lat=0, eta_col=0, stretch=1) "
            "— it is the unguided det baseline."
        )
    return combos, combos.index(zero)


def make_composer(eta_lat, eta_col, stretch, args) -> GuidanceComposer:
    """Composer with per-sample [K] tensors for all three knobs."""
    fns = [
        GuidanceConfig(
            name="lateral", enabled=True, scale=args.lat_scale,
            params={"lambda_lat": args.lambda_lat, "eta_lat": eta_lat},
        ),
        GuidanceConfig(
            name="collision_swerve_batched", enabled=True, scale=args.col_scale,
            params={"eta_col": eta_col, "range": args.col_range},
        ),
    ]
    if bool((stretch != 1.0).any()):
        fns.append(GuidanceConfig(
            name="speed_stretch_batched", enabled=True, scale=args.stretch_scale,
            params={"stretch": stretch},
        ))
    set_cfg = GuidanceSetConfig(functions=fns, global_scale=args.guidance_scale)
    return GuidanceComposer(set_cfg)


@torch.no_grad()
def sweep_scene(model, model_args, npz_path, combos, zero_idx, rcfg, args, device):
    """Run the full grid on one scene. Returns per-scene result dict."""
    data = load_npz_data(npz_path, device)
    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)

    x_ref_np = generate_reference_trajectory(model, model_args, norm_data, device)
    x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device)
    norm_data["reference_trajectory"] = x_ref

    K = len(combos)
    eta_lat = torch.tensor([c[0] for c in combos], device=device, dtype=torch.float32)
    eta_col = torch.tensor([c[1] for c in combos], device=device, dtype=torch.float32)
    stretch = torch.tensor([c[2] for c in combos], device=device, dtype=torch.float32)

    K_data = {}
    for k, v in norm_data.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == 1:
            K_data[k] = v.expand(K, *v.shape[1:]).contiguous()
        else:
            K_data[k] = v

    composer = make_composer(eta_lat, eta_col, stretch, args)
    trajs = _batched_generate_varied_noise(
        model, model_args, K_data,
        noise_min=0.0, noise_max=0.0, first_deterministic=False,
        composer=composer, device=device,
    )  # [K, T, 4]

    breakdowns = compute_reward_batch(trajs.float(), data, rcfg)

    combo_rows = []
    for i, (el, ec, st) in enumerate(combos):
        r = breakdowns[i]
        combo_rows.append({
            "eta_lat": el, "eta_col": ec, "stretch": st,
            "total": float(r.total),
            "static_crossing": bool(r.static_crossing),
            "sc_min_dist": float(r.sc_min_dist),
            "rb_cross": bool(r.rb_crossing),
            "lane_cross": bool(r.lane_crossing),
            "cl": float(r.centerline),
            "stopped": bool(getattr(r, "stopped", False)),
        })

    det = combo_rows[zero_idx]
    # Best = clean (no sc crossing, no rb crossing, not stopped) with max total.
    clean = [c for c in combo_rows
             if not c["static_crossing"] and not c["rb_cross"] and not c["stopped"]]
    best = max(clean, key=lambda c: c["total"]) if clean else None

    sc_n_stopped = int(breakdowns[zero_idx].sc_n_stopped)
    if not det["static_crossing"]:
        status = "already_clean"
    elif best is not None:
        status = "solved"
    else:
        status = "unsolved"

    return {
        "scene": Path(npz_path).name,
        "scene_path": str(npz_path),
        "sc_n_stopped": sc_n_stopped,
        "status": status,
        "det": det,
        "best": best,
        "combos": combo_rows,
        "_trajs": trajs.cpu().numpy(),  # stripped before JSON; used by audit render
    }


def render_audit(result, combos, zero_idx, out_png):
    """Trajectory-fan render: all combos colored by eta_col, det in black."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rlvr.autoresearch.tools.ghost_sim_common import extract_stopped_neighbors
    from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base
    from scenario_generation.visualize import draw_agent_box

    trajs = result["_trajs"]
    fig, ax = plt.subplots(figsize=(11, 11))
    draw_scene_base(ax, result["scene_path"])
    for (x, y, h, length, w) in extract_stopped_neighbors(result["scene_path"]):
        draw_agent_box(ax, x, y, h, length, w, color="crimson", alpha=0.5)

    cmap = plt.get_cmap("coolwarm")
    for i, (el, ec, st) in enumerate(combos):
        if i == zero_idx:
            continue
        color = cmap(0.5 * (ec + 1.0))
        crossed = result["combos"][i]["static_crossing"]
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], "-", color=color,
                lw=0.9, alpha=0.35 if crossed else 0.8)
    ax.plot(trajs[zero_idx, :, 0], trajs[zero_idx, :, 1], "-", color="black",
            lw=2.2, label="det (no guidance)")
    if result["best"] is not None:
        b = result["best"]
        bi = result["combos"].index(b)
        ax.plot(trajs[bi, :, 0], trajs[bi, :, 1], "-", color="lime", lw=2.2,
                label=(f"best lat={b['eta_lat']:+.2f} col={b['eta_col']:+.2f} "
                       f"st={b['stretch']:.2f} sc={b['sc_min_dist']:+.2f}m"))
    ax.set_title(
        f"{result['scene']}  [{result['status']}]  "
        f"det sc={result['det']['sc_min_dist']:+.2f}m  stopped={result['sc_n_stopped']}"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    span = np.abs(trajs[..., :2]).max() + 10
    ax.set_xlim(-10, max(span, 40))
    ax.set_ylim(-span / 2, span / 2)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ego_shape", required=True, help="WB,L,W (consistency check)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eta_lat_grid", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--eta_col_grid", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--stretch_grid", default="1.0")
    parser.add_argument("--lambda_lat", type=float, default=2.5)
    parser.add_argument("--lat_scale", type=float, default=1.0)
    parser.add_argument("--col_scale", type=float, default=2.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--audit_scenes", type=int, default=0,
                        help="Render trajectory fans for the first N scenes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    rcfg = load_reward_config(args.config)
    if not rcfg.static_collision_enabled:
        raise SystemExit("--config must have static_collision_enabled=true")

    eta_lat_grid = [float(x) for x in args.eta_lat_grid.split(",")]
    eta_col_grid = [float(x) for x in args.eta_col_grid.split(",")]
    stretch_grid = [float(x) for x in args.stretch_grid.split(",")]
    combos, zero_idx = build_grid(eta_lat_grid, eta_col_grid, stretch_grid)
    print(f"[sweep] {len(combos)} combos/scene "
          f"(lat {eta_lat_grid} x col {eta_col_grid} x stretch {stretch_grid})")

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    model, model_args = load_model(args.model_path, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "audit"
    if args.audit_scenes > 0:
        audit_dir.mkdir(exist_ok=True)

    results = []
    for si, p in enumerate(scene_paths):
        try:
            with np.load(p, allow_pickle=True) as npz:
                if "ego_shape" not in set(npz.keys()):
                    print(f"  [skip] {Path(p).name}: missing ego_shape")
                    continue
                es = np.asarray(npz["ego_shape"]).reshape(-1)[:3]
            if not np.allclose(es, ego_shape, atol=1e-2):
                print(f"  [skip] {Path(p).name}: ego_shape={es.tolist()}")
                continue
            res = sweep_scene(model, model_args, p, combos, zero_idx, rcfg, args, device)
        except Exception as e:  # noqa: BLE001
            print(f"  [err ] {Path(p).name}: {e}")
            continue

        if si < args.audit_scenes:
            render_audit(res, combos, zero_idx, audit_dir / f"{Path(p).stem}.png")
        res.pop("_trajs")
        results.append(res)

        b = res["best"]
        best_str = (f"best(lat={b['eta_lat']:+.2f} col={b['eta_col']:+.2f} "
                    f"st={b['stretch']:.2f} sc={b['sc_min_dist']:+.2f})") if b else "NO CLEAN COMBO"
        print(f"  [{si:3d}] {res['status']:13s} det_sc={res['det']['sc_min_dist']:+.3f} "
              f"{best_str}  {Path(p).name}")

    n = len(results)
    n_avoid = sum(r["status"] != "already_clean" for r in results)
    n_solved = sum(r["status"] == "solved" for r in results)
    n_unsolved = sum(r["status"] == "unsolved" for r in results)
    summary = {
        "n_scenes": n,
        "n_already_clean": n - n_avoid,
        "n_needs_avoidance": n_avoid,
        "n_solved": n_solved,
        "n_unsolved": n_unsolved,
        "solve_rate": (n_solved / n_avoid) if n_avoid else None,
        "grid": {"eta_lat": eta_lat_grid, "eta_col": eta_col_grid,
                 "stretch": stretch_grid},
        "guidance_args": {
            "lambda_lat": args.lambda_lat, "lat_scale": args.lat_scale,
            "col_scale": args.col_scale, "col_range": args.col_range,
            "stretch_scale": args.stretch_scale,
            "guidance_scale": args.guidance_scale,
        },
    }
    with open(out_dir / "sweep_labels.json", "w") as f:
        json.dump({"summary": summary, "scenes": results}, f, indent=1)

    print(f"\n[sweep] {n} scenes: {summary['n_already_clean']} already clean, "
          f"{n_solved}/{n_avoid} solved by guidance, {n_unsolved} unsolved")
    print(f"Wrote {out_dir / 'sweep_labels.json'}")


if __name__ == "__main__":
    main()
