#!/usr/bin/env python3
"""Classify scenes as avoidance / non-avoidance using the guidance explorer.

The exploration policy was trained to request lateral / collision-swerve
guidance exactly when the frozen planner's deterministic trajectory needs
an avoidance correction, and to stay inert (eta ~ 0) otherwise. This tool
runs the policy (deterministic = Beta means) over an NPZ scene list and
flags a scene as "avoidance" when the requested guidance exceeds
configurable per-head thresholds — a weak signal below threshold is
treated as not-really-avoidance.

No guided generation and no reward scoring happen here: per scene this is
one deterministic planner pass (for the reference trajectory), one frozen
encoder pass, and the policy head — all BATCHED across scenes
(--batch_size, default 32; requires shape-homogeneous NPZs, e.g. 320
neighbor slots).

Outputs a JSON report (per-scene etas + flag + which head(s) triggered,
plus summary counts and |eta| distributions) and, optionally, plain NPZ
path lists for the two classes, directly usable as dataset lists.

Usage:
    python -m rlvr.autoresearch.tools.classify_avoidance_scenes \
        --model_path <base.pth> --policy_dir <dir with exploration_policy.pth> \
        --scenes <scenes.json> --out <report.json> \
        [--lat_thresh 0.15] [--col_thresh 0.15] [--rule any] \
        [--out_avoidance_list <a.json>] [--out_normal_list <n.json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import (
    det_inference_batched,
    load_model,
)
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer


@torch.no_grad()
def scene_etas(model, model_args, policy, heads, npz_path, device):
    """Deterministic per-head etas in [-1, 1] for one scene.

    Returns (etas, det_traj, norm_data): the unguided deterministic
    trajectory IS the policy's x_ref input — the etas are a judgment of
    that specific baseline plan, not of the scene in isolation.
    """
    data = load_npz_data(npz_path, device)
    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)
    x_ref_np = generate_reference_trajectory(model, model_args, norm_data, device)
    x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device)
    norm_data["reference_trajectory"] = x_ref
    enc = run_frozen_encoder(model, norm_data)
    out = policy(enc, x_ref, deterministic=True)
    etas = {h: float(2.0 * out.dists[h].mean - 1.0) for h in heads}
    return etas, x_ref_np, norm_data


@torch.no_grad()
def render_verdict(model, model_args, scene_path, etas, det, norm_data,
                   is_avoid, triggers, args, device, out_png,
                   verdict_label: str | None = None):
    """Scene render for human judgment: det trajectory + stopped-neighbor
    OBBs + verdict; flagged scenes also show the policy-guided trajectory
    (what the policy wants to do instead)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from guidance_gui.generate_samples import generate_samples
    from rlvr.autoresearch.tools.ghost_sim_common import extract_stopped_neighbors
    from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base
    from scenario_generation.visualize import draw_agent_box

    guided = None
    if is_avoid:
        eta_t = {h: torch.tensor([v], device=device) for h, v in etas.items()}
        composer = make_composer(eta_t, args)
        guided = generate_samples(model=model, model_args=model_args,
                                  data=norm_data, noise_scale=0.0, n_samples=1,
                                  composer=composer, device=device)[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_scene_base(ax, scene_path)
    for (x, y, h, length, w) in extract_stopped_neighbors(scene_path):
        draw_agent_box(ax, x, y, h, length, w, color="crimson", alpha=0.5)
    # ego footprint at t0 (ego frame: rear axle at origin, heading 0)
    wb, length, width = args.ego_shape_t
    draw_agent_box(ax, 0.0, 0.0, 0.0, length, width, color="royalblue",
                   alpha=0.35, wheelbase=wb)
    ax.plot(det[:, 0], det[:, 1], "-", color="black", lw=2.2, label="baseline det")
    if guided is not None:
        ax.plot(guided[:, 0], guided[:, 1], "--", color="lime", lw=2.2,
                label="policy-guided")
    verdict = verdict_label or (
        ("AVOIDANCE [" + "+".join(triggers) + "]") if is_avoid else "normal")
    eta_str = " ".join(f"{h[:3]}={v:+.2f}" for h, v in etas.items())
    ax.set_title(f"{Path(scene_path).stem}\n{verdict}  η: {eta_str}",
                 color=("darkred" if is_avoid else "darkgreen"))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    span = float(np.abs(det[..., :2]).max()) + 12
    ax.set_xlim(-12, max(span, 40))
    ax.set_ylim(-span / 2, span / 2)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def classify(etas: dict[str, float], lat_thresh: float, col_thresh: float,
             rule: str) -> tuple[bool, list[str]]:
    """Return (is_avoidance, triggered_heads) from the per-head etas."""
    triggers = []
    if "lateral" in etas and abs(etas["lateral"]) >= lat_thresh:
        triggers.append("lateral")
    if "collision" in etas and abs(etas["collision"]) >= col_thresh:
        triggers.append("collision")
    if rule == "any":
        return bool(triggers), triggers
    # rule == "both": only count scenes where BOTH heads fire
    return ("lateral" in triggers and "collision" in triggers), triggers


def _pct(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    a = np.abs(np.asarray(vals))
    qs = np.percentile(a, [5, 25, 50, 75, 95])
    return {
        "mean": round(float(a.mean()), 4),
        "p5": round(float(qs[0]), 4), "p25": round(float(qs[1]), 4),
        "p50": round(float(qs[2]), 4), "p75": round(float(qs[3]), 4),
        "p95": round(float(qs[4]), 4), "max": round(float(a.max()), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True,
                        help="frozen base planner checkpoint (must be the "
                             "model the policy was trained against)")
    parser.add_argument("--policy_dir", required=True,
                        help="dir with exploration_policy.pth + "
                             "exploration_policy_config.json")
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--lat_thresh", type=float, default=0.15,
                        help="min |eta_lateral| to count as an avoidance "
                             "request (policy inertness bar on normal scenes "
                             "is ~0.1; below this = weak signal)")
    parser.add_argument("--col_thresh", type=float, default=0.15,
                        help="min |eta_collision| to count as an avoidance "
                             "request")
    parser.add_argument("--rule", choices=["any", "both"], default="any",
                        help="'any': either head over threshold flags the "
                             "scene; 'both': require both heads")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="scenes per batched det+encoder+policy pass")
    parser.add_argument("--render_only_avoidance", action="store_true",
                        help="with --render_dir: render only flagged (and "
                             "clearance-demoted) scenes, skip normals")
    parser.add_argument("--out_avoidance_list", default=None,
                        help="optional JSON list of flagged NPZ paths")
    parser.add_argument("--out_normal_list", default=None,
                        help="optional JSON list of non-flagged NPZ paths")
    parser.add_argument("--render_dir", default=None,
                        help="render per-scene verdict PNGs (ego footprint + "
                             "det trajectory + stopped neighbors; flagged "
                             "scenes also show the policy-guided trajectory) "
                             "+ collages")
    parser.add_argument("--ego_shape", default=None,
                        help="WB,L,W — required with --render_dir or "
                             "--verify_clearance, no default")
    parser.add_argument("--verify_clearance", type=float, default=0.0,
                        help="if > 0: cross-check flagged scenes with the "
                             "canonical det-plan OBB clearance vs stopped "
                             "neighbors and DEMOTE to normal when clearance "
                             ">= this value (the baseline plan was already "
                             "safe — policy false positive). Suggested: 1.5. "
                             "Mirrors the deployment activation gate.")
    # Guidance envelope — only used for the guided trajectory in renders;
    # must match what the policy was trained against.
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--head_protect", type=int, default=0)
    parser.add_argument("--envelope", choices=["v1", "v2"], default="v1")
    parser.add_argument("--lambda_col", type=float, default=3.0)
    parser.add_argument("--slow_composer", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)
    if "lateral" not in heads and "collision" not in heads:
        raise ValueError(
            f"policy heads {heads} contain neither 'lateral' nor 'collision' "
            "— this classifier keys on the avoidance heads")
    if args.rule == "both" and not ("lateral" in heads and "collision" in heads):
        raise ValueError(
            f"--rule both requires BOTH avoidance heads; policy has {heads} "
            "— no scene could ever be flagged")

    with open(args.scenes) as f:
        paths = json.load(f)

    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir or args.verify_clearance > 0:
        if not args.ego_shape:
            raise ValueError("--render_dir / --verify_clearance require "
                             "--ego_shape WB,L,W — no default")
        args.ego_shape_t = tuple(float(x) for x in args.ego_shape.split(","))
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)

    from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data

    rows, per_head_abs = [], {h: [] for h in heads}
    done = 0
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        datas = [load_npz_data(p, device) for p in batch_paths]
        # GT futures are not model inputs and may differ in width across
        # pools (3-col raw vs 4-col) — drop them so mixed lists stack.
        stack_in = [{k: v for k, v in d.items()
                     if k not in ("ego_agent_future", "neighbor_agents_future")}
                    for d in datas]
        batch = _stack_scene_data(stack_in, device)
        norm_batch = _normalize_batch(batch, model_args)
        det_b = det_inference_batched(model, model_args, datas, device,
                                      norm_batch=norm_batch)  # [B, T, 4]
        norm_batch["reference_trajectory"] = det_b
        enc = run_frozen_encoder(model, norm_batch)
        out = policy(enc, det_b, deterministic=True)
        etas_b = {h: (2.0 * out.dists[h].mean - 1.0).cpu() for h in heads}
        det_np = det_b.cpu().numpy()

        for i, sp in enumerate(batch_paths):
            etas = {h: float(etas_b[h][i]) for h in heads}
            det = det_np[i]
            is_avoid, triggers = classify(
                etas, args.lat_thresh, args.col_thresh, args.rule)
            row = {
                "scene": sp,
                "etas": {h: round(v, 4) for h, v in etas.items()},
                "avoidance": is_avoid,
                "triggered": triggers,
            }
            if is_avoid and args.verify_clearance > 0:
                from rlvr.autoresearch.tools.ghost_sim_common import (
                    extract_stopped_neighbors,
                )
                from scenario_generation.explorer_runner import (
                    plan_static_clearance,
                )
                boxes = extract_stopped_neighbors(sp)
                det_clr = (plan_static_clearance(det, boxes, args.ego_shape_t,
                                                 device)
                           if boxes else float("inf"))
                row["det_clearance"] = round(float(det_clr), 3)
                if det_clr >= args.verify_clearance:
                    # baseline plan already safe — policy false positive
                    is_avoid = False
                    row["avoidance"] = False
                    row["demoted"] = True
            rows.append(row)
            for h, v in etas.items():
                per_head_abs[h].append(v)
            if render_dir and (is_avoid or row.get("demoted")
                               or not args.render_only_avoidance):
                # pool-prefix the PNG name (same-basename scenes across pools)
                cls = ("avoid" if is_avoid
                       else "demoted" if row.get("demoted") else "normal")
                png = render_dir / (
                    f"{cls}__{Path(sp).parent.name}__{Path(sp).stem}.png")
                label = (f"DEMOTED (det clearance "
                         f"{row['det_clearance']}m)" if cls == "demoted"
                         else None)
                # per-scene norm_data for the guided-trajectory render
                nd = {k: v.clone() if isinstance(v, torch.Tensor) else v
                      for k, v in datas[i].items()}
                nd = model_args.observation_normalizer(nd)
                nd["reference_trajectory"] = det_b[i:i + 1]
                render_verdict(model, model_args, sp, etas, det, nd,
                               is_avoid or bool(row.get("demoted")), triggers,
                               args, device, png, verdict_label=label)
        done += len(batch_paths)
        if done % (args.batch_size * 4) < args.batch_size or done == len(paths):
            print(f"  [classify] {done}/{len(paths)}")

    n_avoid = sum(r["avoidance"] for r in rows)
    n_demoted = sum(r.get("demoted", False) for r in rows)
    summary = {
        "n_scenes": len(rows),
        "n_avoidance": n_avoid,
        "n_normal": len(rows) - n_avoid,
        "n_demoted_by_clearance": n_demoted,
        "thresholds": {"lat": args.lat_thresh, "col": args.col_thresh,
                       "rule": args.rule,
                       "verify_clearance": args.verify_clearance},
        "policy_dir": args.policy_dir,
        "model_path": args.model_path,
        "abs_eta_distribution": {h: _pct(v) for h, v in per_head_abs.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "scenes": rows}, f, indent=1)

    if args.out_avoidance_list:
        with open(args.out_avoidance_list, "w") as f:
            json.dump([r["scene"] for r in rows if r["avoidance"]], f, indent=1)
    if args.out_normal_list:
        with open(args.out_normal_list, "w") as f:
            json.dump([r["scene"] for r in rows if not r["avoidance"]], f, indent=1)

    if render_dir:
        from rlvr.autoresearch.tools.eval_policy_avoidance import make_collage
        for cls in ("avoid", "demoted", "normal"):
            pngs = sorted(render_dir.glob(f"{cls}__*.png"))
            make_collage(pngs, render_dir / f"collage_{cls}.png")
        print(f"[render] {len(list(render_dir.glob('*__*.png')))} PNGs + "
              f"collages -> {render_dir}")

    print(f"\n[classify] {len(rows)} scenes: {n_avoid} avoidance, "
          f"{len(rows) - n_avoid} normal "
          f"(|eta_lat|>={args.lat_thresh} {args.rule} "
          f"|eta_col|>={args.col_thresh})"
          + (f", {n_demoted} demoted by det clearance "
             f">={args.verify_clearance}m" if args.verify_clearance > 0
             else ""))
    for h, st in summary["abs_eta_distribution"].items():
        if st:
            print(f"  |eta_{h}|: mean {st['mean']} p50 {st['p50']} "
                  f"p95 {st['p95']} max {st['max']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
