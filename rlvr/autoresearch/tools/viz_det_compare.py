#!/usr/bin/env python3
"""Render deterministic trajectories from one or two models side-by-side.

Loads one or two models (each with optional LoRA), runs det inference on
shared scenes, and renders per-scene PNGs + a collage showing trajectories
on the same map. If --config is provided, also scores each trajectory with
the reward config.

Usage:
    python -m rlvr.autoresearch.tools.viz_det_compare \
        --model_a <base.pth> --label_a "Baseline" \
        [--model_b <trained.pth> --label_b "a06"] \
        [--lora_a <lora_dir>] [--lora_b <lora_dir>] \
        --scenes <scenes.json> \
        [--config <reward_config.json> --ego_shape WB,L,W] \
        --output_dir <dir> \
        [--collage_cols 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import (
    aggregate_stats,
    det_inference_batched,
    load_model,
    print_summary,
    reward_breakdown_to_det_dict,
)
from rlvr.autoresearch.tools.render_metadata import path_label, render_tag, write_render_meta
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base, draw_traj
from rlvr.reward import compute_reward_batch


def _load_model_with_lora(model_path, lora_path, device):
    model, model_args = load_model(model_path, device)
    if lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint

        print(f"  Loading LoRA: {lora_path}")
        model = load_lora_checkpoint(model, lora_path)
        model.eval()
    return model, model_args


def _draw_current_neighbors(ax, npz_path):
    """Draw all present neighbors at t=0 using the existing ghost-replay OBB helpers."""
    from rlvr.autoresearch.tools.ghost_replay_openloop import _neighbor_boxes_at, _real_neighbors
    from rlvr.autoresearch.tools.ghost_sim_common import _NB_COLOR
    from rlvr.autoresearch.tools.recovery_sim import _draw_agent_box

    idx_dims, nb_past, _ = _real_neighbors(npz_path)
    if nb_past is None:
        return
    for nx, ny, nh, nl, nw in _neighbor_boxes_at(nb_past, nb_past.shape[1] - 1, idx_dims):
        _draw_agent_box(ax, nx, ny, nh, nl, nw, _NB_COLOR, alpha=0.78, lw=1.3, zorder=14)


def _plans(model, margs, policy, heads, datas, device, args):
    """Deterministic plans [B,T,4] for a batch. Plain det inference unless a guidance policy is
    given, in which case each scene's det plan is re-generated under the policy-chosen etas."""
    det = det_inference_batched(model, margs, datas, device)
    if policy is None:
        return det
    from exploration_policy.utils import run_frozen_encoder
    from guidance_gui.generate_samples import generate_samples
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    out = []
    for bi, data in enumerate(datas):
        det_i = det[bi].cpu().numpy()
        norm = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        norm = margs.observation_normalizer(norm)
        x_ref = torch.from_numpy(np.ascontiguousarray(det_i)).float().unsqueeze(0).to(device)
        norm["reference_trajectory"] = x_ref
        enc = run_frozen_encoder(model, norm)
        pout = policy(enc, x_ref, deterministic=True)
        etas = {h: (2.0 * pout.dists[h].mean - 1.0).reshape(1) for h in heads}
        traj = generate_samples(
            model=model,
            model_args=margs,
            data=norm,
            noise_scale=0.0,
            n_samples=1,
            composer=make_composer(etas, args),
            device=device,
        )[0]
        out.append(torch.as_tensor(traj, device=device))
    return torch.stack(out)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_a", required=True)
    p.add_argument("--model_b", default=None)
    p.add_argument("--lora_a", default=None)
    p.add_argument("--lora_b", default=None)
    p.add_argument("--policy_a", default=None, help="guidance/exploration policy dir for side A")
    p.add_argument("--policy_b", default=None, help="guidance/exploration policy dir for side B")
    p.add_argument("--label_a", default="Model A")
    p.add_argument("--label_b", default="Model B")
    p.add_argument("--scenes", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--ego_shape", default=None, help="WB,L,W; required when --config is set")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--collage_cols", type=int, default=5)
    p.add_argument(
        "--indices", type=int, nargs="*", default=None, help="Render only these scene indices"
    )
    p.add_argument(
        "--max_scenes", type=int, default=None, help="Render only the first N selected scenes"
    )
    p.add_argument(
        "--show_gt",
        action="store_true",
        help="overlay the scene's ego_agent_future (hand-drawn GT) as a 3rd trajectory",
    )
    p.add_argument(
        "--no_viz", action="store_true", help="Skip per-scene PNGs and collage; only print stats."
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.config and not args.ego_shape:
        raise SystemExit("--ego_shape is required when --config is set")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")]) if args.ego_shape else None
    rcfg = load_reward_config(args.config) if args.config else None

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    scene_paths = (
        scene_paths.get("files", scene_paths) if isinstance(scene_paths, dict) else scene_paths
    )
    if args.indices is not None:
        bad = [i for i in args.indices if i < 0 or i >= len(scene_paths)]
        if bad:
            raise SystemExit(f"--indices out of range for {len(scene_paths)} scenes: {bad[:10]}")
        scene_paths = [scene_paths[i] for i in args.indices]
    if args.max_scenes is not None and args.max_scenes > 0:
        scene_paths = scene_paths[: args.max_scenes]

    print(f"Loading model A: {args.model_a}")
    model_a, args_a = _load_model_with_lora(args.model_a, args.lora_a, device)
    policy_a = heads_a = None
    if args.policy_a:
        from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy

        print(f"Loading guidance policy A: {args.policy_a}")
        policy_a, heads_a = load_policy(args.policy_a, args_a, device)

    model_b = args_b = None
    policy_b = heads_b = None
    model_b_path = args.model_b or (args.model_a if args.lora_b or args.policy_b else None)
    if model_b_path:
        print(f"Loading model B: {model_b_path}")
        model_b, args_b = _load_model_with_lora(model_b_path, args.lora_b, device)
        if args.policy_b:
            from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy

            print(f"Loading guidance policy B: {args.policy_b}")
            policy_b, heads_b = load_policy(args.policy_b, args_b, device)

    output_tag = render_tag(args.model_a, args.lora_a, model_b_path, args.lora_b)
    out_dir = Path(args.output_dir) / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_render_meta(
        out_dir,
        tool="viz_det_compare",
        model_a_path=args.model_a,
        model_a_label=path_label(args.model_a),
        lora_a_path=args.lora_a or "",
        lora_a_label=path_label(args.lora_a),
        model_b_path=model_b_path or "",
        model_b_label=path_label(model_b_path),
        lora_b_path=args.lora_b or "",
        lora_b_label=path_label(args.lora_b),
        policy_a_path=args.policy_a or "",
        policy_a_label=path_label(args.policy_a),
        policy_b_path=args.policy_b or "",
        policy_b_label=path_label(args.policy_b),
    )

    results_a, results_b = [], []
    cached_trajs_a, cached_trajs_b, cached_gt = [], [], []

    for start in range(0, len(scene_paths), args.batch_size):
        batch_paths = scene_paths[start : start + args.batch_size]
        datas, valid_paths = [], []

        for sp in batch_paths:
            try:
                with np.load(sp, allow_pickle=True) as npz:
                    raw = set(npz.keys())
                if "ego_shape" not in raw:
                    print(f"  [skip] {Path(sp).name}: missing ego_shape")
                    continue
                d = load_npz_data(sp, device)
                es = d["ego_shape"].cpu().numpy().reshape(-1)[:3]
                if ego_shape is not None and not np.allclose(es, ego_shape, atol=1e-2):
                    print(f"  [skip] {Path(sp).name}: shape mismatch")
                    continue
                datas.append(d)
                valid_paths.append(sp)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {Path(sp).name}: {e}")

        if not datas:
            continue

        det_a = _plans(model_a, args_a, policy_a, heads_a, datas, device, args)
        det_b = (
            _plans(model_b, args_b, policy_b, heads_b, datas, device, args)
            if model_b is not None
            else None
        )

        for bi, sp in enumerate(valid_paths):
            name = Path(sp).stem

            traj_a_np = det_a[bi].cpu().numpy()
            traj_b_np = det_b[bi].cpu().numpy() if det_b is not None else None
            cached_trajs_a.append(traj_a_np)
            cached_trajs_b.append(traj_b_np)

            si = len(results_a)
            if rcfg is not None:
                r_a = compute_reward_batch(det_a[bi : bi + 1], datas[bi], rcfg)[0]
                d_a = reward_breakdown_to_det_dict(r_a)
                results_a.append({"scene": name, "scene_path": str(sp), **d_a})
                sc_a = d_a["det_sc_min_dist"]
                flag_a = "COL" if d_a["det_static_crossing"] else "   "
                if det_b is not None:
                    r_b = compute_reward_batch(det_b[bi : bi + 1], datas[bi], rcfg)[0]
                    d_b = reward_breakdown_to_det_dict(r_b)
                    results_b.append({"scene": name, "scene_path": str(sp), **d_b})
                    sc_b = d_b["det_sc_min_dist"]
                    flag_b = "COL" if d_b["det_static_crossing"] else "   "
                    print(
                        f"  [{si:3d}] {name:30s}  A: {flag_a} sc={sc_a:+.2f}m  "
                        f"B: {flag_b} sc={sc_b:+.2f}m"
                    )
                else:
                    print(f"  [{si:3d}] {name:30s}  A: {flag_a} sc={sc_a:+.2f}m")
            else:
                results_a.append({"scene": name, "scene_path": str(sp)})
                if det_b is not None:
                    results_b.append({"scene": name, "scene_path": str(sp)})
                print(f"  [{si:3d}] {name:30s}  rendered")

            gt_np = None
            if args.show_gt:
                _gt = np.load(sp, allow_pickle=True)["ego_agent_future"]
                gt_np = _gt[:, :4] if _gt.shape[-1] >= 4 else _gt
            cached_gt.append(gt_np)

            if args.no_viz:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            draw_scene_base(ax, sp, draw_stopped_neighbors=False)
            _draw_current_neighbors(ax, sp)
            if gt_np is not None:
                draw_traj(ax, gt_np, "GT (hand-drawn)", "#2ca02c", sp)
            label_a = args.label_a
            label_b = args.label_b
            if rcfg is not None:
                label_a = f"{args.label_a} (sc={sc_a:.2f}m)"
                if det_b is not None:
                    label_b = f"{args.label_b} (sc={sc_b:.2f}m)"
            draw_traj(ax, traj_a_np, label_a, "#1f77b4", sp)
            if traj_b_np is not None:
                draw_traj(ax, traj_b_np, label_b, "#d62728", sp)

            all_trajs = [traj_a_np[:, :2], [[0, 0]]]
            if traj_b_np is not None:
                all_trajs.append(traj_b_np[:, :2])
            all_pts = np.vstack(all_trajs)
            cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
            half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 6
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal")
            ax.legend(fontsize=9, loc="upper left")
            ax.set_title(f"{name}", fontsize=10)

            fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    # --- Aggregate stats ---
    def _to_score_list(results):
        return [
            {
                "sc_min_dist": r["det_sc_min_dist"],
                "rb_min_dist": r["det_rb_min_dist"],
                "cl": r["det_cl"],
                "total": r["det_total"],
                "static_crossing": r["det_static_crossing"],
                "rb_cross": r["det_rb_cross"],
                "lane_cross": r["det_lane_cross"],
                "kin_violated": r["det_kin_violated"],
                "sc_n_stopped": r.get("det_sc_n_stopped", 0),
            }
            for r in results
        ]

    if not results_a:
        raise SystemExit(
            f"All {len(scene_paths)} scenes were skipped (ego_shape mismatch "
            f"or missing). Check --ego_shape matches the NPZs."
        )

    agg_a = agg_b = None
    if rcfg is not None:
        agg_a = aggregate_stats(_to_score_list(results_a))
        print(f"\n--- {args.label_a} ---")
        print_summary(agg_a)
        if results_b:
            agg_b = aggregate_stats(_to_score_list(results_b))
            print(f"--- {args.label_b} ---")
            print_summary(agg_b)

    # --- Collage ---
    if not args.no_viz and results_a:
        print("Creating collage...")
        scored_paths = [r["scene_path"] for r in results_a]
        n = len(scored_paths)
        cols = args.collage_cols
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[None, :]
        elif cols == 1:
            axes = axes[:, None]
        axes_flat = axes.flatten()

        for si, sp in enumerate(scored_paths):
            if si >= len(axes_flat):
                break
            ax = axes_flat[si]

            draw_scene_base(ax, sp, draw_stopped_neighbors=False)
            _draw_current_neighbors(ax, sp)
            if si < len(cached_gt) and cached_gt[si] is not None:
                draw_traj(ax, cached_gt[si], "GT", "#2ca02c", sp)
            draw_traj(ax, cached_trajs_a[si], args.label_a, "#1f77b4", sp)
            if cached_trajs_b[si] is not None:
                draw_traj(ax, cached_trajs_b[si], args.label_b, "#d62728", sp)

            t_a, t_b = cached_trajs_a[si], cached_trajs_b[si]
            all_trajs = [t_a[:, :2], [[0, 0]]]
            if t_b is not None:
                all_trajs.append(t_b[:, :2])
            all_pts = np.vstack(all_trajs)
            cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
            half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 6
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_title(f"[{si}] {results_a[si]['scene']}", fontsize=8)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / "collage.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved collage: {out_dir / 'collage.png'}")

    # --- Save JSON ---
    out_path = out_dir / "det_compare_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_a": args.model_a,
                "lora_a": args.lora_a,
                "label_a": args.label_a,
                "model_b": args.model_b,
                "lora_b": args.lora_b,
                "label_b": args.label_b,
                "aggregate_a": agg_a,
                "aggregate_b": agg_b,
                "scenes_a": results_a,
                "scenes_b": results_b,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
