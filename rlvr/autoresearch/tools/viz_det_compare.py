#!/usr/bin/env python3
"""Compare deterministic trajectories from two models side-by-side.

Loads two models (each with optional LoRA), runs det inference on shared
scenes, scores both with a reward config, and renders per-scene PNGs +
a collage showing both trajectories on the same map.

Usage:
    python -m rlvr.autoresearch.tools.viz_det_compare \
        --model_a <base.pth> --label_a "Baseline" \
        --model_b <trained.pth> --label_b "a06" \
        [--lora_a <lora_dir>] [--lora_b <lora_dir>] \
        --scenes <scenes.json> \
        --config <reward_config.json> \
        --ego_shape WB,L,W \
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
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base, draw_traj
from rlvr.reward import compute_reward_batch


def _load_model_with_lora(model_path, lora_path, device):
    model, model_args = load_model(model_path, device)
    if lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint

        model = load_lora_checkpoint(model, lora_path)
        model.eval()
    return model, model_args


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_a", required=True)
    p.add_argument("--model_b", required=True)
    p.add_argument("--lora_a", default=None)
    p.add_argument("--lora_b", default=None)
    p.add_argument("--label_a", default="Model A")
    p.add_argument("--label_b", default="Model B")
    p.add_argument("--scenes", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ego_shape", required=True, help="WB,L,W")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--collage_cols", type=int, default=5)
    p.add_argument("--no_viz", action="store_true",
                   help="Skip per-scene PNGs and collage; only print stats.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = np.array([float(x) for x in args.ego_shape.split(",")])
    rcfg = load_reward_config(args.config)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"Loading model A: {args.model_a}")
    model_a, args_a = _load_model_with_lora(args.model_a, args.lora_a, device)
    print(f"Loading model B: {args.model_b}")
    model_b, args_b = _load_model_with_lora(args.model_b, args.lora_b, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_a, results_b = [], []

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
                if not np.allclose(es, ego_shape, atol=1e-2):
                    print(f"  [skip] {Path(sp).name}: shape mismatch")
                    continue
                datas.append(d)
                valid_paths.append(sp)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {Path(sp).name}: {e}")

        if not datas:
            continue

        det_a = det_inference_batched(model_a, args_a, datas, device)
        det_b = det_inference_batched(model_b, args_b, datas, device)

        for bi, sp in enumerate(valid_paths):
            name = Path(sp).stem

            r_a = compute_reward_batch(det_a[bi : bi + 1], datas[bi], rcfg)[0]
            r_b = compute_reward_batch(det_b[bi : bi + 1], datas[bi], rcfg)[0]

            d_a = reward_breakdown_to_det_dict(r_a)
            d_b = reward_breakdown_to_det_dict(r_b)

            results_a.append({"scene": name, "scene_path": str(sp), **d_a})
            results_b.append({"scene": name, "scene_path": str(sp), **d_b})

            sc_a = d_a["det_sc_min_dist"]
            sc_b = d_b["det_sc_min_dist"]
            flag_a = "COL" if d_a["det_static_crossing"] else "   "
            flag_b = "COL" if d_b["det_static_crossing"] else "   "
            si = len(results_a) - 1
            print(
                f"  [{si:3d}] {name:30s}  "
                f"A: {flag_a} sc={sc_a:+.2f}m  "
                f"B: {flag_b} sc={sc_b:+.2f}m"
            )

            if args.no_viz:
                continue

            traj_a_np = det_a[bi].cpu().numpy()
            traj_b_np = det_b[bi].cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            draw_scene_base(ax, sp)
            draw_traj(ax, traj_a_np,
                      f"{args.label_a} (sc={sc_a:.2f}m)", "#1f77b4", sp)
            draw_traj(ax, traj_b_np,
                      f"{args.label_b} (sc={sc_b:.2f}m)", "#d62728", sp)

            all_pts = np.vstack([traj_a_np[:, :2], traj_b_np[:, :2], [[0, 0]]])
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

    agg_a = aggregate_stats(_to_score_list(results_a))
    agg_b = aggregate_stats(_to_score_list(results_b))

    print(f"\n--- {args.label_a} ---")
    print_summary(agg_a)
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
            d = load_npz_data(sp, device)
            t_a = det_inference_batched(model_a, args_a, [d], device)[0].cpu().numpy()
            t_b = det_inference_batched(model_b, args_b, [d], device)[0].cpu().numpy()

            draw_scene_base(ax, sp)
            draw_traj(ax, t_a, args.label_a, "#1f77b4", sp)
            draw_traj(ax, t_b, args.label_b, "#d62728", sp)

            all_pts = np.vstack([t_a[:, :2], t_b[:, :2], [[0, 0]]])
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
        json.dump({
            "model_a": args.model_a, "lora_a": args.lora_a, "label_a": args.label_a,
            "model_b": args.model_b, "lora_b": args.lora_b, "label_b": args.label_b,
            "aggregate_a": agg_a, "aggregate_b": agg_b,
            "scenes_a": results_a, "scenes_b": results_b,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
