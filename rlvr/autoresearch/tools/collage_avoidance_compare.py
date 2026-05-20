#!/usr/bin/env python3
"""Collage comparing baseline vs trained model on static-collision avoidance.

Scores both models (DET) on each scene via compute_static_collision_penalty,
picks the best-improved scenes, and assembles a grid from pre-rendered
viz_prism_compare PNGs annotated with sc_min_dist for each model.

Usage:
    python -m rlvr.autoresearch.tools.collage_avoidance_compare \
        --baseline_model <base.pth> \
        --trained_model <merged.pth> \
        --scenes <scenes.json> \
        --config <reward_config.json> \
        --viz_dir <viz_compare_dir with per-scene PNGs> \
        --output <collage.png> \
        [--n 12] [--cols 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch

from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.autoresearch.tools.viz_collision_guidance import _score_traj
from rlvr.autoresearch.tools.viz_prism_compare import _load_model, _det_predict
from preference_optimization.utils import load_npz_data


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--trained_model", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--viz_dir", required=True,
                        help="Dir with per-scene PNGs from viz_prism_compare")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=12, help="Number of scenes in collage")
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = load_reward_config(args.config)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"Loading baseline model...")
    m_base, args_base = _load_model(args.baseline_model, None, device)
    print(f"Loading trained model...")
    m_train, args_train = _load_model(args.trained_model, None, device)

    results = []
    for i, sp in enumerate(scene_paths):
        scene_name = Path(sp).stem
        data = load_npz_data(sp, device)
        data_np = dict(np.load(sp, allow_pickle=True))

        traj_base = _det_predict(m_base, args_base, data)
        traj_train = _det_predict(m_train, args_train, data)

        traj_base_t = torch.from_numpy(traj_base).to(device)
        traj_train_t = torch.from_numpy(traj_train).to(device)

        sc_base = _score_traj(traj_base_t, data_np, rcfg, device)
        sc_train = _score_traj(traj_train_t, data_np, rcfg, device)

        delta = sc_train["sc_min_dist"] - sc_base["sc_min_dist"]

        png_candidates = list(Path(args.viz_dir).glob(f"*_{scene_name}.png"))
        if not png_candidates:
            png_candidates = list(Path(args.viz_dir).glob(f"scene_{i:03d}_*.png"))
        png_path = str(png_candidates[0]) if png_candidates else None

        results.append({
            "idx": i,
            "scene": scene_name,
            "base_sc_min": sc_base["sc_min_dist"],
            "train_sc_min": sc_train["sc_min_dist"],
            "base_zone": sc_base["zone"],
            "train_zone": sc_train["zone"],
            "base_crossing": sc_base["static_crossing"],
            "train_crossing": sc_train["static_crossing"],
            "delta": delta,
            "png": png_path,
        })
        zone_b = sc_base["zone"]
        zone_t = sc_train["zone"]
        tag = "FIXED" if sc_base["static_crossing"] and not sc_train["static_crossing"] else ""
        print(f"  [{i:2d}] {scene_name}: base={sc_base['sc_min_dist']:+.2f}m ({zone_b}) "
              f"train={sc_train['sc_min_dist']:+.2f}m ({zone_t}) "
              f"Δ={delta:+.2f}m {tag}")

    improved = [r for r in results if r["delta"] > 0.01 and r["png"]]
    improved.sort(key=lambda r: (-int(r["base_crossing"] and not r["train_crossing"]),
                                  -r["delta"]))

    selected = improved[:args.n]
    if not selected:
        print("No improved scenes found!")
        return

    n_fixed = sum(1 for r in selected if r["base_crossing"] and not r["train_crossing"])
    n_margin = len(selected) - n_fixed
    print(f"\nSelected {len(selected)} scenes: {n_fixed} collision-fixed, {n_margin} margin-improved")

    rows = (len(selected) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(7 * args.cols, 7 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for k, r in enumerate(selected):
        ax = axes_flat[k]
        img = mpimg.imread(r["png"])
        ax.imshow(img)
        ax.axis("off")

        base_d = r["base_sc_min"]
        train_d = r["train_sc_min"]
        delta = r["delta"]
        tag = ""
        if r["base_crossing"] and not r["train_crossing"]:
            tag = "  [COLLISION FIXED]"
        elif r["base_crossing"] and r["train_crossing"]:
            tag = "  [BOTH COLLIDE]"

        base_color = "#cc0000" if r["base_crossing"] else "#333333"
        train_color = "#006600" if not r["train_crossing"] else "#cc0000"

        title = (f"{r['scene']}\n"
                 f"baseline: {base_d:+.2f}m ({r['base_zone']})  →  "
                 f"trained: {train_d:+.2f}m ({r['train_zone']})  "
                 f"Δ={delta:+.2f}m{tag}")
        ax.set_title(title, fontsize=10, fontweight="bold" if tag else "normal")

    fig.suptitle(f"Static Obstacle Avoidance: Baseline vs Trained Model\n"
                 f"{n_fixed} collisions fixed, {n_margin} margins improved",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.3)
    print(f"\nCollage saved to {args.output}")

    summary_path = Path(args.output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {summary_path}")


if __name__ == "__main__":
    main()
