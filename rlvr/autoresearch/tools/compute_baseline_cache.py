"""Precompute baseline and GT path lengths for a scene list.

Saves a JSON file mapping scene_path -> {"baseline_path": float, "gt_path": float}.
This file is used by run_experiment.py --baseline_cache to compute progress ratios
without re-running the base model every experiment.

Usage:
    python -m rlvr.autoresearch.tools.compute_baseline_cache \
        --model_path /path/to/base_model.pth \
        --scenes /path/to/scenes.json \
        --output /path/to/baseline_cache.json
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from guidance_gui.generate_samples import generate_samples
from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data


def main():
    parser = argparse.ArgumentParser(
        description="Precompute baseline/GT paths for progress ratio metrics"
    )
    parser.add_argument("--model_path", type=Path, required=True, help="Path to base model .pth")
    parser.add_argument("--scenes", type=Path, required=True, help="JSON list of scene NPZ paths")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON cache file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args = load_model(args.model_path, device)
    model.eval()

    with open(args.scenes) as f:
        scenes = json.load(f)

    cache = {}
    for i, path in enumerate(scenes):
        data = load_npz_data(path, device)
        norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        if "delay" not in norm:
            norm["delay"] = torch.zeros(1, dtype=torch.long, device=device)
        normalizer = copy.deepcopy(model_args.observation_normalizer)
        with torch.no_grad():
            norm = normalizer(norm)
            traj = generate_samples(model, model_args, norm, 0.0, 1, None, device)
        baseline_pl = float(np.linalg.norm(np.diff(traj[0, :, :2], axis=0), axis=1).sum())

        with np.load(path) as raw:
            gt = raw["ego_agent_future"]
            gt_pl = float(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum())

        cache[path] = {"baseline_path": baseline_pl, "gt_path": gt_pl}

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(scenes)} scenes processed")

    with open(args.output, "w") as f:
        json.dump(cache, f, indent=2)

    bpls = [v["baseline_path"] for v in cache.values()]
    gpls = [v["gt_path"] for v in cache.values()]
    ratios = [b / max(g, 1e-3) for b, g in zip(bpls, gpls)]
    print(f"\nSaved {len(cache)} scenes to {args.output}")
    print(f"Baseline: mean_path={np.mean(bpls):.1f}m, mean_gt={np.mean(gpls):.1f}m")
    print(
        f"Progress ratio (base/GT): p5={np.percentile(ratios, 5):.2f} p25={np.percentile(ratios, 25):.2f} med={np.median(ratios):.2f}"
    )


if __name__ == "__main__":
    main()
