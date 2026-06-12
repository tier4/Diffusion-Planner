"""Detailed centerline + road-border metrics for a list of LoRA checkpoints.

Evaluates each LoRA (loaded onto a shared base model) on a val scene set and
prints per-scene distribution statistics for centerline (cl_mean) and
road-border distance (rb_dist_min), plus threshold-bucket counts. Uses the
same reward code path as training (compute_reward_batch), so numbers are
directly comparable to per-epoch training logs.

This complements eval_checkpoint_sweep.py, which sweeps multiple epochs of
one run; this tool compares multiple runs side-by-side with richer stats.

With --dump_json, the output JSON includes a `per_scene` dict of parallel
arrays (cl_mean, rb_dist_min, rb_near_frac, rb_wide_frac, lane_near_frac,
lane_wide_frac, path, rb_crossing, lane_crossing) so downstream tools can
rank scenes by worst cl / tightest rb without re-running the eval.

Usage:
    python -m rlvr.autoresearch.tools.eval_detailed_metrics \
        --model_path /path/to/base.pth \
        --scenes /path/to/val.json \
        --config /path/to/grpo_config.json \
        --loras run1/lora_epoch_009 run2/lora_epoch_005 \
        --labels run1_ep9 run2_ep5 \
        --dump_json /path/to/out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.model_utils import load_model
from rlvr.autoresearch.run_experiment import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.closed_loop.batched_rollout import _batched_generate
from rlvr.grpo_config import GRPOConfig
from rlvr.reward import compute_reward_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_RB_BUCKETS = (0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20)
_CL_BUCKETS = (0.10, 0.15, 0.20, 0.30, 0.50, 1.00)
_PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _summary(arr, name):
    a = np.asarray(arr, dtype=np.float64)
    qs = np.percentile(a, _PCTS)
    q_str = " ".join(f"p{p}={q:+.3f}" for p, q in zip(_PCTS, qs))
    return (f"{name}: n={a.size} mean={a.mean():+.3f} "
            f"min={a.min():+.3f} max={a.max():+.3f} | {q_str}")


def _threshold_buckets(arr, thresholds, comparison="lt", label_fmt="< {thr:.2f}"):
    """Print count of elements below/above each threshold."""
    a = np.asarray(arr)
    for thr in thresholds:
        n = int((a < thr).sum()) if comparison == "lt" else int((a > thr).sum())
        if n == 0 and comparison == "lt":
            continue
        print(f"    {label_fmt.format(thr=thr)}: {n} scenes ({100 * n / a.size:.1f}%)")


def eval_one(lora_path, scenes, model_args, reward_cfg, base_model, batch_size=150):
    lora_model = load_lora_checkpoint(base_model, Path(lora_path), is_trainable=False)
    lora_model = lora_model.to(DEVICE).eval()

    cl_per_scene, rb_min_per_scene = [], []
    rb_near_per_scene, rb_wide_per_scene = [], []
    lane_near_per_scene, lane_wide_per_scene = [], []
    path_per_scene = []
    rb_cross_per_scene, lane_cross_per_scene = [], []

    all_data = []
    for sp in scenes:
        try:
            all_data.append(load_npz_data(sp, DEVICE))
        except Exception as e:
            print(f"  [skip] {Path(sp).name}: {e}")

    for start in range(0, len(all_data), batch_size):
        chunk = all_data[start:start + batch_size]
        batch = {}
        for k in chunk[0]:
            vals = [d[k] for d in chunk]
            batch[k] = torch.cat(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals[0]
        normalizer = copy.deepcopy(model_args.observation_normalizer)
        norm = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        norm = normalizer(norm)
        with torch.no_grad():
            det = _batched_generate(lora_model, model_args, norm,
                                    noise_scale=0.0, composer=None, device=DEVICE)
        for i in range(len(chunk)):
            r = compute_reward_batch(det[i:i+1], chunk[i], reward_cfg)[0]
            cl_per_scene.append(r.centerline)
            rb_min_per_scene.append(r.rb_min_dist)
            rb_near_per_scene.append(r.rb_near_penalty)
            rb_wide_per_scene.append(r.rb_wide_penalty)
            lane_near_per_scene.append(r.lane_near_frac)
            lane_wide_per_scene.append(r.lane_wide_frac)
            rb_cross_per_scene.append(int(bool(r.rb_crossing)))
            lane_cross_per_scene.append(int(bool(r.lane_crossing)))
            traj = det[i].cpu().numpy()
            path_per_scene.append(
                float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
            )

    n = len(cl_per_scene)
    rb_min_arr = np.asarray(rb_min_per_scene)
    cl_arr = np.asarray(cl_per_scene)

    print(f"\nN scenes: {n}")
    print(f"rb_cross: {sum(rb_cross_per_scene)}/{n} ({100*sum(rb_cross_per_scene)/n:.1f}%)")
    print(f"lane_dep: {sum(lane_cross_per_scene)}/{n} ({100*sum(lane_cross_per_scene)/n:.1f}%)")
    print(f"path_mean: {np.mean(path_per_scene):.2f}m")
    print(_summary(cl_per_scene, "cl (per-scene mean cl)"))
    print(_summary(rb_min_arr, "rb_dist_min (per-scene min to border)"))
    print(_summary(rb_near_per_scene, "rb_near_frac"))
    print(_summary(rb_wide_per_scene, "rb_wide_frac"))
    print(_summary(lane_near_per_scene, "lane_near_frac"))
    print(_summary(lane_wide_per_scene, "lane_wide_frac"))

    print("rb_dist_min threshold buckets:")
    _threshold_buckets(rb_min_arr, _RB_BUCKETS, "lt", "< {thr:.2f}m")
    print("cl magnitude buckets (|cl| > thr):")
    _threshold_buckets(np.abs(cl_arr), _CL_BUCKETS, "gt", "|cl| > {thr:.2f}")

    return {
        "n_scenes": n,
        "rb_cross": sum(rb_cross_per_scene),
        "lane_dep": sum(lane_cross_per_scene),
        "path_mean": float(np.mean(path_per_scene)),
        "cl_mean": float(np.mean(cl_arr)),
        "rb_dist_min_min": float(rb_min_arr.min()),
        "rb_dist_min_p5": float(np.percentile(rb_min_arr, 5)),
        "per_scene": {
            "cl_mean": [float(x) for x in cl_per_scene],
            "rb_dist_min": [float(x) for x in rb_min_per_scene],
            "rb_near_frac": [float(x) for x in rb_near_per_scene],
            "rb_wide_frac": [float(x) for x in rb_wide_per_scene],
            "lane_near_frac": [float(x) for x in lane_near_per_scene],
            "lane_wide_frac": [float(x) for x in lane_wide_per_scene],
            "path": [float(x) for x in path_per_scene],
            "rb_crossing": list(rb_cross_per_scene),
            "lane_crossing": list(lane_cross_per_scene),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=Path)
    ap.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    ap.add_argument("--config", required=True, type=Path,
                    help="Training GRPO config JSON (for reward thresholds)")
    ap.add_argument("--loras", nargs="+", required=True, help="LoRA dirs to evaluate")
    ap.add_argument("--labels", nargs="+", default=None, help="Labels (defaults to dir names)")
    ap.add_argument("--batch_size", type=int, default=150)
    ap.add_argument("--dump_json", type=Path, default=None,
                    help="Optional JSON dump of summary metrics per LoRA")
    args = ap.parse_args()

    reward_cfg = load_reward_config(str(args.config))
    reward_cfg.enable_lane_departure = True
    _ = GRPOConfig.from_json(str(args.config))  # validates config schema

    base_model, model_args = load_model(args.model_path, device=DEVICE)

    with open(args.scenes) as f:
        scenes = json.load(f)

    labels = args.labels or [
        Path(l).parent.name + "/" + Path(l).name for l in args.loras
    ]
    assert len(labels) == len(args.loras), "labels must match --loras count"

    results = {}
    for lora, label in zip(args.loras, labels):
        print(f"\n=========== {label} ===========")
        results[label] = eval_one(
            lora, scenes, model_args, reward_cfg, base_model,
            batch_size=args.batch_size,
        )

    if args.dump_json:
        args.dump_json.write_text(json.dumps(results, indent=2))
        print(f"\nDumped summary → {args.dump_json}")


if __name__ == "__main__":
    main()
