"""Sweep LoRA checkpoints: evaluate LD, path, stopped, lane/border metrics, and L2 loss.

Supports block ablation (no_block0, no_block1, no_block2) for each epoch.

Usage:
    python -m rlvr.autoresearch.tools.eval_checkpoint_sweep \
        --model_path /path/to/base.pth \
        --lora_dir /path/to/experiment_dir \
        --scenes /path/to/val_scenes.json \
        --epochs 5 7 9 10 12 \
        --block_ablations 0 1 \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_block_ablation(lora_dir: Path, block: int) -> Path:
    """Zero out a specific block's LoRA weights, return new lora dir."""
    out = lora_dir.parent / f"{lora_dir.name}_no_block{block}"
    out.mkdir(exist_ok=True)
    state = load_file(str(lora_dir / "adapter_model.safetensors"))
    filtered = {
        k: (torch.zeros_like(v) if f"blocks.{block}." in k else v)
        for k, v in state.items()
    }
    save_file(filtered, str(out / "adapter_model.safetensors"))
    shutil.copy2(lora_dir / "adapter_config.json", out / "adapter_config.json")
    return out


def eval_checkpoint(model_path: str, lora_path: str, scenes: list[str]) -> dict:
    """Evaluate a single checkpoint. Returns metrics dict."""
    from rlvr.autoresearch.tools.eval_lane_border_distance import load_model
    from rlvr.grpo_sampler import generate_samples
    from rlvr.grpo_trainer import load_npz_data
    from rlvr.reward import RewardConfig, compute_reward_batch

    model, args = load_model(model_path, lora_path)
    config = RewardConfig(enable_lane_departure=True)

    ld, stopped = 0, 0
    paths, ln_nears, ln_wides, rb_nears = [], [], [], []

    for p in scenes:
        data = load_npz_data(p, DEVICE)
        norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        norm = args.observation_normalizer(norm)
        traj = generate_samples(model, args, norm, 0.0, 1, None, DEVICE)[0]
        traj_t = torch.tensor(traj[None], device=DEVICE, dtype=torch.float32)
        r = compute_reward_batch(traj_t, data, config)[0]

        if r.lane_crossing:
            ld += 1
        pl = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
        paths.append(pl)
        # Stopped: path < 2m AND GT path > 5m
        gt = np.load(p, allow_pickle=True)["ego_agent_future"]
        gt_path = float(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum())
        if pl < 2.0 and gt_path > 5.0:
            stopped += 1
        ln_nears.append(r.lane_near_frac)
        ln_wides.append(r.lane_wide_frac)
        rb_nears.append(r.rb_near_frac)

    del model
    torch.cuda.empty_cache()

    n = len(scenes)
    return {
        "ld": ld,
        "n": n,
        "stopped": stopped,
        "path_mean": float(np.mean(paths)),
        "path_median": float(np.median(paths)),
        "paths_lt5m": sum(1 for p in paths if p < 5),
        "ln_near_mean": float(np.mean(ln_nears)),
        "ln_near_p95": float(np.percentile(ln_nears, 95)),
        "ln_wide_mean": float(np.mean(ln_wides)),
        "ln_wide_p95": float(np.percentile(ln_wides, 95)),
        "rb_near_mean": float(np.mean(rb_nears)),
        "rb_near_p95": float(np.percentile(rb_nears, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep LoRA checkpoints with block ablation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True, help="Experiment dir containing lora_epoch_NNN/")
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True, help="Epochs to evaluate")
    parser.add_argument("--block_ablations", type=int, nargs="*", default=[], help="Block indices to ablate (e.g. 0 1)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.scenes) as f:
        scenes = json.load(f)

    lora_base = Path(args.lora_dir)
    results = []

    for ep in args.epochs:
        lora_ep = lora_base / f"lora_epoch_{ep:03d}"
        if not lora_ep.exists():
            print(f"  Skipping ep{ep} (not found)")
            continue

        # Full model
        m = eval_checkpoint(args.model_path, str(lora_ep), scenes)
        m["epoch"] = ep
        m["variant"] = "full"
        results.append(m)
        print(
            f"ep{ep:>3d} full      | LD={m['ld']:>2}/{m['n']} | stop={m['stopped']:>2} "
            f"| path={m['path_mean']:>5.1f}m med={m['path_median']:>5.1f}m <5m={m['paths_lt5m']:>2} "
            f"| ln_w_p95={m['ln_wide_p95']:.4f} | rb_n_p95={m['rb_near_p95']:.4f}"
        )
        sys.stdout.flush()

        # Block ablations
        for block in args.block_ablations:
            abl_dir = create_block_ablation(lora_ep, block)
            m = eval_checkpoint(args.model_path, str(abl_dir), scenes)
            m["epoch"] = ep
            m["variant"] = f"no_block{block}"
            results.append(m)
            print(
                f"ep{ep:>3d} no_blk{block}  | LD={m['ld']:>2}/{m['n']} | stop={m['stopped']:>2} "
                f"| path={m['path_mean']:>5.1f}m med={m['path_median']:>5.1f}m <5m={m['paths_lt5m']:>2} "
                f"| ln_w_p95={m['ln_wide_p95']:.4f} | rb_n_p95={m['rb_near_p95']:.4f}"
            )
            sys.stdout.flush()

    # Save results
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {out / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
