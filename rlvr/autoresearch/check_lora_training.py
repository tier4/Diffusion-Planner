#!/usr/bin/env python3
"""Check if LoRA weights changed after training and if deterministic output differs.

Usage:
    python3 rlvr/scripts/check_lora_training.py <experiment_dir> [--scene <npz_path>]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data as _load_raw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_with_lora(base_model_path, lora_path=None):
    args_file = str(os.path.dirname(base_model_path)) + "/args.json"
    args = Config(args_file)
    model = Diffusion_Planner(args)
    ckpt = torch.load(base_model_path, map_location=DEVICE)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    if lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint

        model = load_lora_checkpoint(model, lora_path)
        model.eval()

    return model, args


def run_deterministic(model, args, npz_path):
    data = _load_raw(npz_path, DEVICE)
    data["delay"] = torch.zeros(1, dtype=torch.long, device=DEVICE)
    norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    norm = args.observation_normalizer(norm)
    traj = generate_samples(model, args, norm, 0.0, 1, None, DEVICE)[0]
    return traj


def check_lora_weights(exp_dir):
    """Check if LoRA weights exist and differ from zero."""
    lora_dirs = sorted(glob.glob(f"{exp_dir}/lora_*"))
    if not lora_dirs:
        print("No LoRA checkpoints found yet.")
        return

    for lora_dir in lora_dirs:
        adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(lora_dir, "adapter_model.bin")
        if not os.path.exists(adapter_path):
            print(f"  {os.path.basename(lora_dir)}: no adapter file found")
            continue

        if adapter_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            weights = load_file(adapter_path)
        else:
            weights = torch.load(adapter_path, map_location="cpu")

        total_norm = 0
        n_params = 0
        for k, v in weights.items():
            total_norm += v.float().norm().item() ** 2
            n_params += v.numel()
        total_norm = total_norm**0.5

        print(f"  {os.path.basename(lora_dir)}: {n_params} params, norm={total_norm:.6f}")
        if total_norm < 1e-6:
            print("    WARNING: LoRA weights are all zero — training had no effect!")


def compare_trajectories(base_model_path, exp_dir, npz_path):
    """Compare deterministic output before and after LoRA training."""
    print(f"\nComparing deterministic trajectory on: {os.path.basename(npz_path)}")

    # Baseline
    model_base, args = load_model_with_lora(base_model_path)
    traj_base = run_deterministic(model_base, args, npz_path)
    pl_base = np.linalg.norm(np.diff(traj_base[:, :2], axis=0), axis=1).sum()
    del model_base
    torch.cuda.empty_cache()

    # After each LoRA epoch
    lora_dirs = sorted(glob.glob(f"{exp_dir}/lora_*"))
    for lora_dir in lora_dirs:
        try:
            model_lora, args = load_model_with_lora(base_model_path, lora_dir)
            traj_lora = run_deterministic(model_lora, args, npz_path)
            pl_lora = np.linalg.norm(np.diff(traj_lora[:, :2], axis=0), axis=1).sum()

            diff = np.linalg.norm(traj_base[:, :2] - traj_lora[:, :2], axis=1)
            max_diff = diff.max()
            mean_diff = diff.mean()

            print(
                f"  {os.path.basename(lora_dir)}: path={pl_lora:.1f}m (base={pl_base:.1f}m), "
                f"max_diff={max_diff:.3f}m, mean_diff={mean_diff:.3f}m"
            )
            if max_diff < 0.01:
                print("    WARNING: trajectory barely changed — LoRA may not be training!")

            del model_lora
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {os.path.basename(lora_dir)}: ERROR {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Experiment directory")
    parser.add_argument("--scene", default=None, help="NPZ path for trajectory comparison")
    parser.add_argument("--base-model", required=True, help="Path to base model .pth")
    args = parser.parse_args()

    print(f"Experiment: {args.exp_dir}")
    print("\n--- LoRA Weight Check ---")
    check_lora_weights(args.exp_dir)

    if args.scene:
        compare_trajectories(args.base_model, args.exp_dir, args.scene)
    else:
        print("No --scene provided. Skipping trajectory comparison.")
        print("Provide --scene <npz_path> to compare deterministic trajectories.")


if __name__ == "__main__":
    main()
