"""Visualize exploration policy per-scene guidance outputs.

Loads a trained exploration policy checkpoint and runs deterministic inference
on each scene to show the learned eta_lat/eta_lon guidance parameters.

Usage:
    python -m rlvr.autoresearch.tools.viz_explorer_per_scene \
        --model_path /path/to/base_model.pth \
        --exp_dir /path/to/experiment_dir \
        --scenes /path/to/scenes.json \
        --epoch 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.grpo_config import GRPOConfig


def analyze_explorer(
    policy_model: torch.nn.Module,
    model_args,
    explorer: ExplorationPolicy,
    scene_paths: list[str],
    device: torch.device,
) -> dict:
    """Run explorer on all scenes and return per-scene eta outputs."""
    eta_lats, eta_lons, scene_names = [], [], []

    for path in scene_paths:
        try:
            data = load_npz_data(path, device)
            if "delay" not in data:
                data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
            norm_data = model_args.observation_normalizer(data)

            with torch.no_grad():
                x_ref_np = generate_reference_trajectory(
                    policy_model, model_args, norm_data, device,
                )
                x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device)
                scene_enc = run_frozen_encoder(policy_model, norm_data)
                output = explorer(scene_enc, x_ref, deterministic=True)

            eta_lats.append(output.eta_lat.item())
            eta_lons.append(output.eta_lon.item())
            scene_names.append(Path(path).stem)
        except Exception as e:
            print(f"  Skip {Path(path).stem}: {e}")

    return {
        "eta_lat": np.array(eta_lats),
        "eta_lon": np.array(eta_lons),
        "scenes": scene_names,
    }


def load_explorer(
    exp_dir: Path, epoch: int, model_args, device: torch.device,
) -> ExplorationPolicy | None:
    """Load exploration policy from experiment checkpoint."""
    config_path = exp_dir / "grpo_config.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return None
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = GRPOConfig()
    for k, v in cfg_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # Find checkpoint
    ckpt_dir = exp_dir / f"lora_epoch_{epoch:03d}"
    if not ckpt_dir.exists():
        for e in range(epoch, 0, -1):
            ckpt_dir = exp_dir / f"lora_epoch_{e:03d}"
            if ckpt_dir.exists():
                epoch = e
                break
        else:
            print(f"No checkpoint dirs found in {exp_dir}")
            return None

    policy_path = ckpt_dir / "exploration_policy.pth"
    if not policy_path.exists():
        print(f"No exploration_policy.pth in {ckpt_dir}")
        return None

    ep_config = ExplorationPolicyConfig(
        hidden_dim=config.exploration_hidden_dim,
        n_mixer_layers=config.exploration_n_mixer_layers,
        n_attn_heads=config.exploration_n_attn_heads,
        dropout=config.exploration_dropout,
        learning_rate=config.exploration_lr,
        encoder_hidden_dim=model_args.hidden_dim,
        head_init=config.exploration_head_init,
        head_raw_scale=config.exploration_head_raw_scale,
    )
    explorer = ExplorationPolicy(ep_config, ref_seq_len=model_args.future_len).to(device)
    state = torch.load(policy_path, map_location=device, weights_only=False)
    explorer.load_state_dict(state, strict=False)
    explorer.eval()
    print(f"Loaded explorer from {ckpt_dir} (epoch {epoch})")
    return explorer


def main():
    parser = argparse.ArgumentParser(description="Visualize exploration policy per-scene outputs")
    parser.add_argument("--model_path", type=Path, required=True, help="Base model .pth")
    parser.add_argument("--exp_dir", type=Path, required=True, help="Experiment directory (contains grpo_config.json)")
    parser.add_argument("--scenes", type=Path, required=True, help="JSON list of scene NPZ paths")
    parser.add_argument("--epoch", type=int, default=10, help="Checkpoint epoch to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model, model_args = load_model(args.model_path, device)
    policy_model.eval()

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    explorer = load_explorer(args.exp_dir, args.epoch, model_args, device)
    if explorer is None:
        return

    result = analyze_explorer(policy_model, model_args, explorer, scene_paths, device)
    lat = result["eta_lat"]
    lon = result["eta_lon"]
    n = len(lat)

    if n == 0:
        print("No scenes processed successfully.")
        return

    print(f"\n{'='*70}")
    print(f"Exploration Policy Per-Scene Analysis ({n} scenes)")
    print(f"{'='*70}")
    print(f"  eta_lat: mean={lat.mean():.4f}, std={lat.std():.4f}, min={lat.min():.4f}, max={lat.max():.4f}")
    print(f"  eta_lon: mean={lon.mean():.4f}, std={lon.std():.4f}, min={lon.min():.4f}, max={lon.max():.4f}")
    print(f"  Scene-to-scene lat range: {lat.max() - lat.min():.4f}")
    print(f"  Scene-to-scene lon range: {lon.max() - lon.min():.4f}")

    print(f"\n  {'Scene':<50} {'eta_lat':>8} {'eta_lon':>8}")
    print(f"  {'-'*66}")
    for i in range(n):
        print(f"  {result['scenes'][i]:<50} {lat[i]:>+8.4f} {lon[i]:>+8.4f}")

    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    print(f"\n  Extremes:")
    print(f"  Most LEFT:   {result['scenes'][lat_order[-1]]:<50} lat={lat[lat_order[-1]]:+.4f}")
    print(f"  Most RIGHT:  {result['scenes'][lat_order[0]]:<50} lat={lat[lat_order[0]]:+.4f}")
    print(f"  Fastest:     {result['scenes'][lon_order[-1]]:<50} lon={lon[lon_order[-1]]:+.4f}")
    print(f"  Slowest:     {result['scenes'][lon_order[0]]:<50} lon={lon[lon_order[0]]:+.4f}")


if __name__ == "__main__":
    main()
