"""Integration test: exploration policy + real v4 model + real NPZ scenes.

Loads the v4 base model, creates an ExplorationPolicy, runs it to get
(eta_lat, eta_lon), then generates guided trajectories and compares them
to the deterministic (unguided) output.

Usage:
    python exploration_policy/test_integration.py \
        --model_path <path/to/model.pth> \
        --npz_list <path/to/scenes.json>

    Or via environment variables:
    EXPLORATION_POLICY_MODEL_PATH=<model.pth> \
    EXPLORATION_POLICY_NPZ_LIST=<scenes.json> \
    python exploration_policy/test_integration.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

# Default paths via environment variables (avoids hardcoding machine-specific paths).
_ENV_MODEL = os.environ.get("EXPLORATION_POLICY_MODEL_PATH")
_ENV_NPZ = os.environ.get("EXPLORATION_POLICY_NPZ_LIST")
DEFAULT_MODEL = Path(_ENV_MODEL) if _ENV_MODEL else None
DEFAULT_NPZ_LIST = Path(_ENV_NPZ) if _ENV_NPZ else None


def main():
    parser = argparse.ArgumentParser(description="Exploration policy integration test")
    parser.add_argument(
        "--model_path", type=Path, default=DEFAULT_MODEL,
        help="Path to model .pth (or set EXPLORATION_POLICY_MODEL_PATH env var)",
    )
    parser.add_argument(
        "--npz_list", type=Path, default=DEFAULT_NPZ_LIST,
        help="Path to JSON scene list (or set EXPLORATION_POLICY_NPZ_LIST env var)",
    )
    parser.add_argument("--n_scenes", type=int, default=3, help="Number of scenes to test")
    parser.add_argument("--n_trajectories", type=int, default=4, help="Policy-guided trajectories per scene")
    args = parser.parse_args()

    if args.model_path is None or args.npz_list is None:
        parser.error(
            "Both --model_path and --npz_list are required unless provided via "
            "EXPLORATION_POLICY_MODEL_PATH and EXPLORATION_POLICY_NPZ_LIST env vars."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"\nLoading model from {args.model_path}...")
    from preference_optimization.model_utils import load_model
    model, model_args = load_model(args.model_path, device)
    model.eval()
    print(f"  Model loaded. hidden_dim={model_args.hidden_dim}, future_len={model_args.future_len}")

    # --- Load scenes ---
    with open(args.npz_list) as f:
        npz_paths = json.load(f)
    npz_paths = [p for p in npz_paths[:args.n_scenes] if Path(p).exists()]
    print(f"  Testing on {len(npz_paths)} scenes")

    # --- Create exploration policy ---
    from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
    from exploration_policy.utils import run_frozen_encoder, generate_reference_trajectory

    ep_config = ExplorationPolicyConfig(
        hidden_dim=128,
        n_mixer_layers=2,
        n_attn_heads=4,
        dropout=0.0,
        encoder_hidden_dim=model_args.hidden_dim,
    )
    policy = ExplorationPolicy(ep_config, ref_seq_len=model_args.future_len).to(device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Exploration policy: {n_params:,} params")

    # --- Verify zero-init ---
    with torch.no_grad():
        zero_fused = torch.zeros(1, ep_config.hidden_dim, device=device)
        lat_dist, lon_dist = policy.guidance_head(zero_fused)
        print(f"  Zero-init check: alpha_lat={lat_dist.concentration1.item():.4f}, "
              f"beta_lat={lat_dist.concentration0.item():.4f} "
              f"(mean_eta={2*lat_dist.mean.item()-1:.6f})")

    # --- Run per scene ---
    from preference_optimization.utils import load_npz_data
    from guidance_gui.generate_samples import generate_samples
    from diffusion_planner.model.guidance.composer import GuidanceComposer
    from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig

    for scene_idx, npz_path in enumerate(npz_paths):
        print(f"\n{'='*60}")
        print(f"Scene {scene_idx}: {Path(npz_path).stem}")

        # Load data
        data = load_npz_data(npz_path, device)
        if "delay" not in data:
            data["delay"] = torch.zeros(1, dtype=torch.long, device=device)

        # Normalize for model
        norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        norm_data = model_args.observation_normalizer(norm_data)

        # 1. Deterministic trajectory (no guidance)
        with torch.no_grad():
            det_traj = generate_samples(
                model, model_args, norm_data,
                noise_scale=0.0, n_samples=1, composer=None, device=device,
            )[0]  # (T, 4)
        print(f"  Deterministic: endpoint=({det_traj[-1,0]:.2f}, {det_traj[-1,1]:.2f})")

        # 2. Generate reference trajectory (LoRA-disabled, same as det for base model)
        with torch.no_grad():
            x_ref = generate_reference_trajectory(model, model_args, norm_data, device)
        x_ref_t = torch.from_numpy(x_ref).unsqueeze(0).to(device)  # [1, T, 4]
        print(f"  Reference:     endpoint=({x_ref[-1,0]:.2f}, {x_ref[-1,1]:.2f})")

        # 3. Get frozen encoder output
        with torch.no_grad():
            scene_encoding = run_frozen_encoder(model, norm_data)  # [1, N, D]
        print(f"  Scene encoding: shape={list(scene_encoding.shape)}")

        # 4. Run exploration policy to get eta values
        # Store reference trajectory for guidance functions
        norm_data["reference_trajectory"] = x_ref_t

        guided_trajs = []
        for k in range(args.n_trajectories):
            with torch.no_grad():
                policy_output = policy(scene_encoding, x_ref_t, deterministic=(k == 0))

            eta_lat = policy_output.eta_lat[0].item()
            eta_lon = policy_output.eta_lon[0].item()
            value = policy_output.value[0].item()

            # Build lateral+longitudinal guidance with policy's eta
            guidance_fns = [
                GuidanceConfig(
                    name="lateral", enabled=True, scale=1.0,
                    params={"lambda_lat": 2.5, "eta_lat": eta_lat},
                ),
                GuidanceConfig(
                    name="longitudinal", enabled=True, scale=1.0,
                    params={"lambda_lon": 0.25, "eta_lon": eta_lon},
                ),
            ]
            set_cfg = GuidanceSetConfig(functions=guidance_fns, global_scale=0.5)
            composer = GuidanceComposer(set_cfg)

            with torch.no_grad():
                guided = generate_samples(
                    model, model_args, norm_data,
                    noise_scale=0.5, n_samples=1, composer=composer, device=device,
                )[0]  # (T, 4)

            # Compute displacement from deterministic
            disp = np.linalg.norm(guided[:, :2] - det_traj[:, :2], axis=-1)
            ade = disp.mean()
            fde = disp[-1]

            mode = "det_policy" if k == 0 else f"sample_{k}"
            print(f"  [{mode}] η_lat={eta_lat:+.3f}, η_lon={eta_lon:+.3f}, "
                  f"V={value:+.3f}, ADE={ade:.2f}m, FDE={fde:.2f}m, "
                  f"endpoint=({guided[-1,0]:.2f}, {guided[-1,1]:.2f})")
            guided_trajs.append(guided)

        # 5. Verify trajectories are actually different
        if len(guided_trajs) >= 2:
            pairwise_diffs = []
            for i in range(len(guided_trajs)):
                for j in range(i+1, len(guided_trajs)):
                    d = np.linalg.norm(
                        guided_trajs[i][:, :2] - guided_trajs[j][:, :2], axis=-1
                    ).mean()
                    pairwise_diffs.append(d)
            mean_diversity = np.mean(pairwise_diffs)
            print(f"  Trajectory diversity (mean pairwise ADE): {mean_diversity:.2f}m")

    print(f"\n{'='*60}")
    print("Integration test complete!")


if __name__ == "__main__":
    main()
