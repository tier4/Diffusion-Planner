#!/usr/bin/env python3
"""Visualize actual guided vs unguided trajectories using the real inference pipeline.

Runs the DiT twice per scene: once without guidance, once with the explorer's eta
applied through GuidanceComposer — exactly like training.

Usage:
    python -m rlvr.autoresearch.tools.viz_guidance_actual \
      --model_path /path/to/best_model.pth \
      --policy_path /path/to/lora_epoch_NNN/exploration_policy.pth \
      --scenes /path/to/scenes.json \
      --output /path/to/output.png \
      [--lora_path /path/to/lora_epoch_NNN] \
      [--indices 25 26 27 28 29 30] \
      [--lambda_lat 2.5] [--lambda_lon 0.25] [--guidance_scale 0.5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preference_optimization.utils import load_npz_data
from preference_optimization.lora_utils import load_lora_checkpoint
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from exploration_policy import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import get_frozen_encoder, generate_reference_trajectory
from guidance_gui.generate_samples import generate_samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    model_dir = Path(model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    args = Config(str(args_path))
    model = Diffusion_Planner(args)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, args


def draw_road_borders(ax, data_raw, max_dist=25):
    if 'lanes' in data_raw:
        lanes = data_raw['lanes']
        for s in range(lanes.shape[0]):
            for bx_idx, by_idx in [(4, 5), (6, 7)]:
                pts = []
                for p in range(lanes.shape[1]):
                    cx, cy = lanes[s, p, 0], lanes[s, p, 1]
                    bx, by = lanes[s, p, bx_idx], lanes[s, p, by_idx]
                    if abs(cx) > max_dist or abs(cy) > max_dist:
                        continue
                    if abs(cx) < 0.01 and abs(cy) < 0.01:
                        continue
                    if abs(bx) > 0.01 or abs(by) > 0.01:
                        pts.append([cx + bx, cy + by])
                if pts:
                    pts = np.array(pts)
                    ax.plot(pts[:, 0], pts[:, 1], '-', color='orange',
                            linewidth=0.8, alpha=0.4)
    if 'route_lanes' in data_raw:
        rl = data_raw['route_lanes']
        for s in range(rl.shape[0]):
            for bx_idx, by_idx in [(4, 5), (6, 7)]:
                pts = []
                for p in range(rl.shape[1]):
                    cx, cy = rl[s, p, 0], rl[s, p, 1]
                    bx, by = rl[s, p, bx_idx], rl[s, p, by_idx]
                    if abs(cx) > max_dist or abs(cy) > max_dist:
                        continue
                    if abs(cx) < 0.01 and abs(cy) < 0.01:
                        continue
                    if abs(bx) > 0.01 or abs(by) > 0.01:
                        pts.append([cx + bx, cy + by])
                if pts:
                    pts = np.array(pts)
                    ax.plot(pts[:, 0], pts[:, 1], '-', color='red',
                            linewidth=1.5, alpha=0.6)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA checkpoint dir for DiT weights")
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--n_scenes", type=int, default=9)
    parser.add_argument("--lambda_lat", type=float, default=2.5)
    parser.add_argument("--lambda_lon", type=float, default=0.25)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--raw_scale", type=float, default=10.0)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # Load base model
    model, model_args = load_model(args.model_path)

    # Load LoRA if specified
    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()
        print(f"Loaded LoRA from {args.lora_path}")

    # Load exploration policy
    policy_config = ExplorationPolicyConfig(
        hidden_dim=128, head_init='zeros', head_raw_scale=args.raw_scale
    )
    policy = ExplorationPolicy(policy_config).to(device)
    ckpt = torch.load(args.policy_path, map_location=device)
    policy.load_state_dict(ckpt, strict=False)
    policy.eval()
    encoder = get_frozen_encoder(model)

    with open(args.scenes) as f:
        all_scenes = json.load(f)

    if args.indices:
        indices = args.indices
    else:
        step = max(1, len(all_scenes) // args.n_scenes)
        indices = list(range(0, len(all_scenes), step))[:args.n_scenes]

    n = len(indices)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, si in enumerate(indices):
        npz_path = all_scenes[si]
        data_raw = np.load(npz_path)
        data = load_npz_data(npz_path, device)
        norm_data = {k: v.clone() for k, v in data.items()}
        norm_data = model_args.observation_normalizer(norm_data)

        # Get explorer's eta
        scene_enc = encoder(norm_data)
        x_ref = generate_reference_trajectory(model, model_args, norm_data, device)
        if isinstance(x_ref, np.ndarray):
            x_ref_t = torch.from_numpy(x_ref).float().to(device)
        else:
            x_ref_t = x_ref
        if x_ref_t.dim() == 2:
            x_ref_t = x_ref_t.unsqueeze(0)

        output = policy(scene_enc, x_ref_t, deterministic=True)
        eta_lat = output.lat_dist.mean.item() * 2 - 1
        eta_lon = output.lon_dist.mean.item() * 2 - 1

        # 1) Unguided deterministic trajectory
        traj_no_guide = generate_samples(
            model, model_args, norm_data, 0.0, 1, None, device
        )[0]  # [T, 4]

        # Use unguided trajectory as the reference for guidance
        # (this is what guidance pushes away from)
        norm_data["reference_trajectory"] = torch.tensor(
            traj_no_guide[None], device=device, dtype=torch.float32
        )

        # 2) Guided trajectory with explorer's eta
        guidance_fns = [
            GuidanceConfig(
                name="lateral", enabled=True, scale=1.0,
                params={"lambda_lat": args.lambda_lat, "eta_lat": eta_lat},
            ),
            GuidanceConfig(
                name="longitudinal", enabled=True, scale=1.0,
                params={"lambda_lon": args.lambda_lon, "eta_lon": eta_lon},
            ),
        ]
        set_cfg = GuidanceSetConfig(
            functions=guidance_fns, global_scale=args.guidance_scale
        )
        composer = GuidanceComposer(set_cfg)

        traj_guided = generate_samples(
            model, model_args, norm_data, 0.0, 1, composer, device
        )[0]  # [T, 4]

        # GT
        gt = data_raw['ego_agent_future']
        ego_past = data_raw['ego_agent_past']

        T = min(80, traj_no_guide.shape[0], traj_guided.shape[0])

        ax = axes[plot_idx]
        draw_road_borders(ax, data_raw)

        ax.plot(ego_past[:, 0], ego_past[:, 1], 'k-', linewidth=3,
                label='ego past', zorder=5)
        ax.plot(gt[:T, 0], gt[:T, 1], 'g-', linewidth=2.5,
                label='GT', zorder=4)
        ax.plot(traj_no_guide[:T, 0], traj_no_guide[:T, 1], 'b-',
                linewidth=2.5, label='no guidance', zorder=3)
        ax.plot(traj_guided[:T, 0], traj_guided[:T, 1], 'm--',
                linewidth=2.5, label='with guidance', zorder=6)

        # Arrows showing actual shift direction
        for t in [10, 20, 35]:
            if t < T:
                ax.annotate('', xy=(traj_guided[t, 0], traj_guided[t, 1]),
                            xytext=(traj_no_guide[t, 0], traj_no_guide[t, 1]),
                            arrowprops=dict(arrowstyle='->', color='magenta', lw=2))

        name = Path(npz_path).stem[-20:]
        lat_cm = eta_lat * args.lambda_lat * 100
        lon_pct = eta_lon * args.lambda_lon * 100
        ax.set_title(
            f'Scene {si}: {name}\n'
            f'η_lat={eta_lat:.3f} ({lat_cm:.0f}cm), η_lon={eta_lon:.3f} ({lon_pct:.1f}%)',
            fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        all_x = np.concatenate([traj_no_guide[:T, 0], traj_guided[:T, 0],
                                gt[:T, 0], ego_past[:, 0]])
        all_y = np.concatenate([traj_no_guide[:T, 1], traj_guided[:T, 1],
                                gt[:T, 1], ego_past[:, 1]])
        margin = 3
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        'Actual Guided Inference\n'
        'Blue=unguided, Magenta=guided (real DiT output), Green=GT, Red=road borders',
        fontsize=13)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f'Saved to {args.output}')


if __name__ == "__main__":
    main()
