#!/usr/bin/env python3
"""Visualize exploration policy's effect on trajectories.

For each scene: generates a reference (unguided) trajectory and an explorer-guided
trajectory, then plots both on the same scene with road borders and lane geometry.

Usage:
    python -m rlvr.autoresearch.tools.viz_explorer_trajectories \
      --model_path /path/to/best_model.pth \
      --exp_dir /path/to/experiment_dir \
      --scenes /path/to/scenes.json \
      --epoch 10 \
      --output_dir /path/to/output \
      [--lora_path /path/to/lora_dir] \
      [--n_scenes 10] \
      [--indices 0 5 10 15] \
      [--lambda_lat 2.5] [--lambda_lon 0.25] [--guidance_scale 0.5] \
      [--cols 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from diffusion_planner.utils.config import Config
from matplotlib.patches import Rectangle

from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from guidance_gui.generate_samples import generate_samples
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.grpo_config import GRPOConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model / explorer loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, lora_path: str | None = None):
    """Load base model and optionally apply LoRA."""
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
    model.to(DEVICE)
    if lora_path:
        model = load_lora_checkpoint(model, lora_path)
    model.eval()
    return model, args


def load_explorer(
    exp_dir: Path, epoch: int, model_args, device: torch.device,
) -> ExplorationPolicy | None:
    """Load exploration policy from experiment checkpoint."""
    cfg_path = exp_dir / "grpo_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        config = GRPOConfig()
        for k, v in cfg_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
    else:
        config = GRPOConfig()

    # Find checkpoint dir
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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_borders_and_lanes(ax, npz_path: str):
    """Draw road borders (line_strings ch3) and lane boundaries."""
    npz = np.load(npz_path)

    # Lane boundaries (left/right offsets from centerline)
    lanes = npz["lanes"]
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        if lane.shape[1] > 7:
            pts = lane[:, :2]
            lb, rb = lane[:, 4:6], lane[:, 6:8]
            v = np.abs(pts).sum(axis=1) > 0.1
            if v.sum() > 1:
                ax.plot((pts + lb)[v, 0], (pts + lb)[v, 1], "-",
                        color="#bbb", alpha=0.5, lw=0.7)
                ax.plot((pts + rb)[v, 0], (pts + rb)[v, 1], "-",
                        color="#bbb", alpha=0.5, lw=0.7)
        # Centerline
        valid = np.abs(lane[:, :2]).sum(axis=1) > 0.1
        if valid.sum() > 1:
            ax.plot(lane[valid, 0], lane[valid, 1], "--",
                    color="#999", alpha=0.3, lw=0.5)

    # Road borders from line_strings channel 3
    ls = npz["line_strings"]
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        if ls.shape[-1] >= 4 and line[:, 3].max() > 0.5:
            valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
            if valid.sum() > 1:
                ax.plot(line[valid, 0], line[valid, 1],
                        color="red", lw=3, alpha=0.7, zorder=4)

    return npz


def draw_trajectory(ax, traj, label, color, linestyle="-", zorder=10):
    """Draw a trajectory with dots every 3 steps and endpoint footprint."""
    pl = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
    ax.plot(traj[:, 0], traj[:, 1], linestyle, color=color, lw=2,
            alpha=0.6, zorder=zorder)
    ax.plot(traj[::3, 0], traj[::3, 1], "o", color=color, ms=3.5,
            alpha=0.9, mew=0, zorder=zorder + 1,
            label=f"{label} ({pl:.1f}m)")


def draw_footprints(ax, traj, color, zorder=8):
    """Draw ego footprints at regular intervals along trajectory."""
    wb, length, width = 2.75, 4.34, 1.70
    ro = (length - wb) / 2

    for ts in range(5, len(traj), 10):
        cx, cy = traj[ts, 0], traj[ts, 1]
        cos_h, sin_h = traj[ts, 2], traj[ts, 3]
        hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
        if hn > 0.01:
            heading = np.arctan2(sin_h / hn, cos_h / hn)
            t_rot = (mtransforms.Affine2D().rotate(heading).translate(cx, cy)
                     + ax.transData)
            ax.add_patch(Rectangle(
                (-ro, -width / 2), length, width, lw=0.5,
                ec=color, fc=color, alpha=0.15, zorder=zorder, transform=t_rot,
            ))

    # Endpoint footprint (stronger)
    t_end = len(traj) - 1
    cx, cy = traj[t_end, 0], traj[t_end, 1]
    cos_h, sin_h = traj[t_end, 2], traj[t_end, 3]
    hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
    if hn > 0.01:
        heading = np.arctan2(sin_h / hn, cos_h / hn)
        t_rot = (mtransforms.Affine2D().rotate(heading).translate(cx, cy)
                 + ax.transData)
        ax.add_patch(Rectangle(
            (-ro, -width / 2), length, width, lw=1.5,
            ec=color, fc=color, alpha=0.4, zorder=zorder + 1, transform=t_rot,
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Visualize exploration policy's effect on trajectories",
    )
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Base model .pth")
    parser.add_argument("--exp_dir", type=Path, required=True,
                        help="Experiment dir (contains grpo_config.json)")
    parser.add_argument("--scenes", type=Path, required=True,
                        help="JSON list of NPZ scene paths")
    parser.add_argument("--epoch", type=int, default=10,
                        help="Checkpoint epoch to load")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save output images")
    parser.add_argument("--lora_path", type=Path, default=None,
                        help="LoRA checkpoint dir (optional)")
    parser.add_argument("--n_scenes", type=int, default=10,
                        help="Number of scenes (evenly spaced) if --indices not given")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Specific scene indices to visualize")
    parser.add_argument("--lambda_lat", type=float, default=2.5,
                        help="Lateral guidance lambda")
    parser.add_argument("--lambda_lon", type=float, default=0.25,
                        help="Longitudinal guidance lambda")
    parser.add_argument("--guidance_scale", type=float, default=0.5,
                        help="Global guidance scale")
    parser.add_argument("--cols", type=int, default=3,
                        help="Columns in grid layout")
    args = parser.parse_args()

    device = torch.device(DEVICE)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load scenes
    with open(args.scenes) as f:
        all_scenes = json.load(f)

    if args.indices:
        indices = args.indices
    else:
        step = max(1, len(all_scenes) // args.n_scenes)
        indices = list(range(0, len(all_scenes), step))[:args.n_scenes]

    # Load model
    lora_str = str(args.lora_path) if args.lora_path else None
    print(f"Loading model from {args.model_path}")
    model, model_args = load_model(str(args.model_path), lora_str)

    # Load exploration policy
    explorer = load_explorer(args.exp_dir, args.epoch, model_args, device)
    if explorer is None:
        print("Failed to load explorer. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Per-scene: generate reference + guided trajectories, collect eta values
    # -----------------------------------------------------------------------
    results = []  # list of dicts per scene

    for si in indices:
        npz_path = all_scenes[si]
        name = Path(npz_path).stem
        print(f"  Scene {si}: {name}")

        try:
            data = load_npz_data(npz_path, device)
            if "delay" not in data:
                data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
            norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in data.items()}
            norm_data = model_args.observation_normalizer(norm_data)

            # 1) Reference trajectory (deterministic, no guidance)
            traj_ref = generate_reference_trajectory(
                model, model_args, norm_data, device,
            )  # (T, 4) numpy

            # 2) Get explorer's eta values
            scene_enc = run_frozen_encoder(model, norm_data)
            x_ref_t = torch.from_numpy(traj_ref).unsqueeze(0).float().to(device)
            output = explorer(scene_enc, x_ref_t, deterministic=True)
            eta_lat = output.eta_lat.item()
            eta_lon = output.eta_lon.item()

            # 3) Set reference trajectory for guidance displacement
            norm_data["reference_trajectory"] = torch.tensor(
                traj_ref[None], device=device, dtype=torch.float32,
            )

            # 4) Guided trajectory using explorer's etas
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
                functions=guidance_fns, global_scale=args.guidance_scale,
            )
            composer = GuidanceComposer(set_cfg)

            traj_guided = generate_samples(
                model, model_args, norm_data, 0.0, 1, composer, device,
            )[0]  # (T, 4) numpy

            results.append({
                "si": si,
                "npz_path": npz_path,
                "name": name,
                "traj_ref": traj_ref,
                "traj_guided": traj_guided,
                "eta_lat": eta_lat,
                "eta_lon": eta_lon,
            })
        except Exception as e:
            print(f"    SKIP: {e}")

    if not results:
        print("No scenes processed successfully.")
        return

    # -----------------------------------------------------------------------
    # Plot grid
    # -----------------------------------------------------------------------
    n = len(results)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).reshape(-1)

    for plot_idx, res in enumerate(results):
        ax = axes[plot_idx]
        npz_path = res["npz_path"]
        traj_ref = res["traj_ref"]
        traj_guided = res["traj_guided"]

        # Scene geometry
        npz = draw_borders_and_lanes(ax, npz_path)

        # GT trajectory
        gt = npz["ego_agent_future"]
        ax.plot(gt[:, 0], gt[:, 1], "g-", lw=2, alpha=0.5, zorder=5)
        ax.plot(gt[::3, 0], gt[::3, 1], "go", ms=3, alpha=0.7, mew=0,
                zorder=6, label="GT")

        # Ego box at origin
        wb, length, width = 2.75, 4.34, 1.70
        es = npz.get("ego_shape", None)
        if es is not None and len(es) >= 3:
            wb, length, width = float(es[0]), float(es[1]), float(es[2])
        ro = (length - wb) / 2
        ax.add_patch(Rectangle(
            (-ro, -width / 2), length, width,
            lw=2, ec="black", fc="#3366cc", alpha=0.9, zorder=20,
        ))

        # Reference trajectory (blue)
        draw_trajectory(ax, traj_ref, "Reference", "blue", "-", zorder=10)
        draw_footprints(ax, traj_ref, "blue", zorder=8)

        # Guided trajectory (red)
        draw_trajectory(ax, traj_guided, "Explorer-guided", "red", "--", zorder=12)
        draw_footprints(ax, traj_guided, "red", zorder=11)

        # Shift arrows at a few timesteps
        T = min(80, traj_ref.shape[0], traj_guided.shape[0])
        for t in [10, 20, 35, 55]:
            if t < T:
                dx = traj_guided[t, 0] - traj_ref[t, 0]
                dy = traj_guided[t, 1] - traj_ref[t, 1]
                if np.sqrt(dx ** 2 + dy ** 2) > 0.05:
                    ax.annotate(
                        "", xy=(traj_guided[t, 0], traj_guided[t, 1]),
                        xytext=(traj_ref[t, 0], traj_ref[t, 1]),
                        arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5),
                    )

        # Auto-zoom
        all_pts = np.vstack([
            traj_ref[:, :2], traj_guided[:, :2], gt[:, :2], [[0, 0]],
        ])
        cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
        half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7, loc="upper left")

        # Annotation with eta values
        lat_cm = res["eta_lat"] * args.lambda_lat * 100
        lon_pct = res["eta_lon"] * args.lambda_lon * 100
        ax.set_title(
            f"[{res['si']}] {res['name'][-25:]}\n"
            f"eta_lat={res['eta_lat']:+.3f} ({lat_cm:+.0f}cm)  "
            f"eta_lon={res['eta_lon']:+.3f} ({lon_pct:+.1f}%)",
            fontsize=9,
        )

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Exploration Policy: Reference (blue) vs Guided (red)\n"
        f"lambda_lat={args.lambda_lat}, lambda_lon={args.lambda_lon}, "
        f"guidance_scale={args.guidance_scale}",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = args.output_dir / "explorer_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid: {out_path}")

    # Also save individual per-scene images
    for res in results:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(10, 10))
        npz = draw_borders_and_lanes(ax_s, res["npz_path"])

        gt = npz["ego_agent_future"]
        ax_s.plot(gt[:, 0], gt[:, 1], "g-", lw=2, alpha=0.5, zorder=5)
        ax_s.plot(gt[::3, 0], gt[::3, 1], "go", ms=3, alpha=0.7, mew=0,
                  zorder=6, label="GT")

        es = npz.get("ego_shape", None)
        wb = float(es[0]) if es is not None and len(es) >= 1 else 2.75
        length = float(es[1]) if es is not None and len(es) >= 2 else 4.34
        width = float(es[2]) if es is not None and len(es) >= 3 else 1.70
        ro = (length - wb) / 2
        ax_s.add_patch(Rectangle(
            (-ro, -width / 2), length, width,
            lw=2, ec="black", fc="#3366cc", alpha=0.9, zorder=20,
        ))

        draw_trajectory(ax_s, res["traj_ref"], "Reference", "blue", "-", zorder=10)
        draw_footprints(ax_s, res["traj_ref"], "blue", zorder=8)
        draw_trajectory(ax_s, res["traj_guided"], "Explorer-guided", "red", "--", zorder=12)
        draw_footprints(ax_s, res["traj_guided"], "red", zorder=11)

        T = min(80, res["traj_ref"].shape[0], res["traj_guided"].shape[0])
        for t in [10, 20, 35, 55]:
            if t < T:
                dx = res["traj_guided"][t, 0] - res["traj_ref"][t, 0]
                dy = res["traj_guided"][t, 1] - res["traj_ref"][t, 1]
                if np.sqrt(dx ** 2 + dy ** 2) > 0.05:
                    ax_s.annotate(
                        "", xy=(res["traj_guided"][t, 0], res["traj_guided"][t, 1]),
                        xytext=(res["traj_ref"][t, 0], res["traj_ref"][t, 1]),
                        arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5),
                    )

        all_pts = np.vstack([
            res["traj_ref"][:, :2], res["traj_guided"][:, :2], gt[:, :2], [[0, 0]],
        ])
        cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
        half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
        ax_s.set_xlim(cx - half, cx + half)
        ax_s.set_ylim(cy - half, cy + half)
        ax_s.set_aspect("equal")
        ax_s.grid(True, alpha=0.15)
        ax_s.legend(fontsize=9, loc="upper left")

        lat_cm = res["eta_lat"] * args.lambda_lat * 100
        lon_pct = res["eta_lon"] * args.lambda_lon * 100
        ax_s.set_title(
            f"Scene {res['si']}: {res['name']}\n"
            f"eta_lat={res['eta_lat']:+.3f} ({lat_cm:+.0f}cm)  "
            f"eta_lon={res['eta_lon']:+.3f} ({lon_pct:+.1f}%)",
            fontsize=11,
        )

        fig_s.tight_layout()
        scene_out = args.output_dir / f"scene_{res['si']:04d}.png"
        fig_s.savefig(scene_out, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  Saved: {scene_out}")

    print(f"\nDone. {len(results)} scenes visualized in {args.output_dir}")


if __name__ == "__main__":
    main()
