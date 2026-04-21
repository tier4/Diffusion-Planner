#!/usr/bin/env python3
"""Multi-model comparison visualization on the same scenes.

Overlays baseline + multiple trained models on each scene with proper road borders,
lanes, ego footprints, and border distance annotations.

Usage:
    python -m rlvr.autoresearch.compare_models \
      --base_model /path/to/best_model.pth \
      --models p6m_ep3:/path/to/merged_epoch3.pth \
               zi_ep4:/path/to/merged_epoch4.pth \
               zi_ep7:/path/to/merged_epoch7.pth \
      --scenes /path/to/prob_scenes.json \
      --output_dir /path/to/output \
      --n_scenes 8
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from diffusion_planner.loss import compute_ego_edge_points, point_to_segment_distance
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from matplotlib.patches import Rectangle

from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_COLORS = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0", "#00BCD4"]
BASELINE_COLOR = "#666666"
GT_COLOR = "#2E7D32"


def load_merged_model(model_path, args_json=None):
    """Load a merged model (base weights already included)."""
    model_dir = Path(model_path).parent
    if args_json:
        args = Config(args_json)
    else:
        args = Config(str(model_dir / "args.json"))
    model = Diffusion_Planner(args)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, args


def infer_deterministic(model, args, npz_path, reward_config=None):
    """Generate deterministic trajectory and compute reward."""
    if reward_config is None:
        reward_config = RewardConfig()
    data = load_npz_data(npz_path, DEVICE)
    norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    norm = args.observation_normalizer(norm)
    traj = generate_samples(model, args, norm, 0.0, 1, None, DEVICE)[0]
    traj_t = torch.tensor(traj[None], device=DEVICE, dtype=torch.float32)
    r = compute_reward_batch(traj_t, data, reward_config)[0]
    return traj, data, r


def compute_min_border_dist(traj, data):
    """Compute minimum border distance for a trajectory."""
    ego_traj = torch.tensor(traj[None], device=DEVICE, dtype=torch.float32)
    es = data.get("ego_shape")
    if es is not None:
        ego_shape = es[0] if es.dim() > 1 else es
    else:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=DEVICE)
    ls = data["line_strings"]
    if ls.dim() == 4:
        ls = ls[0]
    rb_mask = (ls[..., 3] > 0.5).any(dim=-1)
    if not rb_mask.any():
        return float("inf")
    try:
        ego_edge = compute_ego_edge_points(ego_traj, ego_shape[None], n_interp=0)
        T, K = ego_edge.shape[1], ego_edge.shape[2]
        seg_a = ls[rb_mask, :-1, :2]
        seg_b = ls[rb_mask, 1:, :2]
        M, S, _ = seg_a.shape
        seg_valid = ((seg_a.abs().sum(-1) > 1e-6) & (seg_b.abs().sum(-1) > 1e-6)).bool()
        seg_a_flat = seg_a.reshape(M * S, 2)
        seg_b_flat = seg_b.reshape(M * S, 2)
        seg_valid_flat = seg_valid.reshape(M * S).bool()
        p = ego_edge[0].reshape(T * K, 1, 2)
        a = seg_a_flat[None, :, :]
        b = seg_b_flat[None, :, :]
        dist = point_to_segment_distance(p, a, b)
        dist = torch.where(seg_valid_flat[None, :], dist, torch.full_like(dist, float("inf")))
        min_per_point = dist.min(dim=-1).values.reshape(T, K)
        min_per_ts = min_per_point.min(dim=-1).values
        return float(min_per_ts.min().item())
    except Exception:
        return float("inf")


def draw_map(ax, npz_path):
    """Draw lanes and road borders."""
    npz = np.load(npz_path)

    # Lanes
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
                ax.plot((pts + lb)[v, 0], (pts + lb)[v, 1], "-", color="#ccc", alpha=0.5, lw=0.7)
                ax.plot((pts + rb)[v, 0], (pts + rb)[v, 1], "-", color="#ccc", alpha=0.5, lw=0.7)

    # Road borders (thick red)
    ls = npz["line_strings"]
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        if ls.shape[-1] >= 4 and line[:, 3].max() > 0.5:
            v = np.abs(line[:, :2]).sum(axis=1) > 0.1
            if v.sum() > 1:
                ax.plot(line[v, 0], line[v, 1], color="red", lw=3, alpha=0.7, zorder=4)

    # GT
    gt = npz["ego_agent_future"]
    ax.plot(gt[:, 0], gt[:, 1], "-", color=GT_COLOR, lw=2.5, alpha=0.4, zorder=5)
    ax.plot(gt[::5, 0], gt[::5, 1], "o", color=GT_COLOR, ms=3, alpha=0.6, mew=0, zorder=6, label="GT")

    # Ego box at t=0
    wb, length, width = 2.75, 4.34, 1.70
    es = npz.get("ego_shape", None)
    if es is not None and len(es) >= 3:
        wb, length, width = float(es[0]), float(es[1]), float(es[2])
    ro = (length - wb) / 2
    ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=2, ec="black", fc="#3366cc",
                            alpha=0.9, zorder=20))
    return npz


def draw_trajectory(ax, traj, label, color, npz, show_footprints=True):
    """Draw a trajectory with optional footprints."""
    wb, length, width = 2.75, 4.34, 1.70
    es = npz.get("ego_shape", None)
    if es is not None and len(es) >= 3:
        wb, length, width = float(es[0]), float(es[1]), float(es[2])
    ro = (length - wb) / 2

    pl = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2.5, alpha=0.6, zorder=10)
    ax.plot(traj[::5, 0], traj[::5, 1], "o", color=color, ms=3, alpha=0.8, mew=0, zorder=11,
            label=f"{label}")

    if show_footprints:
        # Footprints every 20 steps
        for ts in range(10, len(traj), 20):
            cx, cy = traj[ts, 0], traj[ts, 1]
            cos_h, sin_h = traj[ts, 2], traj[ts, 3]
            hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
            if hn > 0.01:
                heading = np.arctan2(sin_h / hn, cos_h / hn)
                t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
                ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=0.5, ec=color, fc=color,
                                       alpha=0.12, zorder=8, transform=t_rot))

        # Endpoint footprint
        t_end = len(traj) - 1
        cx, cy = traj[t_end, 0], traj[t_end, 1]
        cos_h, sin_h = traj[t_end, 2], traj[t_end, 3]
        hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
        if hn > 0.01:
            heading = np.arctan2(sin_h / hn, cos_h / hn)
            t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
            ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=1.5, ec=color, fc=color,
                                    alpha=0.35, zorder=9, transform=t_rot))


def auto_zoom(ax, all_trajs, npz):
    """Set axis limits to fit all trajectories + GT."""
    gt = npz["ego_agent_future"][:, :2]
    pts = [gt, np.array([[0, 0]])]
    for t in all_trajs:
        pts.append(t[:, :2])
    all_pts = np.vstack(pts)
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
    half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison on scenes")
    parser.add_argument("--base_model", type=str, required=True, help="Baseline model .pth")
    parser.add_argument("--base_args", type=str, default=None, help="Baseline args.json (optional)")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="name:path pairs for merged models, e.g. p6m_ep3:/path/to/merged.pth")
    parser.add_argument("--args_jsons", type=str, nargs="*", default=None,
                        help="args.json paths for each model (same order as --models)")
    parser.add_argument("--scenes", type=str, required=True, help="JSON list of NPZ paths")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--n_scenes", type=int, default=8)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--config", type=Path, default=None,
                        help="GRPO training config JSON. When given, the "
                             "CROSS / rb_crossing labels match the training "
                             "run's rb_cross_thresh and gate settings.")
    args = parser.parse_args()

    reward_config = load_reward_config(args.config) if args.config is not None else RewardConfig()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        indices = args.indices
    else:
        step = max(1, len(scenes) // args.n_scenes)
        indices = list(range(0, len(scenes), step))[:args.n_scenes]

    # Parse model specs
    model_specs = []
    for i, spec in enumerate(args.models):
        parts = spec.split(":", 1)
        name = parts[0]
        path = parts[1] if len(parts) > 1 else parts[0]
        aj = args.args_jsons[i] if args.args_jsons and i < len(args.args_jsons) else None
        model_specs.append((name, path, aj))

    # Load baseline
    print(f"Loading baseline: {args.base_model}")
    base_model, base_args = load_merged_model(args.base_model, args.base_args)

    # Load trained models
    models = {}
    for name, path, aj in model_specs:
        print(f"Loading {name}: {path}")
        m, a = load_merged_model(path, aj)
        models[name] = (m, a)

    # Generate comparison grid
    n = len(indices)
    rows = (n + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(10 * args.cols, 9 * rows))
    if rows == 1 and args.cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif args.cols == 1:
        axes = axes[:, None]
    axes_flat = axes.flatten()

    for plot_idx, si in enumerate(indices):
        if plot_idx >= len(axes_flat):
            break
        ax = axes_flat[plot_idx]
        npz_path = scenes[si]
        print(f"  Scene {si}: {Path(npz_path).stem}")

        # Draw map (lanes, borders, GT, ego box)
        npz = draw_map(ax, npz_path)

        # Baseline trajectory
        traj_b, data_b, r_b = infer_deterministic(base_model, base_args, npz_path, reward_config)
        min_d_b = compute_min_border_dist(traj_b, data_b)
        draw_trajectory(ax, traj_b, f"Base (d={min_d_b:.2f}m)", BASELINE_COLOR, npz,
                       show_footprints=False)

        # Each trained model
        all_trajs = [traj_b]
        title_parts = [f"Base: {'CROSS' if r_b.rb_crossing else f'd={min_d_b:.2f}m'}"]

        for mi, (name, _) in enumerate([(n, p) for n, p, _ in model_specs]):
            m, a = models[name]
            traj, data, r = infer_deterministic(m, a, npz_path, reward_config)
            min_d = compute_min_border_dist(traj, data)
            color = MODEL_COLORS[mi % len(MODEL_COLORS)]
            draw_trajectory(ax, traj, f"{name} (d={min_d:.2f}m)", color, npz)
            all_trajs.append(traj)
            title_parts.append(f"{name}: {'CROSS' if r.rb_crossing else f'd={min_d:.2f}m'}")

        auto_zoom(ax, all_trajs, npz)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"[{si}] {Path(npz_path).stem[:40]}\n" + " | ".join(title_parts), fontsize=7)

    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Color legend in suptitle
    model_names = ["Baseline (gray)"] + [f"{n} ({MODEL_COLORS[i]})" for i, (n, _, _) in enumerate(model_specs)]
    fig.suptitle("Multi-model comparison — " + ", ".join(model_names) + f"\nGT (green), Road borders (red)",
                 fontsize=12)
    fig.tight_layout()
    out = out_dir / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
