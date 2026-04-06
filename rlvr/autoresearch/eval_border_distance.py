#!/usr/bin/env python3
"""Evaluate per-scene minimum border distance for a model on a set of scenes.

Computes:
- Per-scene min border distance (meters) across all timesteps
- Per-scene mean border distance
- border_t20 (distance at t=2s, timestep 20)
- Aggregate stats: mean, min, worst-N scenes
- Optional: visualization of worst scenes with distance annotations

Usage:
    # Baseline model
    python rlvr/eval_border_distance.py \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/problem_scenes.json \
        --tag baseline

    # LoRA model
    python rlvr/eval_border_distance.py \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/problem_scenes.json \
        --lora_path /path/to/lora_epoch_003 \
        --tag p6m_ep3

    # With visualization of worst 10 scenes
    python rlvr/eval_border_distance.py \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/problem_scenes.json \
        --lora_path /path/to/lora_epoch_003 \
        --tag p6m_ep3 --visualize --output_dir ~/Pictures/border_viz

    # Use merged model directly
    python rlvr/eval_border_distance.py \
        --merged_model_path /path/to/merged.pth \
        --args_json /path/to/args.json \
        --scenes /path/to/problem_scenes.json \
        --tag p6m_ep3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffusion_planner.dimensions import OUTPUT_T
from diffusion_planner.loss import compute_ego_edge_points, point_to_segment_distance


def load_model_for_eval(args):
    """Load model — either base+LoRA or merged."""
    from preference_optimization.model_utils import load_model
    from diffusion_planner.utils.config import Config

    if args.merged_model_path:
        model, model_args = load_model(Path(args.merged_model_path), "cuda")
        if args.args_json:
            model_args = Config(args.args_json)
        return model, model_args

    model, model_args = load_model(Path(args.model_path), "cuda")
    if args.lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint
        model = load_lora_checkpoint(model, args.lora_path, is_trainable=False)
    return model, model_args


def generate_deterministic_trajectory(model, model_args, data, device="cuda"):
    """Generate a single deterministic trajectory (noise_scale=0)."""
    P = 1 + model_args.predicted_neighbor_num
    ego_current = data["ego_current_state"][:, :4]
    neighbors_current = data["neighbor_agents_past"][:, :P-1, -1, :4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

    xT = current_states[:, :, None, :].expand(-1, -1, OUTPUT_T + 1, -1).clone()
    # noise_scale=0 for deterministic
    data["sampled_trajectories"] = xT.reshape(1, P, -1)

    with torch.no_grad():
        _, decoder_output = model(data)
    return decoder_output["prediction"][:, 0]  # [1, T, 4]


def compute_border_distances(ego_traj, ego_shape, line_strings):
    """Compute per-timestep min distance from ego perimeter to road border.

    Args:
        ego_traj: [1, T, 4] x, y, cos, sin
        ego_shape: [1, 3] or [3]
        line_strings: [1, N, P, D] or [N, P, D]

    Returns:
        min_dists: [T] min distance per timestep (meters)
    """
    if ego_shape.dim() == 1:
        ego_shape = ego_shape.unsqueeze(0)
    if line_strings.dim() == 3:
        line_strings = line_strings.unsqueeze(0)

    ls = line_strings[0]  # [N, P, D]
    ls_xy = ls[..., :2]  # [N, P, 2]
    rb_mask = (ls[..., 3] > 0.5).any(dim=-1)  # [N]

    if not rb_mask.any():
        return torch.full((ego_traj.shape[1],), float("inf"))

    ego_edge_points = compute_ego_edge_points(ego_traj, ego_shape, n_interp=0)
    # ego_edge_points: [1, T, K, 2]

    T, K = ego_edge_points.shape[1], ego_edge_points.shape[2]

    seg_a = ls_xy[rb_mask, :-1, :]  # [M, S, 2]
    seg_b = ls_xy[rb_mask, 1:, :]   # [M, S, 2]
    M, S, _ = seg_a.shape
    seg_valid = ((seg_a.abs().sum(-1) > 1e-6) & (seg_b.abs().sum(-1) > 1e-6)).bool()

    seg_a_flat = seg_a.reshape(M * S, 2)
    seg_b_flat = seg_b.reshape(M * S, 2)
    seg_valid_flat = seg_valid.reshape(M * S).bool()

    p = ego_edge_points[0].reshape(T * K, 1, 2)
    a = seg_a_flat[None, :, :]
    b = seg_b_flat[None, :, :]

    dist = point_to_segment_distance(p, a, b)  # [T*K, M*S]
    dist = torch.where(seg_valid_flat[None, :], dist, torch.full_like(dist, float("inf")))

    min_per_point = dist.min(dim=-1).values.reshape(T, K)
    return min_per_point.min(dim=-1).values  # [T]


def load_npz_data(path, device="cuda"):
    """Load npz and prepare for model."""
    from preference_optimization.utils import load_npz_data as _load
    return _load(path, device)


def visualize_scene_border(ego_traj_np, min_dists_np, data, scene_path, save_path,
                           tag="", rb_crossing=False):
    """Visualize a single scene with border distance annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: map view with trajectory and road borders
    ax = axes[0]
    ls = data["line_strings"][0].cpu().numpy()  # [N, P, D]
    for i in range(ls.shape[0]):
        pts = ls[i, :, :2]
        valid = np.linalg.norm(pts, axis=-1) > 1e-3
        if not valid.any():
            continue
        is_border = ls[i, :, 3] > 0.5
        if is_border.any():
            ax.plot(pts[valid, 0], pts[valid, 1], 'r-', linewidth=2, alpha=0.7)
        else:
            ax.plot(pts[valid, 0], pts[valid, 1], '-', color='orange', linewidth=0.5, alpha=0.3)

    # GT trajectory
    gt = data.get("ego_agent_future")
    if gt is not None:
        gt_np = gt[0].cpu().numpy()
        ax.plot(gt_np[:, 0], gt_np[:, 1], 'g-', linewidth=1.5, alpha=0.5, label='GT')

    # Model trajectory colored by distance
    colors = plt.cm.RdYlGn(np.clip(min_dists_np / 1.0, 0, 1))
    for t in range(len(ego_traj_np) - 1):
        ax.plot(ego_traj_np[t:t+2, 0], ego_traj_np[t:t+2, 1],
                color=colors[t], linewidth=2.5)

    # Mark t=20 (2s) with distance annotation
    if len(min_dists_np) > 20:
        ax.plot(ego_traj_np[20, 0], ego_traj_np[20, 1], 'ko', markersize=8, zorder=5)
        ax.annotate(f't=2s\n{min_dists_np[20]:.2f}m',
                    xy=(ego_traj_np[20, 0], ego_traj_np[20, 1]),
                    fontsize=8, ha='left', fontweight='bold')

    # Mark overall minimum
    t_min = np.argmin(min_dists_np)
    ax.plot(ego_traj_np[t_min, 0], ego_traj_np[t_min, 1], 'r*', markersize=12, zorder=5)
    ax.annotate(f'min={min_dists_np[t_min]:.2f}m\nt={t_min}',
                xy=(ego_traj_np[t_min, 0], ego_traj_np[t_min, 1]),
                fontsize=8, ha='left', color='red', fontweight='bold')

    ax.set_aspect('equal')
    title = (f'{tag} — {Path(scene_path).stem}\n'
             f'rb_cross={"YES" if rb_crossing else "no"}  '
             f'min_dist={min_dists_np.min():.3f}m')
    if len(min_dists_np) > 20:
        title += f'  border_t20={min_dists_np[20]:.3f}m'
    ax.set_title(title)
    ax.legend(fontsize=8)

    # Auto-zoom
    pad = 5
    xmin, xmax = ego_traj_np[:, 0].min() - pad, ego_traj_np[:, 0].max() + pad
    ymin, ymax = ego_traj_np[:, 1].min() - pad, ego_traj_np[:, 1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Right: distance over time
    ax2 = axes[1]
    timesteps = np.arange(len(min_dists_np)) * 0.1
    ax2.plot(timesteps, min_dists_np, 'b-', linewidth=2)
    ax2.axhline(y=0.10, color='r', linestyle='--', label='crossing (10cm)')
    ax2.axhline(y=0.25, color='orange', linestyle='--', label='near (25cm)')
    ax2.axhline(y=0.40, color='y', linestyle='--', label='wide (40cm)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Min border distance (m)')
    ax2.set_title('Distance to road border over time')
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0, top=max(2.0, min_dists_np.max() * 1.1))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--merged_model_path", type=str, default=None)
    parser.add_argument("--args_json", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--worst_n", type=int, default=10, help="Visualize N worst scenes")
    args = parser.parse_args()

    device = "cuda"

    # Load scenes
    with open(args.scenes) as f:
        scene_paths = json.load(f)
    print(f"Evaluating {len(scene_paths)} scenes with tag={args.tag}")

    # Load model
    model, model_args = load_model_for_eval(args)
    model.eval()

    # Normalize function
    norm_fn = model_args.observation_normalizer

    results = []
    for i, path in enumerate(tqdm(scene_paths, desc=f"Border dist [{args.tag}]")):
        try:
            data = load_npz_data(path, device)
            norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in data.items()}
            norm_data = norm_fn(norm_data)

            ego_traj = generate_deterministic_trajectory(model, model_args, norm_data, device)
            # ego_traj: [1, T, 4]

            ego_shape = data["ego_shape"][0]  # [3]
            line_strings = data["line_strings"]  # [1, N, P, D]

            min_dists = compute_border_distances(ego_traj, ego_shape, line_strings)
            min_dists_np = min_dists.cpu().numpy()

            # Check for crossing
            rb_crossing = bool((min_dists_np < 0.10).any())

            results.append({
                "scene_idx": i,
                "scene_path": path,
                "min_dist_overall": float(min_dists_np.min()),
                "mean_dist": float(min_dists_np.mean()),
                "border_t20": float(min_dists_np[20]) if len(min_dists_np) > 20 else float("inf"),
                "rb_crossing": rb_crossing,
                "min_dists": min_dists_np,
                "ego_traj": ego_traj[0].cpu().numpy(),
                "data": data if args.visualize else None,
            })
        except Exception as e:
            print(f"  Error on scene {i}: {e}")
            continue

    if not results:
        print("No results!")
        return

    # Aggregate stats
    min_dists_all = [r["min_dist_overall"] for r in results]
    mean_dists_all = [r["mean_dist"] for r in results]
    border_t20_all = [r["border_t20"] for r in results]
    crossings = sum(1 for r in results if r["rb_crossing"])

    print(f"\n{'='*60}")
    print(f"Border Distance Results — {args.tag}")
    print(f"{'='*60}")
    print(f"  Scenes evaluated: {len(results)}")
    print(f"  rb_crossings: {crossings}/{len(results)}")
    print(f"  min_dist_overall:  mean={np.mean(min_dists_all):.3f}m  "
          f"min={np.min(min_dists_all):.3f}m  p5={np.percentile(min_dists_all, 5):.3f}m")
    print(f"  mean_dist:         mean={np.mean(mean_dists_all):.3f}m  "
          f"min={np.min(mean_dists_all):.3f}m")
    print(f"  border_t20:        mean={np.mean(border_t20_all):.3f}m  "
          f"min={np.min(border_t20_all):.3f}m  p5={np.percentile(border_t20_all, 5):.3f}m")

    # Print worst scenes
    sorted_by_min = sorted(results, key=lambda r: r["min_dist_overall"])
    print(f"\n  Worst {min(10, len(sorted_by_min))} scenes by min distance:")
    for r in sorted_by_min[:10]:
        print(f"    scene {r['scene_idx']:4d}: min={r['min_dist_overall']:.3f}m  "
              f"t20={r['border_t20']:.3f}m  cross={'YES' if r['rb_crossing'] else 'no'}  "
              f"— {Path(r['scene_path']).stem}")

    # Visualize worst scenes
    if args.visualize and args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in sorted_by_min[:args.worst_n]:
            save_path = out_dir / f"{args.tag}_scene{r['scene_idx']:04d}_{Path(r['scene_path']).stem}.png"
            visualize_scene_border(
                r["ego_traj"], r["min_dists"], r["data"], r["scene_path"],
                save_path, tag=args.tag, rb_crossing=r["rb_crossing"]
            )
            print(f"  Saved: {save_path}")

    # Machine-readable summary
    summary = {
        "tag": args.tag,
        "n_scenes": len(results),
        "rb_crossings": crossings,
        "min_dist_mean": float(np.mean(min_dists_all)),
        "min_dist_min": float(np.min(min_dists_all)),
        "min_dist_p5": float(np.percentile(min_dists_all, 5)),
        "mean_dist_mean": float(np.mean(mean_dists_all)),
        "border_t20_mean": float(np.mean(border_t20_all)),
        "border_t20_min": float(np.min(border_t20_all)),
    }
    if args.output_dir:
        summary_path = Path(args.output_dir) / f"{args.tag}_border_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")

    return summary


if __name__ == "__main__":
    main()
