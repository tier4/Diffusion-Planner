#!/usr/bin/env python3
"""Test road border guidance on v4 model with miraikan prob scenes.

Loads v4 base model, runs deterministic inference with and without
road_border guidance at various scales, and saves visualization images.
"""

import json
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
sys.path.insert(0, "/home/danielsanchez/Diffusion-Planner/diffusion_planner")
sys.path.insert(0, "/home/danielsanchez/Diffusion-Planner")

from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.dimensions import OUTPUT_T, POSE_DIM
from diffusion_planner.model.guidance import (
    GuidanceComposer,
    GuidanceConfig,
    GuidanceSetConfig,
)


SSD = "/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207"
MODEL_PATH = f"{SSD}/v4.0/best_model.pth"
PROB_LIST_MIRAIKAN = f"{SSD}/auto_research/v4_prob_miraikan.json"
PROB_LIST_TELEPORT = f"{SSD}/auto_research/v4_prob_teleport.json"
SAVE_DIR = os.path.expanduser("~/Pictures/autoresearch/v4.0")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, device):
    """Load v4 model."""
    args_file = str(Path(model_path).parent / "args.json")
    model_args = Config(args_file)
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt if not isinstance(ckpt, dict) else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, model_args


def load_npz_as_batch(npz_path, model_args, device):
    """Load a single v4 NPZ and prepare as a batch of size 1."""
    d = np.load(npz_path)
    data = {}

    # ego_agent_past: (21, 3) -> need cos/sin format (21, 4)
    ego_past = d["ego_agent_past"]  # (21, 3) [x, y, yaw]
    if ego_past.shape[-1] == 3:
        cos_h = np.cos(ego_past[..., 2:3])
        sin_h = np.sin(ego_past[..., 2:3])
        ego_past = np.concatenate([ego_past[..., :2], cos_h, sin_h], axis=-1)
    data["ego_agent_past"] = torch.tensor(ego_past, dtype=torch.float32).unsqueeze(0)

    # ego_current_state: (10,)
    data["ego_current_state"] = torch.tensor(
        d["ego_current_state"], dtype=torch.float32
    ).unsqueeze(0)

    # ego_agent_future: (80, 3) -> (80, 4) cos/sin
    ego_future = d["ego_agent_future"]  # (80, 3)
    if ego_future.shape[-1] == 3:
        cos_h = np.cos(ego_future[..., 2:3])
        sin_h = np.sin(ego_future[..., 2:3])
        ego_future_4 = np.concatenate([ego_future[..., :2], cos_h, sin_h], axis=-1)
    else:
        ego_future_4 = ego_future
    data["ego_agent_future"] = torch.tensor(ego_future_4, dtype=torch.float32).unsqueeze(0)

    # neighbor_agents_past: (32, 21, 11)
    data["neighbor_agents_past"] = torch.tensor(
        d["neighbor_agents_past"], dtype=torch.float32
    ).unsqueeze(0)

    # lanes: (140, 20, SEGMENT_POINT_DIM)
    data["lanes"] = torch.tensor(d["lanes"], dtype=torch.float32).unsqueeze(0)
    data["lanes_speed_limit"] = torch.tensor(
        d["lanes_speed_limit"], dtype=torch.float32
    ).unsqueeze(0)
    data["lanes_has_speed_limit"] = torch.tensor(
        d["lanes_has_speed_limit"], dtype=torch.bool
    ).unsqueeze(0)

    # route_lanes
    data["route_lanes"] = torch.tensor(d["route_lanes"], dtype=torch.float32).unsqueeze(0)
    data["route_lanes_speed_limit"] = torch.tensor(
        d["route_lanes_speed_limit"], dtype=torch.float32
    ).unsqueeze(0)
    data["route_lanes_has_speed_limit"] = torch.tensor(
        d["route_lanes_has_speed_limit"], dtype=torch.bool
    ).unsqueeze(0)

    # polygons: (10, 40, 3) for v4
    data["polygons"] = torch.tensor(d["polygons"], dtype=torch.float32).unsqueeze(0)

    # line_strings: (60, 20, 4) for v4
    data["line_strings"] = torch.tensor(
        d["line_strings"], dtype=torch.float32
    ).unsqueeze(0)

    # goal_pose: (3,) -> (4,) cos/sin
    gp = d["goal_pose"]
    if gp.shape[-1] == 3:
        gp4 = np.array([gp[0], gp[1], np.cos(gp[2]), np.sin(gp[2])], dtype=np.float32)
    else:
        gp4 = gp
    data["goal_pose"] = torch.tensor(gp4, dtype=torch.float32).unsqueeze(0)

    # turn_indicators
    data["turn_indicators"] = torch.tensor(
        d["turn_indicators"], dtype=torch.int32
    ).unsqueeze(0)

    # ego_shape
    data["ego_shape"] = torch.tensor(d["ego_shape"], dtype=torch.float32).unsqueeze(0)

    # static_objects
    data["static_objects"] = torch.tensor(
        d["static_objects"], dtype=torch.float32
    ).unsqueeze(0)

    # Build sampled_trajectories (noise init)
    P = 1 + model_args.predicted_neighbor_num
    ego_current = data["ego_current_state"][:, :4]  # [1, 4]
    neighbors_current = data["neighbor_agents_past"][:, : P - 1, -1, :4]  # [1, P-1, 4]
    current_states = torch.cat(
        [ego_current[:, None], neighbors_current], dim=1
    )  # [1, P, 4]
    xT = current_states[:, :, None, :].expand(-1, -1, OUTPUT_T + 1, -1).clone()
    # Deterministic: zero noise
    data["sampled_trajectories"] = xT.reshape(1, P, -1)

    # delay: number of initial steps to keep fixed (v4 decoder requirement)
    data["delay"] = torch.zeros(1, dtype=torch.long)

    # Move to device
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(device)

    return data, d


def run_inference(model, data, guidance_composer=None, guidance_scale=0.0):
    """Run deterministic inference, optionally with guidance."""
    if guidance_composer is not None:
        model.decoder._guidance_fn = guidance_composer
        model.decoder._guidance_scale = guidance_scale
    else:
        model.decoder._guidance_fn = None
        model.decoder._guidance_scale = 0.0

    with torch.no_grad():
        _, decoder_output = model(data)

    ego_traj = decoder_output["prediction"][:, 0].cpu().numpy()  # [1, T, 4]
    return ego_traj[0]  # [T, 4]


def visualize_scene(
    npz_data,
    trajectories_dict,
    scene_name,
    save_path,
    view_range=30,
):
    """Visualize scene with road borders, GT, and model trajectories."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrow

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw lane boundaries (left + right from lane segment data)
    lanes = npz_data["lanes"]
    lane_lines = []
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        # Left boundary: center + left offset
        if lane.shape[1] > 7:
            lx = lane[:, 0] + lane[:, 4]
            ly = lane[:, 1] + lane[:, 5]
            rx = lane[:, 0] + lane[:, 6]
            ry = lane[:, 1] + lane[:, 7]
            lane_lines.append(np.column_stack([lx, ly]))
            lane_lines.append(np.column_stack([rx, ry]))
        else:
            # Fallback: just centerline
            ax.plot(lane[:, 0], lane[:, 1], color="lightgray", linewidth=0.5, alpha=0.4)
    if lane_lines:
        lc = LineCollection(lane_lines, colors="lightgray", alpha=0.4, linewidths=0.8)
        ax.add_collection(lc)

    # Draw route lanes (olive dashed, centerline)
    route = npz_data["route_lanes"]
    for i in range(route.shape[0]):
        r = route[i]
        if np.abs(r[:, :2]).sum() < 1e-6:
            continue
        ax.plot(r[:, 0], r[:, 1], color="olive", linewidth=1.5, linestyle="--", alpha=0.5)

    # Draw polygons (gray fill)
    polys = npz_data["polygons"]
    for i in range(polys.shape[0]):
        p = polys[i]
        if np.abs(p[:, :2]).sum() < 1e-6:
            continue
        ax.fill(p[:, 0], p[:, 1], color="lightgray", alpha=0.3, edgecolor="gray", linewidth=0.5)

    # Draw line_strings — road borders in RED, stop lines in orange
    ls = npz_data["line_strings"]
    has_types = ls.shape[-1] >= 4
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        if has_types:
            is_road_border = line[:, 3].max() > 0.5
            is_stop_line = line[:, 2].max() > 0.5
        else:
            is_road_border = False
            is_stop_line = False

        if is_road_border:
            ax.plot(line[:, 0], line[:, 1], color="red", linewidth=2.5, alpha=0.9, zorder=5)
        elif is_stop_line:
            ax.plot(line[:, 0], line[:, 1], color="orange", linewidth=1.5, alpha=0.7)

    # Draw ego car shape at t=0
    ego_shape = npz_data.get("ego_shape", None)
    if ego_shape is not None and len(ego_shape) >= 3:
        wb, length, width = ego_shape[0], ego_shape[1], ego_shape[2]
        rear_overhang = (length - wb) / 2
        corners = np.array([
            [-rear_overhang, -width/2],
            [length - rear_overhang, -width/2],
            [length - rear_overhang, width/2],
            [-rear_overhang, width/2],
            [-rear_overhang, -width/2],
        ])
        ax.fill(corners[:, 0], corners[:, 1], color="blue", alpha=0.6, zorder=12)
        ax.plot(corners[:, 0], corners[:, 1], color="darkblue", linewidth=1.5, zorder=12)
    else:
        ax.plot(0, 0, "bs", markersize=8, zorder=12)

    # Draw ego past
    ego_past = npz_data["ego_agent_past"]
    ax.plot(ego_past[:, 0], ego_past[:, 1], "b-", linewidth=1.5, alpha=0.4, label="Ego past")

    # Draw GT future
    ego_future = npz_data["ego_agent_future"]
    ax.plot(ego_future[:, 0], ego_future[:, 1], "g-", linewidth=2.5, alpha=0.8, label="GT future")
    ax.plot(ego_future[-1, 0], ego_future[-1, 1], "go", markersize=5, alpha=0.8)

    # Draw model trajectories
    colors = ["red", "deepskyblue", "orange", "magenta", "cyan"]
    for idx, (label, traj) in enumerate(trajectories_dict.items()):
        c = colors[idx % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2.5, alpha=0.85, label=label, zorder=10)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=c, markersize=5, zorder=11)

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_title(scene_name, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def compute_min_road_border_dist(npz_data, traj):
    """Compute min distance from trajectory to road borders."""
    ls = npz_data["line_strings"]
    if ls.shape[-1] < 4:
        return float("inf")

    # Get road border points
    border_mask = ls[..., 3] > 0.5  # (60, 20)
    coord_mask = np.linalg.norm(ls[..., :2], axis=-1) > 1e-3
    valid = border_mask & coord_mask
    border_pts = ls[valid, :2]  # (K, 2)

    if len(border_pts) == 0:
        return float("inf")

    # Trajectory points
    traj_xy = traj[:, :2]  # (T, 2)

    # Pairwise distance
    diff = traj_xy[:, None, :] - border_pts[None, :, :]  # (T, K, 2)
    dists = np.linalg.norm(diff, axis=-1)  # (T, K)
    return dists.min()


def main():
    print("Loading v4 model...")
    model, model_args = load_model(MODEL_PATH, DEVICE)
    print(f"  Model loaded on {DEVICE}")

    with open(PROB_LIST_MIRAIKAN) as f:
        miraikan_scenes = json.load(f)
    with open(PROB_LIST_TELEPORT) as f:
        teleport_scenes = json.load(f)
    print(f"  {len(miraikan_scenes)} miraikan, {len(teleport_scenes)} teleport scenes")

    # Build scan list: sample every 5th from each area
    scan_list = []
    for si in range(0, min(len(miraikan_scenes), 700), 5):
        scan_list.append(("M", si, miraikan_scenes[si]))
    for si in range(0, min(len(teleport_scenes), 500), 5):
        scan_list.append(("T", si, teleport_scenes[si]))

    results = []
    for idx, (area, si, npz_path) in enumerate(scan_list):
        scene_name = Path(npz_path).stem
        data, npz_data = load_npz_as_batch(npz_path, model_args, DEVICE)
        traj = run_inference(model, data, None, 0.0)
        min_d = compute_min_road_border_dist(npz_data, traj)
        results.append((area, si, scene_name, min_d, npz_path))
        if idx % 20 == 0:
            print(f"  [{area}] Scene {si}: min_dist={min_d:.3f} m")

    # Sort by distance
    results.sort(key=lambda r: r[3])

    print(f"\n=== Top 20 closest to road border ===")
    for area, si, name, d, path in results[:20]:
        tag = "*** OFFROAD" if d < 0.25 else "** CLOSE" if d < 0.5 else "* NEAR" if d < 1.0 else ""
        print(f"  [{area}] Scene {si}: {name}  min_dist={d:.3f} m {tag}")

    # Visualize top 20
    for area, si, scene_name, min_d, npz_path in results[:20]:
        data, npz_data = load_npz_as_batch(npz_path, model_args, DEVICE)
        traj = run_inference(model, data, None, 0.0)
        trajectories = {"Baseline (no guidance)": traj}
        save_path = os.path.join(SAVE_DIR, f"{area}_scene_{si:04d}_{scene_name}_d{min_d:.2f}.png")
        visualize_scene(npz_data, trajectories, f"[{area}] Scene {si}: {scene_name} (min_d={min_d:.2f}m)", save_path)

    print(f"\nAll images saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
