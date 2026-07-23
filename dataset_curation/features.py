from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _ego_speed_from_state(state: np.ndarray) -> float:
    return float(np.hypot(state[4], state[5]))


def _heading_from_past(past: np.ndarray) -> np.ndarray:
    return np.arctan2(past[:, 3], past[:, 2])


def _curvature(xy: np.ndarray) -> np.ndarray:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return np.abs(dx * ddy - dy * ddx) / denom


def extract_features(npz_path: str) -> dict[str, float]:
    d = np.load(npz_path, allow_pickle=True)

    state = d["ego_current_state"]  # (10,)
    past = d["ego_agent_past"]  # (31, 4)
    future = d["ego_agent_future"]  # (80, 3)
    nbr = d["neighbor_agents_past"]  # (N, 31, 11)
    lanes = d["lanes"]  # (140, 20, 33)
    lanes_sl = d["lanes_speed_limit"]  # (140, 1)
    route = d["route_lanes"]  # (25, 20, 33)
    goal = d["goal_pose"]  # (4,) or (3,)

    ego_speed = _ego_speed_from_state(state)
    ego_accel = float(np.hypot(state[6], state[7]))
    ego_yaw_rate = float(abs(state[9]))

    past_dx = np.diff(past[:, 0])
    past_dy = np.diff(past[:, 1])
    past_speeds = np.hypot(past_dx, past_dy) / 0.1
    nonzero_mask = past_speeds > 0.01

    fut_xy = future[:, :2]
    diffs = np.diff(fut_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    travel_distance = float(seg_lengths.sum())
    endpoint_disp = float(np.hypot(fut_xy[-1, 0] - fut_xy[0, 0], fut_xy[-1, 1] - fut_xy[0, 1]))
    headings = future[:, 2]
    heading_change = float(np.rad2deg(abs(headings[-1] - headings[0])))
    curv = _curvature(fut_xy)
    max_curvature = float(curv.max()) if len(curv) > 0 else 0.0
    path_straightness = float(endpoint_disp / max(travel_distance, 1e-6))

    active_mask = np.any(np.abs(nbr[:, -1, :2]) > 0.01, axis=-1)
    n_active = int(active_mask.sum())
    if n_active > 0:
        dists = np.hypot(nbr[active_mask, -1, 0], nbr[active_mask, -1, 1])
        closest_dist = float(dists.min())
        n_veh = float(nbr[active_mask, -1, 8].sum())
        n_ped = float(nbr[active_mask, -1, 9].sum())
        n_bike = float(nbr[active_mask, -1, 10].sum())
        total_typed = max(n_veh + n_ped + n_bike, 1.0)
        veh_ratio = n_veh / total_typed
        ped_ratio = n_ped / total_typed
        bike_ratio = n_bike / total_typed
    else:
        closest_dist = 999.0
        veh_ratio = ped_ratio = bike_ratio = 0.0

    lane_active = np.any(np.abs(lanes[:, :, :2]) > 0.01, axis=(1, 2))
    active_lanes = lanes[lane_active]
    if len(active_lanes) > 0:
        lane_dirs = active_lanes[:, :, 2:4]
        lane_headings = np.arctan2(lane_dirs[:, :, 1], lane_dirs[:, :, 0])
        per_seg_curvature = np.std(lane_headings, axis=1)
        mean_lane_curv = float(per_seg_curvature.mean())
    else:
        mean_lane_curv = 0.0

    tl_active = lanes[:, :, 8:13]
    has_tl = np.any(tl_active[:, :, :3] > 0.5, axis=(1, 2))
    n_tl_segments = float(has_tl.sum())

    active_sl_mask = lanes_sl[:, 0] > 0.01
    if active_sl_mask.any():
        sl_vals = lanes_sl[active_sl_mask, 0]
        sl_mean = float(sl_vals.mean())
        sl_std = float(sl_vals.std())
    else:
        sl_mean = sl_std = 0.0

    goal_dist = float(np.hypot(goal[0], goal[1]))

    route_active = np.any(np.abs(route[:, :, :2]) > 0.01, axis=(1, 2))
    active_route = route[route_active]
    if len(active_route) > 0:
        route_pts = active_route[:, :, :2].reshape(-1, 2)
        nonzero_pts = route_pts[np.any(np.abs(route_pts) > 0.01, axis=1)]
        if len(nonzero_pts) > 2:
            rd = np.diff(nonzero_pts, axis=0)
            route_length_val = float(np.hypot(rd[:, 0], rd[:, 1]).sum())
            route_h = np.arctan2(rd[:, 1], rd[:, 0])
            route_curv = float(np.abs(np.diff(route_h)).sum())
        else:
            route_length_val = route_curv = 0.0
    else:
        route_length_val = route_curv = 0.0

    return {
        "ego_speed": ego_speed,
        "ego_accel": ego_accel,
        "ego_yaw_rate": ego_yaw_rate,
        "ego_speed_past_mean": float(past_speeds[nonzero_mask].mean())
        if nonzero_mask.any()
        else 0.0,
        "ego_speed_past_std": float(past_speeds[nonzero_mask].std()) if nonzero_mask.any() else 0.0,
        "ego_speed_past_max": float(past_speeds.max()),
        "ego_accel_past_mean": float(np.abs(np.diff(past_speeds)).mean()),
        "ego_accel_past_std": float(np.abs(np.diff(past_speeds)).std()),
        "travel_distance": travel_distance,
        "endpoint_displacement": endpoint_disp,
        "heading_change_deg": heading_change,
        "max_curvature": max_curvature,
        "path_straightness": path_straightness,
        "n_active_neighbors": float(n_active),
        "closest_neighbor_dist": closest_dist,
        "neighbor_vehicle_ratio": veh_ratio,
        "neighbor_ped_ratio": ped_ratio,
        "neighbor_bike_ratio": bike_ratio,
        "mean_lane_curvature": mean_lane_curv,
        "n_traffic_light_segments": n_tl_segments,
        "speed_limit_mean": sl_mean,
        "speed_limit_std": sl_std,
        "goal_distance": goal_dist,
        "route_curvature": route_curv,
        "route_length": route_length_val,
    }


def extract_features_batch(scene_paths: list[str], n_workers: int = 4) -> pd.DataFrame:
    if n_workers <= 1:
        rows = {p: extract_features(p) for p in scene_paths}
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(extract_features, scene_paths))
        rows = dict(zip(scene_paths, results))
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "npz_path"
    return df


def _plot_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_cols = 5
    n_features = len(df.columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], kde=True, bins=50)
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_histograms.png", dpi=150)
    plt.close(fig)

    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, ax=ax2, cmap="coolwarm", center=0, annot=False, xticklabels=True, yticklabels=True
    )
    ax2.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig2.savefig(output_dir / "feature_correlation.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Extract hand-crafted features from NPZ scenes")
    parser.add_argument("--scenes", required=True, help="JSON file with list of NPZ paths")
    parser.add_argument("--output_dir", required=True, help="Directory to write results")
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"Extracting features from {len(scene_paths)} scenes...")
    df = extract_features_batch(scene_paths, n_workers=args.n_workers)
    df.to_parquet(output_dir / "features.parquet")

    print("Generating visualizations...")
    _plot_histograms(df, output_dir)

    print(f"Summary statistics:")
    print(df.describe().to_string())
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
