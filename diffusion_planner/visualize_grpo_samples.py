"""Visualize the diversity of the GRPO sample group.

Reproduces the exact GRPO sampling path used in training (``grpo_epoch._grpo_step`` /
``grpo_utils.sample_group``):

  1. Pick a few scenes from a data list.
  2. (optionally) augment each scene with synthetic adversarial neighbors that are guaranteed
     to collide with the ego GT -- i.e. expand the training data the way GRPO does.
  3. Replicate each scene ``N = num_generations`` times and draw ``N`` ego trajectories in a
     single multi-batch inference pass (random initial diffusion noise -> diverse samples).
  4. Plot, per scene, the ``N`` sampled ego trajectories overlaid on the (augmented) scene
     context (neighbor boxes + futures, GT ego future), and annotate a diversity metric.

This shows how much the policy spreads its ``N`` samples per scene -- the signal GRPO turns
into group-relative advantages.

Example (8 samples, 12 scenes, with synthetic collider augmentation):
    python3 visualize_grpo_samples.py \
        --resume_model_path /path/to/best_model.pth \
        --num_scenes 12 --num_generations 8 --output_path grpo_samples.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from diffusion_planner.grpo_utils import expand_batch, sample_group  # noqa: E402
from diffusion_planner.model.diffusion_planner import Diffusion_Planner  # noqa: E402
from diffusion_planner.train_epoch import heading_to_cos_sin  # noqa: E402
from diffusion_planner.utils.dataset import DiffusionPlannerData  # noqa: E402
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector  # noqa: E402
from diffusion_planner.utils.visualize_input import (  # noqa: E402
    draw_lanes,
    draw_polygons_and_lines,
    draw_route,
    draw_static_objects,
)

# Column layout of a neighbor past row (see loss.py).
_PAST_X, _PAST_Y, _PAST_COS, _PAST_SIN = 0, 1, 2, 3
_PAST_WIDTH, _PAST_LENGTH = 6, 7

_DEFAULT_CKPT = (
    "/mnt/nvme/training_result/"
    "20260503-220950_with_takanawa_16days_weak_smoothing_epoch0060_epoch0080/best_model.pth"
)
_DEFAULT_DATA = "/mnt/nvme/dataset/basic_dataset/path_list_train.json"


def boolean(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_viz_args():
    p = argparse.ArgumentParser(description="Visualize GRPO sample-group diversity")
    p.add_argument("--resume_model_path", type=str, default=_DEFAULT_CKPT)
    p.add_argument("--data_list", type=str, default=_DEFAULT_DATA)
    p.add_argument("--num_scenes", type=int, default=12, help="scenes to visualize")
    p.add_argument("--num_generations", type=int, default=8, help="N samples per scene")
    p.add_argument("--show_road", type=boolean, default=True,
                   help="draw lanes / route / road-borders / static objects")
    p.add_argument("--show_footprint", type=boolean, default=True,
                   help="draw the ego bounding-box footprint along the trajectories")
    p.add_argument("--footprint_stride", type=int, default=20,
                   help="draw a footprint every this many trajectory steps")
    p.add_argument("--grpo_noise_scale", type=float, default=1.0)
    p.add_argument("--aug_mode", type=str, default="synthetic",
                   choices=["synthetic", "none"],
                   help="neighbor augmentation: synthetic colliders / none")
    p.add_argument("--neighbor_inject_max", type=int, default=5)
    p.add_argument("--neighbor_inject_prob", type=float, default=1.0)
    p.add_argument("--pedestrian_prob", type=float, default=0.3,
                   help="(synthetic) fraction of injected colliders that are pedestrians")
    p.add_argument("--bicycle_prob", type=float, default=0.2,
                   help="(synthetic) fraction of injected colliders that are bicycles")
    p.add_argument("--keep_clear_radius", type=float, default=3.0,
                   help="(synthetic) min distance the collider path keeps from the ego t=0 pose")
    p.add_argument("--use_ema", type=boolean, default=False,
                   help="load EMA weights instead of the raw policy")
    p.add_argument("--output_path", type=str, default="grpo_samples.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--zoom_margin", type=float, default=15.0,
                   help="metres of padding around the ego samples+GT; <=0 disables zoom "
                        "(autoscale to full scene incl. far neighbors)")
    return p.parse_known_args()


def build_train_args(v):
    """Build the full training ``args`` (model dims + normalizers) via the trainer's get_args."""
    import sys

    from train_grpo_predictor import get_args

    saved = sys.argv
    sys.argv = [
        "viz",
        "--exp_name", "viz",
        "--save_dir", "/tmp/grpo_viz",
        "--train_set_list", v.data_list,
        "--valid_set_list", v.data_list,
        "--resume_model_path", v.resume_model_path,
        "--diffusion_model_type", "x_start",
        "--num_generations", str(v.num_generations),
        "--grpo_noise_scale", str(v.grpo_noise_scale),
        "--ddp", "False",
        "--device", v.device,
    ]
    try:
        return get_args()
    finally:
        sys.argv = saved


def load_model(args, ckpt_path, use_ema, device):
    model = Diffusion_Planner(args).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    key = "ema_state_dict" if use_ema else "model"
    state = ckpt[key] if key in ckpt else ckpt.get("model", ckpt)
    # checkpoints are saved from a DDP-wrapped model -> strip the "module." prefix.
    state = {k[len("module."):] if k.startswith("module.") else k: val for k, val in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={len(missing)} unexpected={len(unexpected)} "
              f"(first missing: {missing[:1]}, first unexpected: {unexpected[:1]})")
    model.eval()
    return model


def select_batch(data_list, num_scenes, seed, device):
    dataset = DiffusionPlannerData(data_list)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=num_scenes, replace=False)
    batch = default_collate([dataset[int(i)] for i in idx])
    return {k: v.to(device) for k, v in batch.items()}, idx


def _nonzero_rows(xy):
    return xy[np.any(xy != 0.0, axis=-1)]


def _bbox_corners(cx, cy, cos, sin, length, width):
    norm = np.hypot(cos, sin)
    cos, sin = (1.0, 0.0) if norm < 1e-6 else (cos / norm, sin / norm)
    hl, hw = max(length, 0.1) / 2.0, max(width, 0.1) / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
    rot = np.array([[cos, -sin], [sin, cos]])
    return local @ rot.T + np.array([cx, cy])


def draw_road(ax, raw_np, s):
    """Draw lanes / route / road-borders / static objects for scene ``s``.

    Reuses the trainer's input visualizers (diffusion_planner.utils.visualize_input), which
    expect a single-scene dict indexed as ``inputs[key][0]`` in raw npz format.
    """
    scene = {k: raw_np[k][s:s + 1] for k in
             ("lanes", "route_lanes", "line_strings", "polygons", "static_objects")
             if k in raw_np}
    for fn in (draw_lanes, draw_route, draw_polygons_and_lines, draw_static_objects):
        try:
            fn(ax, scene)
        except Exception as e:  # noqa: BLE001 - a missing/odd map element shouldn't kill the plot
            print(f"[draw_road] {fn.__name__} skipped for scene {s}: {e}")


def draw_footprints(ax, traj_xyhead, length, width, color, stride, alpha=0.35, lw=0.8):
    """Draw the ego bounding box along a trajectory at fixed step intervals.

    traj_xyhead: [T, 4] (x, y, cos, sin). Padding rows (x==y==0) are skipped.
    """
    T = traj_xyhead.shape[0]
    for t in range(0, T, max(stride, 1)):
        x, y, cos, sin = traj_xyhead[t]
        if x == 0.0 and y == 0.0:
            continue
        corners = _bbox_corners(x, y, cos, sin, length, width)
        ax.plot(corners[:, 0], corners[:, 1], "-", color=color, lw=lw, alpha=alpha, zorder=3)


def diversity_metrics(samples_xy):
    """samples_xy: [N, T, 2]. Returns (endpoint_spread, mean_path_spread) in metres.

    endpoint_spread: mean pairwise distance between the N final positions.
    mean_path_spread: average over time of the mean pairwise distance between positions.
    """
    n = samples_xy.shape[0]
    if n < 2:
        return 0.0, 0.0
    iu = np.triu_indices(n, k=1)
    endpoints = samples_xy[:, -1, :]  # [N, 2]
    ep = np.linalg.norm(endpoints[iu[0]] - endpoints[iu[1]], axis=-1).mean()
    # per-timestep mean pairwise distance, averaged over time
    diff = samples_xy[iu[0]] - samples_xy[iu[1]]  # [pairs, T, 2]
    path = np.linalg.norm(diff, axis=-1).mean()
    return float(ep), float(path)


@torch.no_grad()
def main():
    v, _ = parse_viz_args()
    torch.manual_seed(v.seed)
    np.random.seed(v.seed)
    device = v.device
    n = v.num_generations

    args = build_train_args(v)
    model = load_model(args, v.resume_model_path, v.use_ema, device)
    print(f"Model loaded from {v.resume_model_path} (ema={v.use_ema})")

    raw, idx = select_batch(v.data_list, v.num_scenes, v.seed, device)
    S = v.num_scenes

    def empty_slots(t):  # [S, Pn] True where the neighbor past track is all-zero padding
        return (t["neighbor_agents_past"] != 0.0).any(dim=(2, 3)).logical_not().cpu().numpy()

    injected_mask = np.zeros((S, raw["neighbor_agents_past"].shape[1]), dtype=bool)
    if v.aug_mode != "none":
        injector = SyntheticColliderInjector(
            pedestrian_prob=v.pedestrian_prob, bicycle_prob=v.bicycle_prob,
            keep_clear_radius=v.keep_clear_radius)
        before = empty_slots(raw)
        raw = injector.inject(raw, v.neighbor_inject_max, v.neighbor_inject_prob)
        # slots that were empty before and are filled after == augmentation-added
        injected_mask = before & ~empty_slots(raw)
        print(f"Augmentation '{v.aug_mode}' applied; "
              f"injected {int(injected_mask.sum())} neighbors across {S} scenes")

    # snapshot the (augmented) scene context for plotting, in raw metres / ego frame.
    raw_np = {k: v.detach().cpu().numpy() for k, v in raw.items()}
    neigh_past = raw_np["neighbor_agents_past"]      # [S, Pn, 31, 11]
    neigh_future = raw_np["neighbor_agents_future"]  # [S, Pn, 80, 3]
    ego_future_gt = raw_np["ego_agent_future"]       # [S, T, 3]
    ego_shape = raw_np["ego_shape"]                  # [S, 3] -> [_, length, width]

    # --- exact GRPO sampling path ---
    exp = expand_batch(raw, n)
    exp["ego_agent_past"] = heading_to_cos_sin(exp["ego_agent_past"])
    exp["goal_pose"] = heading_to_cos_sin(exp["goal_pose"])
    norm_exp = args.observation_normalizer(exp)
    ego_world = sample_group(model, norm_exp, v.grpo_noise_scale, device)  # [S*N, T, 4]
    ego_samples = ego_world.view(S, n, ego_world.shape[1], 4).cpu().numpy()  # [S,N,T,4] (x,y,cos,sin)

    # --- plot ---
    cols = min(4, S)
    rows = int(np.ceil(S / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("turbo")

    for s in range(S):
        ax = axes[s]
        # road context (lanes / route / road-borders / static objects)
        if v.show_road:
            draw_road(ax, raw_np, s)
        # neighbors (augmented scene): DB-injected ones are highlighted in magenta.
        Pn = neigh_past.shape[1]
        for p in range(Pn):
            past = neigh_past[s, p]
            if not np.any(past != 0.0):
                continue
            is_injected = injected_mask[s, p]
            box_color = "magenta" if is_injected else "0.5"
            fut_color = "magenta" if is_injected else "0.75"
            box_lw = 1.4 if is_injected else 0.8
            cur = past[-1]
            corners = _bbox_corners(cur[_PAST_X], cur[_PAST_Y], cur[_PAST_COS], cur[_PAST_SIN],
                                    cur[_PAST_LENGTH], cur[_PAST_WIDTH])
            ax.plot(corners[:, 0], corners[:, 1], "-", color=box_color, lw=box_lw,
                    alpha=0.9 if is_injected else 0.7, zorder=3.5 if is_injected else 2)
            fut = _nonzero_rows(neigh_future[s, p, :, :2])
            if fut.shape[0] > 0:
                ax.plot(fut[:, 0], fut[:, 1], "-", color=fut_color,
                        lw=1.2 if is_injected else 0.8, alpha=0.9 if is_injected else 0.6,
                        zorder=3.5 if is_injected else 2)

        # ego length/width for the footprint boxes
        ego_len, ego_wid = float(ego_shape[s, 1]), float(ego_shape[s, 2])

        # GT ego future (+ footprints)
        gt = _nonzero_rows(ego_future_gt[s, :, :2])
        if gt.shape[0] > 0:
            ax.plot(gt[:, 0], gt[:, 1], "k--", lw=2.0, label="GT ego", zorder=5)
        if v.show_footprint:
            gt_full = ego_future_gt[s]  # [T, 3] (x, y, heading)
            gt_xyh = np.stack([gt_full[:, 0], gt_full[:, 1],
                               np.cos(gt_full[:, 2]), np.sin(gt_full[:, 2])], axis=-1)
            draw_footprints(ax, gt_xyh, ego_len, ego_wid, "black", v.footprint_stride, alpha=0.3)

        # N sampled ego trajectories (+ footprints)
        for i in range(n):
            traj = ego_samples[s, i]  # [T, 4] (x, y, cos, sin)
            color = cmap(i / max(n - 1, 1))
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color,
                    lw=1.5, alpha=0.85, label=f"sample {i}" if s == 0 else None, zorder=4)
            if v.show_footprint:
                draw_footprints(ax, traj, ego_len, ego_wid, color, v.footprint_stride, alpha=0.3)

        # current ego box at the origin (its actual shape and heading)
        ego_cur = raw_np["ego_current_state"][s]  # [10]: x, y, cos, sin, ...
        ego_box = _bbox_corners(ego_cur[0], ego_cur[1], ego_cur[2], ego_cur[3], ego_len, ego_wid)
        ax.plot(ego_box[:, 0], ego_box[:, 1], "-", color="black", lw=2.0, zorder=6)
        ax.plot(0, 0, "k*", ms=10, zorder=6)
        ep, path = diversity_metrics(ego_samples[s, :, :, :2])
        ax.set_title(f"scene #{idx[s]}  endpoint_spread={ep:.2f}m  path_spread={path:.2f}m",
                     fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        # zoom to the ego region so the sample diversity is visible (far neighbors clipped).
        if v.zoom_margin > 0:
            focus = ego_samples[s, :, :, :2].reshape(-1, 2)  # [N*T, 2]
            gt_pts = _nonzero_rows(ego_future_gt[s, :, :2])
            if gt_pts.shape[0] > 0:
                focus = np.concatenate([focus, gt_pts], axis=0)
            focus = np.concatenate([focus, np.zeros((1, 2))], axis=0)  # include ego origin
            lo = focus.min(axis=0) - v.zoom_margin
            hi = focus.max(axis=0) + v.zoom_margin
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])

    for ax in axes[S:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    proxies = [
        plt.Line2D([0], [0], color="magenta", lw=1.4, label="injected collider"),
        plt.Line2D([0], [0], color="0.5", lw=0.8, label="scene neighbor"),
    ]
    fig.legend(handles + proxies, labels + [p.get_label() for p in proxies],
               loc="upper right", fontsize=8, ncol=2)
    all_ep = np.mean([diversity_metrics(ego_samples[s, :, :, :2])[0] for s in range(S)])
    fig.suptitle(
        f"GRPO sample-group diversity: N={n} per scene, noise_scale={v.grpo_noise_scale}, "
        f"aug={v.aug_mode}  |  mean endpoint_spread={all_ep:.2f}m",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(v.output_path, dpi=120)
    print(f"Saved {S} scenes to {v.output_path}  (mean endpoint_spread={all_ep:.2f} m)")


if __name__ == "__main__":
    main()
