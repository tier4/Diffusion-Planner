"""Visualize the GRPO neighbor-collision check for a single data sample.

The GRPO collision reward (``grpo_utils.compute_collision_reward`` ->
``loss.compute_neighbor_collision_penalty``) only evaluates the ego/neighbor distance at a
handful of timesteps -- ``loss._NEIGHBOR_EVAL_STEPS = [0, 20, 40, 60, 79]``. This script picks
one scene, draws a *single* sampled ego trajectory, and renders one panel per eval step so you
can see exactly how the ego bounding box moves and where/when the collision check fires.

Each panel (one eval step ``t``) shows, in the same geometry the penalty uses:
  * every sampled ego bounding box (``compute_ego_bbox_corners``), one color per sample,
    with a red fill on the ones the collision penalty flags at that step,
  * every valid neighbor box at step ``t`` (``center_rect_to_points``), injected colliders in
    magenta,
  * the per-sample neighbor-collision penalty (proximity hinge + SAT overlap) GRPO turns into
    reward at that step.

Example:
    python3 visualize_collision_steps.py \
        --resume_model_path /path/to/best_model.pth \
        --data_list /path/to/path_list_train.json \
        --scene_index 7 --output_path collision_steps.png
"""

import argparse

import numpy as np
import torch
from diffusion_planner.grpo_epoch import _neighbor_future_world
from diffusion_planner.grpo_utils import (
    compute_collision_reward,
    expand_batch,
    sample_group,
)
from diffusion_planner.loss import (
    _NEIGHBOR_EVAL_STEPS,
    compute_ego_bbox_corners,
)
from diffusion_planner.model.guidance.collision import center_rect_to_points
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.neighbor_db import (
    DEFAULT_NEIGHBOR_DB_PATH,
    NeighborPatternDB,
)
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector
from torch.utils.data import default_collate

# reuse the loaders / road drawing from the sample-group visualizer.
from visualize_grpo_samples import (
    boolean,
    build_train_args,
    draw_road,
    figure_legend,
    gray,
    load_model,
)

from scenario_generation.canvas import Figure
from scenario_generation.colormaps import map_values, qualitative


def parse_viz_args():
    p = argparse.ArgumentParser(
        description="Visualize the GRPO neighbor-collision check at the eval steps"
    )
    p.add_argument(
        "--resume_model_path",
        type=str,
        required=True,
        help="path to the GRPO/SFT checkpoint (.pth) to visualize",
    )
    p.add_argument(
        "--data_list",
        type=str,
        required=True,
        help="path to a dataset path-list JSON (e.g. path_list_train.json)",
    )
    p.add_argument(
        "--scene_index",
        type=int,
        default=-1,
        help="dataset index to visualize; <0 picks one at random from --seed",
    )
    p.add_argument("--grpo_noise_scale", type=float, default=3.0)
    p.add_argument(
        "--aug_mode",
        type=str,
        default="db",
        choices=["db", "synthetic", "none"],
        help="neighbor augmentation: DB copy-paste (default) / synthetic colliders / none",
    )
    p.add_argument(
        "--neighbor_db_path",
        type=str,
        default=DEFAULT_NEIGHBOR_DB_PATH,
        help="(db) real-neighbor pattern DB to copy-paste colliding tracks from",
    )
    p.add_argument("--neighbor_db_collision_margin", type=float, default=4.0)
    p.add_argument("--neighbor_min_collision_time", type=float, default=0.8)
    p.add_argument("--neighbor_search_subsample", type=int, default=0)
    p.add_argument("--neighbor_inject_max", type=int, default=1)
    p.add_argument("--neighbor_inject_prob", type=float, default=1.0)
    p.add_argument("--pedestrian_prob", type=float, default=0.3)
    p.add_argument("--bicycle_prob", type=float, default=0.2)
    p.add_argument("--keep_clear_radius", type=float, default=6.0)
    p.add_argument(
        "--collider_straight_line",
        type=boolean,
        default=True,
        help="constant-velocity straight colliders (True) vs curved constant-accel (False)",
    )
    p.add_argument("--show_road", type=boolean, default=True)
    p.add_argument("--use_ema", type=boolean, default=False)
    p.add_argument("--output_path", type=str, default="collision_steps.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--zoom_margin",
        type=float,
        default=12.0,
        help="metres of padding around the ego boxes in each panel",
    )
    p.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="N ego trajectories to draw per eval step (the GRPO group size)",
    )
    return p.parse_known_args()


# neighbor past columns (see loss.py): width=6, length=7.
_N_WIDTH, _N_LENGTH = 6, 7


def _poly(corners):
    """Close a [4, 2] corner array into a [5, 2] polyline."""
    return np.concatenate([corners, corners[:1]], axis=0)


@torch.no_grad()
def main():
    v, _ = parse_viz_args()
    torch.manual_seed(v.seed)
    np.random.seed(v.seed)
    device = v.device

    args = build_train_args(v)
    margin = (
        f"veh={args.neighbor_collision_margin_vehicle} "
        f"ped={args.neighbor_collision_margin_pedestrian} "
        f"bike={args.neighbor_collision_margin_bicycle}"
    )
    model = load_model(args, v.resume_model_path, v.use_ema, device)
    print(f"Model loaded from {v.resume_model_path} (ema={v.use_ema})")

    dataset = DiffusionPlannerData(v.data_list)
    idx = (
        v.scene_index
        if v.scene_index >= 0
        else int(np.random.default_rng(v.seed).integers(len(dataset)))
    )
    raw = default_collate([dataset[idx]])
    raw = {k: val.to(device) for k, val in raw.items()}
    print(f"Scene index: {idx}")

    injected_mask = np.zeros(raw["neighbor_agents_past"].shape[1], dtype=bool)
    if v.aug_mode != "none":
        if v.aug_mode == "db":
            injector = NeighborPatternDB(
                db_path=v.neighbor_db_path,
                collision_margin=v.neighbor_db_collision_margin,
                keep_clear_radius=v.keep_clear_radius,
                min_collision_time=v.neighbor_min_collision_time,
                search_subsample=v.neighbor_search_subsample,
            )
        else:
            injector = SyntheticColliderInjector(
                pedestrian_prob=v.pedestrian_prob,
                bicycle_prob=v.bicycle_prob,
                keep_clear_radius=v.keep_clear_radius,
                straight_line=v.collider_straight_line,
            )
        raw = injector.inject(raw, v.neighbor_inject_max, v.neighbor_inject_prob)
        injected_mask = injector.last_injected_mask.cpu().numpy()[0]
        print(f"Augmentation '{v.aug_mode}' applied; injected {int(injected_mask.sum())} neighbors")

    raw_np = {k: val.detach().cpu().numpy() for k, val in raw.items()}

    # --- exact GRPO sampling path: a group of N samples for the one scene ---
    n = v.num_generations
    exp = expand_batch(raw, n)  # replicate the scene N times
    exp["ego_agent_past"] = heading_to_cos_sin(exp["ego_agent_past"])
    exp["goal_pose"] = heading_to_cos_sin(exp["goal_pose"])
    neighbors_future, neighbor_future_mask = _neighbor_future_world(exp["neighbor_agents_future"])
    neighbors_future_valid = ~neighbor_future_mask
    norm_exp = args.observation_normalizer(exp)
    ego_world = sample_group(model, norm_exp, v.grpo_noise_scale, device)  # [N, T, 4]

    reward, nc_penalty, _ = compute_collision_reward(
        ego_world, norm_exp, neighbors_future, neighbors_future_valid, args
    )
    reward = reward.cpu().numpy()  # [N]
    nc_penalty = nc_penalty.cpu().numpy()  # [N, T]
    T = ego_world.shape[1]

    # --- geometry (same as the collision penalty), per sample ---
    ego_shape = norm_exp["ego_shape"]
    ego_corners = compute_ego_bbox_corners(ego_world, ego_shape).cpu().numpy()  # [N, T, 4, 2]
    ego_xy = ego_world[:, :, :2].cpu().numpy()  # [N, T, 2]

    # neighbors are identical across the group -> use the scene's (row 0) for drawing.
    denorm = args.observation_normalizer.inverse(norm_exp)
    neigh_past = denorm["neighbor_agents_past"][0]  # [Pn, T_past, D]
    Pn = neighbors_future.shape[1]
    neigh_width = torch.clamp(neigh_past[:Pn, -1, _N_WIDTH], min=1e-3)  # [Pn]
    neigh_length = torch.clamp(neigh_past[:Pn, -1, _N_LENGTH], min=1e-3)  # [Pn]

    steps = [s for s in _NEIGHBOR_EVAL_STEPS if s < T]
    P = len(steps)
    scolor = (
        (lambda i: qualitative("tab10", i % 10))
        if n <= 10
        else (lambda i: map_values(i / max(n - 1, 1), "turbo", 0.0, 1.0))
    )

    cols = min(P, 5)
    rows = int(np.ceil(P / cols))
    fig = Figure((5.6 * cols, 5.6 * rows), dpi=120)
    grid = fig.grid(rows, cols, margins=(0.05, 0.98, 0.04, 0.86), wspace=0.2, hspace=0.3)
    axes = [panel for row in grid for panel in row]

    for j, t in enumerate(steps):
        ax = axes[j]
        if v.show_road:
            draw_road(ax, raw_np, 0)

        # neighbor boxes at step t (same rect the penalty builds) -- shared by all samples.
        pos = neighbors_future[0, :, t, :2]
        cos = neighbors_future[0, :, t, 2]
        sin = neighbors_future[0, :, t, 3]
        norm_cs = torch.sqrt(cos**2 + sin**2).clamp_min(1e-6)
        rect = torch.stack(
            [pos[:, 0], pos[:, 1], cos / norm_cs, sin / norm_cs, neigh_length, neigh_width], dim=-1
        )  # [Pn, 6]
        neigh_corners = center_rect_to_points(rect)  # [Pn, 4, 2]
        neigh_corners_np = neigh_corners.cpu().numpy()
        valid_t = neighbors_future_valid[0, :, t]  # [Pn]
        valid_np = valid_t.cpu().numpy()

        for p in range(Pn):
            if not valid_np[p]:
                continue
            is_inj = injected_mask[p] if p < len(injected_mask) else False
            c, lw = ("#ff00ff", 1.6) if is_inj else (gray(0.4), 1.0)
            ax.polygons(
                [neigh_corners_np[p]],
                fill=c if is_inj else None,
                edge=c,
                width=lw,
                alpha=0.25 if is_inj else 0.7,
                z=3.5,
            )

        # N ego boxes at step t, one color per sample; collided samples get a red dashed overlay.
        collide_list = []
        for i in range(n):
            ec = ego_corners[i, t]
            color = scolor(i)
            pen_i = float(nc_penalty[i, t])
            collided = pen_i > 0.0
            ax.polygons([ec], edge=color, width=2.2 if collided else 1.4, alpha=0.95, z=5)
            if collided:
                ax.polygons([ec], fill="#ff0000", edge="#ff0000", width=1.0, alpha=0.18, z=4.5)
                collide_list.append(f"s{i}(p={pen_i:.2f})")
            ax.text(
                tuple(ec.mean(0)), str(i), 7, color, ha="center", va="center", weight="bold", z=6
            )
            ax.lines([ego_xy[i]], color, 0.8, 0.35, z=3)

        k_coll = len(collide_list)
        ax.title(
            f"step {t}   collisions {k_coll}/{n} (margin={margin})\n"
            + ("collide: " + ", ".join(collide_list) if collide_list else "all clear"),
            9,
            color="red" if k_coll else "green",
        )
        ax.equal = True
        ax.grid(0.3)

        # zoom to cover all N ego boxes (the colliding neighbor sits within margin, so in-frame).
        ego_pts_t = ego_corners[:, t].reshape(-1, 2)  # [N*4, 2]
        ctr = ego_pts_t.mean(0)
        half = max(v.zoom_margin, np.abs(ego_pts_t - ctr).max() + 2.0)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)

    for ax in axes[P:]:
        ax.axis_off()

    # legend: one entry per sample (with its scalar reward) + neighbor proxies.
    sample_entries = [(f"s{i} r={reward[i]:.1f}", scolor(i), 2, None) for i in range(n)]
    proxies = [
        ("collision (red fill)", "#ff0000", 2, None),
        ("injected collider", "#ff00ff", 1.6, None),
        ("scene neighbor", gray(0.4), 1.0, None),
    ]
    entries = sample_entries + proxies
    ncol = min(n + 3, 11)
    legend_height = int(np.ceil(len(entries) / ncol)) * 0.3 / (5.6 * rows)
    legend_width = min(0.96, ncol * 1.6 / (5.6 * cols))
    figure_legend(
        fig,
        (0.5 - legend_width / 2, 0.955 - legend_height, legend_width, legend_height),
        entries,
        ncol,
    )
    total_pen = float(nc_penalty.sum())
    fig.suptitle(
        f"GRPO collision check @ eval steps {steps}  |  scene #{idx}, N={n}  "
        f"mean reward={reward.mean():.2f}, best={reward.max():.2f}  total penalty={total_pen:.2f}",
        12,
    )
    fig.save(v.output_path)
    print(f"Saved {P} eval-step panels ({n} samples each) to {v.output_path}")
    n_collide = [int((nc_penalty[:, s] > 0).sum()) for s in steps]
    print(f"  collisions per eval step {steps}: {n_collide}  (out of N={n})")
    print(f"  per-sample reward: {[round(float(r), 2) for r in reward]}")
    print(
        f"  mean reward={reward.mean():.3f}  best={reward.max():.3f}  total penalty={total_pen:.3f}"
    )


if __name__ == "__main__":
    main()
