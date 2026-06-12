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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusion_planner.grpo_epoch import _neighbor_future_world  # noqa: E402
from diffusion_planner.grpo_utils import (  # noqa: E402
    compute_collision_reward,
    expand_batch,
    sample_group,
)
from diffusion_planner.loss import (  # noqa: E402
    _NEIGHBOR_EVAL_STEPS,
    compute_ego_bbox_corners,
)
from diffusion_planner.model.guidance.collision import center_rect_to_points  # noqa: E402
from diffusion_planner.train_epoch import heading_to_cos_sin  # noqa: E402
from diffusion_planner.utils.dataset import DiffusionPlannerData  # noqa: E402
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

# reuse the loaders / road drawing from the sample-group visualizer.
from visualize_grpo_samples import (  # noqa: E402
    boolean,
    build_train_args,
    draw_road,
    load_model,
)


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
        default="synthetic",
        choices=["synthetic", "none"],
        help="neighbor augmentation: synthetic colliders / none",
    )
    p.add_argument("--neighbor_inject_max", type=int, default=1)
    p.add_argument("--neighbor_inject_prob", type=float, default=1.0)
    p.add_argument("--pedestrian_prob", type=float, default=0.3)
    p.add_argument("--bicycle_prob", type=float, default=0.2)
    p.add_argument("--keep_clear_radius", type=float, default=3.0)
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
    margin = args.neighbor_collision_margin
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
        injector = SyntheticColliderInjector(
            pedestrian_prob=v.pedestrian_prob,
            bicycle_prob=v.bicycle_prob,
            keep_clear_radius=v.keep_clear_radius,
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
    base_cmap = plt.get_cmap("tab10" if n <= 10 else "turbo")
    scolor = (lambda i: base_cmap(i % 10)) if n <= 10 else (lambda i: base_cmap(i / max(n - 1, 1)))

    cols = min(P, 5)
    rows = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 5.6 * rows))
    axes = np.atleast_1d(axes).ravel()

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
            c, lw = ("magenta", 1.6) if is_inj else ("0.4", 1.0)
            ax.add_patch(
                Polygon(
                    neigh_corners_np[p],
                    closed=True,
                    fill=is_inj,
                    facecolor=(c if is_inj else "none"),
                    edgecolor=c,
                    lw=lw,
                    alpha=0.25 if is_inj else 0.7,
                    zorder=3.5,
                )
            )

        # N ego boxes at step t, one color per sample; collided samples get a red dashed overlay.
        collide_list = []
        for i in range(n):
            ec = ego_corners[i, t]
            color = scolor(i)
            pen_i = float(nc_penalty[i, t])
            collided = pen_i > 0.0
            ax.add_patch(
                Polygon(
                    ec,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    lw=2.2 if collided else 1.4,
                    alpha=0.95,
                    zorder=5,
                )
            )
            if collided:
                ax.add_patch(
                    Polygon(
                        ec,
                        closed=True,
                        fill=True,
                        facecolor="red",
                        edgecolor="red",
                        lw=1.0,
                        alpha=0.18,
                        zorder=4.5,
                    )
                )
                collide_list.append(f"s{i}(p={pen_i:.2f})")
            ax.annotate(
                str(i),
                xy=tuple(ec.mean(0)),
                fontsize=7,
                color=color,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=6,
            )
            ax.plot(
                ego_xy[i, :, 0], ego_xy[i, :, 1], "-", color=color, lw=0.8, alpha=0.35, zorder=3
            )

        k_coll = len(collide_list)
        ax.set_title(
            f"step {t}   collisions {k_coll}/{n} (margin={margin})\n"
            + ("collide: " + ", ".join(collide_list) if collide_list else "all clear"),
            fontsize=9,
            color=("red" if k_coll else "green"),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        # zoom to cover all N ego boxes (the colliding neighbor sits within margin, so in-frame).
        ego_pts_t = ego_corners[:, t].reshape(-1, 2)  # [N*4, 2]
        ctr = ego_pts_t.mean(0)
        half = max(v.zoom_margin, np.abs(ego_pts_t - ctr).max() + 2.0)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)

    for ax in axes[P:]:
        ax.axis("off")

    # legend: one entry per sample (with its scalar reward) + neighbor proxies.
    sample_handles = [
        plt.Line2D([0], [0], color=scolor(i), lw=2, label=f"s{i} r={reward[i]:.1f}")
        for i in range(n)
    ]
    proxies = [
        plt.Line2D([0], [0], color="red", lw=2, label="collision (red fill)"),
        plt.Line2D([0], [0], color="magenta", lw=1.6, label="injected collider"),
        plt.Line2D([0], [0], color="0.4", lw=1.0, label="scene neighbor"),
    ]
    fig.legend(
        handles=sample_handles + proxies, loc="upper center", fontsize=8, ncol=min(n + 3, 11)
    )
    total_pen = float(nc_penalty.sum())
    fig.suptitle(
        f"GRPO collision check @ eval steps {steps}  |  scene #{idx}, N={n}  "
        f"mean reward={reward.mean():.2f}, best={reward.max():.2f}  total penalty={total_pen:.2f}",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(v.output_path, dpi=120)
    print(f"Saved {P} eval-step panels ({n} samples each) to {v.output_path}")
    n_collide = [int((nc_penalty[:, s] > 0).sum()) for s in steps]
    print(f"  collisions per eval step {steps}: {n_collide}  (out of N={n})")
    print(f"  per-sample reward: {[round(float(r), 2) for r in reward]}")
    print(
        f"  mean reward={reward.mean():.3f}  best={reward.max():.3f}  total penalty={total_pen:.3f}"
    )


if __name__ == "__main__":
    main()
