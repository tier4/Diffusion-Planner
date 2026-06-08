"""Compare two checkpoints' deterministic (temperature 0) ego trajectories on synthetic scenes.

Picks scenes from a data list, injects the same synthetic adversarial colliders used by GRPO
training, then runs each of the two checkpoints once per scene at temperature 0 (``--noise_scale 0`` ->
zero initial diffusion noise, deterministic single trajectory). Each scene panel overlays:

  * model A trajectory (blue) and model B trajectory (red), with end footprints,
  * the GT ego future (black dashed),
  * the (augmented) neighbor boxes + futures, injected colliders in magenta,

and annotates each model's collision reward so you can see, e.g., whether a GRPO-finetuned
checkpoint avoids the injected collider better than its base checkpoint.

Example:
    python3 visualize_model_compare.py \
        --model_path_a /path/to/base/best_model.pth \
        --model_path_b /path/to/grpo/best_model.pth \
        --data_list /path/to/path_list_train.json \
        --num_scenes 12 --output_path model_compare.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffusion_planner.grpo_epoch import _neighbor_future_world  # noqa: E402
from diffusion_planner.grpo_utils import compute_collision_reward, sample_group  # noqa: E402
from diffusion_planner.train_epoch import heading_to_cos_sin  # noqa: E402
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector  # noqa: E402

# reuse the loaders / drawing helpers from the sample-group visualizer.
from visualize_grpo_samples import (  # noqa: E402
    _bbox_corners,
    _nonzero_rows,
    boolean,
    build_train_args,
    draw_footprints,
    draw_road,
    load_model,
    select_batch,
)

# neighbor past columns (see loss.py).
_PAST_X, _PAST_Y, _PAST_COS, _PAST_SIN = 0, 1, 2, 3
_PAST_WIDTH, _PAST_LENGTH = 6, 7

_COLOR_A = "tab:blue"
_COLOR_B = "tab:red"


def parse_viz_args():
    p = argparse.ArgumentParser(description="Compare two checkpoints at temperature 0")
    p.add_argument("--model_path_a", type=str, required=True, help="checkpoint A (.pth)")
    p.add_argument("--model_path_b", type=str, required=True, help="checkpoint B (.pth)")
    p.add_argument("--data_list", type=str, required=True,
                   help="dataset path-list JSON (e.g. path_list_train.json)")
    p.add_argument("--label_a", type=str, default="A")
    p.add_argument("--label_b", type=str, default="B")
    p.add_argument("--num_scenes", type=int, default=12)
    p.add_argument("--show_road", type=boolean, default=True)
    p.add_argument("--show_footprint", type=boolean, default=True)
    p.add_argument("--aug_mode", type=str, default="synthetic", choices=["synthetic", "none"])
    p.add_argument("--neighbor_inject_max", type=int, default=1)
    p.add_argument("--neighbor_inject_prob", type=float, default=1.0)
    p.add_argument("--pedestrian_prob", type=float, default=0.3)
    p.add_argument("--bicycle_prob", type=float, default=0.2)
    p.add_argument("--keep_clear_radius", type=float, default=3.0)
    p.add_argument("--use_ema", type=boolean, default=False)
    p.add_argument("--noise_scale", type=float, default=0.0,
                   help="initial-diffusion-noise scale for the single sample (0 = temperature 0)")
    p.add_argument("--output_path", type=str, default="model_compare.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--zoom_margin", type=float, default=15.0,
                   help="metres of padding around the trajectories+GT; <=0 disables zoom")
    return p.parse_known_args()


@torch.no_grad()
def sample_once(model, norm_exp, args, noise_scale, neighbors_future, neighbors_future_valid,
                device, seed):
    """One trajectory per scene from a fixed-seed initial noise, plus its collision reward."""
    torch.manual_seed(seed)  # both models share this seed -> identical initial latent
    ego_world = sample_group(model, norm_exp, noise_scale, device)  # [S, T, 4]
    reward, _, _ = compute_collision_reward(
        ego_world, norm_exp, neighbors_future, neighbors_future_valid, args)
    return ego_world.cpu().numpy(), reward.cpu().numpy()


@torch.no_grad()
def main():
    v, _ = parse_viz_args()
    torch.manual_seed(v.seed)
    np.random.seed(v.seed)
    device = v.device

    # args (model dims + normalizers) are shared; build them off checkpoint A.
    v.resume_model_path = v.model_path_a
    v.num_generations = 1
    v.grpo_noise_scale = v.noise_scale
    args = build_train_args(v)
    model_a = load_model(args, v.model_path_a, v.use_ema, device)
    model_b = load_model(args, v.model_path_b, v.use_ema, device)
    print(f"Loaded A={v.model_path_a}\n       B={v.model_path_b} (ema={v.use_ema})")

    raw, idx = select_batch(v.data_list, v.num_scenes, v.seed, device)
    S = v.num_scenes

    injected_mask = np.zeros((S, raw["neighbor_agents_past"].shape[1]), dtype=bool)
    if v.aug_mode != "none":
        injector = SyntheticColliderInjector(
            pedestrian_prob=v.pedestrian_prob, bicycle_prob=v.bicycle_prob,
            keep_clear_radius=v.keep_clear_radius)
        raw = injector.inject(raw, v.neighbor_inject_max, v.neighbor_inject_prob)
        injected_mask = injector.last_injected_mask.cpu().numpy()
        print(f"Injected {int(injected_mask.sum())} synthetic colliders across {S} scenes")

    raw_np = {k: val.detach().cpu().numpy() for k, val in raw.items()}
    neigh_past = raw_np["neighbor_agents_past"]
    neigh_future = raw_np["neighbor_agents_future"]
    ego_future_gt = raw_np["ego_agent_future"]
    ego_shape = raw_np["ego_shape"]

    # shared inputs (both models see the identical augmented scene).
    exp = dict(raw)
    exp["ego_agent_past"] = heading_to_cos_sin(exp["ego_agent_past"])
    exp["goal_pose"] = heading_to_cos_sin(exp["goal_pose"])
    neighbors_future, neighbor_future_mask = _neighbor_future_world(exp["neighbor_agents_future"])
    neighbors_future_valid = ~neighbor_future_mask
    norm_exp = args.observation_normalizer(exp)

    ego_a, reward_a = sample_once(
        model_a, norm_exp, args, v.noise_scale, neighbors_future, neighbors_future_valid,
        device, v.seed)
    ego_b, reward_b = sample_once(
        model_b, norm_exp, args, v.noise_scale, neighbors_future, neighbors_future_valid,
        device, v.seed)

    cols = min(4, S)
    rows = int(np.ceil(S / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for s in range(S):
        ax = axes[s]
        if v.show_road:
            draw_road(ax, raw_np, s)

        Pn = neigh_past.shape[1]
        for p in range(Pn):
            past = neigh_past[s, p]
            if not np.any(past != 0.0):
                continue
            is_inj = injected_mask[s, p]
            box_c = "magenta" if is_inj else "0.5"
            cur = past[-1]
            corners = _bbox_corners(cur[_PAST_X], cur[_PAST_Y], cur[_PAST_COS], cur[_PAST_SIN],
                                    cur[_PAST_LENGTH], cur[_PAST_WIDTH])
            ax.plot(corners[:, 0], corners[:, 1], "-", color=box_c,
                    lw=1.4 if is_inj else 0.8, alpha=0.9 if is_inj else 0.7,
                    zorder=3.5 if is_inj else 2)
            fut = _nonzero_rows(neigh_future[s, p, :, :2])
            if fut.shape[0] > 0:
                ax.plot(fut[:, 0], fut[:, 1], "-", color=box_c,
                        lw=1.0 if is_inj else 0.8, alpha=0.8 if is_inj else 0.5, zorder=2)

        ego_len, ego_wid = float(ego_shape[s, 1]), float(ego_shape[s, 2])

        # GT ego future
        gt = _nonzero_rows(ego_future_gt[s, :, :2])
        if gt.shape[0] > 0:
            ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.8,
                    label="GT" if s == 0 else None, zorder=4)

        # the two models' deterministic trajectories
        for ego, color, lbl in ((ego_a, _COLOR_A, v.label_a), (ego_b, _COLOR_B, v.label_b)):
            traj = ego[s]  # [T, 4]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2.0,
                    label=lbl if s == 0 else None, zorder=5)
            if v.show_footprint:
                draw_footprints(ax, traj, ego_len, ego_wid, color, 20, alpha=0.4, mode="tail")

        # current ego box + origin
        ego_cur = raw_np["ego_current_state"][s]
        ego_box = _bbox_corners(ego_cur[0], ego_cur[1], ego_cur[2], ego_cur[3], ego_len, ego_wid)
        ax.plot(ego_box[:, 0], ego_box[:, 1], "-", color="black", lw=2.0, zorder=6)
        ax.plot(0, 0, "k*", ms=10, zorder=6)

        ax.set_title(f"scene #{idx[s]}   {v.label_a} r={reward_a[s]:.2f}   "
                     f"{v.label_b} r={reward_b[s]:.2f}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        if v.zoom_margin > 0:
            focus = np.concatenate([ego_a[s, :, :2], ego_b[s, :, :2], np.zeros((1, 2))], axis=0)
            gt_pts = _nonzero_rows(ego_future_gt[s, :, :2])
            if gt_pts.shape[0] > 0:
                focus = np.concatenate([focus, gt_pts], axis=0)
            lo, hi = focus.min(axis=0) - v.zoom_margin, focus.max(axis=0) + v.zoom_margin
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])

    for ax in axes[S:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, ncol=3)
    fig.suptitle(
        f"Temperature-0 trajectory comparison on synthetic scenes  |  "
        f"{v.label_a} mean r={reward_a.mean():.2f}, {v.label_b} mean r={reward_b.mean():.2f}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(v.output_path, dpi=120)
    print(f"Saved {S} scenes to {v.output_path}")
    print(f"  {v.label_a} mean reward={reward_a.mean():.3f}  "
          f"{v.label_b} mean reward={reward_b.mean():.3f}")


if __name__ == "__main__":
    main()
