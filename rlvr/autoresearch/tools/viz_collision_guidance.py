#!/usr/bin/env python3
"""Generate 16 trajectories per scene with the rsft_v2_col4 variant and
visualise which of them clear a stopped-neighbour collision.

For each scene in ``--scenes``:

1. Generate ``K=16`` trajectories with ``generation_variant=rsft_v2_col4``
   — 1 pure-deterministic (slot 0) + 4 collision-guided CL+SPD slots (slots
   1-4) + 2 standard CL+SPD guided slots (slots 5-6) + 9 noise slots
   (slots 7-15). All through the canonical
   ``rlvr.grpo_trainer_batched.generate_all_scenes_batched`` so this matches
   what training uses.
2. Score every one of the 16 trajectories with
   ``rlvr.reward.compute_static_collision_penalty`` against the NPZ's real
   ``neighbor_agents_future``. Report per-trajectory ``sc_min_dist``,
   ``static_crossing``, zone.
3. Emit one PNG per scene showing the lane network + road borders + all
   stopped-neighbour OBBs + all 16 predicted trajectories drawn in the
   zone colour of their worst clearance. Collision-guided slots are
   drawn bolder and labelled.
4. Emit ``summary.json`` aggregating per-scene zone hits + which
   trajectories cleared the collision.

Usage:

    source /opt/ros/humble/setup.bash && source .venv/bin/activate
    python -m rlvr.autoresearch.tools.viz_collision_guidance \\
        --scenes /media/.../j6_static_collision_mining_<ts>/scene_list.json \\
        --model_path /media/.../x2_model_base/best_model.pth \\
        --config /path/to/reward_config_static_on.json \\
        --output_dir /media/.../j6_static_collision_mining_<ts>/col4_viz \\
        --variant rsft_v2_col4

The reward config JSON MUST set ``static_collision_enabled=true``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.audit_static_collision import (
    _ZONE_COLORS,
    _draw_lane_network_from_tensor,
    _zone_for,
)
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.generation_variants import get_variant
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
    get_generation_config_labels_for_variant,
)
from rlvr.reward import compute_static_collision_penalty


@torch.no_grad()
def _score_traj(
    ego_pred: torch.Tensor,           # (T, 4) ego-centric
    data_np: dict[str, np.ndarray],
    reward_cfg,
    device: str,
) -> dict:
    """Score one predicted trajectory via compute_static_collision_penalty."""
    T = ego_pred.shape[0]
    ego_traj = ego_pred.to(device).unsqueeze(0)

    es = np.asarray(data_np["ego_shape"])
    if es.ndim == 2:
        es = es[0]
    ego_shape = torch.from_numpy(es[:3].astype(np.float32)).to(device)

    nb_fut_raw = np.asarray(data_np["neighbor_agents_future"])
    if nb_fut_raw.ndim == 4:
        nb_fut_raw = nb_fut_raw[0]
    nb_fut_raw = nb_fut_raw[:, :T, :]
    yaw = nb_fut_raw[..., 2:3]
    nb_fut_4 = np.concatenate(
        [nb_fut_raw[..., :2], np.cos(yaw), np.sin(yaw)], axis=-1
    ).astype(np.float32)

    neighbor_futures = torch.from_numpy(nb_fut_4).to(device)
    slot_valid = neighbor_futures[:, :, :2].abs().sum(dim=(1, 2)) > 1e-6
    if not slot_valid.any():
        return {"sc_min_dist": 99.0, "static_crossing": False,
                "sc_n_stopped": 0, "zone": "safe"}
    neighbor_futures = neighbor_futures[slot_valid]
    neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6

    nb_past = np.asarray(data_np["neighbor_agents_past"])
    if nb_past.ndim == 4:
        nb_past = nb_past[0]
    nb_past_valid = nb_past[slot_valid.cpu().numpy()]
    shapes_np = nb_past_valid[:, -1, [6, 7]].astype(np.float32)
    neighbor_shapes = torch.from_numpy(shapes_np).to(device)
    zero = neighbor_shapes.abs().sum(dim=-1) < 1e-3
    if zero.any():
        neighbor_shapes[zero] = torch.tensor([2.0, 4.5], device=device)

    sc = compute_static_collision_penalty(
        ego_traj, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid,
        reward_cfg,
    )
    per_ts = sc["per_timestep_min"][0].cpu().numpy()
    if T > 1:
        sc_min = float(per_ts[1:].min())
    else:
        sc_min = float(per_ts[0])
    n_stopped = int(sc["stopped_mask"].sum().item())

    cross_t = float(reward_cfg.sc_cross_thresh)
    near_t = float(reward_cfg.sc_near_thresh)
    wide_t = float(reward_cfg.sc_wide_thresh)
    cont_t = float(reward_cfg.sc_cont_thresh)
    if sc_min < cross_t:
        zone = "cross"
    elif sc_min < near_t:
        zone = "near"
    elif sc_min < wide_t:
        zone = "wide"
    elif sc_min < cont_t:
        zone = "cont"
    else:
        zone = "safe"

    return {
        "sc_min_dist": sc_min,
        "static_crossing": bool(sc["crossing_gate"][0].item() < 0.5),
        "sc_n_stopped": n_stopped,
        "zone": zone,
    }


def _draw_scene_figure(
    data_np: dict[str, np.ndarray],
    trajs_np: np.ndarray,             # (K, T, 4) ego-centric
    scores: list[dict],
    slot_labels: list[str],
    collision_slots: set[int],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(11, 10))

    # Lane network + road borders.
    lanes = np.asarray(data_np["lanes"]) if "lanes" in data_np else None
    _draw_lane_network_from_tensor(ax, lanes, alpha=0.75)
    ls = np.asarray(data_np["line_strings"]) if "line_strings" in data_np else None
    if ls is not None:
        if ls.ndim == 4:
            ls = ls[0]
        if ls.shape[-1] >= 4:
            for i in range(ls.shape[0]):
                line = ls[i]
                valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=-1) > 0.01)
                if valid.sum() > 1:
                    ax.plot(line[valid, 0], line[valid, 1],
                            color="#dd2222", lw=2.0, alpha=0.5, zorder=3)

    # Stopped-NPC OBBs (from real NPZ — any neighbor whose future shows ~no motion).
    nb_past = np.asarray(data_np["neighbor_agents_past"])
    if nb_past.ndim == 4:
        nb_past = nb_past[0]
    nb_fut = np.asarray(data_np["neighbor_agents_future"])
    if nb_fut.ndim == 4:
        nb_fut = nb_fut[0]
    T_fut = nb_fut.shape[1]
    for i in range(nb_past.shape[0]):
        xy0 = nb_past[i, -1, :2]
        if abs(xy0[0]) + abs(xy0[1]) < 1e-6:
            continue
        # Is this neighbor stopped in its real future? (GT motion)
        fut_xy = nb_fut[i, :, :2]
        fut_valid = np.abs(fut_xy).sum(axis=-1) > 1e-6
        if fut_valid.sum() < 2:
            disp = 0.0
        else:
            disp = float(np.linalg.norm(fut_xy[fut_valid].max(axis=0)
                                         - fut_xy[fut_valid].min(axis=0)))
        if disp >= 0.5:
            continue  # moving — skip
        # Pose from past.
        cos_h = nb_past[i, -1, 2]; sin_h = nb_past[i, -1, 3]
        heading = math.atan2(sin_h, cos_h)
        width = float(nb_past[i, -1, 6])
        length = float(nb_past[i, -1, 7])
        t_rot = mtransforms.Affine2D().rotate(heading).translate(xy0[0], xy0[1]) + ax.transData
        ax.add_patch(Rectangle(
            (-length / 2, -width / 2), length, width,
            lw=1.2, ec="#cc6600", fc="#ffb366", alpha=0.75,
            zorder=14, transform=t_rot,
        ))

    # Ego box at t=0 (predictions are ego-centric so t=0 is origin).
    es = np.asarray(data_np["ego_shape"])
    if es.ndim == 2:
        es = es[0]
    ego_wb, ego_L, ego_W = float(es[0]), float(es[1]), float(es[2])
    cog0 = (ego_wb / 2.0, 0.0)
    t_rot0 = mtransforms.Affine2D().rotate(0.0).translate(cog0[0], cog0[1]) + ax.transData
    ax.add_patch(Rectangle(
        (-ego_L / 2, -ego_W / 2), ego_L, ego_W,
        lw=2.0, ec="#3366cc", fc="#3366cc", alpha=0.30,
        zorder=18, transform=t_rot0,
    ))

    # Slot-category colour scheme so the 16 trajectories are readable:
    #   * pure-deterministic (slot 0)        → blue, bold
    #   * 4 collision-guided (set)           → red, bold — this is what we're probing
    #   * non-collision cl_spd guided (rest) → green, medium
    #   * noise / random                     → light grey, thin
    # Zone info is in the ego-footprint colour at the worst-clearance step
    # (cross=red, near=orange, wide=yellow, safe=blue) — same palette as
    # the audit tool.
    SLOT_DET_COLOR = "#2244bb"
    SLOT_COL_COLOR = "#cc0000"
    SLOT_CLSPD_COLOR = "#1b8a3a"
    SLOT_NOISE_COLOR = "#9a9a9a"

    def _slot_category(k: int) -> str:
        if k == 0:
            return "det"
        if k in collision_slots:
            return "col"
        # cl_spd slots occupy indices 1..(len(cl_spd_configs))
        # If the caller passed a slot_label that isn't "random_*" / "noise_*",
        # we treat it as cl_spd-guided. Noise slots sit between the cl_spd
        # block and random slots; just use the label convention.
        lbl = slot_labels[k] if k < len(slot_labels) else ""
        if lbl.startswith("noise_") or lbl.startswith("random_"):
            return "noise"
        return "clspd"

    K = trajs_np.shape[0]
    cat_color = {
        "det": SLOT_DET_COLOR,
        "col": SLOT_COL_COLOR,
        "clspd": SLOT_CLSPD_COLOR,
        "noise": SLOT_NOISE_COLOR,
    }
    cat_lw = {"det": 2.0, "col": 2.0, "clspd": 1.2, "noise": 0.7}
    cat_alpha = {"det": 0.95, "col": 0.95, "clspd": 0.75, "noise": 0.55}

    for k in range(K):
        cat = _slot_category(k)
        xy = trajs_np[k, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], "-",
                color=cat_color[cat], lw=cat_lw[cat],
                alpha=cat_alpha[cat],
                zorder=26 if cat in ("det", "col") else (22 if cat == "clspd" else 20))

    # Build neighbor tensors once so we can cheaply re-score each slot to
    # find its own worst-clearance timestep (the footprint we want to draw).
    import torch as _t
    import rlvr.reward as _rew_mod
    from rlvr.reward import compute_static_collision_penalty as _scp_fn

    _es = np.asarray(data_np["ego_shape"])
    if _es.ndim == 2:
        _es = _es[0]
    _ego_shape_t = _t.from_numpy(_es[:3].astype(np.float32))
    _nb_fut_raw = np.asarray(data_np["neighbor_agents_future"])
    if _nb_fut_raw.ndim == 4:
        _nb_fut_raw = _nb_fut_raw[0]
    _T = trajs_np.shape[1]
    _nb_fut_raw = _nb_fut_raw[:, :_T, :]
    _yaw = _nb_fut_raw[..., 2:3]
    _nb_fut_4 = np.concatenate(
        [_nb_fut_raw[..., :2], np.cos(_yaw), np.sin(_yaw)], axis=-1
    ).astype(np.float32)
    _neighbor_futures = _t.from_numpy(_nb_fut_4)
    _slot_valid = _neighbor_futures[:, :, :2].abs().sum(dim=(1, 2)) > 1e-6
    if _slot_valid.any():
        _neighbor_futures = _neighbor_futures[_slot_valid]
        _neighbor_valid = _neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6
        _nb_past_valid = nb_past[_slot_valid.cpu().numpy()]
        _shapes = _t.from_numpy(_nb_past_valid[:, -1, [6, 7]].astype(np.float32))
        _z = _shapes.abs().sum(dim=-1) < 1e-3
        if _z.any():
            _shapes[_z] = _t.tensor([2.0, 4.5])
    else:
        _neighbor_futures = _t.zeros(0, _T, 4)
        _shapes = _t.zeros(0, 2)
        _neighbor_valid = _t.zeros(0, _T, dtype=_t.bool)

    _cfg_for_viz = _rew_mod.RewardConfig(
        static_collision_enabled=True, sc_gate_enabled=True,
        sc_cross_thresh=0.2, sc_near_thresh=0.5,
        sc_wide_thresh=1.0, sc_cont_thresh=2.0,
    )

    # Draw an ego footprint at the worst-clearance step of the slots we
    # care about — deterministic (slot 0) and the 4 collision-guided.
    # Footprint face colour = zone at that step (red = cross, orange =
    # near, yellow = wide, blue = safe). Border colour = slot category.
    for k in range(K):
        cat = _slot_category(k)
        if cat not in ("det", "col"):
            continue
        ego_traj_k = _t.from_numpy(trajs_np[k].astype(np.float32)).unsqueeze(0)
        out = _scp_fn(
            ego_traj_k, _ego_shape_t, _neighbor_futures, _shapes, _neighbor_valid,
            _cfg_for_viz,
        )
        per_ts = out["per_timestep_min"][0].cpu().numpy()
        first_cross = out["first_crossing_steps"][0]
        if first_cross is not None:
            t_star = int(first_cross)
        elif _T > 1:
            t_star = int(np.argmin(per_ts[1:])) + 1
        else:
            t_star = 0
        d_star = float(per_ts[t_star])
        cos_k, sin_k = trajs_np[k, t_star, 2], trajs_np[k, t_star, 3]
        hn_k = math.hypot(cos_k, sin_k)
        h_k = math.atan2(sin_k / max(hn_k, 1e-6), cos_k / max(hn_k, 1e-6))
        cx_k, cy_k = trajs_np[k, t_star, :2]
        cog_k = (cx_k + math.cos(h_k) * ego_wb / 2.0,
                 cy_k + math.sin(h_k) * ego_wb / 2.0)
        zone_k = _zone_for(d_star)
        t_rot_k = mtransforms.Affine2D().rotate(h_k).translate(cog_k[0], cog_k[1]) + ax.transData
        ec = cat_color[cat]
        ax.add_patch(Rectangle(
            (-ego_L / 2, -ego_W / 2), ego_L, ego_W,
            lw=1.8, ec=ec, fc=_ZONE_COLORS[zone_k], alpha=0.35,
            zorder=28, transform=t_rot_k,
        ))
        ax.annotate(
            f"#{k} {slot_labels[k]}  d={d_star:.2f} @t={t_star}",
            xy=(cog_k[0], cog_k[1]), fontsize=6, color=ec,
            ha="center", va="center",
            xytext=(0, -10 if (k % 2) else 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor=ec, alpha=0.9),
            zorder=29,
        )

    # Legend for slot categories (trajectory colours).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=SLOT_DET_COLOR, lw=2.0, label="det (slot 0)"),
        Line2D([0], [0], color=SLOT_COL_COLOR, lw=2.0,
               label=f"collision-guided ({len(collision_slots)} slots)"),
        Line2D([0], [0], color=SLOT_CLSPD_COLOR, lw=1.2,
               label="other CL+SPD guided"),
        Line2D([0], [0], color=SLOT_NOISE_COLOR, lw=0.7, label="noise / random"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              framealpha=0.9)

    # Summary: how many of the 16, and specifically of the 4 collision slots, cleared.
    n_safe = sum(1 for s in scores if s["zone"] == "safe")
    n_cross = sum(1 for s in scores if s["zone"] == "cross")
    col_safe = sum(1 for k in collision_slots if scores[k]["zone"] == "safe")
    col_cross = sum(1 for k in collision_slots if scores[k]["zone"] == "cross")
    best_d = max(s["sc_min_dist"] for s in scores)
    best_col_d = max(scores[k]["sc_min_dist"] for k in collision_slots)

    # Auto-zoom so the offender (nearest stopped NPC to any ego predicted
    # pose) is always in frame. Previously we framed on t=0 + nearby NPCs,
    # which cropped out forward offenders like a parked car 10 m ahead.
    pts = [np.array([0.0, 0.0])]
    # Include every stopped NPC within 40 m of ego OR within 5 m of any
    # predicted ego pose.
    traj_xy_flat = trajs_np[:, :, :2].reshape(-1, 2)
    for i in range(nb_past.shape[0]):
        xy0 = nb_past[i, -1, :2]
        if abs(xy0[0]) + abs(xy0[1]) < 1e-6:
            continue
        d_to_ego = float(np.linalg.norm(xy0))
        d_to_path = float(np.linalg.norm(traj_xy_flat - xy0, axis=1).min())
        if d_to_ego <= 40.0 or d_to_path <= 5.0:
            pts.append(xy0)
    # Also include every trajectory endpoint so the plan horizon is visible.
    for k in range(trajs_np.shape[0]):
        pts.append(trajs_np[k, -1, :2])
    stacked = np.vstack([np.array(p).reshape(1, 2) for p in pts])
    cx = float((stacked[:, 0].min() + stacked[:, 0].max()) * 0.5)
    cy = float((stacked[:, 1].min() + stacked[:, 1].max()) * 0.5)
    half = max(float(stacked[:, 0].max() - stacked[:, 0].min()),
               float(stacked[:, 1].max() - stacked[:, 1].min())) * 0.55 + 8.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"{title}\n"
        f"16-gen: {n_safe}/{K} safe, {n_cross}/{K} cross.  "
        f"Collision slots (slots {sorted(collision_slots)}): "
        f"{col_safe}/{len(collision_slots)} safe, {col_cross}/{len(collision_slots)} cross.  "
        f"best d (any slot)={best_d:.2f} m  best d (col slot)={best_col_d:.2f} m",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True,
                        help="JSON list of NPZ paths.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Reward config JSON with static_collision_enabled=true.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="rsft_v2_col4",
                        help="Generation variant name. Default rsft_v2_col4 "
                             "(4 collision-guided slots out of 16).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    reward_cfg = load_reward_config(args.config)
    if not getattr(reward_cfg, "static_collision_enabled", False):
        raise SystemExit(
            f"{args.config} must set static_collision_enabled=true"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scene_paths = list(json.load(f))
    N = len(scene_paths)
    print(f"Loaded {N} scenes.")

    variant = get_variant(args.variant)
    K = 1 + len(variant.cl_spd_configs) + len(variant.noise_configs)
    # Collision-guided slot indices (1..K): slot 0 is pure det, slots
    # 1..len(cl_spd) are cl_spd configs.
    collision_slots = set()
    for i, cfg in enumerate(variant.cl_spd_configs):
        if cfg.get("col", 0.0) > 0.0:
            collision_slots.add(i + 1)  # +1 because slot 0 is pure det
    slot_labels = get_generation_config_labels_for_variant(args.variant, K=K)
    print(f"Variant '{args.variant}': K={K}, "
          f"collision-guided slots={sorted(collision_slots)} "
          f"({len(collision_slots)}/{K})")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    from scenario_generation.simulate import load_model
    print(f"Loading model {args.model_path}")
    model, model_args = load_model(str(args.model_path), device)

    # Load and stack all scenes into one batch.
    print(f"Loading {N} NPZs...")
    all_data = []
    valid_paths = []
    for path in scene_paths:
        try:
            d = load_npz_data(path, device)
            all_data.append(d)
            valid_paths.append(path)
        except Exception as e:
            print(f"  [skip] {Path(path).name}: {e}")
    if not all_data:
        raise SystemExit("No valid scenes loaded.")

    batch_data = _stack_scene_data(all_data, device)
    norm_batch = _normalize_batch(batch_data, model_args)

    # GT max speed per scene (for speed guidance).
    import numpy as _np
    gt_speeds = []
    for d in all_data:
        gt = d.get("ego_agent_future")
        if gt is not None:
            if gt.dim() == 3:
                gt = gt[0]
            gt_np = gt.cpu().numpy()
            vmask = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            if vmask.sum() >= 5:
                vel = _np.diff(gt_np[vmask][:, :2], axis=0) / 0.1
                gt_speeds.append(float(_np.linalg.norm(vel, axis=-1).max()))
            else:
                gt_speeds.append(3.0)
        else:
            gt_speeds.append(3.0)
    gt_max_speed = float(_np.median(gt_speeds))

    # Generate K trajs per scene — batched across all N scenes.
    noise_range = (0.5, 2.0)
    print(f"Generating K={K} trajectories × N={len(all_data)} scenes "
          f"with variant={args.variant} ...")
    all_trajs = generate_all_scenes_batched(
        model, model_args, norm_batch, K=K, noise_range=noise_range,
        device=device, gt_max_speed=gt_max_speed, gen_chunk_size=16,
        longitudinal_eta=0.0, longitudinal_lambda=0.5, longitudinal_scale=10.0,
        lateral_eta=0.0, lateral_lambda=2.0, lateral_scale=5.0,
        speed_stretch=1.0, generation_variant=args.variant,
    )  # (N, K, T, 4)
    print(f"Generated trajectories: {tuple(all_trajs.shape)}")

    # Score + draw per scene.
    summary = []
    for si, path in enumerate(valid_paths):
        try:
            with np.load(path, allow_pickle=True) as raw:
                data_np = {k: raw[k] for k in raw.files if k != "version"}
        except Exception as e:
            print(f"  [skip draw] {path}: {e}")
            continue

        traj_k = all_trajs[si]  # (K, T, 4)
        trajs_np = traj_k.detach().cpu().numpy()
        scores: list[dict] = []
        for k in range(K):
            scores.append(_score_traj(traj_k[k], data_np, reward_cfg, device))

        title_tag = Path(path).stem
        png_path = args.output_dir / f"scene_{si:03d}_{title_tag}.png"
        _draw_scene_figure(
            data_np, trajs_np, scores, slot_labels, collision_slots,
            png_path, title=f"[{si}] {title_tag}",
        )

        row = {
            "scene_idx": si,
            "npz": path,
            "png": str(png_path),
            "scores": scores,
            "n_safe": sum(1 for s in scores if s["zone"] == "safe"),
            "n_cross": sum(1 for s in scores if s["zone"] == "cross"),
            "col_slots_safe": sum(1 for k in collision_slots
                                  if scores[k]["zone"] == "safe"),
            "col_slots_cross": sum(1 for k in collision_slots
                                   if scores[k]["zone"] == "cross"),
            "best_d_any": max(s["sc_min_dist"] for s in scores),
            "best_d_col_slot": max(scores[k]["sc_min_dist"] for k in collision_slots),
        }
        summary.append(row)
        if (si + 1) % 10 == 0:
            print(f"  Drew {si+1}/{len(valid_paths)}")

    # Aggregate stats.
    n_total = len(summary)
    n_any_safe = sum(1 for r in summary if r["n_safe"] >= 1)
    n_col_any_safe = sum(1 for r in summary if r["col_slots_safe"] >= 1)
    med_best_d = float(np.median([r["best_d_any"] for r in summary])) if summary else 0.0
    med_best_col_d = float(np.median([r["best_d_col_slot"] for r in summary])) if summary else 0.0

    payload = {
        "scenes_list": str(args.scenes),
        "model_path": str(args.model_path),
        "reward_config_path": str(args.config),
        "variant": args.variant,
        "K": K,
        "collision_slots": sorted(collision_slots),
        "slot_labels": slot_labels,
        "n_scenes": n_total,
        "n_scenes_with_any_safe_slot": n_any_safe,
        "n_scenes_with_any_col_slot_safe": n_col_any_safe,
        "median_best_d_any_slot": med_best_d,
        "median_best_d_col_slot": med_best_col_d,
        "per_scene": summary,
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {summary_path}")
    print(f"  {n_any_safe}/{n_total} scenes have at least one clearing slot (>=1 m)")
    print(f"  {n_col_any_safe}/{n_total} scenes cleared by at least one COLLISION slot")
    print(f"  median best d (any slot): {med_best_d:.2f} m")
    print(f"  median best d (col slot): {med_best_col_d:.2f} m")


if __name__ == "__main__":
    main()
