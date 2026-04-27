#!/usr/bin/env python3
"""P4 winner-recovery viz: per-perturbed-scene PNG of K=8 with reward.py rank-1 highlighted.

For each scene in --scenes (typically a list of perturbed NPZs from
disturb_and_replay), runs K=N with the configured generation_variant under
the loaded model, ranks the K trajectories with the REAL reward.py
(``compute_reward_batch``), picks rank-1 by total reward, and writes a
per-scene PNG showing:

  * route_lanes centerline (orange, what reward scores against)
  * lane boundaries + road borders (line_strings ch3)
  * K=N trajectories (faint grey)
  * deterministic prediction (blue)
  * rank-1 by reward (red, with footprints @ t=0/20/40/60/79)
  * t0_cl vs top1_cl printed in the title — title prefixed [IMPROVE] when
    rank-1's centerline score beats the perturbed-pose t0 centerline.

Output:
  * <output_dir>/improve/scene_<idx>_step_<n>_<variant>.png  (rank-1 improves)
  * <output_dir>/no_improve/scene_<idx>...png                (rank-1 does not)
  * <output_dir>/summary.json                                (per-scene metrics)
  * <output_dir>/improve_scenes.json                         (kept scene paths)

Reward config (--config) MUST set centerline_usage_mode=baselink. No silent
defaults — the script aborts if the reward config is missing.
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
from diffusion_planner.utils.config import Config
from matplotlib.patches import Rectangle

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.autoresearch.tools.viz_cl_recovery import (
    draw_scene_base,
    draw_traj,
)
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
    get_generation_config_labels_for_variant,
)
from rlvr.reward import compute_centerline_score_batch, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _gt_max_speed(data: dict) -> float:
    """Match the trainer's GT-max-speed estimate for speed guidance scaling."""
    if "ego_agent_future" in data:
        gt = data["ego_agent_future"]
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.detach().cpu().numpy()
        valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if valid.sum() >= 5:
            vel = np.diff(gt_np[valid][:, :2], axis=0) / 0.1
            mx = float(np.linalg.norm(vel, axis=-1).max())
            if mx > 0:
                return mx
    ecs = data["ego_current_state"]
    if ecs.dim() == 2:
        ecs = ecs[0]
    return max(float(torch.linalg.vector_norm(ecs[4:6]).item()), 3.0)


def _t0_centerline(data: dict, ego_shape, device) -> float:
    """Compute the centerline reward at the ego's current pose (t=0).

    Build a 1-step trajectory at the origin and feed it through
    compute_centerline_score_batch — uses whatever usage_mode/cap the reward
    function defaults to (baselink, no cap once the global removal lands).
    """
    traj0 = torch.zeros(1, 1, 4, device=device)
    traj0[0, 0, 2] = 1.0  # cos(0)=1
    return float(compute_centerline_score_batch(traj0, ego_shape, data)[0].item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base model checkpoint (.pth). May already have "
                             "LoRA merged in (in which case omit --lora_path).")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--scenes", type=str, required=True,
                        help="JSON list of NPZ paths (perturbed warm scenes).")
    parser.add_argument("--config", type=str, required=True,
                        help="Reward + generation config JSON. MUST set "
                             "centerline_usage_mode=baselink.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None,
                        help="disturb_and_replay manifest.json — used to "
                             "annotate each PNG with the per-NPZ "
                             "perturbation (kind, lateral offset, yaw, dv).")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=2.0)
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Process only the first N scenes (sanity).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_by_npz: dict[str, dict] = {}
    if args.manifest:
        with open(args.manifest) as f:
            for entry in json.load(f):
                manifest_by_npz[entry["npz"]] = entry

    out_root = Path(args.output_dir)
    (out_root / "improve").mkdir(parents=True, exist_ok=True)
    (out_root / "no_improve").mkdir(parents=True, exist_ok=True)

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model + (optional) LoRA
    device = torch.device(DEVICE)
    model_dir = Path(args.model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    model_args = Config(str(args_path))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()

    # Reward + variant
    rcfg = load_reward_config(args.config)
    if getattr(rcfg, "centerline_usage_mode", "baselink") != "baselink":
        raise SystemExit(
            f"Reward config has centerline_usage_mode="
            f"{rcfg.centerline_usage_mode!r}; only 'baselink' is allowed."
        )
    with open(args.config) as f:
        cfg = json.load(f)
    variant = cfg.get("generation_variant", "default")
    use_route_cl = bool(cfg.get("use_route_cl_guidance", False))
    slot_labels = get_generation_config_labels_for_variant(variant, args.K)

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    if args.max_scenes is not None:
        scene_paths = scene_paths[: args.max_scenes]

    print(f"[viz_p4_recovery] model={args.model_path}")
    print(f"  lora_path={args.lora_path}  variant={variant}  K={args.K}")
    print(f"  use_route_cl_guidance={use_route_cl}")
    print(f"  scenes={len(scene_paths)}")

    summary: list[dict] = []
    improve_paths: list[str] = []

    for si, npz_path in enumerate(scene_paths):
        try:
            data = load_npz_data(npz_path, device)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {Path(npz_path).name}: {e}")
            continue
        es = data.get("ego_shape")
        ego_shape = es[0] if es is not None and es.dim() > 1 else es

        v_high = _gt_max_speed(data)
        t0_cl = _t0_centerline(data, ego_shape, device)

        # K trajectories from the training generation_variant
        batch = _stack_scene_data([data], device)
        norm_batch = _normalize_batch(batch, model_args)
        trajs = generate_all_scenes_batched(
            model, model_args, norm_batch,
            K=args.K, noise_range=(args.noise_min, args.noise_max),
            device=device, gen_chunk_size=args.K,
            gt_max_speed=v_high, generation_variant=variant,
            use_route_cl_guidance=use_route_cl,
        )[0]  # [K, T, 4]

        # Score each of the K with the REAL reward function
        per_k = []
        for ki in range(trajs.shape[0]):
            tr = trajs[ki:ki + 1]
            r = compute_reward_batch(tr, data, rcfg)[0]
            per_k.append({
                "k": ki,
                "total": float(r.total),
                "cl": float(r.centerline),
                "rb_cross": bool(r.rb_crossing),
                "lane_cross": bool(r.lane_crossing),
                "kin_gate": bool(r.kinematic_gate),
                "coll_step": (None if r.collision_step is None
                              else int(r.collision_step)),
            })
        # Rank-1 by total reward (ties → highest CL)
        per_k.sort(key=lambda d: (d["total"], d["cl"]), reverse=True)
        top1 = per_k[0]
        top1_traj = trajs[top1["k"]].cpu().numpy()
        delta = top1["cl"] - t0_cl
        improves = delta > 0.0  # reward.py CL is negative-or-zero (closer-to-0 = better)

        # Deterministic baseline (for the plot — not used in ranking)
        decoder = model.module.decoder if hasattr(model, "module") else model.decoder
        saved_fn = decoder._guidance_fn
        decoder._guidance_fn = None
        try:
            B = norm_batch["ego_current_state"].shape[0]
            P = 1 + model_args.predicted_neighbor_num
            future_len = model_args.future_len
            norm_batch_d = {k: v for k, v in norm_batch.items()}
            norm_batch_d["sampled_trajectories"] = torch.zeros(
                B, P, future_len + 1, 4, device=device
            )
            _, det_out = model(norm_batch_d)
            det_traj = det_out["prediction"][0, 0].detach().cpu().numpy()
        finally:
            decoder._guidance_fn = saved_fn
        det_r = compute_reward_batch(
            torch.tensor(det_traj[None], device=device, dtype=torch.float32),
            data, rcfg,
        )[0]

        # Prepend the perturbed ego pose (origin in ego frame) as a shared
        # t=0 anchor. The model's first predicted step is t=0.1s, so without
        # this both det and rank-1 trajs visually "start" at different points
        # (their first-step predictions). With the origin prepended both
        # trajs share the t=0 footprint at the actual perturbed ego pose.
        origin_step = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=det_traj.dtype)
        det_full = np.concatenate([origin_step, det_traj], axis=0)
        top1_full = np.concatenate([origin_step, top1_traj], axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(11, 11))
        draw_scene_base(ax, npz_path)

        # ------ Draw the PERTURBED ego at t=0 explicitly. ------
        # In ego frame the perturbed ego is at (0, 0) facing +x by definition.
        # We draw it as a thick black-edged footprint with high alpha so the
        # "starting state" is unambiguous. The K-trajectory and det/rank-1
        # footprints at t=0 (which sit at the same spot) are drawn under it.
        with np.load(npz_path, allow_pickle=True) as _d:
            _es_np = _d["ego_shape"] if "ego_shape" in _d.files else None
        wb = float(_es_np[0]) if _es_np is not None and len(_es_np) >= 1 else 4.76
        elen = float(_es_np[1]) if _es_np is not None and len(_es_np) >= 2 else 7.24
        ewid = float(_es_np[2]) if _es_np is not None and len(_es_np) >= 3 else 2.29
        ro = elen - wb
        t_rot0 = mtransforms.Affine2D().rotate(0.0).translate(0.0, 0.0) + ax.transData
        ax.add_patch(Rectangle(
            (-ro, -ewid / 2), elen, ewid, lw=2.0,
            ec="black", fc="#ffe066", alpha=0.85, zorder=20, transform=t_rot0,
        ))
        ax.plot([0], [0], "x", color="black", ms=10, mew=2.5, zorder=21)

        # Faint K=N background — also prepended so all start at the same anchor
        for ki in range(trajs.shape[0]):
            tr = trajs[ki].cpu().numpy()
            tr_full = np.concatenate([origin_step, tr], axis=0)
            ax.plot(tr_full[:, 0], tr_full[:, 1], "-", color="#888888", lw=0.7,
                    alpha=0.45, zorder=6)
        # Det in blue (full = origin + 80 predicted steps)
        draw_traj(ax, det_full,
                  f"Det (cl={det_r.centerline:+.2f}  total={det_r.total:+.1f})",
                  "#1f77b4", npz_path, with_footprints=True)
        # Rank-1 by reward in red
        rank1_label = (
            f"Rank-1 k={top1['k']} {slot_labels[top1['k']][:18]} "
            f"(tot={top1['total']:+.1f}  cl={top1['cl']:+.2f})"
        )
        draw_traj(ax, top1_full, rank1_label, "#d62728", npz_path,
                  with_footprints=True)

        # ------ Perturbation annotation from manifest ------
        meta = manifest_by_npz.get(str(npz_path)) or manifest_by_npz.get(npz_path)
        perturb_str = ""
        if meta is not None:
            lat = meta["lateral_offset_m"]
            lon = meta["longitudinal_offset_m"]
            side = "LEFT" if lat > 0 else ("RIGHT" if lat < 0 else "—")
            perturb_str = (
                f"perturb={meta['kind']}  lat={lat:+.2f}m ({side})  "
                f"lon={lon:+.2f}m  yaw={meta['dtheta_deg']:+.1f}°  "
                f"dv={meta['dv']:+.2f}m/s"
            )
            # Visualize the lateral offset as an arrow from where the ego
            # "would have been" (centerline) to the perturbed pose. In ego
            # frame the centerline is along +x; the lateral perp is +y left.
            # Source pose was at (-dx, -dy) relative to current (perturbed)
            # ego frame. Draw a green arrow showing the displacement.
            sx = -float(meta["dx"])
            sy = -float(meta["dy"])
            ax.annotate(
                "", xy=(0, 0), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.2),
                zorder=22,
            )
            ax.plot([sx], [sy], "o", color="#2ca02c", ms=8, mew=0, zorder=22)
            ax.text(sx, sy + 0.4, "source pose", color="#2ca02c",
                    fontsize=9, ha="center", zorder=22)

        all_pts = np.vstack([
            top1_full[:, :2], det_full[:, :2], np.array([[0.0, 0.0]])
        ])
        cx, cy = float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1]))
        half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")
        prefix = "[IMPROVE] " if improves else "[no-improve] "
        ax.set_title(
            f"{prefix}{Path(npz_path).name}\n"
            f"{perturb_str}\n"
            f"t0_cl={t0_cl:+.3f}  rank1_cl={top1['cl']:+.3f}  Δ={delta:+.3f}  "
            f"(rank1 by reward.total under baselink)",
            fontsize=10,
        )
        out_dir = out_root / ("improve" if improves else "no_improve")
        out_path = out_dir / f"scene_{si:03d}_{Path(npz_path).stem}.png"
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
        plt.close(fig)

        if improves:
            improve_paths.append(str(npz_path))

        summary.append({
            "scene": str(npz_path),
            "scene_idx": si,
            "t0_cl": t0_cl,
            "top1_cl": top1["cl"],
            "delta_cl": delta,
            "improves": improves,
            "top1_k": top1["k"],
            "top1_slot": slot_labels[top1["k"]],
            "top1_total": top1["total"],
            # Rank-1 safety: lets downstream filter scenes whose rank-1 winner
            # would teach the model unsafe behavior (e.g. cutting through lanes
            # or close to road borders) to recover. With enable_lane_departure
            # off in reward.py, lane_cross does NOT gate total reward — so a
            # rank-1 by total reward CAN have lane_cross=True. Filter on these.
            "top1_rb_cross": top1["rb_cross"],
            "top1_lane_cross": top1["lane_cross"],
            "top1_kin_gate": top1["kin_gate"],
            "top1_coll_step": top1["coll_step"],
            "det_cl": float(det_r.centerline),
            "det_total": float(det_r.total),
            "png": str(out_path),
        })
        flag = "IMPROVE" if improves else "       "
        print(
            f"  [{si:3d}] {flag}  t0={t0_cl:+.3f}  top1={top1['cl']:+.3f}  "
            f"Δ={delta:+.3f}  slot={slot_labels[top1['k']][:14]:14s}  "
            f"{Path(npz_path).name}"
        )

    n_imp = sum(1 for s in summary if s["improves"])
    print(f"\n[viz_p4_recovery] {n_imp} / {len(summary)} scenes improve under rank-1")

    with open(out_root / "summary.json", "w") as f:
        json.dump({
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "config": args.config,
            "variant": variant,
            "K": args.K,
            "n_total": len(summary),
            "n_improve": n_imp,
            "scenes": summary,
        }, f, indent=2)
    with open(out_root / "improve_scenes.json", "w") as f:
        json.dump(improve_paths, f, indent=2)
    print(f"  Wrote {out_root}/summary.json")
    print(f"  Wrote {out_root}/improve_scenes.json ({len(improve_paths)} scenes)")


if __name__ == "__main__":
    main()
