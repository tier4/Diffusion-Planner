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
from rlvr.autoresearch.tools.eval_det_avoidance import (
    det_inference_batched,
    reward_breakdown_to_det_dict,
)
from rlvr.autoresearch.tools.percentile_filter_perturbed import is_scene_eligible
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
    """Compute the centerline reward at the ego's CURRENT pose.

    Both ego_current_state and route_lanes are in the same frame (perturbed
    ego at origin). Builds a 1-step trajectory at ego_current_state and
    scores it against route_lanes.
    """
    ecs = data["ego_current_state"]
    if ecs.dim() == 2:
        ecs = ecs[0]
    traj0 = torch.zeros(1, 1, 4, device=device)
    traj0[0, 0, 0] = ecs[0]
    traj0[0, 0, 1] = ecs[1]
    traj0[0, 0, 2] = ecs[2]
    traj0[0, 0, 3] = ecs[3]
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
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip PNG plot (only write summary.json).")
    parser.add_argument("--scene_batch_size", type=int, default=1,
                        help="Scenes batched through K-generation + det forward "
                             "per call. K × scene_batch_size trajectories in "
                             "parallel through the diffusion loop. Requires "
                             "--no_viz when > 1.")
    parser.add_argument(
        "--ego_shape", type=str, required=True,
        help="Ego dimensions as 'WHEEL_BASE,LENGTH,WIDTH' in metres. REQUIRED. "
             "Asserted against every NPZ's ego_shape; mismatch is a hard fail "
             "(prevents silent footprint undersizing in the safety gates).",
    )
    args = parser.parse_args()
    if args.scene_batch_size > 1 and not args.no_viz:
        raise SystemExit("--scene_batch_size > 1 requires --no_viz.")
    if args.scene_batch_size < 1:
        raise SystemExit("--scene_batch_size must be >= 1.")
    _ego_parts = [float(x) for x in args.ego_shape.split(",")]
    if len(_ego_parts) != 3 or any(v <= 0 for v in _ego_parts):
        raise SystemExit(
            f"--ego_shape must be 'WB,LEN,WIDTH' with 3 positive values; "
            f"got {args.ego_shape!r}"
        )
    cli_ego_shape = np.array(_ego_parts, dtype=np.float32)

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

    SBS = args.scene_batch_size
    for batch_start in range(0, len(scene_paths), SBS):
        paths_ok: list = []
        datas_ok: list = []
        orig_indices: list[int] = []
        for gi, p in enumerate(scene_paths[batch_start : batch_start + SBS],
                               start=batch_start):
            try:
                # Check raw NPZ before load_npz_data injects defaults.
                with np.load(p, allow_pickle=True) as f:
                    raw_keys = set(f.files)
                if "ego_shape" not in raw_keys:
                    raise SystemExit(
                        f"NPZ {p} is missing ego_shape; refusing silent "
                        f"fallback. Re-extract / pad upstream."
                    )
                _d = load_npz_data(p, device)
                _es = _d["ego_shape"].cpu().numpy().reshape(-1)[:3]
                if not np.allclose(_es, cli_ego_shape, atol=1e-2):
                    raise SystemExit(
                        f"NPZ {p} has ego_shape={_es.tolist()} but "
                        f"--ego_shape={cli_ego_shape.tolist()}. Mismatch — "
                        f"refusing to score with wrong footprint."
                    )
                datas_ok.append(_d)
                paths_ok.append(p)
                orig_indices.append(gi)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {Path(p).name}: {e}")
        if not datas_ok:
            continue
        B = len(datas_ok)

        ego_shapes = []
        t0_cls = []
        v_highs = []
        for d in datas_ok:
            es = d.get("ego_shape")
            es_one = es[0] if es is not None and es.dim() > 1 else es
            ego_shapes.append(es_one)
            v_highs.append(_gt_max_speed(d))
            t0_cls.append(_t0_centerline(d, es_one, device))
        # Approximation: use max GT speed across the batch. Per-scene speed
        # would require generate_all_scenes_batched to accept a vector, which
        # threads through the guidance system in diffusion_planner/.
        # Conservative: higher speed = stricter speed guidance. Exact when
        # scene_batch_size=1 (the default).
        v_batch = max(v_highs)

        batch = _stack_scene_data(datas_ok, device)
        norm_batch = _normalize_batch(batch, model_args)
        trajs_BKT4 = generate_all_scenes_batched(
            model, model_args, norm_batch,
            K=args.K, noise_range=(args.noise_min, args.noise_max),
            device=device, gen_chunk_size=args.K,
            gt_max_speed=v_batch, generation_variant=variant,
            use_route_cl_guidance=use_route_cl,
        )  # [B, K, T, 4]

        det_trajs_BT4 = det_inference_batched(
            model, model_args, datas_ok, device, norm_batch=norm_batch,
        )

        for bi in range(B):
            si = orig_indices[bi]
            npz_path = paths_ok[bi]
            data = datas_ok[bi]
            t0_cl = t0_cls[bi]
            trajs = trajs_BKT4[bi]  # [K, T, 4]
            det_traj_t = det_trajs_BT4[bi:bi + 1]  # [1, T, 4]

            all_rewards = compute_reward_batch(trajs, data, rcfg)
            per_k = [
                {
                    "k": ki,
                    "total": float(r.total),
                    "cl": float(r.centerline),
                    "rb_cross": bool(r.rb_crossing),
                    "lane_cross": bool(r.lane_crossing),
                    "kin_violated": bool(r.kinematic_violated),
                    "coll_step": (None if r.collision_step is None
                                  else int(r.collision_step)),
                    "static_crossing": bool(r.static_crossing),
                }
                for ki, r in enumerate(all_rewards)
            ]
            per_k.sort(key=lambda d: (d["total"], d["cl"]), reverse=True)
            top1 = per_k[0]
            # delta is trajectory-vs-trajectory: top1_cl − det_cl (both
            # 80-frame trajectory cl scores from compute_reward_batch).
            # Earlier versions compared top1_cl (80-frame) to t0_cl
            # (1-frame at ego_current_state) — apples-to-oranges; that
            # comparison let scenes where rank-1 was no better than
            det_r = compute_reward_batch(det_traj_t, data, rcfg)[0]
            delta = top1["cl"] - float(det_r.centerline)
            top1_better_reward = top1["total"] > float(det_r.total)
            top1_is_different = top1["k"] != 0  # k=0 is deterministic
            improves = (top1_better_reward and top1_is_different
                        and is_scene_eligible(top1, t0_cl=t0_cl))

            if args.no_viz:
                entry = {
                    "scene": str(npz_path),
                    "scene_idx": si,
                    "t0_cl": t0_cl,
                    "top1_cl": top1["cl"],
                    "delta_cl": delta,
                    "improves": improves,
                    "top1_k": top1["k"],
                    "top1_slot": slot_labels[top1["k"]],
                    "top1_total": top1["total"],
                    "top1_rb_cross": top1["rb_cross"],
                    "top1_lane_cross": top1["lane_cross"],
                    "top1_kin_violated": top1["kin_violated"],
                    "top1_coll_step": top1["coll_step"],
                    "top1_static_crossing": top1["static_crossing"],
                    "png": None,
                }
                entry.update(reward_breakdown_to_det_dict(det_r))
                summary.append(entry)
                if improves:
                    improve_paths.append(str(npz_path))
                flag = "IMPROVE" if improves else "       "
                print(
                    f"  [{si:3d}] {flag}  t0={t0_cl:+.3f}  top1={top1['cl']:+.3f}  "
                    f"Δ={delta:+.3f}  slot={slot_labels[top1['k']][:14]:14s}  "
                    f"{Path(npz_path).name}"
                )
                continue

            # Viz path (scene_batch_size == 1 guaranteed)
            top1_traj = trajs[top1["k"]].cpu().numpy()
            det_traj = det_trajs_BT4[bi].cpu().numpy()

            # As of disturb_and_replay 2026-05-12 (commit 3329364), the NPZ
            # is re-anchored to the perturbed ego pose at origin. Trajectory
            # and map data are already in the same frame, no _to_lf needed.
            # Source pose (pre-perturbation ego) sits at (-dx, -dy) in this
            # frame — kept for the annotation arrow only.
            meta = manifest_by_npz.get(str(npz_path)) or manifest_by_npz.get(npz_path)
            dx_pert = float(meta["dx"]) if meta is not None else 0.0
            dy_pert = float(meta["dy"]) if meta is not None else 0.0
            source_pose_xy = (-dx_pert, -dy_pert)
            anchor = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=det_traj.dtype)

            det_full = np.concatenate([anchor, det_traj], axis=0)
            top1_full = np.concatenate([anchor, top1_traj], axis=0)

            fig, ax = plt.subplots(figsize=(11, 11))
            draw_scene_base(ax, npz_path)

            with np.load(npz_path, allow_pickle=True) as _d:
                _es_np = _d["ego_shape"] if "ego_shape" in _d.files else None
            wb = float(_es_np[0]) if _es_np is not None and len(_es_np) >= 1 else 4.76
            elen = float(_es_np[1]) if _es_np is not None and len(_es_np) >= 2 else 7.24
            ewid = float(_es_np[2]) if _es_np is not None and len(_es_np) >= 3 else 2.29
            ro = elen - wb
            # Perturbed ego (now-current) is at origin in the new frame.
            t_rot_pert = (mtransforms.Affine2D()
                          .rotate(0.0)
                          .translate(0.0, 0.0)
                          + ax.transData)
            ax.add_patch(Rectangle(
                (-ro, -ewid / 2), elen, ewid, lw=1.6,
                ec="#7a6500", fc="#ffe066", alpha=0.35, zorder=20,
                transform=t_rot_pert,
            ))
            ax.plot([0.0], [0.0], "x", color="#7a6500",
                    ms=9, mew=2.2, zorder=21)

            for ki in range(trajs.shape[0]):
                tr = trajs[ki].cpu().numpy()
                tr_full = np.concatenate([anchor, tr], axis=0)
                ax.plot(tr_full[:, 0], tr_full[:, 1], "-", color="#888888", lw=0.7,
                        alpha=0.45, zorder=6)
            # Ground truth from NPZ (ego_agent_future, 3-channel x,y,yaw at
            # load time → 4-channel after heading_to_cos_sin). Drawn in green
            # so the user can compare Det / Rank-1 / K=8 candidates against
            # what the bag actually did.
            gt_t = data.get("ego_agent_future")
            if gt_t is not None:
                gt_np = gt_t.detach().cpu().numpy()
                if gt_np.ndim == 3:
                    gt_np = gt_np[0]
                if gt_np.shape[-1] >= 3 and (np.abs(gt_np[:, :2]).sum() > 0.1):
                    if gt_np.shape[-1] == 3:
                        gt_4 = np.zeros((gt_np.shape[0], 4), dtype=np.float32)
                        gt_4[:, 0] = gt_np[:, 0]
                        gt_4[:, 1] = gt_np[:, 1]
                        gt_4[:, 2] = np.cos(gt_np[:, 2])
                        gt_4[:, 3] = np.sin(gt_np[:, 2])
                        gt_np = gt_4
                    gt_full = np.concatenate([anchor, gt_np.astype(det_traj.dtype)], axis=0)
                    draw_traj(ax, gt_full, "GT (bag future)",
                              "#2ca02c", npz_path, with_footprints=False)
            draw_traj(ax, det_full,
                      f"Det (cl={det_r.centerline:+.2f}  total={det_r.total:+.1f})",
                      "#1f77b4", npz_path, with_footprints=True)
            rank1_label = (
                f"Rank-1 k={top1['k']} {slot_labels[top1['k']][:18]} "
                f"(tot={top1['total']:+.1f}  cl={top1['cl']:+.2f})"
            )
            draw_traj(ax, top1_full, rank1_label, "#d62728", npz_path,
                      with_footprints=True)

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
                # Arrow from source pose to perturbed-now ego at origin.
                ax.annotate(
                    "", xy=(0.0, 0.0), xytext=source_pose_xy,
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.2),
                    zorder=22,
                )
                ax.plot([source_pose_xy[0]], [source_pose_xy[1]],
                        "o", color="#2ca02c", ms=8, mew=0, zorder=22)
                ax.text(source_pose_xy[0], source_pose_xy[1] - 0.4, "source pose",
                        color="#2ca02c", fontsize=9, ha="center", zorder=22)

            all_pts = np.vstack([
                top1_full[:, :2], det_full[:, :2],
                np.array([list(source_pose_xy)]),
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
                f"t0_cl={t0_cl:+.3f}(1-frame)  det_cl={float(det_r.centerline):+.3f}  "
                f"rank1_cl={top1['cl']:+.3f}  Δ_top1-det={delta:+.3f}  "
                f"(rank1 by reward.total; Δ apples-to-apples 80-frame)",
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
                "top1_rb_cross": top1["rb_cross"],
                "top1_lane_cross": top1["lane_cross"],
                "top1_kin_violated": top1["kin_violated"],
                "top1_coll_step": top1["coll_step"],
                "top1_static_crossing": top1["static_crossing"],
                "png": str(out_path),
            })
            summary[-1].update(reward_breakdown_to_det_dict(det_r))
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
