#!/usr/bin/env python3
"""PRiSM 3-way comparison: LoRA-less baseline vs warmstart base vs PRiSM-trained model.

For each perturbed scene in --scenes, runs deterministic inference under
each of the three models and overlays their trajectories on a single PNG.
Each model's prediction is drawn as a coloured trajectory + sparse
footprints @ t=0/20/40/60/79; the perturbed ego is rendered at the origin
(yellow filled) and the lateral-offset arrow is drawn from the source
pose to the perturbed pose.

Usage:
    python -m rlvr.autoresearch.tools.viz_prism_compare \\
        --baseline_model /path/to/<baseline>/best_model.pth \\
        --warmstart_model /path/to/<warmstart>/merged.pth \\
        --prism_model /path/to/<warmstart>/merged.pth \\
        --prism_lora /path/to/<run_dir>/lora_epoch_NNN \\
        --scenes /path/to/<perturbed_val>.json \\
        --manifest /path/to/manifest.json \\
        --config /path/to/<reward>.json \\
        --output_dir /path/out \\
        [--max_scenes 12] [--top_delta]   # pick scenes with biggest PRiSM−warmstart Δ

Reward config (--config) MUST set centerline_usage_mode=baselink.
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
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data
from rlvr.reward import compute_centerline_score_batch, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_path: str, lora_path: str | None, device):
    model_dir = Path(model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    margs = Config(str(args_path))
    model = Diffusion_Planner(margs)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    if lora_path:
        model = load_lora_checkpoint(model, lora_path)
        model.eval()
    return model, margs


@torch.no_grad()
def _det_predict(model, model_args, data):
    device = next(model.parameters()).device
    batch = _stack_scene_data([data], device)
    norm_batch = _normalize_batch(batch, model_args)
    B = norm_batch["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    norm_batch["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4, device=device)
    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    saved_fn = decoder._guidance_fn
    decoder._guidance_fn = None
    try:
        _, det_out = model(norm_batch)
    finally:
        decoder._guidance_fn = saved_fn
    return det_out["prediction"][0, 0].detach().cpu().numpy()


def _t0_centerline(data, ego_shape, device):
    traj0 = torch.zeros(1, 1, 4, device=device)
    traj0[0, 0, 2] = 1.0
    return float(compute_centerline_score_batch(traj0, ego_shape, data)[0].item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--warmstart_model", type=str, required=True)
    parser.add_argument("--prism_model", type=str, required=True)
    parser.add_argument("--prism_lora", type=str, default=None)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--top_delta", action="store_true",
                        help="Pick scenes with biggest Δ_cl (PRiSM minus the "
                             "reference model picked by --rank_by). Requires "
                             "--prism_summary and the matching reference summary.")
    parser.add_argument("--rank_by", type=str, default="warmstart",
                        choices=["warmstart", "baseline"],
                        help="Which model PRiSM is compared to for top-Δ ranking.")
    parser.add_argument("--hide_warmstart", action="store_true",
                        help="Don't draw the warmstart trajectory — only "
                             "baseline (LoRA-less) vs PRiSM.")
    parser.add_argument("--warmstart_summary", type=str, default=None)
    parser.add_argument("--baseline_summary", type=str, default=None)
    parser.add_argument("--prism_summary", type=str, default=None)
    parser.add_argument("--baseline_label", type=str, default="baseline")
    parser.add_argument("--warmstart_label", type=str, default="warmstart")
    parser.add_argument("--prism_label", type=str, default="PRiSM")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rcfg = load_reward_config(args.config)
    if getattr(rcfg, "centerline_usage_mode", "baselink") != "baselink":
        raise SystemExit(
            f"Reward config has centerline_usage_mode="
            f"{rcfg.centerline_usage_mode!r}; only 'baselink' is allowed."
        )

    manifest_by_npz = {}
    if args.manifest:
        with open(args.manifest) as f:
            for entry in json.load(f):
                manifest_by_npz[entry["npz"]] = entry

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    # Optional: rank scenes by PRiSM − reference improvement, take top
    if args.top_delta and args.prism_summary:
        ref_path = (args.baseline_summary if args.rank_by == "baseline"
                    else args.warmstart_summary)
        if ref_path is None:
            raise SystemExit(
                f"--top_delta with --rank_by={args.rank_by} requires "
                f"--{'baseline_summary' if args.rank_by == 'baseline' else 'warmstart_summary'}"
            )
        ref = {r["scene"]: r["det_cl"] for r in json.load(open(ref_path))["scenes"]}
        ps  = {r["scene"]: r["det_cl"] for r in json.load(open(args.prism_summary))["scenes"]}
        scene_paths = sorted(scene_paths,
                             key=lambda p: -(ps.get(p, 0) - ref.get(p, 0)))
    if args.max_scenes is not None:
        scene_paths = scene_paths[: args.max_scenes]

    device = torch.device(DEVICE)
    print(f"[viz_prism_compare] loading 3 models...")
    m_base, args_base   = _load_model(args.baseline_model, None, device)
    m_warm, args_warm   = _load_model(args.warmstart_model, None, device)
    m_prism, args_prism = _load_model(args.prism_model, args.prism_lora, device)

    print(f"[viz_prism_compare] processing {len(scene_paths)} scenes...")
    summary = []

    for si, npz_path in enumerate(scene_paths):
        try:
            data_b = load_npz_data(npz_path, device)
        except Exception as e:
            print(f"  [skip] {Path(npz_path).name}: {e}")
            continue
        # Note: each model has its own model_args / normalizer. We pass the same data
        # dict to each, but normalize per-model inside _det_predict.
        es = data_b.get("ego_shape")
        ego_shape = es[0] if es is not None and es.dim() > 1 else es
        t0_cl = _t0_centerline(data_b, ego_shape, device)

        traj_base   = _det_predict(m_base,  args_base,  data_b)
        traj_warm   = _det_predict(m_warm,  args_warm,  data_b)
        traj_prism  = _det_predict(m_prism, args_prism, data_b)

        def _cl(traj):
            r = compute_reward_batch(
                torch.tensor(traj[None], device=device, dtype=torch.float32),
                data_b, rcfg)[0]
            return float(r.centerline)
        cl_base, cl_warm, cl_prism = _cl(traj_base), _cl(traj_warm), _cl(traj_prism)

        # Lookup perturbation offset for this scene from manifest
        meta = manifest_by_npz.get(str(npz_path)) or manifest_by_npz.get(npz_path)
        dx_pert = float(meta["dx"]) if meta is not None else 0.0
        dy_pert = float(meta["dy"]) if meta is not None else 0.0

        # The model output is in an "ego-at-origin" frame: pred[0] ≈ (1, 0) for
        # an ego moving forward at ~10 m/s, regardless of what ego_current_state
        # says about position. Lanelets in the NPZ are in the original world
        # frame (un-shifted). To draw all three trajectories in the lanelet
        # frame STARTING from the perturbed ego pose at (dx, dy), translate
        # raw model output by (dx, dy). Then prepend (dx, dy, 1, 0) so the
        # trajectory's first footprint sits exactly on the perturbed ego pose.
        def _to_lanelet_frame(traj):
            t = traj.copy()
            t[:, 0] += dx_pert
            t[:, 1] += dy_pert
            return t
        traj_base_lf  = _to_lanelet_frame(traj_base)
        traj_warm_lf  = _to_lanelet_frame(traj_warm)
        traj_prism_lf = _to_lanelet_frame(traj_prism)
        anchor = np.array([[dx_pert, dy_pert, 1.0, 0.0]], dtype=traj_base.dtype)
        full_b = np.concatenate([anchor, traj_base_lf],  axis=0)
        full_w = np.concatenate([anchor, traj_warm_lf],  axis=0)
        full_p = np.concatenate([anchor, traj_prism_lf], axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 12))
        draw_scene_base(ax, npz_path)

        # Perturbed ego at (dx, dy) in lanelet frame — yellow translucent
        # rectangle anchored on the actual perturbed pose.
        es_np = np.load(npz_path, allow_pickle=True).get("ego_shape", None)
        wb = float(es_np[0]) if es_np is not None and len(es_np) >= 1 else 4.76
        elen = float(es_np[1]) if es_np is not None and len(es_np) >= 2 else 7.24
        ewid = float(es_np[2]) if es_np is not None and len(es_np) >= 3 else 2.29
        ro = elen - wb
        t_rot_pert = (mtransforms.Affine2D()
                      .rotate(0.0)
                      .translate(dx_pert, dy_pert)
                      + ax.transData)
        ax.add_patch(Rectangle(
            (-ro, -ewid / 2), elen, ewid, lw=1.6,
            ec="#7a6500", fc="#ffe066", alpha=0.35, zorder=20,
            transform=t_rot_pert,
        ))
        ax.plot([dx_pert], [dy_pert], "x", color="#7a6500",
                ms=9, mew=2.2, zorder=21)

        # Model trajectories
        draw_traj(ax, full_b,
                  f"{args.baseline_label}  cl={cl_base:+.3f}",
                  "#1f77b4", npz_path, with_footprints=True)
        if not args.hide_warmstart:
            draw_traj(ax, full_w,
                      f"{args.warmstart_label}  cl={cl_warm:+.3f}",
                      "#ff7f0e", npz_path, with_footprints=True)
        draw_traj(ax, full_p,
                  f"{args.prism_label}  cl={cl_prism:+.3f}",
                  "#d62728", npz_path, with_footprints=True)

        # Perturbation arrow + label.
        # In the lanelet frame: source pose = (0, 0), perturbed pose = (dx, dy).
        # Arrow points from source → perturbed, both visible.
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
            ax.annotate(
                "", xy=(dx_pert, dy_pert), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.2),
                zorder=22,
            )
            ax.plot([0.0], [0.0], "o", color="#2ca02c", ms=8, mew=0, zorder=22)
            ax.text(0.0, -0.4, "source pose", color="#2ca02c",
                    fontsize=9, ha="center", zorder=22)

        all_pts = np.vstack([full_b[:, :2], full_w[:, :2], full_p[:, :2],
                             np.array([[0.0, 0.0]])])
        cx, cy = float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1]))
        half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="upper left")
        delta_pw = cl_prism - cl_warm
        delta_pb = cl_prism - cl_base
        ax.set_title(
            f"{Path(npz_path).name}\n{perturb_str}\n"
            f"t0_cl={t0_cl:+.3f}    "
            f"baseline cl={cl_base:+.3f}   warmstart cl={cl_warm:+.3f}   PRiSM cl={cl_prism:+.3f}\n"
            f"Δ(PRiSM−warmstart)={delta_pw:+.3f}   Δ(PRiSM−baseline)={delta_pb:+.3f}",
            fontsize=10,
        )
        out_path = out_root / f"scene_{si:03d}_{Path(npz_path).stem}.png"
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
        plt.close(fig)

        summary.append({
            "scene": str(npz_path),
            "scene_idx": si,
            "t0_cl": t0_cl,
            "cl_baseline": cl_base,
            "cl_warmstart": cl_warm,
            "cl_prism": cl_prism,
            "delta_prism_minus_warmstart": delta_pw,
            "delta_prism_minus_baseline": delta_pb,
            "png": str(out_path),
        })
        print(f"  [{si:3d}] base={cl_base:+.3f}  warm={cl_warm:+.3f}  PRiSM={cl_prism:+.3f}  "
              f"Δpw={delta_pw:+.3f}  Δpb={delta_pb:+.3f}  {Path(npz_path).name}")

    with open(out_root / "summary.json", "w") as f:
        json.dump({"scenes": summary}, f, indent=2)
    print(f"\nWrote {out_root}/summary.json  + {len(summary)} per-scene PNGs")


if __name__ == "__main__":
    main()
