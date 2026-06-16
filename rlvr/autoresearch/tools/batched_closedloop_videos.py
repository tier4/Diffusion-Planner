#!/usr/bin/env python3
"""Batched closed-loop A/B videos: step ALL scenes together, render in a pool.

Phase 1 (GPU): two legs stepping every scene in lockstep —
  baseline leg: ONE batched deterministic generation per step (all scenes);
  explorer leg: det x_ref + policy etas + guided generation, batched the
  same way (the batched guidances accept [B] etas, one composer call).
Per-scene state update / SG filtering / pose integration mirror
recovery_sim.closed_loop_rollout_with_plans exactly (advance_k=0, dt=0.1).

Phase 2 (CPU pool): ghost_sim_common.run_ghost_sim in render-only mode
(precomputed rollouts; no model, no GPU) — per-step PNGs + WebM per scene.

Usage:
    python -m rlvr.autoresearch.tools.batched_closedloop_videos \
        --model_path <base.pth> --policy_dir <dir> --scenes <json> \
        --output_dir <dir> [--steps 80] [--chunk 25] [--workers 10] \
        [--lambda_lat 5.0] [--lat_scale 2.0] [--col_scale 9.0]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import torch

import rlvr.guidance_batched  # noqa: F401
from exploration_policy.utils import run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.recovery_test import transform_to_new_ego_frame
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_sft_trainer import _smooth_trajectory
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data


@torch.no_grad()
def batched_closed_loop(
    model,
    margs,
    scene_datas,
    device,
    policy=None,
    heads=None,
    gargs=None,
    n_steps: int = 80,
    dt: float = 0.1,
    chunk: int = 25,
):
    """Roll all scenes forward in lockstep. Returns (rollouts, eta_logs);
    rollouts match closed_loop_rollout_with_plans output per scene."""
    from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer

    N = len(scene_datas)
    datas = [
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        for d in scene_datas
    ]
    cum = [[0.0, 0.0, 1.0, 0.0] for _ in range(N)]  # x, y, cos, sin
    positions = [[np.array([0.0, 0.0, 0.0])] for _ in range(N)]
    plans_world = [[] for _ in range(N)]
    eta_logs = [[] for _ in range(N)]
    velocities = []
    for d in datas:
        ecs0 = d["ego_current_state"]
        ecs0 = ecs0[0] if ecs0.dim() == 2 else ecs0
        velocities.append([float(torch.linalg.vector_norm(ecs0[4:6]).item())])

    for step_i in range(n_steps):
        preds = [None] * N
        for c0 in range(0, N, chunk):
            idx = list(range(c0, min(c0 + chunk, N)))
            batch = _stack_scene_data([datas[i] for i in idx], device)
            norm = _normalize_batch(batch, margs)
            det = _batched_generate_varied_noise(
                model,
                margs,
                norm,
                noise_min=0.0,
                noise_max=0.0,
                first_deterministic=False,
                composer=None,
                device=device,
            )  # [B, T, 4]
            if policy is None:
                out = det.cpu().numpy()
            else:
                norm["reference_trajectory"] = det
                enc = run_frozen_encoder(model, norm)
                pout = policy(enc, det, deterministic=True)
                etas = {h: (2.0 * pout.dists[h].mean - 1.0).reshape(-1) for h in heads}
                out = (
                    _batched_generate_varied_noise(
                        model,
                        margs,
                        norm,
                        noise_min=0.0,
                        noise_max=0.0,
                        first_deterministic=False,
                        composer=make_composer(etas, gargs),
                        device=device,
                    )
                    .cpu()
                    .numpy()
                )
                for j, i in enumerate(idx):
                    eta_logs[i].append({h: float(etas[h][j].item()) for h in heads})
            for j, i in enumerate(idx):
                preds[i] = out[j]

        for i in range(N):
            pred = _smooth_trajectory(preds[i], 11, 3)
            cum_x, cum_y, cum_cos, cum_sin = cum[i]

            cur_xy = pred[:, :2].astype(np.float64)
            wx = cum_x + cum_cos * cur_xy[:, 0] - cum_sin * cur_xy[:, 1]
            wy = cum_y + cum_sin * cur_xy[:, 0] + cum_cos * cur_xy[:, 1]
            cur_h = np.arctan2(pred[:, 3], pred[:, 2])
            wh = (cur_h + math.atan2(cum_sin, cum_cos)).astype(np.float64)
            plans_world[i].append(np.stack([wx, wy, wh], axis=-1))

            nx_loc = float(pred[0, 0])
            ny_loc = float(pred[0, 1])
            ncos_loc = float(pred[0, 2])
            nsin_loc = float(pred[0, 3])
            nrm = float(np.hypot(ncos_loc, nsin_loc)) or 1.0
            ncos_loc /= nrm
            nsin_loc /= nrm

            dvx_loc = float(pred[1, 0] - pred[0, 0]) / dt
            dvy_loc = float(pred[1, 1] - pred[0, 1]) / dt
            new_vx = ncos_loc * dvx_loc + nsin_loc * dvy_loc
            new_vy = -nsin_loc * dvx_loc + ncos_loc * dvy_loc

            new_world_x = cum_x + cum_cos * nx_loc - cum_sin * ny_loc
            new_world_y = cum_y + cum_sin * nx_loc + cum_cos * ny_loc
            new_cum_cos = cum_cos * ncos_loc - cum_sin * nsin_loc
            new_cum_sin = cum_sin * ncos_loc + cum_cos * nsin_loc

            positions[i].append(
                np.array([new_world_x, new_world_y, math.atan2(new_cum_sin, new_cum_cos)])
            )
            velocities[i].append(float(np.hypot(new_vx, new_vy)))
            cum[i] = [new_world_x, new_world_y, new_cum_cos, new_cum_sin]

            data = datas[i]
            if "ego_agent_past" in data:
                eap = data["ego_agent_past"].clone()
                old_origin = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=eap.dtype, device=eap.device)
                T = eap.shape[1]
                eap = torch.cat(
                    [eap[:, 1:T], old_origin.view(1, 1, 4).expand(eap.shape[0], 1, 4)], dim=1
                )
                data["ego_agent_past"] = eap
            data = transform_to_new_ego_frame(data, nx_loc, ny_loc, ncos_loc, nsin_loc)
            if "ego_current_state" in data:
                ecs = data["ego_current_state"]
                ecs[..., 0] = 0.0
                ecs[..., 1] = 0.0
                ecs[..., 2] = 1.0
                ecs[..., 3] = 0.0
                ecs[..., 4] = float(new_vx)
                ecs[..., 5] = float(new_vy)
                data["ego_current_state"] = ecs
            datas[i] = data
        if (step_i + 1) % 10 == 0:
            print(f"  [batched-cl] step {step_i + 1}/{n_steps}")

    rollouts = []
    for i in range(N):
        rollouts.append(
            {
                "positions": np.stack(positions[i], axis=0),
                "plans_world": plans_world[i],
                "extra_plans_world": [[] for _ in range(n_steps)],
                "velocities": np.array(velocities[i]),
            }
        )
    return rollouts, eta_logs


def _render_one(job):
    """Worker: render one scene's ghost PNGs + webm from precomputed rollouts."""
    (
        scene_path,
        rollout_a,
        rollout_b,
        eta_log,
        out_dir,
        label_a,
        label_b,
        steps,
        hist_steps,
        webm_fps,
        lambda_spd,
    ) = job
    import matplotlib

    matplotlib.use("Agg")
    from rlvr.autoresearch.tools.ghost_sim_common import (
        GhostSimConfig,
        extract_stopped_neighbors,
        run_ghost_sim,
    )

    # Disambiguate same-basename scenes from different perturbation pools
    # (e.g. train_parallel/ and train_yaw/ both holding scene_0008_var02.npz).
    scene_name = f"{Path(scene_path).parent.name}__{Path(scene_path).stem}"
    cfg = GhostSimConfig(
        model_a_label=label_a,
        model_b_label=label_b,
        steps=steps,
        hist_steps=hist_steps,
        webm_fps=webm_fps,
    )
    cfg.subtitle = scene_name
    data = load_npz_data(scene_path, "cpu")

    def eta_title(step, a_pose, b_pose):
        if not eta_log or step >= len(eta_log):
            return ""
        parts = []
        for h, v in eta_log[step].items():
            if h == "stretch":
                # show the actual factor (1 + lambda_spd * eta), not raw eta
                parts.append(f"str×{1.0 + lambda_spd * v:.2f}")
            else:
                parts.append(f"{h[:3]}={v:+.2f}")
        return "  explorer η: " + " ".join(parts)

    run_ghost_sim(
        scene_path=scene_path,
        model_a=None,
        model_a_args=None,
        model_b=None,
        model_b_args=None,
        scene_data=data,
        output_dir=Path(out_dir) / scene_name,
        cfg=cfg,
        neighbor_boxes=extract_stopped_neighbors(scene_path),
        make_webm=True,
        extra_title_fn=eta_title,
        rollout_a=rollout_a,
        rollout_b=rollout_b,
    )
    return scene_name


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--hist_steps", type=int, default=30)
    parser.add_argument("--webm_fps", type=int, default=10)
    parser.add_argument("--label_a", default="baseline")
    parser.add_argument("--label_b", default="explorer")
    parser.add_argument(
        "--render_only", action="store_true", help="skip phase 1, render from saved rollouts.pkl"
    )
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument(
        "--envelope",
        choices=["v1", "v2"],
        default="v1",
        help="guidance envelope — must match the policy's training labels",
    )
    parser.add_argument("--lambda_col", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "rollouts.pkl"

    with open(args.scenes) as f:
        paths = json.load(f)

    if not args.render_only:
        from rlvr.autoresearch.tools.eval_det_avoidance import load_model
        from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, margs = load_model(args.model_path, device)
        policy, heads = load_policy(args.policy_dir, margs, device)
        datas = [load_npz_data(p, device) for p in paths]
        print(f"[phase1] baseline leg: {len(paths)} scenes x {args.steps} steps")
        ro_base, _ = batched_closed_loop(
            model, margs, datas, device, n_steps=args.steps, chunk=args.chunk
        )
        print(f"[phase1] explorer leg")
        ro_gui, eta_logs = batched_closed_loop(
            model,
            margs,
            datas,
            device,
            policy=policy,
            heads=heads,
            gargs=args,
            n_steps=args.steps,
            chunk=args.chunk,
        )
        with open(pkl, "wb") as f:
            pickle.dump({"paths": paths, "base": ro_base, "gui": ro_gui, "etas": eta_logs}, f)
        print(f"[phase1] saved {pkl}")
        del model, policy, datas
        torch.cuda.empty_cache()
    else:
        with open(pkl, "rb") as f:
            saved = pickle.load(f)
        paths, ro_base, ro_gui, eta_logs = (
            saved["paths"],
            saved["base"],
            saved["gui"],
            saved["etas"],
        )

    jobs = [
        (
            paths[i],
            ro_base[i],
            ro_gui[i],
            eta_logs[i],
            str(out_dir),
            args.label_a,
            args.label_b,
            args.steps,
            args.hist_steps,
            args.webm_fps,
            args.lambda_spd,
        )
        for i in range(len(paths))
    ]
    print(f"[phase2] rendering {len(jobs)} scenes with {args.workers} workers")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for name in pool.imap_unordered(_render_one, jobs):
            print(f"  [done] {name}")
    print("[phase2] all renders complete")


if __name__ == "__main__":
    main()
