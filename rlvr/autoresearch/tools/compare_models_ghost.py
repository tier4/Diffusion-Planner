#!/usr/bin/env python3
"""Ghost-overlay closed-loop sim comparing two models on the same scene.

Runs both models from identical initial conditions, renders per-step PNGs
with both ego footprints (different colors), planned trajectories, stopped
neighbor OBBs, and assembles a WebM clip.

Usage:
    python -m rlvr.autoresearch.tools.compare_models_ghost \
        --model_a <baseline.pth> --label_a baseline \
        --model_b <trained.pth>  --label_b "trained (ep27)" \
        --scenes scene_0038.npz scene_0037.npz ... \
        --output_dir /path/out --steps 80 --make_webm

    # Single scene:
    python -m rlvr.autoresearch.tools.compare_models_ghost \
        --model_a <baseline.pth> --model_b <trained.pth> \
        --scenes scene_0038.npz --output_dir /path/out --make_webm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from preference_optimization.utils import load_npz_data

from rlvr.autoresearch.tools.ghost_sim_common import (
    GhostSimConfig,
    extract_stopped_neighbors,
    load_model,
    run_ghost_sim,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_a", required=True, help="First model (e.g. baseline)")
    parser.add_argument("--lora_a", default=None)
    parser.add_argument("--label_a", default="baseline")
    parser.add_argument("--model_b", default=None, help="Second model; omit with --policy_dir")
    parser.add_argument("--lora_b", default=None)
    parser.add_argument("--label_b", default="trained")
    parser.add_argument("--scenes", nargs="+", required=True, help="NPZ scene paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--advance_k", type=int, default=0)
    parser.add_argument("--view_half_m", type=float, default=30.0)
    parser.add_argument("--ego_wheelbase", type=float, default=4.76,
                        help="Ego wheelbase (m); ego footprint is rear-axle offset by (length-wheelbase)/2")
    parser.add_argument("--make_webm", action="store_true")
    parser.add_argument("--hist_steps", type=int, default=0,
                        help="render N recorded-history frames (gray ego) before the sim")
    parser.add_argument("--n_candidates", type=int, default=0,
                        help="per step, also sample N etas from the policy "
                             "distribution and render their plans as a faint fan")
    parser.add_argument("--webm_fps", type=int, default=10)
    parser.add_argument("--show_lateral", action="store_true",
                        help="Show lateral offset to route centerline")
    parser.add_argument("--policy_dir", default=None,
                        help="exploration-policy dir: model B = model A + guidance "
                             "(per-step policy etas via composer); --model_b ignored")
    parser.add_argument("--sg_smooth", action="store_true",
                        help="Savitzky-Golay smooth both legs' per-step plans "
                             "(11/3) — matches scenario_generation.replay behavior")
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--head_protect", type=int, default=0,
                        help="zero guidance on the first N plan steps "
                             "(closed-loop stall fix; 0 = off)")
    parser.add_argument("--speed_floor", type=float, default=0.0,
                        help="add stock band speed guidance with v_low = this "
                             "fraction of current ego speed (e.g. 0.85) to the "
                             "guided leg — prevents the closed-loop speed-decay "
                             "spiral (the Branch Editor pairs swerve with speed "
                             "guidance the same way). 0 = off")
    parser.add_argument("--speed_scale", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare] loading model A: {args.label_a}")
    model_a, args_a = load_model(args.model_a, args.lora_a, device)
    policy = None
    if args.policy_dir:
        from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy
        policy, policy_heads = load_policy(args.policy_dir, args_a, device)
        model_b, args_b = model_a, args_a
        print(f"[compare] model B = model A + explorer ({args.policy_dir})")
    elif args.model_b:
        print(f"[compare] loading model B: {args.label_b}")
        model_b, args_b = load_model(args.model_b, args.lora_b, device)
    else:
        raise SystemExit("pass either --model_b or --policy_dir")

    def _sg(traj):
        if not args.sg_smooth:
            return traj
        from rlvr.grpo_sft_trainer import _smooth_trajectory
        return _smooth_trajectory(traj, 11, 3)

    def make_predict_fns(eta_log):
        """(predict_a, predict_b): plain[+SG] vs explorer-guided[+SG]."""
        from exploration_policy.utils import run_frozen_encoder
        from rlvr.autoresearch.tools.eval_policy_avoidance import make_composer
        from rlvr.autoresearch.tools.recovery_sim import deterministic_predict
        from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
        from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data

        def predict_a(model, model_args, data):
            return _sg(deterministic_predict(model, model_args, data))

        def predict_b(model, model_args, data):
            det = deterministic_predict(model, model_args, data)
            if policy is None:
                return _sg(det)
            batch = _stack_scene_data([data], device)
            norm = _normalize_batch(batch, model_args)
            x_ref = torch.from_numpy(np.ascontiguousarray(det)).float()
            x_ref = x_ref.unsqueeze(0).to(device)
            norm["reference_trajectory"] = x_ref
            enc = run_frozen_encoder(model, norm)
            out = policy(enc, x_ref, deterministic=True)
            N = max(0, args.n_candidates)
            etas = {
                h: torch.cat([
                    (2.0 * out.dists[h].mean - 1.0).reshape(1),
                    (2.0 * out.dists[h].rsample((N,)).reshape(-1) - 1.0),
                ]) if N else (2.0 * out.dists[h].mean - 1.0).reshape(1)
                for h in policy_heads
            }
            eta_log.append({h: float(v[0].item()) for h, v in etas.items()})
            B = 1 + N
            gen = dict(norm)
            if N:
                for k, v in norm.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                        gen[k] = v.expand(B, *v.shape[1:]).contiguous()
            composer = make_composer(etas, args)
            if args.speed_floor > 0:
                from diffusion_planner.model.guidance.config import GuidanceConfig
                ecs = data["ego_current_state"]
                ecs = ecs[0] if ecs.dim() == 2 else ecs
                v_now = float(torch.linalg.vector_norm(ecs[4:6]).item())
                composer._functions.append(__import__(
                    "diffusion_planner.model.guidance.registry",
                    fromlist=["build"]).build(GuidanceConfig(
                        name="speed", enabled=True, scale=args.speed_scale,
                        params={"v_low": args.speed_floor * v_now,
                                "v_high": max(1.2 * v_now, 8.0)})))
            trajs = _batched_generate_varied_noise(
                model, model_args, gen, noise_min=0.0, noise_max=0.0,
                first_deterministic=False, composer=composer, device=device,
            ).cpu().numpy()
            guided = _sg(trajs[0])
            if N:
                return guided, [trajs[i] for i in range(1, B)]
            return guided

        return predict_a, predict_b

    cfg = GhostSimConfig(
        model_a_label=args.label_a,
        model_b_label=args.label_b,
        view_half_m=args.view_half_m,
        steps=args.steps,
        advance_k=args.advance_k,
        webm_fps=args.webm_fps,
        hist_steps=args.hist_steps,
        show_lateral=args.show_lateral,
        ego_wheelbase=args.ego_wheelbase,
    )

    out_root = Path(args.output_dir)

    for scene_path in args.scenes:
        scene_name = Path(scene_path).stem
        print(f"\n=== {scene_name} ===")

        data = load_npz_data(scene_path, device)
        nb_boxes = extract_stopped_neighbors(scene_path)
        if nb_boxes:
            print(f"  {len(nb_boxes)} stopped neighbor(s)")

        scene_out = out_root / scene_name if len(args.scenes) > 1 else out_root
        cfg.subtitle = scene_name

        eta_log: list[dict] = []
        predict_a, predict_b = make_predict_fns(eta_log)

        def eta_title(step, a_pose, b_pose):
            if not eta_log or step >= len(eta_log):
                return ""
            e = eta_log[step]
            return "  explorer η: " + " ".join(f"{h[:3]}={v:+.2f}" for h, v in e.items())

        run_ghost_sim(
            scene_path=scene_path,
            model_a=model_a, model_a_args=args_a,
            model_b=model_b, model_b_args=args_b,
            scene_data=data,
            output_dir=scene_out,
            cfg=cfg,
            neighbor_boxes=nb_boxes,
            make_webm=args.make_webm,
            extra_title_fn=eta_title if policy is not None else None,
            predict_fn_a=predict_a if args.sg_smooth else None,
            predict_fn_b=predict_b if (policy is not None or args.sg_smooth) else None,
        )


if __name__ == "__main__":
    main()
