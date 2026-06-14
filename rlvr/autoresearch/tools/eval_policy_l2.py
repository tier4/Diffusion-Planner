#!/usr/bin/env python3
"""Open-loop L2 (ADE) of explorer-GUIDED trajectories vs GT on a large val set.

Answers "what would running the explorer online cost in L2?": for every
scene, generate BOTH the plain det trajectory and the policy-guided one
(policy etas + guidance composer, noise=0), and compare their displacement
error against the recorded GT future on identical scenes. Also reports
wall-clock per scene for the policy+guided path (the online-latency axis).

Fully batched: one encoder pass + one det generation + one policy forward +
one guided generation per batch of scenes.

Usage:
    python -m rlvr.autoresearch.tools.eval_policy_l2 \
        --model_path <base.pth> --policy_dir <dir> --scenes <val.json> \
        --output_dir <dir> \
        [--lambda_lat 5.0] [--lat_scale 2.0] [--col_scale 9.0] \
        [--batch_size 32] [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig

import rlvr.guidance_batched  # noqa: F401
from exploration_policy.utils import run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data


def make_composer(etas, args, envelope=None):
    # Resolve each envelope knob explicit-arg -> persisted-envelope -> v1
    # constant, so a policy trained at a non-v1 envelope is scored for L2 at the
    # calibration its etas are bound to, not a hardcoded default.
    from exploration_policy.model import V1_GUIDANCE_ENVELOPE
    from rlvr.autoresearch.tools.eval_policy_avoidance import warn_guidance_envelope_override

    overrides = []

    def knob(name):
        cli = getattr(args, name, None)
        persisted = envelope.get(name) if envelope is not None else None
        if cli is not None:
            if persisted is not None and cli != persisted:
                overrides.append(f"{name}: CLI={cli} overrides persisted={persisted}")
            return cli
        return persisted if persisted is not None else V1_GUIDANCE_ENVELOPE[name]

    unmapped = set(etas) - {"lateral", "collision"}
    if unmapped:
        raise ValueError(
            f"policy heads {sorted(unmapped)} have no guidance mapping in "
            "this tool — evaluating without them would misrepresent the "
            "deployed config"
        )
    fns = []
    if "lateral" in etas:
        fns.append(
            GuidanceConfig(
                name="lateral",
                enabled=True,
                scale=knob("lat_scale"),
                params={"lambda_lat": knob("lambda_lat"), "eta_lat": etas["lateral"]},
            )
        )
    if "collision" in etas:
        fns.append(
            GuidanceConfig(
                name="collision_swerve_batched",
                enabled=True,
                scale=knob("col_scale"),
                params={"eta_col": etas["collision"], "range": knob("col_range")},
            )
        )
    composer = GuidanceComposer(
        GuidanceSetConfig(functions=fns, global_scale=knob("guidance_scale"))
    )
    if overrides:
        warn_guidance_envelope_override(overrides)
    return composer


def ade(pred_xy: torch.Tensor, gt_xy: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    """Mean displacement over valid GT steps, per scene. [B, T, 2] -> [B]."""
    d = (pred_xy - gt_xy).norm(dim=-1)  # [B, T]
    d = d * gt_valid
    return d.sum(dim=1) / gt_valid.sum(dim=1).clamp_min(1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--lambda_lat",
        type=float,
        default=None,
        help="override the policy's persisted guidance envelope",
    )
    parser.add_argument("--lat_scale", type=float, default=None)
    parser.add_argument("--col_scale", type=float, default=None)
    parser.add_argument("--col_range", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    if args.limit:
        scene_paths = scene_paths[: args.limit]
    print(f"[policy_l2] {len(scene_paths)} scenes, heads={heads}")

    det_ades, guided_ades, eta_abs, t_policy = [], [], [], []
    n_skipped = 0
    for start in range(0, len(scene_paths), args.batch_size):
        batch_paths = scene_paths[start : start + args.batch_size]
        datas = []
        for p in batch_paths:
            try:
                datas.append(load_npz_data(p, device))
            except Exception as e:  # noqa: BLE001
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  [skip] {Path(p).name}: {e}")
        if not datas:
            continue
        B = len(datas)
        batch = _stack_scene_data(datas, device)
        norm_batch = _normalize_batch(batch, model_args)

        # 1. det generation (plain planner)
        det = det_inference_batched(
            model, model_args, datas, device, norm_batch=norm_batch
        )  # [B, T, 4]

        # 2. policy forward + guided generation (the online path)
        t0 = time.perf_counter()
        norm_batch_g = dict(norm_batch)
        norm_batch_g["reference_trajectory"] = det
        enc = run_frozen_encoder(model, norm_batch_g)
        out = policy(enc, det, deterministic=True)
        etas = {h: (2.0 * out.dists[h].mean - 1.0) for h in heads}  # [B]
        composer = make_composer(etas, args, envelope=getattr(policy, "guidance_envelope", None))
        guided = _batched_generate_varied_noise(
            model,
            model_args,
            norm_batch_g,
            noise_min=0.0,
            noise_max=0.0,
            first_deterministic=False,
            composer=composer,
            device=device,
        )  # [B, T, 4]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_policy.append((time.perf_counter() - t0) / B)

        # 3. ADE vs GT
        gt = batch["ego_agent_future"]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        gt_xy = gt[..., :2]
        gt_valid = (gt_xy.abs().sum(dim=-1) > 1e-6).float()
        det_ades.append(ade(det[..., :2], gt_xy, gt_valid).cpu())
        guided_ades.append(ade(guided[..., :2], gt_xy, gt_valid).cpu())
        eta_abs.append(torch.stack([etas[h].abs().cpu() for h in heads], dim=-1).max(dim=-1).values)

        done = start + B
        if done % (args.batch_size * 20) < args.batch_size:
            d = torch.cat(det_ades)
            g = torch.cat(guided_ades)
            print(
                f"  [{done}/{len(scene_paths)}] det ADE={d.mean():.4f} "
                f"guided ADE={g.mean():.4f} Δ={(g.mean() - d.mean()) / d.mean() * 100:+.2f}%"
            )

    det_a = torch.cat(det_ades).numpy()
    g_a = torch.cat(guided_ades).numpy()
    e_a = torch.cat(eta_abs).numpy()

    def dist(a):
        return {
            "mean": float(a.mean()),
            "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    acting = e_a > 0.1
    report = {
        "n_scenes": int(len(det_a)),
        "n_skipped": n_skipped,
        "det_ade": dist(det_a),
        "guided_ade": dist(g_a),
        "delta_pct_mean": float((g_a.mean() - det_a.mean()) / det_a.mean() * 100),
        "per_scene_delta": dist(g_a - det_a),
        "n_acting_scenes": int(acting.sum()),
        "acting_delta_mean": float((g_a - det_a)[acting].mean()) if acting.any() else 0.0,
        "policy_path_sec_per_scene": float(np.mean(t_policy)),
        "guidance_args": {
            k: getattr(args, k)
            for k in ("lambda_lat", "lat_scale", "col_scale", "col_range", "guidance_scale")
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "policy_l2.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
