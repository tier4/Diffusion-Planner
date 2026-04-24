#!/usr/bin/env python3
"""Filter candidate training scenes by whether rl_cl guidance can actually
recover the ego's centerline on them.

For each scene NPZ, runs K trajectories with the configured generation
variant (matches training), scores each with reward.py, and keeps only
scenes where:

    max(best_of_K cl_score) - det_cl_score >= improvement_threshold

Drops scenes where even the guidance can't beat the deterministic output
by the selective threshold. Per ``feedback_no_poison_scenes.md``: if the
guidance can't meaningfully recover, training on the scene teaches the
model the WRONG thing.

Usage:
    python -m scenario_generation.tools.verify_recoverable_scenes \\
        --model_path /path/to/merged_or_base.pth \\
        --config /path/to/grpo_config.json \\
        --scenes /path/to/candidate_scenes.json \\
        --output /path/to/recoverable_scenes.json \\
        [--improvement_threshold 0.2] \\
        [--max_scenes 500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True,
                   help="GRPO config JSON (same one used for training)")
    p.add_argument("--scenes", type=Path, required=True,
                   help="JSON list of NPZ paths to verify")
    p.add_argument("--output", type=Path, required=True,
                   help="Filtered JSON list of recoverable scene NPZ paths")
    p.add_argument("--improvement_threshold", type=float, default=0.2,
                   help="Min best_of_K-det reward improvement to keep a scene")
    p.add_argument("--max_scenes", type=int, default=None,
                   help="Cap on # scenes to test (randomly sub-sampled if set)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import sys
    sys.path.insert(0, "/home/danielsanchez/Diffusion-Planner")
    from rlvr.grpo_config import GRPOConfig
    from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
    from rlvr.reward import compute_reward_batch
    from preference_optimization.utils import load_npz_data
    from preference_optimization.lora_utils import load_lora_checkpoint
    from diffusion_planner.utils.config import Config
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from rlvr.grpo_trainer_batched import build_cl_spd_configs, build_noise_configs
    from rlvr.generation_variants import get_variant
    from guidance_gui.generate_samples import generate_samples

    with open(args.config) as f:
        cfg = json.load(f)
    grpo = GRPOConfig.from_json(str(args.config))
    reward_cfg = load_reward_config(str(args.config))

    # Load base model (we don't apply the trained LoRA here — we want to
    # see whether the guidance alone can recover on each scene, so we
    # run against the BASE model. This prevents filtering by already-
    # partially-trained LoRA.).
    args_json = args.model_path.parent / "args.json"
    if not args_json.exists():
        # Assume sibling dir has it
        args_json = args.model_path.parent.parent / "args.json"
    if not args_json.exists():
        args_json = Path("/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/v4.0/args.json")
    model_args = Config(str(args_json))
    model = Diffusion_Planner(model_args).to(args.device)
    ckpt = torch.load(str(args.model_path), map_location=args.device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.eval()

    scenes = json.load(open(args.scenes))
    if args.max_scenes and len(scenes) > args.max_scenes:
        import random; random.seed(42)
        scenes = random.sample(scenes, args.max_scenes)
    print(f"Verifying {len(scenes)} scenes  threshold >= {args.improvement_threshold}")

    variant = get_variant(grpo.generation_variant)
    cl_spd_configs = build_cl_spd_configs(variant, grpo)
    noise_configs = build_noise_configs(variant)
    K = 1 + len(cl_spd_configs) + len(noise_configs)
    print(f"  K={K}  variant={grpo.generation_variant}  use_route_cl={grpo.use_route_cl_guidance}")

    kept, dropped_gate, dropped_no_gain = [], 0, 0
    improvements = []
    for i, path in enumerate(scenes):
        try:
            data = load_npz_data(path, args.device)
        except Exception as e:
            print(f"  [skip] {Path(path).name}: {e}"); continue

        # Gate prefilter — reward on det alone to check existing safety
        normalizer = model_args.observation_normalizer
        norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        norm = normalizer(norm)
        det = generate_samples(model, model_args, norm, noise_scale=0.0,
                               n_samples=1, composer=None, device=args.device)
        det_t = torch.tensor(det, device=args.device, dtype=torch.float32)
        det_reward = compute_reward_batch(det_t, data, reward_cfg)[0]
        if det_reward.rb_crossing or det_reward.collision_step is not None:
            dropped_gate += 1; continue

        # K generations with guidance (same routine training uses)
        # Minimal path: generate_samples with composer for each slot; then best-of-K
        # This is slower than batched but sufficient for pre-filtering
        best_cl = det_reward.centerline
        for slot in cl_spd_configs[:min(4, len(cl_spd_configs))]:  # top 4 slots suffice for recoverability test
            try:
                from guidance_gui.compose import compose_guidance
                gsamples = generate_samples(
                    model, model_args, norm,
                    noise_scale=slot["noise"][1],
                    n_samples=1,
                    composer=compose_guidance(
                        model_args, norm,
                        cl_scale=slot["cl"], spd_scale=slot["spd"],
                        stretch=slot.get("stretch", 1.0),
                        use_route_cl=grpo.use_route_cl_guidance,
                    ),
                    device=args.device,
                )
                gt = torch.tensor(gsamples, device=args.device, dtype=torch.float32)
                greward = compute_reward_batch(gt, data, reward_cfg)[0]
                if greward.centerline > best_cl:
                    best_cl = greward.centerline
            except Exception as e:
                continue

        improvement = best_cl - det_reward.centerline
        improvements.append(improvement)
        if improvement >= args.improvement_threshold:
            kept.append(path)
        else:
            dropped_no_gain += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(scenes)}] kept={len(kept)}  "
                  f"dropped_gate={dropped_gate}  dropped_no_gain={dropped_no_gain}  "
                  f"mean_imp={np.mean(improvements):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(kept, open(args.output, "w"), indent=2)
    imp_arr = np.array(improvements) if improvements else np.array([0.0])
    print(f"\nKept {len(kept)} / {len(scenes)} scenes ({100*len(kept)/max(len(scenes),1):.1f}%)")
    print(f"  dropped by gate:    {dropped_gate}")
    print(f"  dropped by no gain: {dropped_no_gain}")
    print(f"  improvement dist: mean={imp_arr.mean():+.3f} p50={np.median(imp_arr):+.3f} "
          f"p75={np.percentile(imp_arr, 75):+.3f} p95={np.percentile(imp_arr, 95):+.3f}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
