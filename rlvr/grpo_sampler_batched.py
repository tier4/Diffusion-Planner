"""Batched diverse trajectory generation for GRPO training.

Generates K=16 trajectories per scene:
  - 1 deterministic (no guidance, no noise)
  - 7 centerline guidance at varying strengths + small noise (2 batched passes)
  - 8 random guidance configs (2 batched passes)

Total: ~5 forward passes instead of 16 sequential.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from guidance_gui.generate_samples import generate_samples
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_sampler import SamplerConfig


def _expand_data(norm_data, n, device):
    expanded = {}
    for k, v in norm_data.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == 1:
            expanded[k] = v.expand(n, *v.shape[1:]).contiguous()
        else:
            expanded[k] = v
    return expanded


def _batched_cl_pass(model, model_args, norm_data, n, cl_scale, noise_min, noise_max, device):
    """Batch of n trajectories with centerline guidance at cl_scale, varied noise."""
    fns = [GuidanceConfig("centerline_following", enabled=True, scale=cl_scale)]
    comp = GuidanceComposer(GuidanceSetConfig(functions=fns, global_scale=1.0))
    batch = _expand_data(norm_data, n, device)
    return _batched_generate_varied_noise(
        model, model_args, batch,
        noise_min=noise_min, noise_max=noise_max,
        first_deterministic=False, composer=comp, device=device,
    )


def generate_diverse_group_batched(
    model: nn.Module,
    model_args,
    data: dict[str, torch.Tensor],
    config: SamplerConfig,
    device: torch.device,
) -> torch.Tensor:
    """Generate K trajectories with CL-focused diversity in ~5 batched passes.

    Returns:
        [K, T, 4] tensor of ego trajectories.
    """
    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)

    K = config.n_trajectories
    noise_min, noise_max = config.noise_scale_range
    all_trajs = []

    # --- Pass 1: Deterministic (B=1) ---
    det = generate_samples(model, model_args, norm_data, 0.0, 1, None, device)
    all_trajs.append(torch.from_numpy(det[0]).to(device))

    # --- Pass 2: Low CL sweep (B=4, CL=2-5, noise 0-0.5) ---
    trajs_lo = _batched_cl_pass(model, model_args, norm_data, 4, 3.5, 0.0, 0.5, device)
    for i in range(4):
        all_trajs.append(trajs_lo[i])

    # --- Pass 3: High CL sweep (B=3, CL=7-10, noise 0.5-1.0) ---
    trajs_hi = _batched_cl_pass(model, model_args, norm_data, 3, 8.0, 0.5, 1.5, device)
    for i in range(3):
        all_trajs.append(trajs_hi[i])

    # --- Pass 4: Random CL+RB (B=4) ---
    n_rand1 = min(4, K - len(all_trajs))
    if n_rand1 > 0:
        fns1 = [GuidanceConfig("centerline_following", enabled=True, scale=random.uniform(2.0, 8.0))]
        if random.random() < 0.5:
            fns1.append(GuidanceConfig("road_border", enabled=True, scale=random.uniform(0.3, 1.5)))
        comp1 = GuidanceComposer(GuidanceSetConfig(functions=fns1, global_scale=random.uniform(0.3, 1.5)))
        batch1 = _expand_data(norm_data, n_rand1, device)
        trajs_r1 = _batched_generate_varied_noise(
            model, model_args, batch1,
            noise_min=noise_min, noise_max=noise_max,
            first_deterministic=False, composer=comp1, device=device,
        )
        for i in range(n_rand1):
            all_trajs.append(trajs_r1[i])

    # --- Pass 5: Noise-only or light guidance (B=remaining) ---
    n_rand2 = K - len(all_trajs)
    if n_rand2 > 0:
        fns2 = []
        if random.random() < 0.5:
            fns2.append(GuidanceConfig("centerline_following", enabled=True, scale=random.uniform(1.0, 5.0)))
        comp2 = GuidanceComposer(GuidanceSetConfig(functions=fns2, global_scale=random.uniform(0.2, 1.0))) if fns2 else None
        batch2 = _expand_data(norm_data, n_rand2, device)
        trajs_r2 = _batched_generate_varied_noise(
            model, model_args, batch2,
            noise_min=noise_min, noise_max=noise_max,
            first_deterministic=False, composer=comp2, device=device,
        )
        for i in range(n_rand2):
            all_trajs.append(trajs_r2[i])

    return torch.stack(all_trajs[:K])
