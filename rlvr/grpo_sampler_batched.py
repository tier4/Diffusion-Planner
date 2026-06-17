"""Batched diverse trajectory generation for GRPO training.

Generates K=16 trajectories per scene:
  - 1 deterministic (no guidance, no noise)
  - 7 centerline guidance at varying strengths + small noise (2 batched passes)
  - 8 random guidance configs (2 batched passes)

Total: ~5 forward passes instead of 16 sequential.
"""

from __future__ import annotations

import random

import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from torch import nn

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


def _batched_cl_spd_pass(
    model, model_args, norm_data, n, cl_scale, spd_scale, v_high, noise_min, noise_max, device
):
    """Batch of n trajectories with CL + SPD guidance, varied noise."""
    fns = [
        GuidanceConfig("centerline_following", enabled=True, scale=cl_scale),
        GuidanceConfig(
            "speed", enabled=True, scale=spd_scale, params={"v_high": v_high, "v_low": 0.5}
        ),
    ]
    comp = GuidanceComposer(GuidanceSetConfig(functions=fns, global_scale=1.0))
    batch = _expand_data(norm_data, n, device)
    return _batched_generate_varied_noise(
        model,
        model_args,
        batch,
        noise_min=noise_min,
        noise_max=noise_max,
        first_deterministic=False,
        composer=comp,
        device=device,
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
    norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    norm_data = model_args.observation_normalizer(norm_data)

    K = config.n_trajectories
    noise_min, noise_max = config.noise_scale_range
    all_trajs = []

    # --- Pass 1: Deterministic (B=1) ---
    det = generate_samples(model, model_args, norm_data, 0.0, 1, None, device)
    all_trajs.append(torch.from_numpy(det[0]).to(device))

    # Compute GT max speed for speed guidance
    import numpy as _np

    if "ego_agent_future" in data:
        _gt = data["ego_agent_future"]
        if _gt.dim() == 3:
            _gt = _gt[0]
        _gt_np = _gt.cpu().numpy()
        _gt_valid = ~((_gt_np[:, 0] == 0) & (_gt_np[:, 1] == 0))
        if _gt_valid.sum() >= 5:
            _gt_vel = _np.diff(_gt_np[_gt_valid][:, :2], axis=0) / 0.1
            gt_v_high = float(_np.linalg.norm(_gt_vel, axis=-1).max())
        else:
            gt_v_high = 3.0
    else:
        gt_v_high = 3.0

    # --- Pass 2-5: Strong CL + SPD guidance sweep for lane keeping (8 trajectories) ---
    # 8 guided at CL5-10 to ensure ~8-10/16 stay in-lane on hard curves.
    cl_spd_configs = [
        (5.0, 5.0, 0.0, 0.0),  # CL5+SPD5, deterministic
        (8.0, 5.0, 0.0, 0.0),  # CL8+SPD5, deterministic
        (10.0, 8.0, 0.0, 0.0),  # CL10+SPD8, deterministic
        (10.0, 10.0, 0.0, 0.0),  # CL10+SPD10, deterministic
        (5.0, 5.0, 0.3, 0.8),  # CL5+SPD5, noise
        (8.0, 8.0, 0.3, 0.8),  # CL8+SPD8, noise
        (10.0, 8.0, 0.3, 0.8),  # CL10+SPD8, noise
        (10.0, 10.0, 0.5, 1.0),  # CL10+SPD10, noise
    ]
    for cl_scale, spd_scale, n_min, n_max in cl_spd_configs:
        fns_cl = [
            GuidanceConfig("centerline_following", enabled=True, scale=cl_scale),
            GuidanceConfig(
                "speed", enabled=True, scale=spd_scale, params={"v_high": gt_v_high, "v_low": 0.5}
            ),
        ]
        comp_cl = GuidanceComposer(GuidanceSetConfig(functions=fns_cl, global_scale=1.0))
        noise_scale = random.uniform(n_min, n_max) if n_max > 0 else 0.0
        cl_traj = generate_samples(model, model_args, norm_data, noise_scale, 1, comp_cl, device)
        all_trajs.append(torch.from_numpy(cl_traj[0]).to(device))

    # --- Pass 6: Random CL+RB (fill remaining except last 3 for noise-only) ---
    n_rand1 = max(0, min(4, K - len(all_trajs) - 3))
    if n_rand1 > 0:
        fns1 = [
            GuidanceConfig("centerline_following", enabled=True, scale=random.uniform(2.0, 8.0))
        ]
        if random.random() < 0.5:
            fns1.append(GuidanceConfig("road_border", enabled=True, scale=random.uniform(0.3, 1.5)))
        comp1 = GuidanceComposer(
            GuidanceSetConfig(functions=fns1, global_scale=random.uniform(0.3, 1.5))
        )
        batch1 = _expand_data(norm_data, n_rand1, device)
        trajs_r1 = _batched_generate_varied_noise(
            model,
            model_args,
            batch1,
            noise_min=noise_min,
            noise_max=noise_max,
            first_deterministic=False,
            composer=comp1,
            device=device,
        )
        for i in range(n_rand1):
            all_trajs.append(trajs_r1[i])

    # --- Pass 7: Noise-only or light guidance (B=remaining) ---
    n_rand2 = K - len(all_trajs)
    if n_rand2 > 0:
        fns2 = []
        if random.random() < 0.5:
            fns2.append(
                GuidanceConfig("centerline_following", enabled=True, scale=random.uniform(1.0, 5.0))
            )
        comp2 = (
            GuidanceComposer(
                GuidanceSetConfig(functions=fns2, global_scale=random.uniform(0.2, 1.0))
            )
            if fns2
            else None
        )
        batch2 = _expand_data(norm_data, n_rand2, device)
        trajs_r2 = _batched_generate_varied_noise(
            model,
            model_args,
            batch2,
            noise_min=noise_min,
            noise_max=noise_max,
            first_deterministic=False,
            composer=comp2,
            device=device,
        )
        for i in range(n_rand2):
            all_trajs.append(trajs_r2[i])

    return torch.stack(all_trajs[:K])
