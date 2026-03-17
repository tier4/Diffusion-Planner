"""Diverse N-trajectory generation for GRPO training.

Generates N trajectories per scene with randomized guidance and noise
configurations, producing a group of candidates for reward scoring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from guidance_gui.generate_samples import generate_samples


@dataclass
class SamplerConfig:
    n_trajectories: int = 8
    noise_scale_range: tuple[float, float] = (0.5, 4.0)
    guidance_scale_range: tuple[float, float] = (0.1, 2.0)

    enable_guidance: bool = True
    # Per-type toggles: only enabled types enter the random pool
    enable_centerline: bool = True
    enable_anchor: bool = True
    enable_collision: bool = False
    enable_route_following: bool = False
    enable_lane_keeping: bool = False

    # Probability that each enabled type is included for a given trajectory
    guidance_prob: float = 0.5

    centerline_scale_range: tuple[float, float] = (0.5, 3.0)
    anchor_scale_range: tuple[float, float] = (0.5, 3.0)
    collision_scale_range: tuple[float, float] = (0.5, 2.0)
    route_following_scale_range: tuple[float, float] = (0.5, 2.0)
    lane_keeping_scale_range: tuple[float, float] = (0.5, 2.0)

    prototypes_path: str | None = None
    num_prototypes: int = 16


@dataclass
class SampledTrajectory:
    trajectory: np.ndarray               # (T, 4)
    noise_scale: float
    guidance_config: GuidanceSetConfig | None
    is_deterministic: bool
    label: str


def _detect_num_prototypes(path: str) -> int | None:
    """Return number of prototypes, or None if the file is missing/unloadable."""
    if not Path(path).exists():
        return None
    try:
        protos = np.load(path)
        return protos.shape[0]
    except Exception:
        return None


def generate_diverse_group(
    model,
    model_args,
    data: dict[str, torch.Tensor],
    config: SamplerConfig,
    device: torch.device,
) -> list[SampledTrajectory]:
    """Generate N trajectories with diverse noise and guidance configurations.

    Args:
        model: Loaded Diffusion_Planner instance (eval mode).
        model_args: Config object from load_model.
        data: Raw observation dict from load_npz_data (NOT normalized).
        config: SamplerConfig controlling diversity.
        device: Torch device.

    Returns:
        List of N SampledTrajectory instances.
    """
    num_protos = config.num_prototypes
    prototypes_valid = False
    if config.prototypes_path is not None:
        detected = _detect_num_prototypes(config.prototypes_path)
        if detected is not None:
            num_protos = detected
            prototypes_valid = True
        else:
            print(
                f"Warning: prototypes file not found or unloadable: "
                f"{config.prototypes_path} -- anchor guidance disabled"
            )

    norm_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }
    norm_data = model_args.observation_normalizer(norm_data)

    results: list[SampledTrajectory] = []

    # Trajectory 0: deterministic (no noise, no guidance)
    samples = generate_samples(
        model=model,
        model_args=model_args,
        data=norm_data,
        noise_scale=0.0,
        n_samples=1,
        composer=None,
        device=device,
    )
    results.append(SampledTrajectory(
        trajectory=samples[0],
        noise_scale=0.0,
        guidance_config=None,
        is_deterministic=True,
        label="det",
    ))

    # Trajectories 1..N-1: diverse random configs
    for _ in range(1, config.n_trajectories):
        ns = random.uniform(*config.noise_scale_range)
        gs = random.uniform(*config.guidance_scale_range)

        guidance_fns: list[GuidanceConfig] = []
        label_parts = [f"ns={ns:.1f}"]

        if config.enable_guidance:
            # Each enabled type independently coin-flipped
            if config.enable_centerline and random.random() < config.guidance_prob:
                cl_scale = random.uniform(*config.centerline_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="centerline_following", enabled=True, scale=cl_scale,
                ))
                label_parts.append(f"cl={cl_scale:.1f}")

            if (
                config.enable_anchor
                and config.prototypes_path is not None
                and prototypes_valid
                and random.random() < config.guidance_prob
            ):
                anchor_idx = random.randint(0, num_protos - 1)
                anc_scale = random.uniform(*config.anchor_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="anchor_following", enabled=True, scale=anc_scale,
                    params={
                        "prototypes_path": config.prototypes_path,
                        "anchor_index": anchor_idx,
                    },
                ))
                label_parts.append(f"anchor#{anchor_idx}")

            if config.enable_collision and random.random() < config.guidance_prob:
                col_scale = random.uniform(*config.collision_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="collision", enabled=True, scale=col_scale,
                ))
                label_parts.append(f"col={col_scale:.1f}")

            if config.enable_route_following and random.random() < config.guidance_prob:
                rf_scale = random.uniform(*config.route_following_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="route_following", enabled=True, scale=rf_scale,
                ))
                label_parts.append(f"rf={rf_scale:.1f}")

            if config.enable_lane_keeping and random.random() < config.guidance_prob:
                lk_scale = random.uniform(*config.lane_keeping_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="lane_keeping", enabled=True, scale=lk_scale,
                ))
                label_parts.append(f"lk={lk_scale:.1f}")

        composer = None
        set_config = None
        if guidance_fns:
            set_config = GuidanceSetConfig(
                functions=guidance_fns,
                global_scale=gs,
            )
            composer = GuidanceComposer(set_config)

        samples = generate_samples(
            model=model,
            model_args=model_args,
            data=norm_data,
            noise_scale=ns,
            n_samples=1,
            composer=composer,
            device=device,
        )

        results.append(SampledTrajectory(
            trajectory=samples[0],
            noise_scale=ns,
            guidance_config=set_config,
            is_deterministic=False,
            label="+".join(label_parts),
        ))

    return results
