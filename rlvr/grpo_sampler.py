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
    enable_road_border: bool = True
    enable_speed: bool = True
    enable_lateral: bool = False
    enable_longitudinal: bool = False

    # Probability that each enabled type is included for a given trajectory
    guidance_prob: float = 0.5

    centerline_scale_range: tuple[float, float] = (0.5, 3.0)
    anchor_scale_range: tuple[float, float] = (0.5, 3.0)
    collision_scale_range: tuple[float, float] = (0.5, 2.0)
    route_following_scale_range: tuple[float, float] = (0.5, 2.0)
    lane_keeping_scale_range: tuple[float, float] = (0.5, 2.0)
    road_border_scale_range: tuple[float, float] = (0.2, 1.5)

    # Lateral offset range in metres (positive=left, negative=right)
    lateral_offset_range: tuple[float, float] = (-2.0, 2.0)
    lateral_scale_range: tuple[float, float] = (0.5, 3.0)
    # Longitudinal time-shift range in timesteps (at dt=0.1s)
    longitudinal_shift_range: tuple[float, float] = (-10.0, 10.0)
    longitudinal_scale_range: tuple[float, float] = (0.5, 3.0)

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

    # Store deterministic trajectory as reference for lateral/longitudinal guidance.
    # Shape: [1, T, 4] in physical ego-centric metres (x, y, cos, sin).
    # ObservationNormalizer.inverse() passes unknown keys through unchanged.
    if config.enable_lateral or config.enable_longitudinal:
        ref_traj = torch.from_numpy(samples[0]).unsqueeze(0).to(device)  # [1, T, 4]
        norm_data["reference_trajectory"] = ref_traj

    # GT seed REMOVED: raw GT trajectory gets -27 reward due to bad centerline
    # score (heading conversion mismatch), causing GRPO to learn AWAY from GT.
    # The guided deterministic trajectory (LK5+CL3+SPD5) serves as the on-road
    # example instead — it gets top-1 reward on problem scenes.

    # Compute GT speed bounds for speed guidance
    gt_max_speed = None
    gt_min_speed = 0.0
    if "ego_agent_future" in data:
        _gt = data["ego_agent_future"]
        if _gt.dim() == 3:
            _gt = _gt[0]
        _gt_np = _gt.cpu().numpy()
        _gt_valid = ~((_gt_np[:, 0] == 0) & (_gt_np[:, 1] == 0))
        if _gt_valid.sum() >= 10:
            _gt_vel = np.diff(_gt_np[_gt_valid][:, :2], axis=0) / 0.1
            _gt_speeds = np.linalg.norm(_gt_vel, axis=-1)
            gt_max_speed = float(_gt_speeds.max())
            # Use 10th percentile as min speed (avoids noise from near-stop moments)
            gt_min_speed = float(np.percentile(_gt_speeds, 10))

    # Trajectory 2: guided deterministic — strong LK+CL+SPD produces 0% offroad
    # on problem miraikan scenes (verified: LK=5, CL=3, SPD=5 eliminates all offroad).
    guided_fns = [
        GuidanceConfig("lane_keeping", enabled=True, scale=5.0),
        GuidanceConfig("road_border", enabled=True, scale=1.0),
        GuidanceConfig("route_following", enabled=True, scale=1.0),
    ]
    if gt_max_speed is not None:
        guided_fns.append(GuidanceConfig(
            name="speed", enabled=True, scale=5.0,
            params={"v_high": gt_max_speed, "v_low": gt_min_speed},
        ))
    guided_composer = None
    guided_set_cfg = None
    if guided_fns:
        guided_set_cfg = GuidanceSetConfig(functions=guided_fns, global_scale=1.0)
        guided_composer = GuidanceComposer(guided_set_cfg)
    # Generate multiple guided trajectories with varying scales to provide
    # more on-road examples. Each uses zero noise + different guidance strength.
    for g_idx, (lk_s, rb_s, gs_val) in enumerate([
        (5.0, 1.0, 1.0),   # strong guidance
        (3.0, 0.5, 0.5),   # medium guidance
        (2.0, 0.2, 0.3),   # light guidance
    ]):
        g_fns = [
            GuidanceConfig("lane_keeping", enabled=True, scale=lk_s),
            GuidanceConfig("road_border", enabled=True, scale=rb_s),
            GuidanceConfig("route_following", enabled=True, scale=1.0),
        ]
        if gt_max_speed is not None:
            g_fns.append(GuidanceConfig(
                name="speed", enabled=True, scale=5.0,
                params={"v_high": gt_max_speed, "v_low": gt_min_speed},
            ))
        g_set = GuidanceSetConfig(functions=g_fns, global_scale=gs_val)
        g_comp = GuidanceComposer(g_set)
        g_samples = generate_samples(
            model=model, model_args=model_args, data=norm_data,
            noise_scale=0.0, n_samples=1, composer=g_comp, device=device,
        )
        results.append(SampledTrajectory(
            trajectory=g_samples[0],
            noise_scale=0.0,
            guidance_config=g_set,
            is_deterministic=False,
            label=f"guided_{g_idx}",
        ))

    # Remaining trajectories: diverse random configs
    # Account for det(1) + gt_seed(0-1) + guided_det(1) already added
    n_fixed = len(results)
    for _ in range(n_fixed, config.n_trajectories):
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

            if config.enable_road_border and random.random() < config.guidance_prob:
                rb_scale = random.uniform(*config.road_border_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="road_border", enabled=True, scale=rb_scale,
                ))
                label_parts.append(f"rb={rb_scale:.1f}")

            # Lateral guidance: perpendicular offset from deterministic reference
            if config.enable_lateral and random.random() < config.guidance_prob:
                lat_offset = random.uniform(*config.lateral_offset_range)
                lat_scale = random.uniform(*config.lateral_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="lateral", enabled=True, scale=lat_scale,
                    params={"lateral_offset": lat_offset},
                ))
                label_parts.append(f"lat={lat_offset:+.1f}")

            # Longitudinal guidance: time-shift along deterministic reference
            if config.enable_longitudinal and random.random() < config.guidance_prob:
                time_shift = random.uniform(*config.longitudinal_shift_range)
                long_scale = random.uniform(*config.longitudinal_scale_range)
                guidance_fns.append(GuidanceConfig(
                    name="longitudinal", enabled=True, scale=long_scale,
                    params={"time_shift": time_shift},
                ))
                label_parts.append(f"lon={time_shift:+.1f}")

            # Speed guidance caps speed at GT max when enabled and GT available
            if config.enable_speed and gt_max_speed is not None:
                spd_scale = random.uniform(3.0, 8.0)
                guidance_fns.append(GuidanceConfig(
                    name="speed", enabled=True, scale=spd_scale,
                    params={"v_high": gt_max_speed, "v_low": gt_min_speed},
                ))
                label_parts.append(f"spd={spd_scale:.1f}")

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
