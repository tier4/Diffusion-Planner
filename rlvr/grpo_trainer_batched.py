"""Fully batched GRPO trainer: all scenes × all trajectories in ~5 forward passes.

Replaces the sequential scene loop with cross-scene batching. Each guidance
config is applied to ALL scenes simultaneously, producing N_scenes trajectories
per pass. Noise varies per element for diversity.

Layout per epoch:
  1. Load all scene data, stack into batch
  2. For each of ~5 guidance configs: run batched generation for all scenes
  3. Score per-scene (sequential — neighbor data differs per scene)
  4. Stack all scenes' trajectories + advantages for batched GRPO loss
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from torch import nn
from tqdm import tqdm

from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.generation_variants import get_variant
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_batched_grpo_loss
from rlvr.reward import RewardConfig, compute_group_advantages, compute_reward_batch


def _stack_scene_data(all_data: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack N scene dicts (each B=1) into one batch dict (B=N)."""
    batch = {}
    for k in all_data[0]:
        vals = [d[k] for d in all_data]
        if isinstance(vals[0], torch.Tensor):
            batch[k] = torch.cat(vals, dim=0)  # [N, ...]
        else:
            batch[k] = vals[0]
    return batch


def _normalize_batch(batch_data: dict, model_args) -> dict:
    normalizer = copy.deepcopy(model_args.observation_normalizer)
    norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
    return normalizer(norm)


def _chunked_generate(model, model_args, norm_batch, noise_min, noise_max, composer, device, chunk_size=64):
    """Generate trajectories in chunks to avoid OOM."""
    N = norm_batch["ego_current_state"].shape[0]
    all_out = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = {k: v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == N else v
                 for k, v in norm_batch.items()}
        out = _batched_generate_varied_noise(
            model, model_args, chunk,
            noise_min=noise_min, noise_max=noise_max,
            first_deterministic=False, composer=composer, device=device,
        )
        all_out.append(out)
    return torch.cat(all_out, dim=0)


def _build_cl_spd_configs(variant: str) -> list[dict]:
    """Return guided cl_spd slots for the variant (lookup in rlvr.generation_variants)."""
    return get_variant(variant).cl_spd_configs


def _build_noise_configs(variant: str) -> list[dict]:
    """Return noise-only slots for the variant (lookup in rlvr.generation_variants)."""
    return get_variant(variant).noise_configs


def get_generation_config_labels_for_variant(variant: str, K: int = 16) -> list[str]:
    """Full per-slot labels: det_pure + cl_spd configs + noise configs + random."""
    cl_spd = _build_cl_spd_configs(variant)
    noise = _build_noise_configs(variant)
    labels = ["det_pure"] + [c["label"] for c in cl_spd] + [c["label"] for c in noise]
    for i in range(len(labels), K):
        labels.append(f"random_{i}")
    return labels[:K]


def generate_all_scenes_batched(
    model: nn.Module,
    model_args,
    norm_batch: dict[str, torch.Tensor],
    K: int,
    noise_range: tuple[float, float],
    device: torch.device,
    gen_chunk_size: int = 64,
    gt_max_speed: float = 3.0,
    longitudinal_eta: float = 0.0,
    longitudinal_lambda: float = 0.5,
    longitudinal_scale: float = 10.0,
    lateral_eta: float = 0.0,
    lateral_lambda: float = 2.0,
    lateral_scale: float = 5.0,
    speed_stretch: float = 1.0,
    generation_variant: str = "default",
) -> torch.Tensor:
    """Generate K trajectories for all N scenes in ~5 chunked-batched passes.

    Args:
        longitudinal_eta: Longitudinal guidance eta (0=off, >0=faster than ref).
            Applied to CL-guided trajectories when nonzero.
        longitudinal_lambda: Speed scaling constant for longitudinal guidance.
        longitudinal_scale: Guidance scale for longitudinal guidance.
        lateral_eta: Lateral guidance eta (0=off, >0=push left, <0=push right).
            Applied to CL-guided trajectories when nonzero.
        lateral_lambda: Maximum lateral offset in metres for lateral guidance.
        lateral_scale: Guidance scale for lateral guidance.

    Returns:
        [N, K, T, 4] tensor.
    """
    N = norm_batch["ego_current_state"].shape[0]
    noise_min, noise_max = noise_range
    all_k_trajs = []

    # --- Config 1: Deterministic ---
    det_trajs = _chunked_generate(model, model_args, norm_batch, 0.0, 0.0, None, device, gen_chunk_size)
    all_k_trajs.append(det_trajs)

    # Use deterministic trajectory as reference for lon/lat guidance.
    # det_trajs is already in (x, y, cos_yaw, sin_yaw) format — no conversion needed.
    use_lon = abs(longitudinal_eta) > 1e-6
    use_lat = abs(lateral_eta) > 1e-6
    if use_lon or use_lat:
        norm_batch["reference_trajectory"] = det_trajs  # no clone needed, not mutated

    # --- Config 2-9: CL + SPD guidance sweep for lane keeping ---
    # 8 guided trajectories at CL5-10 to ensure ~8-10/16 stay in-lane on curves.
    # Variants can replace the 3 redundant slots with experimental configs.
    cl_spd_configs = _build_cl_spd_configs(generation_variant)
    use_stretch_global = abs(speed_stretch - 1.0) > 1e-6
    for cfg in cl_spd_configs:
        cl_scale = cfg["cl"]
        spd_scale = cfg["spd"]
        n_min, n_max = cfg["noise"]
        # Per-slot stretch overrides global, falls back to global if unset
        cfg_stretch = cfg.get("stretch", speed_stretch)
        cfg_has_noise = n_max > 0
        use_stretch_here = abs(cfg_stretch - 1.0) > 1e-6 and cfg_has_noise
        spd_params = {"stretch": cfg_stretch} if use_stretch_here else {"v_high": gt_max_speed, "v_low": 0.5}
        # Per-slot lateral overrides global lateral_eta
        cfg_lat_eta = cfg.get("lat_eta", lateral_eta)
        cfg_lat_lambda = cfg.get("lat_lambda", lateral_lambda)
        cfg_lat_scale = cfg.get("lat_scale", lateral_scale)
        cfg_use_lat = abs(cfg_lat_eta) > 1e-6
        # Optional per-slot collision guidance
        cfg_col = cfg.get("col", 0.0)
        # Build guidance functions
        fns = []
        if cl_scale > 0:
            fns.append(GuidanceConfig("centerline_following", enabled=True, scale=cl_scale))
        if spd_scale > 0:
            fns.append(GuidanceConfig("speed", enabled=True, scale=spd_scale, params=spd_params))
        if use_lon:
            fns.append(GuidanceConfig(
                "longitudinal", enabled=True, scale=longitudinal_scale,
                params={"eta_lon": longitudinal_eta, "lambda_lon": longitudinal_lambda},
            ))
        if cfg_use_lat:
            fns.append(GuidanceConfig(
                "lateral", enabled=True, scale=cfg_lat_scale,
                params={"eta_lat": cfg_lat_eta, "lambda_lat": cfg_lat_lambda},
            ))
        if cfg_col > 0:
            fns.append(GuidanceConfig("collision", enabled=True, scale=cfg_col))
        comp = GuidanceComposer(GuidanceSetConfig(functions=fns, global_scale=1.0)) if fns else None
        trajs = _chunked_generate(model, model_args, norm_batch, n_min, n_max, comp, device, gen_chunk_size)
        all_k_trajs.append(trajs)

    # Clean up reference_trajectory before random passes (not needed, wastes VRAM on expand)
    norm_batch.pop("reference_trajectory", None)

    # --- Noise-only slots (no guidance, fixed noise ranges) ---
    # Functionally part of the noise/exploration pool but with deterministic
    # noise ranges rather than random sampling.
    for noise_cfg in _build_noise_configs(generation_variant):
        n_min_s, n_max_s = noise_cfg["noise"]
        trajs = _chunked_generate(model, model_args, norm_batch, n_min_s, n_max_s, None, device, gen_chunk_size)
        all_k_trajs.append(trajs)

    # --- Random guidance pool (no road_border to avoid OOM) ---
    n_fixed = len(all_k_trajs)
    n_random = K - n_fixed

    n_per_pass = []
    remaining = n_random
    while remaining > 0:
        n = min(remaining, max(2, remaining // 2))
        n_per_pass.append(n)
        remaining -= n

    for n_pass in n_per_pass:
        fns = []
        if random.random() < 0.7:
            fns.append(GuidanceConfig("centerline_following", enabled=True,
                                       scale=random.uniform(2.0, 8.0)))
        # Skip road_border guidance in generation (causes OOM with large batches)
        # Road border avoidance is handled by the reward function instead
        gs = random.uniform(0.3, 1.5)
        comp = GuidanceComposer(GuidanceSetConfig(functions=fns, global_scale=gs)) if fns else None

        if n_pass > 1:
            expanded = {}
            for k, v in norm_batch.items():
                if isinstance(v, torch.Tensor):
                    expanded[k] = v.repeat(n_pass, *([1] * (v.dim() - 1)))
                else:
                    expanded[k] = v
        else:
            expanded = norm_batch

        trajs = _chunked_generate(model, model_args, expanded, noise_min, noise_max, comp, device, gen_chunk_size)

        T_len = trajs.shape[1]
        trajs = trajs.reshape(n_pass, N, T_len, 4)
        for i in range(n_pass):
            all_k_trajs.append(trajs[i])

    stacked = torch.stack(all_k_trajs[:K], dim=0)
    return stacked.permute(1, 0, 2, 3)


def train_epoch_batched(
    model: nn.Module,
    model_args,
    optimizer: torch.optim.Optimizer,
    scene_paths: list[str],
    config: GRPOConfig,
    reward_config: RewardConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Fully batched epoch: generate all trajs, score, train in bulk."""
    import gc

    # Free memory from previous epoch
    torch.cuda.empty_cache()
    gc.collect()

    K = config.num_generations
    keep = config.rejection_keep

    # 1. Load all scenes
    print(f"  Loading {len(scene_paths)} scenes...")
    all_data = []
    valid_paths = []
    for path in scene_paths:
        try:
            data = load_npz_data(path, device)
            all_data.append(data)
            valid_paths.append(path)
        except Exception as e:
            print(f"  [skip] {Path(path).name}: {e}")

    N = len(all_data)
    if N == 0:
        return {}

    # 2. Stack and normalize
    print(f"  Stacking {N} scenes into batch...")
    batch_data = _stack_scene_data(all_data, device)
    norm_batch = _normalize_batch(batch_data, model_args)

    # Compute per-scene GT max speed for speed guidance (use raw data, not normalized)
    import numpy as _np2
    gt_speeds_list = []
    for d in all_data:
        gt = d.get("ego_agent_future")
        if gt is not None:
            if gt.dim() == 3: gt = gt[0]
            gt_np = gt.cpu().numpy()
            gt_valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            if gt_valid.sum() >= 5:
                vel = _np2.diff(gt_np[gt_valid][:, :2], axis=0) / 0.1
                gt_speeds_list.append(float(_np2.linalg.norm(vel, axis=-1).max()))
            else:
                gt_speeds_list.append(3.0)
        else:
            gt_speeds_list.append(3.0)
    median_gt_speed = float(_np2.median(gt_speeds_list))
    print(f"  Median GT max speed: {median_gt_speed:.1f} m/s")

    # 2b. Apply per-epoch schedules to reward weights and guidance params
    scheduled = config.get_all_scheduled_values(epoch, config.train_epochs)
    reward_weight_names = {
        "w_progress", "w_safety", "w_smooth", "w_feasibility", "w_centerline",
        "stopped_penalty", "underprogress_penalty", "progress_norm_scale",
    }
    for name, value in scheduled.items():
        if name in reward_weight_names and hasattr(reward_config, name):
            setattr(reward_config, name, value)
    if scheduled:
        sched_str = ", ".join(f"{k}={v:.3f}" for k, v in scheduled.items())
        print(f"  [schedule] epoch {epoch}: {sched_str}")

    lon_eta = scheduled.get("longitudinal_eta", 0.0)
    lon_lambda = scheduled.get("longitudinal_lambda", config.lambda_lon)
    lon_scale = scheduled.get("longitudinal_scale", 10.0)
    lat_eta = scheduled.get("lateral_eta", 0.0)
    lat_lambda = scheduled.get("lateral_lambda", config.lambda_lat)
    lat_scale = scheduled.get("lateral_scale", 5.0)
    spd_stretch = scheduled.get("speed_stretch", 1.0)

    # 3. Generate K trajectories for all scenes (batched)
    print(f"  Generating {K} trajectories × {N} scenes (batched)...")
    model.eval()
    with torch.no_grad():
        all_trajs = generate_all_scenes_batched(
            model, model_args, norm_batch, K, config.noise_scale_range, device,
            gt_max_speed=median_gt_speed,
            longitudinal_eta=lon_eta,
            longitudinal_lambda=lon_lambda,
            longitudinal_scale=lon_scale,
            lateral_eta=lat_eta,
            lateral_lambda=lat_lambda,
            lateral_scale=lat_scale,
            speed_stretch=spd_stretch,
        )  # [N, K, T, 4]

    # Free generation memory before scoring + training
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Per-scene reward scoring (sequential — different neighbor data)
    print(f"  Scoring rewards...")
    kept_trajs = []
    kept_advantages = []
    kept_mean_rewards = []
    kept_norm_data = []
    kept_raw_data = []  # raw (unnormalized) per-scene data for logprob path
    kept_lane_dep_fracs = []  # fraction of K trajs that depart lane per scene

    for i in tqdm(range(N), desc="Scoring"):
        traj_K = all_trajs[i]  # [K, T, 4]
        data_i = all_data[i]

        rewards = compute_reward_batch(traj_K, data_i, reward_config)

        # Track lane departure fraction before rejection sampling
        n_lane_dep = sum(1 for r in rewards if r.lane_crossing)
        lane_dep_frac = n_lane_dep / len(rewards)

        if keep and 0 < keep < K:
            reward_vals = np.array([r.total for r in rewards])
            top_idx = np.argsort(reward_vals)[-keep:]
            traj_K = traj_K[top_idx]
            rewards = [rewards[j] for j in top_idx]

        advantages = compute_group_advantages(
            rewards, mode=config.advantage_mode,
            fixed_scale=config.advantage_fixed_scale,
        )

        if np.all(advantages == 0):
            continue

        kept_trajs.append(traj_K)
        kept_advantages.append(advantages)
        kept_mean_rewards.append(float(np.mean([r.total for r in rewards])))
        kept_lane_dep_fracs.append(lane_dep_frac)
        # Extract per-scene norm data (B=1 slice)
        norm_i = {}
        for k, v in norm_batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == N:
                norm_i[k] = v[i:i+1]
            else:
                norm_i[k] = v
        kept_norm_data.append(norm_i)
        # Also keep raw data for logprob path (needs unnormalized data)
        if config.grpo_loss_type == "advantage_logprob":
            kept_raw_data.append(data_i)

    N_kept = len(kept_trajs)
    if N_kept == 0:
        return {}

    # Lane departure scene trimming: drop scenes with highest lane departure fraction.
    # E.g., lane_dep_trim_n=10 drops the 10 scenes where most trajectories leave lane.
    lane_trim = config.lane_dep_trim_n
    if lane_trim > 0 and N_kept > lane_trim:
        # Sort by lane_dep_frac ascending, drop the worst lane_trim scenes (highest fractions)
        sorted_idx = sorted(range(N_kept), key=lambda j: kept_lane_dep_fracs[j])
        keep_idx = sorted_idx[:N_kept - lane_trim]
        n_dropped = N_kept - len(keep_idx)
        avg_dep_dropped = np.mean([kept_lane_dep_fracs[j] for j in sorted_idx[len(keep_idx):]])
        avg_dep_kept = np.mean([kept_lane_dep_fracs[j] for j in keep_idx])
        kept_trajs = [kept_trajs[j] for j in keep_idx]
        kept_advantages = [kept_advantages[j] for j in keep_idx]
        kept_mean_rewards = [kept_mean_rewards[j] for j in keep_idx]
        kept_lane_dep_fracs = [kept_lane_dep_fracs[j] for j in keep_idx]
        kept_norm_data = [kept_norm_data[j] for j in keep_idx]
        if config.grpo_loss_type == "advantage_logprob":
            kept_raw_data = [kept_raw_data[j] for j in keep_idx]
        print(f"  Lane-dep trim: dropped {n_dropped} worst scenes "
              f"(avg_dep={avg_dep_dropped:.0%}), keeping {len(kept_trajs)} "
              f"(avg_dep={avg_dep_kept:.0%})")
        N_kept = len(kept_trajs)

    # Scene trimming by reward
    trim = config.reward_trim_pct
    if trim > 0 and N_kept >= 10:
        n_trim = max(1, int(N_kept * trim))
        mean_rews = kept_mean_rewards
        sorted_idx = sorted(range(N_kept), key=lambda j: mean_rews[j])
        keep_idx = sorted_idx[n_trim:N_kept - n_trim]
        kept_trajs = [kept_trajs[j] for j in keep_idx]
        kept_advantages = [kept_advantages[j] for j in keep_idx]
        kept_mean_rewards = [kept_mean_rewards[j] for j in keep_idx]
        kept_norm_data = [kept_norm_data[j] for j in keep_idx]
        if config.grpo_loss_type == "advantage_logprob":
            kept_raw_data = [kept_raw_data[j] for j in keep_idx]
        print(f"  Trimmed {2*n_trim} scenes, keeping {len(kept_trajs)}/{N_kept}")
        N_kept = len(kept_trajs)

    # 5. Apply KL scheduling (persists on config for logging/checkpointing)
    scheduled_kl = config.get_kl_coef(epoch, config.train_epochs)
    if scheduled_kl != config.kl_coef:
        print(f"  [kl_schedule] epoch {epoch}: kl_coef {config.kl_coef:.4f} -> {scheduled_kl:.4f}")
        config.kl_coef = scheduled_kl

    # 6. Training
    if config.grpo_loss_type == "advantage_logprob":
        return _train_logprob(
            model, model_args, optimizer, config,
            kept_trajs, kept_advantages, kept_raw_data,
            N, N_kept, device,
        )
    else:
        return _train_mse(
            model, model_args, optimizer, config,
            kept_trajs, kept_advantages, kept_norm_data,
            N_kept, device,
        )


def _train_logprob(
    model, model_args, optimizer, config,
    kept_trajs, kept_advantages, kept_raw_data,
    N_total, N_kept, device,
):
    """DDV2-style logprob GRPO: per-scene collect + train."""
    from rlvr.grpo_logprob_loss import collect_logprob_rollout, compute_logprob_grpo_loss

    print(f"  Training on {N_kept} scenes (logprob GRPO, kl_coef={config.kl_coef:.6f})...")

    # Stage 1: Collect rollouts for all scenes (no grad)
    # Uses raw (unnormalized) data — collect_logprob_rollout normalizes internally
    print(f"  Collecting denoising rollouts...")
    rollouts = []
    model.eval()
    for i in tqdm(range(N_kept), desc="Collecting"):
        raw_data_i = kept_raw_data[i]
        # Ensure B=1 format
        data_i = {}
        for k, v in raw_data_i.items():
            if isinstance(v, torch.Tensor):
                data_i[k] = v[:1] if v.shape[0] > 1 else v
            else:
                data_i[k] = v
        rollout = collect_logprob_rollout(
            model=model,
            data=data_i,
            trajectories=kept_trajs[i],
            model_args=model_args,
            config=config,
            device=device,
        )
        rollouts.append(rollout)

    # Stage 2: Optimize using collected rollouts
    print(f"  Optimizing...")
    model.train()
    optimizer.zero_grad()
    all_metrics = {}
    n_scenes = 0
    accum_count = 0

    for i in tqdm(range(N_kept), desc="Training"):
        raw_data_i = kept_raw_data[i]
        data_i = {}
        for k, v in raw_data_i.items():
            if isinstance(v, torch.Tensor):
                data_i[k] = v[:1] if v.shape[0] > 1 else v
            else:
                data_i[k] = v

        loss, metrics = compute_logprob_grpo_loss(
            model=model,
            rollout=rollouts[i],
            advantages=kept_advantages[i],
            data=data_i,
            model_args=model_args,
            config=config,
            device=device,
        )

        scaled_loss = loss / config.grad_accum_groups
        scaled_loss.backward()
        accum_count += 1

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n_scenes += 1

        if accum_count >= config.grad_accum_groups:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    # Flush remaining — rescale gradients for incomplete last group
    if accum_count > 0:
        if accum_count < config.grad_accum_groups:
            scale_fix = config.grad_accum_groups / accum_count
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.mul_(scale_fix)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        optimizer.zero_grad()

    return {k: v / max(n_scenes, 1) for k, v in all_metrics.items()}


def _train_mse(
    model, model_args, optimizer, config,
    kept_trajs, kept_advantages, kept_norm_data,
    N_kept, device,
):
    """Original MSE-based batched GRPO training."""
    print(f"  Training on {N_kept} scenes (batched GRPO, kl_coef={config.kl_coef:.6f})...")
    keep_per = kept_trajs[0].shape[0]

    chunk_size = 1
    model.train()
    optimizer.zero_grad()

    all_metrics = {}
    n_scenes_total = 0
    accum_count = 0
    accum_count_target = config.grad_accum_groups

    for c_start in range(0, N_kept, chunk_size):
        c_end = min(c_start + chunk_size, N_kept)
        c_trajs = kept_trajs[c_start:c_end]
        c_advs = kept_advantages[c_start:c_end]
        c_norms = kept_norm_data[c_start:c_end]
        c_n = len(c_trajs)

        all_kept = torch.cat(c_trajs, dim=0)
        all_adv = np.concatenate(c_advs)

        merged_norm = {}
        for k in c_norms[0]:
            vals = [d[k] for d in c_norms]
            if isinstance(vals[0], torch.Tensor):
                expanded = [v.expand(keep_per, *v.shape[1:]) for v in vals]
                merged_norm[k] = torch.cat(expanded, dim=0)
            else:
                merged_norm[k] = vals[0]

        loss, metrics = compute_batched_grpo_loss(
            policy_model=model,
            trajectories_tensor=all_kept,
            advantages=all_adv,
            data=merged_norm,
            model_args=model_args,
            config=config,
            device=device,
        )

        accum_count += 1
        scaled_loss = loss * c_n / accum_count_target
        scaled_loss.backward()

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v * c_n
        n_scenes_total += c_n

        is_accum_boundary = (c_end % (chunk_size * config.grad_accum_groups) == 0)
        is_last = (c_end == N_kept)
        if is_accum_boundary or is_last:
            if is_last and not is_accum_boundary and accum_count < accum_count_target:
                scale_fix = accum_count_target / accum_count
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.mul_(scale_fix)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    return {k: v / max(n_scenes_total, 1) for k, v in all_metrics.items()}
