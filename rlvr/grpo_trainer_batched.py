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
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data
from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_batched_grpo_loss
from rlvr.reward import RewardConfig, compute_reward_batch, compute_group_advantages


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


def generate_all_scenes_batched(
    model: nn.Module,
    model_args,
    norm_batch: dict[str, torch.Tensor],
    K: int,
    noise_range: tuple[float, float],
    device: torch.device,
    gen_chunk_size: int = 64,
) -> torch.Tensor:
    """Generate K trajectories for all N scenes in ~5 chunked-batched passes.

    Returns:
        [N, K, T, 4] tensor.
    """
    N = norm_batch["ego_current_state"].shape[0]
    noise_min, noise_max = noise_range
    all_k_trajs = []

    # --- Config 1: Deterministic ---
    det_trajs = _chunked_generate(model, model_args, norm_batch, 0.0, 0.0, None, device, gen_chunk_size)
    all_k_trajs.append(det_trajs)

    # --- Config 2-4: CL + SPD guidance sweep (stay in-lane AND drive) ---
    cl_spd_configs = [
        (3.0, 5.0, 0.0, 0.0),   # CL3+SPD5, deterministic
        (5.0, 5.0, 0.0, 0.0),   # CL5+SPD5, deterministic
        (5.0, 8.0, 0.0, 0.5),   # CL5+SPD8, small noise
    ]
    for cl_scale, spd_scale, n_min, n_max in cl_spd_configs:
        fns = [
            GuidanceConfig("centerline_following", enabled=True, scale=cl_scale),
            GuidanceConfig("speed", enabled=True, scale=spd_scale,
                           params={"v_high": 8.0, "v_low": 1.0}),
        ]
        comp = GuidanceComposer(GuidanceSetConfig(functions=fns, global_scale=1.0))
        trajs = _chunked_generate(model, model_args, norm_batch, n_min, n_max, comp, device, gen_chunk_size)
        all_k_trajs.append(trajs)

    # --- Config 5+: Random guidance (no road_border to avoid OOM) ---
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
    if epoch == 1:
        print(f"  [DEBUG] reward_config.reward_mode = {reward_config.reward_mode}")
        print(f"  [DEBUG] reward_config.enable_lane_departure = {reward_config.enable_lane_departure}")
        print(f"  [DEBUG] reward_config.lane_gate_enabled = {reward_config.lane_gate_enabled}")
        print(f"  [DEBUG] reward_config.lane_near_scale = {reward_config.lane_near_scale}")

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

    # 3. Generate K trajectories for all scenes (batched)
    print(f"  Generating {K} trajectories × {N} scenes (batched)...")
    model.eval()
    with torch.no_grad():
        all_trajs = generate_all_scenes_batched(
            model, model_args, norm_batch, K, config.noise_scale_range, device,
        )  # [N, K, T, 4]

    # Free generation memory before scoring + training
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Per-scene reward scoring (sequential — different neighbor data)
    print(f"  Scoring rewards...")
    kept_trajs = []
    kept_advantages = []
    kept_norm_data = []

    for i in tqdm(range(N), desc="Scoring"):
        traj_K = all_trajs[i]  # [K, T, 4]
        data_i = all_data[i]

        rewards = compute_reward_batch(traj_K, data_i, reward_config)

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
        # Extract per-scene norm data (B=1 slice)
        norm_i = {}
        for k, v in norm_batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == N:
                norm_i[k] = v[i:i+1]
            else:
                norm_i[k] = v
        kept_norm_data.append(norm_i)

    N_kept = len(kept_trajs)
    if N_kept == 0:
        return {}

    # Scene trimming
    trim = config.reward_trim_pct
    if trim > 0 and N_kept >= 10:
        n_trim = max(1, int(N_kept * trim))
        mean_rews = [float(t.mean()) for t in kept_trajs]  # rough proxy
        sorted_idx = sorted(range(N_kept), key=lambda j: mean_rews[j])
        keep_idx = sorted_idx[n_trim:N_kept - n_trim]
        kept_trajs = [kept_trajs[j] for j in keep_idx]
        kept_advantages = [kept_advantages[j] for j in keep_idx]
        kept_norm_data = [kept_norm_data[j] for j in keep_idx]
        print(f"  Trimmed {2*n_trim} scenes, keeping {len(kept_trajs)}/{N_kept}")
        N_kept = len(kept_trajs)

    # 5. Batched GRPO training: stack all scenes into one forward pass
    print(f"  Training on {N_kept} scenes (batched GRPO)...")
    keep_per = kept_trajs[0].shape[0]

    # Process one scene at a time: matches sequential trainer's gradient behavior.
    # Each scene's loss is normalized by keep_per trajs only (not cross-scene).
    chunk_size = 1
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    all_metrics = {}
    n_chunks = 0

    for c_start in range(0, N_kept, chunk_size):
        c_end = min(c_start + chunk_size, N_kept)
        c_trajs = kept_trajs[c_start:c_end]
        c_advs = kept_advantages[c_start:c_end]
        c_norms = kept_norm_data[c_start:c_end]
        c_n = len(c_trajs)

        # Stack trajectories: [c_n * keep_per, T, 4]
        all_kept = torch.cat(c_trajs, dim=0)
        all_adv = np.concatenate(c_advs)

        # Stack norm data
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

        # compute_batched_grpo_loss divides by total trajs in chunk (c_n * keep_per).
        # Sequential trainer divides by keep_per only (per scene), then by grad_accum.
        # To match: multiply by c_n to undo the cross-scene averaging, then divide
        # by grad_accum equivalent (n_chunks acts as grad_accum).
        c_n = len(c_trajs)
        # With chunk_size=4: loss already normalized by ~32 trajs (4 scenes × 8).
        # Scale by c_n/grad_accum to match sequential trainer's effective gradient.
        scaled_loss = loss * c_n / config.grad_accum_groups
        scaled_loss.backward()

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n_chunks += 1

        # Step optimizer every few chunks
        if (c_end % (chunk_size * config.grad_accum_groups) == 0) or c_end == N_kept:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            optimizer.zero_grad()

    return {k: v / max(n_chunks, 1) for k, v in all_metrics.items()}
