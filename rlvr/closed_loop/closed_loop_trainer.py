"""Closed-loop exploration trainer.

Collects per-step rollouts → computes GAE → updates explorer with REINFORCE + baseline.
DiT is frozen during Phase 1 (explorer warmup). Phase 2 adds open-loop GRPO for DiT.

Architecture note: the simulation backend (DiT-based state update) is decoupled from
the training loop via RolloutManager. Swapping to an external simulator only requires
replacing how RolloutManager steps the environment.
"""

from __future__ import annotations

import copy
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from torch import nn, optim
from tqdm import tqdm

from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data as _load_npz_data_raw
from rlvr.closed_loop.per_step_reward import StepRewardConfig
from rlvr.closed_loop.rollout import RolloutBuffer, RolloutManager
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_batched_grpo_loss
from rlvr.reward import RewardConfig, compute_group_advantages, compute_reward_batch


class ClosedLoopExplorationTrainer:
    """Closed-loop trainer: rollout -> GAE -> REINFORCE + value baseline.

    Per epoch:
      1. Collect rollouts for all scenes (no grad)
      2. Re-run explorer forward pass with gradients on stored (scene_encoding, x_ref)
      3. REINFORCE loss + value loss + entropy bonus
      4. Step optimizer
    """

    def __init__(
        self,
        policy_model: nn.Module,
        model_args,
        dit_optimizer: optim.Optimizer | None,
        device: torch.device,
        run_dir: Path,
        config: GRPOConfig,
        use_lora: bool = False,
    ):
        self.policy_model = policy_model
        self.model_args = model_args
        self.dit_optimizer = dit_optimizer
        self.device = device
        self.run_dir = run_dir
        self.config = config
        self.use_lora = use_lora

        # Reward config — field-intersection builder so sc_* / future fields
        # are never silently dropped (same fix as GRPOExplorationTrainer).
        from rlvr.grpo_exploration_trainer import reward_config_from_grpo
        self.reward_config = reward_config_from_grpo(config)

        # --- Exploration policy ---
        ep_config = ExplorationPolicyConfig(
            hidden_dim=config.exploration_hidden_dim,
            n_mixer_layers=config.exploration_n_mixer_layers,
            n_attn_heads=config.exploration_n_attn_heads,
            dropout=config.exploration_dropout,
            learning_rate=config.exploration_lr,
            encoder_hidden_dim=model_args.hidden_dim,
            head_init=config.exploration_head_init,
            head_init_std=config.exploration_head_init_std,
            head_raw_scale=config.exploration_head_raw_scale,
            heads=config.exploration_heads,
        )
        self.heads = list(config.exploration_heads)
        self.head_params = {
            "lambda_lat": config.exploration_lambda_lat,
            "lat_scale": config.exploration_lat_scale,
            "col_scale": config.exploration_col_scale,
            "col_range": config.exploration_col_range,
            "lambda_spd": config.exploration_lambda_spd,
            "stretch_scale": config.exploration_stretch_scale,
            "guidance_scale": config.exploration_guidance_scale,
            "lambda_lon": config.exploration_lambda_lon,
        }
        self.exploration_policy = ExplorationPolicy(
            ep_config, ref_seq_len=model_args.future_len,
        ).to(device)

        if config.exploration_checkpoint_path:
            ckpt_path = Path(config.exploration_checkpoint_path)
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=device)
                missing, unexpected = self.exploration_policy.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"  Warning: missing={missing}, unexpected={unexpected}")
                print(f"  Loaded exploration policy from {ckpt_path}")

        n_params = sum(p.numel() for p in self.exploration_policy.parameters())
        print(f"  Exploration policy: {n_params:,} params (hidden={config.exploration_hidden_dim})")

        # Separate optimizer for exploration policy
        self.policy_optimizer = optim.AdamW(
            self.exploration_policy.parameters(),
            lr=config.exploration_lr,
        )

        # Step reward config
        self.step_reward_config = StepRewardConfig(
            w_progress=1.0,
            w_alive=config.closed_loop_alive_bonus,
            w_collision=10.0,
            w_rb_crossing=5.0,
        )

        # Rollout manager — batched version for GPU parallelism
        from rlvr.closed_loop.batched_rollout import BatchedRolloutManager
        self.batched_rollout_manager = BatchedRolloutManager(
            policy_model=policy_model,
            model_args=model_args,
            exploration_policy=self.exploration_policy,
            device=device,
            lambda_lat=config.exploration_lambda_lat,
            lambda_lon=config.exploration_lambda_lon,
            guidance_scale=config.exploration_guidance_scale,
            rollout_steps=config.closed_loop_rollout_steps,
            noise_range=tuple(config.noise_scale_range),
            gamma=config.closed_loop_gamma,
            gae_lambda=config.closed_loop_gae_lambda,
            step_reward_config=self.step_reward_config,
            reward_config=self.reward_config,
            batch_size=config.closed_loop_batch_size,
            drop_last=config.closed_loop_drop_last,
            heads=self.heads,
            head_params=self.head_params,
        )
        # Enable online explorer updates during rollout (PlannerRFT-style)
        if config.closed_loop_online_interval > 0:
            self.batched_rollout_manager.online_update_interval = config.closed_loop_online_interval
            self.batched_rollout_manager.online_lr = config.exploration_lr
            self.batched_rollout_manager.online_entropy_coef = config.exploration_entropy_coef
            self.batched_rollout_manager.online_value_coef = config.closed_loop_value_coef
            self.batched_rollout_manager.explorer_mini_batch = config.closed_loop_explorer_mini_batch
        # Keep sequential rollout as fallback
        self.rollout_manager = RolloutManager(
            policy_model=policy_model,
            model_args=model_args,
            exploration_policy=self.exploration_policy,
            device=device,
            lambda_lat=config.exploration_lambda_lat,
            lambda_lon=config.exploration_lambda_lon,
            guidance_scale=config.exploration_guidance_scale,
            rollout_steps=config.closed_loop_rollout_steps,
            noise_range=tuple(config.noise_scale_range),
            gamma=config.closed_loop_gamma,
            gae_lambda=config.closed_loop_gae_lambda,
            step_reward_config=self.step_reward_config,
            reward_config=self.reward_config,
        )

        # Lateral/longitudinal guidance parameters
        self.lambda_lat = config.exploration_lambda_lat
        self.lambda_lon = config.exploration_lambda_lon
        self.guidance_scale = config.exploration_guidance_scale

        # Tracking
        self.train_log: list[dict] = []
        self.eval_log: list[dict] = []

    def _load_npz(self, npz_path: str) -> dict[str, torch.Tensor]:
        data = _load_npz_data_raw(npz_path, self.device)
        if "delay" not in data:
            data["delay"] = torch.zeros(1, dtype=torch.long, device=self.device)
        return data

    def _run_dit_grpo(self, scene_paths: list[str], epoch: int) -> dict:
        """Run batched N×K GRPO for fast training.

        Processes chunks of N scenes × K trajectories with batched inference.
        Uses varied per-trajectory noise and batched GRPO loss.
        ~6x faster than sequential. Use sequential only for zi reproduction.
        """
        from rlvr.closed_loop.batched_rollout import (
            _batched_encoder,
            _batched_generate,
            _batched_generate_varied_noise,
        )

        if self.exploration_policy is not None:
            self.exploration_policy.eval()
        self.dit_optimizer.zero_grad()

        K = self.config.num_generations
        # GRPO chunk: N scenes per batch. N×K must fit GPU.
        grpo_batch = min(self.config.closed_loop_batch_size, max(16, 256 // K))
        noise_min, noise_max = self.config.noise_scale_range
        rejection_keep = self.config.rejection_keep

        total_dit_loss = 0.0
        n_groups = 0
        dit_accum = 0

        # Load all scenes
        all_data = []
        all_norm = []
        for path in scene_paths:
            try:
                data = self._load_npz(path)
                # Skip stationary scenes
                if "ego_agent_future" in data:
                    gt = data["ego_agent_future"]
                    if gt.dim() == 3: gt = gt[0]
                    if torch.diff(gt[:, :2], dim=0).norm(dim=-1).sum() < 1.0:
                        continue
                norm = copy.deepcopy(self.model_args.observation_normalizer)(
                    {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                )
                all_data.append(data)
                all_norm.append(norm)
            except Exception:
                continue

        N_total = len(all_data)
        if N_total == 0:
            return {"dit_loss": 0.0, "dit_groups": 0}

        pbar = tqdm(total=N_total, desc=f"Epoch {epoch} DiT GRPO")

        for chunk_start in range(0, N_total, grpo_batch):
            chunk_data = all_data[chunk_start:chunk_start + grpo_batch]
            chunk_norm = all_norm[chunk_start:chunk_start + grpo_batch]
            N_chunk = len(chunk_data)

            # Stack into batch
            batch_norm = {}
            for k in chunk_norm[0]:
                vals = [d[k] for d in chunk_norm]
                if isinstance(vals[0], torch.Tensor):
                    batch_norm[k] = torch.cat(vals, dim=0)
                else:
                    batch_norm[k] = vals[0]

            self.policy_model.eval()

            with torch.no_grad():
                scene_encoding = _batched_encoder(self.policy_model, batch_norm)

                import contextlib
                inner = self.policy_model.module if hasattr(self.policy_model, "module") else self.policy_model
                disable_ctx = inner.disable_adapter() if hasattr(inner, "disable_adapter") else contextlib.nullcontext()
                with disable_ctx:
                    ref_trajs = _batched_generate(
                        self.policy_model, self.model_args, batch_norm,
                        noise_scale=0.0, composer=None, device=self.device,
                    )
                batch_norm["x_ref"] = ref_trajs
                batch_norm["reference_trajectory"] = ref_trajs  # Required by lateral/longitudinal guidance

                # Explorer etas — K samples per head, flattened to [N*K]
                if self.exploration_policy is not None:
                    policy_out = self.exploration_policy(scene_encoding, ref_trajs, deterministic=False)
                    etas_NK = {
                        h: (2.0 * policy_out.dists[h].rsample((K,)).squeeze(-1)
                            - 1.0).T.reshape(-1)
                        for h in self.heads
                    }
                else:
                    etas_NK = {h: torch.zeros(N_chunk * K, device=self.device)
                               for h in self.heads}

                # Expand to N×K
                NK_data = {}
                for k_key, v in batch_norm.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == N_chunk:
                        NK_data[k_key] = v.unsqueeze(1).expand(
                            -1, K, *v.shape[1:]
                        ).reshape(N_chunk * K, *v.shape[1:])
                    else:
                        NK_data[k_key] = v

                composer = self.batched_rollout_manager._build_batched_composer(
                    etas_NK)

                all_trajs = _batched_generate_varied_noise(
                    self.policy_model, self.model_args, NK_data,
                    noise_min=noise_min, noise_max=noise_max,
                    first_deterministic=False, composer=composer, device=self.device,
                )
                T_len = all_trajs.shape[1]
                all_trajs = all_trajs.reshape(N_chunk, K, T_len, 4)

            # Per-scene: reward scoring + rejection (must be per-scene for neighbor data)
            kept_trajs = []
            kept_advantages = []
            kept_norm_data = []

            for local_i in range(N_chunk):
                traj_K = all_trajs[local_i]
                data_i = chunk_data[local_i]

                rewards = compute_reward_batch(traj_K, data_i, self.reward_config)

                if rejection_keep > 0 and rejection_keep < K:
                    reward_vals = np.array([r.total for r in rewards])
                    top_idx = np.argsort(reward_vals)[-rejection_keep:]
                    traj_K = traj_K[top_idx]
                    rewards = [rewards[i] for i in top_idx]

                advantages = compute_group_advantages(
                    rewards, mode=self.config.advantage_mode,
                    fixed_scale=self.config.advantage_fixed_scale,
                )

                if np.all(advantages == 0):
                    pbar.update(1)
                    continue

                kept_trajs.append(traj_K)
                kept_advantages.append(advantages)
                norm_i = {k: (v[local_i:local_i+1] if isinstance(v, torch.Tensor) and v.shape[0] == N_chunk else v) for k, v in batch_norm.items()}
                kept_norm_data.append(norm_i)
                pbar.update(1)

            # Multi-scene batched GRPO loss: stack scenes into one forward pass
            if kept_trajs:
                self.policy_model.train()
                N_kept = len(kept_trajs)
                keep_per = kept_trajs[0].shape[0]  # trajectories per scene after rejection

                # Stack all scenes' trajectories: [N_kept * keep_per, T, 4]
                all_kept = torch.cat(kept_trajs, dim=0)
                # Stack advantages: [N_kept * keep_per]
                all_adv = np.concatenate(kept_advantages)
                # Stack norm data: expand each scene's B=1 to B=keep_per, then concat
                merged_norm = {}
                for k in kept_norm_data[0]:
                    vals = [d[k] for d in kept_norm_data]
                    if isinstance(vals[0], torch.Tensor):
                        # Each is [1, ...], expand to [keep_per, ...] then cat
                        expanded = [v.expand(keep_per, *v.shape[1:]) for v in vals]
                        merged_norm[k] = torch.cat(expanded, dim=0)  # [N_kept*keep_per, ...]
                    else:
                        merged_norm[k] = vals[0]

                dit_loss, _ = compute_batched_grpo_loss(
                    policy_model=self.policy_model,
                    trajectories_tensor=all_kept,
                    advantages=all_adv,
                    data=merged_norm,
                    model_args=self.model_args,
                    config=self.config,
                    device=self.device,
                )

                scaled_loss = dit_loss / max(N_kept / self.config.grad_accum_groups, 1)
                scaled_loss.backward()
                dit_accum += N_kept
                total_dit_loss += dit_loss.item() * N_kept
                n_groups += N_kept

                if dit_accum >= self.config.grad_accum_groups:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.policy_model.parameters() if p.requires_grad], max_norm=5.0)
                    self.dit_optimizer.step()
                    self.dit_optimizer.zero_grad()
                    dit_accum = 0

        pbar.close()

        if dit_accum > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad], max_norm=5.0)
            self.dit_optimizer.step()
            self.dit_optimizer.zero_grad()

        avg_dit_loss = total_dit_loss / max(n_groups, 1)
        print(f"  DiT GRPO: {n_groups} groups, avg_loss={avg_dit_loss:.4f}")
        return {"dit_loss": avg_dit_loss, "dit_groups": n_groups}

    def train_epoch(
        self,
        scene_paths: list[str],
        epoch: int,
    ) -> dict:
        """One epoch of closed-loop training.

        Args:
            scene_paths: List of NPZ file paths to train on.
            epoch: Current epoch number (for logging).

        Returns:
            Dict with training metrics.
        """
        # --- Phase 1: Collect rollouts (no grad) ---
        self.exploration_policy.eval()
        self.policy_model.eval()

        rollout_buffers: list[RolloutBuffer] = []
        total_steps = 0
        total_return = 0.0
        n_terminal = 0
        n_collision = 0
        n_rb = 0

        # Use batched rollout for GPU parallelism
        use_batched = self.config.closed_loop_batch_size > 1
        if use_batched:
            print(f"  Batched rollout: {len(scene_paths)} scenes, batch_size={self.config.closed_loop_batch_size}")
            rollout_buffers = self.batched_rollout_manager.run_rollouts(scene_paths)
            for buf in rollout_buffers:
                total_steps += buf.episode_length
                total_return += buf.total_return
                if buf.steps[-1].terminal:
                    n_terminal += 1
                    last = buf.steps[-1]
                    if last.reward < -8.0:
                        n_collision += 1
                    elif last.reward < -3.0:
                        n_rb += 1
        else:
            for path in tqdm(scene_paths, desc=f"Epoch {epoch} rollout"):
                buf = self.rollout_manager.run_rollout(path)
                if buf is not None and len(buf.steps) > 0:
                    rollout_buffers.append(buf)
                    total_steps += buf.episode_length
                    total_return += buf.total_return
                    if buf.steps[-1].terminal:
                        n_terminal += 1
                        last = buf.steps[-1]
                        if last.reward < -8.0:
                            n_collision += 1
                        elif last.reward < -3.0:
                            n_rb += 1

        n_scenes = len(rollout_buffers)
        if n_scenes == 0:
            print(f"  [epoch {epoch}] No valid rollouts collected")
            return {}

        avg_episode_len = total_steps / n_scenes
        avg_return = total_return / n_scenes

        # --- Phase 2: Train explorer (with grad) ---
        # Per-scene stepping: optimizer steps after each scene's rollout data.
        # This gives N_scenes updates per epoch (not 1), which is essential
        # for the policy to learn scene-dependent guidance.
        self.exploration_policy.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_train_steps = 0

        for buf in rollout_buffers:
            if buf.advantages is None:
                continue

            self.policy_optimizer.zero_grad()

            # Normalize advantages across this rollout
            adv = buf.advantages
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            n_steps_in_buf = len(buf.steps)

            for t, step in enumerate(buf.steps):
                # Move stored tensors to device
                scene_enc = step.scene_encoding.to(self.device)
                x_ref = step.x_ref.to(self.device)

                # Re-run forward pass with gradients
                policy_out = self.exploration_policy(scene_enc, x_ref, deterministic=False)

                # Recompute log_prob for the SAME sampled eta values
                step_etas = step.etas_01 or {"lateral": step.eta_lat_01,
                                             "longitudinal": step.eta_lon_01}
                log_prob = sum(
                    policy_out.dists[h].log_prob(torch.tensor(
                        v, dtype=torch.float32, device=self.device,
                    ).clamp(1e-6, 1 - 1e-6))
                    for h, v in step_etas.items()
                )

                # REINFORCE loss
                advantage = adv[t].to(self.device)
                reinforce_loss = -(log_prob * advantage.detach())

                # Value loss
                value_target = buf.value_targets[t].to(self.device)
                value_loss = (policy_out.value.squeeze() - value_target.detach()) ** 2

                # Entropy bonus
                entropy = sum(policy_out.dists[h].entropy() for h in step_etas)

                step_loss = (
                    reinforce_loss
                    + self.config.closed_loop_value_coef * value_loss
                    - self.config.exploration_entropy_coef * entropy
                )
                step_loss = step_loss / n_steps_in_buf  # normalize by steps in THIS scene
                step_loss.backward()

                total_policy_loss += reinforce_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_train_steps += 1

            # Step optimizer after each scene
            torch.nn.utils.clip_grad_norm_(self.exploration_policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()

        # --- Collect eta statistics (per configured head) ---
        def _step_etas(s):
            return s.etas_01 or {"lateral": s.eta_lat_01,
                                 "longitudinal": s.eta_lon_01}

        head_stats: dict[str, float] = {}
        for h in self.heads:
            vals = [_step_etas(s)[h] * 2 - 1
                    for buf in rollout_buffers for s in buf.steps
                    if h in _step_etas(s)]
            mean = sum(vals) / len(vals) if vals else 0.0
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.0
            per_scene = [
                sum(_step_etas(s)[h] * 2 - 1 for s in buf.steps) / len(buf.steps)
                for buf in rollout_buffers if buf.steps
            ]
            svar = 0.0
            if len(per_scene) > 1:
                m = sum(per_scene) / len(per_scene)
                svar = (sum((x - m) ** 2 for x in per_scene) / len(per_scene)) ** 0.5
            head_stats[f"eta_{h}_mean"] = mean
            head_stats[f"eta_{h}_std"] = std
            head_stats[f"scene_var_{h}"] = svar
        # Legacy aliases used by run_experiment dashboards
        eta_lat_mean = head_stats.get("eta_lateral_mean", 0.0)
        eta_lon_mean = head_stats.get("eta_longitudinal_mean", 0.0)

        metrics = {
            "epoch": epoch,
            "n_scenes": n_scenes,
            "avg_episode_len": avg_episode_len,
            "avg_return": avg_return,
            "n_terminal": n_terminal,
            "n_collision": n_collision,
            "n_rb_crossing": n_rb,
            "policy_loss": total_policy_loss / max(n_train_steps, 1),
            "value_loss": total_value_loss / max(n_train_steps, 1),
            "entropy": total_entropy / max(n_train_steps, 1),
            "eta_lat_mean": eta_lat_mean,
            "eta_lon_mean": eta_lon_mean,
            "eta_lat_std": head_stats.get("eta_lateral_std", 0.0),
            "eta_lon_std": head_stats.get("eta_longitudinal_std", 0.0),
            "scene_var_lat": head_stats.get("scene_var_lateral", 0.0),
            "scene_var_lon": head_stats.get("scene_var_longitudinal", 0.0),
            **head_stats,
        }

        self.train_log.append(metrics)

        # --- Free rollout data from GPU before GRPO to avoid OOM ---
        # (eta stats already collected above from rollout_buffers)
        del rollout_buffers
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # --- Reset RNG before GRPO to match zi's random state ---
        # The CL rollout consumes ~12K RNG draws which diverges from zi.
        # Re-seeding ensures GRPO gets the same noise as zi for fair comparison.
        torch.manual_seed(42 + epoch)
        torch.cuda.manual_seed_all(42 + epoch)

        # --- Phase 3: DiT GRPO (if not frozen) ---

        if not self.config.closed_loop_freeze_dit and self.dit_optimizer is not None:
            dit_metrics = self._run_dit_grpo(scene_paths, epoch)
            metrics.update(dit_metrics)


        print(
            f"  Epoch {epoch}: "
            f"scenes={n_scenes}, "
            f"ep_len={avg_episode_len:.1f}/{self.config.closed_loop_rollout_steps}, "
            f"return={avg_return:.2f}, "
            f"terminal={n_terminal} (coll={n_collision}, rb={n_rb}), "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            + ", ".join(
                f"η_{h}={head_stats[f'eta_{h}_mean']:+.4f}"
                f"±{head_stats[f'eta_{h}_std']:.4f}"
                f" (scene_var {head_stats[f'scene_var_{h}']:.4f})"
                for h in self.heads)
        )
        if "dit_loss" in metrics:
            print(f"  DiT GRPO loss: {metrics['dit_loss']:.4f}")

        return metrics

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log training metrics to TSV file."""
        import pandas as pd

        log_entry = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in metrics.items()},
        }
        self.train_log.append(log_entry)

        df = pd.DataFrame(self.train_log)
        log_path = self.run_dir / "closed_loop_train_log.tsv"
        df.to_csv(log_path, sep="\t", index=False)

    def save_checkpoint(self, epoch: int, args_dict: dict | None = None) -> None:
        """Save both DiT (LoRA) and exploration policy checkpoints."""
        if self.use_lora:
            from preference_optimization.lora_utils import save_lora_checkpoint

            lora_dir = str(self.run_dir / f"lora_epoch_{epoch:03d}")
            save_lora_checkpoint(self.policy_model, lora_dir)
            self.config.to_json(Path(lora_dir) / "grpo_config.json")

            # Save exploration policy
            torch.save(
                self.exploration_policy.state_dict(),
                Path(lora_dir) / "exploration_policy.pth",
            )
            torch.save(
                self.policy_optimizer.state_dict(),
                Path(lora_dir) / "policy_optimizer.pth",
            )

            # Symlink latest
            latest_link = self.run_dir / "lora_latest"
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)
            latest_link.symlink_to(f"lora_epoch_{epoch:03d}")
        elif self.config.closed_loop_freeze_dit:
            # Frozen DiT: per-epoch policy-only checkpoints (the DiT is the
            # unchanged base model — re-serializing it every epoch wastes
            # gigabytes and loses the per-epoch history needed for sweeps).
            ep_dir = self.run_dir / f"policy_epoch_{epoch:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.exploration_policy.state_dict(),
                       ep_dir / "exploration_policy.pth")
            torch.save(self.policy_optimizer.state_dict(),
                       ep_dir / "policy_optimizer.pth")
            self.config.to_json(ep_dir / "grpo_config.json")
        else:
            checkpoint_data = {
                "epoch": epoch,
                "model": self.policy_model.state_dict(),
                "exploration_policy": self.exploration_policy.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            }
            torch.save(checkpoint_data, self.run_dir / "latest.pth")

        print(f"  Saved checkpoint: epoch {epoch}")
