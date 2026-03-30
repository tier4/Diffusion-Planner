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
from torch import nn, optim
from tqdm import tqdm

from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data as _load_npz_data_raw
from rlvr.closed_loop.per_step_reward import StepRewardConfig
from rlvr.closed_loop.rollout import RolloutBuffer, RolloutManager
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_grpo_loss
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

        # Reward config
        self.reward_config = RewardConfig(
            w_safety=config.w_safety,
            w_progress=config.w_progress,
            w_smooth=config.w_smooth,
            w_feasibility=config.w_feasibility,
            w_centerline=config.w_centerline,
            near_edge_scale=config.near_edge_scale,
            wide_edge_scale=config.wide_edge_scale,
            stopped_penalty=config.stopped_penalty,
        )

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
        )
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

        # Rollout manager
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
        """Run open-loop GRPO to train DiT using explorer-guided trajectories.

        Same as GRPOExplorationTrainer but only the DiT part — explorer
        is already trained in the closed-loop phase.
        """
        self.exploration_policy.eval()
        self.dit_optimizer.zero_grad()

        total_dit_loss = 0.0
        n_groups = 0
        dit_accum = 0

        for path in tqdm(scene_paths, desc=f"Epoch {epoch} DiT GRPO"):
            try:
                data = self._load_npz(path)
            except Exception:
                continue

            # Normalize
            normalizer = copy.deepcopy(self.model_args.observation_normalizer)
            norm_data = {}
            for k, v in data.items():
                norm_data[k] = v.clone() if isinstance(v, torch.Tensor) else v
            norm_data = normalizer(norm_data)

            # Eval mode for trajectory generation (decoder needs eval mode for inference)
            self.policy_model.eval()

            # Frozen encoder + reference trajectory
            with torch.no_grad():
                scene_encoding = run_frozen_encoder(self.policy_model, norm_data)
                x_ref_np = generate_reference_trajectory(
                    self.policy_model, self.model_args, norm_data, self.device,
                )
                x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(self.device)
                norm_data["x_ref"] = x_ref

                # Explorer produces guidance
                policy_out = self.exploration_policy(scene_encoding, x_ref, deterministic=False)

                # Sample K etas from the learned distribution
                K = self.config.num_generations
                lat_dist = policy_out.lat_dist
                lon_dist = policy_out.lon_dist
                eta_lat_01 = lat_dist.rsample((K,)).squeeze(-1)
                eta_lon_01 = lon_dist.rsample((K,)).squeeze(-1)
                eta_lat_vals = 2.0 * eta_lat_01 - 1.0
                eta_lon_vals = 2.0 * eta_lon_01 - 1.0

                # Generate K trajectories
                trajectories = []
                noise_min, noise_max = self.config.noise_scale_range
                for k in range(K):
                    eta_lat = eta_lat_vals[k].item()
                    eta_lon = eta_lon_vals[k].item()
                    guidance_fns = [
                        GuidanceConfig(
                            name="lateral", enabled=True, scale=1.0,
                            params={"lambda_lat": self.lambda_lat, "eta_lat": eta_lat},
                        ),
                        GuidanceConfig(
                            name="longitudinal", enabled=True, scale=1.0,
                            params={"lambda_lon": self.lambda_lon, "eta_lon": eta_lon},
                        ),
                    ]
                    set_cfg = GuidanceSetConfig(
                        functions=guidance_fns, global_scale=self.guidance_scale,
                    )
                    composer = GuidanceComposer(set_cfg)
                    noise = 0.0 if k == 0 else random.uniform(noise_min, noise_max)
                    traj = generate_samples(
                        model=self.policy_model, model_args=self.model_args,
                        data=norm_data, noise_scale=noise, n_samples=1,
                        composer=composer, device=self.device,
                    )[0]
                    trajectories.append(traj)

                # Score trajectories
                traj_batch = torch.tensor(
                    np.stack(trajectories), device=self.device, dtype=torch.float32,
                )

            # Reward scoring outside no_grad (compute_reward_batch has its own)
            rewards = compute_reward_batch(traj_batch, data, self.reward_config)

            # Rejection sampling
            if self.config.rejection_keep > 0 and self.config.rejection_keep < K:
                keep = self.config.rejection_keep
                reward_vals = np.array([r.total for r in rewards])
                top_idx = np.argsort(reward_vals)[-keep:]
                traj_batch = traj_batch[top_idx]
                rewards = [rewards[i] for i in top_idx]

            advantages = compute_group_advantages(
                rewards, mode=self.config.advantage_mode,
                fixed_scale=self.config.advantage_fixed_scale,
            )

            if np.all(advantages == 0):
                continue

            # Train mode for GRPO loss computation (needs diffusion_time)
            self.policy_model.train()

            # DiT GRPO loss (with gradients)
            dit_loss, _ = compute_grpo_loss(
                policy_model=self.policy_model,
                trajectories=traj_batch.cpu().numpy(),
                advantages=advantages,
                data=norm_data,
                model_args=self.model_args,
                config=self.config,
                device=self.device,
            )

            scaled_loss = dit_loss / self.config.grad_accum_groups
            scaled_loss.backward()
            dit_accum += 1
            total_dit_loss += dit_loss.item()
            n_groups += 1

            if dit_accum >= self.config.grad_accum_groups:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    max_norm=5.0,
                )
                self.dit_optimizer.step()
                self.dit_optimizer.zero_grad()
                dit_accum = 0

        # Flush remaining
        if dit_accum > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
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

        for path in tqdm(scene_paths, desc=f"Epoch {epoch} rollout"):
            buf = self.rollout_manager.run_rollout(path)
            if buf is not None and len(buf.steps) > 0:
                rollout_buffers.append(buf)
                total_steps += buf.episode_length
                total_return += buf.total_return
                if buf.steps[-1].terminal:
                    n_terminal += 1
                    last = buf.steps[-1]
                    if last.reward < -8.0:  # collision penalty
                        n_collision += 1
                    elif last.reward < -3.0:  # rb crossing penalty
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
                eta_lat_01_t = torch.tensor(
                    step.eta_lat_01, dtype=torch.float32, device=self.device
                ).clamp(1e-6, 1 - 1e-6)
                eta_lon_01_t = torch.tensor(
                    step.eta_lon_01, dtype=torch.float32, device=self.device
                ).clamp(1e-6, 1 - 1e-6)

                log_prob = (
                    policy_out.lat_dist.log_prob(eta_lat_01_t)
                    + policy_out.lon_dist.log_prob(eta_lon_01_t)
                )

                # REINFORCE loss
                advantage = adv[t].to(self.device)
                reinforce_loss = -(log_prob * advantage.detach())

                # Value loss
                value_target = buf.value_targets[t].to(self.device)
                value_loss = (policy_out.value.squeeze() - value_target.detach()) ** 2

                # Entropy bonus
                entropy = (
                    policy_out.lat_dist.entropy()
                    + policy_out.lon_dist.entropy()
                )

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

        # --- Collect eta statistics ---
        eta_lats = [s.eta_lat_01 * 2 - 1 for buf in rollout_buffers for s in buf.steps]
        eta_lons = [s.eta_lon_01 * 2 - 1 for buf in rollout_buffers for s in buf.steps]
        eta_lat_mean = sum(eta_lats) / len(eta_lats) if eta_lats else 0.0
        eta_lon_mean = sum(eta_lons) / len(eta_lons) if eta_lons else 0.0
        eta_lat_std = (sum((e - eta_lat_mean) ** 2 for e in eta_lats) / len(eta_lats)) ** 0.5 if eta_lats else 0.0
        eta_lon_std = (sum((e - eta_lon_mean) ** 2 for e in eta_lons) / len(eta_lons)) ** 0.5 if eta_lons else 0.0

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
            "eta_lat_std": eta_lat_std,
            "eta_lon_std": eta_lon_std,
        }

        self.train_log.append(metrics)

        # --- Phase 3: DiT GRPO (if not frozen) ---
        if not self.config.closed_loop_freeze_dit and self.dit_optimizer is not None:
            dit_metrics = self._run_dit_grpo(scene_paths, epoch)
            metrics.update(dit_metrics)

        lat_shift_cm = eta_lat_mean * self.lambda_lat * 100
        lon_shift_pct = eta_lon_mean * self.lambda_lon * 100

        print(
            f"  Epoch {epoch}: "
            f"scenes={n_scenes}, "
            f"ep_len={avg_episode_len:.1f}/{self.config.closed_loop_rollout_steps}, "
            f"return={avg_return:.2f}, "
            f"terminal={n_terminal} (coll={n_collision}, rb={n_rb}), "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            f"η_lat={eta_lat_mean:.4f}±{eta_lat_std:.4f} ({lat_shift_cm:+.1f}cm), "
            f"η_lon={eta_lon_mean:.4f}±{eta_lon_std:.4f} ({lon_shift_pct:+.1f}%)"
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
        else:
            checkpoint_data = {
                "epoch": epoch,
                "model": self.policy_model.state_dict(),
                "exploration_policy": self.exploration_policy.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            }
            torch.save(checkpoint_data, self.run_dir / "latest.pth")

        print(f"  Saved checkpoint: epoch {epoch}")
