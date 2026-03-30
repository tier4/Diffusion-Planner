"""Joint GRPO + Exploration Policy trainer.

Extends the standard GRPO training with a learned exploration policy that
outputs (eta_lat, eta_lon) from Beta distributions. The policy and DiT
planner are trained simultaneously:

- Policy: REINFORCE with GRPO group-relative advantages + entropy + KL
- DiT: Standard GRPO diffusion loss (unchanged)

The exploration policy samples K eta values from one distribution per scene,
generating K deterministic trajectories (noise=0) scored by the reward function.

This is a SEPARATE file from grpo_trainer.py to avoid breaking existing
GRPO training when exploration policy is disabled.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from exploration_policy.loss import compute_exploration_loss, _get_init_distributions
from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from torch.distributions import kl_divergence as kl_div
from guidance_gui.generate_samples import generate_samples
from preference_optimization.utils import load_npz_data as _load_npz_data_raw
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_grpo_loss
from rlvr.grpo_sampler import PolicyGroupMetadata, SamplerConfig
from rlvr.reward import RewardConfig, compute_group_advantages, compute_reward_batch


def _load_npz(npz_path, device):
    data = _load_npz_data_raw(npz_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data


class GRPOExplorationTrainer:
    """Joint trainer for DiT planner (GRPO) + exploration policy (REINFORCE).

    Per scene:
      1. Frozen encoder → scene_encoding
      2. LoRA-disabled DiT → x_ref (reference trajectory)
      3. Exploration policy(scene_encoding, x_ref) → Beta distributions
      4. Sample K η values → K trajectories (noise=0, lat+lon guidance)
      5. Score → GRPO advantages
      6. DiT loss: standard GRPO diffusion loss
      7. Policy loss: REINFORCE + entropy + KL

    Supports inverse KL scheduling: high DiT KL (stable planner) + low policy
    KL (free exploration) early, then swap as policy learns.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        model_args,
        dit_optimizer: optim.Optimizer,
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

        assert config.inner_epochs == 1, (
            f"GRPOExplorationTrainer only supports on-policy training (inner_epochs=1), "
            f"got inner_epochs={config.inner_epochs}. For multi-epoch training, use GRPOTrainer."
        )

        # Reward config (same as standard GRPO)
        self.reward_config = RewardConfig(
            w_safety=config.w_safety,
            w_progress=config.w_progress,
            w_smooth=config.w_smooth,
            w_feasibility=config.w_feasibility,
            w_centerline=config.w_centerline,
            near_edge_scale=config.near_edge_scale,
            wide_edge_scale=config.wide_edge_scale,
            cont_edge_scale=config.cont_edge_scale,
            max_lat_accel=config.max_lat_accel,
            lat_accel_scale=config.lat_accel_scale,
            enable_overprogress=config.enable_overprogress,
            overprogress_margin=config.overprogress_margin,
            overprogress_penalty=config.overprogress_penalty,
            stopped_penalty=config.stopped_penalty,
            reward_mode=config.reward_mode,
        )

        # Sampler config (only used for eval, not for policy-guided generation)
        self.sampler_config = SamplerConfig(
            n_trajectories=config.num_generations,
            noise_scale_range=tuple(config.noise_scale_range),
            guidance_scale_range=tuple(config.guidance_scale_range),
            enable_guidance=config.enable_guidance and config.sampling_randomization,
            enable_centerline=config.enable_centerline,
            enable_anchor=config.enable_anchor,
            enable_collision=config.enable_collision,
            enable_route_following=config.enable_route_following,
            enable_lane_keeping=config.enable_lane_keeping,
            enable_road_border=config.enable_road_border,
            enable_speed=config.enable_speed,
            guidance_prob=config.guidance_prob,
            prototypes_path=config.prototypes_path,
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

        # Lateral/longitudinal guidance parameters (from GRPOConfig)
        self.lambda_lat = config.exploration_lambda_lat
        self.lambda_lon = config.exploration_lambda_lon
        self.guidance_scale = config.exploration_guidance_scale

        # Tracking
        self.train_log: list[dict] = []
        self.eval_log: list[dict] = []
        self.best_det_reward: float = float("-inf")
        self.best_epoch: int = 0

    def generate_policy_guided_group(
        self,
        npz_path: str,
    ) -> dict | None:
        """Generate K policy-guided trajectories, score, compute advantages.

        Returns dict with keys: npz_path, data, trajectories, reward_breakdowns,
        advantages, policy_meta (PolicyGroupMetadata), det_trajectory.
        """
        try:
            data = _load_npz(npz_path, self.device)
        except Exception as e:
            print(f"  [exploration] skipping {npz_path}: {e}")
            return None

        # Skip scenes where GT barely moves
        if "ego_agent_future" in data:
            gt = data["ego_agent_future"]
            if gt.dim() == 3:
                gt = gt[0]
            gt_path = torch.diff(gt[:, :2], dim=0).norm(dim=-1).sum()
            if gt_path < 1.0:
                return None

        # Normalize data
        norm_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        norm_data = self.model_args.observation_normalizer(norm_data)

        self.policy_model.eval()
        self.exploration_policy.eval()

        with torch.no_grad():
            # 1. Deterministic trajectory (eval only, outside training group)
            det_traj = generate_samples(
                model=self.policy_model, model_args=self.model_args,
                data=norm_data, noise_scale=0.0, n_samples=1,
                composer=None, device=self.device,
            )[0]  # (T, 4)

            # 2. Reference trajectory (LoRA-disabled base model output)
            x_ref_np = generate_reference_trajectory(
                self.policy_model, self.model_args, norm_data, self.device,
            )
            x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(self.device)  # [1, T, 4]
            norm_data["reference_trajectory"] = x_ref

            # 3. Frozen encoder → scene encoding
            scene_encoding = run_frozen_encoder(self.policy_model, norm_data)

            # 4. Exploration policy → Beta distributions (deterministic=True to
            #    just get distributions; we sample K values ourselves below)
            policy_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
            lat_dist = policy_output.lat_dist
            lon_dist = policy_output.lon_dist

            # 5. Sample K η values from the distributions
            # lat_dist has batch_shape=[1] (B=1 scene), so rsample((K,)) gives [K,1].
            # Squeeze the singleton batch dim to get [K].
            K = self.config.num_generations
            eta_lat_01 = lat_dist.rsample((K,)).squeeze(-1)  # [K] in (0, 1)
            eta_lon_01 = lon_dist.rsample((K,)).squeeze(-1)  # [K] in (0, 1)
            eta_lat_vals = 2.0 * eta_lat_01 - 1.0  # [K] in [-1, 1]
            eta_lon_vals = 2.0 * eta_lon_01 - 1.0  # [K] in [-1, 1]

            # 6. Generate K trajectories — batched: expand scene to B=K
            from rlvr.closed_loop.batched_rollout import _batched_generate
            noise_min, noise_max = self.config.noise_scale_range

            # Expand scene data from B=1 to B=K
            K_data = {}
            for k_key, v in norm_data.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                    K_data[k_key] = v.expand(K, *v.shape[1:]).contiguous()
                else:
                    K_data[k_key] = v

            # Build batched composer with K etas
            guidance_fns = [
                GuidanceConfig(
                    name="lateral", enabled=True, scale=1.0,
                    params={"lambda_lat": self.lambda_lat, "eta_lat": eta_lat_vals},
                ),
                GuidanceConfig(
                    name="longitudinal", enabled=True, scale=1.0,
                    params={"lambda_lon": self.lambda_lon, "eta_lon": eta_lon_vals},
                ),
            ]
            set_cfg = GuidanceSetConfig(functions=guidance_fns, global_scale=self.guidance_scale)
            composer = GuidanceComposer(set_cfg)

            # Per-trajectory noise scales: k=0 deterministic, rest random
            # Apply via pre-built noisy xT with per-element scales
            from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
            traj_tensor = _batched_generate_varied_noise(
                self.policy_model, self.model_args, K_data,
                noise_min=noise_min, noise_max=noise_max,
                first_deterministic=True,
                composer=composer, device=self.device,
            )  # [K, T, 4]
            trajectories = [traj_tensor[k].cpu().numpy() for k in range(K)]

        # 7. Score all K trajectories
        traj_batch = torch.tensor(
            np.stack(trajectories), device=self.device, dtype=torch.float32,
        )
        reward_breakdowns = compute_reward_batch(traj_batch, data, self.reward_config)
        advantages = compute_group_advantages(
            reward_breakdowns,
            mode=self.config.advantage_mode,
            fixed_scale=self.config.advantage_fixed_scale,
        )

        # Store sampled eta values (in 0-1 space for log_prob recomputation)
        # and scene data for recomputing the policy forward pass with grad.
        # We do NOT store log_probs here — they must be recomputed with the
        # live computation graph during train_on_groups for gradients to flow.
        policy_meta = PolicyGroupMetadata(
            log_probs=torch.zeros(K, device=self.device),  # placeholder, recomputed in train
            lat_dist_params=(lat_dist.concentration1.detach(), lat_dist.concentration0.detach()),
            lon_dist_params=(lon_dist.concentration1.detach(), lon_dist.concentration0.detach()),
            eta_lat_samples=eta_lat_vals.detach(),
            eta_lon_samples=eta_lon_vals.detach(),
        )

        return {
            "npz_path": npz_path,
            "data": data,
            "norm_data": norm_data,
            "trajectories": trajectories,
            "reward_breakdowns": reward_breakdowns,
            "advantages": advantages,
            "policy_meta": policy_meta,
            "det_trajectory": det_traj,
            "scene_encoding": scene_encoding.detach(),
            "x_ref": x_ref.detach(),
            "eta_lat_01": eta_lat_01.detach(),  # [K] in (0,1), for log_prob recomputation
            "eta_lon_01": eta_lon_01.detach(),  # [K] in (0,1), for log_prob recomputation
        }

    def train_on_groups(
        self,
        groups: list[dict],
        epoch: int,
        progress_callback=None,
    ) -> dict[str, float]:
        """Train both DiT and exploration policy on collected groups."""
        if not groups:
            return _empty_metrics()

        # Check if policy should be frozen this epoch
        freeze_after = self.config.exploration_freeze_after_epoch
        policy_frozen = freeze_after > 0 and epoch > freeze_after
        if policy_frozen and epoch == freeze_after + 1:
            print(f"  [policy_freeze] Freezing exploration policy after epoch {freeze_after}")

        all_metrics: dict[str, float] = {}
        num_groups = 0
        # Track per-scene η to measure scene-dependence (not just the mean)
        per_scene_eta_lat: list[float] = []
        per_scene_eta_lon: list[float] = []

        self.policy_model.train()
        if not policy_frozen:
            self.exploration_policy.train()
        else:
            self.exploration_policy.eval()
        self.dit_optimizer.zero_grad()
        if not policy_frozen:
            self.policy_optimizer.zero_grad()

        dit_accum = 0
        n_policy_accum = 0

        for group_idx, group in enumerate(tqdm(groups, desc=f"Epoch {epoch}")):
            advantages_np = group["advantages"]
            if np.all(advantages_np == 0):
                continue

            # --- DiT GRPO loss (batched: all trajectories in one forward pass) ---
            from rlvr.grpo_loss import compute_batched_grpo_loss
            traj_list = group["trajectories"]
            traj_tensor = torch.tensor(
                np.stack(traj_list) if isinstance(traj_list[0], np.ndarray) else traj_list,
                device=self.device, dtype=torch.float32,
            )
            dit_loss, dit_metrics = compute_batched_grpo_loss(
                policy_model=self.policy_model,
                trajectories_tensor=traj_tensor,
                advantages=advantages_np,
                data=group["data"],
                model_args=self.model_args,
                config=self.config,
                device=self.device,
            )

            scaled_dit_loss = dit_loss / self.config.grad_accum_groups
            scaled_dit_loss.backward()
            dit_accum += 1

            if dit_accum >= self.config.grad_accum_groups:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    max_norm=5.0,
                )
                self.dit_optimizer.step()
                self.dit_optimizer.zero_grad()
                dit_accum = 0

            # --- Exploration policy loss (REINFORCE or inner PPO) ---
            scene_encoding = group["scene_encoding"]  # [1, N, D] detached
            x_ref = group["x_ref"]                    # [1, T, 4] detached
            eta_lat_01 = group["eta_lat_01"]          # [K] detached sampled values
            eta_lon_01 = group["eta_lon_01"]          # [K] detached sampled values

            # Skip grad computation entirely when policy is frozen
            grad_enabled = not policy_frozen
            advantages_t = torch.tensor(advantages_np, device=self.device, dtype=torch.float32)

            inner_epochs = self.config.exploration_inner_epochs
            clip_eps = self.config.exploration_clip_epsilon

            # Compute old log_probs (detached) for PPO ratio when inner_epochs > 1
            if inner_epochs > 1:
                with torch.no_grad():
                    old_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
                    old_lp = old_output.lat_dist.log_prob(eta_lat_01) + old_output.lon_dist.log_prob(eta_lon_01)
                    if old_lp.dim() > 1:
                        old_lp = old_lp.squeeze(-1)
                    old_log_probs = old_lp.detach()

            for inner_ep in range(inner_epochs):
                # Forward pass — only build autograd graph when policy is not frozen
                with torch.set_grad_enabled(grad_enabled):
                    policy_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
                lat_dist = policy_output.lat_dist
                lon_dist = policy_output.lon_dist

                log_probs = lat_dist.log_prob(eta_lat_01) + lon_dist.log_prob(eta_lon_01)
                if log_probs.dim() > 1:
                    log_probs = log_probs.squeeze(-1)

                if inner_epochs > 1:
                    # PPO clipped objective
                    ratio = (log_probs - old_log_probs).exp()
                    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                    surr1 = ratio * advantages_t
                    surr2 = clipped_ratio * advantages_t
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    # Entropy bonus + KL penalty (same as REINFORCE path)
                    entropy_value = (lat_dist.entropy() + lon_dist.entropy()).mean()
                    init_lat, init_lon = _get_init_distributions(self.device)
                    kl_value = (kl_div(lat_dist, init_lat) + kl_div(lon_dist, init_lon)).mean()

                    policy_loss = (
                        ppo_loss
                        + self.config.exploration_entropy_coef * (-entropy_value)
                        + self.config.exploration_kl_coef * kl_value
                    )
                    policy_metrics = {
                        "exploration_policy_loss": ppo_loss.item(),
                        "exploration_entropy": entropy_value.item(),
                        "exploration_kl": kl_value.item(),
                        "exploration_total_loss": policy_loss.item(),
                        "exploration_eta_lat_mean": lat_dist.mean.mean().item() * 2 - 1,
                        "exploration_eta_lon_mean": lon_dist.mean.mean().item() * 2 - 1,
                        "exploration_eta_lat_std": (lat_dist.variance.mean().item() * 4) ** 0.5,
                        "exploration_eta_lon_std": (lon_dist.variance.mean().item() * 4) ** 0.5,
                    }

                    # Step per inner epoch per group
                    if not policy_frozen:
                        self.policy_optimizer.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.exploration_policy.parameters(), max_norm=1.0,
                        )
                        self.policy_optimizer.step()
                else:
                    # Original REINFORCE (single step, accumulated across groups)
                    policy_loss, policy_metrics = compute_exploration_loss(
                        advantages=advantages_t,
                        log_probs=log_probs,
                        lat_dist=lat_dist,
                        lon_dist=lon_dist,
                        entropy_coef=self.config.exploration_entropy_coef,
                        kl_coef=self.config.exploration_kl_coef,
                    )
                    if not policy_frozen:
                        policy_loss.backward()
                        # Per-group stepping: immediate per-scene gradient signal
                        if self.config.exploration_step_per_group:
                            n_policy_accum += 1
                            if n_policy_accum >= self.config.exploration_grad_accum_groups:
                                torch.nn.utils.clip_grad_norm_(
                                    self.exploration_policy.parameters(), max_norm=1.0,
                                )
                                self.policy_optimizer.step()
                                self.policy_optimizer.zero_grad()
                                n_policy_accum = 0

            if not self.config.exploration_step_per_group:
                n_policy_accum += 1

            # Track per-scene η for variance computation
            per_scene_eta_lat.append(lat_dist.mean.mean().item() * 2 - 1)
            per_scene_eta_lon.append(lon_dist.mean.mean().item() * 2 - 1)

            # Merge metrics
            for k, v in dit_metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            for k, v in policy_metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            num_groups += 1

            if progress_callback is not None:
                progress_callback({
                    "epoch": epoch,
                    "group": group_idx + 1,
                    "total_groups": len(groups),
                    **dit_metrics, **policy_metrics,
                })

        # Flush remaining DiT gradients
        if dit_accum > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            self.dit_optimizer.step()
            self.dit_optimizer.zero_grad()

        # Policy optimizer step: only needed for REINFORCE (inner_epochs=1)
        # without per-group stepping. PPO and per-group both step above.
        if n_policy_accum > 0 and self.config.exploration_inner_epochs <= 1 and not policy_frozen:
            for p in self.exploration_policy.parameters():
                if p.grad is not None:
                    p.grad.div_(n_policy_accum)
            torch.nn.utils.clip_grad_norm_(
                self.exploration_policy.parameters(), max_norm=1.0,
            )
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()

        if num_groups == 0:
            return _empty_metrics()

        result = {k: v / num_groups for k, v in all_metrics.items()}
        # Add per-scene η variance (measures scene-dependence of policy output)
        if per_scene_eta_lat:
            import numpy as _np
            result["exploration_eta_lat_scene_std"] = float(_np.std(per_scene_eta_lat))
            result["exploration_eta_lon_scene_std"] = float(_np.std(per_scene_eta_lon))
        return result

    def train_epoch(
        self,
        npz_paths: list[str],
        epoch: int,
        progress_callback=None,
    ) -> dict[str, float]:
        """Full epoch: generate policy-guided groups, then train both networks."""
        # Apply inverse KL scheduling:
        # DiT KL decays (high→low): keep planner stable early, let it adapt later
        scheduled_dit_kl = self.config.get_kl_coef(epoch, self.config.train_epochs)
        if scheduled_dit_kl != self.config.kl_coef:
            print(f"  [kl_schedule] epoch {epoch}: dit_kl {self.config.kl_coef:.4f} -> {scheduled_dit_kl:.4f}")
            self.config.kl_coef = scheduled_dit_kl

        # Policy KL ramps (low→high): free exploration early, anchor learned policy later
        scheduled_policy_kl = self.config.get_exploration_kl_coef(epoch, self.config.train_epochs)
        if scheduled_policy_kl != self.config.exploration_kl_coef:
            print(f"  [kl_schedule] epoch {epoch}: policy_kl {self.config.exploration_kl_coef:.4f} -> {scheduled_policy_kl:.4f}")
            self.config.exploration_kl_coef = scheduled_policy_kl

        print(f"  Generating policy-guided groups for {len(npz_paths)} scenes (K={self.config.num_generations})...")
        groups = []
        for npz_path in tqdm(npz_paths, desc="Generating groups"):
            group = self.generate_policy_guided_group(npz_path)
            if group is not None:
                groups.append(group)

        print(f"  Generated {len(groups)} valid groups")
        if not groups:
            return _empty_metrics()

        random.shuffle(groups)
        return self.train_on_groups(groups, epoch, progress_callback)

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log training metrics."""
        import pandas as pd

        log_entry = {
            "epoch": epoch,
            "dit_kl_coef": self.config.kl_coef,
            "policy_entropy_coef": self.config.exploration_entropy_coef,
            "policy_kl_coef": self.config.exploration_kl_coef,
            **{f"train_{k}": v for k, v in metrics.items()},
        }
        self.train_log.append(log_entry)

        df = pd.DataFrame(self.train_log)
        log_path = self.run_dir / "grpo_exploration_train_log.tsv"
        df.to_csv(log_path, sep="\t", index=False)

        def _fmt(v: float) -> str:
            if v == 0.0:
                return "0"
            if abs(v) < 0.001:
                return f"{v:.2e}"
            return f"{v:.4f}"

        dit_loss = _fmt(metrics.get("loss", 0))
        pol_loss = _fmt(metrics.get("exploration_total_loss", 0))
        entropy = _fmt(metrics.get("exploration_entropy", 0))
        eta_lat = _fmt(metrics.get("exploration_eta_lat_mean", 0))
        eta_lon = _fmt(metrics.get("exploration_eta_lon_mean", 0))
        eta_std = _fmt(metrics.get("exploration_eta_lat_std", 0))

        scene_std_lat = _fmt(metrics.get("exploration_eta_lat_scene_std", 0))
        scene_std_lon = _fmt(metrics.get("exploration_eta_lon_scene_std", 0))
        print(
            f"  Epoch {epoch}: DiT_loss={dit_loss}, "
            f"Policy_loss={pol_loss}, Entropy={entropy}, "
            f"η_lat={eta_lat}, η_lon={eta_lon}, η_std={eta_std}, "
            f"scene_var_lat={scene_std_lat}, scene_var_lon={scene_std_lon}"
        )

    def save_checkpoint(self, epoch: int, args_dict: dict) -> None:
        """Save both DiT and exploration policy checkpoints."""
        if self.use_lora:
            from preference_optimization.lora_utils import save_lora_checkpoint

            lora_dir = str(self.run_dir / f"lora_epoch_{epoch:03d}")
            save_lora_checkpoint(self.policy_model, lora_dir)
            torch.save(
                {"epoch": epoch, "optimizer": self.dit_optimizer.state_dict()},
                Path(lora_dir) / "optimizer.pth",
            )
            self.config.to_json(Path(lora_dir) / "grpo_config.json")

            # Save exploration policy
            torch.save(
                self.exploration_policy.state_dict(),
                Path(lora_dir) / "exploration_policy.pth",
            )
            self.exploration_policy.config.to_json(
                Path(lora_dir) / "exploration_policy_config.json"
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
                "dit_optimizer": self.dit_optimizer.state_dict(),
                "exploration_policy": self.exploration_policy.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "args": args_dict,
            }
            latest_path = self.run_dir / "latest.pth"
            torch.save(checkpoint_data, latest_path)
            self.config.to_json(self.run_dir / "grpo_config.json")
            self.exploration_policy.config.to_json(
                self.run_dir / "exploration_policy_config.json"
            )

            if epoch % 5 == 0:
                epoch_path = self.run_dir / f"epoch_{epoch:03d}.pth"
                torch.save(checkpoint_data, epoch_path)
                print(f"  Saved checkpoint: {epoch_path}")


def _empty_metrics() -> dict[str, float]:
    return {
        "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0,
        "mean_advantage": 0.0, "advantage_std": 0.0,
        "exploration_total_loss": 0.0, "exploration_entropy": 0.0,
        "exploration_kl": 0.0,
    }
