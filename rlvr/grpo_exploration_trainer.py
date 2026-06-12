"""Joint GRPO + Exploration Policy trainer.

DEPRECATED for guidance-explorer training: the canonical pipeline is
supervised regression on sweep-derived robust labels
(rlvr.train_explorer_regression) with optional closed-loop RL polish
(rlvr.closed_loop.closed_loop_trainer.ClosedLoopExplorationTrainer).
This open-loop group-GRPO trainer is kept for reproducing older runs only.

Extends the standard GRPO training with a learned exploration policy that
outputs (eta_lat, eta_lon) from Beta distributions. The policy and DiT
planner are trained simultaneously:

- Policy: advantage-weighted log_prob (advantage_logprob) or MSE regression
  toward best eta (best_sample_mse). Controlled by exploration_loss_type.
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
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from torch import nn, optim
from torch.distributions import kl_divergence as kl_div
from tqdm import tqdm

import rlvr.guidance_batched  # noqa: F401 -- registers collision_swerve_batched etc.
from exploration_policy.loss import _get_init_distributions, compute_exploration_loss
from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
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


def reward_config_from_grpo(config: GRPOConfig) -> RewardConfig:
    """Build a RewardConfig from every GRPOConfig field that RewardConfig shares.

    Field-name intersection reproduces exactly the mapping run_experiment.py
    uses for its train_reward_config (verified: 52 shared fields, no renames),
    and cannot silently drop newly added reward fields the way a hand-copied
    kwargs list can (the old list here omitted all static-collision fields).
    """
    from dataclasses import fields as _dc_fields

    reward_field_names = {f.name for f in _dc_fields(RewardConfig)}
    kwargs = {
        name: getattr(config, name)
        for name in reward_field_names
        if hasattr(config, name)
    }
    return RewardConfig(**kwargs)


class GRPOExplorationTrainer:
    """Joint trainer for DiT planner (GRPO) + exploration policy.

    Per scene:
      1. Frozen encoder → scene_encoding
      2. LoRA-disabled DiT → x_ref (reference trajectory)
      3. Exploration policy(scene_encoding, x_ref) → Beta distributions
      4. Sample K η values → K trajectories (noise=0, lat+lon guidance)
      5. Score → group-relative advantages
      6. DiT loss: standard GRPO diffusion loss
      7. Policy loss: advantage_logprob or best_sample_mse (see exploration_loss_type)

    Supports inverse KL scheduling: high DiT KL (stable planner) + low policy
    KL (free exploration) early, then swap as policy learns.
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
        self.train_dit = config.train_dit

        if self.train_dit and dit_optimizer is None:
            raise ValueError(
                "train_dit=True requires a dit_optimizer; pass train_dit=False "
                "in the config for frozen-DiT policy-only training."
            )

        assert config.inner_epochs == 1, (
            f"GRPOExplorationTrainer only supports on-policy training (inner_epochs=1), "
            f"got inner_epochs={config.inner_epochs}. For multi-epoch training, use GRPOTrainer."
        )

        # Reward config: every GRPOConfig field shared with RewardConfig
        # (including the static-collision sc_* family — the previous
        # hand-copied list silently dropped them).
        self.reward_config = reward_config_from_grpo(config)

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

        # --- Exploration policy (or random guidance mode) ---
        self.random_guidance_mode = config.random_guidance_mode
        self.use_explorer = self.random_guidance_mode == "explorer"

        self.heads = list(config.exploration_heads)
        if not self.use_explorer and self.heads != ["lateral", "longitudinal"]:
            raise ValueError(
                f"random_guidance_mode={self.random_guidance_mode!r} supports only "
                f"the default ['lateral', 'longitudinal'] heads, got {self.heads}."
            )

        if self.use_explorer:
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
                heads=self.heads,
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

            self.policy_optimizer = optim.AdamW(
                self.exploration_policy.parameters(),
                lr=config.exploration_lr,
            )
        else:
            self.exploration_policy = None
            self.policy_optimizer = None
            print(f"  Random guidance mode: {self.random_guidance_mode} (no explorer network)")

        # Lateral/longitudinal guidance parameters (from GRPOConfig)
        self.lambda_lat = config.exploration_lambda_lat
        self.lambda_lon = config.exploration_lambda_lon
        self.guidance_scale = config.exploration_guidance_scale

        # Tracking
        self.train_log: list[dict] = []
        self.eval_log: list[dict] = []
        self.best_det_reward: float = float("-inf")
        self.best_epoch: int = 0

    def _build_guidance_fns(self, eta_vals: dict[str, torch.Tensor]) -> list[GuidanceConfig]:
        """Map head-name -> GuidanceConfig with per-sample eta tensors.

        Default ["lateral", "longitudinal"] reproduces the original hard-coded
        pair exactly; "collision"/"stretch" use the batched variants from
        rlvr.guidance_batched.
        """
        cfg = self.config
        fns = []
        for head in self.heads:
            eta = eta_vals[head]
            if head == "lateral":
                fns.append(GuidanceConfig(
                    name="lateral", enabled=True, scale=cfg.exploration_lat_scale,
                    params={"lambda_lat": self.lambda_lat, "eta_lat": eta},
                ))
            elif head == "longitudinal":
                fns.append(GuidanceConfig(
                    name="longitudinal", enabled=True, scale=1.0,
                    params={"lambda_lon": self.lambda_lon, "eta_lon": eta},
                ))
            elif head == "collision":
                fns.append(GuidanceConfig(
                    name="collision_swerve_batched", enabled=True,
                    scale=cfg.exploration_col_scale,
                    params={"eta_col": eta, "range": cfg.exploration_col_range},
                ))
            elif head == "stretch":
                fns.append(GuidanceConfig(
                    name="speed_stretch_batched", enabled=True,
                    scale=cfg.exploration_stretch_scale,
                    params={"stretch": 1.0 + cfg.exploration_lambda_spd * eta},
                ))
            else:
                raise ValueError(f"unknown exploration head {head!r}")
        return fns

    def _sample_random_eta(self, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random η values based on random_guidance_mode."""
        mode = self.random_guidance_mode
        if mode == "uniform":
            eta_lat = torch.rand(K, device=self.device) * 2 - 1  # U[-1, 1]
            eta_lon = torch.rand(K, device=self.device) * 2 - 1
        elif mode == "narrow":
            eta_lat = torch.rand(K, device=self.device) - 0.5  # U[-0.5, 0.5]
            eta_lon = (torch.rand(K, device=self.device) - 0.5) * 0.5  # U[-0.25, 0.25]
        elif mode == "gaussian":
            eta_lat = torch.randn(K, device=self.device) * 0.3  # N(0, 0.3)
            eta_lon = torch.randn(K, device=self.device) * 0.15  # N(0, 0.15)
            eta_lat.clamp_(-1, 1)
            eta_lon.clamp_(-1, 1)
        elif mode == "none":
            eta_lat = torch.zeros(K, device=self.device)
            eta_lon = torch.zeros(K, device=self.device)
        else:
            raise ValueError(f"Unknown random_guidance_mode: {mode}")
        return eta_lat, eta_lon

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
        if self.use_explorer:
            self.exploration_policy.eval()

        with torch.no_grad():
            # 1. Deterministic trajectory (eval only, outside training group)
            det_traj = generate_samples(
                model=self.policy_model, model_args=self.model_args,
                data=norm_data, noise_scale=0.0, n_samples=1,
                composer=None, device=self.device,
            )[0]  # (T, 4)

            K = self.config.num_generations

            if self.use_explorer:
                # Explorer path: Beta distributions → η sampling
                x_ref_np = generate_reference_trajectory(
                    self.policy_model, self.model_args, norm_data, self.device,
                )
                x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(self.device)
                norm_data["reference_trajectory"] = x_ref
                scene_encoding = run_frozen_encoder(self.policy_model, norm_data)

                policy_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
                dists = policy_output.dists

                eta_01 = {
                    h: dists[h].rsample((K,)).squeeze(-1) for h in self.heads
                }
                if self.config.exploration_pin_zero_eta:
                    # Slot 0 = forced η=0 (0.5 in Beta space): the unguided
                    # reference the group advantages compare against. Excluded
                    # from the policy log-prob gradient in train_on_groups.
                    for h in self.heads:
                        eta_01[h][0] = 0.5
                eta_vals = {h: 2.0 * v - 1.0 for h, v in eta_01.items()}
            else:
                # Random guidance path: sample η directly (default heads only,
                # validated in __init__)
                eta_lat_vals, eta_lon_vals = self._sample_random_eta(K)

                # Generate reference trajectory (needed by lateral/longitudinal guidance)
                x_ref_np = generate_reference_trajectory(
                    self.policy_model, self.model_args, norm_data, self.device,
                )
                x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(self.device)
                norm_data["reference_trajectory"] = x_ref

                scene_encoding = None
                dists = None
                eta_vals = {"lateral": eta_lat_vals, "longitudinal": eta_lon_vals}
                eta_01 = {h: (v + 1.0) / 2.0 for h, v in eta_vals.items()}

            # 6. Generate K trajectories — batched with per-trajectory varied noise
            from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
            noise_min, noise_max = self.config.noise_scale_range

            # Expand scene data from B=1 to B=K
            K_data = {}
            for k_key, v in norm_data.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                    K_data[k_key] = v.expand(K, *v.shape[1:]).contiguous()
                else:
                    K_data[k_key] = v

            # Build batched composer with K etas per configured head
            guidance_fns = self._build_guidance_fns(eta_vals)
            set_cfg = GuidanceSetConfig(functions=guidance_fns, global_scale=self.guidance_scale)
            composer = GuidanceComposer(set_cfg)

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

        # Store sampled eta values for recomputation during training.
        # PolicyGroupMetadata is lat/lon-specific bookkeeping kept for the
        # default head layout only (train_on_groups recomputes log-probs).
        if self.use_explorer and self.heads == ["lateral", "longitudinal"]:
            policy_meta = PolicyGroupMetadata(
                log_probs=torch.zeros(K, device=self.device),
                lat_dist_params=(dists["lateral"].concentration1.detach(),
                                 dists["lateral"].concentration0.detach()),
                lon_dist_params=(dists["longitudinal"].concentration1.detach(),
                                 dists["longitudinal"].concentration0.detach()),
                eta_lat_samples=eta_vals["lateral"].detach(),
                eta_lon_samples=eta_vals["longitudinal"].detach(),
            )
        else:
            policy_meta = None

        return {
            "npz_path": npz_path,
            "data": data,
            "norm_data": norm_data,
            "trajectories": trajectories,
            "reward_breakdowns": reward_breakdowns,
            "advantages": advantages,
            "policy_meta": policy_meta,
            "det_trajectory": det_traj,
            "scene_encoding": scene_encoding.detach() if scene_encoding is not None else None,
            "x_ref": x_ref.detach() if x_ref is not None else None,
            "eta_01": {h: v.detach() for h, v in eta_01.items()},
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
        if not self.use_explorer:
            policy_frozen = True  # No explorer to train
        if policy_frozen and epoch == freeze_after + 1 and self.use_explorer:
            print(f"  [policy_freeze] Freezing exploration policy after epoch {freeze_after}")

        all_metrics: dict[str, float] = {}
        num_groups = 0
        per_scene_eta: dict[str, list[float]] = {h: [] for h in self.heads}
        # random-guidance branch logs the legacy lateral/longitudinal pair
        for _h in ("lateral", "longitudinal"):
            per_scene_eta.setdefault(_h, [])

        if self.train_dit:
            self.policy_model.train()
        else:
            self.policy_model.eval()
        if self.use_explorer:
            if not policy_frozen:
                self.exploration_policy.train()
            else:
                self.exploration_policy.eval()
        if self.train_dit:
            self.dit_optimizer.zero_grad()
        if self.use_explorer and not policy_frozen:
            self.policy_optimizer.zero_grad()

        dit_accum = 0
        n_policy_accum = 0

        for group_idx, group in enumerate(tqdm(groups, desc=f"Epoch {epoch}")):
            advantages_np = group["advantages"]
            if np.all(advantages_np == 0):
                continue

            # --- DiT GRPO loss (batched) — skipped entirely when the DiT is frozen ---
            if self.train_dit:
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
            else:
                dit_metrics = {}

            # --- Exploration policy loss (advantage_logprob, best_sample_mse, or PPO) ---
            if self.use_explorer:
                scene_encoding = group["scene_encoding"]
                x_ref = group["x_ref"]
                eta_01 = group["eta_01"]

                grad_enabled = not policy_frozen
                advantages_t = torch.tensor(advantages_np, device=self.device, dtype=torch.float32)

                inner_epochs = self.config.exploration_inner_epochs
                clip_eps = self.config.exploration_clip_epsilon

                def _sum_log_probs(dists_by_head):
                    lp = sum(dists_by_head[h].log_prob(eta_01[h]) for h in self.heads)
                    return lp.squeeze(-1) if lp.dim() > 1 else lp

                if inner_epochs > 1:
                    with torch.no_grad():
                        old_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
                        old_log_probs = _sum_log_probs(old_output.dists).detach()

                for inner_ep in range(inner_epochs):
                    with torch.set_grad_enabled(grad_enabled):
                        policy_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
                    dists = policy_output.dists

                    log_probs = _sum_log_probs(dists)

                    # Pinned slot 0 is a forced (not sampled) action — exclude it
                    # from log-prob-based policy gradients. best_sample_mse keeps
                    # all slots: η=0 winning is a valid regression target.
                    if self.config.exploration_pin_zero_eta:
                        pg_log_probs = log_probs[1:]
                        pg_advantages = advantages_t[1:]
                    else:
                        pg_log_probs = log_probs
                        pg_advantages = advantages_t

                    if inner_epochs > 1:
                        pg_old_log_probs = (
                            old_log_probs[1:]
                            if self.config.exploration_pin_zero_eta
                            else old_log_probs
                        )
                        ratio = (pg_log_probs - pg_old_log_probs).exp()
                        clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                        surr1 = ratio * pg_advantages
                        surr2 = clipped_ratio * pg_advantages
                        ppo_loss = -torch.min(surr1, surr2).mean()

                        entropy_value = sum(d.entropy().mean() for d in dists.values())
                        init_dist, _ = _get_init_distributions(self.device)
                        kl_value = sum(kl_div(d, init_dist).mean() for d in dists.values())

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
                        }
                        for h, d in dists.items():
                            policy_metrics[f"exploration_eta_{h}_mean"] = d.mean.mean().item() * 2 - 1
                            policy_metrics[f"exploration_eta_{h}_std"] = (d.variance.mean().item() * 4) ** 0.5

                        if not policy_frozen:
                            self.policy_optimizer.zero_grad()
                            policy_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.exploration_policy.parameters(), max_norm=1.0,
                            )
                            self.policy_optimizer.step()
                    elif self.config.exploration_loss_type == "best_sample_mse":
                        # Ranked SFT for explorer: MSE regression of policy mean toward best eta.
                        # Unlike advantage_logprob which uses all K samples, this directly supervises
                        # the policy to output the best-reward eta for each scene.
                        best_idx = advantages_t.argmax()
                        rsft_loss = sum(
                            (dists[h].mean.squeeze() - eta_01[h][best_idx].detach()) ** 2
                            for h in self.heads
                        )

                        policy_loss = rsft_loss
                        policy_metrics = {
                            "exploration_policy_loss": rsft_loss.item(),
                            "exploration_entropy": float(sum(d.entropy().mean() for d in dists.values())),
                            "exploration_kl": 0.0,
                            "exploration_total_loss": rsft_loss.item(),
                        }
                        for h, d in dists.items():
                            policy_metrics[f"exploration_eta_{h}_mean"] = d.mean.mean().item() * 2 - 1
                            policy_metrics[f"exploration_eta_{h}_std"] = (d.variance.mean().item() * 4) ** 0.5
                    else:
                        policy_loss, policy_metrics = compute_exploration_loss(
                            advantages=pg_advantages,
                            log_probs=pg_log_probs,
                            dists=dists,
                            entropy_coef=self.config.exploration_entropy_coef,
                            kl_coef=self.config.exploration_kl_coef,
                            action_cost_coef=self.config.exploration_action_cost,
                        )
                    # Backward pass for advantage_logprob and best_sample_mse paths
                    # (PPO path handles its own backward+step above)
                    if not policy_frozen and inner_epochs <= 1:
                        policy_loss.backward()
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

                for h in self.heads:
                    per_scene_eta[h].append(
                        dists[h].mean.mean().item() * 2 - 1)
            else:
                # Random guidance: no policy to train, just track η stats
                eta_lat_01 = group["eta_01"]["lateral"]
                eta_lon_01 = group["eta_01"]["longitudinal"]
                eta_lat_vals = 2.0 * eta_lat_01 - 1.0
                eta_lon_vals = 2.0 * eta_lon_01 - 1.0
                policy_metrics = {
                    "exploration_policy_loss": 0.0,
                    "exploration_entropy": 0.0,
                    "exploration_kl": 0.0,
                    "exploration_total_loss": 0.0,
                    "exploration_eta_lat_mean": eta_lat_vals.mean().item(),
                    "exploration_eta_lon_mean": eta_lon_vals.mean().item(),
                    "exploration_eta_lat_std": eta_lat_vals.std().item(),
                    "exploration_eta_lon_std": eta_lon_vals.std().item(),
                }
                per_scene_eta["lateral"].append(eta_lat_vals.mean().item())
                per_scene_eta["longitudinal"].append(eta_lon_vals.mean().item())

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
        if self.train_dit and dit_accum > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            self.dit_optimizer.step()
            self.dit_optimizer.zero_grad()

        # Policy optimizer step: only needed for non-PPO (inner_epochs=1)
        # without per-group stepping. PPO and per-group both step above.
        if self.use_explorer and n_policy_accum > 0 and self.config.exploration_inner_epochs <= 1 and not policy_frozen:
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
        # Add per-scene η variance (measures scene-dependence of policy
        # output), keyed by the ACTUAL head name (previously heads[0]/[-1]
        # were logged as lat/lon regardless of the head spec).
        import numpy as _np
        for h, vals in per_scene_eta.items():
            if vals:
                result[f"exploration_eta_{h}_scene_std"] = float(_np.std(vals))
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

        # Scene-level reward trimming: drop top and bottom X% of scenes by mean reward
        trim = self.config.reward_trim_pct
        if trim > 0 and len(groups) >= 10:
            n = len(groups)
            n_trim = max(1, int(n * trim))
            mean_rewards = [np.mean([r.total for r in g["reward_breakdowns"]]) for g in groups]
            sorted_idx = sorted(range(n), key=lambda i: mean_rewards[i])
            keep_idx = sorted_idx[n_trim:n - n_trim]
            groups = [groups[i] for i in keep_idx]
            print(f"  Trimmed {2*n_trim} scenes ({trim*100:.0f}% each end), keeping {len(groups)}/{n}")

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

    @torch.no_grad()
    def evaluate_policy_guided(
        self,
        npz_paths: list[str],
        reward_config: RewardConfig | None = None,
        label: str = "policy-eval",
    ) -> dict[str, float]:
        """Deterministic policy-guided eval: per scene, run the policy with
        deterministic=True (Beta means → η), generate ONE guided trajectory
        (noise=0) plus the unguided det trajectory, score both.

        Unlike evaluate_checkpoint (which ignores the policy and is constant
        when the DiT is frozen), this measures what the explorer actually does.
        Returns aggregate metrics keyed like evaluate_checkpoint (reward_mean,
        rb_crossings, collision_rate) plus guidance-specific extras.
        """
        if not self.use_explorer:
            raise ValueError("evaluate_policy_guided requires random_guidance_mode='explorer'")
        reward_config = reward_config or self.reward_config

        self.policy_model.eval()
        self.exploration_policy.eval()

        rows: list[dict] = []
        for npz_path in tqdm(npz_paths, desc=label):
            try:
                data = _load_npz(npz_path, self.device)
            except Exception as e:
                print(f"  [{label}] skipping {npz_path}: {e}")
                continue
            norm_data = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
            norm_data = self.model_args.observation_normalizer(norm_data)

            x_ref_np = generate_reference_trajectory(
                self.policy_model, self.model_args, norm_data, self.device,
            )
            x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(self.device)
            norm_data["reference_trajectory"] = x_ref
            scene_encoding = run_frozen_encoder(self.policy_model, norm_data)

            policy_output = self.exploration_policy(scene_encoding, x_ref, deterministic=True)
            etas = {
                h: (2.0 * policy_output.dists[h].mean - 1.0).reshape(1)
                for h in self.heads
            }

            guidance_fns = self._build_guidance_fns(etas)
            set_cfg = GuidanceSetConfig(functions=guidance_fns, global_scale=self.guidance_scale)
            composer = GuidanceComposer(set_cfg)

            guided_traj = generate_samples(
                model=self.policy_model, model_args=self.model_args,
                data=norm_data, noise_scale=0.0, n_samples=1,
                composer=composer, device=self.device,
            )[0]
            det_traj = generate_samples(
                model=self.policy_model, model_args=self.model_args,
                data=norm_data, noise_scale=0.0, n_samples=1,
                composer=None, device=self.device,
            )[0]

            traj_batch = torch.tensor(
                np.stack([guided_traj, det_traj]),
                device=self.device, dtype=torch.float32,
            )
            guided_bd, det_bd = compute_reward_batch(traj_batch, data, reward_config)
            deviation = float(
                np.linalg.norm(
                    np.asarray(guided_traj)[:, :2] - np.asarray(det_traj)[:, :2], axis=-1
                ).mean()
            )
            rows.append({
                "npz_path": npz_path,
                **{f"eta_{h}": float(v.item()) for h, v in etas.items()},
                "reward": guided_bd.total,
                "det_reward": det_bd.total,
                "static_crossing": bool(guided_bd.static_crossing),
                "det_static_crossing": bool(det_bd.static_crossing),
                "sc_min_dist": float(guided_bd.sc_min_dist),
                "det_sc_min_dist": float(det_bd.sc_min_dist),
                "rb_crossing": bool(guided_bd.rb_crossing),
                "collision": guided_bd.collision_step is not None,
                "sc_n_stopped": int(guided_bd.sc_n_stopped),
                "deviation": deviation,
            })

        self.last_eval_rows = rows  # per-scene detail for eval/viz tools
        return aggregate_policy_eval(rows)

    def save_checkpoint(self, epoch: int, args_dict: dict) -> None:
        """Save both DiT and exploration policy checkpoints.

        With train_dit=False only the exploration policy (+ its optimizer and
        config) is saved — the frozen DiT is the unchanged input checkpoint.
        """
        if not self.train_dit:
            torch.save(
                self.exploration_policy.state_dict(),
                self.run_dir / f"exploration_policy_epoch_{epoch:03d}.pth",
            )
            torch.save(
                self.exploration_policy.state_dict(),
                self.run_dir / "exploration_policy.pth",
            )
            torch.save(
                self.policy_optimizer.state_dict(),
                self.run_dir / "policy_optimizer.pth",
            )
            self.exploration_policy.config.to_json(
                self.run_dir / "exploration_policy_config.json"
            )
            self.config.to_json(self.run_dir / "grpo_config.json")
            return
        if self.use_lora:
            from preference_optimization.lora_utils import save_lora_checkpoint

            lora_dir = str(self.run_dir / f"lora_epoch_{epoch:03d}")
            save_lora_checkpoint(self.policy_model, lora_dir)
            torch.save(
                {"epoch": epoch, "optimizer": self.dit_optimizer.state_dict()},
                Path(lora_dir) / "optimizer.pth",
            )
            self.config.to_json(Path(lora_dir) / "grpo_config.json")

            # Save exploration policy (if using explorer)
            if self.use_explorer:
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
                "args": args_dict,
            }
            if self.use_explorer:
                checkpoint_data["exploration_policy"] = self.exploration_policy.state_dict()
                checkpoint_data["policy_optimizer"] = self.policy_optimizer.state_dict()
            latest_path = self.run_dir / "latest.pth"
            torch.save(checkpoint_data, latest_path)
            self.config.to_json(self.run_dir / "grpo_config.json")
            if self.use_explorer:
                self.exploration_policy.config.to_json(
                    self.run_dir / "exploration_policy_config.json"
                )

            if epoch % 5 == 0:
                epoch_path = self.run_dir / f"epoch_{epoch:03d}.pth"
                torch.save(checkpoint_data, epoch_path)
                print(f"  Saved checkpoint: {epoch_path}")


def aggregate_policy_eval(rows: list[dict]) -> dict[str, float]:
    """Aggregate per-scene policy-guided eval rows into summary metrics.

    Keys mirror evaluate_checkpoint (reward_mean, rb_crossings,
    collision_rate) so run_experiment best-tracking works unchanged, plus
    guidance-specific extras. Scenes are split into avoidance
    (sc_n_stopped > 0) vs normal for the per-type |η| inertness metrics.
    """
    if not rows:
        return {
            "reward_mean": float("-inf"), "rb_crossings": 999,
            "collision_rate": 1.0, "n_scenes": 0,
        }
    avoid = [r for r in rows if r["sc_n_stopped"] > 0]
    normal = [r for r in rows if r["sc_n_stopped"] == 0]

    def _mean(vals):
        return float(np.mean(vals)) if len(vals) else 0.0

    out = {
        "n_scenes": len(rows),
        "reward_mean": _mean([r["reward"] for r in rows]),
        "det_reward_mean": _mean([r["det_reward"] for r in rows]),
        "rb_crossings": int(sum(r["rb_crossing"] for r in rows)),
        "collision_rate": _mean([float(r["collision"]) for r in rows]),
        "static_crossings": int(sum(r["static_crossing"] for r in rows)),
        "det_static_crossings": int(sum(r["det_static_crossing"] for r in rows)),
        "sc_min_dist_mean": _mean([r["sc_min_dist"] for r in avoid]),
        "det_sc_min_dist_mean": _mean([r["det_sc_min_dist"] for r in avoid]),
        "deviation_mean": _mean([r["deviation"] for r in rows]),
        "deviation_max": float(max((r["deviation"] for r in rows), default=0.0)),
        "n_avoidance_scenes": len(avoid),
        "deviation_mean_normal": _mean([r["deviation"] for r in normal]),
    }
    eta_keys = [k for k in rows[0] if k.startswith("eta_")]
    for k in eta_keys:
        suffix = k[len("eta_"):]
        # Short aliases for the original layout: lateral->lat, longitudinal->lon
        suffix = {"lateral": "lat", "longitudinal": "lon"}.get(suffix, suffix)
        out[f"eta_{suffix}_abs_avoid"] = _mean([abs(r[k]) for r in avoid])
        out[f"eta_{suffix}_abs_normal"] = _mean([abs(r[k]) for r in normal])
    return out


def _empty_metrics() -> dict[str, float]:
    return {
        "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0,
        "mean_advantage": 0.0, "advantage_std": 0.0,
        "exploration_total_loss": 0.0, "exploration_entropy": 0.0,
        "exploration_kl": 0.0,
    }
