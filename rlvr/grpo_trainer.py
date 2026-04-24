"""GRPO Trainer — manages reinforcement fine-tuning loop.

Supports two modes via GRPOConfig:
- On-policy (inner_epochs=1): Single gradient step per rollout batch.
- Multi-epoch (inner_epochs>1): Multiple gradient steps per rollout with
  PPO-clipped importance sampling for higher sample efficiency.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from preference_optimization.utils import (
    calculate_ade,
    generate_deterministic_trajectory,
)
from preference_optimization.utils import (
    load_npz_data as _load_npz_data_raw,
)
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_direct_best_loss, compute_grpo_loss, compute_log_probs
from rlvr.grpo_sampler import SamplerConfig, generate_diverse_group
from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_group_advantages,
    compute_reward_batch,
)


def load_npz_data(npz_path, device):
    """Wrapper around preference_optimization load_npz_data that adds v4 delay key."""
    data = _load_npz_data_raw(npz_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)
    return data


class GRPOTrainer:
    """Trainer for GRPO with configurable inner-loop strategy.

    Outer loop (per epoch):
        1. Generate N trajectories per scene (expensive diffusion sampling)
        2. Score with rule-based rewards, compute advantages
        3. Store old_log_probs (behavior reference for importance sampling)

    Inner loop (M steps per rollout batch):
        - M=1 (on-policy): single gradient step, no importance sampling
        - M>1 (multi-epoch): PPO-clipped updates reusing the same rollout
    """

    def __init__(
        self,
        policy_model: nn.Module,
        model_args,
        optimizer: optim.Optimizer,
        device: torch.device,
        run_dir: Path,
        config: GRPOConfig,
        use_lora: bool = False,
    ):
        self.policy_model = policy_model
        self.model_args = model_args
        self.optimizer = optimizer
        self.device = device
        self.run_dir = run_dir
        self.config = config
        self.use_lora = use_lora

        # Build sub-configs from the master config
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
            enable_lateral=config.enable_lateral,
            enable_longitudinal=config.enable_longitudinal,
            lambda_lat=config.lambda_lat,
            lambda_lon=config.lambda_lon,
            guidance_prob=config.guidance_prob,
            prototypes_path=config.prototypes_path,
        )
        self.reward_config = RewardConfig(
            w_safety=config.w_safety,
            w_progress=config.w_progress,
            w_smooth=config.w_smooth,
            w_feasibility=config.w_feasibility,
            w_centerline=config.w_centerline,
            rb_near_scale=config.rb_near_scale,
            rb_wide_scale=config.rb_wide_scale,
            rb_cont_scale=config.rb_cont_scale,
            rb_gate_enabled=config.rb_gate_enabled,
            rb_penalty_mode=config.rb_penalty_mode,
            rb_cross_thresh=config.rb_cross_thresh,
            rb_near_thresh=config.rb_near_thresh,
            rb_wide_thresh=config.rb_wide_thresh,
            rb_cont_thresh=config.rb_cont_thresh,
            max_lat_accel=config.max_lat_accel,
            lat_accel_scale=config.lat_accel_scale,
            enable_overprogress=config.enable_overprogress,
            overprogress_margin=config.overprogress_margin,
            overprogress_penalty=config.overprogress_penalty,
            stopped_penalty=config.stopped_penalty,
            reward_mode=config.reward_mode,
            enable_lane_departure=config.enable_lane_departure,
            lane_gate_enabled=config.lane_gate_enabled,
            lane_near_scale=config.lane_near_scale,
            lane_wide_scale=config.lane_wide_scale,
            lane_cont_scale=config.lane_cont_scale,
            lane_cross_thresh=config.lane_cross_thresh,
            lane_near_thresh=config.lane_near_thresh,
            lane_wide_thresh=config.lane_wide_thresh,
            lane_cont_thresh=config.lane_cont_thresh,
            max_yaw_rate=config.max_yaw_rate,
            max_steer=config.max_steer,
            kinematic_margin=config.kinematic_margin,
            underprogress_penalty=config.underprogress_penalty,
            underprogress_threshold=config.underprogress_threshold,
            underprogress_reference=config.underprogress_reference,
        )

        # Evaluation: fixed scene subset from validation set, sampled once
        self._eval_sampler_config = SamplerConfig(
            n_trajectories=8,
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
            enable_lateral=config.enable_lateral,
            enable_longitudinal=config.enable_longitudinal,
            lambda_lat=config.lambda_lat,
            lambda_lon=config.lambda_lon,
            guidance_prob=config.guidance_prob,
            prototypes_path=config.prototypes_path,
        )
        self._eval_scene_paths: list[str] | None = None
        self.eval_log: list[dict] = []
        self.best_det_reward: float = float("-inf")
        self.best_epoch: int = 0

        self.train_log: list[dict] = []

    # Expose beta/grad_accum as properties so the GUI can tweak them
    @property
    def beta(self) -> float:
        return self.config.kl_coef

    @beta.setter
    def beta(self, value: float):
        self.config.kl_coef = value

    @property
    def grad_accum_groups(self) -> int:
        return self.config.grad_accum_groups

    @grad_accum_groups.setter
    def grad_accum_groups(self, value: int):
        self.config.grad_accum_groups = value

    def generate_and_score_group(self, npz_path: str) -> dict | None:
        """Generate N trajectories, score, compute advantages, store old_log_probs.

        Returns dict with keys:
            npz_path, data, trajectories, reward_breakdowns, advantages,
            old_log_probs (Tensor (N,) — behavior reference for IS ratio)
        """
        try:
            data = load_npz_data(npz_path, self.device)
        except Exception as e:
            print(f"  [grpo] skipping {npz_path}: {e}")
            return None

        # Skip scenes where GT barely moves (<1m)
        if "ego_agent_future" in data:
            gt = data["ego_agent_future"]
            if gt.dim() == 3:
                gt = gt[0]
            gt_path = torch.diff(gt[:, :2], dim=0).norm(dim=-1).sum()
            if gt_path < 1.0:
                return None

        # Skip scenes where ego starts offroad at t=0 — these are poison
        # because the model can't control the starting position.
        from rlvr.reward import _build_lane_polygons, _ego_on_road_polygon
        t0_traj = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], device=self.device)
        es = data.get("ego_shape")
        if es is not None:
            if es.dim() == 2:
                es = es[0]
            es = es[:3]
        else:
            es = torch.tensor([2.79, 4.34, 1.70], device=self.device)
        lanes = data["lanes"]
        if lanes.dim() == 4:
            lanes = lanes[0]
        lane_polys = _build_lane_polygons(lanes)
        _, frac_t0, _, _ = _ego_on_road_polygon(t0_traj, es, lane_polys)
        # Use generous threshold — if even 1% of ego perimeter is offroad
        # at t=0, the scene is likely at a boundary where clean driving is impossible.
        if frac_t0[0, 0].item() > 0.01:
            return None

        self.policy_model.eval()
        with torch.no_grad():
            # Use batched generation (~3 forward passes instead of K sequential)
            from rlvr.grpo_sampler_batched import generate_diverse_group_batched
            traj_batch = generate_diverse_group_batched(
                model=self.policy_model,
                model_args=self.model_args,
                data=data,
                config=self.sampler_config,
                device=self.device,
            )  # [K, T, 4]

        trajectories = [traj_batch[k].cpu().numpy() for k in range(traj_batch.shape[0])]
        reward_breakdowns = compute_reward_batch(
            traj_batch, data, self.reward_config,
        )

        # Rejection sampling: keep only top K trajectories by reward
        keep = self.config.rejection_keep
        if keep and 0 < keep < len(trajectories):
            totals = [r.total for r in reward_breakdowns]
            top_indices = sorted(range(len(totals)), key=lambda i: totals[i], reverse=True)[:keep]
            top_indices.sort()  # preserve order
            trajectories = [trajectories[i] for i in top_indices]
            reward_breakdowns = [reward_breakdowns[i] for i in top_indices]
            traj_batch = traj_batch[top_indices]

        advantages = compute_group_advantages(
            reward_breakdowns,
            mode=self.config.advantage_mode,
            fixed_scale=self.config.advantage_fixed_scale,
        )

        # Store old log-probs and the (noise, t) used to compute them.
        # Reusing the same (noise, t) during training ensures a consistent
        # importance sampling ratio.
        # Use batched log-prob computation (1 forward pass for all K trajs)
        from rlvr.grpo_loss import compute_batched_trajectory_losses
        B = data["ego_current_state"].shape[0]
        P = 1 + self.model_args.predicted_neighbor_num
        future_len = self.model_args.future_len
        eps = 1e-3
        old_noise = torch.randn(1, P, future_len, 4, device=self.device)
        old_t = torch.rand(1, device=self.device) * (1 - eps) + eps

        was_training = self.policy_model.training
        self.policy_model.train()
        with torch.no_grad():
            losses = compute_batched_trajectory_losses(
                self.policy_model, data, traj_batch, self.model_args,
                old_noise, old_t, self.device,
            )
            old_log_probs = -losses  # (K,)
        if not was_training:
            self.policy_model.eval()

        result = {
            "npz_path": npz_path,
            "data": data,
            "trajectories": trajectories,
            "reward_breakdowns": reward_breakdowns,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
            "old_noise": old_noise,
            "old_t": old_t,
        }

        # For logprob loss: also collect the denoising rollout chain
        if self.config.grpo_loss_type == "advantage_logprob":
            from rlvr.grpo_logprob_loss import collect_logprob_rollout
            was_training = self.policy_model.training
            self.policy_model.eval()
            rollout = collect_logprob_rollout(
                model=self.policy_model,
                data=data,
                trajectories=traj_batch,
                model_args=self.model_args,
                config=self.config,
                device=self.device,
            )
            if was_training:
                self.policy_model.train()
            result["rollout"] = rollout

        return result

    def train_on_groups(
        self,
        groups: list[dict],
        epoch: int,
        progress_callback=None,
    ) -> dict[str, float]:
        """Train on collected groups with M inner epochs and gradient accumulation.

        For M=1 (on-policy): single pass, no importance sampling.
        For M>1 (multi-epoch): reuse rollouts with PPO clipping.
        """
        if not groups:
            return _empty_metrics()

        M = self.config.inner_epochs
        if M > 1 and self.config.grpo_loss_type == "advantage_logprob":
            raise ValueError(
                "inner_epochs > 1 is not supported with grpo_loss_type='advantage_logprob'. "
                "Logprob GRPO uses on-policy REINFORCE without importance sampling."
            )
        all_metrics: dict[str, float] = {}
        total_inner_steps = 0

        for inner_epoch in range(M):
            random.shuffle(groups)
            self.policy_model.train()
            self.optimizer.zero_grad()

            num_groups = 0
            accum_count = 0

            desc = f"Epoch {epoch}" if M == 1 else f"Epoch {epoch} inner {inner_epoch + 1}/{M}"
            for group_idx, group in enumerate(tqdm(groups, desc=desc)):
                advantages = group["advantages"]

                if np.all(advantages == 0):
                    continue

                if self.config.grpo_loss_type == "advantage_logprob":
                    # DDV2-style log-probability GRPO loss
                    from rlvr.grpo_logprob_loss import compute_logprob_grpo_loss
                    rollout = group.get("rollout")
                    if rollout is None:
                        continue
                    loss, metrics = compute_logprob_grpo_loss(
                        model=self.policy_model,
                        rollout=rollout,
                        advantages=advantages,
                        data=group["data"],
                        model_args=self.model_args,
                        config=self.config,
                        device=self.device,
                    )
                elif self.config.loss_mode == "direct_best":
                    # Direct regression: find best trajectory, regress det output toward it
                    rewards = group["reward_breakdowns"]
                    best_idx = int(np.argmax([r.total for r in rewards]))
                    best_traj = group["trajectories"][best_idx]

                    loss, metrics = compute_direct_best_loss(
                        policy_model=self.policy_model,
                        best_trajectory=best_traj,
                        data=group["data"],
                        model_args=self.model_args,
                        device=self.device,
                        config=self.config,
                    )
                else:
                    if M == 1:
                        # Batched GRPO loss: all K trajectories in ONE forward pass
                        from rlvr.grpo_loss import compute_batched_grpo_loss
                        traj_tensor = torch.tensor(
                            np.stack(group["trajectories"]),
                            device=self.device, dtype=torch.float32,
                        )
                        loss, metrics = compute_batched_grpo_loss(
                            policy_model=self.policy_model,
                            trajectories_tensor=traj_tensor,
                            advantages=advantages,
                            data=group["data"],
                            model_args=self.model_args,
                            config=self.config,
                            device=self.device,
                        )
                    else:
                        # Sequential GRPO loss for inner_epochs > 1 (importance sampling)
                        old_lp = group.get("old_log_probs")
                        old_noise = group.get("old_noise")
                        old_t = group.get("old_t")
                        loss, metrics = compute_grpo_loss(
                            policy_model=self.policy_model,
                            trajectories=group["trajectories"],
                            advantages=advantages,
                            data=group["data"],
                            model_args=self.model_args,
                            config=self.config,
                            device=self.device,
                            old_log_probs=old_lp,
                            old_noise=old_noise,
                            old_t=old_t,
                        )

                scaled_loss = loss / self.config.grad_accum_groups
                scaled_loss.backward()
                accum_count += 1

                if accum_count >= self.config.grad_accum_groups:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.policy_model.parameters() if p.requires_grad],
                        max_norm=5.0,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0

                for k, v in metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
                num_groups += 1
                total_inner_steps += 1

                if progress_callback is not None:
                    progress_callback({
                        "epoch": epoch,
                        "inner_epoch": inner_epoch + 1,
                        "inner_epochs_total": M,
                        "group": group_idx + 1,
                        "total_groups": len(groups),
                        **metrics,
                    })

            # Flush remaining accumulated gradients
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    max_norm=5.0,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        if total_inner_steps == 0:
            return _empty_metrics()

        return {k: v / total_inner_steps for k, v in all_metrics.items()}

    def train_epoch(
        self,
        npz_paths: list[str],
        epoch: int,
        progress_callback=None,
    ) -> dict[str, float]:
        """Full epoch: generate groups for all scenes, then train (with inner epochs)."""
        # Apply KL schedule
        scheduled_kl = self.config.get_kl_coef(epoch, self.config.train_epochs)
        if scheduled_kl != self.config.kl_coef:
            print(f"  [kl_schedule] epoch {epoch}: kl_coef {self.config.kl_coef:.4f} -> {scheduled_kl:.4f}")
            self.config.kl_coef = scheduled_kl

        print(f"  Generating trajectory groups for {len(npz_paths)} scenes (N={self.config.num_generations})...")
        groups = []
        for npz_path in tqdm(npz_paths, desc="Generating groups"):
            group = self.generate_and_score_group(npz_path)
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

        return self.train_on_groups(groups, epoch, progress_callback)

    def setup_eval_scenes(self, valid_npz_paths: list[str], n_scenes: int = 50) -> None:
        """Sample and fix the validation scenes used for per-epoch evaluation.

        Called once before training. The same scenes are reused every epoch
        so reward trends are comparable across epochs.
        """
        eval_scenes_path = self.run_dir / "eval_scenes.json"
        if eval_scenes_path.exists():
            with open(eval_scenes_path) as f:
                self._eval_scene_paths = json.load(f)
            print(f"  Loaded {len(self._eval_scene_paths)} fixed eval scenes from {eval_scenes_path}")
            return

        rng = np.random.default_rng(42)
        n = min(n_scenes, len(valid_npz_paths))
        indices = rng.choice(len(valid_npz_paths), size=n, replace=False)
        self._eval_scene_paths = [valid_npz_paths[i] for i in indices]

        with open(eval_scenes_path, "w") as f:
            json.dump(self._eval_scene_paths, f, indent=2)
        print(f"  Fixed {n} eval scenes (from {len(valid_npz_paths)} validation) -> {eval_scenes_path}")

    @torch.no_grad()
    def evaluate_rewards(self, epoch: int, seed: int = 42) -> dict[str, float]:
        """Evaluate on fixed validation scenes: deterministic + stochastic trajectories.

        For each scene:
        1. Generate 1 deterministic trajectory (noise=0, no guidance) — the deployment output
        2. Generate 8 stochastic trajectories (diverse noise/guidance) — for distribution stats

        Uses a fixed random seed for reproducibility across epochs and runs.
        """
        if not self._eval_scene_paths:
            return {}

        from guidance_gui.generate_samples import generate_samples

        self.policy_model.eval()

        # Fix all random seeds for reproducible evaluation across runs.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Deterministic trajectory metrics (the deployment-relevant output)
        det_totals = []
        det_collisions = 0
        det_offroad = []
        det_rb_crossings = 0
        det_rb_near = []
        det_components = {k: [] for k in ["safety", "progress", "smoothness", "feasibility", "centerline"]}

        # Stochastic group metrics (for distribution/diversity stats)
        all_totals = []
        all_collisions = 0
        all_offroad = []
        scene_spreads = []
        scene_means = []

        for path in self._eval_scene_paths:
            try:
                data = load_npz_data(path, self.device)

                # Normalize data once for generate_samples
                norm_data = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                }
                norm_data = self.model_args.observation_normalizer(norm_data)

                # 1. Deterministic trajectory (noise=0, no guidance)
                det_traj = generate_samples(
                    self.policy_model, self.model_args, norm_data,
                    noise_scale=0.0, n_samples=1, composer=None, device=self.device,
                )  # (1, T, 4)
                det_traj_t = torch.tensor(det_traj, device=self.device, dtype=torch.float32)
                det_reward = compute_reward_batch(det_traj_t, data, self.reward_config)[0]

                det_totals.append(det_reward.total)
                if det_reward.collision_step is not None:
                    det_collisions += 1
                det_offroad.append(det_reward.off_road_fraction)
                if det_reward.rb_crossing:
                    det_rb_crossings += 1
                det_rb_near.append(det_reward.rb_near_penalty)
                det_components["safety"].append(det_reward.safety)
                det_components["progress"].append(det_reward.progress)
                det_components["smoothness"].append(det_reward.smoothness)
                det_components["feasibility"].append(det_reward.feasibility)
                det_components["centerline"].append(det_reward.centerline)

                # 2. Stochastic group (8 diverse trajectories)
                sampled = generate_diverse_group(
                    self.policy_model, self.model_args, data,
                    self._eval_sampler_config, self.device,
                )
                trajs = torch.tensor(
                    np.stack([s.trajectory for s in sampled]),
                    device=self.device, dtype=torch.float32,
                )
                rewards = compute_reward_batch(trajs, data, self.reward_config)

                totals = [r.total for r in rewards]
                all_totals.extend(totals)
                all_collisions += sum(1 for r in rewards if r.collision_step is not None)
                all_offroad.extend([r.off_road_fraction for r in rewards])
                scene_spreads.append(max(totals) - min(totals))
                scene_means.append(float(np.mean(totals)))

            except Exception as e:
                print(f"  [eval] skipping {path}: {e}")

        if not det_totals:
            return {}

        n_scenes = len(det_totals)
        det_arr = np.array(det_totals)
        det_offroad_arr = np.array(det_offroad)
        totals_arr = np.array(all_totals)
        offroad_arr = np.array(all_offroad)
        spreads_arr = np.array(scene_spreads)
        scene_means_arr = np.array(scene_means)
        cfg = self.reward_config

        eval_metrics = {
            "epoch": epoch,
            "n_scenes": n_scenes,
            # Deterministic (deployment) metrics
            "det_reward_mean": float(det_arr.mean()),
            "det_reward_median": float(np.median(det_arr)),
            "det_reward_std": float(det_arr.std()),
            "det_collision_rate": det_collisions / n_scenes,
            "det_offroad_mean": float(det_offroad_arr.mean()),
            "det_rb_crossings": det_rb_crossings,
            "det_rb_near_mean": float(np.mean(det_rb_near)) if det_rb_near else 0.0,
            "det_w_safety": float(np.mean(det_components["safety"]) * cfg.w_safety),
            "det_w_progress": float(np.mean(det_components["progress"]) * cfg.w_progress),
            "det_w_smooth": float(np.mean(det_components["smoothness"]) * cfg.w_smooth),
            "det_w_feasibility": float(np.mean(det_components["feasibility"]) * cfg.w_feasibility),
            "det_w_centerline": float(np.mean(det_components["centerline"]) * cfg.w_centerline),
            # Stochastic group metrics
            "group_reward_mean": float(totals_arr.mean()),
            "group_reward_median": float(np.median(totals_arr)),
            "group_scene_mean": float(scene_means_arr.mean()),
            "group_collision_rate": all_collisions / len(all_totals),
            "group_offroad_mean": float(offroad_arr.mean()),
            "group_spread_mean": float(spreads_arr.mean()),
        }

        self.eval_log.append(eval_metrics)
        df = pd.DataFrame(self.eval_log)
        eval_log_path = self.run_dir / "grpo_eval_log.tsv"
        df.to_csv(eval_log_path, sep="\t", index=False)

        # Track best deterministic reward and save best checkpoint
        det_mean = eval_metrics["det_reward_mean"]
        is_best = det_mean > self.best_det_reward
        if is_best:
            self.best_det_reward = det_mean
            self.best_epoch = epoch
            self._save_best_checkpoint(epoch)

        best_tag = " ** NEW BEST **" if is_best else f" (best: epoch {self.best_epoch} = {self.best_det_reward:+.1f})"

        print(
            f"  Eval (epoch {epoch}, {n_scenes} scenes):\n"
            f"    DET:   reward={det_arr.mean():+.1f} median={np.median(det_arr):+.1f}  "
            f"collision={det_collisions/n_scenes:.1%}  rb_cross={det_rb_crossings}/{n_scenes}  rb_near={np.mean(det_rb_near):.2f}{best_tag}\n"
            f"    GROUP: reward={totals_arr.mean():+.1f} scene_mean={scene_means_arr.mean():+.1f}  "
            f"collision={all_collisions/len(all_totals):.1%}  offroad={offroad_arr.mean():.1%}  "
            f"spread={spreads_arr.mean():.1f}"
        )

        # Write machine-readable summary for agentic consumption
        self._write_run_summary(epoch, eval_metrics, is_best)

        return eval_metrics

    def _save_best_checkpoint(self, epoch: int) -> None:
        """Copy the current checkpoint as the best model."""
        if self.use_lora:
            from preference_optimization.lora_utils import save_lora_checkpoint
            best_dir = str(self.run_dir / "lora_best")
            save_lora_checkpoint(self.policy_model, best_dir)
            self.config.to_json(Path(best_dir) / "grpo_config.json")
            # Write epoch marker
            with open(Path(best_dir) / "best_epoch.json", "w") as f:
                json.dump({"epoch": epoch, "det_reward_mean": self.best_det_reward}, f, indent=2)
            print(f"  Saved best checkpoint (epoch {epoch}) -> {best_dir}")
        else:
            best_path = self.run_dir / "best.pth"
            latest_path = self.run_dir / "latest.pth"
            if latest_path.exists():
                shutil.copy2(latest_path, best_path)
            with open(self.run_dir / "best_epoch.json", "w") as f:
                json.dump({"epoch": epoch, "det_reward_mean": self.best_det_reward}, f, indent=2)
            print(f"  Saved best checkpoint (epoch {epoch}) -> {best_path}")

    def _write_run_summary(self, epoch: int, eval_metrics: dict, is_best: bool) -> None:
        """Write a machine-readable JSON summary of the current run state.

        Designed for agentic consumption: an auto-research agent can read this
        file to decide whether to continue training, adjust hyperparameters,
        or stop early.
        """
        # Gather training metrics from the latest epoch
        train_metrics = self.train_log[-1] if self.train_log else {}

        summary = {
            "run_dir": str(self.run_dir),
            "current_epoch": epoch,
            "total_epochs": self.config.train_epochs,
            "best_epoch": self.best_epoch,
            "best_det_reward": self.best_det_reward,
            "is_best_this_epoch": is_best,
            "config": {
                "num_generations": self.config.num_generations,
                "inner_epochs": self.config.inner_epochs,
                "kl_coef": self.config.kl_coef,
                "learning_rate": self.config.learning_rate,
                "grad_accum_groups": self.config.grad_accum_groups,
                "use_lora": self.config.use_lora,
                "lora_rank": self.config.lora_rank,
            },
            "eval": eval_metrics,
            "train": train_metrics,
            "eval_history": [
                {
                    "epoch": e["epoch"],
                    "det_reward_mean": e["det_reward_mean"],
                    "det_collision_rate": e["det_collision_rate"],
                    "det_offroad_mean": e["det_offroad_mean"],
                    "group_reward_mean": e["group_reward_mean"],
                }
                for e in self.eval_log
            ],
        }

        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def save_checkpoint(self, epoch: int, args_dict: dict) -> None:
        """Save model checkpoint (LoRA adapters or full state dict)."""
        if self.use_lora:
            from preference_optimization.lora_utils import save_lora_checkpoint

            lora_dir = str(self.run_dir / f"lora_epoch_{epoch:03d}")
            save_lora_checkpoint(self.policy_model, lora_dir)
            torch.save(
                {"epoch": epoch, "optimizer": self.optimizer.state_dict()},
                Path(lora_dir) / "optimizer.pth",
            )
            # Save the config alongside the checkpoint
            self.config.to_json(Path(lora_dir) / "grpo_config.json")
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
                "optimizer": self.optimizer.state_dict(),
                "args": args_dict,
            }
            latest_path = self.run_dir / "latest.pth"
            torch.save(checkpoint_data, latest_path)
            self.config.to_json(self.run_dir / "grpo_config.json")

            if epoch % 5 == 0:
                epoch_path = self.run_dir / f"epoch_{epoch:03d}.pth"
                torch.save(checkpoint_data, epoch_path)
                print(f"  Saved checkpoint: {epoch_path}")

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log training metrics to TSV file."""
        log_entry = {"epoch": epoch, "kl_coef": self.config.kl_coef, **{f"train_{k}": v for k, v in metrics.items()}}
        self.train_log.append(log_entry)

        df = pd.DataFrame(self.train_log)
        log_path = self.run_dir / "grpo_train_log.tsv"
        df.to_csv(log_path, sep="\t", index=False)

        M = self.config.inner_epochs
        mode_str = f"M={M}" + (" PPO-clip" if M > 1 else " on-policy")
        clip_str = f"  ClipFrac={metrics.get('clip_fraction', 0):.3f}" if M > 1 else ""

        def _fmt(v: float) -> str:
            """Adaptive formatting: use scientific notation for very small values."""
            if v == 0.0:
                return "0"
            if abs(v) < 0.001:
                return f"{v:.2e}"
            return f"{v:.6f}"

        loss = _fmt(metrics.get("loss", 0))
        ploss = _fmt(metrics.get("policy_loss", 0))
        kl = _fmt(metrics.get("kl_loss", 0))
        logp = _fmt(metrics.get("mean_policy_logprob", 0))
        print(
            f"  Epoch {epoch} [{mode_str}]: "
            f"Loss={loss}, PolicyLoss={ploss}, KL={kl}, "
            f"MeanLogProb={logp}{clip_str}"
        )

    def save_epoch1_baselines(self, npz_paths: list[str]) -> None:
        """Save deterministic trajectories as epoch-1 reference for drift tracking.

        Saves ALL scenes — downstream consumers include
        `underprogress_reference="baseline"` path-length lookup
        (rlvr/grpo_sft_trainer.py), which needs every training scene to have
        an anchor; subsampling causes unfixed scenes to fall back to the "det"
        reference and mask path-collapse behavior.
        """
        baseline_path = self.run_dir / "epoch1_baselines.npz"
        if baseline_path.exists():
            return

        self.policy_model.eval()
        paths_list: list[str] = []
        trajs_list: list[np.ndarray] = []

        for npz_path in npz_paths:
            try:
                obs = load_npz_data(npz_path, self.device)
                traj = generate_deterministic_trajectory(
                    self.policy_model, self.model_args, obs, self.device,
                )
                paths_list.append(str(npz_path))
                trajs_list.append(traj)
            except Exception as e:
                print(f"  [baseline] skipping {npz_path}: {e}")

        if not paths_list:
            return

        np.savez(
            baseline_path,
            paths=np.array(paths_list),
            trajectories=np.stack(trajs_list),
        )
        print(f"  Saved epoch-1 baselines for {len(paths_list)} samples")

    def compute_trajectory_drift(self) -> str:
        """Compute ADE between current model and epoch-1 baselines."""
        baseline_path = self.run_dir / "epoch1_baselines.npz"
        if not baseline_path.exists():
            return ""

        saved = np.load(baseline_path, allow_pickle=True)
        paths_list = saved["paths"].tolist()
        baselines = saved["trajectories"]

        self.policy_model.eval()
        ades: list[float] = []
        for npz_path, baseline_traj in zip(paths_list, baselines):
            try:
                obs = load_npz_data(npz_path, self.device)
                current_traj = generate_deterministic_trajectory(
                    self.policy_model, self.model_args, obs, self.device,
                )
                ades.append(calculate_ade(current_traj, baseline_traj))
            except Exception:
                pass

        if not ades:
            return "Drift vs epoch 1: N/A"

        mean_ade = float(np.mean(ades))
        std_ade = float(np.std(ades))
        max_ade = float(np.max(ades))
        msg = (
            f"Drift vs epoch 1: mean={mean_ade:.3f}m  "
            f"std={std_ade:.3f}m  max={max_ade:.3f}m  (n={len(ades)})"
        )
        print(f"  {msg}")
        return msg


def _empty_metrics() -> dict[str, float]:
    return {
        "loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0,
        "mean_advantage": 0.0, "advantage_std": 0.0,
        "clip_fraction": 0.0, "approx_kl_behavior": 0.0,
    }
