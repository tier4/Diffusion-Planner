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
    load_npz_data,
)

from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_grpo_loss, compute_log_probs
from rlvr.grpo_sampler import SamplerConfig, generate_diverse_group
from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_group_advantages,
    compute_reward_batch,
)


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
            guidance_prob=config.guidance_prob,
            prototypes_path=config.prototypes_path,
        )
        self.reward_config = RewardConfig(
            w_safety=config.w_safety,
            w_progress=config.w_progress,
            w_smooth=config.w_smooth,
            w_feasibility=config.w_feasibility,
            w_centerline=config.w_centerline,
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
            guidance_prob=config.guidance_prob,
            prototypes_path=config.prototypes_path,
        )
        self._eval_scene_paths: list[str] | None = None
        self.eval_log: list[dict] = []

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

        self.policy_model.eval()
        with torch.no_grad():
            sampled = generate_diverse_group(
                model=self.policy_model,
                model_args=self.model_args,
                data=data,
                config=self.sampler_config,
                device=self.device,
            )

        trajectories = [st.trajectory for st in sampled]

        # Score with rewards
        traj_batch = torch.tensor(
            np.stack(trajectories), device=self.device, dtype=torch.float32,
        )
        reward_breakdowns = compute_reward_batch(
            traj_batch, data, self.reward_config,
        )
        advantages = compute_group_advantages(reward_breakdowns)

        # Store old log-probs and the (noise, t) used to compute them.
        # Reusing the same (noise, t) during training ensures a consistent
        # importance sampling ratio.
        old_log_probs, old_noise, old_t = compute_log_probs(
            self.policy_model, trajectories, data, self.model_args, self.device,
        )

        return {
            "npz_path": npz_path,
            "data": data,
            "trajectories": trajectories,
            "reward_breakdowns": reward_breakdowns,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
            "old_noise": old_noise,
            "old_t": old_t,
        }

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

                # For M=1, old_log_probs is ignored inside compute_grpo_loss
                old_lp = group.get("old_log_probs") if M > 1 else None
                old_noise = group.get("old_noise") if M > 1 else None
                old_t = group.get("old_t") if M > 1 else None

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
        print(f"  Generating trajectory groups for {len(npz_paths)} scenes (N={self.config.num_generations})...")
        groups = []
        for npz_path in tqdm(npz_paths, desc="Generating groups"):
            group = self.generate_and_score_group(npz_path)
            if group is not None:
                groups.append(group)

        print(f"  Generated {len(groups)} valid groups")
        if not groups:
            return _empty_metrics()

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
        """Evaluate reward distribution on fixed validation scenes.

        Generates 8 trajectories per scene, scores them, and returns
        summary statistics. Results are appended to eval_log and saved to TSV.

        Uses a fixed random seed so that trajectory generation is deterministic
        across epochs and across different training runs, enabling fair
        comparison of different hyperparameter configurations.
        """
        if not self._eval_scene_paths:
            return {}

        self.policy_model.eval()

        # Fix all random seeds for reproducible evaluation across runs.
        # The model weights are the only variable between evaluations.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        all_totals = []
        all_collisions = 0
        all_offroad = []
        scene_spreads = []
        components = {k: [] for k in ["safety", "progress", "smoothness", "feasibility", "centerline"]}

        for path in self._eval_scene_paths:
            try:
                data = load_npz_data(path, self.device)
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
                for r in rewards:
                    components["safety"].append(r.safety)
                    components["progress"].append(r.progress)
                    components["smoothness"].append(r.smoothness)
                    components["feasibility"].append(r.feasibility)
                    components["centerline"].append(r.centerline)
            except Exception as e:
                print(f"  [eval] skipping {path}: {e}")

        if not all_totals:
            return {}

        n_trajs = len(all_totals)
        totals_arr = np.array(all_totals)
        offroad_arr = np.array(all_offroad)
        spreads_arr = np.array(scene_spreads)
        cfg = self.reward_config

        eval_metrics = {
            "epoch": epoch,
            "n_scenes": len(scene_spreads),
            "n_trajs": n_trajs,
            "reward_mean": float(totals_arr.mean()),
            "reward_std": float(totals_arr.std()),
            "reward_median": float(np.median(totals_arr)),
            "reward_min": float(totals_arr.min()),
            "reward_max": float(totals_arr.max()),
            "spread_mean": float(spreads_arr.mean()),
            "collision_rate": all_collisions / n_trajs,
            "offroad_rate": float((offroad_arr > 0.1).sum() / n_trajs),
            "offroad_mean": float(offroad_arr.mean()),
            "w_safety_mean": float(np.mean(components["safety"]) * cfg.w_safety),
            "w_progress_mean": float(np.mean(components["progress"]) * cfg.w_progress),
            "w_smooth_mean": float(np.mean(components["smoothness"]) * cfg.w_smooth),
            "w_feasibility_mean": float(np.mean(components["feasibility"]) * cfg.w_feasibility),
            "w_centerline_mean": float(np.mean(components["centerline"]) * cfg.w_centerline),
        }

        self.eval_log.append(eval_metrics)
        df = pd.DataFrame(self.eval_log)
        eval_log_path = self.run_dir / "grpo_eval_log.tsv"
        df.to_csv(eval_log_path, sep="\t", index=False)

        print(
            f"  Eval (epoch {epoch}, {len(scene_spreads)} scenes): "
            f"reward={totals_arr.mean():+.1f}±{totals_arr.std():.1f}  "
            f"collision={all_collisions/n_trajs:.1%}  "
            f"offroad={offroad_arr.mean():.1%}  "
            f"spread={spreads_arr.mean():.1f}"
        )

        return eval_metrics

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
        log_entry = {"epoch": epoch, **{f"train_{k}": v for k, v in metrics.items()}}
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
        """Save deterministic trajectories as epoch-1 reference for drift tracking."""
        baseline_path = self.run_dir / "epoch1_baselines.npz"
        if baseline_path.exists():
            return

        self.policy_model.eval()
        paths_list: list[str] = []
        trajs_list: list[np.ndarray] = []

        for npz_path in npz_paths[:100]:
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
