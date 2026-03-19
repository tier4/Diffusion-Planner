"""Automated GRPO research: runs experiments, evaluates, reports.

Searches for training configurations that eliminate off-road behavior on
problematic scenes while preserving validation performance.

Usage:
    source .venv/bin/activate
    python rlvr/auto_research.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from guidance_gui.generate_samples import generate_samples
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_trainer import GRPOTrainer
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JST = timezone(timedelta(hours=9))

# --- Data paths ---
SSD = Path("/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207")
BASE_MODEL = SSD / "xx1-best-model/v3.0/best_model.pth"
POOL_71K = SSD / "xx1_grpo_cleansed_data/path_list.json"
PROB_SCENES = SSD / "path_lists/merged_20260216_20260224/path_list.json"
VALID_SCENES = SSD / "xx1_validation_data/xx1_real_valid/path_list.json"
OUTPUT_DIR = SSD / "auto_research"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str

    # GRPO config overrides (merged on top of defaults)
    grpo_overrides: dict = field(default_factory=dict)

    # Training set composition
    n_prob_scenes: int = 50
    n_normal_scenes: int = 50
    prob_scene_seed: int = 42
    normal_scene_seed: int = 123

    # Training budget
    max_epochs: int = 10
    max_minutes: int = 30


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    name: str
    config: dict
    status: str = "pending"  # pending, running, completed, failed

    # Problematic scene eval (all 100, deterministic)
    prob_det_reward_mean: float = 0.0
    prob_det_offroad_mean: float = 0.0
    prob_det_collision_rate: float = 0.0

    # Validation eval (50 scenes, deterministic)
    val_det_reward_mean: float = 0.0
    val_det_offroad_mean: float = 0.0

    # Per-epoch history
    epoch_history: list[dict] = field(default_factory=list)

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_minutes: float = 0.0

    # Paths
    run_dir: str = ""
    best_checkpoint: str = ""
    error: str = ""


class AutoResearcher:
    """Autonomous GRPO experiment runner."""

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.master_log_path = self.output_dir / "master_log.json"
        self.results: list[ExperimentResult] = []

        # Load scene lists
        with open(POOL_71K) as f:
            self.pool_71k = json.load(f)
        with open(PROB_SCENES) as f:
            all_prob = json.load(f)
        self.prob_100 = all_prob[:100]
        with open(VALID_SCENES) as f:
            self.valid_all = json.load(f)

        # Fixed eval subsets
        rng = np.random.default_rng(42)
        val_idx = rng.choice(len(self.valid_all), size=50, replace=False)
        self.val_50 = [self.valid_all[i] for i in val_idx]

        self._save_master_log()
        print(f"AutoResearcher initialized:")
        print(f"  Output: {self.output_dir}")
        print(f"  Prob scenes: {len(self.prob_100)}")
        print(f"  Val scenes: {len(self.val_50)}")
        print(f"  Pool: {len(self.pool_71k)}")

    def _save_master_log(self):
        data = [asdict(r) for r in self.results]
        with open(self.master_log_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_training_set(
        self,
        n_prob: int,
        n_normal: int,
        prob_seed: int = 42,
        normal_seed: int = 123,
    ) -> list[str]:
        """Create a training set mixing problematic + normal scenes."""
        rng_prob = np.random.default_rng(prob_seed)
        rng_norm = np.random.default_rng(normal_seed)

        prob_idx = rng_prob.choice(len(self.prob_100), size=min(n_prob, len(self.prob_100)), replace=False)
        prob_scenes = [self.prob_100[i] for i in prob_idx]

        norm_idx = rng_norm.choice(len(self.pool_71k), size=min(n_normal, len(self.pool_71k)), replace=False)
        normal_scenes = [self.pool_71k[i] for i in norm_idx]

        combined = prob_scenes + normal_scenes
        random.Random(42).shuffle(combined)
        return combined

    def _build_grpo_config(self, overrides: dict) -> GRPOConfig:
        """Build GRPOConfig from defaults + overrides."""
        config = GRPOConfig()
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    @torch.no_grad()
    def evaluate_checkpoint(
        self,
        model,
        model_args,
        scene_paths: list[str],
        reward_config: RewardConfig,
        label: str = "",
    ) -> dict:
        """Evaluate deterministic trajectory on a set of scenes.

        Returns dict with reward_mean, reward_std, offroad_mean, collision_rate,
        and per-component breakdowns.
        """
        model.eval()

        totals = []
        offroads = []
        collisions = 0
        components = {k: [] for k in ["safety", "progress", "smoothness", "feasibility", "centerline"]}

        for path in scene_paths:
            try:
                data = load_npz_data(path, DEVICE)
                norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                norm_data = model_args.observation_normalizer(norm_data)

                det_traj = generate_samples(
                    model, model_args, norm_data,
                    noise_scale=0.0, n_samples=1, composer=None, device=DEVICE,
                )
                det_traj_t = torch.tensor(det_traj, device=DEVICE, dtype=torch.float32)
                reward = compute_reward_batch(det_traj_t, data, reward_config)[0]

                totals.append(reward.total)
                offroads.append(reward.off_road_fraction)
                if reward.collision_step is not None:
                    collisions += 1
                components["safety"].append(reward.safety)
                components["progress"].append(reward.progress)
                components["smoothness"].append(reward.smoothness)
                components["feasibility"].append(reward.feasibility)
                components["centerline"].append(reward.centerline)
            except Exception as e:
                print(f"  [eval] skipping {Path(path).name}: {e}")

        n = len(totals)
        if n == 0:
            return {"n_scenes": 0, "reward_mean": 0, "offroad_mean": 0, "collision_rate": 0}

        result = {
            "n_scenes": n,
            "reward_mean": float(np.mean(totals)),
            "reward_std": float(np.std(totals)),
            "reward_median": float(np.median(totals)),
            "offroad_mean": float(np.mean(offroads)),
            "offroad_pct_nonzero": float(np.mean([o > 0 for o in offroads])),
            "collision_rate": collisions / n,
        }
        for k, v in components.items():
            result[f"{k}_mean"] = float(np.mean(v))

        tag = f" [{label}]" if label else ""
        print(f"  Eval{tag}: {n} scenes, reward={result['reward_mean']:+.2f}, "
              f"offroad={result['offroad_mean']:.1%}, collision={result['collision_rate']:.1%}")
        return result

    def run_experiment(self, exp_config: ExperimentConfig) -> ExperimentResult:
        """Run a single training experiment with evaluation."""
        result = ExperimentResult(
            name=exp_config.name,
            config=asdict(exp_config),
            status="running",
            start_time=datetime.now(JST).isoformat(),
        )
        self.results.append(result)
        self._save_master_log()

        try:
            print(f"\n{'='*70}")
            print(f"EXPERIMENT: {exp_config.name}")
            print(f"  {exp_config.description}")
            print(f"{'='*70}")

            # Build config
            grpo_config = self._build_grpo_config(exp_config.grpo_overrides)
            grpo_config.train_epochs = exp_config.max_epochs

            # Create training set
            train_paths = self.create_training_set(
                n_prob=exp_config.n_prob_scenes,
                n_normal=exp_config.n_normal_scenes,
                prob_seed=exp_config.prob_scene_seed,
                normal_seed=exp_config.normal_scene_seed,
            )
            print(f"  Training set: {len(train_paths)} scenes "
                  f"({exp_config.n_prob_scenes} prob + {exp_config.n_normal_scenes} normal)")

            # Setup experiment directory
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = self.output_dir / f"{timestamp}_{exp_config.name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result.run_dir = str(run_dir)

            # Copy base model
            checkpoint_path = run_dir / "latest.pth"
            shutil.copy2(BASE_MODEL, checkpoint_path)
            args_json = BASE_MODEL.parent / "args.json"
            shutil.copy2(args_json, run_dir / "args.json")
            grpo_config.to_json(run_dir / "grpo_config.json")

            # Save training set
            train_list_path = run_dir / "train_scenes.json"
            with open(train_list_path, "w") as f:
                json.dump(train_paths, f)

            # Load model
            policy_model, model_args = load_model(checkpoint_path, DEVICE)

            # Apply LoRA
            if grpo_config.use_lora:
                from preference_optimization.lora_utils import apply_lora
                policy_model = apply_lora(
                    policy_model,
                    r=grpo_config.lora_rank,
                    lora_alpha=grpo_config.lora_alpha,
                    lora_dropout=grpo_config.lora_dropout,
                )

            # Optimizer
            trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=grpo_config.learning_rate)

            # Reward config
            reward_config = RewardConfig(
                w_safety=grpo_config.w_safety,
                w_progress=grpo_config.w_progress,
                w_smooth=grpo_config.w_smooth,
                w_feasibility=grpo_config.w_feasibility,
                w_centerline=grpo_config.w_centerline,
            )

            # Create trainer
            trainer = GRPOTrainer(
                policy_model=policy_model,
                model_args=model_args,
                optimizer=optimizer,
                device=DEVICE,
                run_dir=run_dir,
                config=grpo_config,
                use_lora=grpo_config.use_lora,
            )

            # Eval base model first
            print("\n  Evaluating base model...")
            base_prob = self.evaluate_checkpoint(
                policy_model, model_args, self.prob_100, reward_config, "base-prob"
            )
            base_val = self.evaluate_checkpoint(
                policy_model, model_args, self.val_50, reward_config, "base-val"
            )

            # Set up eval scenes (use our fixed val_50)
            trainer._eval_scene_paths = self.val_50

            # Training loop with time budget
            start_time = time.time()
            deadline = start_time + exp_config.max_minutes * 60

            args_dict = {"exp_name": exp_config.name}
            best_prob_reward = float("-inf")

            for epoch in range(1, exp_config.max_epochs + 1):
                if time.time() > deadline:
                    print(f"  Time budget exhausted at epoch {epoch}")
                    break

                print(f"\n  --- Epoch {epoch}/{exp_config.max_epochs} ---")

                if epoch == 1:
                    trainer.save_epoch1_baselines(train_paths)

                metrics = trainer.train_epoch(train_paths, epoch)
                trainer.log_metrics(epoch, metrics)
                trainer.save_checkpoint(epoch, args_dict)

                # Evaluate on both scene sets
                prob_eval = self.evaluate_checkpoint(
                    policy_model, model_args, self.prob_100, reward_config, f"epoch{epoch}-prob"
                )
                val_eval = self.evaluate_checkpoint(
                    policy_model, model_args, self.val_50, reward_config, f"epoch{epoch}-val"
                )

                epoch_result = {
                    "epoch": epoch,
                    "train_metrics": metrics,
                    "prob_eval": prob_eval,
                    "val_eval": val_eval,
                }
                result.epoch_history.append(epoch_result)

                # Track best
                if prob_eval["reward_mean"] > best_prob_reward:
                    best_prob_reward = prob_eval["reward_mean"]
                    result.prob_det_reward_mean = prob_eval["reward_mean"]
                    result.prob_det_offroad_mean = prob_eval["offroad_mean"]
                    result.prob_det_collision_rate = prob_eval["collision_rate"]
                    result.val_det_reward_mean = val_eval["reward_mean"]
                    result.val_det_offroad_mean = val_eval["offroad_mean"]
                    if grpo_config.use_lora:
                        result.best_checkpoint = str(run_dir / f"lora_epoch_{epoch:03d}")
                    else:
                        # For full fine-tuning, save a copy as best
                        best_path = run_dir / "best.pth"
                        latest = run_dir / "latest.pth"
                        if latest.exists():
                            shutil.copy2(latest, best_path)
                        result.best_checkpoint = str(best_path)

                self._save_master_log()

                # Early stopping: if prob reward hasn't improved in 3 epochs
                if epoch >= 4:
                    recent = [h["prob_eval"]["reward_mean"] for h in result.epoch_history[-3:]]
                    if all(r <= result.epoch_history[-4]["prob_eval"]["reward_mean"] for r in recent):
                        print(f"  Early stopping: no prob improvement in 3 epochs")
                        break

            result.status = "completed"
            result.end_time = datetime.now(JST).isoformat()
            result.duration_minutes = (time.time() - start_time) / 60

            print(f"\n  Experiment {exp_config.name} completed in {result.duration_minutes:.1f} min")
            print(f"  Best prob reward: {result.prob_det_reward_mean:+.2f}, "
                  f"offroad: {result.prob_det_offroad_mean:.1%}")
            print(f"  Val reward: {result.val_det_reward_mean:+.2f}")

            # Cleanup: remove per-epoch checkpoints except best and latest
            self._cleanup_checkpoints(run_dir, result.best_checkpoint)

            # Free GPU memory
            del policy_model, trainer, optimizer
            torch.cuda.empty_cache()

        except Exception as e:
            result.status = "failed"
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            result.end_time = datetime.now(JST).isoformat()
            print(f"  FAILED: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

        self._save_master_log()
        return result

    def _cleanup_checkpoints(self, run_dir: Path, best_checkpoint: str):
        """Remove per-epoch checkpoints except best to save disk space."""
        best_dir = Path(best_checkpoint).name if best_checkpoint else ""
        for d in run_dir.glob("lora_epoch_*"):
            if d.name != best_dir and d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        # Remove lora_latest symlink
        latest_link = run_dir / "lora_latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        # For full fine-tuning, remove epoch_XXX.pth files
        for f in run_dir.glob("epoch_*.pth"):
            if str(f) != best_checkpoint:
                f.unlink(missing_ok=True)

    def get_top_k(self, k: int = 3) -> list[ExperimentResult]:
        """Return top k experiments by prob det_reward_mean."""
        completed = [r for r in self.results if r.status == "completed"]
        completed.sort(key=lambda r: r.prob_det_reward_mean, reverse=True)
        return completed[:k]

    def run_final_comparison(self, top_k: list[ExperimentResult], n_val_scenes: int = 500):
        """Extended evaluation of top candidates on larger validation set."""
        print(f"\n{'='*70}")
        print(f"FINAL COMPARISON: Top {len(top_k)} experiments on {n_val_scenes} validation scenes")
        print(f"{'='*70}")

        rng = np.random.default_rng(99)
        val_idx = rng.choice(len(self.valid_all), size=min(n_val_scenes, len(self.valid_all)), replace=False)
        val_extended = [self.valid_all[i] for i in val_idx]

        for result in top_k:
            if not result.best_checkpoint or not Path(result.best_checkpoint).exists():
                print(f"  Skipping {result.name}: checkpoint not found at {result.best_checkpoint}")
                continue

            print(f"\n  --- {result.name} ---")

            # Load model + checkpoint
            grpo_config = GRPOConfig.from_json(Path(result.run_dir) / "grpo_config.json")

            if grpo_config.use_lora:
                policy_model, model_args = load_model(
                    Path(result.run_dir) / "latest.pth", DEVICE
                )
                from preference_optimization.lora_utils import load_lora_checkpoint
                policy_model = load_lora_checkpoint(
                    policy_model, result.best_checkpoint, is_trainable=False,
                )
            else:
                # Full fine-tuning: load the best.pth directly
                policy_model, model_args = load_model(
                    result.best_checkpoint, DEVICE
                )

            reward_config = RewardConfig(
                w_safety=grpo_config.w_safety,
                w_progress=grpo_config.w_progress,
                w_smooth=grpo_config.w_smooth,
                w_feasibility=grpo_config.w_feasibility,
                w_centerline=grpo_config.w_centerline,
            )

            # Full prob eval
            prob_full = self.evaluate_checkpoint(
                policy_model, model_args, self.prob_100, reward_config, f"{result.name}-prob-full"
            )
            # Extended val eval
            val_ext = self.evaluate_checkpoint(
                policy_model, model_args, val_extended, reward_config, f"{result.name}-val-{n_val_scenes}"
            )

            result.config["final_prob_eval"] = prob_full
            result.config["final_val_eval"] = val_ext

            del policy_model
            torch.cuda.empty_cache()

        self._save_master_log()

    def generate_report(self) -> str:
        """Generate markdown report of all experiments."""
        lines = [
            "# Auto-Research Report: GRPO Off-Road Improvement",
            f"\nGenerated: {datetime.now(JST).isoformat()}",
            f"\nTotal experiments: {len(self.results)}",
            f"Completed: {sum(1 for r in self.results if r.status == 'completed')}",
            f"Failed: {sum(1 for r in self.results if r.status == 'failed')}",
            "",
        ]

        # Baseline
        lines.append("## Base Model Performance")
        for r in self.results:
            if r.epoch_history and "prob_eval" in r.epoch_history[0]:
                # First experiment has base eval in epoch_history
                break

        # Results table
        lines.append("\n## Results Summary")
        lines.append("| Experiment | Loss Mode | Prob Reward | Prob Offroad | Val Reward | Duration |")
        lines.append("|---|---|---|---|---|---|")

        for r in self.results:
            if r.status != "completed":
                lines.append(f"| {r.name} | — | FAILED | — | — | — |")
                continue
            loss_mode = r.config.get("grpo_overrides", {}).get("loss_mode", "diffusion")
            lines.append(
                f"| {r.name} | {loss_mode} | {r.prob_det_reward_mean:+.2f} | "
                f"{r.prob_det_offroad_mean:.1%} | {r.val_det_reward_mean:+.2f} | "
                f"{r.duration_minutes:.0f}m |"
            )

        # Top 3
        top3 = self.get_top_k(3)
        if top3:
            lines.append("\n## Top 3 Experiments")
            for i, r in enumerate(top3, 1):
                lines.append(f"\n### #{i}: {r.name}")
                lines.append(f"- Prob reward: {r.prob_det_reward_mean:+.2f}")
                lines.append(f"- Prob offroad: {r.prob_det_offroad_mean:.1%}")
                lines.append(f"- Val reward: {r.val_det_reward_mean:+.2f}")
                lines.append(f"- Checkpoint: {r.best_checkpoint}")
                lines.append(f"- Config: `{json.dumps(r.config.get('grpo_overrides', {}), indent=2)}`")

                if r.config.get("final_prob_eval"):
                    fp = r.config["final_prob_eval"]
                    fv = r.config.get("final_val_eval", {})
                    lines.append(f"- Final prob eval: reward={fp.get('reward_mean', 0):+.2f}, "
                                 f"offroad={fp.get('offroad_mean', 0):.1%}")
                    lines.append(f"- Final val eval ({fv.get('n_scenes', 0)} scenes): "
                                 f"reward={fv.get('reward_mean', 0):+.2f}")

        # Per-experiment details
        lines.append("\n## Experiment Details")
        for r in self.results:
            lines.append(f"\n### {r.name}")
            lines.append(f"Status: {r.status}")
            if r.error:
                lines.append(f"Error: {r.error[:200]}")
            if r.epoch_history:
                lines.append("\n| Epoch | Prob Reward | Prob Offroad | Val Reward |")
                lines.append("|---|---|---|---|")
                for h in r.epoch_history:
                    pe = h.get("prob_eval", {})
                    ve = h.get("val_eval", {})
                    lines.append(
                        f"| {h['epoch']} | {pe.get('reward_mean', 0):+.2f} | "
                        f"{pe.get('offroad_mean', 0):.1%} | {ve.get('reward_mean', 0):+.2f} |"
                    )

        report = "\n".join(lines)
        report_path = self.output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
        return report

    def run(self, deadline: datetime):
        """Main research loop: run experiments until deadline."""
        print(f"\n{'#'*70}")
        print(f"AUTO-RESEARCH STARTING")
        print(f"Deadline: {deadline.isoformat()}")
        print(f"{'#'*70}")

        # =====================================================================
        # Phase 1: Baseline eval + quick probe of loss modes
        # =====================================================================

        experiments_queue: list[ExperimentConfig] = []

        # Finding: GRPO at lr=1e-5 produces LoRA_B weights of ~0.0001 magnitude,
        # which moves the deterministic output by only ~0.005m. Need weights at
        # ~0.01-0.1 range to meaningfully change trajectories. Prioritize high LR.

        # Experiment 1: GRPO with high LR (the critical test)
        experiments_queue.append(ExperimentConfig(
            name="grpo_lr1e3",
            description="Standard GRPO with lr=1e-3 (100x baseline)",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.1,
                "learning_rate": 1e-3,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=50,
            max_epochs=10,
            max_minutes=30,
        ))

        # Experiment 2: GRPO lr=1e-4 (10x baseline)
        experiments_queue.append(ExperimentConfig(
            name="grpo_lr1e4",
            description="Standard GRPO with lr=1e-4 (10x baseline)",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.1,
                "learning_rate": 1e-4,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=50,
            max_epochs=10,
            max_minutes=30,
        ))

        # Experiment 3: GRPO lr=1e-3, no KL, 100% prob scenes
        experiments_queue.append(ExperimentConfig(
            name="grpo_lr1e3_prob_only",
            description="GRPO lr=1e-3, no KL, only problematic scenes",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.0,
                "learning_rate": 1e-3,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=0,
            max_epochs=10,
            max_minutes=30,
        ))

        # Experiment 4: direct_best with high LR
        experiments_queue.append(ExperimentConfig(
            name="direct_best_lr1e3",
            description="BC toward best traj at low-t, lr=1e-3",
            grpo_overrides={
                "loss_mode": "direct_best",
                "direct_loss_weight": 1.0,
                "diffusion_t_range": [0.001, 0.1],
                "diffusion_k_steps": 4,
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.0,
                "learning_rate": 1e-3,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=0,
            max_epochs=10,
            max_minutes=30,
        ))

        # Experiment 5: diffusion_low_t with high LR
        experiments_queue.append(ExperimentConfig(
            name="low_t_lr1e3",
            description="GRPO with t near 0 and lr=1e-3",
            grpo_overrides={
                "loss_mode": "diffusion_low_t",
                "diffusion_t_range": [0.001, 0.1],
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.1,
                "learning_rate": 1e-3,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=50,
            max_epochs=10,
            max_minutes=30,
        ))

        # Experiment 6: Heavy reward weights + high LR
        experiments_queue.append(ExperimentConfig(
            name="grpo_lr1e3_heavy_reward",
            description="GRPO lr=1e-3 with 2x feasibility+centerline weights",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.1,
                "learning_rate": 1e-3,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
                "w_feasibility": 10.0,
                "w_centerline": 10.0,
                "w_progress": 1.0,
            },
            n_prob_scenes=50,
            n_normal_scenes=50,
            max_epochs=10,
            max_minutes=30,
        ))

        # Experiment 7: Full fine-tuning (no LoRA), high LR
        experiments_queue.append(ExperimentConfig(
            name="full_ft_lr1e4",
            description="Full fine-tuning (no LoRA) with lr=1e-4",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.0,
                "learning_rate": 1e-4,
                "use_lora": False,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=50,
            max_epochs=10,
            max_minutes=30,
        ))

        # =====================================================================
        # Phase 1: Run initial experiments
        # =====================================================================
        for exp_config in experiments_queue:
            if datetime.now(JST) > deadline - timedelta(hours=2):
                print(f"\nApproaching deadline, stopping new experiments for final comparison")
                break
            self.run_experiment(exp_config)

        # =====================================================================
        # Phase 2: Adaptive follow-up based on results
        # =====================================================================
        self._run_adaptive_phase(deadline)

        # =====================================================================
        # Phase 3: Final comparison
        # =====================================================================
        top3 = self.get_top_k(3)
        if top3:
            self.run_final_comparison(top3, n_val_scenes=500)

        # Generate report
        self.generate_report()

        print(f"\n{'#'*70}")
        print(f"AUTO-RESEARCH COMPLETE")
        print(f"Results: {self.master_log_path}")
        print(f"Report: {self.output_dir / 'report.md'}")
        print(f"{'#'*70}")

    def _run_adaptive_phase(self, deadline: datetime):
        """Generate follow-up experiments based on Phase 1 results."""
        completed = [r for r in self.results if r.status == "completed"]
        if not completed:
            print("No completed experiments to analyze for adaptive phase.")
            return

        # Find best so far
        best = max(completed, key=lambda r: r.prob_det_reward_mean)
        print(f"\nAdaptive phase: best so far is '{best.name}' "
              f"(prob reward={best.prob_det_reward_mean:+.2f}, offroad={best.prob_det_offroad_mean:.1%})")

        follow_ups: list[ExperimentConfig] = []

        # Based on findings: lr=1e-3 works but is unstable.
        # Key experiments: moderate LR (5e-4), more normal scenes, stronger KL.

        # Experiment A: lr=5e-4 (sweet spot hypothesis)
        follow_ups.append(ExperimentConfig(
            name="grpo_lr5e4_balanced",
            description="GRPO lr=5e-4, kl=0.2, 50 prob + 150 normal (balanced)",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 15,
                "kl_coef": 0.2,
                "learning_rate": 5e-4,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=150,
            max_epochs=15,
            max_minutes=35,
        ))

        # Experiment B: lr=1e-3 with strong KL (0.5) + more normal scenes
        follow_ups.append(ExperimentConfig(
            name="grpo_lr1e3_hi_kl",
            description="GRPO lr=1e-3, kl=0.5, 50 prob + 200 normal (stabilized)",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 10,
                "kl_coef": 0.5,
                "learning_rate": 1e-3,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=200,
            max_epochs=10,
            max_minutes=35,
        ))

        # Experiment C: lr=5e-4 with heavy feasibility weights
        follow_ups.append(ExperimentConfig(
            name="grpo_lr5e4_heavy_feas",
            description="GRPO lr=5e-4, heavy feasibility/centerline, 50p+150n",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 15,
                "kl_coef": 0.2,
                "learning_rate": 5e-4,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
                "w_feasibility": 10.0,
                "w_centerline": 10.0,
                "w_progress": 1.0,
            },
            n_prob_scenes=50,
            n_normal_scenes=150,
            max_epochs=15,
            max_minutes=35,
        ))

        # Experiment D: lr=3e-4 (even more conservative)
        follow_ups.append(ExperimentConfig(
            name="grpo_lr3e4_long",
            description="GRPO lr=3e-4, kl=0.2, 50p+100n, 15 epochs",
            grpo_overrides={
                "loss_mode": "diffusion",
                "num_generations": 16,
                "train_epochs": 15,
                "kl_coef": 0.2,
                "learning_rate": 3e-4,
                "lora_rank": 64,
                "lora_alpha": 64,
                "guidance_prob": 0.7,
                "enable_route_following": True,
                "enable_lane_keeping": True,
            },
            n_prob_scenes=50,
            n_normal_scenes=100,
            max_epochs=15,
            max_minutes=35,
        ))

        # Run follow-ups until deadline
        for exp_config in follow_ups:
            if datetime.now(JST) > deadline - timedelta(hours=1, minutes=30):
                print(f"\nApproaching deadline, stopping for final comparison")
                break
            # Skip if name already exists
            if any(r.name == exp_config.name for r in self.results):
                exp_config.name = exp_config.name + "_v2"
            self.run_experiment(exp_config)


def main():
    deadline = datetime(2026, 3, 20, 22, 0, 0, tzinfo=JST)
    researcher = AutoResearcher()
    researcher.run(deadline)


if __name__ == "__main__":
    main()
