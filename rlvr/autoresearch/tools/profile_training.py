"""Profile training epoch performance for RSFT, Exploration, and Closed-Loop trainers.

Requires CUDA. Instruments key phases with wall-clock + CUDA timing and produces
a breakdown table.

Usage:
    python -m rlvr.autoresearch.tools.profile_training \
        --model_path <base_model.pth> \
        --scenes <scenes.json> \
        --mode rsft|explorer|closed_loop \
        [--config <config.json>] \
        [--n_scenes 10] \
        [--n_epochs 1]
"""

import argparse
import contextlib
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TimerRecord:
    name: str
    calls: int = 0
    total_wall: float = 0.0
    total_cuda: float = 0.0


class Profiler:
    """Lightweight profiler that wraps functions with CUDA-synced timing."""

    def __init__(self):
        self.records: dict[str, TimerRecord] = {}
        self._stack: list[str] = []
        self._originals: list[tuple] = []  # (obj, attr, original_fn) for unpatching

    def _get_or_create(self, name: str) -> TimerRecord:
        if name not in self.records:
            self.records[name] = TimerRecord(name=name)
        return self.records[name]

    @contextlib.contextmanager
    def region(self, name: str):
        """Context manager for timing a code region."""
        rec = self._get_or_create(name)
        self._stack.append(name)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()

            wall_elapsed = time.perf_counter() - wall_start
            cuda_elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> s

            rec.calls += 1
            rec.total_wall += wall_elapsed
            rec.total_cuda += cuda_elapsed
            self._stack.pop()

    def wrap_function(self, obj, attr: str, label: str | None = None):
        """Monkey-patch obj.attr with a timed wrapper."""
        original = getattr(obj, attr)
        name = label or f"{type(obj).__name__}.{attr}"
        rec = self._get_or_create(name)

        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            wall_start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                end_event.record()
                torch.cuda.synchronize()
                rec.calls += 1
                rec.total_wall += time.perf_counter() - wall_start
                rec.total_cuda += start_event.elapsed_time(end_event) / 1000.0
            return result

        setattr(obj, attr, wrapper)
        self._originals.append((obj, attr, original))

    def wrap_module_function(self, module, func_name: str, label: str | None = None):
        """Monkey-patch a module-level function."""
        original = getattr(module, func_name)
        name = label or f"{module.__name__}.{func_name}"
        rec = self._get_or_create(name)

        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            wall_start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                end_event.record()
                torch.cuda.synchronize()
                rec.calls += 1
                rec.total_wall += time.perf_counter() - wall_start
                rec.total_cuda += start_event.elapsed_time(end_event) / 1000.0
            return result

        setattr(module, func_name, wrapper)
        self._originals.append((module, func_name, original))

    def unpatch_all(self):
        for obj, attr, original in self._originals:
            setattr(obj, attr, original)
        self._originals.clear()

    def report(self, total_epoch_wall: float) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 90)
        lines.append("TRAINING PROFILE REPORT")
        lines.append("=" * 90)
        lines.append(
            f"{'Phase':<45} {'Calls':>6} {'Wall(s)':>9} {'CUDA(s)':>9} {'Wall%':>7} {'Avg(ms)':>9}"
        )
        lines.append("-" * 90)

        sorted_recs = sorted(self.records.values(), key=lambda r: r.total_wall, reverse=True)
        for rec in sorted_recs:
            pct = (rec.total_wall / total_epoch_wall * 100) if total_epoch_wall > 0 else 0
            avg_ms = (rec.total_wall / rec.calls * 1000) if rec.calls > 0 else 0
            lines.append(
                f"  {rec.name:<43} {rec.calls:>6} {rec.total_wall:>9.2f} {rec.total_cuda:>9.2f} {pct:>6.1f}% {avg_ms:>9.1f}"
            )

        lines.append("-" * 90)
        lines.append(f"  {'TOTAL EPOCH':<43} {'':>6} {total_epoch_wall:>9.2f}")
        lines.append("  Note: nested phases overlap; Wall% is not additive.")
        lines.append("=" * 90)

        lines.append(
            f"\nGPU Memory: peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB, "
            f"reserved={torch.cuda.max_memory_reserved() / 1e9:.2f}GB"
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loading (shared)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")


def load_model(model_path: str):
    """Load the base model + LoRA setup."""
    from preference_optimization.lora_utils import apply_lora
    from preference_optimization.model_utils import load_model as _load_model

    model, model_args = _load_model(Path(model_path), DEVICE)
    model = apply_lora(model, r=16, lora_alpha=16, lora_dropout=0.05)
    model.to(DEVICE)
    return model, model_args


def load_scene_paths(scenes_json: str, n_scenes: int) -> list[str]:
    """Load scene paths from JSON, limit to n_scenes."""
    with open(scenes_json) as f:
        paths = json.load(f)
    if not paths:
        raise ValueError(f"No scenes in {scenes_json}")
    if isinstance(paths[0], dict):
        paths = [p["path"] for p in paths]
    result = paths[:n_scenes]
    if not result:
        raise ValueError(f"No scenes selected (n_scenes={n_scenes})")
    return result


# ---------------------------------------------------------------------------
# RSFT profiling
# ---------------------------------------------------------------------------


def profile_rsft(
    model,
    model_args,
    scene_paths,
    config_path: str | None,
    profiler: Profiler,
    epoch: int = 1,
    n_epochs: int = 1,
):
    """Profile Ranked SFT training epoch."""
    from rlvr.grpo_config import GRPOConfig
    from rlvr.reward import RewardConfig

    if config_path:
        config = GRPOConfig.from_json(config_path)
    else:
        config = GRPOConfig()
        config.ranked_sft_mode = "gt_neighbor"
        config.neighbor_reg_weight = 1.0
        config.neighbor_reg_only = True
        config.num_generations = 16
        config.noise_scale_range = [0.5, 2.0]
        config.diffusion_k_steps = 8
        config.grad_accum_groups = 8
        config.sft_batch_size = 1
        config.train_epochs = n_epochs

    reward_config = RewardConfig(enable_overprogress=True)
    reward_config.enable_lane_departure = True
    reward_config.stopped_penalty = 100.0

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    # Instrument key functions
    import rlvr.grpo_sft_trainer as sft_mod
    import rlvr.grpo_trainer_batched as batched_mod
    import rlvr.reward as reward_mod
    from rlvr.closed_loop import batched_rollout as br_mod

    profiler.wrap_module_function(
        batched_mod, "generate_all_scenes_batched", "generation.batched_all_scenes"
    )
    profiler.wrap_module_function(batched_mod, "_chunked_generate", "generation.chunked_generate")
    profiler.wrap_module_function(
        br_mod, "_batched_generate_varied_noise", "generation.varied_noise"
    )
    profiler.wrap_module_function(reward_mod, "compute_reward_batch", "reward.compute_batch")
    profiler.wrap_module_function(reward_mod, "compute_road_border_penalty", "reward.road_border")
    profiler.wrap_module_function(
        reward_mod, "compute_lane_departure_penalty", "reward.lane_departure"
    )
    profiler.wrap_module_function(reward_mod, "compute_safety_score_batch", "reward.safety")
    profiler.wrap_module_function(reward_mod, "compute_progress_score_batch", "reward.progress")
    profiler.wrap_module_function(
        reward_mod, "compute_feasibility_score_batch", "reward.feasibility"
    )
    profiler.wrap_module_function(reward_mod, "compute_centerline_score_batch", "reward.centerline")
    profiler.wrap_module_function(
        sft_mod, "_compute_sft_diffusion_loss", "training.sft_diffusion_loss"
    )
    profiler.wrap_module_function(sft_mod, "_smooth_trajectory", "cpu.sg_filter")

    import preference_optimization.utils as po_utils

    profiler.wrap_module_function(po_utils, "load_npz_data", "io.load_npz")

    from rlvr.grpo_sft_trainer import train_epoch_ranked_sft

    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()

    metrics = train_epoch_ranked_sft(
        model=model,
        model_args=model_args,
        optimizer=optimizer,
        scene_paths=scene_paths,
        config=config,
        reward_config=reward_config,
        device=DEVICE,
        epoch=epoch,
        run_dir=None,
    )

    total_wall = time.perf_counter() - wall_start
    return total_wall, metrics


# ---------------------------------------------------------------------------
# Explorer profiling
# ---------------------------------------------------------------------------


def profile_explorer(
    model,
    model_args,
    scene_paths,
    config_path: str | None,
    profiler: Profiler,
    epoch: int = 1,
    n_epochs: int = 1,
):
    """Profile Guidance Explorer training epoch."""
    import tempfile

    from rlvr.grpo_config import GRPOConfig
    from rlvr.grpo_exploration_trainer import GRPOExplorationTrainer

    if config_path:
        config = GRPOConfig.from_json(config_path)
    else:
        config = GRPOConfig()
        config.use_exploration_policy = True
        config.num_generations = 16
        config.noise_scale_range = [0.5, 2.0]
        config.diffusion_k_steps = 8
        config.grad_accum_groups = 8
        config.train_epochs = n_epochs
        config.exploration_loss_type = "advantage_logprob"

    config.use_exploration_policy = True

    dit_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    with tempfile.TemporaryDirectory(prefix="profile_explorer_") as tmp_dir:
        run_dir = Path(tmp_dir)

        trainer = GRPOExplorationTrainer(
            policy_model=model,
            model_args=model_args,
            dit_optimizer=dit_optimizer,
            device=DEVICE,
            run_dir=run_dir,
            config=config,
        )

        # Instrument key functions
        import rlvr.reward as reward_mod
        from rlvr import grpo_loss as loss_mod
        from rlvr.closed_loop import batched_rollout as br_mod

        profiler.wrap_module_function(
            br_mod, "_batched_generate_varied_noise", "generation.varied_noise"
        )
        profiler.wrap_module_function(reward_mod, "compute_reward_batch", "reward.compute_batch")
        profiler.wrap_module_function(
            reward_mod, "compute_road_border_penalty", "reward.road_border"
        )
        profiler.wrap_module_function(
            reward_mod, "compute_lane_departure_penalty", "reward.lane_departure"
        )
        profiler.wrap_module_function(reward_mod, "compute_safety_score_batch", "reward.safety")
        profiler.wrap_module_function(reward_mod, "compute_progress_score_batch", "reward.progress")
        profiler.wrap_module_function(
            reward_mod, "compute_feasibility_score_batch", "reward.feasibility"
        )
        profiler.wrap_module_function(
            reward_mod, "compute_centerline_score_batch", "reward.centerline"
        )
        profiler.wrap_module_function(loss_mod, "compute_batched_grpo_loss", "training.grpo_loss")

        from exploration_policy import utils as ep_utils

        profiler.wrap_module_function(
            ep_utils, "generate_reference_trajectory", "generation.reference_traj"
        )
        profiler.wrap_module_function(ep_utils, "run_frozen_encoder", "generation.frozen_encoder")

        profiler.wrap_function(trainer, "generate_policy_guided_group", "phase.generate_group")
        profiler.wrap_function(trainer, "train_on_groups", "phase.train_on_groups")

        import preference_optimization.utils as po_utils

        profiler.wrap_module_function(po_utils, "load_npz_data", "io.load_npz")

        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()

        metrics = trainer.train_epoch(scene_paths, epoch=epoch)

        total_wall = time.perf_counter() - wall_start

    return total_wall, metrics


# ---------------------------------------------------------------------------
# Closed-Loop profiling
# ---------------------------------------------------------------------------


def profile_closed_loop(
    model,
    model_args,
    scene_paths,
    config_path: str | None,
    profiler: Profiler,
    epoch: int = 1,
    n_epochs: int = 1,
):
    """Profile Closed-Loop Exploration training epoch."""
    import tempfile

    from rlvr.closed_loop.closed_loop_trainer import ClosedLoopExplorationTrainer
    from rlvr.grpo_config import GRPOConfig

    if config_path:
        config = GRPOConfig.from_json(config_path)
    else:
        config = GRPOConfig()
        config.use_closed_loop = True
        config.closed_loop_rollout_steps = 40
        config.closed_loop_batch_size = 16
        config.num_generations = 16
        config.noise_scale_range = [0.5, 2.0]
        config.diffusion_k_steps = 8
        config.grad_accum_groups = 8
        config.train_epochs = n_epochs

    config.use_closed_loop = True

    dit_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    with tempfile.TemporaryDirectory(prefix="profile_closed_loop_") as tmp_dir:
        run_dir = Path(tmp_dir)

        trainer = ClosedLoopExplorationTrainer(
            policy_model=model,
            model_args=model_args,
            dit_optimizer=dit_optimizer,
            device=DEVICE,
            run_dir=run_dir,
            config=config,
        )

        # Instrument key functions — MUST patch module globals for same-module references
        import rlvr.reward as reward_mod
        from rlvr import grpo_loss as loss_mod
        from rlvr.closed_loop import batched_rollout as br_mod
        from rlvr.closed_loop import per_step_reward as step_reward_mod

        profiler.wrap_module_function(
            br_mod, "_batched_generate_varied_noise", "cl.generation.varied_noise"
        )
        profiler.wrap_module_function(br_mod, "_batched_generate", "cl.generation.batched_generate")
        profiler.wrap_module_function(br_mod, "_batched_encoder", "cl.encoder")
        profiler.wrap_module_function(br_mod, "_load_npz", "cl.io.load_npz")
        profiler.wrap_module_function(step_reward_mod, "compute_step_reward", "cl.step_reward_orig")
        profiler.wrap_module_function(br_mod, "compute_step_reward", "cl.step_reward")
        profiler.wrap_module_function(reward_mod, "compute_reward_batch", "reward.compute_batch")
        profiler.wrap_module_function(
            reward_mod, "compute_road_border_penalty", "reward.road_border"
        )
        profiler.wrap_module_function(
            reward_mod, "compute_lane_departure_penalty", "reward.lane_departure"
        )
        profiler.wrap_module_function(reward_mod, "compute_safety_score_batch", "reward.safety")
        profiler.wrap_module_function(reward_mod, "compute_progress_score_batch", "reward.progress")
        profiler.wrap_module_function(
            reward_mod, "compute_feasibility_score_batch", "reward.feasibility"
        )
        profiler.wrap_module_function(
            reward_mod, "compute_centerline_score_batch", "reward.centerline"
        )
        profiler.wrap_module_function(loss_mod, "compute_batched_grpo_loss", "training.grpo_loss")

        from exploration_policy import utils as ep_utils

        profiler.wrap_module_function(
            ep_utils, "generate_reference_trajectory", "generation.reference_traj"
        )
        profiler.wrap_module_function(ep_utils, "run_frozen_encoder", "generation.frozen_encoder")

        if hasattr(trainer, "batched_rollout_manager"):
            profiler.wrap_function(
                trainer.batched_rollout_manager, "run_rollouts", "phase.batched_rollouts"
            )
            profiler.wrap_function(
                trainer.batched_rollout_manager, "_normalize_batch", "cl.normalize"
            )
            profiler.wrap_function(
                trainer.batched_rollout_manager, "_build_batched_composer", "cl.build_composer"
            )
        if hasattr(trainer, "rollout_manager"):
            profiler.wrap_function(trainer.rollout_manager, "run_rollout", "phase.single_rollout")

        profiler.wrap_function(trainer, "_run_dit_grpo", "phase.dit_grpo")

        from rlvr.closed_loop import state_update as su_mod

        profiler.wrap_module_function(su_mod, "update_scene_state", "cl.state_update_orig")
        profiler.wrap_module_function(br_mod, "update_scene_state", "cl.state_update")
        profiler.wrap_module_function(
            br_mod, "transform_positions_to_ego_frame", "cl.transform_positions"
        )

        import preference_optimization.utils as po_utils

        profiler.wrap_module_function(po_utils, "load_npz_data", "io.load_npz")

        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()

        metrics = trainer.train_epoch(scene_paths, epoch=epoch)

        total_wall = time.perf_counter() - wall_start

    return total_wall, metrics


# ---------------------------------------------------------------------------
# Evaluation profiling (runs after each epoch)
# ---------------------------------------------------------------------------


def profile_eval(model, model_args, scene_paths, profiler: Profiler):
    """Profile the evaluation pass that runs between epochs."""
    import rlvr.reward as reward_mod
    from rlvr.autoresearch.run_experiment import evaluate_checkpoint
    from rlvr.closed_loop import batched_rollout as br_mod
    from rlvr.reward import RewardConfig

    profiler.wrap_module_function(br_mod, "_batched_generate", "eval.batched_generate")
    profiler.wrap_module_function(reward_mod, "compute_reward_batch", "eval.reward_compute_batch")

    reward_config = RewardConfig(enable_overprogress=True)
    reward_config.enable_lane_departure = True

    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()

    with profiler.region("eval.total"):
        result = evaluate_checkpoint(model, model_args, scene_paths, reward_config, "profile-eval")

    total_wall = time.perf_counter() - wall_start
    return total_wall, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA. No GPU detected.")

    parser = argparse.ArgumentParser(description="Profile training epoch (requires CUDA)")
    parser.add_argument("--model_path", required=True, help="Base model .pth path")
    parser.add_argument("--scenes", required=True, help="Scene list JSON")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["rsft", "explorer", "closed_loop", "eval"],
        help="Training mode to profile",
    )
    parser.add_argument("--config", required=True, help="GRPO training config JSON (required)")
    parser.add_argument("--n_scenes", type=int, default=50, help="Number of scenes (default: 50)")
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to profile (default: 1)"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, model_args = load_model(args.model_path)
    print(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
    )

    scene_paths = load_scene_paths(args.scenes, args.n_scenes)
    print(f"Loaded {len(scene_paths)} scenes")

    # Warmup: one forward pass to initialize CUDA kernels
    print("Warming up CUDA...")
    model.eval()
    with torch.no_grad():
        from preference_optimization.utils import load_npz_data

        warmup_data = load_npz_data(scene_paths[0], DEVICE)
        norm_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in warmup_data.items()
        }
        norm_data = model_args.observation_normalizer(norm_data)
        from rlvr.closed_loop.batched_rollout import _batched_generate

        _ = _batched_generate(
            model, model_args, norm_data, noise_scale=0.0, composer=None, device=DEVICE
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    for epoch_i in range(args.n_epochs):
        profiler = Profiler()

        try:
            print(f"\n{'=' * 60}")
            print(
                f"PROFILING: mode={args.mode}, n_scenes={len(scene_paths)}, epoch={epoch_i + 1}/{args.n_epochs}"
            )
            print(f"{'=' * 60}\n")

            ep = epoch_i + 1
            if args.mode == "rsft":
                total_wall, metrics = profile_rsft(
                    model,
                    model_args,
                    scene_paths,
                    args.config,
                    profiler,
                    epoch=ep,
                    n_epochs=args.n_epochs,
                )
            elif args.mode == "explorer":
                total_wall, metrics = profile_explorer(
                    model,
                    model_args,
                    scene_paths,
                    args.config,
                    profiler,
                    epoch=ep,
                    n_epochs=args.n_epochs,
                )
            elif args.mode == "closed_loop":
                total_wall, metrics = profile_closed_loop(
                    model,
                    model_args,
                    scene_paths,
                    args.config,
                    profiler,
                    epoch=ep,
                    n_epochs=args.n_epochs,
                )
            elif args.mode == "eval":
                total_wall, metrics = profile_eval(model, model_args, scene_paths, profiler)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            print(profiler.report(total_wall))

            if isinstance(metrics, dict):
                print("\nTraining metrics:")
                for k, v in sorted(metrics.items()):
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
        finally:
            profiler.unpatch_all()


if __name__ == "__main__":
    main()
