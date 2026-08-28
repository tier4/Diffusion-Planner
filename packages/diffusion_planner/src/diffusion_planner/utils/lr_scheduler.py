from dataclasses import dataclass, fields
from typing import Any

from timm.scheduler.scheduler import Scheduler
from timm.scheduler.scheduler_factory import create_scheduler_v2
from torch.optim import Optimizer

# "constant" is not a timm schedule; it is built below as a step schedule that never
# decays, so warm-up behaves exactly as it does for the other shapes.
LR_SCHEDULER_CHOICES = ("cosine", "tanh", "step", "multistep", "poly", "constant")


@dataclass
class LRScheduleConfig:
    """Everything timm's factory needs. All durations are in optimizer steps."""

    lr_scheduler: str = "cosine"
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    min_lr: float = 0.0

    warmup_lr: float = 1e-7
    warmup_prefix: bool = True
    decay_steps: int = 10000
    decay_milestones: tuple[int, ...] = (20000, 40000)
    decay_rate: float = 0.1
    cycle_mul: float = 1.0
    cycle_decay: float = 0.5
    cycle_limit: int = 1
    k_decay: float = 1.0

    @classmethod
    def from_args(cls, args: Any) -> "LRScheduleConfig":
        """Take whichever fields ``args`` defines, and default the rest."""
        overrides = {
            f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)
        }
        return cls(**overrides)


def build_lr_scheduler(optimizer: Optimizer, args: Any, total_steps: int) -> Scheduler:
    """Build the configured schedule over ``total_steps`` optimizer steps."""
    cfg = LRScheduleConfig.from_args(args)
    if cfg.lr_scheduler not in LR_SCHEDULER_CHOICES:
        raise ValueError(
            f"unknown lr_scheduler {cfg.lr_scheduler!r}, expected one of {LR_SCHEDULER_CHOICES}"
        )
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    sched, decay_rate, decay_steps = cfg.lr_scheduler, cfg.decay_rate, cfg.decay_steps
    if sched == "constant":
        sched, decay_rate, decay_steps = "step", 1.0, total_steps

    # A prefixed warm-up is played before the schedule rather than over it, so the
    # schedule itself only gets the remaining steps if the run is to end on time.
    sched_steps = total_steps - cfg.warmup_steps if cfg.warmup_prefix else total_steps
    if sched_steps <= 0:
        raise ValueError(
            f"warmup_steps ({cfg.warmup_steps}) must be below total_steps ({total_steps})"
        )

    # The factory is epoch-denominated and multiplies its epoch arguments by
    # updates_per_epoch; passing 1 makes that conversion the identity, so every duration
    # below is read directly as a step count.
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=sched,
        num_epochs=sched_steps,
        decay_epochs=decay_steps,
        decay_milestones=list(cfg.decay_milestones),
        decay_rate=decay_rate,
        min_lr=cfg.min_lr,
        warmup_lr=cfg.warmup_lr,
        warmup_epochs=cfg.warmup_steps,
        warmup_prefix=cfg.warmup_prefix,
        cycle_mul=cfg.cycle_mul,
        cycle_decay=cfg.cycle_decay,
        cycle_limit=cfg.cycle_limit,
        k_decay=cfg.k_decay,
        step_on_epochs=False,
        updates_per_epoch=1,
    )
    if scheduler is None:
        raise ValueError(f"timm could not create LR scheduler {sched!r}")
    return scheduler


def set_base_lr(scheduler: Scheduler, lr: float) -> None:
    """Re-anchor a schedule on a new peak LR (e.g. a new ``--learning_rate`` on resume)."""
    for group in scheduler.optimizer.param_groups:
        group["initial_lr"] = lr
    scheduler.base_values = [lr for _ in scheduler.base_values]


def describe_lr_scheduler(args: Any, total_steps: int) -> str:
    cfg = LRScheduleConfig.from_args(args)
    return (
        f"LR schedule: {cfg.lr_scheduler} over {total_steps} steps, "
        f"peak={cfg.learning_rate:g}, min={cfg.min_lr:g}, "
        f"warmup={cfg.warmup_steps} steps from {cfg.warmup_lr:g}"
    )
