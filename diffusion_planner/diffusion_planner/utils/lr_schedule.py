import warnings

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiplicativeLR,
    SequentialLR,
)


def WarmupConstantLR(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    """Linear warmup for ``warm_up_epoch`` epochs, then HOLD the lr constant.

    This is the original schedule, formerly (mis)named
    ``CosineAnnealingWarmUpRestarts``: the post-warmup phase is
    ``MultiplicativeLR(lambda=1.0)``, so the learning rate never decays. Kept
    under an accurate name for backward-compatible reproduction of pre-existing
    runs and resumed checkpoints. New training should prefer
    :func:`WarmupCosineAnnealingLR`.

    Called with ``scheduler.step()`` once per epoch, so the sub-schedulers count
    in epochs.
    """
    assert epoch >= warm_up_epoch
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed = MultiplicativeLR(optimizer, lr_lambda=lambda e: 1.0)
    return SequentialLR(optimizer, schedulers=[warmup, fixed], milestones=[warm_up_epoch])


def WarmupCosineAnnealingLR(
    optimizer,
    epoch,
    warm_up_epoch,
    start_factor=0.1,
    eta_min=1e-6,
    cosine_t_max=None,
):
    """Linear warmup for ``warm_up_epoch`` epochs, then cosine-decay to
    ``eta_min`` over the remaining epochs.

    Replaces the constant post-warmup phase of :func:`WarmupConstantLR`, which
    let training collapse right after warmup: the learning rate stayed at its
    peak for the whole run, so every model peaked in the warmup epochs and then
    degraded. Decaying with a proper cosine schedule (matching original planTF)
    keeps the lr low enough after the initial phase to actually converge.

    Called with ``scheduler.step()`` once per epoch, so the sub-schedulers count
    in epochs. ``max(..., 1)`` guards the ``warm_up_epoch == 1`` /
    ``epoch == warm_up_epoch`` edge cases so the sub-schedulers always get
    positive spans.
    """
    assert epoch >= warm_up_epoch
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=max(warm_up_epoch - 1, 1))
    t_max = max(epoch - warm_up_epoch, 1) if cosine_t_max is None else cosine_t_max
    if t_max <= 0:
        raise ValueError(f"cosine_t_max must be positive, got {t_max}")
    cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm_up_epoch])


# Backward-compatible alias. The original public name advertised cosine annealing
# but the implementation held the lr constant after warmup; keep the name bound
# to that exact (constant) behavior so existing imports and resumed checkpoints
# reproduce unchanged. New code should call WarmupCosineAnnealingLR directly (or
# select it via TrainConfig.lr_schedule_type="cosine").
CosineAnnealingWarmUpRestarts = WarmupConstantLR


def build_lr_scheduler(
    optimizer,
    epoch,
    warm_up_epoch,
    schedule_type="cosine",
    start_factor=0.1,
    eta_min=1e-6,
    cosine_t_max=None,
):
    """Dispatch to the configured lr schedule.

    ``schedule_type="cosine"`` (default) -> :func:`WarmupCosineAnnealingLR`;
    ``"constant"`` -> :func:`WarmupConstantLR` (the legacy behavior).
    """
    if schedule_type == "cosine":
        return WarmupCosineAnnealingLR(
            optimizer,
            epoch,
            warm_up_epoch,
            start_factor=start_factor,
            eta_min=eta_min,
            cosine_t_max=cosine_t_max,
        )
    if schedule_type == "constant":
        return WarmupConstantLR(
            optimizer,
            epoch,
            warm_up_epoch,
            start_factor=start_factor,
        )
    raise ValueError(
        f"Unknown lr schedule_type: {schedule_type!r} (expected 'cosine' or 'constant')"
    )


def rebuild_lr_scheduler(
    optimizer,
    epoch,
    warm_up_epoch,
    completed_epochs,
    learning_rate,
    schedule_type="cosine",
    start_factor=0.1,
    eta_min=1e-6,
    cosine_t_max=None,
):
    """Rebuild a scheduler at the LR corresponding to a resumed epoch.

    Loading an optimizer checkpoint restores the scheduled LR, but the training
    entry point also supports selecting a new base LR when resuming. Directly
    assigning that base LR would restart training at the peak and leave the
    scheduler's internal epoch out of sync. Rebuilding and fast-forwarding keeps
    both the optimizer LR and scheduler state aligned.
    """
    if not 0 <= completed_epochs <= epoch:
        raise ValueError(f"completed_epochs must be in [0, {epoch}], got {completed_epochs}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["initial_lr"] = learning_rate

    scheduler = build_lr_scheduler(
        optimizer,
        epoch,
        warm_up_epoch,
        schedule_type=schedule_type,
        start_factor=start_factor,
        eta_min=eta_min,
        cosine_t_max=cosine_t_max,
    )
    if completed_epochs:
        # Fast-forwarding does not update parameters. Suppress PyTorch's generic
        # ordering warning because there is intentionally no optimizer step here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
            )
            for _ in range(completed_epochs):
                scheduler.step()
    return scheduler


def apply_legacy_final_phase_lr(
    optimizer,
    current_epoch,
    total_epochs,
    base_lr,
    schedule_type,
    final_epoch_count=10,
):
    """Apply the pre-cosine final LR step-down only for the legacy schedule."""
    if final_epoch_count < 0:
        raise ValueError(f"final_epoch_count must be non-negative, got {final_epoch_count}")
    if schedule_type != "constant" or current_epoch < total_epochs - final_epoch_count:
        return None

    if current_epoch >= total_epochs - final_epoch_count // 2:
        adjusted_lr = base_lr * 0.01
    else:
        adjusted_lr = base_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = adjusted_lr
    return adjusted_lr


def resolve_lr_schedule_steps(
    total_epochs,
    warm_up_epochs,
    steps_per_epoch,
    interval="epoch",
    cosine_t_max_epochs=0,
    completed_epochs=0,
):
    """Convert user-facing epoch durations to scheduler-step durations."""
    if interval not in {"epoch", "batch"}:
        raise ValueError(f"Unknown lr scheduler interval: {interval!r}")
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if cosine_t_max_epochs < 0:
        raise ValueError(f"cosine_t_max_epochs must be non-negative, got {cosine_t_max_epochs}")

    multiplier = steps_per_epoch if interval == "batch" else 1
    cosine_t_max = cosine_t_max_epochs * multiplier if cosine_t_max_epochs else None
    return {
        "total": total_epochs * multiplier,
        "warmup": warm_up_epochs * multiplier,
        "completed": completed_epochs * multiplier,
        "cosine_t_max": cosine_t_max,
    }
