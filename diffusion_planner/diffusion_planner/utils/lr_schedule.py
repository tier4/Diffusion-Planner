from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiplicativeLR,
    SequentialLR,
)


def final_phase_lr(base_lr, epoch, train_epochs, lr_schedule="constant", final_epoch_count=10):
    """LR override for the last ``final_epoch_count`` epochs of the constant schedule.

    Returns the adjusted LR (``0.1 * base_lr`` for the first half of the final
    phase, ``0.01 * base_lr`` for the second half), or ``None`` when no override
    applies. Under ``"cosine"`` there is never an override: the annealed schedule
    already decays smoothly to 0, and stamping ``0.1 * base_lr`` over its tail
    would *raise* the LR back up in the very epochs it is meant to settle.
    """
    if lr_schedule == "cosine":
        return None
    if epoch < train_epochs - final_epoch_count:
        return None
    if epoch >= train_epochs - final_epoch_count // 2:
        return base_lr * 0.01
    return base_lr * 0.1


def CosineAnnealingWarmUpRestarts(
    optimizer, epoch, warm_up_epoch, start_factor=0.1, lr_schedule="constant"
):
    """Linear warm-up, then either hold the LR or anneal it to zero.

    Args:
        lr_schedule: ``"constant"`` (default, unchanged behaviour) holds the LR at its
            configured value after warm-up. ``"cosine"`` decays it to 0 over the
            remaining epochs.

    The default is ``"constant"`` because that is what this function has always done
    despite its name, and every existing checkpoint was trained that way.

    Why ``"cosine"`` exists: at a constant LR the weights do not settle, so a saved
    checkpoint is one arbitrary sample of a random walk around the minimum. Measured on
    a 2.3k-scene set (72 steps/epoch, 200 epochs), two functionally near-identical runs
    landed 23 points apart in closed-loop recovery rate at the final checkpoint, and
    validation loss swung 6-9 points within the last twenty epochs. With cosine decay
    the same comparison differed by 0.6 points and each run was flat across its late
    checkpoints. On small datasets this makes A/B comparisons meaningful; on
    full-size datasets the effect is unmeasured.
    """
    assert epoch >= warm_up_epoch
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}; expected 'constant' or 'cosine'")
    T_warmup = warm_up_epoch

    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    if lr_schedule == "cosine":
        post_warmup = CosineAnnealingLR(optimizer, T_max=max(epoch - T_warmup, 1), eta_min=0.0)
    else:
        post_warmup = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, post_warmup], milestones=[T_warmup]
    )

    return scheduler
