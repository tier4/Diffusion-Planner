"""Tests for the warmup + post-warmup lr schedules.

Covers the fix where the post-warmup phase held the lr constant (the old
`CosineAnnealingWarmUpRestarts`) vs. the new cosine decay, plus the
backward-compatible alias and the dispatcher.
"""

import pytest
import torch
from diffusion_planner.utils.lr_schedule import (
    CosineAnnealingWarmUpRestarts,
    WarmupConstantLR,
    WarmupCosineAnnealingLR,
    apply_legacy_final_phase_lr,
    build_lr_scheduler,
    rebuild_lr_scheduler,
    resolve_lr_schedule_steps,
)

PEAK_LR = 1e-3
WARMUP = 5
EPOCHS = 25
ETA_MIN = 1e-6


def _make_optimizer():
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=PEAK_LR)


def _run(scheduler, epochs=EPOCHS):
    """Return the lr seen at the start of each epoch (step() once per epoch)."""
    optimizer = scheduler.optimizer
    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def _run_training_loop(schedule_type, epochs=EPOCHS, warmup=WARMUP):
    """Mirror the LR-related ordering in model_training."""
    optimizer = _make_optimizer()
    scheduler = build_lr_scheduler(
        optimizer,
        epochs,
        warmup,
        schedule_type=schedule_type,
        eta_min=ETA_MIN,
    )
    lrs = []
    for current_epoch in range(epochs):
        apply_legacy_final_phase_lr(
            optimizer,
            current_epoch=current_epoch,
            total_epochs=epochs,
            base_lr=PEAK_LR,
            schedule_type=schedule_type,
        )
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_warmup_ramps_from_start_factor_to_peak():
    opt = _make_optimizer()
    lrs = _run(WarmupCosineAnnealingLR(opt, EPOCHS, WARMUP, start_factor=0.1, eta_min=ETA_MIN))
    # first epoch starts at start_factor * peak, peak reached by end of warmup
    assert lrs[0] == pytest.approx(0.1 * PEAK_LR, rel=1e-6)
    assert max(lrs) == pytest.approx(PEAK_LR, rel=1e-6)
    assert lrs[WARMUP] == pytest.approx(PEAK_LR, rel=1e-6)


def test_constant_holds_peak_after_warmup():
    opt = _make_optimizer()
    lrs = _run(WarmupConstantLR(opt, EPOCHS, WARMUP, start_factor=0.1))
    # every post-warmup epoch stays at the peak lr
    post_warmup = lrs[WARMUP:]
    assert all(lr == pytest.approx(PEAK_LR, rel=1e-6) for lr in post_warmup)


def test_cosine_decays_after_warmup():
    opt = _make_optimizer()
    lrs = _run(WarmupCosineAnnealingLR(opt, EPOCHS, WARMUP, start_factor=0.1, eta_min=ETA_MIN))
    post_warmup = lrs[WARMUP:]
    # strictly decreasing after the peak, and lands near eta_min by the end
    assert all(b < a for a, b in zip(post_warmup, post_warmup[1:]))
    assert lrs[-1] < 0.5 * PEAK_LR
    assert lrs[-1] >= ETA_MIN


def test_cosine_differs_from_constant():
    opt_c = _make_optimizer()
    opt_k = _make_optimizer()
    cosine = _run(WarmupCosineAnnealingLR(opt_c, EPOCHS, WARMUP, eta_min=ETA_MIN))
    constant = _run(WarmupConstantLR(opt_k, EPOCHS, WARMUP))
    # identical through warmup, diverge afterwards
    assert cosine[:WARMUP] == pytest.approx(constant[:WARMUP], rel=1e-6)
    assert cosine[-1] < constant[-1]


def test_alias_preserves_legacy_constant_behavior():
    # The old public name must keep the exact (constant) behavior for compat.
    opt_alias = _make_optimizer()
    opt_const = _make_optimizer()
    alias = _run(CosineAnnealingWarmUpRestarts(opt_alias, EPOCHS, WARMUP))
    constant = _run(WarmupConstantLR(opt_const, EPOCHS, WARMUP))
    assert alias == pytest.approx(constant, rel=1e-6)


def test_build_lr_scheduler_dispatch():
    opt_cos = _make_optimizer()
    opt_con = _make_optimizer()
    cosine = _run(build_lr_scheduler(opt_cos, EPOCHS, WARMUP, schedule_type="cosine"))
    constant = _run(build_lr_scheduler(opt_con, EPOCHS, WARMUP, schedule_type="constant"))
    assert cosine[-1] < constant[-1]
    with pytest.raises(ValueError):
        build_lr_scheduler(_make_optimizer(), EPOCHS, WARMUP, schedule_type="bogus")


def test_cosine_handles_short_warmup_edge_case():
    # warm_up_epoch == 1 and epoch == warm_up_epoch must not raise.
    opt = _make_optimizer()
    WarmupCosineAnnealingLR(opt, 1, 1)
    opt2 = _make_optimizer()
    sched = WarmupCosineAnnealingLR(opt2, 10, 1)
    _run(sched, epochs=10)


def test_training_loop_does_not_override_cosine_in_final_ten_epochs():
    lrs = _run_training_loop("cosine", epochs=20, warmup=5)
    expected = _run(
        build_lr_scheduler(
            _make_optimizer(),
            20,
            5,
            schedule_type="cosine",
            eta_min=ETA_MIN,
        ),
        epochs=20,
    )
    assert lrs == pytest.approx(expected)
    assert all(b < a for a, b in zip(lrs[5:], lrs[6:]))


def test_short_training_loop_distinguishes_cosine_from_legacy_constant():
    cosine = _run_training_loop("cosine", epochs=2, warmup=1)
    constant = _run_training_loop("constant", epochs=2, warmup=1)
    assert cosine != pytest.approx(constant)
    assert constant == pytest.approx([PEAK_LR * 0.01] * 2)


def test_resume_rebuild_matches_uninterrupted_cosine_schedule():
    optimizer = _make_optimizer()
    uninterrupted_scheduler = build_lr_scheduler(
        optimizer,
        100,
        5,
        schedule_type="cosine",
        eta_min=ETA_MIN,
    )
    uninterrupted = _run(uninterrupted_scheduler, epochs=100)

    resumed_optimizer = _make_optimizer()
    resumed_scheduler = rebuild_lr_scheduler(
        resumed_optimizer,
        100,
        5,
        completed_epochs=50,
        learning_rate=PEAK_LR,
        schedule_type="cosine",
        eta_min=ETA_MIN,
    )
    resumed = _run(resumed_scheduler, epochs=50)

    assert resumed == pytest.approx(uninterrupted[50:])


def test_resume_rebuild_rebases_schedule_to_new_learning_rate():
    new_lr = PEAK_LR / 2
    optimizer = _make_optimizer()
    scheduler = rebuild_lr_scheduler(
        optimizer,
        EPOCHS,
        WARMUP,
        completed_epochs=10,
        learning_rate=new_lr,
        schedule_type="cosine",
        eta_min=ETA_MIN,
    )

    expected_optimizer = _make_optimizer()
    expected_optimizer.param_groups[0]["lr"] = new_lr
    expected_optimizer.param_groups[0]["initial_lr"] = new_lr
    expected = _run(
        build_lr_scheduler(
            expected_optimizer,
            EPOCHS,
            WARMUP,
            schedule_type="cosine",
            eta_min=ETA_MIN,
        ),
        epochs=EPOCHS,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected[10])
    assert _run(scheduler, epochs=EPOCHS - 10) == pytest.approx(expected[10:])


def test_all_cosine_parameters_are_configurable():
    start_factor = 0.25
    eta_min = 2e-5
    cosine_t_max = 7
    optimizer = _make_optimizer()
    scheduler = build_lr_scheduler(
        optimizer,
        EPOCHS,
        WARMUP,
        schedule_type="cosine",
        start_factor=start_factor,
        eta_min=eta_min,
        cosine_t_max=cosine_t_max,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(PEAK_LR * start_factor)
    cosine = scheduler._schedulers[1]
    assert cosine.eta_min == eta_min
    assert cosine.T_max == cosine_t_max


def test_batch_interval_converts_epoch_durations_to_optimizer_steps():
    resolved = resolve_lr_schedule_steps(
        total_epochs=20,
        warm_up_epochs=5,
        steps_per_epoch=8,
        interval="batch",
        cosine_t_max_epochs=12,
        completed_epochs=7,
    )
    assert resolved == {
        "total": 160,
        "warmup": 40,
        "completed": 56,
        "cosine_t_max": 96,
    }


def test_epoch_interval_keeps_epoch_durations():
    resolved = resolve_lr_schedule_steps(
        total_epochs=20,
        warm_up_epochs=5,
        steps_per_epoch=8,
        interval="epoch",
        cosine_t_max_epochs=0,
        completed_epochs=7,
    )
    assert resolved == {
        "total": 20,
        "warmup": 5,
        "completed": 7,
        "cosine_t_max": None,
    }


def test_batch_interval_resume_matches_uninterrupted_schedule():
    resolved = resolve_lr_schedule_steps(
        total_epochs=20,
        warm_up_epochs=5,
        steps_per_epoch=8,
        interval="batch",
        cosine_t_max_epochs=12,
        completed_epochs=7,
    )
    optimizer = _make_optimizer()
    uninterrupted_scheduler = build_lr_scheduler(
        optimizer,
        resolved["total"],
        resolved["warmup"],
        schedule_type="cosine",
        eta_min=ETA_MIN,
        cosine_t_max=resolved["cosine_t_max"],
    )
    uninterrupted = _run(uninterrupted_scheduler, epochs=resolved["total"])

    resumed_optimizer = _make_optimizer()
    resumed_scheduler = rebuild_lr_scheduler(
        resumed_optimizer,
        resolved["total"],
        resolved["warmup"],
        completed_epochs=resolved["completed"],
        learning_rate=PEAK_LR,
        schedule_type="cosine",
        eta_min=ETA_MIN,
        cosine_t_max=resolved["cosine_t_max"],
    )
    resumed = _run(
        resumed_scheduler,
        epochs=resolved["total"] - resolved["completed"],
    )
    assert resumed == pytest.approx(uninterrupted[resolved["completed"] :])
