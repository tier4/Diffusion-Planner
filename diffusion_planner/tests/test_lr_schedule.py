"""LR schedule integration tests.

Replicates train.py's exact per-epoch sequence — final-phase override, then the
training step (which reads optimizer.param_groups), then scheduler.step() — and
asserts on the LR the optimizer would actually use, not the standalone scheduler
state.
"""

import math

import torch
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts, final_phase_lr

BASE_LR = 1e-4
EPOCHS = 20
WARM_UP = 5


def _observed_lrs(lr_schedule: str) -> list[float]:
    """LR seen by the training step at every epoch, in train.py's order."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    sched = CosineAnnealingWarmUpRestarts(opt, EPOCHS, WARM_UP, lr_schedule=lr_schedule)
    lrs = []
    for epoch in range(EPOCHS):
        adjusted = final_phase_lr(BASE_LR, epoch, EPOCHS, lr_schedule)
        if adjusted is not None:
            for pg in opt.param_groups:
                pg["lr"] = adjusted
        lrs.append(opt.param_groups[0]["lr"])  # what optimizer.step() uses
        sched.step()
    return lrs


def test_constant_schedule_keeps_final_phase_override():
    """Default behavior is byte-identical: flat LR, then the 0.1x/0.01x tail."""
    lrs = _observed_lrs("constant")
    assert all(math.isclose(lr, BASE_LR) for lr in lrs[WARM_UP : EPOCHS - 10])
    assert all(math.isclose(lr, BASE_LR * 0.1) for lr in lrs[EPOCHS - 10 : EPOCHS - 5])
    assert all(math.isclose(lr, BASE_LR * 0.01) for lr in lrs[EPOCHS - 5 :])
    print("  [PASS] constant schedule: flat + 0.1x/0.01x final phase")


def test_cosine_schedule_never_overridden():
    """Cosine must decay monotonically to ~0 after warm-up with NO tail bump:
    the final-phase override would otherwise raise the LR back to 0.1x base in
    the very epochs the schedule is meant to settle."""
    lrs = _observed_lrs("cosine")
    post = lrs[WARM_UP - 1 :]
    assert all(b <= a + 1e-12 for a, b in zip(post, post[1:])), (
        f"cosine LR increased somewhere after warm-up: {post}"
    )
    assert lrs[-1] < BASE_LR * 0.02, f"cosine tail did not anneal: final LR {lrs[-1]:.2e}"
    assert all(final_phase_lr(BASE_LR, e, EPOCHS, "cosine") is None for e in range(EPOCHS)), (
        "final_phase_lr must never fire under cosine"
    )
    print("  [PASS] cosine schedule: monotone decay, no final-phase override")


def test_warmup_identical_for_both_schedules():
    c, a = _observed_lrs("constant"), _observed_lrs("cosine")
    assert c[:WARM_UP] == a[:WARM_UP], "warm-up must not depend on the schedule choice"
    print("  [PASS] warm-up identical for both schedules")


if __name__ == "__main__":
    test_constant_schedule_keeps_final_phase_override()
    test_cosine_schedule_never_overridden()
    test_warmup_identical_for_both_schedules()
    print("All tests passed!")
