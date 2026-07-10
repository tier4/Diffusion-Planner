"""R2LPL-style expert-disagreement conflict detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ConflictResult:
    expert_disagreement: bool
    expert_disagreement_step: int | None
    max_deviation: float
    reason: str = ""
    model_end_progress: float = 0.0
    expert_end_progress: float = 0.0
    model_end_speed: float = 0.0
    expert_end_speed: float = 0.0


def _validate_disagreement_thresholds(**thresholds: float) -> None:
    """Reject non-positive thresholds (loud), shared by both detector variants."""
    for name, value in thresholds.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


def _disagreement_decision(
    *,
    model_mid_progress: float,
    model_end_progress: float,
    expert_mid_progress: float,
    expert_end_progress: float,
    model_end_speed: float,
    expert_end_speed: float,
    expert_mean_speed: float,
    wait_speed_mps: float,
    wait_progress_m: float,
    forward_progress_gap_m: float,
    lag_progress_gap_m: float,
    moving_speed_mps: float,
) -> tuple[bool, str]:
    """R2LPL 3-branch expert-disagreement decision on horizon summary scalars.

    This is the frozen port of ``_model_expert_disagreement`` from the reference
    (``lpl_planner/rollout/rollout_cl_generator.py``). Written once and shared by
    every detector variant so the branch arithmetic stays identical.
    """
    expert_waiting = expert_end_progress <= wait_progress_m and expert_mean_speed <= wait_speed_mps
    model_forward = (
        model_end_progress >= wait_progress_m + forward_progress_gap_m
        or model_end_speed >= moving_speed_mps
    )
    if expert_waiting and model_forward:
        return True, "expert_wait_model_forward"

    progress_gap = expert_end_progress - model_end_progress
    mid_progress_gap = expert_mid_progress - model_mid_progress
    if (
        expert_end_speed >= moving_speed_mps
        and max(progress_gap, mid_progress_gap) >= lag_progress_gap_m
    ):
        return True, "model_lagging_expert"

    if model_end_progress - expert_end_progress >= forward_progress_gap_m:
        return True, "model_ahead_expert"

    return False, ""


def _progress_and_speed(traj: torch.Tensor, *, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    xy = traj[..., :2].to(torch.float32)
    if xy.shape[-2] == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=xy.device)
        return empty, empty
    if xy.shape[-2] == 1:
        return (
            torch.zeros(1, dtype=torch.float32, device=xy.device),
            torch.zeros(1, dtype=torch.float32, device=xy.device),
        )
    step = torch.linalg.norm(xy[1:] - xy[:-1], dim=-1)
    progress = torch.cat(
        [torch.zeros(1, dtype=torch.float32, device=xy.device), torch.cumsum(step, dim=0)]
    )
    # speed[t] is the speed of the interval ENDING at t (t=0 has no prior interval, so
    # 0). Pad with a leading zero rather than duplicating step[0], which would bias
    # the mean speed and misalign speed[0] with the progress convention above.
    speed = torch.cat([torch.zeros(1, dtype=torch.float32, device=xy.device), step / dt])
    return progress, speed


def detect_expert_disagreement(
    rollout: torch.Tensor,
    gt: torch.Tensor,
    *,
    enabled: bool,
    wait_speed_mps: float,
    wait_progress_m: float,
    forward_progress_gap_m: float,
    lag_progress_gap_m: float,
    moving_speed_mps: float,
    dt: float = 0.1,
) -> ConflictResult:
    """Detect R2LPL conflict between model rollout behavior and logged expert behavior."""
    if not enabled:
        return ConflictResult(False, None, 0.0)
    _validate_disagreement_thresholds(
        wait_speed_mps=wait_speed_mps,
        wait_progress_m=wait_progress_m,
        forward_progress_gap_m=forward_progress_gap_m,
        lag_progress_gap_m=lag_progress_gap_m,
        moving_speed_mps=moving_speed_mps,
        dt=dt,
    )
    if rollout.shape[-1] < 2 or gt.shape[-1] < 2:
        raise ValueError("rollout and gt must have xy in the first two channels")

    horizon = min(int(rollout.shape[-2]), int(gt.shape[-2]))
    if horizon <= 1:
        return ConflictResult(False, None, 0.0)

    model = rollout[:horizon]
    expert = gt[:horizon]
    model_progress, model_speed = _progress_and_speed(model, dt=dt)
    expert_progress, expert_speed = _progress_and_speed(expert, dt=dt)
    if model_progress.numel() == 0 or expert_progress.numel() == 0:
        return ConflictResult(False, None, 0.0)

    mid_idx = horizon // 2
    model_mid_progress = float(model_progress[mid_idx].item())
    model_end_progress = float(model_progress[-1].item())
    expert_mid_progress = float(expert_progress[mid_idx].item())
    expert_end_progress = float(expert_progress[-1].item())
    model_end_speed = float(model_speed[-1].item())
    expert_end_speed = float(expert_speed[-1].item())
    expert_mean_speed = float(expert_speed.mean().item())
    max_dev = float(torch.linalg.norm(model[:, :2] - expert[:, :2], dim=-1).max().item())

    flag, reason = _disagreement_decision(
        model_mid_progress=model_mid_progress,
        model_end_progress=model_end_progress,
        expert_mid_progress=expert_mid_progress,
        expert_end_progress=expert_end_progress,
        model_end_speed=model_end_speed,
        expert_end_speed=expert_end_speed,
        expert_mean_speed=expert_mean_speed,
        wait_speed_mps=wait_speed_mps,
        wait_progress_m=wait_progress_m,
        forward_progress_gap_m=forward_progress_gap_m,
        lag_progress_gap_m=lag_progress_gap_m,
        moving_speed_mps=moving_speed_mps,
    )
    return ConflictResult(
        flag,
        None,  # not localized: flagged from horizon summaries, no step index
        max_dev,
        reason,
        model_end_progress,
        expert_end_progress,
        model_end_speed,
        expert_end_speed,
    )


def detect_expert_disagreement_projected(
    model_progress: np.ndarray,
    model_speed: np.ndarray,
    expert_progress: np.ndarray,
    expert_speed: np.ndarray,
    *,
    wait_speed_mps: float,
    wait_progress_m: float,
    forward_progress_gap_m: float,
    lag_progress_gap_m: float,
    moving_speed_mps: float,
) -> ConflictResult:
    """Faithful port of the reference ``_model_expert_disagreement``.

    Operates on PRE-COMPUTED progress + speed arrays (route arc-length progress and
    real speed), NOT on xy paths — matching the reference, which reads progress from
    trajectory column 0 and speed from column 3. Applies the reference's expert +1
    step shift: ``horizon = min(len(model_progress), len(expert_progress) - 1)``, the
    model uses ``[:horizon]`` and the expert uses ``[1 : horizon + 1]``. The expert
    arrays must therefore start at the CURRENT state (index 0), so the shift lands the
    first future expert sample on the model's first predicted step (the reference
    builds ``expert = [log_current_state] + future_states`` for the same reason).

    ``max_deviation`` is reported as ``max|model_progress - expert_progress|`` over the
    aligned horizon purely for downstream ranking; it does NOT influence the flag.
    """
    _validate_disagreement_thresholds(
        wait_speed_mps=wait_speed_mps,
        wait_progress_m=wait_progress_m,
        forward_progress_gap_m=forward_progress_gap_m,
        lag_progress_gap_m=lag_progress_gap_m,
        moving_speed_mps=moving_speed_mps,
    )
    model_progress = np.asarray(model_progress, dtype=np.float32).reshape(-1)
    model_speed = np.asarray(model_speed, dtype=np.float32).reshape(-1)
    expert_progress = np.asarray(expert_progress, dtype=np.float32).reshape(-1)
    expert_speed = np.asarray(expert_speed, dtype=np.float32).reshape(-1)
    if model_progress.shape[0] != model_speed.shape[0]:
        raise ValueError("model_progress and model_speed must have equal length")
    if expert_progress.shape[0] != expert_speed.shape[0]:
        raise ValueError("expert_progress and expert_speed must have equal length")

    horizon = min(int(model_progress.shape[0]), int(expert_progress.shape[0]) - 1)
    if horizon <= 1:
        return ConflictResult(False, None, 0.0)

    model_prog = model_progress[:horizon]
    model_spd = model_speed[:horizon]
    expert_prog = expert_progress[1 : horizon + 1]
    expert_spd = expert_speed[1 : horizon + 1]
    mid_idx = horizon // 2

    model_mid_progress = float(model_prog[mid_idx])
    model_end_progress = float(model_prog[-1])
    expert_mid_progress = float(expert_prog[mid_idx])
    expert_end_progress = float(expert_prog[-1])
    model_end_speed = float(model_spd[-1])
    expert_end_speed = float(expert_spd[-1])
    expert_mean_speed = float(np.mean(np.abs(expert_spd)))
    max_dev = float(np.max(np.abs(model_prog - expert_prog)))

    flag, reason = _disagreement_decision(
        model_mid_progress=model_mid_progress,
        model_end_progress=model_end_progress,
        expert_mid_progress=expert_mid_progress,
        expert_end_progress=expert_end_progress,
        model_end_speed=model_end_speed,
        expert_end_speed=expert_end_speed,
        expert_mean_speed=expert_mean_speed,
        wait_speed_mps=wait_speed_mps,
        wait_progress_m=wait_progress_m,
        forward_progress_gap_m=forward_progress_gap_m,
        lag_progress_gap_m=lag_progress_gap_m,
        moving_speed_mps=moving_speed_mps,
    )
    return ConflictResult(
        flag,
        None,  # not localized: flagged from horizon summaries, no step index
        max_dev,
        reason,
        model_end_progress,
        expert_end_progress,
        model_end_speed,
        expert_end_speed,
    )
