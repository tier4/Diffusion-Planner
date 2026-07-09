"""R2LPL-style expert-disagreement conflict detection."""

from __future__ import annotations

from dataclasses import dataclass

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
    speed = torch.cat([step[:1] / dt, step / dt])
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
    for name, value in {
        "wait_speed_mps": wait_speed_mps,
        "wait_progress_m": wait_progress_m,
        "forward_progress_gap_m": forward_progress_gap_m,
        "lag_progress_gap_m": lag_progress_gap_m,
        "moving_speed_mps": moving_speed_mps,
        "dt": dt,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
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

    expert_waiting = expert_end_progress <= wait_progress_m and expert_mean_speed <= wait_speed_mps
    model_forward = (
        model_end_progress >= wait_progress_m + forward_progress_gap_m
        or model_end_speed >= moving_speed_mps
    )
    if expert_waiting and model_forward:
        return ConflictResult(
            True,
            0,
            max_dev,
            "expert_wait_model_forward",
            model_end_progress,
            expert_end_progress,
            model_end_speed,
            expert_end_speed,
        )

    progress_gap = expert_end_progress - model_end_progress
    mid_progress_gap = expert_mid_progress - model_mid_progress
    if (
        expert_end_speed >= moving_speed_mps
        and max(progress_gap, mid_progress_gap) >= lag_progress_gap_m
    ):
        return ConflictResult(
            True,
            0,
            max_dev,
            "model_lagging_expert",
            model_end_progress,
            expert_end_progress,
            model_end_speed,
            expert_end_speed,
        )

    if model_end_progress - expert_end_progress >= forward_progress_gap_m:
        return ConflictResult(
            True,
            0,
            max_dev,
            "model_ahead_expert",
            model_end_progress,
            expert_end_progress,
            model_end_speed,
            expert_end_speed,
        )

    return ConflictResult(
        False,
        None,
        max_dev,
        "",
        model_end_progress,
        expert_end_progress,
        model_end_speed,
        expert_end_speed,
    )
