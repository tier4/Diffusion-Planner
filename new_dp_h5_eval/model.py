"""ONNX inference boundary for the new planner."""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import torch

from .schema import (
    INITIAL_NOISE_SHAPE,
    MODEL_INPUT_NAMES,
    NUM_AGENTS,
    TURN_LOGIT_DIM,
)


def _providers_for_local_rank(
    providers: list[str] | None,
) -> list[str | tuple[str, dict[str, str]]] | None:
    """Bind GPU execution providers to the torchrun local rank when present."""
    local_rank = os.environ.get("LOCAL_RANK")
    if providers is None or local_rank is None:
        return providers
    return [
        (provider, {"device_id": local_rank})
        if provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        else provider
        for provider in providers
    ]


def normalize_frame(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(frame)
    for key in (
        "ego_agent_past",
        "neighbor_agents_past",
        "ego_agent_future",
        "neighbor_agents_future",
        "goal_pose",
    ):
        if key in frame:
            out[key] = frame[key].copy()
            out[key][..., :2] /= 50.0
    for key in ("lanes", "route_lanes", "intersection_area", "stop_lines", "road_borders"):
        if key in frame:
            out[key] = frame[key] / 50.0
    for key in ("lanes_speed_limit", "route_lanes_speed_limit"):
        if key in frame:
            out[key] = frame[key] / 15.0
    for key in ("agent_shape", "ego_shape"):
        if key in frame:
            out[key] = frame[key] / 10.0
    return out


def seeded_initial_noise(seeds: list[int]) -> np.ndarray:
    """Create the new-DP sampler noise deterministically, one seed per batch item."""
    return np.stack(
        [
            torch.randn(INITIAL_NOISE_SHAPE, generator=torch.Generator().manual_seed(seed)).numpy()
            for seed in seeds
        ]
    )


def decode_onnx_outputs(
    trajectory: np.ndarray, turn_logits: np.ndarray, *, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and denormalize the current sampler's two ONNX outputs."""
    trajectory = np.asarray(trajectory)
    turn_logits = np.asarray(turn_logits)
    expected_trajectory = (batch_size, NUM_AGENTS, *INITIAL_NOISE_SHAPE[1:])
    if trajectory.shape != expected_trajectory:
        raise ValueError(f"unexpected ONNX trajectory shape: {trajectory.shape}")
    if turn_logits.shape != (batch_size, TURN_LOGIT_DIM):
        raise ValueError(f"unexpected new-DP turn-logit shape: {turn_logits.shape}")
    trajectory = trajectory.copy()
    trajectory[..., :2] *= 50.0
    yaw = trajectory[..., 2:4]
    trajectory[..., 2:4] = yaw / np.maximum(np.linalg.norm(yaw, axis=-1, keepdims=True), 1e-6)
    return trajectory, turn_logits


def legacy_feedback_turn_logits(turn_logits: np.ndarray) -> np.ndarray:
    """Adapt new-DP state logits for the legacy closed-loop feedback decoder.

    New DP predicts `[DISABLE, LEFT, RIGHT]`.  The legacy decoder additionally
    has `NONE` (0) and `KEEP` (4), which are not next-state predictions and
    must never be selected for a new-DP model.
    """
    turn_logits = np.asarray(turn_logits)
    if turn_logits.ndim != 2 or turn_logits.shape[1] != TURN_LOGIT_DIM:
        raise ValueError(f"unexpected new-DP turn-logit shape: {turn_logits.shape}")
    legacy = np.full((turn_logits.shape[0], 5), -1e9, dtype=np.float32)
    legacy[:, 1:4] = turn_logits
    return legacy


class NewDpOnnxRunner:
    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        self.session = ort.InferenceSession(
            model_path, providers=_providers_for_local_rank(providers)
        )
        actual = {item.name for item in self.session.get_inputs()}
        expected = set(MODEL_INPUT_NAMES) | {"initial_noise"}
        if actual != expected:
            raise ValueError(
                f"ONNX input mismatch; missing={expected - actual}, extra={actual - expected}"
            )

    def predict(
        self, frames: list[dict[str, np.ndarray]], seeds: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(frames) != len(seeds) or not frames:
            raise ValueError("frames and seeds must have the same non-zero length")
        normalized = [normalize_frame(frame) for frame in frames]
        feed = {
            key: np.stack([np.asarray(f[key], dtype=np.float32) for f in normalized])
            for key in MODEL_INPUT_NAMES
        }
        feed["initial_noise"] = seeded_initial_noise(seeds)
        trajectory, turn_logits = self.session.run(None, feed)
        return decode_onnx_outputs(trajectory, turn_logits, batch_size=len(frames))
