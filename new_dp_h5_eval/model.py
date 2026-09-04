"""ONNX inference boundary for the new planner."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from .schema import MODEL_INPUT_NAMES


def normalize_frame(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(frame)
    for key in ("ego_agent_past", "neighbor_agents_past", "ego_agent_future",
                "neighbor_agents_future", "goal_pose"):
        if key in frame:
            out[key] = frame[key].copy(); out[key][..., :2] /= 50.0
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


class NewDpOnnxRunner:
    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        self.session = ort.InferenceSession(model_path, providers=providers)
        actual = {item.name for item in self.session.get_inputs()}
        expected = set(MODEL_INPUT_NAMES) | {"initial_noise"}
        if actual != expected:
            raise ValueError(f"ONNX input mismatch; missing={expected-actual}, extra={actual-expected}")

    def predict(self, frames: list[dict[str, np.ndarray]], seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if len(frames) != len(seeds) or not frames:
            raise ValueError("frames and seeds must have the same non-zero length")
        normalized = [normalize_frame(frame) for frame in frames]
        feed = {key: np.stack([np.asarray(f[key], dtype=np.float32) for f in normalized])
                for key in MODEL_INPUT_NAMES}
        rng_noise = [torch.randn((321, 80, 4), generator=torch.Generator().manual_seed(seed)).numpy()
                     for seed in seeds]
        feed["initial_noise"] = np.stack(rng_noise)
        trajectory, turn_logits = self.session.run(None, feed)
        if np.asarray(trajectory).shape != (len(frames), 321, 80, 4):
            raise ValueError(f"unexpected ONNX trajectory shape: {np.asarray(trajectory).shape}")
        if np.asarray(turn_logits).shape != (len(frames), 3):
            raise ValueError(f"unexpected new-DP turn-logit shape: {np.asarray(turn_logits).shape}")
        trajectory = np.asarray(trajectory); trajectory[..., :2] *= 50.0
        yaw = trajectory[..., 2:4]
        trajectory[..., 2:4] = yaw / np.maximum(np.linalg.norm(yaw, axis=-1, keepdims=True), 1e-6)
        return trajectory, np.asarray(turn_logits)
