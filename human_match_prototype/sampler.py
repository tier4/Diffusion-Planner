"""Draw N candidate futures per NPZ frame from the deployed full ONNX model,
capturing the encoder scene embedding in the same session run."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.encoder_inference import (
    ENCODER_OUTPUT_NAME,
    _add_encoder_output,
)

FLOAT_KEYS = [
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "polygons",
    "line_strings",
    "turn_indicators",
]
BOOL_KEYS = ["lanes_has_speed_limit", "route_lanes_has_speed_limit"]
DEFAULT_EGO_SHAPE = [2.75, 4.34, 1.70]  # wheelbase, length, width (x2)

# --- Normalization / noise convention (validated by the Step-2 smoke gates) ---
# The deployed ONNX model does NOT normalize internally, so scene inputs are
# normalized externally via Config(args.json).observation_normalizer, exactly
# as the prior latent-OOD work validated. The diffusion noise fed into
# `sampled_trajectories` is 0.5 * randn and is injected AFTER normalization
# (the normalizer does not touch that key), matching ros_scripts/torch2onnx.py's
# 0.5-scaled Gaussian. This convention passed all smoke plausibility gates; see
# .superpowers/sdd/task-4-report.md for the recorded gate results.


def _heading_to_cos_sin(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 4:
        return x
    return torch.cat([x[..., :2], x[..., 2:3].cos(), x[..., 2:3].sin()], dim=-1)


@dataclass
class SampleResult:
    ego_samples: np.ndarray  # (N, 80, 3) [x, y, yaw]
    embedding: np.ndarray  # (256,) L2-normalized
    human_future: np.ndarray  # (80, 3) [x, y, yaw]


class TrajectorySampler:
    def __init__(self, args_path: str, onnx_path: str, device: str = "cuda"):
        self.config = Config(str(args_path))
        self.obs_norm = self.config.observation_normalizer
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            _add_encoder_output(str(onnx_path)), providers=providers
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _build_batch(self, data, num_samples: int, seed: int, temperature: float = 0.5) -> dict:
        g = torch.Generator().manual_seed(seed)
        n_agents = data["neighbor_agents_past"].shape[0] + 1  # 320 + ego = 321
        inputs = {
            "sampled_trajectories": temperature
            * torch.randn(num_samples, n_agents, 81, 4, generator=g, dtype=torch.float32)
        }
        inputs["ego_agent_past"] = _heading_to_cos_sin(
            torch.tensor(np.asarray(data["ego_agent_past"]), dtype=torch.float32)
        ).unsqueeze(0)
        for key in FLOAT_KEYS:
            if key == "turn_indicators" and key not in data.files:
                # Some NPZ frames lack turn_indicators; fall back to zeros(31).
                val = np.zeros(31, dtype=np.float32)
            else:
                val = np.asarray(data[key], dtype=np.float32)
            inputs[key] = torch.tensor(val).unsqueeze(0)
        for key in BOOL_KEYS:
            inputs[key] = torch.tensor(np.asarray(data[key])).bool().unsqueeze(0)
        goal = torch.tensor(np.asarray(data["goal_pose"], dtype=np.float32)).unsqueeze(0)
        inputs["goal_pose"] = _heading_to_cos_sin(goal)
        ego_shape = (
            np.asarray(data["ego_shape"], dtype=np.float32)
            if "ego_shape" in data.files
            else np.asarray(DEFAULT_EGO_SHAPE, dtype=np.float32)
        )
        inputs["ego_shape"] = torch.tensor(ego_shape).reshape(1, 3)
        # Tile every batch=1 tensor to num_samples (noise already has N rows)
        for key, val in inputs.items():
            if key != "sampled_trajectories":
                inputs[key] = val.repeat(num_samples, *([1] * (val.dim() - 1)))
        inputs = self.obs_norm(inputs)
        ort_inputs = {}
        for key, val in inputs.items():
            arr = val.cpu().numpy()
            ort_inputs[key] = arr.astype(np.bool_) if key in BOOL_KEYS else arr.astype(np.float32)
        ort_inputs["delay"] = np.zeros((1, 1), dtype=np.float32)
        missing = self._input_names - set(ort_inputs)
        if missing:
            raise KeyError(f"ONNX inputs not built: {missing}")
        return ort_inputs

    def sample(
        self, npz_path: str | Path, num_samples: int = 64, seed: int = 0, temperature: float = 0.5
    ) -> SampleResult:
        data = np.load(npz_path, allow_pickle=True)
        ort_inputs = self._build_batch(data, num_samples, seed, temperature)
        pred, enc = self.session.run(["prediction", ENCODER_OUTPUT_NAME], ort_inputs)
        ego = pred[:, 0]  # (N, 80, 4) denormalized ego-frame [x, y, cos, sin]
        yaw = np.arctan2(ego[..., 3], ego[..., 2])[..., None]
        ego_samples = np.concatenate([ego[..., :2], yaw], axis=-1)
        z = torch.from_numpy(enc).mean(dim=1)
        z = F.normalize(z, dim=-1)[0].numpy()  # identical scene per row; take row 0
        human = np.asarray(data["ego_agent_future"], dtype=np.float32)
        return SampleResult(
            ego_samples=ego_samples.astype(np.float32),
            embedding=z.astype(np.float32),
            human_future=human,
        )
