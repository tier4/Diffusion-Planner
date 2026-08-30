"""Run planner sampling for one dashboard frame."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray

from diffusion_planner.data import PlannerDataNormalizer
from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.onnx import PLANNER_INPUT_NAMES


def run_inference(
    model: DiffusionPlanner,
    frame_data: Mapping[str, Any],
    *,
    device: str,
    num_steps: int,
    time_epsilon: float,
    noise_scale: float,
    seed: int,
) -> tuple[NDArray[np.float32], float]:
    """Sample one trajectory batch and return the prediction and elapsed seconds."""
    torch_device = torch.device(device)
    normalizer = PlannerDataNormalizer()
    normalized_frame = normalizer(
        {key: np.asarray(value) for key, value in frame_data.items()}
    )
    input_data = {
        key: torch.as_tensor(value, device=torch_device).unsqueeze(0)
        for key, value in normalized_frame.items()
    }
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    neighbor_count = normalized_frame["neighbor_agents_past"].shape[0]
    initial_noise = noise_scale * torch.randn(
        (1, neighbor_count + 1, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
        device=torch_device,
        dtype=torch.float32,
        generator=generator,
    )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    start = perf_counter()
    with torch.inference_mode():
        prediction, _ = model.sample(
            input_data,
            initial_noise=initial_noise,
            num_steps=num_steps,
            time_epsilon=time_epsilon,
        )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = perf_counter() - start
    prediction_array = prediction[0].detach().float().cpu().numpy()
    prediction_array = normalizer.denormalize_trajectory(prediction_array)
    return prediction_array.astype(np.float32, copy=False), elapsed


def run_onnx_inference(
    session: ort.InferenceSession,
    frame_data: Mapping[str, Any],
    *,
    noise_scale: float,
    seed: int,
) -> tuple[NDArray[np.float32], float]:
    """Run a fixed-step sampler ONNX for one dashboard frame."""
    normalizer = PlannerDataNormalizer()
    normalized_frame = normalizer(
        {key: np.asarray(value) for key, value in frame_data.items()}
    )
    neighbor_count = normalized_frame["neighbor_agents_past"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    initial_noise = torch.randn(
        (1, neighbor_count + 1, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
        generator=generator,
        dtype=torch.float32,
    ).numpy() * np.float32(noise_scale)
    available_inputs = {value.name for value in session.get_inputs()}
    inputs = {
        name: np.asarray(normalized_frame[name], dtype=np.float32)[None]
        for name in PLANNER_INPUT_NAMES
        if name in available_inputs
    }
    inputs["initial_noise"] = initial_noise

    start = perf_counter()
    prediction = session.run(None, inputs)[0]
    assert isinstance(prediction, np.ndarray), "ONNX prediction must be an ndarray"
    elapsed = perf_counter() - start
    prediction_array = normalizer.denormalize_trajectory(np.asarray(prediction[0]))
    return prediction_array.astype(np.float32, copy=False), elapsed


def run_turn_indicator_inference(
    model: DiffusionPlanner,
    frame_data: Mapping[str, Any],
    *,
    device: str,
) -> tuple[NDArray[np.float32], int, float]:
    """Predict the next turn indicator and return probabilities and elapsed seconds."""
    torch_device = torch.device(device)
    normalized_frame = PlannerDataNormalizer()(
        {key: np.asarray(value) for key, value in frame_data.items()}
    )
    input_data = {
        key: torch.as_tensor(value, device=torch_device).unsqueeze(0)
        for key, value in normalized_frame.items()
    }
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    start = perf_counter()
    with torch.inference_mode():
        logits = model.predict_turn_indicator(input_data)
        probabilities = torch.softmax(logits, dim=-1)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = perf_counter() - start
    probability_array = probabilities[0].detach().float().cpu().numpy()
    predicted_report = int(probability_array.argmax()) + 1
    return probability_array.astype(np.float32, copy=False), predicted_report, elapsed
