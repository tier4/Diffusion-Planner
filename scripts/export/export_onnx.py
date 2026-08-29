#!/usr/bin/env python3
"""Export a training checkpoint into scene-encoder and trajectory-decoder ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from diffusion_planner.data import (
    FillUnknownTrafficLightFutures,
    PlannerDataNormalizer,
    PlannerDataset,
)
from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.onnx import (
    SCENE_INPUT_NAMES,
    DiffusionPlannerSamplerOnnxWrapper,
    SceneEncoderOnnxWrapper,
    TrajectoryDecoderOnnxWrapper,
)
from diffusion_planner.utils.checkpoint import load_model

SCENE_OUTPUT_NAMES = ("scene", "scene_mask", "agent_pose", "agent_mask")
DECODER_INPUT_NAMES = (
    "x",
    "x_mask",
    "scene",
    "scene_mask",
    "agent_pose",
    "time",
)
SAMPLER_INPUT_NAMES = ("initial_noise", *SCENE_INPUT_NAMES, "turn_indicators")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--parquet-path",
        type=Path,
        required=True,
        help="Training dataset Parquet index used to create representative inputs.",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--opset-version", type=int, default=20)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _load_frame(parquet_path: Path, frame_index: int) -> dict[str, torch.Tensor]:
    dataset = PlannerDataset(
        parquet_path,
        file_capacity=1,
        transforms=[FillUnknownTrafficLightFutures(), PlannerDataNormalizer()],
    )
    try:
        if not 0 <= frame_index < len(dataset):
            raise IndexError(
                f"frame-index {frame_index} is outside dataset with {len(dataset)} rows"
            )
        return dataset[frame_index]
    finally:
        dataset.close()


def _scene_inputs(frame: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(
        frame[name].unsqueeze(0).repeat((2,) + (1,) * frame[name].ndim)
        for name in SCENE_INPUT_NAMES
    )


def _export(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    opset_version: int,
) -> None:
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=False,
        dynamic_axes={name: {0: "batch"} for name in (*input_names, *output_names)},
        external_data=False,
        optimize=True,
    )
    print(f"exported: {path}")


def _ort_outputs(path: Path, inputs: dict[str, torch.Tensor]) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        None,
        {name: value.detach().cpu().numpy() for name, value in inputs.items()},
    )
    return [np.asarray(output) for output in outputs]


def _validate(
    path: Path,
    input_names: tuple[str, ...],
    inputs: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    for batch_size in (1, 2):
        batch_inputs = tuple(value[:batch_size] for value in inputs)
        actual = _ort_outputs(path, dict(zip(input_names, batch_inputs, strict=True)))
        for index, (torch_value, ort_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            np.testing.assert_allclose(
                ort_value,
                torch_value[:batch_size].detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=(
                    f"ONNX output {index} differs from PyTorch at batch "
                    f"size {batch_size}"
                ),
            )
    print(f"validated: {path}")


def _validate_all(
    validations: list[
        tuple[
            Path,
            tuple[str, ...],
            tuple[torch.Tensor, ...],
            tuple[torch.Tensor, ...],
        ]
    ],
) -> None:
    """Run every ONNX validation before reporting collected failures."""
    failures: list[tuple[Path, Exception]] = []
    for path, input_names, inputs, expected in validations:
        try:
            _validate(path, input_names, inputs, expected)
        except Exception as error:
            failures.append((path, error))
            print(f"validation failed: {path}\n{error}")

    if failures:
        failed_paths = ", ".join(str(path) for path, _ in failures)
        raise RuntimeError(
            f"{len(failures)} ONNX validation(s) failed after all validations ran: "
            f"{failed_paths}"
        ) from failures[0][1]


def main() -> None:
    args = _parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model = load_model(checkpoint_path, DiffusionPlanner).eval()
    frame = _load_frame(args.parquet_path, args.frame_index)
    scene_inputs = _scene_inputs(frame)
    scene_wrapper = SceneEncoderOnnxWrapper(model.scene_encoder).eval()
    with torch.no_grad():
        scene_outputs = scene_wrapper(*scene_inputs)

    scene, scene_mask, agent_pose, agent_mask = scene_outputs
    batch_size, agents = agent_mask.shape
    decoder_inputs = (
        torch.randn(batch_size, agents, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
        agent_mask,
        scene,
        scene_mask,
        agent_pose,
        torch.full((batch_size,), 0.5),
    )
    decoder_wrapper = TrajectoryDecoderOnnxWrapper(model.trajectory_decoder).eval()
    with torch.no_grad():
        decoder_output = decoder_wrapper(*decoder_inputs)
    turn_indicators = frame["turn_indicators"].unsqueeze(0).repeat(batch_size, 1)
    sampler_inputs = (decoder_inputs[0], *scene_inputs, turn_indicators)
    sampler_wrapper = DiffusionPlannerSamplerOnnxWrapper(model).eval()
    with torch.no_grad():
        sampler_output = sampler_wrapper(*sampler_inputs)

    output_dir = checkpoint_path.parent / "onnx" / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scene_encoder.onnx"
    decoder_path = output_dir / "trajectory_decoder.onnx"
    sampler_path = output_dir / "diffusion_planner_sampler.onnx"
    _export(
        scene_wrapper,
        scene_inputs,
        scene_path,
        SCENE_INPUT_NAMES,
        SCENE_OUTPUT_NAMES,
        args.opset_version,
    )
    _export(
        decoder_wrapper,
        decoder_inputs,
        decoder_path,
        DECODER_INPUT_NAMES,
        ("x0_prediction",),
        args.opset_version,
    )
    _export(
        sampler_wrapper,
        sampler_inputs,
        sampler_path,
        SAMPLER_INPUT_NAMES,
        ("trajectory", "turn_indicator_logits"),
        args.opset_version,
    )
    if not args.skip_validation:
        _validate_all(
            [
                (scene_path, SCENE_INPUT_NAMES, scene_inputs, scene_outputs),
                (
                    decoder_path,
                    DECODER_INPUT_NAMES,
                    decoder_inputs,
                    (decoder_output,),
                ),
                (
                    sampler_path,
                    SAMPLER_INPUT_NAMES,
                    sampler_inputs,
                    (sampler_output,),
                ),
            ]
        )


if __name__ == "__main__":
    main()
