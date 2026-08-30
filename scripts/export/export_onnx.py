#!/usr/bin/env python3
"""Export a training checkpoint as a complete planner sampler ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from diffusion_planner.data.dimensions import (
    MAX_NUM_NEIGHBORS,
    PLANNER_INPUT_SHAPES,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.onnx import (
    PLANNER_INPUT_NAMES,
    DiffusionPlannerOnnxWrapper,
)
from diffusion_planner.utils.checkpoint import load_model

SAMPLER_INPUT_NAMES = ("initial_noise", *PLANNER_INPUT_NAMES)
VALIDATION_FRAME_PATH = Path(__file__).resolve().parent / "fixtures" / "frame.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opset-version", type=int, default=20)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _load_validation_inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    """Load the repository fixture and replicate its frame along the batch axis."""
    with np.load(VALIDATION_FRAME_PATH) as fixture:
        missing = set(PLANNER_INPUT_NAMES).difference(fixture.files)
        if missing:
            raise ValueError(
                f"Validation fixture is missing tensors: {', '.join(sorted(missing))}"
            )
        inputs: list[torch.Tensor] = []
        for name in PLANNER_INPUT_NAMES:
            array = np.asarray(fixture[name], dtype=np.float32)
            expected_shape = PLANNER_INPUT_SHAPES[name]
            if array.shape != expected_shape:
                raise ValueError(
                    f"Validation tensor {name!r} has shape {array.shape}, "
                    f"expected {expected_shape}"
                )
            inputs.append(
                torch.from_numpy(array)
                .unsqueeze(0)
                .repeat((batch_size,) + (1,) * array.ndim)
            )
        return tuple(inputs)


def _make_onnx_inputs(
    *, batch_size: int = 2, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """Create every export input randomly from the canonical shape definitions."""
    generator = torch.Generator().manual_seed(seed)
    initial_noise = torch.randn(
        batch_size,
        MAX_NUM_NEIGHBORS + 1,
        TRAJECTORY_LENGTH,
        TRAJECTORY_DIM,
        generator=generator,
    )
    planner_inputs = tuple(
        torch.randn(batch_size, *PLANNER_INPUT_SHAPES[name], generator=generator)
        for name in PLANNER_INPUT_NAMES
    )
    return (initial_noise, *planner_inputs)


def _make_validation_inputs(
    planner_inputs: tuple[torch.Tensor, ...], *, seed: int = 1
) -> tuple[torch.Tensor, ...]:
    """Combine the real validation frame with deterministic random noise."""
    batch_size = planner_inputs[0].shape[0]
    generator = torch.Generator().manual_seed(seed)
    initial_noise = torch.randn(
        batch_size,
        MAX_NUM_NEIGHBORS + 1,
        TRAJECTORY_LENGTH,
        TRAJECTORY_DIM,
        generator=generator,
    )
    return (initial_noise, *planner_inputs)


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


def main() -> None:
    args = _parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model = load_model(checkpoint_path, DiffusionPlanner).eval()
    export_inputs = _make_onnx_inputs()
    sampler_wrapper = DiffusionPlannerOnnxWrapper(model).eval()

    output_dir = checkpoint_path.parent / "onnx" / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_path = output_dir / "ml_planner.onnx"
    _export(
        sampler_wrapper,
        export_inputs,
        sampler_path,
        SAMPLER_INPUT_NAMES,
        ("trajectory", "turn_indicator_logits"),
        args.opset_version,
    )
    if not args.skip_validation:
        validation_planner_inputs = _load_validation_inputs()
        validation_inputs = _make_validation_inputs(validation_planner_inputs)
        with torch.no_grad():
            expected_outputs = sampler_wrapper(*validation_inputs)
        _validate(
            sampler_path,
            SAMPLER_INPUT_NAMES,
            validation_inputs,
            expected_outputs,
        )


if __name__ == "__main__":
    main()
