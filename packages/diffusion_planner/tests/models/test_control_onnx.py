"""ONNX export tests for the control-to-pose rollout.

The rollout is the one piece of the sampler that runs vendored numerical code, so
it gets its own export test: a Python-side shape baked into the trace produces a
graph that only accepts the batch size it was exported with, and that failure is
invisible until the ROS side runs the model at a different batch size.
"""

from __future__ import annotations

import io

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from diffusion_planner.data.dimensions import (
    CONTROL_DIM,
    EGO_HISTORY_LENGTH,
    EGO_STATE_DIM,
    EGO_VELOCITY_INDEX,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner


class EgoRollout(nn.Module):
    """Expose `ego_poses_from_control` as a standalone exportable graph."""

    def __init__(self, planner: DiffusionPlanner) -> None:
        super().__init__()
        self.planner = planner

    def forward(
        self, control: torch.Tensor, ego_agent_past: torch.Tensor
    ) -> torch.Tensor:
        """Roll control out into poses for one batch of ego histories."""
        return self.planner.ego_poses_from_control(
            control, {"ego_agent_past": ego_agent_past}
        )


def make_rollout_inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(control, ego_agent_past)` for a batch of moving ego vehicles."""
    generator = torch.Generator().manual_seed(0)
    control = (
        torch.randn(batch, TRAJECTORY_LENGTH, CONTROL_DIM, generator=generator) * 0.1
    )
    ego_past = torch.zeros(batch, EGO_HISTORY_LENGTH, EGO_STATE_DIM)
    ego_past[..., 2] = 1.0
    ego_past[..., EGO_VELOCITY_INDEX] = 6.0
    return control, ego_past


def export_rollout(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> bytes:
    """Export `module` with a dynamic batch axis and return the serialized model."""
    buffer = io.BytesIO()
    names = ("control", "ego_agent_past", "poses")
    torch.onnx.export(
        module,
        inputs,
        buffer,
        input_names=names[:2],
        output_names=names[2:],
        opset_version=20,
        dynamo=False,
        dynamic_axes={name: {0: "batch"} for name in names},
    )
    return buffer.getvalue()


class TestControlRolloutOnnx:
    def test_export_accepts_every_batch_size(self):
        planner = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            element_mixer_hidden_dim=8,
        ).eval()
        module = EgoRollout(planner).eval()
        control, ego_past = make_rollout_inputs(batch=3)

        session = ort.InferenceSession(
            export_rollout(module, (control, ego_past)),
            providers=["CPUExecutionProvider"],
        )

        for batch in (1, 2, 3):
            outputs = session.run(
                None,
                {
                    "control": control[:batch].numpy(),
                    "ego_agent_past": ego_past[:batch].numpy(),
                },
            )
            expected = module(control[:batch], ego_past[:batch]).detach().numpy()
            assert outputs[0].shape == (batch, TRAJECTORY_LENGTH, 4)
            np.testing.assert_allclose(outputs[0], expected, rtol=1e-4, atol=1e-5)
