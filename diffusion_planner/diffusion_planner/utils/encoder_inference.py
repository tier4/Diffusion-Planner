"""Encoder-only inference for latent OOD embedding extraction.

The Latent OOD Stage 1 tooling (``latent_ood.py``, ``build_latent_ood_bank.py``,
``score_latent_ood.py``) was originally written against a ``.pth`` checkpoint.
No ``.pth`` exists for the deployed x2 model — the training machine that
produced it is unreachable — but the full ONNX export is available at
``/opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx``.

That ONNX graph fuses encoder + decoder + sampler into ~20k nodes and its
weights are unnamed ``onnx::MatMul_XXXX`` initializers, so PyTorch weights
cannot be reconstructed from it. Instead, for ``.onnx`` inputs this module
taps the intermediate encoder-output tensor
(``/model/encoder/fusion/norm/LayerNormalization_output_0``) by adding it as
an extra graph output, and runs the (pruned) subgraph needed to produce it
via onnxruntime.

``EncoderInference`` also accepts a ``.pth`` checkpoint (should one become
available again from a future training run) and runs the equivalent PyTorch
encoder path, so callers don't need to know which backend is in use.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper

from diffusion_planner.utils.config import Config
from diffusion_planner.utils.onnx_export import ENCODER_INPUT_NAMES

ENCODER_OUTPUT_NAME = "/model/encoder/fusion/norm/LayerNormalization_output_0"


def _add_encoder_output(onnx_path: str) -> str:
    """Load the full ONNX model, add the encoder output as a graph output, save to a temp file.

    Returns the path to the modified model. onnxruntime only executes the
    subgraph needed to produce the requested output(s), so declaring this as
    an additional output (rather than replacing the graph outputs) is enough
    to run encoder-only inference without touching the decoder/sampler nodes.
    """
    model = onnx.load(onnx_path)

    existing_names = {out.name for out in model.graph.output}
    if ENCODER_OUTPUT_NAME not in existing_names:
        encoder_out = helper.make_tensor_value_info(ENCODER_OUTPUT_NAME, TensorProto.FLOAT, None)
        model.graph.output.append(encoder_out)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return tmp.name


def _load_torch_checkpoint(config_obj: Config, ckpt_path: str, device: str) -> torch.nn.Module:
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner

    model = Diffusion_Planner(config_obj)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    return model


class EncoderInference:
    """Extract L2-normalized scene embeddings from a Diffusion Planner model.

    Accepts either a ``.pth`` checkpoint (loaded as a regular PyTorch
    ``Diffusion_Planner`` and run through ``model.encoder``) or a ``.onnx``
    export (encoder subgraph run via onnxruntime). Callers use the same
    ``encode_batch`` interface either way.

    Usage:
        encoder = EncoderInference(args_path, model_path, device="cuda")
        z = encoder.encode_batch(batch)  # [B, hidden_dim], L2-normalized
    """

    def __init__(self, args_path: Path, model_path: Path, device: str = "cuda"):
        config_obj = Config(str(args_path))
        self.obs_norm = config_obj.observation_normalizer
        self.config = config_obj
        self.device = device

        model_path = Path(model_path)
        suffix = model_path.suffix.lower()
        if suffix == ".onnx":
            self._backend = "onnx"
            self._init_onnx(model_path)
        elif suffix == ".pth":
            self._backend = "torch"
            self._torch_model = _load_torch_checkpoint(config_obj, str(model_path), device)
        else:
            raise ValueError(
                f"Unsupported model_path extension '{suffix}' for {model_path} "
                "(expected .pth or .onnx)"
            )

    def _init_onnx(self, onnx_path: Path) -> None:
        modified_onnx = _add_encoder_output(str(onnx_path))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in self.device
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(modified_onnx, providers=providers)
        self._encoder_output_name = ENCODER_OUTPUT_NAME
        # ONNX declares some inputs (e.g. the speed-limit masks) as BOOL; feeding
        # them as float32 raises a type-mismatch error from onnxruntime, so look
        # up each input's declared dtype instead of assuming float everywhere.
        self._input_is_bool = {
            info.name: info.type == "tensor(bool)" for info in self._session.get_inputs()
        }
        # onnxruntime's InferenceSession.run() validates that every *declared*
        # graph input is fed, even ones (decoder/sampler-only, e.g.
        # `sampled_trajectories`, `ego_current_state`, `delay`) that aren't
        # ancestors of the requested encoder output. Precompute zero-filled
        # placeholders for those so encode_batch only has to reason about the
        # encoder's own inputs.
        self._unused_input_specs = [
            (info.name, info.shape)
            for info in self._session.get_inputs()
            if info.name not in ENCODER_INPUT_NAMES
        ]

    @staticmethod
    def _heading_to_cos_sin(x: torch.Tensor) -> torch.Tensor:
        """Convert (x, y, heading) to (x, y, cos(heading), sin(heading)). Idempotent on 4-col input."""
        if x.shape[-1] == 4:
            return x
        return torch.cat([x[..., :2], x[..., 2:3].cos(), x[..., 2:3].sin()], dim=-1)

    def encode_batch(self, batch: dict, normalize: bool = True) -> torch.Tensor:
        """Run the encoder on a batch, return [B, hidden_dim] embeddings.

        Args:
            normalize: If True (default), L2-normalize the output. Set False to
                get raw mean-pooled vectors for methods that need their own preprocessing.
        """
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        if "ego_agent_past" in batch:
            batch["ego_agent_past"] = self._heading_to_cos_sin(batch["ego_agent_past"])
        if "goal_pose" in batch:
            batch["goal_pose"] = self._heading_to_cos_sin(batch["goal_pose"])
        batch = self.obs_norm(batch)

        if self._backend == "onnx":
            encoding = self._encode_onnx(batch)
        else:
            with torch.no_grad():
                encoding = self._torch_model.encoder(batch)

        z = encoding.mean(dim=1)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z

    def _encode_onnx(self, batch: dict) -> torch.Tensor:
        ort_inputs = {}
        for name in ENCODER_INPUT_NAMES:
            val = batch[name]
            arr = val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
            if self._input_is_bool.get(name, False):
                arr = arr.astype(np.bool_)
            else:
                arr = arr.astype(np.float32)
            ort_inputs[name] = arr

        batch_size = ort_inputs[ENCODER_INPUT_NAMES[0]].shape[0]
        for name, shape in self._unused_input_specs:
            concrete_shape = [batch_size if isinstance(d, str) else d for d in shape]
            dtype = np.bool_ if self._input_is_bool.get(name, False) else np.float32
            ort_inputs[name] = np.zeros(concrete_shape, dtype=dtype)

        ort_outputs = self._session.run([self._encoder_output_name], ort_inputs)
        return torch.from_numpy(ort_outputs[0]).to(self.device)
