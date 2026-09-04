"""``torch.compile`` for the closed-loop evaluation forward pass.

``nn.MultiheadAttention`` takes a fused fastpath in eval mode that dynamo cannot trace, so a
compiled model always runs the decomposed path instead. Both are correct; they round differently
in the last float32 bit, and a closed loop feeds that difference back into its own next input.
Closed-loop metrics therefore shift once. No backend choice avoids this -- it is structural.

``cudagraphs`` replays the same kernels in the same order and was the faster of the two
candidates on a real rollout. Inductor generates its own reductions and was no faster.
"""

from __future__ import annotations

import contextlib
import logging

import torch

logger = logging.getLogger(__name__)


class _CloneEncoderOutput(torch.nn.Module):
    """Clone the encoder's output so the DiT's next cudagraph replay cannot overwrite it.

    The encoding is computed once per inference and read by every DPM-solver step, but a
    cudagraph-managed output buffer belongs to the graph that produced it and is reused on the
    next replay.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs).clone()


@contextlib.contextmanager
def compiled_for_inference(model):
    """Compile the encoder and the DiT for the duration, then put the model back as it was.

    Scoped rather than permanent because the training loop hands over its LIVE model: compiling
    in place would leave ``model.encoder`` wrapped and rename the decoder's parameter keys
    (``_orig_mod.`` prefixes), which would then land in the next checkpoint.

    The fastpath switch is entered here rather than left to the caller because compilation is
    lazy and the flag is not in dynamo's guard set: setting it afterwards would leave the graph
    traced under the old value while reading back as changed.

    Callers must run inference under ``no_grad`` -- an output carrying an autograd graph drops
    compile off the CUDA-graph fast path silently -- and call :func:`mark_inference_step` once
    per inference.
    """
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    dit = getattr(decoder, "dit", None)
    if not isinstance(encoder, torch.nn.Module) or not isinstance(dit, torch.nn.Module):
        # e.g. the ONNX adapter from simulate.load_onnx_model, which runs its own runtime.
        logger.info("%s has nothing to compile; running it as-is.", type(model).__name__)
        yield model
        return

    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    model.encoder = _CloneEncoderOutput(torch.compile(encoder, backend="cudagraphs"))
    decoder.dit = torch.compile(dit, backend="cudagraphs")
    try:
        yield model
    finally:
        model.encoder = encoder
        decoder.dit = dit
        torch.backends.mha.set_fastpath_enabled(fastpath)


def mark_inference_step() -> None:
    """Open a new cudagraph step.

    Unconditional on purpose: the rollout should not have to know whether the model it was handed
    is compiled, and skipping it on a compiled model corrupts the replayed buffers.
    """
    torch.compiler.cudagraph_mark_step_begin()
