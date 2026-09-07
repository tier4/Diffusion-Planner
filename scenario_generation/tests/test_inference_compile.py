"""Smoke tests for scenario_generation.inference_compile.

The compiled forward itself needs a GPU, so what is worth pinning here is the scoping: the
training loop hands over its live model, and a wrapper left behind would rename the decoder's
parameter keys in the next checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from scenario_generation.inference_compile import compiled_for_inference, mark_inference_step


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = nn.Linear(4, 4)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = _Decoder()


def test_model_and_fastpath_are_restored_on_exit():
    model = _Model()
    encoder, dit = model.encoder, model.decoder.dit
    fastpath = torch.backends.mha.get_fastpath_enabled()

    with compiled_for_inference(model):
        assert model.encoder is not encoder
        assert not torch.backends.mha.get_fastpath_enabled()

    assert model.encoder is encoder
    assert model.decoder.dit is dit
    assert torch.backends.mha.get_fastpath_enabled() == fastpath
    assert set(model.state_dict()) == set(_Model().state_dict())


def test_a_model_with_nothing_to_compile_passes_through():
    """The ONNX adapter runs its own runtime and has no encoder/dit to hand to dynamo."""
    model = object()
    with compiled_for_inference(model) as handed_back:
        assert handed_back is model


def test_mark_inference_step_is_safe_without_a_compiled_model():
    mark_inference_step()
