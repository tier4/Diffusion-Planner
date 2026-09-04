"""An ONNX session that fell back to CPU must say so, not run unaccelerated in silence."""

from __future__ import annotations

import sys
import types

import pytest

from scenario_generation.simulate import CPU_EP, CUDA_EP, TENSORRT_EP, _require_accelerator


class _Session:
    def __init__(self, active):
        self._active = active

    def get_providers(self):
        return self._active


def test_a_gpu_request_that_landed_on_cpu_raises():
    """ORT lists providers the build contains, not the ones that loaded; it warns and continues."""
    session = _Session([CPU_EP])

    with pytest.raises(RuntimeError, match="runs on"):
        _require_accelerator(session, [TENSORRT_EP, CUDA_EP, CPU_EP], "m.onnx")


def test_the_message_names_the_way_out():
    session = _Session([CPU_EP])

    with pytest.raises(RuntimeError, match=r"providers=\['CPUExecutionProvider'\]"):
        _require_accelerator(session, [CUDA_EP, CPU_EP], "m.onnx")


@pytest.mark.parametrize("active", [[TENSORRT_EP, CPU_EP], [CUDA_EP, CPU_EP]])
def test_either_accelerator_satisfies_the_request(active):
    """TensorRT partitions the graph and leaves the rest to CPU, so CPU being present is normal."""
    _require_accelerator(_Session(active), [TENSORRT_EP, CUDA_EP, CPU_EP], "m.onnx")


def test_asking_for_cpu_is_not_a_failure():
    _require_accelerator(_Session([CPU_EP]), [CPU_EP], "m.onnx")


def test_the_default_does_not_reach_for_tensorrt(monkeypatch):
    """TensorRT partitions the graph up front and refuses ops it cannot build, so defaulting to
    it turns a model it dislikes into a session that never opens. It has to be asked for."""
    seen = {}

    class _Session:
        def __init__(self, path, providers=None, provider_options=None):
            seen["providers"] = providers

        def get_providers(self):
            return [CUDA_EP, CPU_EP]

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    import scenario_generation.simulate as simulate

    monkeypatch.setitem(
        sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=_Session)
    )
    simulate._OnnxModel("m.onnx", "cuda")

    assert seen["providers"] == [CUDA_EP, CPU_EP]


def test_asking_for_cpu_by_device_does_not_request_a_gpu_provider(monkeypatch):
    seen = {}

    class _Session:
        def __init__(self, path, providers=None, provider_options=None):
            seen["providers"] = providers

        def get_providers(self):
            return [CPU_EP]

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    import scenario_generation.simulate as simulate

    monkeypatch.setitem(
        sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=_Session)
    )
    simulate._OnnxModel("m.onnx", "cpu")

    assert seen["providers"] == [CPU_EP]
