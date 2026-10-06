"""Drive an ML Planner sampler ONNX from scenario_sim.

Its inputs come from ``openscenario_python.MlPlannerObserver``, which runs ``autoware_ml_planner``'s
own preprocessing on what the simulator publishes, so they are built as on the vehicle.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from new_dp_h5_eval.model import decode_onnx_outputs


class MlPlannerOnnx:
    """A sampler ONNX, run with zero noise as the deployed node does."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(path, providers=providers)
        # ORT falls back to CPU without an error, which turns every case into a timeout.
        if (
            device.startswith("cuda")
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError(f"CUDAExecutionProvider did not load for {path}")
        self.shapes = {i.name: i.shape[1:] for i in self.session.get_inputs()}
        self._noise = np.zeros((1, *self.shapes.pop("initial_noise")), dtype=np.float32)

    def predict(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
        """Observer inputs -> (ego-frame plan ``[T, 4]`` of x, y, cos, sin; turn report 1/2/3)."""
        feed = {"initial_noise": self._noise}
        for name, shape in self.shapes.items():
            value = inputs[name]
            # The preprocessing's sizes are fixed when it is compiled, not read from the graph.
            if value.ndim != len(shape) or any(
                isinstance(d, int) and d != n for d, n in zip(shape, value.shape)
            ):
                raise ValueError(
                    f"{name}: the preprocessing built {value.shape}, the graph takes {shape}"
                )
            feed[name] = value[None]
        trajectory, logits = self.session.run(None, feed)
        trajectory, logits = decode_onnx_outputs(trajectory, logits, batch_size=1)
        return trajectory[0, 0], int(np.argmax(logits[0])) + 1
