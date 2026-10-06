"""Drive a planner ONNX from scenario_sim.

Its inputs come from the simulator's observer (``openscenario_python.MlPlannerObserver`` or
``DiffusionPlannerObserver``), which runs the node's own preprocessing on what the simulator
publishes, so they are built as on the vehicle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from new_dp_h5_eval.model import decode_onnx_outputs
from scenario_generation.simulate import decode_turn_indicator


def load_planner(path: str, device: str = "cuda") -> MlPlannerOnnx | DiffusionPlannerOnnx:
    """An ML Planner sampler takes its own noise; a Diffusion-Planner export does not."""
    if not path.endswith(".onnx"):
        raise ValueError(f"{path}: scenario_sim takes an .onnx model")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda")
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(path, providers=providers)
    # ORT falls back to CPU without an error, which turns every case into a timeout.
    if device.startswith("cuda") and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDAExecutionProvider did not load for {path}")
    if any(i.name == "initial_noise" for i in session.get_inputs()):
        return MlPlannerOnnx(session)
    return DiffusionPlannerOnnx(session, str(Path(path).parent / "args.json"))


class MlPlannerOnnx:
    """A sampler ONNX, run with zero noise as the deployed node does."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.shapes = {i.name: i.shape[1:] for i in self.session.get_inputs()}
        self._noise = np.zeros((1, *self.shapes.pop("initial_noise")), dtype=np.float32)

    def observer(self, osp, ego_ref: str, route_ids, goal_pose):
        return osp.MlPlannerObserver(ego_ref, route_ids, goal_pose)

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


class DiffusionPlannerOnnx:
    """A Diffusion-Planner export, normalized with the args.json beside it."""

    def __init__(self, session: ort.InferenceSession, args_path: str) -> None:
        self.session = session
        self.args_path = args_path

    def observer(self, osp, ego_ref: str, route_ids, goal_pose):
        return osp.DiffusionPlannerObserver(ego_ref, route_ids, goal_pose, self.args_path)

    def predict(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
        """Observer inputs -> (ego-frame plan ``[T, 4]`` of x, y, cos, sin; turn class 0-4)."""
        feed = {}
        for i in self.session.get_inputs():
            if i.type == "tensor(bool)":
                # The node's inference marks a lane's speed limit as present where it is positive.
                value = inputs[i.name.replace("_has_", "_")] > 0
            else:
                value = inputs[i.name]
            # The observer returns flat arrays; reshape raises if the compiled sizes differ.
            feed[i.name] = value.reshape(1, *i.shape[1:])
        prediction, logits = self.session.run(["prediction", "turn_indicator_logit"], feed)
        return prediction[0, 0], int(decode_turn_indicator(torch.from_numpy(logits[0])))
