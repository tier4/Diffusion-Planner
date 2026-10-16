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
from new_dp_h5_eval.schema import EgoPastLayout
from new_dp_h5_eval.vehicle import kinematic_steering
from scenario_generation.simulate import decode_turn_indicator

# The vehicle a model that reads the wheel base is evaluated for, as the planner node's launch
# gives it: beside the ONNX, like a Diffusion-Planner export's args.json.
VEHICLE_INFO = "vehicle_info.param.yaml"


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
        return MlPlannerOnnx(session, Path(path).parent / VEHICLE_INFO)
    return DiffusionPlannerOnnx(session, str(Path(path).parent / "args.json"))


def vehicle_spec(vehicle_info: Path) -> list[float]:
    """``[base_link_to_front, length, width, wheel_base]`` of a ``vehicle_info.param.yaml``,
    composed as the planner node composes its ``VehicleSpec``."""
    import yaml  # noqa: PLC0415

    ((_, node),) = yaml.safe_load(vehicle_info.read_text()).items()
    v = {k: float(x) for k, x in node["ros__parameters"].items()}
    return [
        v["wheel_base"] + v["front_overhang"],
        v["front_overhang"] + v["wheel_base"] + v["rear_overhang"],
        v["left_overhang"] + v["wheel_tread"] + v["right_overhang"],
        v["wheel_base"],
    ]


class Observer:
    """The simulator's observer, also given what the vehicle reports besides its messages.

    The headless ego has no steering state, so a model that reads steering gets the kinematic
    angle of the ego's turn, as the native-H5 closed loop gives a placed ego.
    """

    def __init__(self, observer, wheel_base: float | None = None) -> None:
        self._observer = observer
        self._wheel_base = wheel_base
        self._steering = 0.0

    def observe(self, turn_indicator_report: int, ego_state: dict) -> None:
        if self._wheel_base is None:
            self._observer.observe(turn_indicator_report)
            return
        twist = ego_state["twist"]
        self._steering = kinematic_steering(
            self._wheel_base, float(twist["linear_x"]), float(twist["angular_z"]), self._steering
        )
        self._observer.observe(turn_indicator_report, self._steering)

    def inputs(self) -> dict[str, np.ndarray]:
        return self._observer.inputs()


class MlPlannerOnnx:
    """A sampler ONNX, run with zero noise as the deployed node does."""

    def __init__(self, session: ort.InferenceSession, vehicle_info: Path | None = None) -> None:
        self.session = session
        self.shapes = {i.name: i.shape[1:] for i in self.session.get_inputs()}
        self._noise = np.zeros((1, *self.shapes.pop("initial_noise")), dtype=np.float32)
        self.layout = EgoPastLayout.of(int(self.shapes["ego_agent_past"][-1]))
        # A graph that reads steering also reads the wheel base, which the simulator's vehicle
        # parameters need not carry for the vehicle the model is evaluated for.
        self.vehicle_spec = None
        if self.layout.steering is not None:
            if vehicle_info is None or not vehicle_info.is_file():
                raise FileNotFoundError(
                    f"{vehicle_info}: a model that reads the wheel base needs the evaluated "
                    "vehicle's vehicle_info.param.yaml beside it"
                )
            self.vehicle_spec = vehicle_spec(vehicle_info)

    def check_simulator(self, osp) -> None:
        """Refuse a simulator whose ML Planner preprocessing builds another ego history width."""
        compiled = getattr(osp, "ML_PLANNER_EGO_HISTORY_DIM", EgoPastLayout.of(6).width)
        if compiled != self.layout.width:
            raise ValueError(
                f"the simulator's ML Planner preprocessing builds a {compiled}-column ego history,"
                f" the graph takes {self.layout.width}; use the overlay built for this model"
            )

    def observer(self, osp, ego_ref: str, route_ids, goal_pose) -> Observer:
        # Before any step, so an overlay built for the other interface stops the run at once.
        self.check_simulator(osp)
        if self.vehicle_spec is None:
            return Observer(osp.MlPlannerObserver(ego_ref, route_ids, goal_pose))
        return Observer(
            osp.MlPlannerObserver(ego_ref, route_ids, goal_pose, self.vehicle_spec),
            self.vehicle_spec[3],
        )

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

    def observer(self, osp, ego_ref: str, route_ids, goal_pose) -> Observer:
        return Observer(osp.DiffusionPlannerObserver(ego_ref, route_ids, goal_pose, self.args_path))

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
