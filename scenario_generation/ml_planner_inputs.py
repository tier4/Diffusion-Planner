"""Drive an ML Planner sampler ONNX from scenario_sim.

The planner was trained on what Autoware's ``autoware_ml_planner`` node feeds it, so each input is
built the way that node builds it: the map selection and resampling come from the simulator
(``openscenario_python.MapFeatures``, which follows the node's preprocessing), and the histories
are kept here on the node's 0.1 s grid, with what the node sees right after it starts -- no
observation yet -- where the rollout has none.

Which tensors to build, and how large, is read off the ONNX graph, so a model whose inputs differ
fails at load with the names it asks for instead of being fed a stale layout.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from new_dp_h5_eval.model import decode_onnx_outputs, normalize_frame
from scenario_generation.scenario_sim_scene import baselink_xyh, centroid_xyh

DT = 0.1
# The node treats a signal it has not heard for longer than this as unknown.
TRAFFIC_LIGHT_TIMEOUT_S = 0.2
_TL_TIMEOUT_TICKS = int(round(TRAFFIC_LIGHT_TIMEOUT_S / DT))
# Amber turns red this many steps after it came on, when the future is inferred.
_AMBER_STEPS = 30
_MOVING_SPEED_MPS = 0.2
_MAX_YAW_RATE = 0.95
_MASK_RANGE_M = 100.0

# TrafficLightElement values.
_TL_RED, _TL_AMBER, _TL_GREEN, _TL_WHITE = 1, 2, 3, 4
_TL_CIRCLE, _TL_LEFT, _TL_RIGHT, _TL_UP, _TL_DOWN, _TL_DOWN_LEFT, _TL_DOWN_RIGHT = (
    1,
    2,
    3,
    4,
    7,
    8,
    9,
)
_ARROWS = {_TL_LEFT, _TL_RIGHT, _TL_UP, _TL_DOWN, _TL_DOWN_LEFT, _TL_DOWN_RIGHT}
# Channels of the signal one-hot: green, amber, red, unknown, white-or-no-light, arrow.
_CH_GREEN, _CH_AMBER, _CH_RED, _CH_UNKNOWN, _CH_NONE, _CH_ARROW = range(6)
_TURN_TO_SHAPE = {0: _TL_UP, 1: _TL_LEFT, 2: _TL_RIGHT}

# simulator (type, subtype) -> [vehicle, pedestrian, bicycle]; anything else is not an agent.
_VEHICLE_SUBTYPES = {1, 2, 3, 4, 5}  # car, truck, bus, trailer, motorcycle
_SUBTYPE_BICYCLE = 6


def _agent_label(state: dict) -> int | None:
    sim_type, subtype = int(state["type"]), int(state.get("subtype", 0))
    if sim_type == 2 or subtype == 7:
        return 1
    if sim_type == 1:
        if subtype in _VEHICLE_SUBTYPES:
            return 0
        if subtype == _SUBTYPE_BICYCLE:
            return 2
    return None


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    z: float
    yaw: float

    def to_ego(self, xy: np.ndarray, frame: "Pose2") -> np.ndarray:
        c, s = math.cos(frame.yaw), math.sin(frame.yaw)
        d = np.asarray(xy, dtype=np.float64) - (frame.x, frame.y)
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    def relative(self, frame: "Pose2") -> tuple[float, float, float, float]:
        x, y = self.to_ego(np.array([self.x, self.y]), frame)
        dyaw = self.yaw - frame.yaw
        return float(x), float(y), math.cos(dyaw), math.sin(dyaw)

    def as_pose7(self) -> list[float]:
        return [self.x, self.y, self.z, 0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2)]


def _pose_of(state: dict, *, centroid: bool) -> Pose2:
    x, y, yaw = centroid_xyh(state) if centroid else baselink_xyh(state)
    return Pose2(x, y, float(state["pose"]["z"]), yaw)


def _light_status(turn_direction: int, elements: list) -> tuple[int, bool]:
    """(color, is_arrow) of the bulb that governs a lane: the node's own choice among bulbs."""
    lit = [e for e in elements if e[0] != 0]
    if not lit:
        return 0, False
    if len(lit) > 1:
        target = _TURN_TO_SHAPE.get(turn_direction, 0)
        lit = (
            [e for e in lit if e[1] == target] or [e for e in lit if e[1] == _TL_CIRCLE] or lit
        )
    color, shape, _status, _conf = max(lit, key=lambda e: e[3])
    return color, shape in _ARROWS


def infer_traffic_light_future(past: np.ndarray, steps: int) -> np.ndarray:
    """Hold the current signal, except that amber turns red once it has been on for 30 steps."""
    n = past.shape[0]
    future = np.zeros((n, steps, past.shape[2]), dtype=np.float32)
    current = past[:, -1]
    for i in range(n):
        if not current[i].any():
            continue
        future[i] = current[i]
        if current[i, _CH_AMBER] > 0.5:
            on = 0
            for t in range(past.shape[1] - 1, -1, -1):
                if past[i, t, _CH_AMBER] <= 0.5:
                    break
                on += 1
            red_from = max(0, _AMBER_STEPS - on)
            future[i, red_from:] = 0.0
            future[i, red_from:, _CH_RED] = 1.0
    return future


class MlPlannerOnnx:
    """A sampler ONNX with its input layout, run with zero noise as the deployed node does."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(path, providers=providers)
        # ORT falls back to CPU without an error, which turns every case into a timeout.
        if device.startswith("cuda") and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(f"CUDAExecutionProvider did not load for {path}")
        self.shapes = {i.name: tuple(i.shape[1:]) for i in self.session.get_inputs()}
        missing = set(self.shapes) - {"initial_noise"} - set(INPUT_BUILDERS)
        if missing:
            raise ValueError(f"{path} takes inputs this rollout cannot build: {sorted(missing)}")
        self._noise = np.zeros((1, *self.shapes["initial_noise"]), dtype=np.float32)

    def predict(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
        """-> (ego-frame plan ``[T, 4]`` of x, y, cos, sin; turn indicator report 1/2/3)."""
        feed = {k: v[None].astype(np.float32) for k, v in normalize_frame(inputs).items()}
        feed["initial_noise"] = self._noise
        trajectory, logits = self.session.run(None, feed)
        trajectory, logits = decode_onnx_outputs(trajectory, logits, batch_size=1)
        return trajectory[0, 0], int(np.argmax(logits[0])) + 1


class MlPlannerObservation:
    """Per-rollout histories, and the model inputs they make at the current tick."""

    def __init__(self, model: MlPlannerOnnx, map_features, route_ids, goal_pose) -> None:
        self.shapes = model.shapes
        self.history = self.shapes["ego_agent_past"][0]
        self.features = map_features
        self.route_ids = [int(i) for i in route_ids]
        self.goal = Pose2(float(goal_pose[0]), float(goal_pose[1]), 0.0, float(goal_pose[2]))
        self.ego: deque = deque(maxlen=self.history)
        self.turn: deque = deque(maxlen=self.history)
        # Longer than the window: a slot near its start may hold a signal heard just before it.
        self.lights: deque = deque(maxlen=self.history + _TL_TIMEOUT_TICKS)
        # name -> [(tick, Pose2)] over the window.
        self.agents: dict[str, deque] = {}
        self.current: dict = {}
        self.ego_box: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.tick = -1

    def record(self, states: dict, ego_name: str, light_groups: dict, turn_report: int) -> None:
        self.tick += 1
        ego = states[ego_name]
        tw = ego["twist"]
        self.ego.append((_pose_of(ego, centroid=False), float(tw["linear_x"]), float(tw["angular_z"])))
        bbox = ego["bounding_box"]
        length, width = float(bbox["dimensions"]["x"]), float(bbox["dimensions"]["y"])
        self.ego_box = (float(bbox["center"]["x"]) + length / 2, length, width)
        self.turn.append(float(turn_report))
        self.lights.append(light_groups)
        self.current = {}
        for name, st in states.items():
            label = _agent_label(st) if name != ego_name else None
            if label is None:
                continue
            pose = _pose_of(st, centroid=True)
            self.agents.setdefault(name, deque(maxlen=self.history)).append((self.tick, pose))
            dims = st["bounding_box"]["dimensions"]
            self.current[name] = (label, float(dims["y"]), float(dims["x"]), pose)

    def build(self) -> dict[str, np.ndarray]:
        frame = self.ego[-1][0]
        s = self.shapes
        sel = self.features.select(
            frame.as_pose7(),
            self.route_ids,
            max_lanes=s["lanes"][0],
            max_route_lanes=s["route_lanes"][0],
            max_intersection_areas=s["intersection_area"][0],
            max_stop_lines=s["stop_lines"][0],
            max_road_borders=s["road_borders"][0],
            range_m=_MASK_RANGE_M,
        )
        ctx = _Context(self, frame, sel)
        return {name: INPUT_BUILDERS[name](ctx) for name in s if name != "initial_noise"}


    def ego_history(self, frame: Pose2) -> np.ndarray:
        rows = list(self.ego)
        rows = [rows[0]] * (self.history - len(rows)) + rows  # the node holds its first message
        out = np.zeros((self.history, 6), dtype=np.float32)
        for t, (pose, v, yaw_rate) in enumerate(rows):
            out[t, :4] = pose.relative(frame)
            out[t, 4] = v
            out[t, 5] = 0.0 if abs(v) < _MOVING_SPEED_MPS else np.clip(yaw_rate, -_MAX_YAW_RATE, _MAX_YAW_RATE)
        return out

    def neighbors(self, frame: Pose2):
        ranked = sorted(
            self.current.items(),
            key=lambda kv: math.hypot(*kv[1][3].to_ego(np.array([kv[1][3].x, kv[1][3].y]), frame)),
        )
        return ranked[: self.shapes["neighbor_agents_past"][0]]

    def neighbor_history(self, name: str, frame: Pose2) -> np.ndarray:
        out = np.zeros((self.history, 4), dtype=np.float32)
        seen = list(self.agents[name])
        first_tick = self.tick - self.history + 1
        k = 0
        for t in range(self.history):
            grid = first_tick + t
            while k + 1 < len(seen) and seen[k + 1][0] <= grid:
                k += 1
            if seen[k][0] <= grid:
                out[t] = seen[k][1].relative(frame)
        return out

    def light_history(self, lanes: dict) -> np.ndarray:
        n = len(lanes["lanelet_id"])
        out = np.zeros((n, self.history, 6), dtype=np.float32)
        observed = list(self.lights)
        offset = self.history - len(observed)
        for i in range(n):
            tl_id, turn = int(lanes["traffic_light_id"][i]), int(lanes["turn_direction"][i])
            if tl_id < 0:
                out[i, :, _CH_NONE] = 1.0
                continue
            for t in range(self.history):
                elements = None
                for back in range(_TL_TIMEOUT_TICKS + 1):
                    j = t - offset - back
                    if 0 <= j and tl_id in observed[j]:
                        elements = observed[j][tl_id]
                        break
                if elements is None:
                    out[i, t, _CH_UNKNOWN] = 1.0
                    continue
                color, arrow = _light_status(turn, elements)
                out[i, t, _CH_GREEN] = color == _TL_GREEN
                out[i, t, _CH_AMBER] = color == _TL_AMBER
                out[i, t, _CH_RED] = color == _TL_RED
                out[i, t, _CH_UNKNOWN] = color == 0
                out[i, t, _CH_NONE] = color == _TL_WHITE
                out[i, t, _CH_ARROW] = arrow
        return out


class _Context:
    """One tick's derived values, shared by the input builders so each is computed once."""

    def __init__(self, obs: MlPlannerObservation, frame: Pose2, sel: dict) -> None:
        self.obs, self.frame, self.sel = obs, frame, sel
        self._cache: dict = {}

    def once(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def pad(self, name: str, rows: np.ndarray) -> np.ndarray:
        out = np.zeros(self.obs.shapes[name], dtype=np.float32)
        out[: len(rows)] = rows
        return out


def _lane_tensor(ctx: _Context, key: str, name: str) -> np.ndarray:
    lanes = ctx.sel[key]
    c = lanes["center"]
    return ctx.pad(name, np.concatenate([c, lanes["left"] - c, lanes["right"] - c], axis=-1))


def _lane_types(ctx: _Context, key: str, name: str) -> np.ndarray:
    types = ctx.sel[key]["boundary_type"].astype(np.int64)
    half = ctx.obs.shapes[name][-1] // 2
    rows = np.zeros((len(types), 2 * half), dtype=np.float32)
    rows[np.arange(len(types)), types[:, 0]] = 1.0
    rows[np.arange(len(types)), half + types[:, 1]] = 1.0
    return ctx.pad(name, rows)


def _speed_limit(ctx: _Context, key: str, name: str) -> np.ndarray:
    return ctx.pad(name, np.nan_to_num(ctx.sel[key]["speed_limit_mps"], nan=0.0)[:, None])


def _light_past(ctx: _Context, key: str) -> np.ndarray:
    return ctx.once(("tl", key), lambda: ctx.obs.light_history(ctx.sel[key]))


def _light_past_input(ctx: _Context, key: str, name: str) -> np.ndarray:
    return ctx.pad(name, _light_past(ctx, key))


def _light_future_input(ctx: _Context, key: str, name: str) -> np.ndarray:
    steps = ctx.obs.shapes[name][1]
    return ctx.pad(name, infer_traffic_light_future(_light_past(ctx, key), steps))


def _neighbors(ctx: _Context):
    return ctx.once("neighbors", lambda: ctx.obs.neighbors(ctx.frame))


def _neighbor_past(ctx: _Context) -> np.ndarray:
    rows = [ctx.obs.neighbor_history(name, ctx.frame) for name, _ in _neighbors(ctx)]
    return ctx.pad("neighbor_agents_past", np.array(rows).reshape(-1, *ctx.obs.shapes["neighbor_agents_past"][1:]))


def _agent_shape(ctx: _Context) -> np.ndarray:
    rows = [(w, length) for _, (_, w, length, _) in _neighbors(ctx)]
    return ctx.pad("agent_shape", np.array(rows, dtype=np.float32).reshape(-1, 2))


def _agent_label_input(ctx: _Context) -> np.ndarray:
    labels = [label for _, (label, *_rest) in _neighbors(ctx)]
    rows = np.zeros((len(labels), ctx.obs.shapes["agent_label"][-1]), dtype=np.float32)
    rows[np.arange(len(labels)), labels] = 1.0
    return ctx.pad("agent_label", rows)


def _goal_pose(ctx: _Context) -> np.ndarray:
    return np.array(ctx.obs.goal.relative(ctx.frame), dtype=np.float32)


INPUT_BUILDERS = {
    "ego_agent_past": lambda ctx: ctx.obs.ego_history(ctx.frame),
    "neighbor_agents_past": _neighbor_past,
    "agent_shape": _agent_shape,
    "agent_label": _agent_label_input,
    "lanes": lambda ctx: _lane_tensor(ctx, "lanes", "lanes"),
    "lane_types": lambda ctx: _lane_types(ctx, "lanes", "lane_types"),
    "lanes_speed_limit": lambda ctx: _speed_limit(ctx, "lanes", "lanes_speed_limit"),
    "lane_traffic_light_past": lambda ctx: _light_past_input(ctx, "lanes", "lane_traffic_light_past"),
    "lane_traffic_light_future": lambda ctx: _light_future_input(
        ctx, "lanes", "lane_traffic_light_future"
    ),
    "route_lanes": lambda ctx: _lane_tensor(ctx, "route_lanes", "route_lanes"),
    "route_lane_types": lambda ctx: _lane_types(ctx, "route_lanes", "route_lane_types"),
    "route_lanes_speed_limit": lambda ctx: _speed_limit(
        ctx, "route_lanes", "route_lanes_speed_limit"
    ),
    "route_traffic_light_past": lambda ctx: _light_past_input(
        ctx, "route_lanes", "route_traffic_light_past"
    ),
    "route_traffic_light_future": lambda ctx: _light_future_input(
        ctx, "route_lanes", "route_traffic_light_future"
    ),
    "intersection_area": lambda ctx: ctx.pad("intersection_area", ctx.sel["intersection_areas"]),
    "stop_lines": lambda ctx: ctx.pad("stop_lines", ctx.sel["stop_lines"]),
    "road_borders": lambda ctx: ctx.pad("road_borders", ctx.sel["road_borders"]),
    "goal_pose": _goal_pose,
    "ego_shape": lambda ctx: np.array(ctx.obs.ego_box, dtype=np.float32),
    "turn_indicators": lambda ctx: np.array(
        [ctx.obs.turn[0]] * (ctx.obs.history - len(ctx.obs.turn)) + list(ctx.obs.turn),
        dtype=np.float32,
    ),
}
