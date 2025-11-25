import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROS_PKG_ROOT = REPO_ROOT / "diffusion_planner_ros"
if str(ROS_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS_PKG_ROOT))

from diffusion_planner.dimensions import (
    INPUT_T,
    MAX_NUM_NEIGHBORS,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    OUTPUT_T,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
)
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    create_lane_tensor,
    create_line_tensor,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DT = 0.1
MAP_DEVICE = torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GPUDRIVE simulation_data.json into npz files that mimic the "
            "Diffusion-Planner dataset expected by train_predictor.py."
        )
    )
    parser.add_argument("simulation_data_path", type=Path)
    parser.add_argument("vector_map_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--step", type=int, default=1, help="Sliding window stride.")
    return parser.parse_args()


ArrayOrFloat = Union[np.ndarray, float]


def wrap_angle(angle: ArrayOrFloat) -> ArrayOrFloat:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transform_to_local(
    xs: np.ndarray,
    ys: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
) -> np.ndarray:
    dx = xs - anchor_x
    dy = ys - anchor_y
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    x_local = cos_yaw * dx + sin_yaw * dy
    y_local = -sin_yaw * dx + cos_yaw * dy
    return np.stack([x_local, y_local], axis=-1)


def create_transform_matrices(
    x: float,
    y: float,
    z: float,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation = np.array([x, y, z], dtype=np.float32)
    bl2map = np.eye(4, dtype=np.float32)
    bl2map[:3, :3] = rotation
    bl2map[:3, 3] = translation
    map2bl = np.eye(4, dtype=np.float32)
    map2bl[:3, :3] = rotation.T
    map2bl[:3, 3] = -rotation.T @ translation
    return bl2map, map2bl


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).detach().cpu().numpy()


@dataclass
class SimulationSeries:
    step_keys: List[str]
    pos_x: np.ndarray
    pos_y: np.ndarray
    pos_z: np.ndarray
    yaw: np.ndarray
    goal_x: np.ndarray
    goal_y: np.ndarray
    vehicle_length: np.ndarray
    vehicle_width: np.ndarray
    vehicle_height: np.ndarray
    agent_ids: np.ndarray

    @property
    def num_steps(self) -> int:
        return self.pos_x.shape[0]

    @property
    def num_agents(self) -> int:
        return self.pos_x.shape[1]


def _stack_field(step_items: List[tuple[str, dict]], field_name: str) -> np.ndarray:
    arrays = []
    for _, data in step_items:
        global_state = data.get("global_ego_state")
        if global_state is None:
            raise KeyError(f"'global_ego_state' is missing under step {data}")
        field_value = global_state.get(field_name)
        if field_value is None:
            raise KeyError(f"{field_name} is missing in global_ego_state")
        arrays.append(np.asarray(field_value[0], dtype=np.float32))
    return np.stack(arrays, axis=0)


def load_simulation_series(json_path: Path) -> SimulationSeries:
    LOGGER.info("Loading %s", json_path)
    raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    if not raw_data:
        raise ValueError(f"{json_path} does not contain any steps.")
    step_items = sorted(raw_data.items(), key=lambda kv: int(kv[0].split("_")[1]))
    pos_x = _stack_field(step_items, "pos_x")
    pos_y = _stack_field(step_items, "pos_y")
    pos_z = _stack_field(step_items, "pos_z")
    yaw = _stack_field(step_items, "rotation_angle")
    goal_x = _stack_field(step_items, "goal_x")
    goal_y = _stack_field(step_items, "goal_y")
    vehicle_length = _stack_field(step_items, "vehicle_length")
    vehicle_width = _stack_field(step_items, "vehicle_width")
    vehicle_height = _stack_field(step_items, "vehicle_height")
    agent_ids = _stack_field(step_items, "id").astype(np.int64)
    return SimulationSeries(
        step_keys=[key for key, _ in step_items],
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
        yaw=yaw,
        goal_x=goal_x,
        goal_y=goal_y,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        vehicle_height=vehicle_height,
        agent_ids=agent_ids,
    )


def compute_goal_pose(goal_x: float, goal_y: float) -> np.ndarray:
    heading = wrap_angle(math.atan2(goal_y, goal_x))
    return np.asarray([goal_x, goal_y, heading], dtype=np.float32)


def build_current_state(
    x_series: np.ndarray,
    y_series: np.ndarray,
    yaw_series: np.ndarray,
    center_idx: int,
    dt: float,
) -> np.ndarray:
    vx_local = 0.0
    vy_local = 0.0
    yaw_rate = 0.0
    if center_idx > 0:
        dx = x_series[center_idx] - x_series[center_idx - 1]
        dy = y_series[center_idx] - y_series[center_idx - 1]
        vx_map = dx / dt
        vy_map = dy / dt
        heading = yaw_series[center_idx]
        cos_yaw = math.cos(heading)
        sin_yaw = math.sin(heading)
        vx_local = cos_yaw * vx_map + sin_yaw * vy_map
        vy_local = -sin_yaw * vx_map + cos_yaw * vy_map
        yaw_rate = wrap_angle(yaw_series[center_idx] - yaw_series[center_idx - 1]) / dt
    ego_state = np.zeros(10, dtype=np.float32)
    ego_state[2] = 1.0  # heading cos in ego frame
    ego_state[4] = vx_local
    ego_state[5] = vy_local
    ego_state[9] = yaw_rate
    return ego_state


def save_sequences(
    simulation: SimulationSeries,
    vector_map,
    save_dir: Path,
    dataset_name: str,
    step_stride: int,
) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    past_steps = INPUT_T + 1
    output_t = OUTPUT_T
    dt = DEFAULT_DT
    num_steps = simulation.num_steps
    last_center = num_steps - output_t - 1
    if last_center < past_steps - 1:
        LOGGER.warning(
            "Not enough frames (%d) to create samples that need %d history + %d future.",
            num_steps,
            past_steps,
            output_t,
        )
        return 0

    zeros_static = np.zeros((5, 10), dtype=np.float32)
    samples_written = 0
    for agent_idx in range(simulation.num_agents):
        x_series = simulation.pos_x[:, agent_idx]
        y_series = simulation.pos_y[:, agent_idx]
        z_series = simulation.pos_z[:, agent_idx]
        yaw_series = simulation.yaw[:, agent_idx]
        goal_x_series = simulation.goal_x[:, agent_idx]
        goal_y_series = simulation.goal_y[:, agent_idx]
        agent_id = int(simulation.agent_ids[0, agent_idx])

        center_indices = range(past_steps - 1, last_center + 1, step_stride)
        for center_idx in center_indices:
            anchor_x = x_series[center_idx]
            anchor_y = y_series[center_idx]
            anchor_z = z_series[center_idx]
            anchor_yaw = yaw_series[center_idx]
            past_idx = np.arange(center_idx - past_steps + 1, center_idx + 1)
            future_idx = np.arange(center_idx + 1, center_idx + 1 + output_t)

            past_xy = transform_to_local(
                x_series[past_idx],
                y_series[past_idx],
                anchor_x,
                anchor_y,
                anchor_yaw,
            )
            past_yaw = wrap_angle(yaw_series[past_idx] - anchor_yaw).astype(np.float32)
            ego_past = np.concatenate([past_xy, past_yaw[:, None]], axis=1).astype(np.float32)

            future_xy = transform_to_local(
                x_series[future_idx],
                y_series[future_idx],
                anchor_x,
                anchor_y,
                anchor_yaw,
            )
            future_yaw = wrap_angle(yaw_series[future_idx] - anchor_yaw).astype(np.float32)
            ego_future = np.concatenate([future_xy, future_yaw[:, None]], axis=1).astype(
                np.float32
            )

            goal_pose = compute_goal_pose(goal_x_series[center_idx], goal_y_series[center_idx])
            ego_current_state = build_current_state(
                x_series, y_series, yaw_series, center_idx, dt
            )

            _, map2bl_matrix = create_transform_matrices(anchor_x, anchor_y, anchor_z, anchor_yaw)
            traffic_light_recognition: dict = {}
            lanes_tensor, lanes_speed_limit, lanes_has_speed_limit = create_lane_tensor(
                vector_map.lanelets.values(),
                map2bl_mat4x4=map2bl_matrix,
                center_x=anchor_x,
                center_y=anchor_y,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=NUM_SEGMENTS_IN_LANE,
                dev=MAP_DEVICE,
                do_sort=True,
            )
            route_tensor, route_speed_limit, route_has_speed_limit = create_lane_tensor(
                vector_map.lanelets.values(),
                map2bl_mat4x4=map2bl_matrix,
                center_x=anchor_x,
                center_y=anchor_y,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=NUM_SEGMENTS_IN_ROUTE,
                dev=MAP_DEVICE,
                do_sort=True,
            )
            polygon_tensor = create_line_tensor(
                vector_map.polygons.values(),
                map2bl_matrix,
                anchor_x,
                anchor_y,
                NUM_POLYGONS,
                POINTS_PER_POLYGON,
                MAP_DEVICE,
            )
            line_string_tensor = create_line_tensor(
                vector_map.line_strings.values(),
                map2bl_matrix,
                anchor_x,
                anchor_y,
                NUM_LINE_STRINGS,
                POINTS_PER_LINE_STRING,
                MAP_DEVICE,
            )
            lanes_np = tensor_to_numpy(lanes_tensor)
            lanes_speed_np = tensor_to_numpy(lanes_speed_limit)
            lanes_has_speed_np = tensor_to_numpy(lanes_has_speed_limit).astype(bool)
            route_np = tensor_to_numpy(route_tensor)
            route_speed_np = tensor_to_numpy(route_speed_limit)
            route_has_speed_np = tensor_to_numpy(route_has_speed_limit).astype(bool)
            polygons_np = tensor_to_numpy(polygon_tensor)
            line_strings_np = tensor_to_numpy(line_string_tensor)

            neighbor_past = np.zeros(
                (MAX_NUM_NEIGHBORS, past_steps, 11), dtype=np.float32
            )
            neighbor_future = np.zeros(
                (MAX_NUM_NEIGHBORS, output_t, 3), dtype=np.float32
            )
            turn_indicators = np.zeros(past_steps, dtype=np.int32)

            sample = {
                "version": 2,
                "ego_agent_past": ego_past,
                "ego_current_state": ego_current_state,
                "ego_agent_future": ego_future,
                "neighbor_agents_past": neighbor_past,
                "neighbor_agents_future": neighbor_future,
                "static_objects": zeros_static,
                "lanes": lanes_np,
                "lanes_speed_limit": lanes_speed_np,
                "lanes_has_speed_limit": lanes_has_speed_np,
                "route_lanes": route_np,
                "route_lanes_speed_limit": route_speed_np,
                "route_lanes_has_speed_limit": route_has_speed_np,
                "turn_indicators": turn_indicators,
                "goal_pose": goal_pose,
                "polygons": polygons_np,
                "line_strings": line_strings_np,
            }
            token = f"{dataset_name}_agent{agent_id}_{center_idx:05d}"
            np.savez(save_dir / f"{token}.npz", **sample)
            samples_written += 1

    return samples_written


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    simulation = load_simulation_series(args.simulation_data_path)
    vector_map = convert_lanelet(str(args.vector_map_path))
    dataset_name = args.simulation_data_path.stem
    total_samples = save_sequences(
        simulation=simulation,
        vector_map=vector_map,
        save_dir=args.save_dir,
        dataset_name=dataset_name,
        step_stride=args.step,
    )
    LOGGER.info("Created %d samples at %s", total_samples, args.save_dir)


if __name__ == "__main__":
    main()
