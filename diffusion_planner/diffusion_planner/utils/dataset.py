import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list, past_neighbor_num, predicted_neighbor_num, future_len):
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)

        ego_agent_past = data["ego_agent_past"].astype(np.float32)
        ego_current_state = data["ego_current_state"]
        ego_agent_future = data["ego_agent_future"].astype(np.float32)

        neighbor_agents_past = data["neighbor_agents_past"][: self._past_neighbor_num]
        neighbor_agents_future = data["neighbor_agents_future"][: self._predicted_neighbor_num]

        lanes = data["lanes"]
        lanes_speed_limit = data["lanes_speed_limit"]
        lanes_has_speed_limit = data["lanes_has_speed_limit"]

        route_lanes = data["route_lanes"]
        route_lanes_speed_limit = data["route_lanes_speed_limit"]
        route_lanes_has_speed_limit = data["route_lanes_has_speed_limit"]

        polygons = data["polygons"]
        line_strings = data["line_strings"]

        static_objects = data["static_objects"]

        turn_indicator = data["turn_indicator"]

        goal_pose = data["goal_pose"]

        # wheel_base, length, width
        ego_shape = np.array([2.75, 4.34, 1.70], dtype=np.float32)

        data = {
            "ego_agent_past": ego_agent_past,
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "polygons": polygons,
            "line_strings": line_strings,
            "static_objects": static_objects,
            "turn_indicator": turn_indicator,
            "goal_pose": goal_pose,
            "ego_shape": ego_shape,
        }

        return tuple(data.values())
