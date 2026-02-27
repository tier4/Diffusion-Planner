"""
Coordinate utilities for RLVR: converts between ego-centric NPZ data and
MGRS local Cartesian map frame used by TeraSim/SUMO.

All .npz files store data in the ego base_link frame at t=0.
The companion .json sidecar stores the world-frame ego pose as MGRS local
Cartesian (x, y) and quaternion (qx, qy, qz, qw).
"""

import json

import numpy as np
from scipy.spatial.transform import Rotation


def load_bl2map(json_path: str) -> np.ndarray:
    """
    Build the 4x4 homogeneous base_link-to-map transform from the JSON sidecar.

    The JSON stores the ego pose at t=0 in MGRS local Cartesian coordinates:
      x, y, z  — translation in meters
      qx, qy, qz, qw  — rotation as quaternion

    Returns:
        (4, 4) float64 transform matrix: p_map = bl2map @ p_bl
    """
    meta = json.load(open(json_path))
    rot = Rotation.from_quat([meta["qx"], meta["qy"], meta["qz"], meta["qw"]])
    bl2map = np.eye(4)
    bl2map[:3, :3] = rot.as_matrix()
    bl2map[:3, 3] = [meta["x"], meta["y"], meta["z"]]
    return bl2map


def ego_centric_to_map(xy_bl: np.ndarray, bl2map: np.ndarray) -> np.ndarray:
    """
    Transform (N, 2) ego-centric XY coordinates to MGRS map-frame XY.

    Points are assumed to lie on the z=0 plane of the base_link frame.

    Args:
        xy_bl:  (N, 2) array of [x, y] in ego base_link frame
        bl2map: (4, 4) base_link-to-map transform

    Returns:
        (N, 2) array of [x, y] in MGRS map frame
    """
    n = len(xy_bl)
    pts_h = np.hstack([xy_bl, np.zeros((n, 1)), np.ones((n, 1))])  # (N, 4)
    pts_map = (bl2map @ pts_h.T).T                                   # (N, 4)
    return pts_map[:, :2]


def heading_bl_to_map(cos_h: float, sin_h: float, bl2map: np.ndarray) -> float:
    """
    Convert a heading encoded as (cos, sin) in the base_link frame to a map-frame
    yaw angle (radians, counter-clockwise from +X axis).

    The rotation is additive: the map-frame heading is the ego's own yaw plus
    the heading relative to the ego.

    Args:
        cos_h:  cos(heading) in base_link frame
        sin_h:  sin(heading) in base_link frame
        bl2map: (4, 4) base_link-to-map transform

    Returns:
        yaw angle in radians, counter-clockwise from +X
    """
    ego_yaw = np.arctan2(bl2map[1, 0], bl2map[0, 0])
    agent_yaw = np.arctan2(sin_h, cos_h)
    return ego_yaw + agent_yaw


def ros_yaw_to_sumo_angle(yaw_rad: float) -> float:
    """
    Convert ROS/Autoware yaw (counter-clockwise from +X axis, radians) to
    SUMO angle (clockwise from north/+Y axis, degrees).

    Formula: sumo_angle = (90 - yaw_deg) mod 360

    Args:
        yaw_rad: heading in radians, CCW from +X

    Returns:
        heading in degrees, CW from +Y (SUMO convention)
    """
    return float(np.degrees(np.pi / 2 - yaw_rad) % 360)


def extract_spawn_states(npz_path: str, json_path: str) -> dict:
    """
    Extract t=0 spawn states for the ego and all active NPCs, plus the ego's
    ground-truth future trajectory — all in MGRS map frame.

    Args:
        npz_path:  path to .npz data file
        json_path: path to companion .json sidecar

    Returns:
        {
            "ego": {
                "x": float, "y": float,
                "yaw_rad": float, "sumo_angle": float,
                "vx": float, "length": float, "width": float
            },
            "npcs": [
                {
                    "id": str, "x": float, "y": float,
                    "yaw_rad": float, "sumo_angle": float,
                    "vx": float, "width": float, "length": float,
                    "class": int   # 0=vehicle, 1=pedestrian, 2=bicycle
                },
                ...
            ],
            "ego_future_map": np.ndarray (80, 3)  # [x, y, yaw_rad] in map frame
        }
    """
    data = np.load(npz_path, allow_pickle=True)
    meta = json.load(open(json_path))
    bl2map = load_bl2map(json_path)

    # Ego at t=0: position comes directly from JSON (it IS the origin of bl2map)
    ego_yaw_map = float(np.arctan2(bl2map[1, 0], bl2map[0, 0]))
    ego = {
        "x":          float(meta["x"]),
        "y":          float(meta["y"]),
        "yaw_rad":    ego_yaw_map,
        "sumo_angle": ros_yaw_to_sumo_angle(ego_yaw_map),
        "vx":         float(data["ego_current_state"][4]),
        "length":     float(data["ego_shape"][1]),
        "width":      float(data["ego_shape"][2]),
    }

    # NPCs: neighbor_agents_past has shape (32, 21, 11)
    # Index -1 along axis=1 gives the most recent (t=0) state per NPC
    # Fields: [x, y, cos_yaw, sin_yaw, vx, vy, w, l, is_veh, is_ped, is_bike]
    npc_past = data["neighbor_agents_past"]  # (32, 21, 11)
    npcs = []
    for i in range(npc_past.shape[0]):
        row = npc_past[i, -1]  # (11,) at t=0
        if np.all(row == 0):
            continue  # slot is empty
        xy_map = ego_centric_to_map(row[:2].reshape(1, 2), bl2map)[0]
        yaw_map = heading_bl_to_map(float(row[2]), float(row[3]), bl2map)
        npcs.append({
            "id":         f"npc_{i}",
            "x":          float(xy_map[0]),
            "y":          float(xy_map[1]),
            "yaw_rad":    float(yaw_map),
            "sumo_angle": ros_yaw_to_sumo_angle(float(yaw_map)),
            "vx":         float(np.sqrt(row[4] ** 2 + row[5] ** 2)),
            "width":      float(row[6]),
            "length":     float(row[7]),
            "class":      int(np.argmax(row[8:11])),  # 0=veh, 1=ped, 2=bike
        })

    # Ego future in map frame: ego_agent_future shape (80, 3) = [x, y, yaw_rad]
    ego_future_bl = data["ego_agent_future"]  # (80, 3)
    ego_future_map_xy = ego_centric_to_map(ego_future_bl[:, :2], bl2map)
    ego_yaw_offset = float(np.arctan2(bl2map[1, 0], bl2map[0, 0]))
    ego_future_map = np.column_stack([
        ego_future_map_xy,
        ego_yaw_offset + ego_future_bl[:, 2],
    ])  # (80, 3): [x, y, yaw_rad] in map frame

    return {
        "ego":            ego,
        "npcs":           npcs,
        "ego_future_map": ego_future_map,
    }
