"""venv-safe neighbor-agent input builders, ported from the C++ AgentData.

This module deliberately avoids any ROS / rosbag2_py import so it can run both under the
Python-3.10 ROS env (used by parse_rosbag.py) and in a plain venv (used by the closed-loop
Perception Reproducer). The builders read tracked-object attributes by duck typing, so the
caller may pass real ROS TrackedObject msgs OR lightweight stand-ins with the same fields.
"""

from collections import deque

import numpy as np
from diffusion_planner.dimensions import INPUT_T
from diffusion_planner_ros.utils import pose_to_mat4x4

PAST_TIME_STEPS = INPUT_T + 1

# Autoware ObjectClassification labels.
_CLS_UNKNOWN = 0
_CLS_CAR = 1
_CLS_TRUCK = 2
_CLS_BUS = 3
_CLS_TRAILER = 4
_CLS_MOTORCYCLE = 5
_CLS_BICYCLE = 6
_CLS_PEDESTRIAN = 7

# Model labels.
_AGENT_VEHICLE = 0
_AGENT_PEDESTRIAN = 1
_AGENT_BICYCLE = 2


def _highest_prob_label(classification) -> int:
    """Mirror autoware object_recognition_utils::getHighestProbLabel (first strict max wins)."""
    best_label = _CLS_UNKNOWN
    best_prob = -1.0
    for c in classification:
        if c.probability > best_prob:
            best_prob = c.probability
            best_label = c.label
    return best_label


def _get_model_label(classification) -> int:
    label = _highest_prob_label(classification)
    if label in (_CLS_CAR, _CLS_TRUCK, _CLS_BUS, _CLS_MOTORCYCLE, _CLS_TRAILER):
        return _AGENT_VEHICLE
    if label == _CLS_BICYCLE:
        return _AGENT_BICYCLE
    if label == _CLS_PEDESTRIAN:
        return _AGENT_PEDESTRIAN
    return _AGENT_VEHICLE


def _is_unknown_object(classification) -> bool:
    return _highest_prob_label(classification) == _CLS_UNKNOWN


def _agent_state_from_object(obj):
    """Capture the raw per-state info (pose in map frame + attributes)."""
    pose_map = pose_to_mat4x4(obj.kinematics.pose_with_covariance.pose)
    twist = obj.kinematics.twist_with_covariance.twist.linear
    return {
        "pose_map": pose_map,
        "vx": twist.x,
        "vy": twist.y,
        "width": obj.shape.dimensions.y,
        "length": obj.shape.dimensions.x,
        "label": _get_model_label(obj.classification),
    }


def _agent_state_array(state, map2bl_matrix_4x4):
    """C++ AgentState::as_array, computed after transform to ego frame (11 features)."""
    pose_bl = map2bl_matrix_4x4 @ state["pose_map"]
    cos_yaw = pose_bl[0, 0]
    sin_yaw = pose_bl[1, 0]
    # Normalize to a pure heading cos/sin, matching rotation_matrix_to_cos_sin (atan2 then cos/sin).
    yaw = np.arctan2(sin_yaw, cos_yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    velocity_norm = np.hypot(state["vx"], state["vy"])
    label = state["label"]
    return (
        pose_bl[0, 3],
        pose_bl[1, 3],
        cos_yaw,
        sin_yaw,
        velocity_norm * cos_yaw,
        velocity_norm * sin_yaw,
        state["width"],
        state["length"],
        float(label == _AGENT_VEHICLE),
        float(label == _AGENT_PEDESTRIAN),
        float(label == _AGENT_BICYCLE),
    )


def _latest_state_distance(history, map2bl_matrix_4x4):
    pose_bl = map2bl_matrix_4x4 @ history[-1]["pose_map"]
    return np.hypot(pose_bl[0, 3], pose_bl[1, 3])


def build_neighbor_past(data_list, i, map2bl_matrix_4x4, max_num_objects, time_length):
    """Port of AgentData.update_histories + transformed_and_trimmed_histories + flatten.

    Returns (tensor (max_num_objects, time_length, 11), agent_ids) where agents are sorted
    by latest-state ego distance and trimmed to max_num_objects, matching the C++ converter.
    agent_ids is the ordered list of object ids kept (used for the future, which must share
    the same ordering).
    """
    start_idx = max(0, i - time_length + 1)
    histories = {}  # object_id(bytes) -> deque[state] (max time_length)
    for frame_idx in range(start_idx, i + 1):
        objects = data_list[frame_idx].tracked_objects.objects
        found_ids = set()
        for obj in objects:
            if _is_unknown_object(obj.classification):
                continue
            oid = bytes(obj.object_id.uuid)
            state = _agent_state_from_object(obj)
            if oid in histories:
                dq = histories[oid]
                if len(dq) >= time_length:
                    dq.popleft()
                dq.append(state)
            else:
                # New agent: fill the whole history with the first observation.
                histories[oid] = deque([state] * time_length, maxlen=time_length)
            found_ids.add(oid)
        # Drop agents not present in this frame.
        for oid in [k for k in histories if k not in found_ids]:
            del histories[oid]

    items = list(histories.items())
    items.sort(key=lambda kv: _latest_state_distance(kv[1], map2bl_matrix_4x4))
    items = items[:max_num_objects]

    out = np.zeros((max_num_objects, time_length, 11), dtype=np.float32)
    for a, (_oid, history) in enumerate(items):
        for t, state in enumerate(history):
            out[a, t] = _agent_state_array(state, map2bl_matrix_4x4)
    agent_ids = [oid for oid, _ in items]
    return out, agent_ids


def build_neighbor_future(data_list, i, map2bl_matrix_4x4, agent_ids, max_num_objects, out_t):
    """Port of process_neighbor_agents_and_future's future half.

    For each past agent (same order), seed an OUTPUT_T-length history with the current-frame
    state, then walk future frames appending the same id until it disappears. Returns
    (max_num_objects, out_t, 3) of [x, y, heading] in ego frame.
    """
    current_objs = {
        bytes(o.object_id.uuid): o for o in data_list[i].tracked_objects.objects
    }
    # Pre-index future frames by object id.
    future_maps = []
    for t in range(1, out_t + 1):
        fidx = i + t
        if fidx >= len(data_list):
            future_maps.append(None)
        else:
            future_maps.append(
                {bytes(o.object_id.uuid): o for o in data_list[fidx].tracked_objects.objects}
            )

    out = np.zeros((max_num_objects, out_t, 3), dtype=np.float32)
    for a, oid in enumerate(agent_ids):
        seed_obj = current_objs.get(oid)
        if seed_obj is None:
            continue
        dq = deque(maxlen=out_t)
        dq.append(_agent_state_from_object(seed_obj))
        for fmap in future_maps:
            if fmap is None:
                break
            fo = fmap.get(oid)
            if fo is None:
                break
            dq.append(_agent_state_from_object(fo))
        for t, state in enumerate(dq):
            arr = _agent_state_array(state, map2bl_matrix_4x4)
            out[a, t, 0] = arr[0]
            out[a, t, 1] = arr[1]
            out[a, t, 2] = np.arctan2(arr[3], arr[2])
    return out
