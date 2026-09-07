"""Raw tensor inspection controls for the frame browser."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
from numpy.typing import NDArray


@dataclass(frozen=True)
class TensorDisplaySpec:
    """Human-readable axis and feature names for a known frame tensor."""

    leading_axes: tuple[str, ...] = ()
    row_axis: str = "index"
    fields: tuple[str, ...] | None = None


_LANE_TYPES = (
    "crosswalk",
    "curbstone",
    "guard_rail",
    "line_thick",
    "line_thin",
    "pedestrian_marking",
    "road_border",
    "road_shoulder",
    "virtual",
    "zebra_marking",
)
_LANE_FIELDS = (
    "x",
    "y",
    "left_offset_x",
    "left_offset_y",
    "right_offset_x",
    "right_offset_y",
    *(f"left_type_{name}" for name in _LANE_TYPES),
    *(f"right_type_{name}" for name in _LANE_TYPES),
)
_POSE_FIELDS = ("x", "y", "cos_yaw", "sin_yaw")
_EGO_FIELDS = (*_POSE_FIELDS, "velocity", "yaw_rate")
_NEIGHBOR_FIELDS = _POSE_FIELDS
_TRAFFIC_LIGHT_FIELDS = (
    "green",
    "amber",
    "red",
    "unknown",
    "white_or_none",
    "is_arrow",
)

_TENSOR_SPECS: dict[str, TensorDisplaySpec] = {
    "ego_agent_past": TensorDisplaySpec(row_axis="time_index", fields=_EGO_FIELDS),
    "ego_agent_future": TensorDisplaySpec(row_axis="time_index", fields=_EGO_FIELDS),
    "neighbor_agents_past": TensorDisplaySpec(
        leading_axes=("agent",), row_axis="time_index", fields=_NEIGHBOR_FIELDS
    ),
    "agent_shape": TensorDisplaySpec(row_axis="agent", fields=("width", "length")),
    "agent_label": TensorDisplaySpec(
        row_axis="agent",
        fields=("is_vehicle", "is_pedestrian", "is_bicycle", "is_unknown"),
    ),
    "neighbor_agents_future": TensorDisplaySpec(
        leading_axes=("agent",), row_axis="time_index", fields=_POSE_FIELDS
    ),
    "lanes": TensorDisplaySpec(
        leading_axes=("lane",), row_axis="point_index", fields=_LANE_FIELDS
    ),
    "route_lanes": TensorDisplaySpec(
        leading_axes=("route_lane",), row_axis="point_index", fields=_LANE_FIELDS
    ),
    "intersection_area": TensorDisplaySpec(
        leading_axes=("intersection_area",),
        row_axis="point_index",
        fields=("x", "y"),
    ),
    "stop_lines": TensorDisplaySpec(
        leading_axes=("stop_line",), row_axis="point_index", fields=("x", "y")
    ),
    "road_borders": TensorDisplaySpec(
        leading_axes=("road_border",), row_axis="point_index", fields=("x", "y")
    ),
    "lane_traffic_light_past": TensorDisplaySpec(
        leading_axes=("lane",), row_axis="time_index", fields=_TRAFFIC_LIGHT_FIELDS
    ),
    "route_traffic_light_past": TensorDisplaySpec(
        leading_axes=("route_lane",),
        row_axis="time_index",
        fields=_TRAFFIC_LIGHT_FIELDS,
    ),
    "ego_shape": TensorDisplaySpec(
        fields=("base_link_to_front", "vehicle_length", "vehicle_width")
    ),
    "goal_pose": TensorDisplaySpec(fields=_POSE_FIELDS),
    "lanes_speed_limit": TensorDisplaySpec(
        row_axis="lane_index", fields=("speed_limit",)
    ),
    "route_lanes_speed_limit": TensorDisplaySpec(
        row_axis="route_lane_index", fields=("speed_limit",)
    ),
    "turn_indicators": TensorDisplaySpec(row_axis="time_index", fields=("value",)),
    "turn_indicators_future": TensorDisplaySpec(
        row_axis="time_index", fields=("value",)
    ),
    "predicted_trajectory": TensorDisplaySpec(
        leading_axes=("agent",), row_axis="time_index", fields=_POSE_FIELDS
    ),
}


def _field_names(spec: TensorDisplaySpec, width: int) -> list[str]:
    if spec.fields is not None and len(spec.fields) == width:
        return list(spec.fields)
    return [f"feature_{index}" for index in range(width)]


def _leading_axis_names(spec: TensorDisplaySpec, count: int) -> list[str]:
    names = list(spec.leading_axes[:count])
    names.extend(f"axis_{index}" for index in range(len(names), count))
    return names


def _slice_tensor(
    tensor_name: str,
    array: NDArray[np.generic],
    spec: TensorDisplaySpec,
    key_prefix: str,
) -> tuple[NDArray[np.generic], tuple[int, ...]]:
    leading_count = max(0, array.ndim - 2)
    selected: list[int] = []
    for axis, axis_name in enumerate(_leading_axis_names(spec, leading_count)):
        selected.append(
            st.slider(
                f"{axis_name.replace('_', ' ').title()} index",
                min_value=0,
                max_value=array.shape[axis] - 1,
                value=0,
                step=1,
                key=f"{key_prefix}::{tensor_name}::axis-{axis}",
            )
        )
    if not selected:
        return array, ()
    selection = (*selected, *(slice(None) for _ in range(2)))
    return array[selection], tuple(selected)


def _table_data(
    array: NDArray[np.generic], spec: TensorDisplaySpec
) -> Mapping[str, Sequence[Any] | NDArray[np.generic]]:
    if array.ndim == 0:
        return {"value": [array.item()]}
    if array.ndim == 1:
        if spec.fields is not None and len(spec.fields) == array.shape[0]:
            return {"field": list(spec.fields), "value": array}
        value_name = (
            spec.fields[0]
            if spec.fields is not None and len(spec.fields) == 1
            else "value"
        )
        return {spec.row_axis: np.arange(array.shape[0]), value_name: array}
    if array.ndim != 2:
        raise ValueError(
            f"Expected a tensor slice with at most 2 dimensions, got {array.shape}"
        )
    columns: dict[str, Sequence[Any] | NDArray[np.generic]] = {
        spec.row_axis: np.arange(array.shape[0])
    }
    for index, name in enumerate(_field_names(spec, array.shape[1])):
        columns[name] = array[:, index]
    return columns


def render_tensor_inspector(
    frame_data: Mapping[str, Any], *, key_prefix: str = "tensor-inspector"
) -> None:
    """Render a collapsed raw-data inspector for the selected frame."""
    with st.expander("Tensor Inspector", expanded=False):
        tensor_names = sorted(frame_data)
        if not tensor_names:
            st.info("The selected frame does not contain tensor data.")
            return

        tensor_name = st.selectbox(
            "Tensor",
            tensor_names,
            key=f"{key_prefix}::tensor",
        )
        array = np.asarray(frame_data[tensor_name])
        st.caption(f"Shape: {array.shape} · Dtype: {array.dtype}")

        spec = _TENSOR_SPECS.get(tensor_name, TensorDisplaySpec())
        displayed, selected = _slice_tensor(tensor_name, array, spec, key_prefix)
        if selected:
            prefix = ", ".join(str(index) for index in selected)
            st.code(
                f"{tensor_name}[{prefix}, :, :]  shape={displayed.shape}", language=None
            )
        else:
            st.code(f"{tensor_name}[:]  shape={displayed.shape}", language=None)

        st.dataframe(_table_data(displayed, spec), width="stretch", hide_index=True)
