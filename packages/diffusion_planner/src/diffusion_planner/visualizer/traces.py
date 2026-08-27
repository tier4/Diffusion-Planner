"""Plotly trace factories for diffusion planner frame layers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseTraceType

from .frame import FrameData
from .schema import (
    AgentLabelIndex,
    AgentShapeIndex,
    LaneIndex,
    NeighborIndex,
    TrafficLightIndex,
)
from .style import FramePlotOptions, VisualizerStyle


def _joined_lines(
    lines: Iterable[NDArray[np.generic]],
) -> tuple[list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    for line in lines:
        if len(line) == 0:
            continue
        x.extend(float(value) for value in line[:, 0])
        y.extend(float(value) for value in line[:, 1])
        x.append(None)
        y.append(None)
    return x, y


def _line_trace(
    lines: Iterable[NDArray[np.generic]],
    *,
    name: str,
    color: str,
    width: float,
    legendgroup: str,
    dash: str | None = None,
    showlegend: bool = True,
    marker_size: float | None = None,
) -> go.Scattergl | None:
    x, y = _joined_lines(lines)
    if not x:
        return None
    return go.Scattergl(
        x=x,
        y=y,
        mode="lines+markers" if marker_size is not None else "lines",
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        line={"color": color, "width": width, "dash": dash},
        marker={"color": color, "size": marker_size}
        if marker_size is not None
        else None,
        hoverinfo="skip",
    )


def _lane_lines(lanes: NDArray[np.generic], kind: str) -> list[NDArray[np.generic]]:
    lines: list[NDArray[np.generic]] = []
    for lane in lanes[FrameData.valid_rows(lanes)]:
        center = lane[:, [LaneIndex.X, LaneIndex.Y]]
        if kind == "center":
            lines.append(center)
        elif kind == "left":
            lines.append(
                center + lane[:, [LaneIndex.LEFT_OFFSET_X, LaneIndex.LEFT_OFFSET_Y]]
            )
        elif kind == "right":
            lines.append(
                center + lane[:, [LaneIndex.RIGHT_OFFSET_X, LaneIndex.RIGHT_OFFSET_Y]]
            )
        else:
            raise ValueError(f"Unknown lane line kind: {kind}")
    return lines


def _route_polygon_traces(
    lanes: NDArray[np.generic], style: VisualizerStyle
) -> list[BaseTraceType]:
    """Create one filled polygon from the left and right boundaries of each route lane."""
    traces: list[BaseTraceType] = []
    for index, lane in enumerate(lanes[FrameData.valid_rows(lanes)]):
        center = lane[:, [LaneIndex.X, LaneIndex.Y]]
        left = center + lane[:, [LaneIndex.LEFT_OFFSET_X, LaneIndex.LEFT_OFFSET_Y]]
        right = center + lane[:, [LaneIndex.RIGHT_OFFSET_X, LaneIndex.RIGHT_OFFSET_Y]]
        polygon = np.concatenate((left, right[::-1], left[:1]), axis=0)
        traces.append(
            go.Scatter(
                x=polygon[:, 0],
                y=polygon[:, 1],
                mode="lines",
                fill="toself",
                fillcolor=style.route_fill_color,
                line={"color": style.route_color, "width": 1},
                name="Route",
                legendgroup="route",
                showlegend=index == 0,
                hoverinfo="skip",
            )
        )
    return traces


def create_lane_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[BaseTraceType]:
    """Create local-lane and route-lane traces."""
    traces: list[BaseTraceType] = []
    lanes = frame["lanes"]
    center = _line_trace(
        _lane_lines(lanes, "center"),
        name="Lanes",
        color=style.lane_color,
        width=style.lane_width,
        legendgroup="lanes",
    )
    if center is not None:
        traces.append(center)
    if options.show_lane_boundaries:
        for kind in ("left", "right"):
            boundary = _line_trace(
                _lane_lines(lanes, kind),
                name="Lanes boundaries",
                color=style.lane_boundary_color,
                width=max(1.0, style.lane_width * 0.6),
                legendgroup="lanes",
                showlegend=kind == "left",
            )
            if boundary is not None:
                traces.append(boundary)

    if options.show_speed_limits:
        traces.extend(_create_speed_limit_trace(frame, "lanes", lanes, "lanes"))
    if options.show_traffic_lights:
        traces.extend(_create_traffic_light_traces(frame, "lanes", lanes, "lanes"))

    route_lanes = frame["route_lanes"]
    traces.extend(_route_polygon_traces(route_lanes, style))
    if options.show_speed_limits:
        traces.extend(
            _create_speed_limit_trace(frame, "route_lanes", route_lanes, "route")
        )
    if options.show_traffic_lights:
        traces.extend(
            _create_traffic_light_traces(frame, "route_lanes", route_lanes, "route")
        )
    return traces


def _create_speed_limit_trace(
    frame: FrameData,
    lane_key: str,
    lanes: NDArray[np.generic],
    legendgroup: str,
) -> list[BaseTraceType]:
    speed_key = f"{lane_key}_speed_limit"
    speeds = frame.get(speed_key)
    if speeds is None:
        return []
    valid_indices = np.flatnonzero(FrameData.valid_rows(lanes))
    x: list[float] = []
    y: list[float] = []
    text: list[str] = []
    for index in valid_indices:
        speed = float(np.ravel(speeds[index])[0])
        if speed <= 0:
            continue
        midpoint = lanes[index, len(lanes[index]) // 2, :2]
        x.append(float(midpoint[0]))
        y.append(float(midpoint[1]))
        text.append(f"{speed:.1f} m/s")
    if not x:
        return []
    return [
        go.Scattergl(
            x=x,
            y=y,
            text=text,
            mode="markers+text",
            textposition="top center",
            marker={"size": 5, "color": "#374151"},
            name="Speed limits",
            legendgroup=legendgroup,
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        )
    ]


def _create_traffic_light_traces(
    frame: FrameData,
    lane_key: str,
    lanes: NDArray[np.generic],
    legendgroup: str,
) -> list[BaseTraceType]:
    traffic_key = (
        "lane_traffic_light_past" if lane_key == "lanes" else "route_traffic_light_past"
    )
    traffic = frame.get(traffic_key)
    if traffic is None:
        return []

    colors = ("#22c55e", "#f59e0b", "#ef4444", "#6b7280")
    names = ("Green light", "Amber light", "Red light", "Unknown light")
    valid_lanes = FrameData.valid_rows(lanes)
    traces: list[BaseTraceType] = []
    for state, (color, name) in enumerate(zip(colors, names, strict=True)):
        x: list[float] = []
        y: list[float] = []
        symbols: list[str] = []
        traffic_indices: list[int] = []
        for index in np.flatnonzero(valid_lanes):
            latest = traffic[index, -1]
            if latest.shape[-1] <= TrafficLightIndex.UNKNOWN:
                continue
            color_state = latest[: TrafficLightIndex.WHITE_OR_NONE + 1]
            if not np.any(color_state) or int(np.argmax(color_state)) != state:
                continue
            endpoint = lanes[index, -1, :2]
            x.append(float(endpoint[0]))
            y.append(float(endpoint[1]))
            traffic_indices.append(int(index))
            is_arrow = (
                latest.shape[-1] > TrafficLightIndex.IS_ARROW
                and latest[TrafficLightIndex.IS_ARROW] > 0.5
            )
            symbols.append("triangle-up" if is_arrow else "circle")
        if x:
            traces.append(
                go.Scattergl(
                    x=x,
                    y=y,
                    customdata=traffic_indices,
                    mode="markers",
                    marker={"size": 8, "color": color, "symbol": symbols},
                    name=name,
                    legendgroup=f"{legendgroup}-traffic",
                    showlegend=lane_key == "route_lanes",
                    hovertemplate=(
                        f"{name}<br>{traffic_key} index=%{{customdata}}"
                        "<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>"
                    ),
                )
            )
    return traces


def create_map_element_traces(
    frame: FrameData, style: VisualizerStyle
) -> list[BaseTraceType]:
    """Create intersection-area, stop-line, and road-border traces."""
    traces: list[BaseTraceType] = []
    intersection_areas = frame["intersection_area"]
    area_lines = []
    for area in intersection_areas[FrameData.valid_rows(intersection_areas)]:
        area_lines.append(np.concatenate((area, area[:1]), axis=0))
    x, y = _joined_lines(area_lines)
    if x:
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=style.polygon_color,
                line={"color": style.polygon_color, "width": 1},
                name="Intersection areas",
                legendgroup="map-elements",
                hoverinfo="skip",
            )
        )

    stop_lines = frame["stop_lines"]
    stop_line_trace = _line_trace(
        (line[:, :2] for line in stop_lines[FrameData.valid_rows(stop_lines)]),
        name="Stop lines",
        color=style.stop_line_color,
        width=2.0,
        legendgroup="map-elements",
    )
    if stop_line_trace is not None:
        traces.append(stop_line_trace)

    road_borders = frame["road_borders"]
    road_border_trace = _line_trace(
        (line[:, :2] for line in road_borders[FrameData.valid_rows(road_borders)]),
        name="Road borders",
        color=style.road_border_color,
        width=1.5,
        legendgroup="map-elements",
    )
    if road_border_trace is not None:
        traces.append(road_border_trace)
    return traces


def _pose_lines(array: NDArray[np.generic]) -> list[NDArray[np.generic]]:
    lines: list[NDArray[np.generic]] = []
    if array.ndim == 2:
        valid = FrameData.valid_steps(array)
        if np.any(valid):
            lines.append(array[valid, :2])
        return lines
    for poses in array:
        valid = FrameData.valid_steps(poses)
        if np.any(valid):
            lines.append(poses[valid, :2])
    return lines


def _neighbor_trajectory_trace(
    neighbors: NDArray[np.generic],
    *,
    name: str,
    color: str,
    width: float,
    legendgroup: str = "neighbors",
    dash: str | None = None,
) -> go.Scattergl | None:
    """Create neighbor trajectories with their first-axis index in hover data."""
    x: list[float | None] = []
    y: list[float | None] = []
    neighbor_indices: list[int | None] = []
    for neighbor_index, poses in enumerate(neighbors):
        valid = FrameData.valid_steps(poses)
        if not np.any(valid):
            continue
        points = poses[valid, :2]
        x.extend(float(value) for value in points[:, 0])
        y.extend(float(value) for value in points[:, 1])
        neighbor_indices.extend([neighbor_index] * len(points))
        x.append(None)
        y.append(None)
        neighbor_indices.append(None)
    if not x:
        return None
    return go.Scattergl(
        x=x,
        y=y,
        customdata=neighbor_indices,
        mode="lines",
        name=name,
        legendgroup=legendgroup,
        line={"color": color, "width": width, "dash": dash},
        hovertemplate=(
            "neighbor_index=%{customdata}<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>"
        ),
    )


def create_agent_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[BaseTraceType]:
    """Create ego and neighboring-agent history/future traces."""
    traces: list[BaseTraceType] = []
    if options.show_agent_history:
        ego_past = _line_trace(
            _pose_lines(frame["ego_agent_past"]),
            name="Ego past",
            color=style.past_trajectory_color,
            width=style.history_width,
            legendgroup="ego",
            marker_size=style.trajectory_marker_size,
        )
        if ego_past is not None:
            traces.append(ego_past)

        neighbors = frame["neighbor_agents_past"]
        neighbor_past = _neighbor_trajectory_trace(
            neighbors,
            name="Neighbors past",
            color=style.past_trajectory_color,
            width=style.history_width,
        )
        if neighbor_past is not None:
            traces.append(neighbor_past)
        traces.extend(
            _create_neighbor_boxes(
                neighbors,
                frame["agent_shape"],
                frame["agent_label"],
                style,
            )
        )

    if options.show_agent_future:
        ego_future = frame.get("ego_agent_future")
        if ego_future is not None:
            trace = _line_trace(
                _pose_lines(ego_future),
                name="Ego future",
                color=style.future_trajectory_color,
                width=style.future_width,
                legendgroup="ego",
                marker_size=style.trajectory_marker_size,
            )
            if trace is not None:
                traces.append(trace)
        neighbor_future = frame.get("neighbor_agents_future")
        if neighbor_future is not None:
            trace = _neighbor_trajectory_trace(
                neighbor_future,
                name="Neighbors future",
                color=style.future_trajectory_color,
                width=style.future_width,
            )
            if trace is not None:
                traces.append(trace)
    return traces


def create_prediction_traces(
    prediction: NDArray[np.generic],
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[BaseTraceType]:
    """Create ego and neighbor traces for a sampled `(A, T, 4)` trajectory."""
    if not options.show_agent_prediction:
        return []
    traces: list[BaseTraceType] = []
    ego_prediction = _line_trace(
        _pose_lines(prediction[0]),
        name="Ego prediction",
        color=style.ego_prediction_color,
        width=style.future_width,
        legendgroup="prediction",
        dash="dash",
        marker_size=style.trajectory_marker_size,
    )
    if ego_prediction is not None:
        traces.append(ego_prediction)
    neighbor_prediction = _neighbor_trajectory_trace(
        prediction[1:],
        name="Neighbors prediction",
        color=style.neighbor_prediction_color,
        width=style.future_width,
        legendgroup="prediction",
        dash="dash",
    )
    if neighbor_prediction is not None:
        traces.append(neighbor_prediction)
    return traces


def create_prediction_footprint_traces(
    frame: FrameData,
    prediction: NDArray[np.generic],
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[BaseTraceType]:
    """Create ego footprint outlines along a sampled trajectory."""
    if not options.show_prediction_footprints or len(prediction) == 0:
        return []

    return _create_ego_trajectory_footprint_traces(
        frame,
        prediction[0],
        options.prediction_footprint_stride,
        "Ego prediction footprints",
        "prediction",
        style.ego_prediction_color,
    )


def create_ego_future_footprint_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[BaseTraceType]:
    """Create ego footprint outlines along the labeled future trajectory."""
    ego_future = frame.get("ego_agent_future")
    if (
        not options.show_ego_future_footprints
        or ego_future is None
        or len(ego_future) == 0
    ):
        return []

    return _create_ego_trajectory_footprint_traces(
        frame,
        ego_future,
        options.ego_future_footprint_stride,
        "Ego future footprints",
        "ego",
        style.ego_future_color,
    )


def _create_ego_trajectory_footprint_traces(
    frame: FrameData,
    trajectory: NDArray[np.generic],
    stride: int,
    name: str,
    legendgroup: str,
    color: str,
) -> list[BaseTraceType]:
    """Create sampled ego footprint outlines for one trajectory."""

    base_link_to_front, length, width = (
        float(value) for value in frame["ego_shape"][:3]
    )
    base_link_to_rear = length - base_link_to_front
    local_corners = np.array(
        [
            [-base_link_to_rear, -width / 2],
            [base_link_to_front, -width / 2],
            [base_link_to_front, width / 2],
            [-base_link_to_rear, width / 2],
            [-base_link_to_rear, -width / 2],
        ]
    )
    step_indices = list(range(0, len(trajectory), stride))
    if step_indices and step_indices[-1] != len(trajectory) - 1:
        step_indices.append(len(trajectory) - 1)

    x: list[float | None] = []
    y: list[float | None] = []
    customdata: list[int | None] = []
    for step in step_indices:
        state = trajectory[step]
        cos_yaw, sin_yaw = float(state[2]), float(state[3])
        if cos_yaw**2 + sin_yaw**2 <= 0.5:
            continue
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = local_corners @ rotation.T + state[:2]
        x.extend(float(value) for value in corners[:, 0])
        y.extend(float(value) for value in corners[:, 1])
        customdata.extend([step] * len(corners))
        x.append(None)
        y.append(None)
        customdata.append(None)
    if not x:
        return []
    return [
        go.Scattergl(
            x=x,
            y=y,
            customdata=customdata,
            mode="lines",
            name=name,
            legendgroup=legendgroup,
            line={"color": color, "width": 1},
            hovertemplate="step=%{customdata}<extra></extra>",
        )
    ]


def _create_neighbor_boxes(
    neighbors: NDArray[np.generic],
    agent_shapes: NDArray[np.generic],
    agent_labels: NDArray[np.generic],
    style: VisualizerStyle,
) -> list[BaseTraceType]:
    """Create an oriented footprint box for each neighbor's current state."""
    valid_indices = np.flatnonzero(FrameData.valid_rows(neighbors))
    valid_neighbors = neighbors[valid_indices]
    if len(valid_neighbors) == 0:
        return []
    current = valid_neighbors[:, -1]
    shapes = agent_shapes[valid_indices]
    labels = agent_labels[valid_indices]
    label_names = np.array(["vehicle", "pedestrian", "bicycle"])
    traces: list[BaseTraceType] = []
    for neighbor_index, state, shape, label_one_hot in zip(
        valid_indices, current, shapes, labels, strict=True
    ):
        x = float(state[NeighborIndex.X])
        y = float(state[NeighborIndex.Y])
        cos_yaw = float(state[NeighborIndex.COS_YAW])
        sin_yaw = float(state[NeighborIndex.SIN_YAW])
        width = float(shape[AgentShapeIndex.WIDTH])
        length = float(shape[AgentShapeIndex.LENGTH])
        label = (
            label_names[int(np.argmax(label_one_hot))]
            if np.any(
                label_one_hot[
                    AgentLabelIndex.IS_VEHICLE : AgentLabelIndex.IS_BICYCLE + 1
                ]
            )
            else "unknown"
        )
        if width <= 0.0 or length <= 0.0 or cos_yaw**2 + sin_yaw**2 <= 0.5:
            continue

        corners = np.array(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
                [-length / 2, -width / 2],
            ]
        )
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = corners @ rotation.T + np.array([x, y])
        customdata = np.tile(
            np.array([neighbor_index, label, x, y, length, width], dtype=object),
            (len(corners), 1),
        )
        traces.append(
            go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(217, 119, 6, 0.25)",
                line={"color": style.neighbor_color, "width": 1.5},
                customdata=customdata,
                name="Neighbors current",
                legendgroup="neighbors",
                showlegend=False,
                hovertemplate=(
                    "neighbor_index=%{customdata[0]}<br>%{customdata[1]}"
                    "<br>x=%{customdata[2]:.2f} m<br>y=%{customdata[3]:.2f} m"
                    "<br>length=%{customdata[4]:.2f} m"
                    "<br>width=%{customdata[5]:.2f} m<extra></extra>"
                ),
            )
        )
    return traces


def create_annotation_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[BaseTraceType]:
    """Create ego footprint and goal-pose traces."""
    traces: list[BaseTraceType] = []
    ego_past = frame["ego_agent_past"]
    current = ego_past[-1]
    if options.show_ego_shape:
        base_link_to_front, length, width = (
            float(value) for value in frame["ego_shape"][:3]
        )
        base_link_to_rear = length - base_link_to_front
        corners = np.array(
            [
                [-base_link_to_rear, -width / 2],
                [base_link_to_front, -width / 2],
                [base_link_to_front, width / 2],
                [-base_link_to_rear, width / 2],
                [-base_link_to_rear, -width / 2],
            ]
        )
        cos_yaw, sin_yaw = float(current[2]), float(current[3])
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = corners @ rotation.T + current[:2]
        traces.append(
            go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(220, 38, 38, 0.25)",
                line={"color": style.ego_color, "width": 2},
                name="Ego footprint",
                legendgroup="ego",
                customdata=np.full((len(corners), 1), base_link_to_front),
                hovertemplate="base_link_to_front=%{customdata[0]:.2f} m<extra></extra>",
            )
        )

    if options.show_goal:
        goal = frame["goal_pose"]
        heading_length = 3.0
        x = float(goal[0])
        y = float(goal[1])
        dx = heading_length * float(goal[2])
        dy = heading_length * float(goal[3])
        traces.append(
            go.Scattergl(
                x=[x, x + dx],
                y=[y, y + dy],
                mode="lines+markers",
                marker={"size": [10, 5], "color": style.goal_color},
                line={"color": style.goal_color, "width": 3},
                name="Goal pose",
                legendgroup="goal",
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<extra>Goal</extra>",
            )
        )
    return traces


def create_frame_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
    prediction: NDArray[np.generic] | None = None,
) -> list[BaseTraceType]:
    """Create all enabled traces for one frame."""
    return [
        *create_map_element_traces(frame, style),
        *create_lane_traces(frame, style, options),
        *create_agent_traces(frame, style, options),
        *create_ego_future_footprint_traces(frame, style, options),
        *(
            create_prediction_traces(prediction, style, options)
            if prediction is not None
            else []
        ),
        *(
            create_prediction_footprint_traces(frame, prediction, style, options)
            if prediction is not None
            else []
        ),
        *create_annotation_traces(frame, style, options),
    ]
