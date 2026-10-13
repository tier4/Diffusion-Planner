from pathlib import Path

import numpy as np
import torch

from diffusion_planner.dimensions import *
from scenario_generation.canvas import Figure, Panel

_NAMED_COLORS = {
    "r": "#ff0000",
    "red": "#ff0000",
    "green": "#008000",
    "yellow": "#ffff00",
    "purple": "#800080",
    "black": "#000000",
    "blue": "#0000ff",
    "orange": "#ffa500",
    "gray": "#808080",
    "brown": "#a52a2a",
    "olive": "#808000",
}


def _hex(color: str) -> str:
    """A ``#rrggbb`` colour for a matplotlib colour name, or the colour itself when already hex."""
    return color if color.startswith("#") else _NAMED_COLORS[color]


def fancy_arrow(x, y, dx, dy, width, head_width, head_length) -> np.ndarray:
    """Outline of a matplotlib ``FancyArrow`` from ``(x, y)`` along ``(dx, dy)`` plus its head."""
    distance = float(np.hypot(dx, dy))
    length = distance + head_length
    half = np.array(
        [
            [0.0, 0.0],
            [-head_length, -head_width / 2],
            [-head_length, -width / 2],
            [-length, -width / 2],
            [-length, 0.0],
        ]
    )
    half[:, 0] += head_length
    outline = np.concatenate([half[:-1], (half * [1, -1])[-2::-1]])
    cos, sin = (dx / distance, dy / distance) if distance > 0 else (1.0, 0.0)
    rotated = outline @ np.array([[cos, sin], [-sin, cos]])
    return rotated + np.array([x + dx, y + dy])


def get_traffic_light_color(traffic_light):
    """Get traffic light color from traffic light array."""
    if traffic_light[TRAFFIC_LIGHT_GREEN - TRAFFIC_LIGHT] == 1:
        return "green"
    elif traffic_light[TRAFFIC_LIGHT_YELLOW - TRAFFIC_LIGHT] == 1:
        return "yellow"
    elif traffic_light[TRAFFIC_LIGHT_RED - TRAFFIC_LIGHT] == 1:
        return "red"
    elif traffic_light[TRAFFIC_LIGHT_WHITE - TRAFFIC_LIGHT] == 1:
        return "purple"
    elif traffic_light[TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT - TRAFFIC_LIGHT] == 1:
        return "black"
    else:
        return "purple"
        # raise ValueError(f"Unknown traffic light state: {traffic_light}")


_TURN_INDICATOR_LABELS = {
    0: "Straight",
    1: "Straight",
    2: "Left",
    3: "Right",
    4: "Keep",
}


def turn_indicator_int_to_str(turn_indicator):
    """Convert turn indicator integer to string."""
    label = _TURN_INDICATOR_LABELS.get(int(turn_indicator))
    if label is None:
        raise ValueError(f"Unknown turn command: {turn_indicator}")
    return label


def draw_bounding_box(ax: Panel, x, y, cos, sin, len_x, len_y, color, alpha):
    """
    Draw a bounding box at the specified position with given dimensions and heading.

    Args:
        ax: canvas panel
        x, y: center position
        cos, sin: orientation in radians
        len_x, len_y: length and width of the bounding box
        color: color of the bounding box
        alpha: transparency
    """
    dx_coeff = [+1, +1, -1, -1]
    dy_coeff = [+1, -1, -1, +1]
    for d in range(4):
        curr_dx = dx_coeff[(d + 0) % 4] * (len_x / 2)
        curr_dy = dy_coeff[(d + 0) % 4] * (len_y / 2)
        next_dx = dx_coeff[(d + 1) % 4] * (len_x / 2)
        next_dy = dy_coeff[(d + 1) % 4] * (len_y / 2)
        # rotate
        rot_cdx = curr_dx * cos - curr_dy * sin
        rot_cdy = curr_dx * sin + curr_dy * cos
        rot_ndx = next_dx * cos - next_dy * sin
        rot_ndy = next_dx * sin + next_dy * cos

        line_color = "red" if (d == 0) else color
        ax.lines(
            [np.array([[x + rot_cdx, y + rot_cdy], [x + rot_ndx, y + rot_ndy]])],
            _hex(line_color),
            width=1,
            alpha=alpha,
            z=2,
        )


def draw_ego_vehicle(ax: Panel, inputs, show_future=True):
    """Draw ego vehicle, its past and future trajectories."""
    ego_state = inputs["ego_current_state"][0]
    ego_x, ego_y = ego_state[0], ego_state[1]
    ego_heading = np.arctan2(ego_state[3], ego_state[2])

    # Ego vehicle's length and width
    shape = np.asarray(inputs["ego_shape"]).reshape(-1)
    car_length = float(shape[1])
    car_width = float(shape[2])
    dx = car_length / 2 * np.cos(ego_heading)
    dy = car_length / 2 * np.sin(ego_heading)

    # Draw the ego vehicle as an arrow
    ax.polygons(
        [fancy_arrow(ego_x, ego_y, dx, dy, car_width / 2, car_width, car_length / 3)],
        fill=_hex("r"),
        edge=_hex("r"),
        width=1.0,
        alpha=0.7,
        z=1,
    )

    # Draw past trajectory
    if "ego_agent_past" in inputs:
        ego_past = inputs["ego_agent_past"][0]
        ax.lines(
            [np.stack([ego_past[:, 0], ego_past[:, 1]], 1)],
            _hex("orange"),
            alpha=0.5,
            dash="dash",
            z=2,
            label="Ego Past Trajectory",
        )

        # plot as arrow
        # for t in range(0, ego_past.shape[0], 4):
        #     cos = ego_past[t, 2]
        #     sin = ego_past[t, 3]
        #     ax.arrow(
        #         ego_past[t, 0],
        #         ego_past[t, 1],
        #         cos,
        #         sin,
        #         width=0.3,
        #         head_width=0.8,
        #         head_length=0.3,
        #         fc="orange",
        #         ec="orange",
        #         alpha=0.5,
        #     )

    # Draw future trajectory
    if show_future and "ego_agent_future" in inputs:
        ego_future = inputs["ego_agent_future"][0]
        valid_indices = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if np.any(valid_indices):
            valid_future = ego_future[valid_indices]
            t_values = np.linspace(0, 1, len(valid_future))
            colors = [(1.0 * t, 0.0, 1.0 * (1 - t)) for t in t_values]
            ax.markers(valid_future[:, :2], "o", size=np.sqrt(20), color=colors, alpha=0.5, z=1)

        # plot as arrow
        # for t in range(0, ego_future.shape[0], 4):
        #     if ego_future[t, 0] == 0 and ego_future[t, 1] == 0:
        #         continue
        #     cos = np.cos(ego_future[t, 2])
        #     sin = np.sin(ego_future[t, 2])
        #     ax.arrow(
        #         ego_future[t, 0],
        #         ego_future[t, 1],
        #         cos,
        #         sin,
        #         width=0.3,
        #         head_width=0.8,
        #         head_length=0.3,
        #         fc="blue",
        #         ec="blue",
        #         alpha=0.5,
        #     )

        # Draw bounding boxes at 4 seconds and 8 seconds
        for j in [40 - 1, 80 - 1]:  # 4 seconds and 8 seconds
            if ego_future[j, 0] == 0 and ego_future[j, 1] == 0:
                continue
            cos = np.cos(ego_future[j, 2])
            sin = np.sin(ego_future[j, 2])
            draw_bounding_box(
                ax,
                ego_future[j, 0],
                ego_future[j, 1],
                cos,
                sin,
                car_length,
                car_width,
                "orange",
                0.1,
            )

    return ego_x, ego_y, ego_state


def draw_neighbor_agents(ax: Panel, inputs):
    """Draw neighbor agents with their trajectories and bounding boxes."""
    neighbors = inputs["neighbor_agents_past"][0]
    last_timestep = neighbors.shape[1] - 1

    past_lines = []
    past_colors = []
    velocity_arrows = []
    future_scatter_xy = []
    future_scatter_colors = []

    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i, last_timestep]
        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            continue

        n_x, n_y = neighbor[0], neighbor[1]
        n_cos, n_sin = neighbor[2], neighbor[3]
        vel_x, vel_y = neighbor[4], neighbor[5]
        len_y, len_x = neighbor[6], neighbor[7]

        # Set color based on vehicle type
        vehicle_type = np.argmax(neighbor[8:11]) if neighbor.shape[0] > 8 else 0
        color = ["blue", "green", "purple"][vehicle_type] if vehicle_type < 3 else "blue"

        # Collect past trajectory
        past_points = np.array(
            [[neighbors[i, t, 0], neighbors[i, t, 1]] for t in range(last_timestep + 1)]
        )
        if len(past_points) > 1:
            past_lines.append(past_points)
            past_colors.append(color)

        # Collect bounding box lines
        draw_bounding_box(ax, n_x, n_y, n_cos, n_sin, len_x, len_y, color, 0.5)

        # Collect future trajectory
        if "neighbor_agents_future" in inputs:
            neighbor_future = inputs["neighbor_agents_future"][0][i]
            valid_indices = ~((neighbor_future[:, 0] == 0) & (neighbor_future[:, 1] == 0))
            if np.any(valid_indices):
                valid_future = neighbor_future[valid_indices]
                future_scatter_xy.append(valid_future[:, :2])
                t_values = np.linspace(0, 1, len(valid_future))
                colors = [(1.0 * t, 0.0, 1.0 * (1 - t)) for t in t_values]
                future_scatter_colors.extend(colors)

        # Collect velocity arrows
        v = np.sqrt(vel_x**2 + vel_y**2)
        if v > 0.1:
            velocity_arrows.append((n_x, n_y, vel_x / 2, vel_y / 2))

    # Batch drawing
    for color in dict.fromkeys(past_colors):
        ax.lines(
            [line for line, c in zip(past_lines, past_colors) if c == color],
            _hex(color),
            width=1,
            alpha=0.6,
            dash="dash",
            z=2,
        )

    if future_scatter_xy:
        ax.markers(
            np.concatenate(future_scatter_xy),
            "o",
            size=np.sqrt(8),
            color=future_scatter_colors,
            alpha=0.5,
            z=1,
        )

    # Draw velocity arrows
    if velocity_arrows:
        ax.polygons(
            [fancy_arrow(x, y, dx, dy, 0.2, 0.5, 0.3) for x, y, dx, dy in velocity_arrows],
            fill=_hex("orange"),
            edge=_hex("orange"),
            width=1.0,
            alpha=0.6,
            z=1,
        )


def draw_static_objects(ax: Panel, inputs):
    """Draw static objects."""
    static_objects = inputs["static_objects"][0]

    for i in range(static_objects.shape[0]):
        obj = static_objects[i]
        if np.sum(np.abs(obj[:4])) < 1e-6:
            continue

        obj_x, obj_y = obj[0], obj[1]
        obj_heading = np.arctan2(obj[3], obj[2])
        obj_width = obj[4] if obj.shape[0] > 4 else 1.0
        obj_length = obj[5] if obj.shape[0] > 5 else 1.0

        obj_type = np.argmax(obj[-4:]) if obj.shape[0] >= 10 else 0
        colors = ["orange", "gray", "yellow", "brown"]
        obj_color = colors[obj_type % len(colors)]

        corner = np.array([obj_x - obj_length / 2, obj_y - obj_width / 2])
        cos, sin = np.cos(obj_heading), np.sin(obj_heading)
        local = np.array([[0, 0], [obj_length, 0], [obj_length, obj_width], [0, obj_width]])
        rect = corner + local @ np.array([[cos, sin], [-sin, cos]])
        ax.polygons([rect], fill=_hex(obj_color), edge=_hex(obj_color), width=1.0, alpha=0.4, z=1)


def draw_lanes(ax: Panel, inputs):
    """Draw lane boundaries."""
    lanes = inputs["lanes"][0]
    lane_lines = []
    lane_colors = []

    for i in range(lanes.shape[0]):
        traffic_light = lanes[i, 0, 8:13]
        color = get_traffic_light_color(traffic_light)

        # Left boundary
        lx = lanes[i, :, 0] + lanes[i, :, 4]
        ly = lanes[i, :, 1] + lanes[i, :, 5]
        lane_lines.append(np.array([lx, ly]).T)
        lane_colors.append(color)

        # Right boundary
        rx = lanes[i, :, 0] + lanes[i, :, 6]
        ry = lanes[i, :, 1] + lanes[i, :, 7]
        lane_lines.append(np.array([rx, ry]).T)
        lane_colors.append(color)

    for color in dict.fromkeys(lane_colors):
        ax.lines(
            [line for line, c in zip(lane_lines, lane_colors) if c == color],
            _hex(color),
            width=1,
            alpha=0.25,
            z=2,
        )


def draw_route(ax: Panel, inputs, color="olive", label=None):
    """Draw route lanes."""
    route_lanes = inputs["route_lanes"][0]
    ax.lines(
        [route_lanes[i, :, :2] for i in range(route_lanes.shape[0])],
        _hex(color),
        width=2,
        alpha=0.5,
        dash="dash",
        z=2,
        label=label,
    )


def draw_goal_pose(ax: Panel, inputs):
    """Draw goal pose."""
    if "goal_pose" in inputs:
        goal_x, goal_y, goal_cos, goal_sin = inputs["goal_pose"][0]
        goal_dx = 2 * goal_cos
        goal_dy = 2 * goal_sin
        ax.polygons(
            [fancy_arrow(goal_x, goal_y, goal_dx, goal_dy, 0.5, 1.0, 1.0)],
            fill=_hex("blue"),
            edge=_hex("blue"),
            width=1.0,
            alpha=0.7,
            z=1,
            label="Goal Pose",
        )


def draw_polygons_and_lines(ax: Panel, inputs):
    """Draw polygons and line strings."""
    if "polygons" in inputs:
        for i in range(inputs["polygons"].shape[1]):
            polygon = inputs["polygons"][0, i]
            if np.sum(np.abs(polygon)) < 1e-6:
                continue
            ax.polygons([polygon[:, :2]], fill=_hex("gray"), edge=_hex("gray"), alpha=0.5, z=1)

    if "line_strings" in inputs:
        for i in range(inputs["line_strings"].shape[1]):
            line_string = inputs["line_strings"][0, i]
            if np.sum(np.abs(line_string)) < 1e-6:
                continue
            # Check one-hot type: [x, y, stop_line, road_border]
            is_road_border = line_string.shape[-1] > 3 and np.any(line_string[:, 3] > 0.5)
            if is_road_border:
                ax.lines([line_string[:, :2]], _hex("red"), width=1, z=2)
            else:
                ax.lines([line_string[:, :2]], _hex("orange"), width=1, z=2)


def setup_axis(ax: Panel, ego_x, ego_y, ego_state, view_range, inputs):
    """Setup axis properties and add status text."""
    ego_vel_x, ego_vel_y = ego_state[4], ego_state[5]
    ego_acc_x, ego_acc_y = ego_state[6], ego_state[7]
    ego_steering = ego_state[8]
    ego_yaw_rate = ego_state[9]

    ax.xlabel("X [m]")
    ax.ylabel("Y [m]")
    ax.equal = True
    ax.grid(0.3)
    ax.set_xlim(ego_x - view_range, ego_x + view_range)
    ax.set_ylim(ego_y - view_range, ego_y + view_range)

    # Handle turn indicator
    turn_indicator_text_gt = "There is no turn command"
    turn_indicator_text_pred = "There is no predicted turn command"

    if "turn_indicators" in inputs:
        turn_indicator = inputs["turn_indicators"][0][-1]
        if inputs["turn_indicators"][0][-2] == turn_indicator:
            # Same as previous timestep — prefix with "Keep" to indicate sustained signal
            label = turn_indicator_int_to_str(turn_indicator)
            turn_indicator_text_gt = label if label == "Keep" else f"Keep {label}"
        else:
            turn_indicator_text_gt = turn_indicator_int_to_str(turn_indicator)

    if "turn_indicator_pred" in inputs:
        turn_indicator_pred = inputs["turn_indicator_pred"]
        turn_indicator_text_pred = turn_indicator_int_to_str(turn_indicator_pred)

    ax.text(
        (view_range - 1, view_range - 1),
        f"VelocityX: {ego_vel_x:.2f} m/s\n"
        f"VelocityY: {ego_vel_y:.2f} m/s\n"
        f"AccelerationX: {ego_acc_x:.2f} m/s²\n"
        f"AccelerationY: {ego_acc_y:.2f} m/s²\n"
        f"Steering: {ego_steering:.2f} rad\n"
        f"Yaw Rate: {ego_yaw_rate:.2f} rad/s\n"
        f"Turn Command GT: {turn_indicator_text_gt}\n"
        f"Turn Command PR: {turn_indicator_text_pred}",
        8,
        _hex("red"),
        ha="right",
        va="top",
        z=3,
    )


def visualize_inputs(
    inputs: dict,
    save_path: Path | None = None,
    ax: Panel | None = None,
    view_ranges: list = None,
    annotation: str | None = None,
    annotation_color: str = "red",
    show_neighbors: bool = True,
    show_ego_future: bool = True,
    route_color: str = "olive",
    route_label: str | None = None,
):
    """
    Draw the input data of the diffusion_planner model on the xy plane.

    Args:
        inputs: Input data dictionary
        save_path: Path to save the visualization
        ax: Canvas panel (for single plot compatibility)
        view_ranges: List of view ranges in meters [60, 120] for multi-range visualization
        annotation: Optional banner drawn at the top of the figure. Drawn with a fixed
            size/position so the figure height stays constant across frames (e.g. the
            reason a frame would be dropped, or a faint "OK" for kept frames).
        annotation_color: Colour of the annotation banner.

    Returns:
        For single range: ax
        For multi-range: (fig, axes)
    """
    # Default behavior: single 60m range for backward compatibility
    if view_ranges is None:
        view_ranges = [60]

    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    for key in inputs:
        inputs[key] = to_numpy(inputs[key])

    # Handle single axis case for backward compatibility
    if ax is not None and len(view_ranges) == 1:
        # Single plot mode
        ego_x, ego_y, ego_state = draw_ego_vehicle(ax, inputs, show_future=show_ego_future)
        if show_neighbors:
            draw_neighbor_agents(ax, inputs)
        draw_static_objects(ax, inputs)
        draw_lanes(ax, inputs)
        draw_route(ax, inputs, color=route_color, label=route_label)
        draw_goal_pose(ax, inputs)
        draw_polygons_and_lines(ax, inputs)
        setup_axis(ax, ego_x, ego_y, ego_state, view_ranges[0], inputs)

        if save_path is not None:
            ax.figure.save(save_path)
        return ax

    # Multi-plot mode
    fig = Figure((10 * len(view_ranges), 8), dpi=100)
    axes = fig.grid(1, len(view_ranges), margins=(0.07, 0.98, 0.07, 0.92), wspace=0.2)[0]

    for i, view_range in enumerate(view_ranges):
        current_ax = axes[i]

        # Draw all components
        ego_x, ego_y, ego_state = draw_ego_vehicle(current_ax, inputs, show_future=show_ego_future)
        if show_neighbors:
            draw_neighbor_agents(current_ax, inputs)
        draw_static_objects(current_ax, inputs)
        draw_lanes(current_ax, inputs)
        draw_route(current_ax, inputs, color=route_color, label=route_label)
        draw_goal_pose(current_ax, inputs)
        draw_polygons_and_lines(current_ax, inputs)
        setup_axis(current_ax, ego_x, ego_y, ego_state, view_range, inputs)

        # Add title to distinguish different ranges
        current_ax.title(f"View Range: {view_range}m")

    if annotation is not None:
        fig.suptitle(annotation, size=12, weight="bold", color=_hex(annotation_color))

    if save_path is not None:
        fig.save(save_path)
        return fig, axes

    return fig, axes
