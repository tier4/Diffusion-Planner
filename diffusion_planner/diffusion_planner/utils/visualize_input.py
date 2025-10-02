from pathlib import Path

import matplotlib

# 高速化のためAggバックエンドを使用（GUIなし、ファイル出力特化）
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from diffusion_planner.dimensions import *
from diffusion_planner.utils.normalizer import ObservationNormalizer


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


def turn_indicator_int_to_str(turn_indicator):
    """Convert turn indicator integer to string."""
    if turn_indicator == 0:
        return "turn_indicator=0"
    if turn_indicator == 1:
        return "None"
    elif turn_indicator == 2:
        return "<-"
    elif turn_indicator == 3:
        return "->"
    else:
        raise ValueError(f"Unknown turn command: {turn_indicator}")


def draw_bounding_box(ax, x, y, cos, sin, len_x, len_y, color, alpha):
    """
    Draw a bounding box at the specified position with given dimensions and heading.

    Args:
        ax: matplotlib axis
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
        ax.add_line(
            plt.Line2D(
                [x + rot_cdx, x + rot_ndx],
                [y + rot_cdy, y + rot_ndy],
                color=line_color,
                alpha=alpha,
                linewidth=1,
            )
        )


def draw_ego_vehicle(ax, inputs):
    """Draw ego vehicle, its past and future trajectories."""
    ego_state = inputs["ego_current_state"][0]
    ego_x, ego_y = ego_state[0], ego_state[1]
    ego_heading = np.arctan2(ego_state[3], ego_state[2])

    # Ego vehicle's length and width
    car_length = 4.5
    car_width = 2.0
    dx = car_length / 2 * np.cos(ego_heading)
    dy = car_length / 2 * np.sin(ego_heading)

    # Draw the ego vehicle as an arrow
    ax.arrow(
        ego_x,
        ego_y,
        dx,
        dy,
        width=car_width / 2,
        head_width=car_width,
        head_length=car_length / 3,
        fc="r",
        ec="r",
        alpha=0.7,
    )

    # Draw past trajectory
    if "ego_agent_past" in inputs:
        ego_past = inputs["ego_agent_past"][0]
        ax.plot(
            ego_past[:, 0],
            ego_past[:, 1],
            color="orange",
            alpha=0.5,
            linestyle="--",
            label="Ego Past Trajectory",
        )

    # Draw future trajectory
    if "ego_agent_future" in inputs:
        ego_future = inputs["ego_agent_future"][0]
        valid_indices = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if np.any(valid_indices):
            valid_future = ego_future[valid_indices]
            t_values = np.linspace(0, 1, len(valid_future))
            colors = [[1.0 * t, 0.0, 1.0 * (1 - t)] for t in t_values]
            ax.scatter(valid_future[:, 0], valid_future[:, 1], c=colors, alpha=0.5, s=20)

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


def draw_neighbor_agents(ax, inputs):
    """Draw neighbor agents with their trajectories and bounding boxes."""
    neighbors = inputs["neighbor_agents_past"][0]
    last_timestep = neighbors.shape[1] - 1

    past_lines = []
    past_colors = []
    current_boxes = []
    velocity_arrows = []
    future_scatter_x = []
    future_scatter_y = []
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
                future_scatter_x.extend(valid_future[:, 0])
                future_scatter_y.extend(valid_future[:, 1])
                t_values = np.linspace(0, 1, len(valid_future))
                colors = [[1.0 * t, 0.0, 1.0 * (1 - t)] for t in t_values]
                future_scatter_colors.extend(colors)

        # Collect velocity arrows
        v = np.sqrt(vel_x**2 + vel_y**2)
        if v > 0.1:
            velocity_arrows.append((n_x, n_y, vel_x / 2, vel_y / 2))

    # Batch drawing
    if past_lines:
        lc_past = LineCollection(
            past_lines, colors=past_colors, alpha=0.6, linewidths=1, linestyles="--"
        )
        ax.add_collection(lc_past)

    if current_boxes:
        lc_boxes = LineCollection(current_boxes, colors="gray", alpha=0.5, linewidths=1)
        ax.add_collection(lc_boxes)

    if future_scatter_x:
        ax.scatter(future_scatter_x, future_scatter_y, c=future_scatter_colors, alpha=0.5, s=8)

    # Draw velocity arrows
    for x, y, dx, dy in velocity_arrows:
        ax.arrow(
            x,
            y,
            dx,
            dy,
            width=0.2,
            head_width=0.5,
            head_length=0.3,
            fc="orange",
            ec="orange",
            alpha=0.6,
        )


def draw_static_objects(ax, inputs):
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

        rect = plt.Rectangle(
            (obj_x - obj_length / 2, obj_y - obj_width / 2),
            obj_length,
            obj_width,
            angle=np.degrees(obj_heading),
            color=obj_color,
            alpha=0.4,
        )
        ax.add_patch(rect)


def draw_lanes(ax, inputs):
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

    if lane_lines:
        lc_lanes = LineCollection(lane_lines, colors=lane_colors, alpha=0.25, linewidths=1)
        ax.add_collection(lc_lanes)


def draw_route(ax, inputs):
    """Draw route lanes."""
    route_lanes = inputs["route_lanes"][0]

    for i in range(route_lanes.shape[0]):
        ax.plot(
            route_lanes[i, :, 0],
            route_lanes[i, :, 1],
            alpha=0.5,
            linewidth=2,
            color="olive",
            linestyle="--",
        )


def draw_goal_pose(ax, inputs):
    """Draw goal pose."""
    if "goal_pose" in inputs:
        goal_x, goal_y, goal_cos, goal_sin = inputs["goal_pose"][0]
        goal_dx = 2 * goal_cos
        goal_dy = 2 * goal_sin
        ax.arrow(
            goal_x,
            goal_y,
            goal_dx,
            goal_dy,
            width=0.5,
            head_width=1.0,
            head_length=1.0,
            fc="blue",
            ec="blue",
            alpha=0.7,
            label="Goal Pose",
        )


def draw_polygons_and_lines(ax, inputs):
    """Draw polygons and line strings."""
    if "polygons" in inputs:
        for i in range(inputs["polygons"].shape[1]):
            polygon = inputs["polygons"][0, i]
            if np.sum(np.abs(polygon)) < 1e-6:
                continue
            ax.fill(polygon[:, 0], polygon[:, 1], color="gray", alpha=0.5)

    if "line_strings" in inputs:
        for i in range(inputs["line_strings"].shape[1]):
            line_string = inputs["line_strings"][0, i]
            if np.sum(np.abs(line_string)) < 1e-6:
                continue
            ax.plot(line_string[:, 0], line_string[:, 1], color="red")


def setup_axis(ax, ego_x, ego_y, ego_state, view_range, inputs):
    """Setup axis properties and add status text."""
    ego_vel_x, ego_vel_y = ego_state[4], ego_state[5]
    ego_acc_x, ego_acc_y = ego_state[6], ego_state[7]
    ego_steering = ego_state[8]
    ego_yaw_rate = ego_state[9]

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(ego_x - view_range, ego_x + view_range)
    ax.set_ylim(ego_y - view_range, ego_y + view_range)

    # Handle turn indicator
    turn_indicator_text_gt = "There is no turn command"
    turn_indicator_text_pred = "There is no predicted turn command"

    if "turn_indicators" in inputs:
        turn_indicator = inputs["turn_indicators"][0][-1]
        turn_indicator_text_gt = turn_indicator_int_to_str(turn_indicator)

    if "turn_indicator_pred" in inputs:
        turn_indicator_pred = inputs["turn_indicator_pred"]
        turn_indicator_text_pred = turn_indicator_int_to_str(turn_indicator_pred)

    ax.text(
        view_range - 1,
        view_range - 1,
        f"VelocityX: {ego_vel_x:.2f} m/s\n"
        f"VelocityY: {ego_vel_y:.2f} m/s\n"
        f"AccelerationX: {ego_acc_x:.2f} m/s²\n"
        f"AccelerationY: {ego_acc_y:.2f} m/s²\n"
        f"Steering: {ego_steering:.2f} rad\n"
        f"Yaw Rate: {ego_yaw_rate:.2f} rad/s\n"
        f"Turn Command GT: {turn_indicator_text_gt}\n"
        f"Turn Command PR: {turn_indicator_text_pred}",
        fontsize=8,
        color="red",
        ha="right",
        va="top",
    )


def visualize_inputs(
    inputs: dict,
    obs_normalizer: ObservationNormalizer,
    save_path: Path | None = None,
    ax: None = None,
    view_ranges: list = None,
):
    """
    Draw the input data of the diffusion_planner model on the xy plane.

    Args:
        inputs: Input data dictionary
        obs_normalizer: Observation normalizer
        save_path: Path to save the visualization
        ax: Matplotlib axis (for single plot compatibility)
        view_ranges: List of view ranges in meters [60, 120] for multi-range visualization

    Returns:
        For single range: ax
        For multi-range: (fig, axes)
    """
    # Default behavior: single 60m range for backward compatibility
    if view_ranges is None:
        view_ranges = [60]

    # Prepare data
    inputs = obs_normalizer.inverse(inputs)

    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    for key in inputs:
        inputs[key] = to_numpy(inputs[key])

    # Handle single axis case for backward compatibility
    if ax is not None and len(view_ranges) == 1:
        # Single plot mode
        ego_x, ego_y, ego_state = draw_ego_vehicle(ax, inputs)
        draw_neighbor_agents(ax, inputs)
        draw_static_objects(ax, inputs)
        draw_lanes(ax, inputs)
        draw_route(ax, inputs)
        draw_goal_pose(ax, inputs)
        draw_polygons_and_lines(ax, inputs)
        setup_axis(ax, ego_x, ego_y, ego_state, view_ranges[0], inputs)

        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
        return ax

    # Multi-plot mode
    fig, axes = plt.subplots(1, len(view_ranges), figsize=(10 * len(view_ranges), 8))
    if len(view_ranges) == 1:
        axes = [axes]

    for i, view_range in enumerate(view_ranges):
        current_ax = axes[i]

        # Draw all components
        ego_x, ego_y, ego_state = draw_ego_vehicle(current_ax, inputs)
        draw_neighbor_agents(current_ax, inputs)
        draw_static_objects(current_ax, inputs)
        draw_lanes(current_ax, inputs)
        draw_route(current_ax, inputs)
        draw_goal_pose(current_ax, inputs)
        draw_polygons_and_lines(current_ax, inputs)
        setup_axis(current_ax, ego_x, ego_y, ego_state, view_range, inputs)

        # Add title to distinguish different ranges
        current_ax.set_title(f"View Range: {view_range}m")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        return fig, axes

    return fig, axes
