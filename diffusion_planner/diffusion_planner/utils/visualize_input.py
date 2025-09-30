from pathlib import Path

import matplotlib

# 高速化のためAggバックエンドを使用（GUIなし、ファイル出力特化）
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from diffusion_planner.utils.normalizer import ObservationNormalizer


def draw_bounding_box(ax, x, y, heading, len_x, len_y, color, alpha):
    """
    Draw a bounding box at the specified position with given dimensions and heading.

    Args:
        ax: matplotlib axis
        x, y: center position
        heading: orientation in radians
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
        rot_cdx = curr_dx * np.cos(heading) - curr_dy * np.sin(heading)
        rot_cdy = curr_dx * np.sin(heading) + curr_dy * np.cos(heading)
        rot_ndx = next_dx * np.cos(heading) - next_dy * np.sin(heading)
        rot_ndy = next_dx * np.sin(heading) + next_dy * np.cos(heading)

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


def visualize_inputs(
    inputs: dict,
    obs_normalizer: ObservationNormalizer,
    save_path: Path | None = None,
    ax: None = None,
):
    """
    draw the input data of the diffusion_planner model on the xy plane
    """
    view_range = 60
    inputs = obs_normalizer.inverse(inputs)

    # Function to convert PyTorch tensors to NumPy arrays
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    for key in inputs:
        inputs[key] = to_numpy(inputs[key])

    """
    for key in inputs:
        print(f"{key}={inputs[key].shape}")

    ego_agent_past=(1, 20, 3)
    ego_current_state=(1, 10)
    ego_agent_future=(1, 80, 3)
    neighbor_agents_past=(1, 32, 21, 11)
    neighbor_agents_future=(1, 32, 80, 3)
    static_objects=(1, 5, 10)
    lanes=(1, 70, 20, 13)
    lanes_speed_limit=(1, 70, 1)
    lanes_has_speed_limit=(1, 70, 1)
    route_lanes=(1, 25, 20, 13)
    route_lanes_speed_limit=(1, 25, 1)
    route_lanes_has_speed_limit=(1, 25, 1)
    turn_indicator=(1,)
    goal_pose=(1, 4)
    polygons,
    line_strings
    """

    # initialize the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # ==== Ego ====
    ego_state = inputs["ego_current_state"][0]  # Use the first sample in the batch
    ego_x, ego_y = ego_state[0], ego_state[1]
    ego_heading = np.arctan2(ego_state[3], ego_state[2])
    ego_vel_x = ego_state[4]
    ego_vel_y = ego_state[5]
    ego_acc_x = ego_state[6]
    ego_acc_y = ego_state[7]
    ego_steering = ego_state[8]
    ego_yaw_rate = ego_state[9]

    # Ego vehicle's length and width
    car_length = 4.5  # Assumed value for vehicle length
    car_width = 2.0  # Assumed value for vehicle width
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

    if "ego_agent_past" in inputs:
        ego_past = inputs["ego_agent_past"][0]  # Use the first sample in the batch
        ego_past_x = ego_past[:, 0]
        ego_past_y = ego_past[:, 1]
        ax.plot(
            ego_past_x,
            ego_past_y,
            color="orange",
            alpha=0.5,
            linestyle="--",
            label="Ego Past Trajectory",
        )

    if "ego_agent_future" in inputs:
        ego_future = inputs["ego_agent_future"][0]

        # 有効な未来軌跡点のみを抽出
        valid_indices = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if np.any(valid_indices):
            valid_future = ego_future[valid_indices]

            # 一括scatter描画
            t_values = np.linspace(0, 1, len(valid_future))
            colors = [[1.0 * t, 0.0, 1.0 * (1 - t)] for t in t_values]
            ax.scatter(valid_future[:, 0], valid_future[:, 1], c=colors, alpha=0.5, s=20)

        # Draw bounding boxes at 4 seconds and 8 seconds for ego vehicle
        for j in [40 - 1, 80 - 1]:  # 4 seconds and 8 seconds
            ego_future_x = ego_future[j, 0]
            ego_future_y = ego_future[j, 1]
            if ego_future_x == 0 and ego_future_y == 0:
                continue
            draw_bounding_box(
                ax,
                ego_future_x,
                ego_future_y,
                ego_future[j, 2],
                car_length,
                car_width,
                "orange",
                0.1,
            )

    # ==== Neighbor agents ====
    neighbors = inputs["neighbor_agents_past"][0]  # Use the first sample in the batch
    last_timestep = neighbors.shape[1] - 1

    # データを事前に収集して一括描画
    past_lines = []
    past_colors = []
    current_boxes = []
    velocity_arrows = []
    future_scatter_x = []
    future_scatter_y = []
    future_scatter_colors = []

    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i, last_timestep]

        # Skip zero vectors (masked objects)
        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            continue

        n_x, n_y = neighbor[0], neighbor[1]
        n_heading = np.arctan2(neighbor[3], neighbor[2])
        vel_x, vel_y = neighbor[4], neighbor[5]
        len_y, len_x = neighbor[6], neighbor[7]

        # Set color and shape dimensions based on the vehicle type
        vehicle_type = np.argmax(neighbor[8:11]) if neighbor.shape[0] > 8 else 0
        if vehicle_type == 0:  # Vehicle
            color = "blue"
        elif vehicle_type == 1:  # Pedestrian
            color = "green"
        else:  # Bicycle
            color = "purple"

        # 過去の軌跡データを収集（LineCollectionで使用）
        past_points = np.array(
            [[neighbors[i, t, 0], neighbors[i, t, 1]] for t in range(last_timestep + 1)]
        )
        if len(past_points) > 1:
            past_lines.append(past_points)
            past_colors.append(color)

        # Bounding boxの線を収集
        box_lines = []
        dx_coeff = [+1, +1, -1, -1]
        dy_coeff = [+1, -1, -1, +1]
        for d in range(4):
            curr_dx = dx_coeff[d] * (len_x / 2)
            curr_dy = dy_coeff[d] * (len_y / 2)
            next_dx = dx_coeff[(d + 1) % 4] * (len_x / 2)
            next_dy = dy_coeff[(d + 1) % 4] * (len_y / 2)

            rot_cdx = curr_dx * np.cos(n_heading) - curr_dy * np.sin(n_heading)
            rot_cdy = curr_dx * np.sin(n_heading) + curr_dy * np.cos(n_heading)
            rot_ndx = next_dx * np.cos(n_heading) - next_dy * np.sin(n_heading)
            rot_ndy = next_dx * np.sin(n_heading) + next_dy * np.cos(n_heading)

            box_lines.append([[n_x + rot_cdx, n_y + rot_cdy], [n_x + rot_ndx, n_y + rot_ndy]])
        current_boxes.extend(box_lines)

        # 未来の軌跡データを収集
        if "neighbor_agents_future" in inputs:
            neighbor_future = inputs["neighbor_agents_future"][0][i]
            valid_indices = ~((neighbor_future[:, 0] == 0) & (neighbor_future[:, 1] == 0))
            if np.any(valid_indices):
                valid_future = neighbor_future[valid_indices]
                future_scatter_x.extend(valid_future[:, 0])
                future_scatter_y.extend(valid_future[:, 1])

                # 色のグラデーション
                t_values = np.linspace(0, 1, len(valid_future))
                colors = [[1.0 * t, 0.0, 1.0 * (1 - t)] for t in t_values]
                future_scatter_colors.extend(colors)

        # 速度矢印データを収集（簡略化）
        v = np.sqrt(vel_x**2 + vel_y**2)
        if v > 0.1:  # 最小速度閾値
            velocity_arrows.append((n_x, n_y, vel_x / 2, vel_y / 2))

    # 一括描画の実行
    if past_lines:
        # 過去の軌跡をLineCollectionで一括描画
        lc_past = LineCollection(
            past_lines, colors=past_colors, alpha=0.6, linewidths=1, linestyles="--"
        )
        ax.add_collection(lc_past)

    if current_boxes:
        # Bounding boxesをLineCollectionで一括描画
        lc_boxes = LineCollection(current_boxes, colors="gray", alpha=0.5, linewidths=1)
        ax.add_collection(lc_boxes)

    if future_scatter_x:
        # 未来の軌跡を一括scatter描画
        ax.scatter(future_scatter_x, future_scatter_y, c=future_scatter_colors, alpha=0.5, s=8)

    # 速度矢印（数を制限して描画）
    for arrow_data in velocity_arrows[:10]:  # 最大10個まで
        x, y, dx, dy = arrow_data
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

    # ==== Static objects ====
    static_objects = inputs["static_objects"][0]  # Use the first sample in the batch

    for i in range(static_objects.shape[0]):
        obj = static_objects[i]

        # Skip zero vectors (masked objects)
        if np.sum(np.abs(obj[:4])) < 1e-6:
            continue

        obj_x, obj_y = obj[0], obj[1]
        obj_heading = np.arctan2(obj[3], obj[2])
        obj_width = obj[4] if obj.shape[0] > 4 else 1.0
        obj_length = obj[5] if obj.shape[0] > 5 else 1.0

        # Set color based on the object type
        obj_type = np.argmax(obj[-4:]) if obj.shape[0] >= 10 else 0
        colors = ["orange", "gray", "yellow", "brown"]
        obj_color = colors[obj_type % len(colors)]

        # Draw the object as a rectangle
        rect = plt.Rectangle(
            (obj_x - obj_length / 2, obj_y - obj_width / 2),
            obj_length,
            obj_width,
            angle=np.degrees(obj_heading),
            color=obj_color,
            alpha=0.4,
        )
        ax.add_patch(rect)

    def get_traffic_light_color(traffic_light):
        if traffic_light[0] == 1:
            return "green"
        elif traffic_light[1] == 1:
            return "yellow"
        elif traffic_light[2] == 1:
            return "red"
        elif traffic_light[3] == 1:
            return "gray"
        elif traffic_light[4] == 1:
            return "black"
        return "purple"

    # ==== Lanes ====
    lanes = inputs["lanes"][0]  # Use the first sample in the batch

    # Lane境界線をLineCollectionで一括描画
    lane_lines = []
    lane_colors = []

    for i in range(lanes.shape[0]):
        traffic_light = lanes[i, 0, 8:13]
        color = get_traffic_light_color(traffic_light)

        # 左境界線
        lx = lanes[i, :, 0] + lanes[i, :, 4]
        ly = lanes[i, :, 1] + lanes[i, :, 5]
        left_points = np.array([lx, ly]).T
        lane_lines.append(left_points)
        lane_colors.append(color)

        # 右境界線
        rx = lanes[i, :, 0] + lanes[i, :, 6]
        ry = lanes[i, :, 1] + lanes[i, :, 7]
        right_points = np.array([rx, ry]).T
        lane_lines.append(right_points)
        lane_colors.append(color)

    if lane_lines:
        lc_lanes = LineCollection(lane_lines, colors=lane_colors, alpha=0.25, linewidths=1)
        ax.add_collection(lc_lanes)

    # ==== Route ====
    route_lanes = inputs["route_lanes"][0]  # Use the first sample in the batch
    route_lanes_speed_limit = inputs["route_lanes_speed_limit"][0]
    route_lanes_has_speed_limit = inputs["route_lanes_has_speed_limit"][0]

    for i in range(route_lanes.shape[0]):
        traffic_light = route_lanes[i, 0, 8:13]
        color = get_traffic_light_color(traffic_light)

        # center line
        ax.plot(
            route_lanes[i, :, 0],
            route_lanes[i, :, 1],
            alpha=0.5,
            linewidth=2,
            color="olive",
            linestyle="--",
        )

        # print speed limit
        # ax.text(
        #     (left_x + next_left_x) / 2,
        #     (left_y + next_left_y) / 2,
        #     f"Limit({route_lanes_has_speed_limit[i][0]})={route_lanes_speed_limit[i][0]:.1f}",
        #     fontsize=8,
        #     color="black",
        # )

    # ==== Goal Pose ====
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

    # ==== Polygons ====
    if "polygons" in inputs:
        for i in range(inputs["polygons"].shape[1]):
            polygon = inputs["polygons"][0, i]
            if np.sum(np.abs(polygon)) < 1e-6:
                continue
            ax.fill(polygon[:, 0], polygon[:, 1], color="gray", alpha=0.5)

    # ==== Line Strings ====
    if "line_strings" in inputs:
        for i in range(inputs["line_strings"].shape[1]):
            line_string = inputs["line_strings"][0, i]
            if np.sum(np.abs(line_string)) < 1e-6:
                continue
            ax.plot(line_string[:, 0], line_string[:, 1], color="red")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # print status
    def turn_indicator_int_to_str(turn_indicator):
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

    if "turn_indicator" in inputs:
        turn_indicator = inputs["turn_indicator"][0]
        turn_indicator_text_gt = turn_indicator_int_to_str(turn_indicator)
    else:
        turn_indicator_text_gt = "There is no turn command"

    if "turn_indicator_pred" in inputs:
        turn_indicator_pred = inputs["turn_indicator_pred"]
        turn_indicator_text_pred = turn_indicator_int_to_str(turn_indicator_pred)
    else:
        turn_indicator_text_pred = "There is no predicted turn command"

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

    ax.set_xlim(ego_x - view_range, ego_x + view_range)
    ax.set_ylim(ego_y - view_range, ego_y + view_range)

    if save_path is None:
        return ax

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
