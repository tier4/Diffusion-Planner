import cv2
import numpy as np
import torch


def visualize_inputs_cv2(
    inputs: dict,
    image_size: tuple = (600, 600),
    view_range: float = 100.0,
) -> np.ndarray:
    """
    Draw the input data of the diffusion_planner model as a cv2 image
    Returns the image as a numpy array suitable for neural network input
    """
    for key in inputs:
        inputs[key] = inputs[key].detach().cpu().numpy()

    # Create a blank image
    img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # Get ego position for centering
    ego_state = inputs["ego_current_state"][0]
    ego_x, ego_y = ego_state[0], ego_state[1]

    # Define coordinate transformation functions
    def world_to_image(x, y):
        """Convert world coordinates to image coordinates"""
        # Center the ego vehicle in the image
        img_x = int(((x - ego_x) + view_range) * image_size[0] / (2 * view_range))
        img_y = int((view_range - (y - ego_y)) * image_size[1] / (2 * view_range))
        return img_x, img_y

    def draw_rotated_rect(img, center_x, center_y, width, height, angle, color, thickness=-1):
        """Draw a rotated rectangle"""
        center = world_to_image(center_x, center_y)
        # Scale dimensions to image space
        scaled_width = int(width * image_size[0] / (2 * view_range))
        scaled_height = int(height * image_size[1] / (2 * view_range))

        # Create rectangle points
        rect = cv2.boxPoints(
            ((center[0], center[1]), (scaled_width, scaled_height), -np.degrees(angle))
        )
        rect = np.intp(rect)

        if thickness == -1:
            cv2.fillPoly(img, [rect], color)
        else:
            cv2.drawContours(img, [rect], 0, color, thickness)

    def draw_arrow(img, start_x, start_y, end_x, end_y, color, thickness=2):
        """Draw an arrow from start to end"""
        start = world_to_image(start_x, start_y)
        end = world_to_image(end_x, end_y)
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.3)

    # ==== Lanes (draw first as background) ====
    lanes = inputs["lanes"][0]
    for i in range(lanes.shape[0]):
        traffic_light = lanes[i, 0, 8:13]
        if traffic_light[0] == 1:
            color = (0, 255, 0)  # Green
        elif traffic_light[1] == 1:
            color = (0, 255, 255)  # Yellow
        elif traffic_light[2] == 1:
            color = (0, 0, 255)  # Red
        elif traffic_light[3] == 1:
            color = (128, 128, 128)  # Gray
        elif traffic_light[4] == 1:
            color = (0, 0, 0)  # no traffic light
        else:
            color = (255, 0, 0)  # Default color (Blue)

        # Draw lane boundaries with low opacity
        points_l = []
        points_r = []

        for j in range(lanes.shape[1]):
            if np.sum(np.abs(lanes[i, j, :2])) < 1e-6:
                break

            cx, cy = lanes[i, j, 0:2]
            lx = cx + lanes[i, j, 4]
            ly = cy + lanes[i, j, 5]
            rx = cx + lanes[i, j, 6]
            ry = cy + lanes[i, j, 7]

            points_l.append(world_to_image(lx, ly))
            points_r.append(world_to_image(rx, ry))

        points_l = np.array(points_l, np.int32)
        points_r = np.array(points_r, np.int32)

        # Draw directly without transparency
        cv2.polylines(img, [points_l], False, color, 2)
        cv2.polylines(img, [points_r], False, color, 2)

    # ==== Route lanes ====
    route_lanes = inputs["route_lanes"][0]
    for i in range(route_lanes.shape[0]):
        points_center = []

        for j in range(route_lanes.shape[1]):
            if np.sum(np.abs(route_lanes[i, j, :2])) < 1e-6:
                break
            cx, cy = route_lanes[i, j, 0:2]
            points_center.append(world_to_image(cx, cy))

        pts_center = np.array(points_center, np.int32)
        cv2.polylines(img, [pts_center], False, (255, 0, 255), 8)

    # ==== Static objects ====
    static_objects = inputs["static_objects"][0]
    for i in range(static_objects.shape[0]):
        obj = static_objects[i]

        if np.sum(np.abs(obj[:4])) < 1e-6:
            continue

        obj_x, obj_y = obj[0], obj[1]
        obj_heading = np.arctan2(obj[3], obj[2])
        obj_width = obj[4] if obj.shape[0] > 4 else 1.0
        obj_length = obj[5] if obj.shape[0] > 5 else 1.0

        # Set color based on object type
        obj_type = np.argmax(obj[-4:]) if obj.shape[0] >= 10 else 0
        colors = [
            (0, 165, 255),
            (128, 128, 128),
            (0, 255, 255),
            (42, 42, 165),
        ]  # Orange, Gray, Yellow, Brown
        obj_color = colors[obj_type % len(colors)]

        # Draw directly without transparency
        draw_rotated_rect(img, obj_x, obj_y, obj_length, obj_width, obj_heading, obj_color)

    # ==== Neighbor agents ====
    neighbors = inputs["neighbor_agents_past"][0]
    last_timestep = neighbors.shape[1] - 1

    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i, last_timestep]

        if np.sum(np.abs(neighbor[:4])) < 1e-6:
            continue

        n_x, n_y = neighbor[0], neighbor[1]
        n_heading = np.arctan2(neighbor[3], neighbor[2])
        vel_x, vel_y = neighbor[4], neighbor[5]
        len_y, len_x = neighbor[6], neighbor[7]

        # Set color based on vehicle type
        vehicle_type = np.argmax(neighbor[8:11]) if neighbor.shape[0] > 8 else 0
        if vehicle_type == 0:  # Vehicle
            color = (255, 0, 0)  # Blue
        elif vehicle_type == 1:  # Pedestrian
            color = (0, 255, 0)  # Green
        else:  # Bicycle
            color = (255, 0, 255)  # Purple

        # Draw past trajectory
        past_points = []
        for t in range(last_timestep + 1):
            px, py = neighbors[i, t, 0], neighbors[i, t, 1]
            if np.sum(np.abs(neighbors[i, t, :2])) > 1e-6:
                past_points.append(world_to_image(px, py))

        if len(past_points) > 1:
            pts = np.array(past_points, np.int32)
            cv2.polylines(img, [pts], False, color, 1)

        # Draw bounding box
        draw_rotated_rect(img, n_x, n_y, len_x, len_y, n_heading, color, thickness=2)

        # Draw velocity arrow
        if np.sqrt(vel_x**2 + vel_y**2) > 0.1:
            draw_arrow(img, n_x, n_y, n_x + vel_x / 2, n_y + vel_y / 2, (0, 165, 255), 2)

    # ==== Ego past trajectory ====
    if "ego_agent_past" in inputs:
        ego_past = inputs["ego_agent_past"][0]
        past_points = []
        for i in range(ego_past.shape[0]):
            px, py = ego_past[i, 0], ego_past[i, 1]
            past_points.append(world_to_image(px, py))

        if len(past_points) > 1:
            pts = np.array(past_points, np.int32)
            cv2.polylines(img, [pts], False, (0, 165, 255), 2)  # Orange

    # ==== Ego vehicle ====
    ego_heading = np.arctan2(ego_state[3], ego_state[2])
    car_length = 4.5
    car_width = 2.0

    # Draw ego vehicle as a filled rectangle
    draw_rotated_rect(img, ego_x, ego_y, car_length, car_width, ego_heading, (0, 0, 255))

    # ==== Goal pose ====
    if "goal_pose" in inputs:
        goal_x, goal_y, goal_yaw = inputs["goal_pose"][0]
        goal_end_x = goal_x + 3 * np.cos(goal_yaw)
        goal_end_y = goal_y + 3 * np.sin(goal_yaw)
        draw_arrow(img, goal_x, goal_y, goal_end_x, goal_end_y, (255, 0, 0), 3)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Return the image array (BGR format)
    return img


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    target = args.target

    if target.is_dir():
        # すべてのnpzファイルを取得
        npz_path_list = sorted(target.glob("*.npz"))
    else:
        assert target.suffix == ".npz", (
            "Target must be a .npz file or a directory containing .npz files"
        )
        npz_path_list = [target]

    def process_one(npz_path: Path) -> None:
        loaded = np.load(npz_path)

        data = {}
        for key, value in loaded.items():
            if key == "map_name" or key == "token":
                continue
            # add batch size axis
            data[key] = torch.tensor(np.expand_dims(value, axis=0))

        img = visualize_inputs_cv2(data)

        # Save image instead of displaying due to OpenCV GUI support issue
        output_path = npz_path.with_suffix(".png")
        cv2.imwrite(str(output_path), img)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, npz_path): npz_path for npz_path in npz_path_list}
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
