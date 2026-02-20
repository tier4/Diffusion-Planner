# Annotation UI (Lichtblick)

This extension recreates the annotation workflow as multiple Lichtblick panels using **ROS2 topics**.

The backend server is:

- `preference_optimization/annotation_ros_node.py`

WebSocket transport is no longer used for panel data/control.

## Architecture

- Python backend runs as a ROS2 node and publishes annotation state + visualization topics.
- The extension subscribes/publishes through `PanelExtensionContext` ROS topic APIs.
- For 3D trajectory rendering, use the Autoware Lichtblick converter extension (no local dummy trajectory converter).

## Prerequisite: Install Autoware Lichtblick Plugins First

Before installing this annotation extension, install Tier IV's Autoware converter extension first:

```bash
git clone https://github.com/tier4/AutowareLichtblickPlugins.git
cd AutowareLichtblickPlugins
bash ./install.sh
```

This provides the planning/perception converters used by the 3D panel, including trajectory visualization.

## Quick Start

1. Start ROS2 and foxglove bridge (example):

```bash
ros2 run foxglove_bridge foxglove_bridge
```

2. Start the annotation ROS node:

```bash
python -m preference_optimization.annotation_ros_node \
  --model-path /path/to/model.pth \
  --npz-list /path/to/train_npz_list.json \
  --device cuda:0
```

3. Install this extension (from this folder):

```bash
npm install
npm run local-install
```

4. In Lichtblick, import the [dpo.json](./layouts/dpo.json) as layout configuration.

## ROS Topic Contract

### Command / state

- `/annotation/cmd` (`std_msgs/msg/String`): panel commands (JSON payload)
- `/annotation/state` (`std_msgs/msg/String`): serialized UI state (JSON payload)

### Trajectory topics

- `/annotation/data/trajectory/deterministic` (`autoware_planning_msgs/msg/Trajectory`)
- `/annotation/data/trajectory/stochastic` (`autoware_planning_msgs/msg/Trajectory`)
- `/annotation/data/trajectory/ground_truth` (`autoware_planning_msgs/msg/Trajectory`)
- `/annotation/data/trajectory/ego_history` (`autoware_planning_msgs/msg/Trajectory`)
- `/annotation/data/trajectory/gt_snippet` (`autoware_planning_msgs/msg/Trajectory`)

### 3D context topics

- `/annotation/data/map_markers` (`visualization_msgs/msg/MarkerArray`)
- `/annotation/data/footprints` (`visualization_msgs/msg/MarkerArray`)
- `/annotation/data/tracked_objects` (`autoware_perception_msgs/msg/TrackedObjects`)
- `/tf` (dynamic transforms, including deterministic/stochastic base links)

## 3D Panel Notes

- Set fixed frame to `map`.
- Ensure Autoware converter extension is enabled for:
  - `autoware_planning_msgs/msg/Trajectory -> foxglove.SceneUpdate`
- If a trajectory topic is visible but not rendered, verify:
  - topic has non-empty `points`
  - `header.frame_id` is valid and present in TF
  - 3D per-topic setting `viewPath` is enabled

## Image Panels

Use native Lichtblick Image panels for camera topics as needed. These are independent from the annotation extension panels.
