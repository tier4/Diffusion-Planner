#!/bin/bash
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <input rosbag>"
    exit 1
fi

SCRIPT_DIR=$(readlink -f $(dirname $0))
INPUT_ROSBAG_DIR=$(readlink -f $1)
OUTPUT_ROSBAG_DIR=$2

cd $SCRIPT_DIR
set +eu
source ~/autoware/install/setup.bash
set -eu

declare -a TOPICS=(
    # Camera
    # Gen1
    "/sensing/camera/camera0/camera_info"
    "/sensing/camera/camera0/image_rect_color/compressed"
    "/sensing/camera/camera1/camera_info"
    "/sensing/camera/camera1/image_rect_color/compressed"
    "/sensing/camera/camera2/camera_info"
    "/sensing/camera/camera2/image_rect_color/compressed"
    "/sensing/camera/camera3/camera_info"
    "/sensing/camera/camera3/image_rect_color/compressed"
    "/sensing/camera/camera4/camera_info"
    "/sensing/camera/camera4/image_rect_color/compressed"
    "/sensing/camera/camera5/camera_info"
    "/sensing/camera/camera5/image_rect_color/compressed"
    "/sensing/camera/camera6/camera_info"
    "/sensing/camera/camera6/image_rect_color/compressed"
    "/sensing/camera/camera6/image_raw/compressed"

    # Gen2
    "/sensing/camera/camera0/camera_info"
    "/sensing/camera/camera0/image_raw/compressed"
    "/sensing/camera/camera1/camera_info"
    "/sensing/camera/camera1/image_raw/compressed"
    "/sensing/camera/camera2/camera_info"
    "/sensing/camera/camera2/image_raw/compressed"
    "/sensing/camera/camera3/camera_info"
    "/sensing/camera/camera3/image_raw/compressed"
    "/sensing/camera/camera4/camera_info"
    "/sensing/camera/camera4/image_raw/compressed"
    "/sensing/camera/camera5/camera_info"
    "/sensing/camera/camera5/image_raw/compressed"
    "/sensing/camera/camera6/camera_info"
    "/sensing/camera/camera6/image_raw/compressed"
    "/sensing/camera/camera7/camera_info"
    "/sensing/camera/camera7/image_raw/compressed"
    "/sensing/camera/camera8/camera_info"
    "/sensing/camera/camera8/image_raw/compressed"
    "/sensing/camera/camera9/camera_info"
    "/sensing/camera/camera9/image_raw/compressed"
    "/sensing/camera/camera10/camera_info"
    "/sensing/camera/camera10/image_raw/compressed"

    # LiDAR
    # Gen1
    "/sensing/lidar/left/velodyne_packets"
    "/sensing/lidar/rear/velodyne_packets"
    "/sensing/lidar/right/velodyne_packets"
    "/sensing/lidar/top/velodyne_packets"

    # Gen2
    "/sensing/lidar/top/pandar_packets"
    "/sensing/lidar/side_left/pandar_packets"
    "/sensing/lidar/side_right/pandar_packets"
    "/sensing/lidar/front_left/pandar_packets"
    "/sensing/lidar/front_right/pandar_packets"
)

METADATA=$INPUT_ROSBAG_DIR/metadata.yaml
STORAGE=$(cat $METADATA | grep storage_identifier)  #  "  storage_identifier: mcap" or "  storage_identifier: sqlite3"
EXT=$(echo $STORAGE | awk '{print $2}')
PLAY_CMD="ros2 bag filter --storage=$EXT $INPUT_ROSBAG_DIR -o $OUTPUT_ROSBAG_DIR -x"

for TOPIC in "${TOPICS[@]}"; do
    PLAY_CMD+=" $TOPIC"
done

echo "Executing command: $PLAY_CMD"
eval $PLAY_CMD
