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
    "/localization/kinematic_state"
    "/localization/acceleration"
    "/perception/object_recognition/tracking/objects"
    "/perception/traffic_light_recognition/traffic_signals"
    "/planning/mission_planning/route"
    "/vehicle/status/turn_indicators_status"
    "/vehicle/status/velocity_status"
    "/tf"
    "/tf_static"
)

METADATA=$INPUT_ROSBAG_DIR/metadata.yaml
STORAGE=$(cat $METADATA | grep storage_identifier)  #  "  storage_identifier: mcap" or "  storage_identifier: sqlite3"
EXT=$(echo $STORAGE | awk '{print $2}')
PLAY_CMD="ros2 bag filter --storage=$EXT $INPUT_ROSBAG_DIR -o $OUTPUT_ROSBAG_DIR -i"

for TOPIC in "${TOPICS[@]}"; do
    PLAY_CMD+=" $TOPIC"
done

echo "Executing command: $PLAY_CMD"
eval $PLAY_CMD
