#!/bin/bash
set -eux
target_time_dir=$(readlink -f $1)

ros2 bag reindex $target_time_dir
/home/ubuntu/sakoda/Diffusion-Planner_npf/ros_scripts/filter_rosbag_all.sh $target_time_dir
rm -rf $target_time_dir
