#!/bin/bash
set -eux

cd $(dirname $0)

vcs import --recursive src < data_converter.repos
vcs pull src
rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
colcon build \
  --symlink-install \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  --packages-up-to autoware_diffusion_planner_data_converter
