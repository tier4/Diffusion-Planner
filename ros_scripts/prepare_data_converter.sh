#!/bin/bash
set -eux

cd ~/autoware/src/core/autoware_msgs
git checkout 1.3.0
cd ~/autoware
rm -rf build/autoware_*_msgs
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TOOL=ON --packages-up-to autoware_diffusion_planner
