#!/bin/bash
set -eux

cd $(dirname $0)

colcon build \
  --symlink-install \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-Wno-error=unused-variable" \
  --packages-up-to autoware_diffusion_planner_tools
  
