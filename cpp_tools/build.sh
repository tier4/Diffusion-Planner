#!/bin/bash
set -eux

cd $(dirname $0)

if [[ "${1:-}" == "--test" ]]; then
  echo "[INFO] Build and run unit tests with colcon test"
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to autoware_diffusion_planner_tools
  colcon test --packages-select autoware_diffusion_planner_tools
  colcon test-result --all
  exit 0
fi

colcon build \
  --symlink-install \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  --packages-up-to autoware_diffusion_planner_tools
