#!/bin/bash
set -eux

cd $(dirname $0)

if [[ "${1:-}" == "--test" ]]; then
  echo "[INFO] Build and run unit tests with colcon test"
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBUILD_TESTING=1 --packages-up-to autoware_diffusion_planner_tools
  colcon test --packages-select autoware_diffusion_planner_tools
  colcon test-result --all
  exit 0
fi

colcon build \
  --symlink-install \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DBUILD_TESTING=0 \
  -DSSV2_HEADLESS_EGO=ON \
  -DCMAKE_CXX_FLAGS=-DSSV2_HEADLESS_EGO \
  --packages-up-to autoware_diffusion_planner_tools openscenario_python autoware_lanelet2_extension_python behavior_tree_plugin
