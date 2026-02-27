cmake_minimum_required(VERSION 3.14)
project(autoware_diffusion_planner_tools)

find_package(autoware_cmake REQUIRED)
autoware_package()

# Add compile option to suppress deprecation warnings (optional)
add_compile_options(-Wno-deprecated-declarations)

# Suppress narrowing conversion warnings from TensorRT headers
add_compile_options(-Wno-narrowing)

ament_auto_add_executable(data_converter
  src/data_converter.cpp
)

ament_auto_add_executable(inference_tool
  src/inference_tool.cpp
)

ament_auto_package(
  INSTALL_TO_SHARE
)
