cmake_minimum_required(VERSION 3.14)
project(autoware_diffusion_planner_tools)

find_package(autoware_cmake REQUIRED)
autoware_package()

# Add compile option to suppress deprecation warnings (optional)
add_compile_options(-Wno-deprecated-declarations)

# Suppress narrowing conversion warnings from TensorRT headers
add_compile_options(-Wno-narrowing)

# rosbag2_storage >= 0.26 (Jazzy+): offered_qos_profiles changed from
# std::string to std::vector<rclcpp::QoS>. Pass a compile-time flag.
find_package(rosbag2_storage REQUIRED)
if(rosbag2_storage_VERSION VERSION_GREATER_EQUAL "0.26.0")
  add_compile_definitions(ROSBAG2_QOS_PROFILES_AS_VECTOR)
endif()

ament_auto_add_executable(data_converter
  src/data_converter.cpp
)

ament_auto_add_executable(inference_tool
  src/inference_tool.cpp
)

ament_auto_add_executable(benchmark_tool
  src/benchmark_tool.cpp
)

ament_auto_package(
  INSTALL_TO_SHARE
)
