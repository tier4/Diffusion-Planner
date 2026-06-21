cmake_minimum_required(VERSION 3.14)
project(autoware_diffusion_planner_tools)

include(FetchContent)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(cli11)

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
  src/cli/converter_options.cpp
  src/conversion/data_converter.cpp
  src/io/frame_writer.cpp
  src/io/npz_frame_writer.cpp
  src/io/projector_factory.cpp
  src/processing/ego_sequence.cpp
  src/processing/frame_processor.cpp
  src/processing/frame_skip_decision.cpp
  src/processing/neighbor_processor.cpp
  src/processing/sequence_builder.cpp
  src/rosbag/parsed_bag_data.cpp
  src/rosbag/missing_topics.cpp
  src/utils/cnpy.cpp
)
target_include_directories(data_converter PRIVATE include src)
target_link_libraries(data_converter yaml-cpp ZLIB::ZLIB CLI11::CLI11 fmt::fmt)

ament_auto_add_executable(parse_rosbag_for_directory_with_map_version
  src/parse_rosbag_for_directory_with_map_version.cpp
  src/cli/converter_options.cpp
  src/conversion/data_converter.cpp
  src/io/frame_writer.cpp
  src/io/npz_frame_writer.cpp
  src/io/projector_factory.cpp
  src/processing/ego_sequence.cpp
  src/processing/frame_processor.cpp
  src/processing/frame_skip_decision.cpp
  src/processing/neighbor_processor.cpp
  src/processing/sequence_builder.cpp
  src/rosbag/parsed_bag_data.cpp
  src/rosbag/missing_topics.cpp
  src/utils/cnpy.cpp
)
target_include_directories(parse_rosbag_for_directory_with_map_version PRIVATE include src)
target_link_libraries(parse_rosbag_for_directory_with_map_version yaml-cpp ZLIB::ZLIB Threads::Threads CLI11::CLI11 fmt::fmt)

ament_auto_add_executable(inference_tool
  src/inference_tool.cpp
)

ament_auto_add_executable(map_exporter
  src/map_exporter.cpp
)
target_link_libraries(map_exporter yaml-cpp)

ament_auto_add_executable(benchmark_tool
  src/benchmark_tool.cpp
)

ament_auto_package(
  INSTALL_TO_SHARE
)

# --- GoogleTest for unit tests ---
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Existing test
  ament_add_gtest(test_timestamp_stats test/test_timestamp_stats.cpp)
  if(TARGET test_timestamp_stats)
    target_include_directories(test_timestamp_stats PRIVATE include)
  endif()

  # cli/converter_options — no ROS message deps, no heavy linking
  ament_add_gtest(test_converter_options
    test/test_converter_options.cpp
    src/cli/converter_options.cpp
  )
  if(TARGET test_converter_options)
    target_include_directories(test_converter_options PRIVATE include src)
    target_link_libraries(test_converter_options CLI11::CLI11)
  endif()

  # processing/frame_filters — header-only pure functions; needs autoware_diffusion_planner dims
  ament_add_gtest(test_frame_filters
    test/test_frame_filters.cpp
  )
  if(TARGET test_frame_filters)
    target_include_directories(test_frame_filters PRIVATE include src)
    ament_target_dependencies(test_frame_filters autoware_diffusion_planner)
  endif()

  # io/frame_writer pure builders — needs nav_msgs + autoware_diffusion_planner
  ament_add_gtest(test_frame_writer
    test/test_frame_writer.cpp
    src/io/frame_writer.cpp
  )
  if(TARGET test_frame_writer)
    target_include_directories(test_frame_writer PRIVATE include src)
    ament_target_dependencies(test_frame_writer
      nav_msgs
      autoware_diffusion_planner
    )
  endif()

  # processing/frame_skip_decision — pure logic, needs autoware_diffusion_planner dims
  ament_add_gtest(test_frame_skip_decision
    test/test_frame_skip_decision.cpp
    src/processing/frame_skip_decision.cpp
  )
  if(TARGET test_frame_skip_decision)
    target_include_directories(test_frame_skip_decision PRIVATE include src)
    ament_target_dependencies(test_frame_skip_decision autoware_diffusion_planner)
  endif()

  # rosbag/check_missing_topics — needs ROS message types but NOT rosbag2_storage
  ament_add_gtest(test_missing_topics
    test/test_missing_topics.cpp
    src/rosbag/missing_topics.cpp
  )
  if(TARGET test_missing_topics)
    target_include_directories(test_missing_topics PRIVATE include src)
    ament_target_dependencies(test_missing_topics
      nav_msgs
      autoware_perception_msgs
      autoware_planning_msgs
      autoware_vehicle_msgs
      geometry_msgs
    )
  endif()
endif()
