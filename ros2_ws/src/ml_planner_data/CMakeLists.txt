cmake_minimum_required(VERSION 3.14)
project(ml_planner_data)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# The current Autoware fork exposes the planner preprocessing as
# autoware_diffusion_planner, but does not build its component library on
# machines without CUDA/TensorRT.  Dataset generation is CPU-only, so compile
# the shared preprocessing sources directly into this Python extension.
set(DIFFUSION_PLANNER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../deps/autoware_universe/planning/autoware_diffusion_planner"
)

# Non-ament dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Python bindings (FrameDataCache)
pybind11_add_module(_ml_planner_data
  src/pybind_module.cpp
  src/bag_dataset_builder.cpp
  src/skip_index.cpp
  src/bag_frame_reader.cpp
  src/frame_data_cache.cpp
  src/label_builder.cpp
  src/topic_config.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/input_builder.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/agent.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/ego_history.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/ego_shape.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/goal_pose.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/map.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/turn_indicators.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/items/traffic_signals.cpp
  ${DIFFUSION_PLANNER_DIR}/src/preprocessing/preprocessing_utils.cpp
  ${DIFFUSION_PLANNER_DIR}/src/utils/utils.cpp
)
target_include_directories(_ml_planner_data PUBLIC
  "${DIFFUSION_PLANNER_DIR}/include"
)
ament_target_dependencies(_ml_planner_data PUBLIC
  ${${PROJECT_NAME}_FOUND_BUILD_DEPENDS}
)
install(TARGETS _ml_planner_data
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

ament_auto_package(INSTALL_TO_SHARE config)
